"""
Train the pairwise model and tune the decision layer, then save a bundle for
scripts/predict.py to run inference from.

Training does not need the whole dataset. The pairwise model sees roughly 90
candidate pairs per entity, so a few hundred thousand entities already give
tens of millions of training rows - far past the point where more helps a
38-feature GBDT. What cannot be subsampled is inference over the full test set,
which is why that lives in a separate, shardable script.

`--max-fit-entities` bounds training memory regardless of how large the input
is, which is the difference between a run that fits on a laptop or a Kaggle
session and one that is killed for using 30 GB.

    python3 scripts/train_model.py --train-dir dataset/train \
        --bundle model.pkl --max-fit-entities 150000
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd


def _project_root():
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.isdir(os.path.join(here, "src", "matching")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    raise RuntimeError("could not locate the project root containing src/matching")


sys.path.insert(0, _project_root())

from src.matching import io as match_io
from src.matching import metrics, resolve, splits
from src.matching.blocking import CHANNELS, channel_contribution, generate_candidates_cached
from src.matching.decide import select_matches, tune_threshold, tune_tiered
from src.matching.model import PairwiseMatcher, label_pairs
from src.matching.pair_features import (
    FEATURE_NAMES, build_idf, build_pair_table, build_record_views, set_embedding_feature,
)
from src.matching.persist import save_bundle
from src.preprocessing.pipeline import preprocess_dataframe

logger = logging.getLogger("train_model")


def _load_sources(data_dir, prefix):
    frames = {}
    for source in ("source1", "source2", "source3"):
        parquet = os.path.join(data_dir, f"{prefix}_{source}_processed.parquet")
        tsv = os.path.join(data_dir, f"{prefix}_{source}.tsv")
        if os.path.exists(parquet):
            frames[source] = pd.read_parquet(parquet)
        elif os.path.exists(tsv):
            frames[source] = preprocess_dataframe(pd.read_csv(tsv, sep="\t", dtype=str))
        else:
            raise FileNotFoundError(f"neither {parquet} nor {tsv} exists")
    pool = pd.concat([frames["source2"], frames["source3"]], ignore_index=True)
    return frames["source1"], pool


def _tokens(df, column):
    if column not in df.columns:
        return []
    return [t.split() for t in df[column].fillna("").astype(str)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--bundle", default="model.pkl")
    parser.add_argument("--report-file", default="reports/training.json")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-fit-entities", type=int, default=150_000,
                        help="cap on entities used to fit the model; bounds memory")
    parser.add_argument("--max-tune-entities", type=int, default=50_000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--tune-fraction", type=float, default=0.25)
    parser.add_argument("--strategy", default="tiered",
                        choices=("tiered", "expected_f05", "threshold", "top1"))
    parser.add_argument("--conflict-stage", default="post", choices=("none", "pre", "post"))
    parser.add_argument("--channels", default=None)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--blocking-threads", type=int, default=-1)
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.time()
    report = {}
    rng = np.random.default_rng(args.seed)

    s1_df, pool_df = _load_sources(args.train_dir, "train")
    gt_path = args.ground_truth or os.path.join(args.train_dir, "train_ground_truth.tsv")
    ground_truth = match_io.load_ground_truth(gt_path)
    logger.info("loaded | source1=%d pool=%d labels=%d", len(s1_df), len(pool_df),
                len(ground_truth))

    country_of = dict(zip(s1_df["entity_id"], s1_df["country_clean"].fillna("UNKNOWN")))
    one_to_one = resolve.check_one_to_one(ground_truth)
    report["one_to_one_check"] = one_to_one
    conflict_stage = args.conflict_stage
    if not one_to_one["holds"] and conflict_stage != "none":
        logger.warning("one-to-one assumption violated; disabling conflict resolution")
        conflict_stage = "none"

    report["all_empty_baseline"] = metrics.all_empty_baseline(ground_truth)
    logger.info("all-empty baseline = %.4f", report["all_empty_baseline"])

    s1_ids = s1_df["entity_id"].tolist()
    train_ids, val_ids = splits.holdout_split(
        s1_ids, ground_truth, country_of, args.val_fraction, args.seed
    )
    fit_ids, tune_ids = splits.holdout_split(
        train_ids, ground_truth, country_of, args.tune_fraction, args.seed + 1
    )

    # Capping is what keeps memory flat as the dataset grows. Sampled at random
    # from an already stratified split, so the class mix is preserved.
    def cap(ids, limit, label):
        if limit and len(ids) > limit:
            kept = sorted(rng.choice(np.array(ids, dtype=object), size=limit, replace=False))
            logger.info("capped %s entities %d -> %d", label, len(ids), limit)
            return list(kept)
        return ids

    fit_ids = cap(fit_ids, args.max_fit_entities, "fit")
    tune_ids = cap(tune_ids, args.max_tune_entities, "tune")
    val_ids = cap(val_ids, args.max_tune_entities, "val")
    logger.info("fit=%d tune=%d val=%d", len(fit_ids), len(tune_ids), len(val_ids))
    report["split"] = {"fit": len(fit_ids), "tune": len(tune_ids), "val": len(val_ids)}

    channels = ([c.strip() for c in args.channels.split(",") if c.strip()]
                if args.channels else [c for c in CHANNELS if c != "embedding"])
    encoder = None
    if args.embeddings:
        from src.matching.embeddings import DEFAULT_MODEL, build_encoder

        if "embedding" not in channels:
            channels.append("embedding")
        encoder = build_encoder(args.embedding_model or DEFAULT_MODEL,
                                device=args.embedding_device)

    blocking_config = {"n_threads": args.blocking_threads, "channels": channels,
                       "embedding_encoder": encoder}

    # Only the entities actually used are blocked: blocking the full training
    # Source 1 would dominate the runtime for rows the model never sees.
    used_ids = sorted(set(fit_ids) | set(tune_ids) | set(val_ids))
    used_df = s1_df[s1_df["entity_id"].isin(used_ids)].reset_index(drop=True)
    logger.info("blocking %d of %d training entities", len(used_df), len(s1_df))

    t0 = time.time()
    candidates = generate_candidates_cached(
        used_df, pool_df, blocking_config, args.cache_dir, tag="train"
    )
    report["blocking_seconds"] = round(time.time() - t0, 1)

    val_gt = {sid: ground_truth.get(sid, set()) for sid in val_ids}
    report["blocking"] = metrics.blocking_report(candidates, val_gt, len(pool_df))
    report["blocking_channels"] = channel_contribution(candidates, val_gt)
    logger.info("blocking: pair recall %.4f | %.1f candidates/entity",
                report["blocking"]["pair_recall"],
                report["blocking"]["candidates_per_entity_mean"])

    s1_views, pool_views = build_record_views(used_df), build_record_views(pool_df)
    name_column = ("business_name_core" if "business_name_core" in used_df.columns
                   else "business_name_clean")
    name_idf = build_idf(_tokens(s1_df, name_column), _tokens(pool_df, name_column))
    addr_idf = build_idf(_tokens(s1_df, "business_address_clean"),
                         _tokens(pool_df, "business_address_clean"))

    embedding = None
    if encoder is not None:
        from src.matching.embeddings import build_embedding_lookup

        s1_map, s1_vec = build_embedding_lookup(used_df, encoder, args.cache_dir, "train_s1")
        pool_map, pool_vec = build_embedding_lookup(pool_df, encoder, args.cache_dir, "train_pool")
        embedding = (s1_map, s1_vec, pool_map, pool_vec)

    def featurise(ids):
        subset = {sid: candidates.get(sid, {}) for sid in ids}
        X, index = build_pair_table(subset, s1_views, pool_views, name_idf, addr_idf)
        if embedding is not None and len(X):
            from src.matching.embeddings import pair_cosines

            set_embedding_feature(X, pair_cosines(index, *embedding))
        return X, index

    X_train, train_index = featurise(fit_ids)
    y_train = label_pairs(train_index, ground_truth)
    logger.info("train pairs: %d (%d positive)", len(y_train), int(y_train.sum()))
    report["train_pairs"] = int(len(y_train))

    matcher = PairwiseMatcher(random_state=args.seed).fit(
        X_train, y_train, train_index["s1_entity_id"].to_numpy(dtype=object)
    )
    del X_train, train_index, y_train

    def score(ids):
        X, index = featurise(ids)
        out = {sid: [] for sid in ids}
        if len(X):
            probs = matcher.predict_proba(X)
            for s1_id, cand_id, prob in zip(
                index["s1_entity_id"], index["candidate_entity_id"], probs
            ):
                out[s1_id].append((cand_id, float(prob)))
        return out

    tune_scored = score(tune_ids)
    tune_gt = {sid: ground_truth.get(sid, set()) for sid in tune_ids}
    tuned = tune_threshold(tune_scored, tune_gt, max_k=args.max_k)
    tuned_tiered = tune_tiered(tune_scored, tune_gt, max_k=args.max_k)
    report["tuned_threshold"], report["tuned_tiered"] = tuned, tuned_tiered
    logger.info("tuned threshold: %s", tuned)
    logger.info("tuned tiered:    %s", tuned_tiered)

    val_scored = score(val_ids)
    kwargs = ({"t_high": tuned["t_high"], "ratio": tuned["ratio"], "max_k": args.max_k}
              if args.strategy == "threshold" else
              {"t_first": tuned_tiered["t_first"], "t_rest": tuned_tiered["t_rest"],
               "ratio": tuned_tiered["ratio"], "max_k": args.max_k}
              if args.strategy == "tiered" else
              {"t_high": tuned["t_high"]} if args.strategy == "top1" else
              {"max_k": args.max_k})

    working = val_scored
    if conflict_stage == "pre":
        working, _ = resolve.resolve_scored_pairs(working)
    predictions = select_matches(working, args.strategy, **kwargs)
    if conflict_stage == "post":
        predictions, _ = resolve.resolve_predictions(predictions, working)

    score_value, per_entity = metrics.macro_f05(predictions, val_gt)
    report["validation_macro_f05"] = score_value
    report["errors"] = metrics.error_attribution(predictions, candidates, val_gt)
    report["slices"] = metrics.slice_report(per_entity, val_gt, country_of)
    report["best_validation"] = {"strategy": args.strategy,
                                 "conflict_stage": conflict_stage,
                                 "macro_f05": score_value}
    logger.info("validation macro F0.5 = %.4f (%s, conflict=%s)",
                score_value, args.strategy, conflict_stage)

    save_bundle(
        args.bundle, matcher, name_idf, addr_idf, tuned, tuned_tiered,
        blocking_config, args.strategy, conflict_stage, args.max_k, FEATURE_NAMES,
        extra={"embedding_model": args.embedding_model, "seed": args.seed,
               "fit_entities": len(fit_ids)},
    )

    report["total_seconds"] = round(time.time() - started, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.report_file)) or ".", exist_ok=True)
    with open(args.report_file, "w") as handle:
        json.dump(report, handle, indent=2, default=str)

    print(f"\nvalidation macro F0.5 : {score_value:.4f}")
    print(f"all-empty baseline    : {report['all_empty_baseline']:.4f}")
    print(f"bundle                : {args.bundle}")
    print(f"\nNext: python3 scripts/predict.py --bundle {args.bundle} "
          f"--test-dir dataset/test --output-dir output --shards 20")


if __name__ == "__main__":
    main()
