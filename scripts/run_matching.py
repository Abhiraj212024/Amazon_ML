"""
End-to-end matching pipeline: blocking -> pairwise model -> conflict resolution
-> set selection, scored with the official macro F0.5.

Runs an ablation over the decision strategy and the conflict-resolution stage,
so the value of each filtering layer is visible rather than assumed.

    python3 scripts/run_matching.py --train-dir dataset/train
    python3 scripts/run_matching.py --train-dir dataset/train \
        --test-dir dataset/test --output-dir output
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
    """
    Locate the directory that holds `src/`, searching upward from this file.

    The repository keeps scripts beside `src/`, while the submission package
    places them under `src/` so that all source sits there as the challenge
    requires. Searching upward makes the same file work in both layouts.
    """
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
from src.matching import metrics, resolve, scorecache, splits
from src.matching.blocking import channel_contribution, generate_candidates_cached
from src.matching.decide import select_matches, tune_threshold, tune_tiered
from src.matching.model import PairwiseMatcher, label_pairs
from src.matching.pair_features import (
    build_idf, build_pair_table, build_record_views, set_embedding_feature,
)
from src.preprocessing.pipeline import preprocess_dataframe

logger = logging.getLogger("run_matching")


def _load_sources(data_dir, prefix):
    processed_paths = {
        source: os.path.join(data_dir, f"{prefix}_{source}_processed.parquet")
        for source in ("source1", "source2", "source3")
    }
    processed_exists = [os.path.exists(path) for path in processed_paths.values()]
    use_processed = all(processed_exists)
    if any(processed_exists) and not use_processed:
        missing = [path for path, exists in zip(processed_paths.values(), processed_exists) if not exists]
        raise FileNotFoundError(f"processed dataset is incomplete; missing {missing}")

    frames = {}
    for source in ("source1", "source2", "source3"):
        path = processed_paths[source] if use_processed else os.path.join(
            data_dir, f"{prefix}_{source}.tsv"
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"expected raw TSV file; missing {path}")
        if use_processed:
            frames[source] = pd.read_parquet(path)
        else:
            frames[source] = preprocess_dataframe(pd.read_csv(path, sep="\t", dtype=str))
    pool = pd.concat([frames["source2"], frames["source3"]], ignore_index=True)
    return frames["source1"], pool


def _tokens(df, column):
    if column not in df.columns:
        return []
    return [t.split() for t in df[column].fillna("").astype(str)]


def _featurise(candidates, s1_views, pool_views, name_idf, addr_idf, s1_ids, embedding):
    """Build the feature matrix, filling the cosine column when embeddings are on."""
    subset = {sid: candidates.get(sid, {}) for sid in s1_ids}
    X, pair_index = build_pair_table(subset, s1_views, pool_views, name_idf, addr_idf)

    if embedding is not None and len(X):
        from src.matching.embeddings import pair_cosines

        started = time.time()
        set_embedding_feature(X, pair_cosines(pair_index, *embedding))
        logger.info("embedding cosines for %d pairs in %.1fs", len(X), time.time() - started)

    return X, pair_index


def _score_pairs(matcher, candidates, s1_views, pool_views, name_idf, addr_idf, s1_ids,
                 embedding=None, cache_dir=None, cache_tag=None, cache_extra=None):
    """
    Featurise and score, returning dict s1_id -> [(candidate_id, prob), ...].

    Featurising and scoring is several minutes at real scale, and it is the
    loop paid on every decision-layer experiment - which is where most of the
    loss sits. Caching it on the candidate set plus the model configuration
    makes a full threshold sweep effectively free.
    """
    subset = {sid: candidates.get(sid, {}) for sid in s1_ids}
    key = None
    if cache_dir and cache_tag:
        key = scorecache.fingerprint(subset, cache_extra)
        cached = scorecache.load(cache_dir, cache_tag, key)
        if cached is not None:
            return cached["scored"], cached.get("labels"), cached.get("probs")

    started = time.time()
    X, pair_index = _featurise(
        candidates, s1_views, pool_views, name_idf, addr_idf, s1_ids, embedding
    )

    scored = {sid: [] for sid in s1_ids}
    if len(X) == 0:
        return scored, None, None

    probs = matcher.predict_proba(X)
    for s1_id, cand_id, prob in zip(
        pair_index["s1_entity_id"], pair_index["candidate_entity_id"], probs
    ):
        scored[s1_id].append((cand_id, float(prob)))
    logger.info("scored %d pairs in %.1fs", len(X), time.time() - started)

    if key is not None:
        scorecache.save(cache_dir, cache_tag, key,
                        {"scored": scored, "labels": None, "probs": probs})
    return scored, None, probs


def _strategy_kwargs(strategy, tuned, tuned_tiered, max_k):
    """Tuned parameters for one strategy, in the form select_matches expects."""
    if strategy == "threshold":
        return {"t_high": tuned["t_high"], "ratio": tuned["ratio"], "max_k": max_k}
    if strategy == "tiered":
        return {
            "t_first": tuned_tiered["t_first"], "t_rest": tuned_tiered["t_rest"],
            "ratio": tuned_tiered["ratio"], "max_k": max_k,
        }
    if strategy == "top1":
        return {"t_high": tuned["t_high"]}
    return {"max_k": max_k}


def _evaluate(scored, candidates, ground_truth, country_of, strategy, conflict_stage,
              strategy_kwargs=None):
    """Apply one (strategy, conflict_stage) combination and score it."""
    strategy_kwargs = strategy_kwargs or {}
    working = scored
    conflict_report = None

    if conflict_stage == "pre":
        working, conflict_report = resolve.resolve_scored_pairs(working)

    predictions = select_matches(working, strategy, **strategy_kwargs)

    if conflict_stage == "post":
        predictions, conflict_report = resolve.resolve_predictions(predictions, working)

    score, per_entity = metrics.macro_f05(predictions, ground_truth)
    return {
        "strategy": strategy,
        "conflict_stage": conflict_stage,
        "macro_f05": score,
        "conflict_report": conflict_report,
        "errors": metrics.error_attribution(predictions, candidates, ground_truth),
        "slices": metrics.slice_report(per_entity, ground_truth, country_of),
    }, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", default=None,
                        help="when given, also predict the test set and write submission files")
    parser.add_argument("--ground-truth", default=None,
                        help="path to train_ground_truth.tsv (defaults to --train-dir/train_ground_truth.tsv)")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--report-file", default="reports/matching_report.json")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--tune-fraction", type=float, default=0.25,
                        help="share of the training entities reserved for tuning thresholds")
    parser.add_argument("--holdout-country", default=None,
                        help="validate on this country only, as a proxy for the unseen test country")
    parser.add_argument("--strategy", default="tiered",
                        choices=("tiered", "expected_f05", "threshold", "top1"))
    parser.add_argument("--conflict-stage", default="post", choices=("none", "pre", "post"))
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--cache-dir", default=None,
                        help="reuse blocking output across runs; the key covers the data "
                             "and the blocking config, so it invalidates itself")
    parser.add_argument("--blocking-threads", type=int, default=-1,
                        help="threads for the sparse top-k product (-1 = all cores)")
    parser.add_argument("--channels", default=None,
                        help="comma-separated blocking channels to run. Measured unique "
                             "recall: addr_char 12.4%%, rare_token 0.29%%, numeric 0.24%%, "
                             "name_char 0.20%%, name_word 0.04%%. Pruning the cheap ones "
                             "buys back most of the blocking time.")
    parser.add_argument("--max-candidates", type=int, default=0,
                        help="cap candidates kept per Source 1 entity after the channels "
                             "are unioned (0 = no cap). Smaller candidate sets are ranked "
                             "higher by the organisers, and featurising plus scoring is "
                             "linear in this number. Sweep it with scripts/tune_blocking.py")
    parser.add_argument("--max-df-char", type=float, default=1.0,
                        help="drop char n-grams appearing in more than this share of the pool")
    parser.add_argument("--max-k", type=int, default=10,
                        help="most matches predictable for one entity")
    parser.add_argument("--embeddings", action="store_true",
                        help="enable the dense embedding channel and cosine feature "
                             "(needs sentence-transformers and hnswlib)")
    parser.add_argument("--embedding-model", default=None,
                        help="model name or local path; 'hashing' uses a deterministic "
                             "offline stand-in that exercises the path without a download")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.time()
    report = {}

    logger.info("loading training data")
    s1_df, pool_df = _load_sources(args.train_dir, "train")
    ground_truth_path = args.ground_truth or os.path.join(args.train_dir, "train_ground_truth.tsv")
    ground_truth = match_io.load_ground_truth(ground_truth_path)
    logger.info(
        "training data loaded | source1=%d pool=%d labels=%d",
        len(s1_df), len(pool_df), len(ground_truth),
    )
    country_of = dict(zip(s1_df["entity_id"], s1_df["country_clean"].fillna("UNKNOWN")))

    # --- assumption check that gates conflict resolution -------------------
    one_to_one = resolve.check_one_to_one(ground_truth)
    report["one_to_one_check"] = one_to_one
    logger.info(
        "one-to-one check: %d/%d matched records claimed by >1 S1 entity (holds=%s)",
        one_to_one["n_shared_records"], one_to_one["n_matched_records"], one_to_one["holds"],
    )
    conflict_stage = args.conflict_stage
    if not one_to_one["holds"] and conflict_stage != "none":
        logger.warning(
            "ground truth violates the one-to-one assumption (rate %.4f); "
            "conflict resolution disabled",
            one_to_one["violation_rate"],
        )
        conflict_stage = "none"

    # --- the floor every model has to beat ---------------------------------
    report["all_empty_baseline"] = metrics.all_empty_baseline(ground_truth)
    logger.info("all-empty baseline (singleton fraction) = %.4f", report["all_empty_baseline"])

    # --- split by entity, keeping the full pool for validation -------------
    s1_ids = s1_df["entity_id"].tolist()
    if args.holdout_country:
        train_ids, val_ids = splits.leave_one_country_out(
            s1_ids, country_of, args.holdout_country
        )
    else:
        train_ids, val_ids = splits.holdout_split(
            s1_ids, ground_truth, country_of, args.val_fraction, args.seed
        )
    # Thresholds are hyperparameters: tuning them on the validation slice and
    # then reporting on it inflates the score. Carve a separate tuning slice out
    # of the training entities instead, so validation stays untouched.
    fit_ids, tune_ids = splits.holdout_split(
        train_ids, ground_truth, country_of, args.tune_fraction, args.seed + 1
    )
    report["split"] = splits.split_summary(fit_ids, val_ids, ground_truth, country_of)
    report["split"]["tune"] = {"n_entities": len(tune_ids)}
    logger.info(
        "split: %d fit / %d tune / %d val entities", len(fit_ids), len(tune_ids), len(val_ids)
    )

    # --- stage A: blocking -------------------------------------------------
    logger.info("generating candidates")
    t0 = time.time()
    from src.matching.blocking import CHANNELS

    if args.channels:
        channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    else:
        channels = [c for c in CHANNELS if c != "embedding"]
    if args.embeddings and "embedding" not in channels:
        channels.append("embedding")

    encoder = None
    if args.embeddings:
        from src.matching.embeddings import DEFAULT_MODEL, build_encoder

        model_name = args.embedding_model or DEFAULT_MODEL
        logger.info("loading embedding model %s", model_name)
        encoder_started = time.time()
        encoder = build_encoder(
            model_name, args.embedding_batch_size, args.embedding_device
        )
        logger.info("embedding model ready in %.1fs", time.time() - encoder_started)

    blocking_config = {
        "n_threads": args.blocking_threads,
        "channels": channels,
        "embedding_encoder": encoder,
        "max_candidates": args.max_candidates,
        "max_df_char": args.max_df_char,
    }
    report["blocking_config"] = {"channels": channels, "embeddings": bool(args.embeddings)}
    logger.info("blocking channels: %s", ", ".join(channels))
    candidates = generate_candidates_cached(
        s1_df, pool_df, blocking_config, args.cache_dir, tag="train"
    )
    report["blocking_seconds"] = time.time() - t0
    logger.info("candidate generation complete in %.1fs", report["blocking_seconds"])

    val_gt = {sid: ground_truth.get(sid, set()) for sid in val_ids}
    report["blocking"] = metrics.blocking_report(candidates, val_gt, len(pool_df))
    report["candidate_totals"] = {
        "train_entities": len(candidates),
        "train_candidate_pairs": sum(len(v) for v in candidates.values()),
    }
    report["blocking_channels"] = channel_contribution(candidates, val_gt)
    logger.info(
        "blocking: pair recall %.4f | %.1f candidates/entity | reduction %.5f",
        report["blocking"]["pair_recall"],
        report["blocking"]["candidates_per_entity_mean"],
        report["blocking"]["reduction_ratio"],
    )

    # --- stage B: pairwise model ------------------------------------------
    feature_started = time.time()
    s1_views, pool_views = build_record_views(s1_df), build_record_views(pool_df)
    name_column = "business_name_core" if "business_name_core" in s1_df.columns else "business_name_clean"
    name_idf = build_idf(_tokens(s1_df, name_column), _tokens(pool_df, name_column))
    addr_idf = build_idf(
        _tokens(s1_df, "business_address_clean"), _tokens(pool_df, "business_address_clean")
    )
    logger.info("record views and IDF features ready in %.1fs", time.time() - feature_started)

    embedding = None
    if encoder is not None:
        from src.matching.embeddings import build_embedding_lookup

        started = time.time()
        s1_map, s1_vectors = build_embedding_lookup(s1_df, encoder, args.cache_dir, "train_s1")
        pool_map, pool_vectors = build_embedding_lookup(pool_df, encoder, args.cache_dir, "train_pool")
        embedding = (s1_map, s1_vectors, pool_map, pool_vectors)
        report["embedding"] = {
            "model": getattr(encoder, "model_name", "unknown"),
            "dimensions": int(s1_vectors.shape[1]),
            "encode_seconds": round(time.time() - started, 1),
            "device": getattr(encoder, "device", None),
        }
        logger.info("record embeddings ready in %.1fs", time.time() - started)

    logger.info("featurising training pairs")
    X_train, train_index = _featurise(
        candidates, s1_views, pool_views, name_idf, addr_idf, fit_ids, embedding
    )
    y_train = label_pairs(train_index, ground_truth)
    logger.info("train pairs: %d (%d positive)", len(y_train), int(y_train.sum()))

    fit_started = time.time()
    matcher = PairwiseMatcher(random_state=args.seed).fit(
        X_train, y_train, train_index["s1_entity_id"].to_numpy(dtype=object)
    )
    logger.info("pairwise model and calibration complete in %.1fs", time.time() - fit_started)
    report["feature_importance"] = dict(
        list(matcher.feature_importance(X_train, y_train).items())[:15]
    )

    logger.info("scoring tuning and validation pairs")
    score_started = time.time()
    cache_extra = {
        "seed": args.seed, "channels": channels, "embeddings": bool(args.embeddings),
        "embedding_model": args.embedding_model, "fit_entities": len(fit_ids),
    }
    tune_scored, _, _ = _score_pairs(
        matcher, candidates, s1_views, pool_views, name_idf, addr_idf, tune_ids,
        embedding, args.cache_dir, "tune", cache_extra,
    )
    val_scored, _, val_probs = _score_pairs(
        matcher, candidates, s1_views, pool_views, name_idf, addr_idf, val_ids,
        embedding, args.cache_dir, "val", cache_extra,
    )
    logger.info("tuning and validation scoring complete in %.1fs", time.time() - score_started)

    # --- stages C and D: ablation -----------------------------------------
    tune_gt = {sid: ground_truth.get(sid, set()) for sid in tune_ids}
    tuned = tune_threshold(tune_scored, tune_gt, max_k=args.max_k)
    tuned_tiered = tune_tiered(tune_scored, tune_gt, max_k=args.max_k)
    report["tuned_threshold"] = tuned
    report["tuned_tiered"] = tuned_tiered
    logger.info("tuned threshold: %s", tuned)
    logger.info("tuned tiered:    %s", tuned_tiered)
    for name, params in (("threshold", tuned), ("tiered", tuned_tiered)):
        if params.get("on_grid_edge"):
            # an optimum sitting on an edge means the grid, not the data, chose it
            logger.warning(
                "%s optimum sits on a grid edge for %s; widen the range",
                name, ", ".join(params["on_grid_edge"]),
            )

    # how well the probabilities behave as probabilities, which is exactly what
    # expected_f05 relies on when it weighs adding another candidate
    val_flat_pairs = [(s, c, p) for s, pairs in val_scored.items() for c, p in pairs]
    if val_flat_pairs:
        val_labels = np.array(
            [1 if c in ground_truth.get(s, ()) else 0 for s, c, _ in val_flat_pairs],
            dtype=int,
        )
        report["calibration"] = scorecache.calibration_report(
            [p for _, _, p in val_flat_pairs], val_labels
        )
        logger.info(
            "calibration: brier=%.4f mean_predicted=%.4f observed=%.4f",
            report["calibration"]["brier_score"],
            report["calibration"]["mean_predicted"],
            report["calibration"]["observed_rate"],
        )

    combos = (
        [(args.strategy, conflict_stage)]
        if args.no_ablation
        else [(s, c) for s in ("top1", "threshold", "expected_f05", "tiered")
              for c in (["none"] if conflict_stage == "none" else ["none", "pre", "post"])]
    )

    ablation, chosen_predictions = [], None
    for strategy, stage in combos:
        kwargs = _strategy_kwargs(strategy, tuned, tuned_tiered, args.max_k)
        result, predictions = _evaluate(
            val_scored, candidates, val_gt, country_of, strategy, stage, kwargs
        )
        ablation.append(result)
        logger.info(
            "  %-13s conflict=%-4s -> macro F0.5 = %.4f",
            strategy, stage, result["macro_f05"],
        )
        if (strategy, stage) == (args.strategy, conflict_stage):
            chosen_predictions = predictions

    report["ablation"] = ablation
    best = max(ablation, key=lambda r: r["macro_f05"])
    report["best_validation"] = {
        "strategy": best["strategy"],
        "conflict_stage": best["conflict_stage"],
        "macro_f05": best["macro_f05"],
    }
    report["selected_validation"] = {
        "strategy": args.strategy,
        "conflict_stage": conflict_stage,
        "macro_f05": next(
            r["macro_f05"] for r in ablation
            if r["strategy"] == args.strategy and r["conflict_stage"] == conflict_stage
        ),
    }

    # --- test set prediction -----------------------------------------------
    if args.test_dir:
        logger.info("predicting the test set")
        test_started = time.time()
        test_s1, test_pool = _load_sources(args.test_dir, "test")
        logger.info("test data loaded | source1=%d pool=%d", len(test_s1), len(test_pool))
        test_candidates = generate_candidates_cached(
            test_s1, test_pool, blocking_config, args.cache_dir, tag="test"
        )
        logger.info("test candidate generation complete in %.1fs", time.time() - test_started)

        test_s1_views, test_pool_views = build_record_views(test_s1), build_record_views(test_pool)
        test_ids = test_s1["entity_id"].tolist()
        test_embedding = None
        if encoder is not None:
            from src.matching.embeddings import build_embedding_lookup

            t_s1_map, t_s1_vec = build_embedding_lookup(test_s1, encoder, args.cache_dir, "test_s1")
            t_pool_map, t_pool_vec = build_embedding_lookup(test_pool, encoder, args.cache_dir, "test_pool")
            test_embedding = (t_s1_map, t_s1_vec, t_pool_map, t_pool_vec)

        test_scored, _, _ = _score_pairs(
            matcher, test_candidates, test_s1_views, test_pool_views,
            name_idf, addr_idf, test_ids, test_embedding,
            args.cache_dir, "test", cache_extra,
        )

        working = test_scored
        if conflict_stage == "pre":
            working, _ = resolve.resolve_scored_pairs(working)
        test_predictions = select_matches(
            working, args.strategy,
            **_strategy_kwargs(args.strategy, tuned, tuned_tiered, args.max_k),
        )
        if conflict_stage == "post":
            test_predictions, _ = resolve.resolve_predictions(test_predictions, working)

        logger.info("test scoring and selection complete in %.1fs", time.time() - test_started)

        match_io.write_matching_results(
            os.path.join(args.output_dir, "matching_results.tsv"), test_ids, test_predictions
        )
        match_io.write_candidate_pairs(
            os.path.join(args.output_dir, "candidate_pairs.tsv"), test_ids, test_candidates
        )
        report["candidate_totals"]["test_entities"] = len(test_candidates)
        report["candidate_totals"]["test_candidate_pairs"] = sum(
            len(v) for v in test_candidates.values()
        )
        report["test"] = {
            "n_entities": len(test_ids),
            "n_predicted_ids": sum(len(v) for v in test_predictions.values()),
            "predicted_singleton_rate": sum(1 for v in test_predictions.values() if not v) / max(len(test_ids), 1),
            "countries": sorted(set(test_s1["country_clean"].fillna("UNKNOWN"))),
        }
        logger.info("wrote submission files to %s/", args.output_dir)

    report["total_seconds"] = time.time() - started
    os.makedirs(os.path.dirname(os.path.abspath(args.report_file)), exist_ok=True)
    with open(args.report_file, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    logger.info("report written to %s", args.report_file)

    print("\n=== summary " + "=" * 48)
    print(f"all-empty baseline (must beat) : {report['all_empty_baseline']:.4f}")
    print(f"blocking pair recall           : {report['blocking']['pair_recall']:.4f}")
    print(f"candidates / entity (mean)     : {report['blocking']['candidates_per_entity_mean']:.1f}")
    print(f"best validation macro F0.5     : {best['macro_f05']:.4f} "
          f"({best['strategy']}, conflict={best['conflict_stage']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
