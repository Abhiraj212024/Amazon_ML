"""
Run inference from a saved bundle, in resumable shards.

At full scale a single-process run is not viable: blocking the whole test set
against the whole pool takes longer than a Kaggle session allows, and the
feature matrix for every candidate pair does not fit in memory at once. This
splits the Source 1 side into shards and processes them one at a time, so peak
memory is set by the shard size rather than by the dataset, and a session that
is cut short resumes from the shards already finished.

Each shard is blocked against the *whole* pool - sharding the pool as well
would lose candidates.

    python3 scripts/predict.py --bundle model.pkl --test-dir dataset/test \
        --output-dir output --shards 20
"""
import argparse
import json
import logging
import os
import pickle
import sys
import time

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
from src.matching import resolve
from src.matching.blocking import PoolIndex, generate_candidates
from src.matching.decide import select_matches
from src.matching.pair_features import (
    build_pair_table, build_record_views, set_embedding_feature, FEATURE_NAMES,
)
from src.matching.persist import load_bundle
from src.preprocessing.pipeline import preprocess_dataframe

logger = logging.getLogger("predict")


def _load_sources(data_dir, prefix):
    """Prefer the preprocessed parquet, fall back to the raw TSV."""
    frames = {}
    for source in ("source1", "source2", "source3"):
        parquet = os.path.join(data_dir, f"{prefix}_{source}_processed.parquet")
        tsv = os.path.join(data_dir, f"{prefix}_{source}.tsv")
        started = time.time()
        if os.path.exists(parquet):
            frames[source] = pd.read_parquet(parquet)
        elif os.path.exists(tsv):
            frames[source] = preprocess_dataframe(pd.read_csv(tsv, sep="\t", dtype=str))
        else:
            raise FileNotFoundError(f"neither {parquet} nor {tsv} exists")
        logger.info("loaded %s | rows=%d | %.1fs", source, len(frames[source]),
                    time.time() - started)
    pool = pd.concat([frames["source2"], frames["source3"]], ignore_index=True)
    return frames["source1"], pool


def _strategy_kwargs(bundle):
    strategy, max_k = bundle["strategy"], bundle["max_k"]
    tuned, tiered = bundle["tuned_threshold"], bundle["tuned_tiered"]
    if strategy == "threshold":
        return {"t_high": tuned["t_high"], "ratio": tuned["ratio"], "max_k": max_k}
    if strategy == "tiered":
        return {"t_first": tiered["t_first"], "t_rest": tiered["t_rest"],
                "ratio": tiered["ratio"], "max_k": max_k}
    if strategy == "top1":
        return {"t_high": tuned["t_high"]}
    return {"max_k": max_k}


def _shard_bounds(n_entities, shards):
    size = max(1, -(-n_entities // shards))
    return [(start, min(start + size, n_entities)) for start in range(0, n_entities, size)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, help="model bundle from train_model.py")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--prefix", default="test")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--shard-dir", default=None,
                        help="where partial shard results live (default: <output-dir>/shards)")
    parser.add_argument("--shards", type=int, default=20,
                        help="more shards means lower peak memory and finer resume")
    parser.add_argument("--only-shards", default=None,
                        help="comma-separated shard indices, for splitting across sessions")
    parser.add_argument("--blocking-threads", type=int, default=-1)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--report-file", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.time()

    bundle = load_bundle(args.bundle, FEATURE_NAMES)
    matcher = bundle["matcher"]
    blocking_config = dict(bundle["blocking_config"])
    blocking_config["n_threads"] = args.blocking_threads
    channels = blocking_config.get("channels", [])
    logger.info("channels: %s | strategy: %s | conflict: %s",
                ", ".join(channels), bundle["strategy"], bundle["conflict_stage"])

    encoder = None
    if "embedding" in channels:
        from src.matching.embeddings import build_encoder

        model_name = bundle["extra"].get("embedding_model")
        encoder = build_encoder(model_name, device=args.embedding_device)
        blocking_config["embedding_encoder"] = encoder

    s1_df, pool_df = _load_sources(args.test_dir, args.prefix)
    entity_ids = s1_df["entity_id"].astype(str).tolist()

    pool_embedding = None
    if encoder is not None:
        from src.matching.embeddings import build_embedding_lookup

        pool_map, pool_vectors = build_embedding_lookup(
            pool_df, encoder, args.shard_dir or os.path.join(args.output_dir, "shards"),
            "predict_pool",
        )
        pool_embedding = (pool_map, pool_vectors)

    # built once and reused by every shard: without it the pool's vectorisers
    # and inverted indexes are rebuilt per shard, which measured 2.6x the
    # unsharded cost at ten shards and gets worse as shards are added
    blocking_config["pool_index"] = PoolIndex()

    shard_dir = args.shard_dir or os.path.join(args.output_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    bounds = _shard_bounds(len(entity_ids), args.shards)
    wanted = None
    if args.only_shards:
        wanted = {int(v) for v in args.only_shards.split(",") if v.strip()}

    logger.info("%d entities in %d shards of ~%d", len(entity_ids), len(bounds),
                bounds[0][1] - bounds[0][0] if bounds else 0)

    kwargs = _strategy_kwargs(bundle)
    done = 0
    for index, (start, stop) in enumerate(bounds):
        if wanted is not None and index not in wanted:
            continue
        path = os.path.join(shard_dir, f"shard_{index:05d}.pkl")
        if os.path.exists(path):
            logger.info("shard %d/%d already done, skipping", index + 1, len(bounds))
            done += 1
            continue

        shard_started = time.time()
        shard_df = s1_df.iloc[start:stop]
        # the shard is blocked against the WHOLE pool; sharding the pool would
        # drop candidates that only that part of the pool contains
        candidates = generate_candidates(shard_df, pool_df, blocking_config)

        # Views are built only for the candidates this shard actually produced.
        # One per pool record costs about a kilobyte, so materialising the whole
        # pool runs to gigabytes and was the largest avoidable allocation in the
        # run - it is what made this OOM on a laptop regardless of shard count.
        needed = {cand_id for matches in candidates.values() for cand_id in matches}
        pool_views = build_record_views(pool_df, only=needed)
        shard_views = build_record_views(shard_df)
        embedding = None
        if encoder is not None:
            from src.matching.embeddings import build_embedding_lookup, pair_cosines

            shard_map, shard_vectors = build_embedding_lookup(shard_df, encoder)
            embedding = (shard_map, shard_vectors, pool_embedding[0], pool_embedding[1])

        X, pair_index = build_pair_table(
            candidates, shard_views, pool_views, bundle["name_idf"], bundle["addr_idf"]
        )
        if embedding is not None and len(X):
            from src.matching.embeddings import pair_cosines

            set_embedding_feature(X, pair_cosines(pair_index, *embedding))

        scored = {sid: [] for sid in shard_df["entity_id"].astype(str)}
        if len(X):
            probs = matcher.predict_proba(X)
            for s1_id, cand_id, prob in zip(
                pair_index["s1_entity_id"], pair_index["candidate_entity_id"], probs
            ):
                scored[s1_id].append((cand_id, float(prob)))

        working = scored
        if bundle["conflict_stage"] == "pre":
            working, _ = resolve.resolve_scored_pairs(working)
        predictions = select_matches(working, bundle["strategy"], **kwargs)

        # keep only the scores of predicted ids: global conflict resolution runs
        # at merge time and needs nothing else, so the shard file stays small
        predicted_scores = {
            s1_id: [(c, p) for c, p in working.get(s1_id, []) if c in predictions[s1_id]]
            for s1_id in predictions
        }
        payload = {
            "predictions": {k: sorted(v) for k, v in predictions.items()},
            "predicted_scores": predicted_scores,
            "candidates": {k: sorted(v) for k, v in candidates.items()},
        }
        tmp = path + ".tmp"
        with open(tmp, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

        done += 1
        logger.info(
            "shard %d/%d | entities=%d candidates=%d pool_views=%d pairs=%d | %.1fs",
            index + 1, len(bounds), stop - start, len(needed), len(pool_views),
            len(X), time.time() - shard_started,
        )
        del pool_views, shard_views, X, pair_index, candidates

    if wanted is not None:
        logger.info("finished the requested shards; re-run without --only-shards to merge")
        return

    # --- merge -------------------------------------------------------------
    logger.info("merging %d shards", len(bounds))
    predictions, candidates, scored = {}, {}, {}
    missing = []
    for index in range(len(bounds)):
        path = os.path.join(shard_dir, f"shard_{index:05d}.pkl")
        if not os.path.exists(path):
            missing.append(index)
            continue
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        predictions.update({k: set(v) for k, v in payload["predictions"].items()})
        candidates.update(payload["candidates"])
        scored.update(payload["predicted_scores"])

    if missing:
        sys.exit(f"ERROR: shards not finished: {missing}. Re-run to complete them.")

    report = {"n_entities": len(entity_ids), "shards": len(bounds)}
    if bundle["conflict_stage"] == "post":
        # resolution must be global: two entities in different shards can claim
        # the same record, which no single shard can see
        predictions, conflict = resolve.resolve_predictions(predictions, scored)
        report["conflict"] = conflict
        logger.info("global conflict resolution: %s", conflict)

    os.makedirs(args.output_dir, exist_ok=True)
    match_io.write_matching_results(
        os.path.join(args.output_dir, "matching_results.tsv"), entity_ids, predictions
    )
    match_io.write_candidate_pairs(
        os.path.join(args.output_dir, "candidate_pairs.tsv"), entity_ids, candidates
    )

    report["n_predicted_ids"] = sum(len(v) for v in predictions.values())
    report["predicted_singleton_rate"] = (
        sum(1 for v in predictions.values() if not v) / max(len(entity_ids), 1)
    )
    report["total_seconds"] = round(time.time() - started, 1)
    if args.report_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_file)) or ".", exist_ok=True)
        with open(args.report_file, "w") as handle:
            json.dump(report, handle, indent=2, default=str)

    logger.info("wrote submission files to %s", args.output_dir)
    print(f"\nentities={report['n_entities']} "
          f"predicted_ids={report['n_predicted_ids']} "
          f"singleton_rate={report['predicted_singleton_rate']:.4f} "
          f"in {report['total_seconds']:.0f}s")


if __name__ == "__main__":
    main()
