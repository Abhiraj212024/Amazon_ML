"""
Check that the ground truth and the source files describe the same world.

Run this before trusting any score. The commonest way to get a meaningless
result is to subsample the three source files independently of the ground
truth: the labels still reference Source 2/3 records that are no longer in the
pool, so blocking cannot possibly retrieve them and pair recall collapses to
roughly the sampling rate, however good the pipeline is.

    python3 scripts/diagnose_data.py --train-dir dataset/train

The number to look at is `recall_ceiling`. It is the best pair recall any
blocking strategy could achieve on these files. If it is far below 1.0 the data
is inconsistent, not the model.
"""
import argparse
import json
import logging
import os
import sys
import time
from collections import Counter

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.matching.io import load_ground_truth

logger = logging.getLogger("diagnose_data")


def _read_ids(data_dir, prefix, source):
    """entity_id column from the processed parquet if present, else the raw TSV."""
    started = time.time()
    parquet = os.path.join(data_dir, f"{prefix}_{source}_processed.parquet")
    if os.path.exists(parquet):
        ids = set(pd.read_parquet(parquet, columns=["entity_id"])["entity_id"].astype(str))
        logger.info(
            "loaded %s | records=%d | elapsed=%.1fs",
            parquet, len(ids), time.time() - started,
        )
        return ids
    tsv = os.path.join(data_dir, f"{prefix}_{source}.tsv")
    if os.path.exists(tsv):
        ids = set(pd.read_csv(tsv, sep="\t", dtype=str, usecols=["entity_id"])["entity_id"].astype(str))
        logger.info(
            "loaded %s | records=%d | elapsed=%.1fs",
            tsv, len(ids), time.time() - started,
        )
        return ids
    raise FileNotFoundError(f"neither {parquet} nor {tsv} exists")


def diagnose(train_dir, ground_truth_path=None):
    started = time.time()
    if ground_truth_path:
        gt_path = ground_truth_path
    else:
        local_gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
        repo_gt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "train",
            "train_ground_truth.tsv",
        )
        gt_path = local_gt_path if os.path.exists(local_gt_path) else repo_gt_path
    logger.info("loading ground truth from %s", os.path.abspath(gt_path))
    ground_truth = load_ground_truth(gt_path)
    logger.info("ground truth loaded | rows=%d", len(ground_truth))

    logger.info("loading source identifiers from %s", os.path.abspath(train_dir))
    s1_ids = _read_ids(train_dir, "train", "source1")
    s2_ids = _read_ids(train_dir, "train", "source2")
    s3_ids = _read_ids(train_dir, "train", "source3")
    pool_ids = s2_ids | s3_ids
    logger.info(
        "source identifiers loaded | source1=%d source2=%d source3=%d pool=%d",
        len(s1_ids), len(s2_ids), len(s3_ids), len(pool_ids),
    )

    # --- do the labelled entities exist in source1? ------------------------
    gt_s1 = set(ground_truth)
    s1_without_labels = s1_ids - gt_s1
    labels_without_s1 = gt_s1 - s1_ids

    # --- do the referenced matches exist in the pool? ----------------------
    total_pairs = present_pairs = 0
    missing_by_prefix = Counter()
    entities_fully_present = entities_with_truth = 0

    logger.info("checking labelled matches against the pool")
    total_labels = len(ground_truth)
    for label_number, (s1_id, matches) in enumerate(ground_truth.items(), start=1):
        if label_number % 250000 == 0:
            logger.info("checked %d/%d labelled entities", label_number, total_labels)
        if s1_id not in s1_ids:
            continue
        if not matches:
            continue
        entities_with_truth += 1
        present = 0
        for match_id in matches:
            total_pairs += 1
            if match_id in pool_ids:
                present_pairs += 1
                present += 1
            else:
                missing_by_prefix[match_id.split("-")[0]] += 1
        if present == len(matches):
            entities_fully_present += 1

    scored_entities = len(s1_ids & gt_s1)
    singletons = sum(
        1 for s1_id in s1_ids & gt_s1 if not ground_truth[s1_id]
    )
    # a labelled entity whose matches have all vanished now scores 0 no matter
    # what, because the label still says it has matches
    unwinnable = sum(
        1 for s1_id in s1_ids & gt_s1
        if ground_truth[s1_id] and not (ground_truth[s1_id] & pool_ids)
    )

    report = {
        "source1_records": len(s1_ids),
        "pool_records": len(pool_ids),
        "ground_truth_rows": len(ground_truth),
        "source1_without_label": len(s1_without_labels),
        "labels_without_source1_record": len(labels_without_s1),
        "true_pairs_referenced": total_pairs,
        "true_pairs_present_in_pool": present_pairs,
        "recall_ceiling": present_pairs / total_pairs if total_pairs else 1.0,
        "entities_with_all_matches_present": (
            entities_fully_present / entities_with_truth if entities_with_truth else 1.0
        ),
        "missing_matches_by_source": dict(missing_by_prefix),
        "scored_entities": scored_entities,
        "singleton_rate": singletons / scored_entities if scored_entities else 0.0,
        "unwinnable_entities": unwinnable,
        "unwinnable_rate": unwinnable / scored_entities if scored_entities else 0.0,
    }
    logger.info("diagnosis complete in %.1fs", time.time() - started)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--report-file", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = diagnose(args.train_dir, args.ground_truth)
    print(json.dumps(report, indent=2))

    ceiling = report["recall_ceiling"]
    print("\n" + "=" * 62)
    if ceiling < 0.95:
        print(f"PROBLEM: only {ceiling:.1%} of the labelled matches exist in the pool.")
        print("No blocking strategy can exceed that recall on these files, and")
        print(f"{report['unwinnable_rate']:.1%} of entities cannot score above 0 at all.")
        print("If you subsampled the sources, redo it with scripts/make_subsample.py,")
        print("which keeps each sampled entity's full match set.")
    else:
        print(f"OK: {ceiling:.1%} of labelled matches are present. Data is consistent.")
    print("=" * 62)

    if args.report_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_file)), exist_ok=True)
        with open(args.report_file, "w") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
