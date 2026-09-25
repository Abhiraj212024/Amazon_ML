"""
Build a smaller but *label-consistent* slice of the training data.

Sampling the three source files independently breaks the ground truth: the
labels still name Source 2/3 records that are no longer in the pool, so pair
recall is capped at roughly the sampling rate and every score is meaningless.

This script samples Source 1 *entities* instead, then keeps:
  * every Source 2/3 record those entities are labelled against, and
  * a proportional number of unrelated pool records as distractors, because a
    thinned pool makes the matching problem artificially easy.

The ground truth is filtered to the sampled entities, so the recall ceiling
stays at 1.0. Verify with scripts/diagnose_data.py afterwards.

    python3 scripts/make_subsample.py \
        --train-dir dataset/train --output-dir dataset_5pct/train --fraction 0.05
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.matching.io import load_ground_truth


def _read_source(data_dir, prefix, source):
    tsv = os.path.join(data_dir, f"{prefix}_{source}.tsv")
    if os.path.exists(tsv):
        return pd.read_csv(tsv, sep="\t", dtype=str), "tsv"
    parquet = os.path.join(data_dir, f"{prefix}_{source}_processed.parquet")
    if os.path.exists(parquet):
        return pd.read_parquet(parquet), "parquet"
    raise FileNotFoundError(f"neither {tsv} nor {parquet} exists")


def _stratify_key(s1_df, ground_truth):
    """Sample within (country, has-matches) cells so the slice keeps their mix."""
    country = (
        s1_df["country"].fillna("UNKNOWN").astype(str)
        if "country" in s1_df.columns
        else pd.Series(["UNKNOWN"] * len(s1_df), index=s1_df.index)
    )
    has_match = s1_df["entity_id"].map(lambda e: bool(ground_truth.get(e)))
    return country + "|" + has_match.astype(str)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fraction", type=float, default=0.05,
                        help="share of Source 1 entities to keep")
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--distractor-multiplier", type=float, default=1.0,
                        help="pool distractors kept, relative to the sampling fraction. "
                             "1.0 keeps the pool-to-entity ratio of the full data; "
                             "raise it to make the slice harder than the full problem")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    s1_df, _ = _read_source(args.train_dir, "train", "source1")
    s2_df, _ = _read_source(args.train_dir, "train", "source2")
    s3_df, _ = _read_source(args.train_dir, "train", "source3")
    gt_path = args.ground_truth or os.path.join(args.train_dir, "train_ground_truth.tsv")
    ground_truth = load_ground_truth(gt_path)

    # --- sample Source 1 entities, stratified ------------------------------
    keys = _stratify_key(s1_df, ground_truth)
    keep_rows = []
    for _, group in s1_df.groupby(keys, sort=True):
        n_keep = max(1, int(round(args.fraction * len(group))))
        keep_rows.extend(rng.choice(group.index.to_numpy(), size=min(n_keep, len(group)),
                                    replace=False).tolist())

    s1_sample = s1_df.loc[sorted(keep_rows)].copy()
    kept_entities = set(s1_sample["entity_id"].astype(str))

    # --- keep every labelled match of those entities -----------------------
    required = set()
    for entity_id in kept_entities:
        required |= ground_truth.get(entity_id, set())

    # --- plus proportional distractors, so the pool is not thinned ---------
    pool_fraction = min(1.0, args.fraction * args.distractor_multiplier)
    sampled_pool = {}
    for name, df in (("source2", s2_df), ("source3", s3_df)):
        ids = df["entity_id"].astype(str)
        is_required = ids.isin(required)
        others = df.index[~is_required].to_numpy()
        n_distractors = min(len(others), int(round(pool_fraction * len(df))))
        chosen = rng.choice(others, size=n_distractors, replace=False) if n_distractors else []
        sampled_pool[name] = df.loc[sorted(set(df.index[is_required]).union(chosen))].copy()

    # --- filter the ground truth to the sampled entities -------------------
    gt_rows = [
        {
            "source1_entity_id": entity_id,
            "matched_entity_ids": ",".join(sorted(ground_truth.get(entity_id, set()))),
        }
        for entity_id in sorted(kept_entities)
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    s1_sample.to_csv(os.path.join(args.output_dir, "train_source1.tsv"), sep="\t", index=False)
    sampled_pool["source2"].to_csv(
        os.path.join(args.output_dir, "train_source2.tsv"), sep="\t", index=False)
    sampled_pool["source3"].to_csv(
        os.path.join(args.output_dir, "train_source3.tsv"), sep="\t", index=False)
    pd.DataFrame(gt_rows, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
        os.path.join(args.output_dir, "train_ground_truth.tsv"), sep="\t", index=False)

    n_pool = len(sampled_pool["source2"]) + len(sampled_pool["source3"])
    print(f"source1: {len(s1_df)} -> {len(s1_sample)}")
    print(f"pool:    {len(s2_df) + len(s3_df)} -> {n_pool} "
          f"({len(required)} required matches + distractors)")
    print(f"labels:  {len(ground_truth)} -> {len(gt_rows)}")
    print(f"\nwritten to {args.output_dir}/")
    print("verify with: python3 scripts/diagnose_data.py --train-dir " + args.output_dir)


if __name__ == "__main__":
    main()
