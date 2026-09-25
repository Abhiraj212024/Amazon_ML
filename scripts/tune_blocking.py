"""
Sweep blocking configurations and report the recall / candidate-set trade-off.

Candidate-set size now matters twice over. The organisers say a smaller
candidate set per Source 1 entity ranks higher, and featurising plus scoring is
linear in candidates per entity - the dominant cost once blocking is cached. So
the goal is not maximum recall, it is the smallest candidate set that keeps
recall high enough.

Prints a Pareto table: for each configuration, blocking time, pair recall, and
candidates per entity. Run it on a real slice; the synthetic corpus draws
addresses from a handful of streets and will mislead you about how much a
common n-gram is worth.

    python3 scripts/tune_blocking.py --train-dir dataset_5pct/train
"""
import argparse
import itertools
import json
import logging
import os
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

from src.matching import metrics
from src.matching.blocking import CHANNELS, channel_contribution, generate_candidates, prune_candidates
from src.matching.io import load_ground_truth
from src.preprocessing.pipeline import preprocess_dataframe

logger = logging.getLogger("tune_blocking")


def _load(data_dir, prefix="train"):
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
    return frames["source1"], pd.concat(
        [frames["source2"], frames["source3"]], ignore_index=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--max-entities", type=int, default=20_000,
                        help="cap Source 1 so a sweep stays quick; the pool is kept whole")
    parser.add_argument("--channel-sets", default="all,no_rare_numeric,addr_name_only,addr_only")
    parser.add_argument("--k-values", default="25,15,10")
    parser.add_argument("--max-df-values", default="1.0,0.3,0.1")
    parser.add_argument("--prune-values", default="0,40,25,15,10",
                        help="candidates kept per entity after the union; 0 means no cap")
    parser.add_argument("--report-file", default="reports/blocking_sweep.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    s1_df, pool_df = _load(args.train_dir)
    gt_path = args.ground_truth or os.path.join(args.train_dir, "train_ground_truth.tsv")
    ground_truth = load_ground_truth(gt_path)

    if args.max_entities and len(s1_df) > args.max_entities:
        s1_df = s1_df.sample(args.max_entities, random_state=args.seed).reset_index(drop=True)
    ids = set(s1_df["entity_id"].astype(str))
    gt = {k: v for k, v in ground_truth.items() if k in ids}
    print(f"sweeping on {len(s1_df):,} Source 1 entities against {len(pool_df):,} pool records\n")

    named_sets = {
        "all": list(c for c in CHANNELS if c != "embedding"),
        "no_rare_numeric": ["name_char", "addr_char", "name_word"],
        "addr_name_only": ["name_char", "addr_char"],
        "addr_only": ["addr_char"],
    }
    channel_sets = [c.strip() for c in args.channel_sets.split(",") if c.strip()]
    k_values = [int(v) for v in args.k_values.split(",") if v.strip()]
    max_dfs = [float(v) for v in args.max_df_values.split(",") if v.strip()]
    prunes = [int(v) for v in args.prune_values.split(",") if v.strip()]

    rows = []
    header = f"{'channels':17s} {'k':>3s} {'max_df':>7s} {'prune':>6s} " \
             f"{'time':>7s} {'recall':>7s} {'cands/ent':>10s} {'pairs':>12s}"
    print(header)
    print("-" * len(header))

    for name, k, max_df in itertools.product(channel_sets, k_values, max_dfs):
        channels = named_sets.get(name)
        if channels is None:
            print(f"  unknown channel set {name!r}, skipping")
            continue
        config = {
            "channels": channels, "max_df_char": max_df, "max_df_word": max_df,
            "k_name_char": k, "k_addr_char": k, "k_name_word": k,
            "k_numeric": k, "k_rare_token": k,
        }
        started = time.time()
        candidates = generate_candidates(s1_df, pool_df, config)
        elapsed = time.time() - started

        # pruning is applied afterwards so one blocking pass serves every cap
        for prune in prunes:
            subset = prune_candidates(candidates, prune) if prune else candidates
            report = metrics.blocking_report(subset, gt, len(pool_df))
            row = {
                "channels": name, "k": k, "max_df": max_df, "prune": prune,
                "blocking_seconds": round(elapsed, 1),
                "pair_recall": round(report["pair_recall"], 4),
                "candidates_per_entity": round(report["candidates_per_entity_mean"], 1),
                "n_candidate_pairs": report["n_candidate_pairs"],
            }
            rows.append(row)
            print(f"{name:17s} {k:3d} {max_df:7.2f} {prune or '-':>6} "
                  f"{elapsed:6.1f}s {row['pair_recall']:7.4f} "
                  f"{row['candidates_per_entity']:10.1f} {row['n_candidate_pairs']:12,d}")

    # the knee: for each recall floor, the smallest candidate set that clears it
    print("\nsmallest candidate set at each recall floor:")
    for floor in (0.99, 0.98, 0.97, 0.95, 0.90):
        viable = [r for r in rows if r["pair_recall"] >= floor]
        if not viable:
            print(f"  recall >= {floor:.2f}: none of the swept settings reach it")
            continue
        best = min(viable, key=lambda r: r["candidates_per_entity"])
        print(f"  recall >= {floor:.2f}: {best['candidates_per_entity']:6.1f} cands/entity "
              f"(recall {best['pair_recall']:.4f}) with channels={best['channels']} "
              f"k={best['k']} max_df={best['max_df']} prune={best['prune'] or 'none'}")

    if args.report_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_file)) or ".", exist_ok=True)
        with open(args.report_file, "w") as handle:
            json.dump({"rows": rows, "n_entities": len(s1_df), "n_pool": len(pool_df)},
                      handle, indent=2)
        print(f"\nwrote {args.report_file}")


if __name__ == "__main__":
    main()
