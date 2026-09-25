"""
Estimate what a full-scale run will cost, by timing blocking on several
subsample sizes and fitting the growth curve.

Blocking compares every Source 1 entity against every pool record within a
country, so its cost grows roughly with the product of the two - close to
quadratic in the sampling fraction. That means a 5% sample does about 0.25% of
the full-data work, and extrapolating linearly from it underestimates the real
run badly.

    python3 scripts/scale_probe.py --train-dir dataset/train

Use the printed exponent and projection to decide whether the full run fits on
your machine or needs a bigger box.
"""
import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
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

from src.matching.blocking import generate_candidates
from src.preprocessing.pipeline import preprocess_dataframe

logger = logging.getLogger("scale_probe")


def _time_blocking(sample_dir, threads):
    frames = {}
    for source in ("source1", "source2", "source3"):
        path = os.path.join(sample_dir, f"train_{source}.tsv")
        frames[source] = preprocess_dataframe(pd.read_csv(path, sep="\t", dtype=str))
    s1 = frames["source1"]
    pool = pd.concat([frames["source2"], frames["source3"]], ignore_index=True)

    started = time.time()
    candidates = generate_candidates(s1, pool, {"n_threads": threads})
    elapsed = time.time() - started

    n_candidates = sum(len(v) for v in candidates.values())
    return elapsed, len(s1), len(pool), n_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--fractions", default="0.02,0.05,0.10",
                        help="comma-separated sampling fractions to time")
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    fractions = [float(f) for f in args.fractions.split(",")]
    workdir = tempfile.mkdtemp(prefix="scale_probe_")
    rows = []

    try:
        for fraction in fractions:
            sample_dir = os.path.join(workdir, f"f{fraction}")
            subprocess.run(
                [sys.executable, os.path.join(os.path.dirname(__file__), "make_subsample.py"),
                 "--train-dir", args.train_dir, "--output-dir", sample_dir,
                 "--fraction", str(fraction), "--seed", str(args.seed)],
                check=True, stdout=subprocess.DEVNULL,
            )
            elapsed, n_s1, n_pool, n_cand = _time_blocking(sample_dir, args.threads)
            rows.append((fraction, n_s1, n_pool, elapsed, n_cand))
            print(f"  fraction={fraction:<6.3f} s1={n_s1:>8d} pool={n_pool:>9d} "
                  f"blocking={elapsed:>8.1f}s candidates={n_cand:>10d}")
            shutil.rmtree(sample_dir, ignore_errors=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if len(rows) < 2:
        print("\nneed at least two fractions to fit a curve")
        return

    # fit time ~ fraction**exponent in log space
    f = np.array([r[0] for r in rows], dtype=float)
    t = np.array([r[3] for r in rows], dtype=float)
    if (t <= 0).any():
        print("\nsamples ran too fast to fit a reliable curve; try larger fractions")
        return

    exponent, intercept = np.polyfit(np.log(f), np.log(t), 1)
    projected = float(np.exp(intercept))  # value at fraction = 1.0

    print(f"\n  growth exponent : {exponent:.2f}  "
          f"(1.0 = linear, 2.0 = quadratic in the sampling fraction)")
    print(f"  projected full-data blocking: {projected:.0f}s "
          f"({projected / 60:.1f} min, {projected / 3600:.2f} h)")
    print("\n  This covers blocking only, on this machine, at these thread settings.")
    print("  It scales close to linearly with core count, so a box with N times")
    print("  the cores should land near projected/N.")
    if exponent < 1.3:
        print("  NOTE: the exponent looks low. If the fractions were tiny the timing")
        print("        is dominated by fixed costs; re-run with larger fractions.")


if __name__ == "__main__":
    main()
