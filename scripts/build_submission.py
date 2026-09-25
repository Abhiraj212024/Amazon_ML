"""
Package the final submission zip in the structure the challenge requires.

    <team_name>_submission.zip
    |-- output/
    |   |-- matching_results.tsv
    |   `-- candidate_pairs.tsv
    |-- code/
    |   `-- business_entity_resolution/
    |       |-- src/
    |       |-- README.md
    |       `-- requirements.txt      (pinned)
    `-- Documentation_template.md

Two rules from the challenge shape this script. Submissions that fail
validation are not evaluated, so the two TSVs are validated first and the
archive is refused if they do not pass. And dependencies must be pinned, so
requirements are written from the versions actually installed in the
environment that produced the outputs, not from the repo's loose ranges.

    python3 scripts/build_submission.py \
        --team-name your_team \
        --output-dir output --test-dir dataset/test \
        --report-file reports/matching_full.json
"""
import argparse
import importlib.metadata as importlib_metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import date

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# Everything the pipeline imports at runtime. Pinned from the live environment
# so the package records what actually produced the outputs.
RUNTIME_PACKAGES = [
    "pandas", "numpy", "pyarrow", "scikit-learn", "scipy", "rapidfuzz",
    "sparse_dot_topn",
]
OPTIONAL_PACKAGES = ["lightgbm", "sentence-transformers", "hnswlib", "torch"]

# The spec says "Put all source under src/", so scripts and utils are packaged
# beneath src/ rather than beside it. The scripts locate the project root by
# searching upward for src/matching, so they run unchanged in either layout.
PACKAGED_DIRS = {
    "src/matching": "src/matching",
    "src/preprocessing": "src/preprocessing",
    "scripts": "src/scripts",
    "utils": "src/utils",
}


def _validate_outputs(output_dir, test_dir):
    """
    Run the official validator; a failing submission is not evaluated.

    Returns (ok, detail). Validation needs the test set, because the central
    rule is that every test Source 1 entity appears exactly once - so without
    --test-dir it is skipped rather than reported as a pass.
    """
    validator = os.path.join(REPO_ROOT, "utils", "validate_submission.py")
    if not os.path.exists(validator):
        return None, "validator not found in the repo"
    if not test_dir:
        return None, "no --test-dir given, so the outputs were NOT validated"

    command = [
        sys.executable, validator,
        "--matching", os.path.join(output_dir, "matching_results.tsv"),
        "--candidate", os.path.join(output_dir, "candidate_pairs.tsv"),
    ]
    if test_dir:
        command += ["--test-dir", test_dir, "--check-ids"]

    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def _pinned_requirements():
    lines = [
        "# Pinned from the environment that produced the submitted outputs.",
        f"# Generated {date.today().isoformat()} on Python "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "",
    ]
    missing = []
    for package in RUNTIME_PACKAGES:
        try:
            lines.append(f"{package}=={importlib_metadata.version(package)}")
        except importlib_metadata.PackageNotFoundError:
            missing.append(package)

    optional = []
    for package in OPTIONAL_PACKAGES:
        try:
            optional.append(f"{package}=={importlib_metadata.version(package)}")
        except importlib_metadata.PackageNotFoundError:
            continue
    if optional:
        lines += ["", "# Optional: only needed for the embedding stage (--embeddings)."] + optional

    return "\n".join(lines) + "\n", missing


def _run_readme(report):
    channels = report.get("blocking_config", {}).get("channels", [])
    embeddings = report.get("blocking_config", {}).get("embeddings", False)
    flags = ["    --train-dir dataset/train \\", "    --test-dir dataset/test \\",
             "    --output-dir output \\", "    --cache-dir .cache"]
    if channels:
        flags.insert(0, f"    --channels {','.join(channels)} \\")
    if embeddings:
        flags.append(" \\\n    --embeddings")

    return f"""# Business Entity Resolution - runnable pipeline

Regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv`
from the challenge data using only what is in this folder.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

Place the challenge data as:

```
dataset/train/train_source1.tsv   train_source2.tsv   train_source3.tsv   train_ground_truth.tsv
dataset/test/test_source1.tsv     test_source2.tsv    test_source3.tsv
```

## Check the data before running

Subsampling the sources independently of the ground truth silently caps recall
at the sampling rate, so verify consistency first:

```bash
python3 src/scripts/diagnose_data.py --train-dir dataset/train
```

`recall_ceiling` must be ~1.0.

## Reproduce the submitted outputs

```bash
python3 src/scripts/run_matching.py \\
{chr(10).join(flags)}
```

Then validate:

```bash
python3 src/utils/validate_submission.py \\
    --matching output/matching_results.tsv \\
    --candidate output/candidate_pairs.tsv \\
    --test-dir dataset/test --check-ids
```

## Tests

```bash
python3 src/scripts/test_matching.py
```

## Layout

All source is under `src/`: `src/matching` and `src/preprocessing` are the
packages, `src/scripts` the entry points, `src/utils` the standalone helpers.
The entry points locate the project root by searching upward for
`src/matching`, so they run from this folder without any path setup.
"""


def _methodology_appendix(report):
    """Measured numbers for the write-up, so the prose cites real results."""
    if not report:
        return "\n(no run report supplied, so no measured numbers are included)\n"

    lines = ["", "## Appendix: measured results (auto-generated)", "",
             "Numbers below are read directly from the run report; the prose above",
             "is written by hand.", ""]

    baseline = report.get("all_empty_baseline")
    if baseline is not None:
        lines.append(f"- **All-empty baseline** (singleton fraction, the floor): {baseline:.4f}")

    blocking = report.get("blocking", {})
    if blocking:
        lines += [
            f"- **Blocking pair recall**: {blocking.get('pair_recall', 0):.4f}",
            f"- **Candidates per entity (mean)**: {blocking.get('candidates_per_entity_mean', 0):.1f}",
            f"- **Reduction ratio**: {blocking.get('reduction_ratio', 0):.5f}",
        ]

    best = report.get("best_validation", {})
    if best:
        lines.append(
            f"- **Best validation macro F0.5**: {best.get('macro_f05', 0):.4f} "
            f"({best.get('strategy')}, conflict={best.get('conflict_stage')})"
        )

    channels = report.get("blocking_channels", {})
    if channels:
        lines += ["", "### Blocking channel contribution", "",
                  "| Channel | Recall | Unique recall |", "| --- | --- | --- |"]
        for name, stats in channels.items():
            lines.append(
                f"| `{name}` | {stats['recall']:.4f} | {stats['unique_recall']:.4f} |"
            )

    ablation = report.get("ablation", [])
    if ablation:
        lines += ["", "### Decision strategy / conflict stage ablation", "",
                  "| Strategy | Conflict stage | Macro F0.5 |", "| --- | --- | --- |"]
        for row in ablation:
            lines.append(
                f"| {row['strategy']} | {row['conflict_stage']} | {row['macro_f05']:.4f} |"
            )
        errors = max(ablation, key=lambda r: r["macro_f05"]).get("errors", {})
        if errors:
            lines += ["", "### Error attribution (best setting)", ""]
            for key, value in errors.items():
                lines.append(f"- `{key}`: {value}")

    calibration = report.get("calibration", {})
    if calibration:
        lines += ["", "### Calibration", "",
                  f"- Brier score: {calibration.get('brier_score', 0):.4f}",
                  f"- Mean predicted: {calibration.get('mean_predicted', 0):.4f}",
                  f"- Observed rate: {calibration.get('observed_rate', 0):.4f}"]

    embedding = report.get("embedding")
    if embedding:
        lines += ["", "### Embeddings", "",
                  f"- Model: `{embedding.get('model')}`",
                  f"- Dimensions: {embedding.get('dimensions')}",
                  f"- Device: {embedding.get('device')}"]

    return "\n".join(lines) + "\n"


DEFAULT_DOC = """# Methodology

<!-- Replace this scaffold with your own write-up. The appendix below is
     generated from the run report and contains the measured numbers. -->

## Methodology used

The task is per-Source-1-entity retrieval scored by a macro-average of F0.5.
Expanding F-beta for a predicted set of size m, a true set of size n and c
correct predictions gives `F0.5 = 1.25c / (m + 0.25n)`, and the pipeline is
built around the asymmetry that implies.

Stages: blocking -> pairwise model -> conflict resolution -> set selection.

## Candidate generation / blocking strategy

A union of independent channels, blocked by country, so a true pair only has to
survive one of them. Country is treated as an open set of string labels and
records with a missing country are pooled into every block.

## Model architecture and feature engineering

A gradient-boosted pairwise classifier over string-similarity, IDF-weighted
token-overlap, numeric-address agreement and entity-context features, with
isotonic calibration. Negatives are drawn from the pipeline's own blocking
output.

## Any other relevant information

Conflict resolution exploits Source 1 being deduplicated: a Source 2/3 record
belongs to at most one Source 1 entity, so competing claims are resolved in
favour of the higher-scoring one. The assumption is verified against the
training ground truth before it is applied.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--output-dir", default="output",
                        help="directory holding the two submission TSVs")
    parser.add_argument("--test-dir", default=None,
                        help="test data, used to validate IDs before packaging")
    parser.add_argument("--report-file", default=None,
                        help="run report, used to fill the methodology appendix")
    parser.add_argument("--documentation", default=None,
                        help="the challenge's Documentation_template.md, used as the base")
    parser.add_argument("--dest", default="dist")
    parser.add_argument("--wrap-in-folder", action="store_true",
                        help="nest everything under <team_name>_submission/ inside the zip. "
                             "Off by default: the spec's tree puts output/, code/ and "
                             "Documentation_template.md at the root of the archive.")
    parser.add_argument("--skip-validation", action="store_true",
                        help="package even if the validator fails (not recommended)")
    args = parser.parse_args()

    print("Building submission package\n")

    # --- the two output files must exist and pass the validator ----------
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        path = os.path.join(args.output_dir, name)
        if not os.path.exists(path):
            sys.exit(f"ERROR: missing {path}. Run scripts/run_matching.py --test-dir first.")
        print(f"  found {path} ({os.path.getsize(path):,} bytes)")

    ok, detail = _validate_outputs(args.output_dir, args.test_dir)
    if ok is None:
        print(f"\n  validator: SKIPPED - {detail}")
        print("    pass --test-dir to check the outputs before packaging")
    else:
        print(f"\n  validator: {'PASS' if ok else 'FAIL'}")
    if ok is False:
        print("\n".join(f"    {line}" for line in detail.splitlines()[-15:]))
        if not args.skip_validation:
            sys.exit("\nERROR: refusing to package outputs that fail validation. "
                     "A submission that fails validation is not evaluated.")
        print("    packaging anyway because --skip-validation was passed")

    report = {}
    if args.report_file and os.path.exists(args.report_file):
        with open(args.report_file) as handle:
            report = json.load(handle)
        print(f"  read run report from {args.report_file}")

    requirements, missing = _pinned_requirements()
    if missing:
        print(f"  WARNING: not installed, so not pinned: {', '.join(missing)}")

    # --- assemble ---------------------------------------------------------
    os.makedirs(args.dest, exist_ok=True)
    archive = os.path.join(args.dest, f"{args.team_name}_submission.zip")

    with tempfile.TemporaryDirectory() as staging:
        # the spec's tree puts output/, code/ and the document at the archive
        # root, so no wrapper folder unless explicitly asked for
        root = os.path.join(staging, f"{args.team_name}_submission") if args.wrap_in_folder \
            else os.path.join(staging, "submission")
        code_root = os.path.join(root, "code", "business_entity_resolution")
        os.makedirs(os.path.join(root, "output"), exist_ok=True)
        os.makedirs(code_root, exist_ok=True)

        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            shutil.copy2(os.path.join(args.output_dir, name), os.path.join(root, "output", name))

        for source_rel, packaged_rel in PACKAGED_DIRS.items():
            source = os.path.join(REPO_ROOT, source_rel)
            if os.path.isdir(source):
                shutil.copytree(
                    source, os.path.join(code_root, packaged_rel),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
                )
        # make src/ a package root even though the subpackages carry __init__
        init = os.path.join(code_root, "src", "__init__.py")
        if not os.path.exists(init):
            with open(init, "w") as handle:
                handle.write("")

        with open(os.path.join(code_root, "requirements.txt"), "w") as handle:
            handle.write(requirements)
        with open(os.path.join(code_root, "README.md"), "w") as handle:
            handle.write(_run_readme(report))

        base_doc = DEFAULT_DOC
        if args.documentation and os.path.exists(args.documentation):
            with open(args.documentation) as handle:
                base_doc = handle.read()
            print(f"  using {args.documentation} as the documentation base")
        else:
            print("  no template supplied; wrote a scaffold you must replace with your write-up")
        with open(os.path.join(root, "Documentation_template.md"), "w") as handle:
            handle.write(base_doc.rstrip() + "\n" + _methodology_appendix(report))

        base = staging if args.wrap_in_folder else root
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for folder, _, files in os.walk(root):
                for name in sorted(files):
                    full = os.path.join(folder, name)
                    zf.write(full, os.path.relpath(full, base))

    print(f"\n  wrote {archive} ({os.path.getsize(archive):,} bytes)")
    with zipfile.ZipFile(archive) as zf:
        entries = zf.namelist()
    print(f"  {len(entries)} entries\n")

    prefix = f"{args.team_name}_submission/" if args.wrap_in_folder else ""
    required = [
        f"{prefix}output/matching_results.tsv",
        f"{prefix}output/candidate_pairs.tsv",
        f"{prefix}code/business_entity_resolution/README.md",
        f"{prefix}code/business_entity_resolution/requirements.txt",
        f"{prefix}Documentation_template.md",
    ]
    print("  structure required by the challenge:")
    missing = False
    for entry in required:
        present = entry in entries
        missing = missing or not present
        print(f"    [{'OK' if present else '--'}] {entry}")
    source_files = [e for e in entries if f"{prefix}code/business_entity_resolution/src/" in e]
    print(f"    [{'OK' if source_files else '--'}] "
          f"code/business_entity_resolution/src/ ({len(source_files)} files)")
    if missing or not source_files:
        sys.exit("\nERROR: the archive is missing required entries")

    print("\nBefore submitting, replace the methodology scaffold with your own write-up.")


if __name__ == "__main__":
    main()
