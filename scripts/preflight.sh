#!/usr/bin/env bash
#
# Pre-flight check: prove the pipeline is correct and the data is consistent
# BEFORE spending money or a Kaggle session quota on a full run.
#
#   ./scripts/preflight.sh                             # code checks only
#   ./scripts/preflight.sh --data-dir dataset/train    # also check your real data
#   ./scripts/preflight.sh --data-dir dataset/train --scale-probe
#
# Exit code 0 means safe to launch. Any non-zero means fix it locally first.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
DATA_DIR=""
SCALE_PROBE=0
KEEP=0
QUICK=0
WORK_DIR=""

FAILURES=0
WARNINGS=0

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
    RED=""; GREEN=""; YELLOW=""; BOLD=""; RESET=""
fi

section() { printf '\n%s== %s ==%s\n' "$BOLD" "$1" "$RESET"; }
ok()      { printf '  %s[ OK ]%s %s\n' "$GREEN" "$RESET" "$1"; }
warn()    { printf '  %s[WARN]%s %s\n' "$YELLOW" "$RESET" "$1"; WARNINGS=$((WARNINGS + 1)); }
fail()    { printf '  %s[FAIL]%s %s\n' "$RED" "$RESET" "$1"; FAILURES=$((FAILURES + 1)); }

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir)    DATA_DIR="${2:-}"; shift 2 ;;
        --scale-probe) SCALE_PROBE=1; shift ;;
        --keep)        KEEP=1; shift ;;
        --quick)       QUICK=1; shift ;;
        -h|--help)     usage ;;
        *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

cleanup() {
    if [ -n "$WORK_DIR" ] && [ "$KEEP" -eq 0 ] && [ -d "$WORK_DIR" ]; then
        rm -rf "$WORK_DIR"
    elif [ -n "$WORK_DIR" ] && [ "$KEEP" -eq 1 ]; then
        printf '\nartifacts kept in %s\n' "$WORK_DIR"
    fi
}
trap cleanup EXIT

printf '%sPre-flight check - Business Entity Resolution pipeline%s\n' "$BOLD" "$RESET"
printf 'repo: %s\n' "$REPO_ROOT"

# --------------------------------------------------------------------------
section "Environment"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    fail "$PYTHON not found. Set PYTHON=/path/to/python3 and retry."
    exit 1
fi
ok "$("$PYTHON" --version 2>&1)"

DEP_LOG="$(mktemp -t preflight_deps_XXXXXX)"
"$PYTHON" - <<'PYEOF' | tee "$DEP_LOG"
import importlib, sys
required = ["numpy", "pandas", "sklearn", "scipy", "rapidfuzz"]
optional = {"sparse_dot_topn": "blocking runs ~20x slower without it",
            "lightgbm": "falls back to sklearn HistGradientBoosting (fine)",
            "pyarrow": "needed only for the processed .parquet cache"}
missing = []
for name in required:
    try:
        module = importlib.import_module(name)
        print(f"  [ OK ] {name} {getattr(module, '__version__', '?')}")
    except ImportError:
        missing.append(name)
        print(f"  [FAIL] {name} is missing")
for name, note in optional.items():
    try:
        module = importlib.import_module(name)
        print(f"  [ OK ] {name} {getattr(module, '__version__', '?')}")
    except ImportError:
        print(f"  [WARN] {name} is missing - {note}")
sys.exit(1 if missing else 0)
PYEOF
DEP_STATUS=${PIPESTATUS[0]}
WARNINGS=$((WARNINGS + $(grep -c '\[WARN\]' "$DEP_LOG")))
rm -f "$DEP_LOG"
if [ "$DEP_STATUS" -ne 0 ]; then
    fail "required packages missing - run: pip install -r requirements.txt"
    exit 1
fi

"$PYTHON" - <<'PYEOF'
import os
cores = os.cpu_count() or 1
print(f"  [INFO] cpu cores: {cores}")
try:
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    print(f"  [INFO] system memory: {total:.1f} GB")
except (ValueError, OSError, AttributeError):
    pass
PYEOF

# --------------------------------------------------------------------------
section "Unit tests"

if "$PYTHON" scripts/test_matching.py >/tmp/preflight_tests.log 2>&1; then
    ok "matching tests passed ($(grep -c 'OK$' /tmp/preflight_tests.log) checks)"
else
    fail "matching tests FAILED"
    tail -20 /tmp/preflight_tests.log | sed 's/^/      /'
fi

if [ -f scripts/test_pipeline.py ]; then
    if "$PYTHON" scripts/test_pipeline.py >/tmp/preflight_pre.log 2>&1; then
        ok "preprocessing tests passed"
    else
        fail "preprocessing tests FAILED"
        tail -20 /tmp/preflight_pre.log | sed 's/^/      /'
    fi
fi

# --------------------------------------------------------------------------
if [ "$QUICK" -eq 0 ]; then
    section "End-to-end smoke test (synthetic data)"

    WORK_DIR="$(mktemp -d -t preflight_XXXXXX)"
    SYNTH="$WORK_DIR/data"
    OUT="$WORK_DIR/output"

    if "$PYTHON" scripts/make_synthetic_data.py \
            --output-dir "$SYNTH" --n-train 600 --n-test 200 \
            >"$WORK_DIR/gen.log" 2>&1; then
        ok "synthetic dataset generated"
    else
        fail "synthetic data generation failed"
        tail -20 "$WORK_DIR/gen.log" | sed 's/^/      /'
    fi

    # the generator is label-consistent, so this must come back clean; if it
    # does not, the checker itself is broken and its verdict on real data is
    # worthless
    if "$PYTHON" scripts/diagnose_data.py --train-dir "$SYNTH/train" \
            >"$WORK_DIR/diag.log" 2>&1 && grep -q "^OK:" "$WORK_DIR/diag.log"; then
        ok "data consistency checker self-test passed"
    else
        fail "consistency checker failed on known-good data - do not trust it on yours"
        tail -10 "$WORK_DIR/diag.log" | sed 's/^/      /'
    fi

    if "$PYTHON" scripts/run_matching.py \
            --train-dir "$SYNTH/train" --test-dir "$SYNTH/test" \
            --output-dir "$OUT" --report-file "$WORK_DIR/report.json" \
            --cache-dir "$WORK_DIR/cache" \
            >"$WORK_DIR/run.log" 2>&1; then
        ok "pipeline ran end-to-end"
        grep -E "baseline|pair recall|best validation" "$WORK_DIR/run.log" \
            | sed 's/^/      /' | tail -4
    else
        fail "pipeline run failed"
        tail -25 "$WORK_DIR/run.log" | sed 's/^/      /'
    fi

    # the cache must return the identical candidate set, not merely run
    if [ -f "$OUT/matching_results.tsv" ]; then
        cp "$OUT/matching_results.tsv" "$WORK_DIR/first_results.tsv"
        if "$PYTHON" scripts/run_matching.py \
                --train-dir "$SYNTH/train" --test-dir "$SYNTH/test" \
                --output-dir "$OUT" --report-file "$WORK_DIR/report2.json" \
                --cache-dir "$WORK_DIR/cache" \
                >"$WORK_DIR/run2.log" 2>&1; then
            if grep -q "loading cached candidates" "$WORK_DIR/run2.log" \
               && cmp -s "$WORK_DIR/first_results.tsv" "$OUT/matching_results.tsv"; then
                ok "candidate cache reused and results reproduced exactly"
            else
                fail "cached run did not reproduce the uncached result"
            fi
        else
            fail "cached re-run failed"
        fi
    fi

    if [ -f utils/validate_submission.py ] && [ -f "$OUT/matching_results.tsv" ]; then
        if "$PYTHON" utils/validate_submission.py \
                --matching "$OUT/matching_results.tsv" \
                --candidate "$OUT/candidate_pairs.tsv" \
                --test-dir "$SYNTH/test" --check-ids \
                >"$WORK_DIR/validate.log" 2>&1; then
            ok "submission files pass the official validator"
        else
            fail "submission files REJECTED by the validator"
            tail -20 "$WORK_DIR/validate.log" | sed 's/^/      /'
        fi
    fi
fi

# --------------------------------------------------------------------------
if [ -n "$DATA_DIR" ]; then
    section "Your data: $DATA_DIR"

    if [ ! -d "$DATA_DIR" ]; then
        fail "directory not found: $DATA_DIR"
    else
        DIAG_LOG="$(mktemp -t preflight_diag_XXXXXX)"
        if "$PYTHON" scripts/diagnose_data.py --train-dir "$DATA_DIR" \
                >"$DIAG_LOG" 2>&1; then
            grep -E '"(recall_ceiling|singleton_rate|unwinnable_rate|source1_records|pool_records)"' \
                "$DIAG_LOG" | sed 's/^/      /'
            if grep -q "^OK:" "$DIAG_LOG"; then
                ok "ground truth and sources are consistent"
                # predicting nothing scores the singleton rate; that is the floor
                "$PYTHON" - "$DIAG_LOG" <<'PYEOF'
import json, re, sys
text = open(sys.argv[1]).read()
match = re.search(r'\{.*\}', text, re.S)
if match:
    rate = json.loads(match.group(0)).get("singleton_rate", 0.0)
    print(f"      [INFO] all-empty baseline to beat: {rate:.4f}")
PYEOF
            else
                fail "ground truth references records missing from the sources"
                sed -n '/^PROBLEM/,$p' "$DIAG_LOG" | sed 's/^/      /'
                printf '      %sAny score from this data is meaningless. Fix it with:%s\n' "$BOLD" "$RESET"
                printf '      python3 scripts/make_subsample.py --train-dir <full> --output-dir <slice> --fraction 0.05\n'
            fi
        else
            fail "consistency check could not run"
            tail -15 "$DIAG_LOG" | sed 's/^/      /'
        fi
        rm -f "$DIAG_LOG"
    fi

    if [ "$SCALE_PROBE" -eq 1 ]; then
        section "Scale probe (how long will the full run take?)"
        if ! "$PYTHON" scripts/scale_probe.py --train-dir "$DATA_DIR"; then
            warn "scale probe failed; size the instance from a manual timing instead"
        fi
    fi
else
    printf '\n  %s[INFO]%s pass --data-dir dataset/train to check your real data too\n' \
        "$YELLOW" "$RESET"
fi

# --------------------------------------------------------------------------
printf '\n%s%s%s\n' "$BOLD" "$(printf '=%.0s' {1..62})" "$RESET"
if [ "$FAILURES" -eq 0 ]; then
    printf '%sPREFLIGHT PASSED%s  (%d warnings)\n' "$GREEN" "$RESET" "$WARNINGS"
    printf 'Safe to launch on Kaggle or AWS.\n'
    printf '%s\n' "$(printf '=%.0s' {1..62})"
    exit 0
else
    printf '%sPREFLIGHT FAILED%s  (%d failures, %d warnings)\n' \
        "$RED" "$RESET" "$FAILURES" "$WARNINGS"
    printf 'Fix these locally before spending compute.\n'
    printf '%s\n' "$(printf '=%.0s' {1..62})"
    exit 1
fi
