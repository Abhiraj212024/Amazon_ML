#!/usr/bin/env bash
#
# One entrypoint for a full submission run on a remote machine (SageMaker
# Processing Job, a Kaggle notebook's shell, an EC2 box, anything).
#
#   ./scripts/cloud_run.sh --data-dir /opt/ml/processing/input \
#                          --out-dir  /opt/ml/processing/output \
#                          --team-name your_team
#
# Runs the whole chain and stops at the first failure, because every later step
# is worthless if an earlier one is wrong:
#
#   1. data consistency  - a broken slice caps recall at the sampling rate
#   2. matching          - blocking, model, decision, both TSVs
#   3. validation        - a submission that fails validation is not evaluated
#   4. packaging         - the zip in the structure the challenge requires
#
# --data-dir must contain train/ and test/ in the challenge layout.

set -uo pipefail

if [ -n "${ZSH_VERSION:-}" ]; then
    case "${ZSH_EVAL_CONTEXT:-}" in *:file*)
        printf 'error: run this script, do not source it\n' >&2; return 1 ;;
    esac
elif [ -n "${BASH_VERSION:-}" ] && [ "${BASH_SOURCE[0]}" != "$0" ]; then
    printf 'error: run this script, do not source it\n' >&2; return 1
fi
if [ -z "${BASH_VERSION:-}" ]; then
    command -v bash >/dev/null 2>&1 && exec bash "$0" "$@"
    printf 'error: this script needs bash\n' >&2; exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd)"
if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/scripts/run_matching.py" ]; then
    printf 'error: could not locate the repository root\n' >&2; exit 1
fi
cd "$REPO_ROOT" || exit 1

PYTHON="${PYTHON:-python3}"
DATA_DIR=""
OUT_DIR="output"
TEAM_NAME=""
CACHE_DIR=".cache"
CHANNELS=""
EMBEDDINGS=0
EXTRA=""
SKIP_TESTS=0

while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir)   DATA_DIR="${2:-}"; shift 2 ;;
        --out-dir)    OUT_DIR="${2:-}"; shift 2 ;;
        --team-name)  TEAM_NAME="${2:-}"; shift 2 ;;
        --cache-dir)  CACHE_DIR="${2:-}"; shift 2 ;;
        --channels)   CHANNELS="${2:-}"; shift 2 ;;
        --embeddings) EMBEDDINGS=1; shift ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        --extra)      EXTRA="${2:-}"; shift 2 ;;
        -h|--help)
            printf 'usage: %s --data-dir DIR --team-name NAME [--out-dir DIR]\n' "$0"
            printf '          [--cache-dir DIR] [--channels LIST] [--embeddings]\n'
            printf '          [--skip-tests] [--extra "FLAGS"]\n'
            exit 0 ;;
        *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

[ -z "$DATA_DIR" ]  && { printf 'error: --data-dir is required\n' >&2; exit 2; }
[ -z "$TEAM_NAME" ] && { printf 'error: --team-name is required\n' >&2; exit 2; }

TRAIN_DIR="$DATA_DIR/train"
TEST_DIR="$DATA_DIR/test"
for directory in "$TRAIN_DIR" "$TEST_DIR"; do
    [ -d "$directory" ] || { printf 'error: missing %s\n' "$directory" >&2; exit 2; }
done

REPORTS="$OUT_DIR/reports"
mkdir -p "$OUT_DIR" "$REPORTS" || exit 1
RULE="=============================================================="

step() { printf '\n%s\n== %s\n%s\n' "$RULE" "$1" "$RULE"; }
die()  { printf '\nFAILED at: %s\n' "$1" >&2; exit 1; }

printf 'Cloud run\n  repo : %s\n  data : %s\n  out  : %s\n' "$REPO_ROOT" "$DATA_DIR" "$OUT_DIR"
"$PYTHON" - <<'PYEOF'
import os, platform
print(f"  host : {platform.platform()} | {os.cpu_count()} cores")
try:
    page, pages = os.sysconf("SC_PAGE_SIZE"), os.sysconf("SC_PHYS_PAGES")
    print(f"  mem  : {page * pages / 1e9:.1f} GB")
except (ValueError, OSError, AttributeError):
    pass
PYEOF

if [ "$SKIP_TESTS" -eq 0 ]; then
    step "1/5  Tests"
    "$PYTHON" scripts/test_matching.py >"$REPORTS/tests.log" 2>&1 \
        || { tail -25 "$REPORTS/tests.log"; die "tests"; }
    printf '  passed\n'
fi

step "2/5  Data consistency"
# a slice whose labels reference missing records caps recall at the sampling
# rate, and every score after it is meaningless
"$PYTHON" scripts/diagnose_data.py --train-dir "$TRAIN_DIR" \
    --report-file "$REPORTS/data_diagnosis.json" >"$REPORTS/diagnose.log" 2>&1
grep -E '"(recall_ceiling|singleton_rate|unwinnable_rate|source1_records|pool_records)"' \
    "$REPORTS/diagnose.log" | sed 's/^/  /'
grep -q "^OK:" "$REPORTS/diagnose.log" || {
    sed -n '/^PROBLEM/,$p' "$REPORTS/diagnose.log" | sed 's/^/  /'
    die "data consistency (fix the data before spending compute)"
}
printf '  consistent\n'

step "3/5  Matching"
MATCH_FLAGS="--train-dir $TRAIN_DIR --test-dir $TEST_DIR --output-dir $OUT_DIR"
MATCH_FLAGS="$MATCH_FLAGS --cache-dir $CACHE_DIR --report-file $REPORTS/matching.json"
[ -n "$CHANNELS" ]      && MATCH_FLAGS="$MATCH_FLAGS --channels $CHANNELS"
[ "$EMBEDDINGS" -eq 1 ] && MATCH_FLAGS="$MATCH_FLAGS --embeddings"
[ -n "$EXTRA" ]         && MATCH_FLAGS="$MATCH_FLAGS $EXTRA"
printf '  flags: %s\n\n' "$MATCH_FLAGS"

# shellcheck disable=SC2086
"$PYTHON" scripts/run_matching.py $MATCH_FLAGS 2>&1 | tee "$REPORTS/matching.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "matching"

step "4/5  Validation"
"$PYTHON" utils/validate_submission.py \
    --matching "$OUT_DIR/matching_results.tsv" \
    --candidate "$OUT_DIR/candidate_pairs.tsv" \
    --test-dir "$TEST_DIR" --check-ids 2>&1 | tee "$REPORTS/validate.log" | tail -5
[ "${PIPESTATUS[0]}" -eq 0 ] || die "validation"

step "5/5  Submission package"
"$PYTHON" scripts/build_submission.py \
    --team-name "$TEAM_NAME" \
    --output-dir "$OUT_DIR" --test-dir "$TEST_DIR" \
    --report-file "$REPORTS/matching.json" \
    --dest "$OUT_DIR/dist" 2>&1 | tee "$REPORTS/package.log" | tail -14
[ "${PIPESTATUS[0]}" -eq 0 ] || die "packaging"

printf '\n%s\nCLOUD RUN COMPLETE\n' "$RULE"
printf '  leaderboard upload : %s/matching_results.tsv\n' "$OUT_DIR"
printf '  final package      : %s/dist/%s_submission.zip\n' "$OUT_DIR" "$TEAM_NAME"
printf '  logs and reports   : %s\n' "$REPORTS"
printf '\nThe methodology document in the zip is a scaffold. Replace it with your\n'
printf 'own write-up before submitting.\n%s\n' "$RULE"
