#!/usr/bin/env bash
#
# Runs inside a SageMaker Processing Job container.
#
# SageMaker mounts:
#   /opt/ml/processing/input   <- the dataset, copied from S3
#   /opt/ml/processing/code    <- this repository, as a tarball
#   /opt/ml/processing/output  -> copied back to S3 when the job finishes
#
# Nothing here is SageMaker-specific beyond those paths, so the same script
# runs on a plain EC2 box or locally for debugging.

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-/opt/ml/processing/input}"
CODE_DIR="${CODE_DIR:-/opt/ml/processing/code}"
OUTPUT_DIR="${OUTPUT_DIR:-/opt/ml/processing/output}"

MODE="${MODE:-full}"
TEAM_NAME="${TEAM_NAME:-team}"
SHARDS="${SHARDS:-40}"
TRAIN_SUBDIR="${TRAIN_SUBDIR:-train}"
TEST_SUBDIR="${TEST_SUBDIR:-test}"
MAX_FIT_ENTITIES="${MAX_FIT_ENTITIES:-150000}"
CHANNELS="${CHANNELS:-}"
MAX_CANDIDATES="${MAX_CANDIDATES:-0}"
EXTRA_TRAIN_FLAGS="${EXTRA_TRAIN_FLAGS:-}"

RULE="=============================================================="
step() { printf '\n%s\n== %s\n%s\n' "$RULE" "$1" "$RULE"; }

step "Environment"
python3 --version
python3 - <<'PYEOF'
import os, platform
print(f"  {platform.platform()} | {os.cpu_count()} cores")
try:
    page, pages = os.sysconf("SC_PAGE_SIZE"), os.sysconf("SC_PHYS_PAGES")
    print(f"  memory: {page * pages / 1e9:.1f} GB")
except (ValueError, OSError, AttributeError):
    pass
PYEOF
df -h /tmp | tail -1

step "Unpacking the pipeline"
WORK_DIR=/opt/ml/processing/work
mkdir -p "$WORK_DIR"
tar -xzf "$CODE_DIR"/repo.tar.gz -C "$WORK_DIR"
cd "$WORK_DIR"
ls -1 scripts | head

step "Installing dependencies"
# The SageMaker scikit-learn image already carries numpy/pandas/scikit-learn;
# only the extras are installed here, which keeps job start-up short.
python3 -m pip install --quiet --no-cache-dir rapidfuzz "sparse_dot_topn>=1.1" pyarrow
python3 -c "import rapidfuzz, sparse_dot_topn, pyarrow; print('  extras OK')"

REPORTS="$OUTPUT_DIR/reports"
mkdir -p "$OUTPUT_DIR" "$REPORTS"
CACHE_DIR="${CACHE_DIR:-/tmp/cache}"
mkdir -p "$CACHE_DIR"

step "Data consistency"
# A slice whose labels reference missing records caps recall at the sampling
# rate; every number after that is meaningless, so stop rather than burn the job.
python3 scripts/diagnose_data.py --train-dir "$INPUT_DIR/$TRAIN_SUBDIR" \
    --report-file "$REPORTS/data_diagnosis.json" 2>&1 | tee "$REPORTS/diagnose.log" | tail -20
grep -q "^OK:" "$REPORTS/diagnose.log" || {
    echo "FAILED: training data is not label-consistent"; exit 1; }

TRAIN_FLAGS="--train-dir $INPUT_DIR/$TRAIN_SUBDIR --bundle $OUTPUT_DIR/model.pkl"
TRAIN_FLAGS="$TRAIN_FLAGS --report-file $REPORTS/training.json --cache-dir $CACHE_DIR"
TRAIN_FLAGS="$TRAIN_FLAGS --max-fit-entities $MAX_FIT_ENTITIES"
[ -n "$CHANNELS" ]          && TRAIN_FLAGS="$TRAIN_FLAGS --channels $CHANNELS"
[ "$MAX_CANDIDATES" != "0" ] && TRAIN_FLAGS="$TRAIN_FLAGS --max-candidates $MAX_CANDIDATES"
[ -n "$EXTRA_TRAIN_FLAGS" ] && TRAIN_FLAGS="$TRAIN_FLAGS $EXTRA_TRAIN_FLAGS"

step "Training  (mode=$MODE)"
# shellcheck disable=SC2086
python3 scripts/train_model.py $TRAIN_FLAGS 2>&1 | tee "$REPORTS/training.log"

PREDICT_FLAGS="--bundle $OUTPUT_DIR/model.pkl --test-dir $INPUT_DIR/$TEST_SUBDIR"
PREDICT_FLAGS="$PREDICT_FLAGS --output-dir $OUTPUT_DIR --shards $SHARDS"
PREDICT_FLAGS="$PREDICT_FLAGS --shard-dir $CACHE_DIR/shards --report-file $REPORTS/predict.json"

if [ "$MODE" = "smoke" ]; then
    step "Inference: ONE shard only (smoke mode)"
    # Proves the whole cloud path and measures a shard, so the full run can be
    # sized from a real number instead of an estimate.
    # shellcheck disable=SC2086
    python3 scripts/predict.py $PREDICT_FLAGS --only-shards 0 2>&1 | tee "$REPORTS/predict.log"
    SHARD_SECONDS="$(grep -oE 'shard 1/[0-9]+ .*\| [0-9.]+s' "$REPORTS/predict.log" \
        | grep -oE '[0-9.]+s$' | tr -d 's' | tail -1 || true)"
    printf '\n%s\nSMOKE RUN COMPLETE\n' "$RULE"
    if [ -n "$SHARD_SECONDS" ]; then
        python3 - "$SHARD_SECONDS" "$SHARDS" <<'PYEOF'
import sys
seconds, shards = float(sys.argv[1]), int(sys.argv[2])
total = seconds * shards
print(f"  one shard took {seconds:.1f}s over {shards} shards")
print(f"  projected full inference: {total/60:.1f} min ({total/3600:.2f} h)")
print("  plus one-off pool setup, already paid in this run")
PYEOF
    fi
    printf 'Re-run with MODE=full for the complete submission.\n%s\n' "$RULE"
    exit 0
fi

step "Inference: all $SHARDS shards"
# shellcheck disable=SC2086
python3 scripts/predict.py $PREDICT_FLAGS 2>&1 | tee "$REPORTS/predict.log"

step "Validation"
# A submission that fails validation is not evaluated, so check before packaging.
python3 utils/validate_submission.py \
    --matching "$OUTPUT_DIR/matching_results.tsv" \
    --candidate "$OUTPUT_DIR/candidate_pairs.tsv" \
    --test-dir "$INPUT_DIR/$TEST_SUBDIR" --check-ids 2>&1 | tee "$REPORTS/validate.log" | tail -5

step "Submission package"
python3 scripts/build_submission.py \
    --team-name "$TEAM_NAME" \
    --output-dir "$OUTPUT_DIR" --test-dir "$INPUT_DIR/$TEST_SUBDIR" \
    --report-file "$REPORTS/training.json" \
    --dest "$OUTPUT_DIR/dist" 2>&1 | tee "$REPORTS/package.log" | tail -16

# the model bundle can be large and is not part of the submission
rm -f "$OUTPUT_DIR/model.pkl"

printf '\n%s\nJOB COMPLETE\n' "$RULE"
printf '  leaderboard file : matching_results.tsv\n'
printf '  final package    : dist/%s_submission.zip\n' "$TEAM_NAME"
printf '  logs             : reports/\n%s\n' "$RULE"
