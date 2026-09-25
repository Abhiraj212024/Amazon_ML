#!/usr/bin/env bash
#
# Prove the pipeline runs end to end WITH embeddings, on a slice small enough
# to finish in minutes on a laptop.
#
#   ./scripts/smoke_embeddings.sh --train-dir dataset_5pct/train
#
# This answers "does the embedding path work", not "is it any good". The slice
# is far too small for its score to mean anything; use the full 5% run for that.
#
# Exit code 0 means the embedding path is safe to run at scale.

set -uo pipefail

if [ -n "${ZSH_VERSION:-}" ]; then
    case "${ZSH_EVAL_CONTEXT:-}" in *:file*)
        printf 'error: run this script, do not source it\n' >&2; return 1 ;;
    esac
elif [ -n "${BASH_VERSION:-}" ]; then
    if [ "${BASH_SOURCE[0]}" != "$0" ]; then
        printf 'error: run this script, do not source it\n' >&2; return 1
    fi
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
TRAIN_DIR=""
ENTITIES=400
MODEL=""
DEVICE=""
KEEP=0
WORK_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --train-dir) TRAIN_DIR="${2:-}"; shift 2 ;;
        --entities)  ENTITIES="${2:-}"; shift 2 ;;
        --model)     MODEL="${2:-}"; shift 2 ;;
        --device)    DEVICE="${2:-}"; shift 2 ;;
        --keep)      KEEP=1; shift ;;
        -h|--help)
            printf 'usage: %s --train-dir DIR [--entities N] [--model NAME] [--device mps|cpu|cuda] [--keep]\n' "$0"
            exit 0 ;;
        *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

if [ -z "$TRAIN_DIR" ]; then
    printf 'error: --train-dir is required (e.g. dataset_5pct/train)\n' >&2; exit 2
fi
if [ ! -d "$TRAIN_DIR" ]; then
    printf 'error: no such directory: %s\n' "$TRAIN_DIR" >&2; exit 2
fi

cleanup() {
    if [ -n "$WORK_DIR" ] && [ "$KEEP" -eq 0 ] && [ -d "$WORK_DIR" ]; then
        rm -rf "$WORK_DIR"
    elif [ -n "$WORK_DIR" ] && [ "$KEEP" -eq 1 ]; then
        printf '\nartifacts kept in %s\n' "$WORK_DIR"
    fi
}
trap cleanup EXIT

RULE="=============================================================="
printf 'Embedding smoke test\n  source : %s\n  target : ~%s Source 1 entities\n\n' \
    "$TRAIN_DIR" "$ENTITIES"

# --- dependencies -------------------------------------------------------
if ! "$PYTHON" - <<'PYEOF'
import importlib.util
import sys
missing = [m for m in ("sentence_transformers", "hnswlib")
           if importlib.util.find_spec(m) is None]
if missing:
    print("  missing: " + ", ".join(missing))
    print("  install with: pip install sentence-transformers hnswlib")
    sys.exit(1)
print("  sentence-transformers and hnswlib present")
try:
    import torch
    mps = getattr(torch.backends, "mps", None)
    print(f"  torch {torch.__version__} | mps={bool(mps and mps.is_available())} "
          f"| cuda={torch.cuda.is_available()}")
except ImportError:
    pass
PYEOF
then
    printf '\n%s\nSMOKE TEST CANNOT RUN - install the embedding extras first.\n%s\n' "$RULE" "$RULE"
    exit 1
fi

# --- carve a tiny, label-consistent slice -------------------------------
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/embed_smoke_XXXXXX")"
TINY="$WORK_DIR/train"

TOTAL="$("$PYTHON" - "$TRAIN_DIR" <<'PYEOF'
import os, sys
import pandas as pd
d = sys.argv[1]
parquet = os.path.join(d, "train_source1_processed.parquet")
tsv = os.path.join(d, "train_source1.tsv")
if os.path.exists(parquet):
    print(len(pd.read_parquet(parquet, columns=["entity_id"])))
else:
    print(sum(1 for _ in open(tsv, encoding="utf-8")) - 1)
PYEOF
)"
FRACTION="$("$PYTHON" -c "print(min(1.0, max(0.0005, $ENTITIES / max($TOTAL, 1))))")"
printf '  slicing %s of %s entities (fraction %s)\n' "$ENTITIES" "$TOTAL" "$FRACTION"

# shrinking both sides is the point: encoding cost is driven by the POOL, so
# capping Source 1 alone would leave the slow part untouched
if ! "$PYTHON" scripts/make_subsample.py --train-dir "$TRAIN_DIR" \
        --output-dir "$TINY" --fraction "$FRACTION" >"$WORK_DIR/subsample.log" 2>&1; then
    printf '  FAILED to build the slice\n'; tail -15 "$WORK_DIR/subsample.log" | sed 's/^/      /'
    exit 1
fi
grep -E "^(source1|pool|labels):" "$WORK_DIR/subsample.log" | sed 's/^/  /'

if ! "$PYTHON" scripts/diagnose_data.py --train-dir "$TINY" >"$WORK_DIR/diag.log" 2>&1 \
   || ! grep -q "^OK:" "$WORK_DIR/diag.log"; then
    printf '  FAILED: the slice is not label-consistent\n'
    sed -n '/^PROBLEM/,$p' "$WORK_DIR/diag.log" | sed 's/^/      /'
    exit 1
fi
printf '  slice is label-consistent\n\n'

# --- run it, with embeddings on -----------------------------------------
EXTRA=""
[ -n "$MODEL" ]  && EXTRA="$EXTRA --embedding-model $MODEL"
[ -n "$DEVICE" ] && EXTRA="$EXTRA --embedding-device $DEVICE"

printf 'running the pipeline with --embeddings (first run also downloads the model)\n'
STARTED=$(date +%s)
# shellcheck disable=SC2086
if "$PYTHON" scripts/run_matching.py \
        --train-dir "$TINY" \
        --embeddings \
        --cache-dir "$WORK_DIR/cache" \
        --report-file "$WORK_DIR/report.json" \
        $EXTRA 2>&1 | tee "$WORK_DIR/run.log" \
        | grep -E "embedding|blocking channels|pair recall|macro F0.5|baseline|calibration|tuned"; then
    ELAPSED=$(( $(date +%s) - STARTED ))
else
    printf '\n%s\nSMOKE TEST FAILED - the embedding path does not run.\n%s\n' "$RULE" "$RULE"
    tail -25 "$WORK_DIR/run.log" | sed 's/^/  /'
    exit 1
fi

# --- assert the embedding stage actually did something ------------------
if ! "$PYTHON" - "$WORK_DIR/report.json" <<'PYEOF'
import json, sys
report = json.load(open(sys.argv[1]))

problems = []
if "embedding" not in report:
    problems.append("no embedding section in the report")
else:
    if not report["embedding"].get("dimensions"):
        problems.append("embeddings reported zero dimensions")
if "embedding" not in report.get("blocking_config", {}).get("channels", []):
    problems.append("the embedding channel did not run")

channels = report.get("blocking_channels", {})
if "embedding" in channels:
    recall = channels["embedding"]["recall"]
    print(f"  embedding channel recall: {recall:.4f} "
          f"(unique {channels['embedding']['unique_recall']:.4f})")
    if recall <= 0.0:
        problems.append("the embedding channel retrieved no true pairs")

for problem in problems:
    print(f"  PROBLEM: {problem}")
sys.exit(1 if problems else 0)
PYEOF
then
    printf '\n%s\nSMOKE TEST FAILED - the embedding stage ran but did nothing useful.\n%s\n' "$RULE" "$RULE"
    exit 1
fi

printf '\n  elapsed: %ss\n' "$ELAPSED"
printf '\n%s\n' "$RULE"
printf 'EMBEDDING SMOKE TEST PASSED\n'
printf 'The embedding path works end to end. The score on this slice is\n'
printf 'meaningless - it is far too small. Run the full 5%% slice for that.\n'
printf '%s\n' "$RULE"
