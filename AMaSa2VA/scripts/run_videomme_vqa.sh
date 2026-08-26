#!/usr/bin/env bash
# Backbone-aware Video-MME launcher.
# usage: run_videomme_vqa.sh GPU OUT_DIR MODEL_PATH BACKBONE MODE [DENYLIST]
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 GPU OUT_DIR MODEL_PATH {internvl|qwen|auto} {base|core|feedback} [DENYLIST]" >&2
  exit 2
fi

GPU_ID="$1"
OUT_DIR="$2"
MODEL_PATH="$3"
BACKBONE="$4"
MODE="$5"
DENYLIST="${6:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

case "$BACKBONE" in
  auto|internvl|qwen) ;;
  *) echo "unknown backbone: $BACKBONE" >&2; exit 2 ;;
esac

unset VQA_MEM_ENABLE QA_MEM_ENABLE PL_DENYLIST_PATH || true
case "$MODE" in
  base)
    export VQA_MEM_ENABLE=0
    export QA_MEM_ENABLE=0
    ;;
  core|feedback)
    export VQA_MEM_ENABLE=0
    export QA_MEM_ENABLE=1
    export QA_QUERY_MODE="${QA_QUERY_MODE:-question_with_options}"
    export QA_FUSION_MODE="${QA_FUSION_MODE:-topk_select}"
    export QA_TOP_K="${QA_TOP_K:-4}"
    export QA_RET_TAU="${QA_RET_TAU:-0.4}"
    export QA_RESTORE_TEMPORAL_ORDER="${QA_RESTORE_TEMPORAL_ORDER:-1}"
    if [[ "$MODE" == feedback ]]; then
      if [[ -z "$DENYLIST" || ! -f "$DENYLIST" ]]; then
        echo "feedback mode requires an existing denylist file" >&2
        exit 2
      fi
      export PL_DENYLIST_PATH="$DENYLIST"
    fi
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

mkdir -p "$OUT_DIR"
exec "$PYTHON_BIN" -u "$ROOT_DIR/projects/amasa2va/evaluation/run_videomme.py" \
  --data Video-MME \
  --mode all \
  --nframe "${VQA_NFRAME:-8}" \
  --gpu "$GPU_ID" \
  --model-path "$MODEL_PATH" \
  --model-name "Sa2VA-${BACKBONE}-${MODE}" \
  --backbone "$BACKBONE" \
  --work-dir "$OUT_DIR"
