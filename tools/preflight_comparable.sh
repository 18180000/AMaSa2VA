#!/bin/bash
set -euo pipefail

MODE=""
RUN_DIR=""
MANIFEST_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --run_dir) RUN_DIR="$2"; shift 2 ;;
    --manifest) MANIFEST_PATH="$2"; shift 2 ;;
    *) echo "[FAIL_FAST] UNKNOWN_ARG $1"; exit 2 ;;
  esac
done

if [[ -z "$MODE" || -z "$RUN_DIR" ]]; then
  echo "Usage: $0 --mode <base|core|feedback> --run_dir <RUN_DIR> [--manifest <file>]"
  exit 2
fi

mkdir -p "$RUN_DIR"
PREFLIGHT_LOG="$RUN_DIR/preflight_env.txt"

[[ "${SEED:-}" == "0" ]] || { echo "[FAIL_FAST] EXPERIMENT_NOT_COMPARABLE seed=${SEED:-}"; exit 1; }
[[ "${PYTHONHASHSEED:-}" == "0" ]] || { echo "[FAIL_FAST] EXPERIMENT_NOT_COMPARABLE pythonhashseed=${PYTHONHASHSEED:-}"; exit 1; }
[[ "${CUBLAS_WORKSPACE_CONFIG:-}" == ":4096:8" ]] || { echo "[FAIL_FAST] EXPERIMENT_NOT_COMPARABLE cublas=${CUBLAS_WORKSPACE_CONFIG:-}"; exit 1; }
[[ "${SA2VA_AUDIT_DETERMINISTIC:-}" == "1" ]] || { echo "[FAIL_FAST] EXPERIMENT_NOT_COMPARABLE deterministic=${SA2VA_AUDIT_DETERMINISTIC:-}"; exit 1; }

if [[ -n "$MANIFEST_PATH" && ! -f "$MANIFEST_PATH" ]]; then
  echo "[FAIL_FAST] EXPERIMENT_NOT_COMPARABLE manifest_missing=$MANIFEST_PATH"
  exit 1
fi

GIT_HASH=$(git -C /home/usergjf/sallm_final rev-parse HEAD 2>/dev/null || echo "UNKNOWN")
EVAL_SHA=$(sha256sum /home/usergjf/sallm_final/scripts/sa2va_eval_ref_vos_patched.py | awk '{print $1}')
VERSIONS=$(/opt/anaconda3/envs/torch/bin/python - <<'PY'
import torch, transformers
print(f"torch={torch.__version__} transformers={transformers.__version__}")
PY
)

{
  echo "mode=$MODE"
  echo "git_hash=$GIT_HASH"
  echo "eval_sha=$EVAL_SHA"
  echo "versions=$VERSIONS"
  echo "seed=${SEED:-}"
  echo "pythonhashseed=${PYTHONHASHSEED:-}"
  echo "cublas=${CUBLAS_WORKSPACE_CONFIG:-}"
  echo "deterministic=${SA2VA_AUDIT_DETERMINISTIC:-}"
  if [[ -n "$MANIFEST_PATH" ]]; then
    echo "manifest=$MANIFEST_PATH"
    echo "manifest_sha256=$(sha256sum "$MANIFEST_PATH" | awk '{print $1}')"
  fi
} > "$PREFLIGHT_LOG"

echo "[PROGRESS] PREFLIGHT_OK mode=$MODE run_dir=$RUN_DIR"
