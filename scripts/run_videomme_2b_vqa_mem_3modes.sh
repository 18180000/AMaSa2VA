#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/torch/bin/python}"
GPU_ID="${1:-0}"
OUT_ROOT="${2:-/data0/multi_rvos_benchmark_v1/runs/videomme_2b_vqa_mem_3modes_v1}"
MODEL_PATH="${3:-/data0/pretrained/Sa2VA-InternVL3-2B}"

source "$ROOT_DIR/configs/modes.sh"

FULL_LOG="$OUT_ROOT/logs/full.log"
mkdir -p "$(dirname "$FULL_LOG")"

clear_vqa_mem_env() {
  unset VQA_MEM_ENABLE || true
  unset VQA_MEM_TOPK || true
  unset VQA_MEM_TAU || true
  unset VQA_MEM_ALPHA || true
}

apply_vqa_mem_env() {
  local mode="$1"
  clear_vqa_mem_env
  case "$mode" in
    base)
      export VQA_MEM_ENABLE="0"
      ;;
    core|feedback)
      export VQA_MEM_ENABLE="1"
      export VQA_MEM_TOPK="${VQA_MEM_TOPK:-4}"
      export VQA_MEM_TAU="${VQA_MEM_TAU:-0.4}"
      export VQA_MEM_ALPHA="${VQA_MEM_ALPHA:-0.5}"
      ;;
    *)
      echo "[FAIL_FAST] UNKNOWN_MODE mode=$mode" | tee -a "$FULL_LOG"
      return 2
      ;;
  esac
}

run_rating() {
  local mode="$1"
  local xlsx="$OUT_ROOT/$mode/Sa2VA-InternVL3-2B-LOCAL-MEM/Sa2VA-InternVL3-2B-LOCAL-MEM_Video-MME_8frame_nopack_nosubs.xlsx"
  if [[ ! -f "$xlsx" ]]; then
    echo "[FAIL_FAST] MISSING_XLSX mode=$mode path=$xlsx" | tee -a "$FULL_LOG"
    return 2
  fi
  X="$xlsx" "$PYTHON_BIN" - <<'PY' | tee -a "$FULL_LOG"
import os
import sys
sys.path.insert(0, '/home/usergjf/Sa2VA/Sa2VA-main/sa2va_eval')
from vlmeval.dataset.videomme import VideoMME
path = os.environ['X']
out = VideoMME.evaluate(path, model='exact_matching')
print(f"[PROGRESS] VIDEOMME_RATING path={out} overall={out.get('overall', {}).get('overall', 'NA') if isinstance(out, dict) else 'NA'}")
PY
}

run_mode() {
  local mode="$1"
  local mode_root="$OUT_ROOT/$mode"
  local mode_log="$mode_root/eval.log"
  local denylist_path="$OUT_ROOT/denylist.txt"
  mkdir -p "$mode_root"

  if [[ -f "$mode_root/Sa2VA-InternVL3-2B-LOCAL-MEM/Sa2VA-InternVL3-2B-LOCAL-MEM_Video-MME_8frame_nopack_nosubs_rating.json" ]]; then
    echo "[PROGRESS] RESUME_SKIP mode=$mode reason=rating_exists" | tee -a "$FULL_LOG"
    return 0
  fi

  clear_mode_env
  case "$mode" in
    base)
      apply_mode_env base ""
      ;;
    core)
      apply_mode_env core ""
      ;;
    feedback)
      if [[ -f "$denylist_path" ]]; then
        apply_mode_env feedback "$denylist_path"
      else
        apply_mode_env feedback ""
      fi
      ;;
    *)
      echo "[FAIL_FAST] UNKNOWN_MODE mode=$mode" | tee -a "$FULL_LOG"
      return 2
      ;;
  esac
  apply_vqa_mem_env "$mode"
  echo "[PROGRESS] START mode=$mode data=Video-MME gpu=$GPU_ID mem_enable=${VQA_MEM_ENABLE} adapter=${PL_ADAPTER_CKPT:-off} gate=${PL_GATE_MODE:-off}" | tee -a "$FULL_LOG"

  "$PYTHON_BIN" -u "$ROOT_DIR/tools/run_sa2va_eval_local_2b_mem.py" \
    --data Video-MME \
    --mode all \
    --nframe 8 \
    --gpu "$GPU_ID" \
    --model-path "$MODEL_PATH" \
    --work-dir "$mode_root" \
    2>&1 | tee "$mode_log" | tee -a "$FULL_LOG"

  run_rating "$mode"
  echo "[PROGRESS] MODE_DONE mode=$mode" | tee -a "$FULL_LOG"
}

run_mode base
run_mode core
run_mode feedback
echo "[PROGRESS] RUN_DONE out_root=$OUT_ROOT" | tee -a "$FULL_LOG"
