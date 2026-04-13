#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/usergjf/.conda/envs/sa2va-vqa-clean/bin/python}"
GPU_ID="${1:-7}"
OUT_ROOT="${2:-/data0/multi_rvos_benchmark_v1/runs/videomme_1b_vqa_mem_core_v1}"
MODEL_PATH="${3:-/data0/pretrained/Sa2VA-1B}"

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
  clear_vqa_mem_env
  export VQA_MEM_ENABLE="1"
  export VQA_MEM_TOPK="${VQA_MEM_TOPK:-4}"
  export VQA_MEM_TAU="${VQA_MEM_TAU:-0.4}"
  export VQA_MEM_ALPHA="${VQA_MEM_ALPHA:-0.5}"
}

run_rating() {
  local xlsx="$OUT_ROOT/core/Sa2VA-1B-LOCAL-MEM/Sa2VA-1B-LOCAL-MEM_Video-MME_8frame_nopack_nosubs.xlsx"
  if [[ ! -f "$xlsx" ]]; then
    echo "[FAIL_FAST] MISSING_XLSX path=$xlsx" | tee -a "$FULL_LOG"
    return 2
  fi
  X="$xlsx" python3 - <<'PY' | tee -a "$FULL_LOG"
import json
import os
import pandas as pd

path = os.environ["X"]
df = pd.read_excel(path)
pred = df["prediction"].fillna("").astype(str).str.strip()
ans = df["answer"].fillna("").astype(str).str.strip()
out = {"overall": round((pred == ans).mean(), 3)}
for duration in ["short", "medium", "long"]:
    sub = df[df["duration"].astype(str).str.lower() == duration]
    p = sub["prediction"].fillna("").astype(str).str.strip()
    a = sub["answer"].fillna("").astype(str).str.strip()
    out[duration] = round((p == a).mean(), 3)
print(f"[PROGRESS] VIDEOMME_RATING_EXACT {json.dumps(out, ensure_ascii=False)}")
PY
}

mkdir -p "$OUT_ROOT/core"
clear_mode_env
apply_mode_env core ""
apply_vqa_mem_env

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home/usergjf/stage12_restart_v1/vendor_stubs"
for cupti_dir in \
  /usr/local/cuda-12.8/extras/CUPTI/lib64 \
  /opt/anaconda3/envs/torch/lib/python3.10/site-packages/nvidia/cuda_cupti/lib
do
  if [[ -d "$cupti_dir" ]]; then
    export LD_LIBRARY_PATH="$cupti_dir:${LD_LIBRARY_PATH:-}"
  fi
done

echo "[PROGRESS] START mode=core data=Video-MME gpu=$GPU_ID mem_enable=${VQA_MEM_ENABLE} model=$MODEL_PATH adapter=${PL_ADAPTER_CKPT:-off} gate=${PL_GATE_MODE:-off}" | tee -a "$FULL_LOG"

"$PYTHON_BIN" -u "$ROOT_DIR/tools/run_sa2va_eval_local_mem.py" \
  --data Video-MME \
  --mode all \
  --nframe 8 \
  --gpu "$GPU_ID" \
  --model-path "$MODEL_PATH" \
  --model-name "Sa2VA-1B-LOCAL-MEM" \
  --work-dir "$OUT_ROOT/core" \
  2>&1 | tee -a "$FULL_LOG"

rc=${PIPESTATUS[0]}
if [[ $rc -ne 0 ]]; then
  echo "[FAIL_FAST] MODE_FAILED mode=core rc=$rc" | tee -a "$FULL_LOG"
  exit $rc
fi

run_rating
echo "[PROGRESS] MODE_DONE mode=core" | tee -a "$FULL_LOG"
