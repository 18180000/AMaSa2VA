#!/bin/bash
set -euo pipefail

GPU_ID="${1:?gpu id required}"
PYTHON_BIN="${2:-/opt/anaconda3/envs/torch/bin/python}"
OUT_ROOT="${3:-/data0/multi_rvos_benchmark_v1/runs/xmem_copy_davis17_3modes_4b_v1}"
FACTOR="${4:-2}"
MAX_SAMPLES="${5:-0}"
MODEL_PATH="${6:-/data0/pretrained/Sa2VA-4B-ready}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_RUNNER="$ROOT_DIR/tools/run_eval_with_final_patchfix.py"
SRC_META="/data0/data/sa2va-data/video_datas/davis17/meta_expressions/valid/meta_expressions.json"
SRC_MASK="/data0/data/sa2va-data/video_datas/davis17/valid/mask_dict.json"
SRC_IMAGES="/data0/data/sa2va-data/video_datas/davis17/valid/JPEGImages"
COPY_DIR="$OUT_ROOT/datasets/davis17_xmem_copy_${FACTOR}x"
DATASET_TAG="DAVIS17_XMEM_COPY_${FACTOR}X"
DENYLIST_PATH="/home/usergjf/sallm_final/artifacts/denylist_units.txt"

mkdir -p "$OUT_ROOT/logs" "$ROOT_DIR/artifacts"
FULL_LOG="$OUT_ROOT/logs/full.log"

source "$ROOT_DIR/configs/modes.sh"
SIG_PAIR="$(write_mode_signature_file "$ROOT_DIR/artifacts/mode_signature.txt")"
CORE_SIG="${SIG_PAIR%%|*}"
FEEDBACK_SIG="${SIG_PAIR##*|}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="/data0/hf_home"
export TRANSFORMERS_CACHE="/data0/hf_home/transformers"
export HF_DATASETS_CACHE="/data0/hf_home/datasets"
export TORCH_HOME="/data0/torch_home"
export XDG_CACHE_HOME="/data0/xdg_cache"
export TMPDIR="/data0/tmp"
export SA2VA_AUDIT_DETERMINISTIC=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SEED=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/usergjf/Sa2VA/Sa2VA-main:/home/usergjf/sallm_final:${PYTHONPATH:-}"

run_mode() {
  local mode="$1"
  local denylist_path="${2:-}"
  local run_dir="$OUT_ROOT/$mode"
  mkdir -p "$run_dir"

  local res="$run_dir/$DATASET_TAG/results.json"
  local metrics="$run_dir/$DATASET_TAG/metrics.json"
  if [[ -f "$metrics" ]]; then
    echo "[PROGRESS] RESUME_SKIP mode=$mode reason=metrics_exists path=$metrics" | tee -a "$FULL_LOG"
    return 0
  fi

  apply_mode_env "$mode" "$denylist_path"
  echo "[PROGRESS] START mode=$mode dataset=$DATASET_TAG gpu=$GPU_ID phase=infer" | tee -a "$FULL_LOG"
  echo "[PROGRESS] MODE_SIGNATURE core=$CORE_SIG feedback=$FEEDBACK_SIG" | tee -a "$FULL_LOG"

  "$PYTHON_BIN" -u "$ROOT_DIR/tools/apply_patches_like_final.py" | tee -a "$FULL_LOG"
  "$PYTHON_BIN" -u "$ROOT_DIR/tools/print_vmbank_patch_state.py" | tee -a "$FULL_LOG"

  "$PYTHON_BIN" -u "$EVAL_RUNNER" \
    "$MODEL_PATH" \
    --dataset "$DATASET_TAG" \
    --meta_path "$COPY_DIR/meta_expressions.json" \
    --mask_path "$COPY_DIR/mask_dict.json" \
    --data_root /data0/data/sa2va-data \
    --work_dir "$run_dir" \
    --max_samples "$MAX_SAMPLES" \
    --use_vmbank 1 \
    --vmbank_alpha 0.5 \
    --cap_M 16 \
    --obj_tokens 4 \
    --gate_enable 0 \
    --compress fifo \
    --mem_select topk \
    --mem_topk 4 \
    --mem_sim cosine 2>&1 | tee -a "$FULL_LOG"

  [[ -f "$res" ]] || { echo "[FAIL_FAST] RESULTS_MISSING mode=$mode path=$res" | tee -a "$FULL_LOG"; return 1; }

  echo "[PROGRESS] START mode=$mode dataset=$DATASET_TAG phase=metrics" | tee -a "$FULL_LOG"
  "$PYTHON_BIN" /home/usergjf/Sa2VA/Sa2VA-main/tools/eval/eval_davis.py \
    "$res" \
    --mevis_exp_path "$COPY_DIR/meta_expressions.json" \
    --mevis_mask_path "$COPY_DIR/mask_dict.json" \
    --save_name metrics.json 2>&1 | tee -a "$FULL_LOG"

  [[ -f "$metrics" ]] || { echo "[FAIL_FAST] METRICS_MISSING mode=$mode path=$metrics" | tee -a "$FULL_LOG"; return 1; }
  echo "[PROGRESS] MODE_DONE mode=$mode metrics=$metrics" | tee -a "$FULL_LOG"
}

{
  echo "[PROGRESS] RUN_START out_root=$OUT_ROOT factor=$FACTOR max_samples=$MAX_SAMPLES gpu=$GPU_ID model=$MODEL_PATH"
  "$PYTHON_BIN" -u "$ROOT_DIR/tools/build_refvos_xmem_copyset.py" \
    --src-meta "$SRC_META" \
    --src-mask "$SRC_MASK" \
    --src-images "$SRC_IMAGES" \
    --out-dir "$COPY_DIR" \
    --factor "$FACTOR" \
    --image-link-mode hardlink

  run_mode base
  run_mode core
  run_mode feedback "$DENYLIST_PATH"
  echo "[PROGRESS] RUN_DONE out_root=$OUT_ROOT"
} 2>&1 | tee -a "$FULL_LOG"
