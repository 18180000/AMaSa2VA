#!/bin/bash
set -euo pipefail

DATASET="${1:?dataset required}"
GPU_ID="${2:?gpu id required}"
PYTHON_BIN="${3:?python required}"
OUT_ROOT="${4:?out_root required}"
MODEL_PATH="${5:?model path required}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="/data0/data/sa2va-data"
EVAL_PY="/home/usergjf/sallm_final/scripts/sa2va_eval_ref_vos_patched.py"
EVAL_RUNNER="$ROOT_DIR/tools/run_eval_with_final_patchfix.py"
HELP_FLAGS_CACHE="$ROOT_DIR/artifacts/help_flags.txt"

source "$ROOT_DIR/configs/modes.sh"

case "$DATASET" in
  davis17_valid)
    META_PATH="/home/usergjf/multi_rvos_benchmark_v1/data_links/davis17_valid/meta_expressions.json"
    MASK_PATH="/data0/data/sa2va-data/video_datas/davis17/valid/mask_dict.json"
    ;;
  mevis_valid_u)
    META_PATH="/data0/data/sa2va-data/video_datas/mevis/valid_u/meta_expressions.json"
    MASK_PATH="/data0/data/sa2va-data/video_datas/mevis/valid_u/mask_dict.json"
    ;;
  *)
    echo "[FAIL_FAST] BASE_ONLY_UNSUPPORTED dataset=$DATASET"
    exit 2
    ;;
esac

MODE="base"
MODE_DIR="$OUT_ROOT/$DATASET/$MODE"
LOG_FILE="$MODE_DIR/eval.log"
mkdir -p "$MODE_DIR" "$OUT_ROOT/logs" "$ROOT_DIR/artifacts"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export SEED=0
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SA2VA_AUDIT_DETERMINISTIC=1
export PYTHONUNBUFFERED=1
export HF_HOME="/data0/hf_home"
export TRANSFORMERS_CACHE="/data0/hf_home/transformers"
export HF_DATASETS_CACHE="/data0/hf_home/datasets"
export TORCH_HOME="/data0/torch_home"
export XDG_CACHE_HOME="/data0/xdg_cache"
export TMPDIR="/data0/tmp"
export PYTHONPATH="/home/usergjf/Sa2VA/Sa2VA-main:/home/usergjf/sallm_final:${PYTHONPATH:-}"

apply_mode_env base ""

eval_args=(
  "$MODEL_PATH"
  --dataset "$DATASET"
  --data_root "$DATA_ROOT"
  --use_vmbank 1
  --vmbank_alpha 0.5
  --cap_M 16
  --obj_tokens 4
  --gate_enable 0
  --compress fifo
  --mem_select topk
  --mem_topk 4
  --mem_sim cosine
  --meta_path "$META_PATH"
  --mask_path "$MASK_PATH"
)

echo "[PROGRESS] START dataset=$DATASET mode=base gpu=$GPU_ID model=$MODEL_PATH" | tee "$LOG_FILE"

"$ROOT_DIR/tools/assert_same_as_final.sh" "$EVAL_PY" | tee -a "$LOG_FILE"

"$PYTHON_BIN" -u "$ROOT_DIR/tools/assert_no_unknown_args.py" \
  --python "$PYTHON_BIN" \
  --entry "$EVAL_PY" \
  --allowed-flags-file "$HELP_FLAGS_CACHE" \
  -- "${eval_args[@]}" --work_dir "$MODE_DIR" | tee -a "$LOG_FILE"

"$ROOT_DIR/tools/preflight_comparable.sh" \
  --mode "$MODE" \
  --run_dir "$MODE_DIR" \
  --manifest "$META_PATH" | tee -a "$LOG_FILE"

"$PYTHON_BIN" -u "$EVAL_RUNNER" "${eval_args[@]}" --work_dir "$MODE_DIR" 2>&1 | tee -a "$LOG_FILE"

RESULTS_JSON="$MODE_DIR/$DATASET/results.json"
[[ -f "$RESULTS_JSON" ]] || { echo "[FAIL_FAST] RESULTS_JSON_MISSING path=$RESULTS_JSON" | tee -a "$LOG_FILE"; exit 1; }

if [[ "$DATASET" == "davis17_valid" ]]; then
  "$PYTHON_BIN" /home/usergjf/Sa2VA/Sa2VA-main/tools/eval/eval_davis.py \
    "$RESULTS_JSON" \
    --mevis_exp_path "$META_PATH" \
    --mevis_mask_path "$MASK_PATH" \
    --save_name metrics.json | tee -a "$LOG_FILE"
  "$PYTHON_BIN" /home/usergjf/stage12_restart_v1/tools/dump_unit_metrics_refvos.py \
    --results "$RESULTS_JSON" \
    --exp_path "$META_PATH" \
    --mask_path "$MASK_PATH" \
    --out "$MODE_DIR/$DATASET/unit_jf.json" | tee -a "$LOG_FILE"
elif [[ "$DATASET" == "mevis_valid_u" ]]; then
  "$PYTHON_BIN" -u "$ROOT_DIR/tools/eval_mevis_subset_progress.py" \
    "$RESULTS_JSON" \
    --mevis_exp_path "$META_PATH" \
    --mevis_mask_path "$MASK_PATH" \
    --save_name metrics.json \
    --log_every 20 2>&1 | tee -a "$LOG_FILE"
  "$PYTHON_BIN" /home/usergjf/sallm_final/tools/dump_unit_metrics.py \
    --results "$RESULTS_JSON" \
    --exp_path "$META_PATH" \
    --mask_path "$MASK_PATH" \
    --out "$MODE_DIR/$DATASET/unit_jf.json" | tee -a "$LOG_FILE"
fi

"$PYTHON_BIN" - <<PY | tee -a "$LOG_FILE"
import json
res=json.load(open("$RESULTS_JSON"))
keys=sorted(map(str, res.keys()))
with open("$MODE_DIR/subset_manifest.txt", "w") as f:
    for k in keys:
        f.write(k + "\\n")
print(f"[PROGRESS] SUBSET_MANIFEST_WRITTEN n={len(keys)} out=$MODE_DIR/subset_manifest.txt")
PY

echo "[PROGRESS] MODE_DONE dataset=$DATASET mode=base" | tee -a "$LOG_FILE"
