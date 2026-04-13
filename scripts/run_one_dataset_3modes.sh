#!/bin/bash
set -euo pipefail

DATASET="${1:?dataset required}"
GPU_ID="${2:?gpu id required}"
PYTHON_BIN="${3:?python required}"
OUT_ROOT="${4:?out_root required}"
MODEL_PATH="${5:-/data0/pretrained/Sa2VA-1B}"

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
  revos_valid)
    META_PATH="/home/usergjf/multi_rvos_benchmark_v1/data_links/revos_valid/meta_expressions.json"
    MASK_PATH="/data0/data/sa2va-data/video_datas/revos/mask_dict.json"
    ;;
  mevis_valid_u)
    META_PATH="/data0/data/sa2va-data/video_datas/mevis/valid_u/meta_expressions.json"
    MASK_PATH="/data0/data/sa2va-data/video_datas/mevis/valid_u/mask_dict.json"
    ;;
  rvos_valid)
    echo "[PROGRESS] SKIP dataset=rvos_valid reason=disabled_by_user"
    exit 0
    ;;
  *)
    echo "[FAIL_FAST] UNKNOWN_DATASET dataset=$DATASET"
    exit 2
    ;;
esac

mkdir -p "$OUT_ROOT/$DATASET" "$OUT_ROOT/logs" "$ROOT_DIR/artifacts"

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

SIG_PAIR="$(write_mode_signature_file "$ROOT_DIR/artifacts/mode_signature.txt")"
CORE_SIG="${SIG_PAIR%%|*}"
FEEDBACK_SIG="${SIG_PAIR##*|}"

echo "[PROGRESS] MODE_SIGNATURE core=${CORE_SIG} feedback=${FEEDBACK_SIG}"

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

validate_resume() {
  local mode_dir="$1"
  local metrics="$mode_dir/$DATASET/metrics.json"
  local unit="$mode_dir/$DATASET/unit_jf.json"
  [[ -f "$metrics" && -f "$unit" ]] || return 1
  "$PYTHON_BIN" - <<PY >/dev/null
import json
json.load(open("$metrics"))
u=json.load(open("$unit"))
if not isinstance(u, dict):
    raise SystemExit(1)
valid=sum(1 for k,v in u.items() if "|" in str(k) and isinstance(v, dict) and "mean_JF" in v)
if valid <= 0:
    raise SystemExit(1)
PY
}

write_subset_manifest() {
  local results_json="$1"
  local out_txt="$2"
  "$PYTHON_BIN" - <<PY
import json
res=json.load(open("$results_json"))
keys=sorted(map(str, res.keys()))
with open("$out_txt", "w") as f:
    for k in keys:
        f.write(k + "\n")
print(f"[PROGRESS] SUBSET_MANIFEST_WRITTEN n={len(keys)} out=$out_txt")
PY
}

write_unit_from_results() {
  local results_json="$1"
  local out_unit="$2"
  local mode_dir
  mode_dir="$(dirname "$(dirname "$results_json")")"
  local csv_path="$mode_dir/$DATASET/metrics.csv"
  if [[ "$DATASET" == "revos_valid" && -f "$csv_path" ]]; then
    "$PYTHON_BIN" /home/usergjf/stage12_restart_v1/tools/build_unit_from_revos_csv.py \
      --csv "$csv_path" \
      --out "$out_unit"
    return 0
  fi
  "$PYTHON_BIN" /home/usergjf/stage12_restart_v1/tools/dump_unit_metrics_refvos.py \
    --results "$results_json" \
    --exp_path "$META_PATH" \
    --mask_path "$MASK_PATH" \
    --out "$out_unit"
}

ensure_mode_unit() {
  local mode="$1"
  local mode_dir="$OUT_ROOT/$DATASET/$mode"
  local results_json="$mode_dir/$DATASET/results.json"
  local unit_json="$mode_dir/$DATASET/unit_jf.json"

  [[ -f "$results_json" ]] || { echo "[FAIL_FAST] RESULTS_JSON_MISSING_FOR_UNIT mode=$mode path=$results_json"; return 1; }
  if [[ -f "$unit_json" ]]; then
    if "$PYTHON_BIN" - <<PY >/dev/null
import json
u=json.load(open("$unit_json"))
ok=sum(1 for k,v in u.items() if "|" in str(k) and isinstance(v, dict) and "mean_JF" in v) > 0
raise SystemExit(0 if ok else 1)
PY
    then
      echo "[PROGRESS] RESUME_SKIP dataset=$DATASET mode=${mode}_unit path=$unit_json"
      return 0
    fi
  fi

  if [[ "$DATASET" == "mevis_valid_u" ]]; then
    "$PYTHON_BIN" /home/usergjf/sallm_final/tools/dump_unit_metrics.py \
      --results "$results_json" \
      --exp_path "$META_PATH" \
      --mask_path "$MASK_PATH" \
      --out "$unit_json"
  else
    write_unit_from_results "$results_json" "$unit_json"
  fi

  "$PYTHON_BIN" - <<PY >/dev/null
import json
u=json.load(open("$unit_json"))
ok=sum(1 for k,v in u.items() if "|" in str(k) and isinstance(v, dict) and "mean_JF" in v) > 0
if not ok:
    raise SystemExit("[FAIL_FAST] UNIT_JF_INVALID path=$unit_json")
PY
}

build_denylist() {
  local base_unit="$OUT_ROOT/$DATASET/base/$DATASET/unit_jf.json"
  local core_unit="$OUT_ROOT/$DATASET/core/$DATASET/unit_jf.json"
  local deny_dir="$OUT_ROOT/$DATASET/denylist"
  local deny_file="$deny_dir/${DATASET}_valid.txt"
  mkdir -p "$deny_dir"

  if [[ -s "$deny_file" ]]; then
    echo "[PROGRESS] RESUME_SKIP dataset=$DATASET mode=denylist source=reuse path=$deny_file"
    echo "$deny_file"
    return 0
  fi

  if [[ ! -f "$base_unit" ]]; then
    ensure_mode_unit base
  fi
  ensure_mode_unit core
  [[ -f "$base_unit" && -f "$core_unit" ]] || { echo "[FAIL_FAST] DENYLIST_SOURCE_UNIT_MISSING dataset=$DATASET"; return 1; }

  "$PYTHON_BIN" /home/usergjf/sallm_final/tools/build_denylist_units.py \
    --base "$base_unit" \
    --treat "$core_unit" \
    --threshold -0.003 \
    --out "$deny_file"

  local n_deny=0
  n_deny="$(wc -l < "$deny_file" | tr -d '[:space:]')"
  if [[ "${n_deny:-0}" -le 0 ]]; then
    echo "[FAIL_FAST] DENYLIST_EMPTY dataset=$DATASET threshold=-0.003 path=$deny_file"
    return 1
  fi
  echo "[PROGRESS] DENYLIST_READY dataset=$DATASET source=base_vs_core n_deny=$n_deny path=$deny_file"
  echo "$deny_file"
}

run_mode() {
  local mode="$1"
  local denylist_path="${2:-}"
  local mode_dir="$OUT_ROOT/$DATASET/$mode"
  mkdir -p "$mode_dir"

  if validate_resume "$mode_dir"; then
    echo "[PROGRESS] RESUME_SKIP dataset=$DATASET mode=$mode"
    return 0
  fi

  apply_mode_env "$mode" "$denylist_path"

  local log_file="$mode_dir/eval.log"
  echo "[PROGRESS] START dataset=$DATASET mode=$mode gpu=$GPU_ID" | tee "$log_file"
  echo "[PROGRESS] MODE_SIGNATURE core=${CORE_SIG} feedback=${FEEDBACK_SIG}" | tee -a "$log_file"

  "$ROOT_DIR/tools/assert_same_as_final.sh" "$EVAL_PY" | tee -a "$log_file"

  "$PYTHON_BIN" -u "$ROOT_DIR/tools/assert_no_unknown_args.py" \
    --python "$PYTHON_BIN" \
    --entry "$EVAL_PY" \
    --allowed-flags-file "$HELP_FLAGS_CACHE" \
    -- "${eval_args[@]}" --work_dir "$mode_dir" | tee -a "$log_file"

  "$ROOT_DIR/tools/preflight_comparable.sh" \
    --mode "$mode" \
    --run_dir "$mode_dir" \
    --manifest "$META_PATH" | tee -a "$log_file"

  if [[ "$mode" == "core" || "$mode" == "feedback" ]]; then
    "$PYTHON_BIN" -u "$ROOT_DIR/tools/apply_patches_like_final.py" | tee -a "$log_file"
    PATCH_STATE="$("$PYTHON_BIN" -u "$ROOT_DIR/tools/print_vmbank_patch_state.py")"
    echo "$PATCH_STATE" | tee -a "$log_file"
    echo "$PATCH_STATE" | grep -q '"_run_single_frame_inference_patched": true' || {
      echo "[FAIL_FAST] VMBANK_PATCH_MISSING_AFTER_APPLY dataset=$DATASET mode=$mode" | tee -a "$log_file"
      return 1
    }
  fi

  "$PYTHON_BIN" -u "$EVAL_RUNNER" "${eval_args[@]}" --work_dir "$mode_dir" 2>&1 | tee -a "$log_file"

  local results_json="$mode_dir/$DATASET/results.json"
  [[ -f "$results_json" ]] || { echo "[FAIL_FAST] RESULTS_JSON_MISSING path=$results_json" | tee -a "$log_file"; return 1; }

  if [[ "$DATASET" == "davis17_valid" ]]; then
    "$PYTHON_BIN" /home/usergjf/Sa2VA/Sa2VA-main/tools/eval/eval_davis.py \
      "$results_json" \
      --mevis_exp_path "$META_PATH" \
      --mevis_mask_path "$MASK_PATH" \
      --save_name metrics.json | tee -a "$log_file"
  elif [[ "$DATASET" == "revos_valid" ]]; then
    "$PYTHON_BIN" /home/usergjf/Sa2VA/Sa2VA-main/tools/eval/eval_revos.py \
      "$results_json" \
      --exp_path "$META_PATH" \
      --mask_path "$MASK_PATH" \
      --foreground_mask_path "/data0/data/sa2va-data/video_datas/revos/mask_dict_foreground.json" \
      --save_json_name metrics.json \
      --save_csv_name metrics.csv | tee -a "$log_file"
  elif [[ "$DATASET" == "mevis_valid_u" ]]; then
    "$PYTHON_BIN" -u "$ROOT_DIR/tools/eval_mevis_subset_progress.py" \
      "$results_json" \
      --mevis_exp_path "$META_PATH" \
      --mevis_mask_path "$MASK_PATH" \
      --save_name metrics.json \
      --log_every 20 2>&1 | tee -a "$log_file"
  else
    echo "[FAIL_FAST] METRICS_EVAL_UNSUPPORTED dataset=$DATASET" | tee -a "$log_file"
    return 1
  fi

  ensure_mode_unit "$mode" | tee -a "$log_file"
  write_subset_manifest "$results_json" "$mode_dir/subset_manifest.txt" | tee -a "$log_file"

  [[ -f "$mode_dir/$DATASET/metrics.json" ]] || { echo "[FAIL_FAST] METRICS_JSON_MISSING mode=$mode" | tee -a "$log_file"; return 1; }
  [[ -f "$mode_dir/$DATASET/unit_jf.json" ]] || { echo "[FAIL_FAST] UNIT_JF_JSON_MISSING mode=$mode" | tee -a "$log_file"; return 1; }

  echo "[PROGRESS] MODE_DONE dataset=$DATASET mode=$mode" | tee -a "$log_file"
}

run_mode base
run_mode core
DENY_FILE="$(build_denylist | tail -n 1)"
run_mode feedback "$DENY_FILE"

echo "[PROGRESS] DATASET_DONE dataset=$DATASET"
