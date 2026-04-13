#!/bin/bash
set -euo pipefail

ADAPTER_CKPT_DEFAULT="/data0/pl_adapter_e2e_eval_v1/sa2va_wt/artifacts/pl_adapter_ckpt.pt"

# Keep env key names aligned with /home/usergjf/sallm_final/run_full_mevis_* scripts.
MODE_KEYS_CORE=(
  PL_ADAPTER_CKPT
  PL_ADAPTER_ALPHA
  PL_GATE_MODE
  PL_GATE_C0
  PL_GATE_C1
  PL_GATE_T
  PL_GATE_MIN
  PL_GATE_MAX
  PL_GATE_STATS_EVERY
)
MODE_KEYS_FEEDBACK=(
  PL_ADAPTER_CKPT
  PL_ADAPTER_ALPHA
  PL_GATE_MODE
  PL_GATE_C0
  PL_GATE_C1
  PL_GATE_T
  PL_GATE_MIN
  PL_GATE_MAX
  PL_GATE_STATS_EVERY
  PL_DENYLIST_PATH
)

clear_mode_env() {
  unset PL_ADAPTER_CKPT || true
  unset PL_ADAPTER_ALPHA || true
  unset PL_GATE_MODE || true
  unset PL_GATE_C0 || true
  unset PL_GATE_C1 || true
  unset PL_GATE_T || true
  unset PL_GATE_MIN || true
  unset PL_GATE_MAX || true
  unset PL_GATE_STATS_EVERY || true
  unset PL_DENYLIST_PATH || true
}

apply_mode_env() {
  local mode="$1"
  local denylist_path="${2:-}"
  clear_mode_env
  case "$mode" in
    base)
      export PL_ADAPTER_CKPT=""
      export PL_ADAPTER_ALPHA="0.5"
      export PL_GATE_MODE="off"
      ;;
    core)
      export PL_ADAPTER_CKPT="${ADAPTER_CKPT_DEFAULT}"
      export PL_ADAPTER_ALPHA="0.5"
      export PL_GATE_MODE="cos"
      export PL_GATE_C0="0.80"
      export PL_GATE_C1="0.95"
      export PL_GATE_T="10.0"
      export PL_GATE_MIN="0.0"
      export PL_GATE_MAX="1.0"
      export PL_GATE_STATS_EVERY="200"
      ;;
    feedback)
      export PL_ADAPTER_CKPT="${ADAPTER_CKPT_DEFAULT}"
      export PL_ADAPTER_ALPHA="0.5"
      export PL_GATE_MODE="cos"
      export PL_GATE_C0="0.80"
      export PL_GATE_C1="0.95"
      export PL_GATE_T="10.0"
      export PL_GATE_MIN="0.0"
      export PL_GATE_MAX="1.0"
      export PL_GATE_STATS_EVERY="200"
      export PL_DENYLIST_PATH="$denylist_path"
      ;;
    *)
      echo "[FAIL_FAST] UNKNOWN_MODE mode=$mode" >&2
      return 2
      ;;
  esac
}

mode_signature() {
  local mode="$1"
  local -a keys
  local kvs=""
  if [[ "$mode" == "core" ]]; then
    keys=("${MODE_KEYS_CORE[@]}")
  elif [[ "$mode" == "feedback" ]]; then
    keys=("${MODE_KEYS_FEEDBACK[@]}")
  else
    echo "[FAIL_FAST] SIGNATURE_UNSUPPORTED_MODE mode=$mode" >&2
    return 2
  fi

  for k in "${keys[@]}"; do
    local v="${!k-}"
    if [[ "$k" == "PL_DENYLIST_PATH" && -n "$v" ]]; then
      v="<DATASET_DENYLIST_PATH>"
    fi
    kvs+="$k=$v"$'\n'
  done
  printf "%s" "$kvs" | LC_ALL=C sort | sha256sum | awk '{print $1}'
}

write_mode_signature_file() {
  local out_file="$1"
  mkdir -p "$(dirname "$out_file")"
  clear_mode_env
  apply_mode_env core ""
  local core_sig
  core_sig="$(mode_signature core)"

  clear_mode_env
  apply_mode_env feedback "/tmp/placeholder_denylist.txt"
  local feedback_sig
  feedback_sig="$(mode_signature feedback)"

  cat > "$out_file" <<EOT
core=${core_sig}
feedback=${feedback_sig}
EOT
  echo "$core_sig|$feedback_sig"
}
