#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || "$3" != "--" ]]; then
  echo "usage: $0 OUTPUT_DIR GPU_INDEX -- COMMAND [ARG ...]" >&2
  exit 2
fi

output_dir=$1
gpu_index=$2
shift 3
mkdir -p "$output_dir"

printf '%q ' "$@" >"$output_dir/command.txt"
printf '\n' >>"$output_dir/command.txt"
date -u +%FT%TZ >"$output_dir/started_utc.txt"

nvidia-smi \
  --id="$gpu_index" \
  --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop-ms=200 >"$output_dir/gpu.csv" 2>"$output_dir/gpu_sampler.err" &
sampler_pid=$!

cleanup() {
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
}
trap cleanup EXIT

set +e
/usr/bin/time -v -o "$output_dir/time.txt" "$@" \
  >"$output_dir/stdout.log" 2>"$output_dir/stderr.log"
status=$?
set -e

printf '%s\n' "$status" >"$output_dir/exit_status.txt"
date -u +%FT%TZ >"$output_dir/finished_utc.txt"
exit "$status"
