#!/bin/bash
set -euo pipefail

ENTRY="${1:-/home/usergjf/sallm_final/scripts/sa2va_eval_ref_vos_patched.py}"
FINAL="/home/usergjf/sallm_final/scripts/sa2va_eval_ref_vos_patched.py"

if [[ ! -f "$ENTRY" ]]; then
  echo "[FAIL_FAST] EVAL_ENTRY_NOT_FOUND path=$ENTRY"
  exit 2
fi
if [[ ! -f "$FINAL" ]]; then
  echo "[FAIL_FAST] FINAL_ENTRY_NOT_FOUND path=$FINAL"
  exit 2
fi

ENTRY_REAL="$(realpath "$ENTRY")"
FINAL_REAL="$(realpath "$FINAL")"
ENTRY_SHA="$(sha256sum "$ENTRY_REAL" | awk '{print $1}')"
FINAL_SHA="$(sha256sum "$FINAL_REAL" | awk '{print $1}')"

if [[ "$ENTRY_SHA" != "$FINAL_SHA" ]]; then
  echo "[FAIL_FAST] EVAL_ENTRYPOINT_DIVERGED expected_sha=$FINAL_SHA got_sha=$ENTRY_SHA"
  exit 1
fi

echo "[PROGRESS] ASSERT_SAME_AS_FINAL_PASS entry=$ENTRY_REAL sha=$ENTRY_SHA"
