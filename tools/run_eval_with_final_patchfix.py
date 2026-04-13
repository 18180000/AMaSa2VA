#!/usr/bin/env python3
"""Run sallm_final eval entry with an in-memory marker fix.

No file under /home/usergjf/sallm_final is modified on disk.
"""

import os
import sys


TARGET = "/home/usergjf/sallm_final/scripts/sa2va_eval_ref_vos_patched.py"
NEEDLE = "        sam2_mod._run_single_frame_inference = MethodType(_run_single_frame_inference_profiling, sam2_mod)\n"
PATCH = (
    "        _run_single_frame_inference_profiling._vmbank_patched = True\n"
    "        sam2_mod._run_single_frame_inference = MethodType(_run_single_frame_inference_profiling, sam2_mod)\n"
)


def main() -> int:
    for p in ("/home/usergjf/Sa2VA/Sa2VA-main", "/home/usergjf/sallm_final"):
        if p not in sys.path:
            sys.path.insert(0, p)

    with open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    if PATCH not in src:
        if NEEDLE not in src:
            raise SystemExit("[FAIL_FAST] PATCH_NEEDLE_NOT_FOUND")
        src = src.replace(NEEDLE, PATCH, 1)

    code = compile(src, TARGET, "exec")
    glb = {
        "__name__": "__main__",
        "__file__": TARGET,
        "__package__": None,
        "__cached__": None,
    }
    sys.argv[0] = TARGET
    exec(code, glb, glb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
