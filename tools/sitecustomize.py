#!/usr/bin/env python3
"""Process-wide startup hooks for stage12 restart tools.

Enabled only when STAGE12_METHODTYPE_PATCH=1.
This keeps VMBank patch markers on MethodType-bound wrappers created in
sa2va_eval_ref_vos_patched.py under core/feedback modes.
"""

import os
import types


if os.environ.get("STAGE12_METHODTYPE_PATCH", "0") == "1":
    _orig_method_type = types.MethodType

    def _patched_method_type(func, obj, /):
        name = getattr(func, "__name__", "")
        if (
            "_run_single_frame_inference" in name
            or name == "_get_image_feature_hook"
            or name == "_run_single_frame_inference_profiling"
        ):
            try:
                setattr(func, "_vmbank_patched", True)
            except Exception:
                pass
        return _orig_method_type(func, obj)

    types.MethodType = _patched_method_type
