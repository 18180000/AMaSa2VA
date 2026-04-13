#!/usr/bin/env python3
import json
import os
import sys


def is_patched_callable(bound_callable) -> bool:
    fn = getattr(bound_callable, "__func__", bound_callable)
    return bool(getattr(fn, "_vmbank_patched", False))


class DummySam2:
    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        return None

    def _run_single_frame_inference(self, *args, **kwargs):
        return None, None


def main() -> int:
    patch_root = "/data0/rebuild_vmbank_trigate_v1"
    if patch_root not in sys.path:
        sys.path.insert(0, patch_root)

    from scripts.patch_sam2_integration import apply_sam2_patches  # type: ignore
    from types import MethodType

    sam2 = DummySam2()
    apply_sam2_patches(
        sam2,
        use_vmbank=True,
        vmbank_alpha=0.5,
        cap_M=16,
        obj_tokens=4,
        gate_enable=False,
        compress="fifo",
        mem_select="topk",
        mem_topk=4,
        mem_sim="cosine",
        gate_safe=False,
        tau_read=0.4,
        inject_gate="none",
        sim_tau0=0.2,
        sim_tau1=0.6,
        alpha_base=None,
    )

    # Mirror the final script's extra PL profiling rebind path.
    orig = sam2._run_single_frame_inference

    def _run_single_frame_inference_profiling(self, *args, **kwargs):
        return orig(*args, **kwargs)

    _run_single_frame_inference_profiling._vmbank_patched = True
    sam2._run_single_frame_inference = MethodType(_run_single_frame_inference_profiling, sam2)

    state = {
        "_get_image_feature_patched": is_patched_callable(sam2._get_image_feature),
        "_run_single_frame_inference_patched": is_patched_callable(sam2._run_single_frame_inference),
    }
    print("[PROGRESS] APPLY_PATCHES_LIKE_FINAL", json.dumps(state, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
