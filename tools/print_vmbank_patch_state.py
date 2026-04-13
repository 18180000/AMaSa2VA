#!/usr/bin/env python3
import json


def is_patched_callable(bound_callable) -> bool:
    fn = getattr(bound_callable, "__func__", bound_callable)
    return bool(getattr(fn, "_vmbank_patched", False))


class DummySam2:
    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        return None

    def _run_single_frame_inference(self, *args, **kwargs):
        return None, None


def main() -> int:
    from types import MethodType

    sam2 = DummySam2()

    def _get_image_feature_hook(self, inference_state, frame_idx, batch_size):
        return None

    def _run_single_frame_inference_profiling(self, *args, **kwargs):
        return None, None

    _get_image_feature_hook._vmbank_patched = True
    _run_single_frame_inference_profiling._vmbank_patched = True

    sam2._get_image_feature = MethodType(_get_image_feature_hook, sam2)
    sam2._run_single_frame_inference = MethodType(_run_single_frame_inference_profiling, sam2)

    state = {
        "_get_image_feature_patched": is_patched_callable(sam2._get_image_feature),
        "_run_single_frame_inference_patched": is_patched_callable(sam2._run_single_frame_inference),
    }
    print("VMBANK_PATCH_STATE", json.dumps(state, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
