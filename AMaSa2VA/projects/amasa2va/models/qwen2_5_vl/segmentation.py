"""Validation for the processor-driven Qwen2.5-VL RefVOS path.

The official Qwen model owns its multimodal input construction.  The memory
integration therefore leaves ``predict_forward`` untouched and patches only
its shared SAM2 predictor after this contract has been checked.
"""

from typing import Any

from ..backbones import QWEN, infer_backbone_from_model


def validate_model(model: Any, processor: Any = None) -> None:
    detected = infer_backbone_from_model(model)
    if detected is not None and detected != QWEN:
        raise TypeError("a non-Qwen Sa2VA model was routed to the Qwen adapter")
    if processor is None:
        raise ValueError("the Qwen RefVOS path requires its AutoProcessor instance")
    if not callable(getattr(model, "predict_forward", None)):
        raise TypeError("the model does not expose Sa2VA predict_forward")
