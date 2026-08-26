"""Validation for Sa2VA-1B/2B/4B's native RefVOS path."""

from typing import Any

from ..backbones import QWEN, infer_backbone_from_model


def validate_model(model: Any, processor: Any = None) -> None:
    """Reject accidentally routing a Qwen-specific model through this path."""
    if infer_backbone_from_model(model) == QWEN:
        raise TypeError("a Qwen Sa2VA model was routed to the InternVL/Phi adapter")
    if not callable(getattr(model, "predict_forward", None)):
        raise TypeError("the model does not expose Sa2VA predict_forward")
