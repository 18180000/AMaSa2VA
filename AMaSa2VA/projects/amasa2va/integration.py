"""Thin backbone dispatcher around the existing, shared SAM2 memory patch.

The object memory and prompt fusion are deliberately not duplicated: both
Sa2VA model families eventually call the same SAM2 video predictor.  Only
model/processor validation belongs in the per-backbone adapters below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .models.memory import (
    ObjectCentricMemoryBank,
    PaperMemoryConfig,
    apply_paper_memory_patch,
    locate_sam2_module,
)
from .models.backbones import QWEN, resolve_backbone


@dataclass(frozen=True)
class SegmentationBackboneContext:
    name: str
    model: Any
    processor: Any
    sam2: Any


def prepare_backbone(
    model: Any,
    model_path: str = "",
    backbone: str = "auto",
    processor: Any = None,
) -> SegmentationBackboneContext:
    """Validate and expose the correct RefVOS backend without editing it."""
    resolved = resolve_backbone(model_path, backbone, loaded_model=model)
    if resolved == QWEN:
        from .models.qwen2_5_vl.segmentation import validate_model
    else:
        from .models.internvl.segmentation import validate_model

    validate_model(model, processor=processor)
    return SegmentationBackboneContext(
        name=resolved,
        model=model,
        processor=processor,
        sam2=locate_sam2_module(model),
    )


def apply_memory_for_backbone(
    model: Any,
    config: PaperMemoryConfig,
    model_path: str = "",
    backbone: str = "auto",
    processor: Any = None,
) -> Tuple[SegmentationBackboneContext, ObjectCentricMemoryBank]:
    """Apply the common SAM2 patch after backbone-specific validation."""
    context = prepare_backbone(
        model=model,
        model_path=model_path,
        backbone=backbone,
        processor=processor,
    )
    bank = apply_paper_memory_patch(context.sam2, config)
    return context, bank


def begin_expression(
    context: SegmentationBackboneContext,
    bank: ObjectCentricMemoryBank,
    video_id: str,
    expression_id: str,
) -> None:
    """Clear causal state and attach a stable ID before each expression."""
    bank.clear()
    context.sam2._amasa2va_sample_id = f"{video_id}::{expression_id}"
