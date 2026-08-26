"""Minimal Core-mode integration for an already loaded Sa2VA model."""

from pathlib import Path
from typing import Any

from projects.amasa2va.integration import apply_memory_for_backbone
from projects.amasa2va.models.memory import PaperMemoryConfig


def enable_core_mode(
    model: Any,
    adapter_checkpoint: Path,
    audit_path: Path,
    model_path: str = "",
    backbone: str = "auto",
    processor: Any = None,
):
    context, bank = apply_memory_for_backbone(
        model,
        PaperMemoryConfig(
            capacity=8,
            retrieval_k=4,
            temperature=0.4,
            compression="similarity_merge",
            alpha=0.5,
            cosine_c0=0.8,
            cosine_c1=0.95,
            adapter_checkpoint=str(adapter_checkpoint),
            audit_path=str(audit_path),
        ),
        model_path=model_path,
        backbone=backbone,
        processor=processor,
    )
    return context.sam2, bank


def begin_expression(sam2: Any, bank: Any, video_id: str, expression_id: str) -> None:
    """Reset causal state before invoking the upstream prediction function."""
    bank.clear()
    sam2._amasa2va_sample_id = f"{video_id}::{expression_id}"
