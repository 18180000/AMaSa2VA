"""Resolve the two Sa2VA backbone families without loading model weights.

The task-specific wrappers deliberately stay in their own packages.  This
module only decides which package should be used, so importing it is safe in
launch scripts and unit tests that do not have PyTorch/Transformers installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional


INTERNVL = "internvl"
QWEN = "qwen"
SUPPORTED_BACKBONES = (INTERNVL, QWEN)

_ALIASES = {
    "internvl": INTERNVL,
    "phi": INTERNVL,
    "phi2": INTERNVL,
    "other": INTERNVL,
    "qwen": QWEN,
    "qwen2": QWEN,
    "qwen2.5": QWEN,
    "qwen2_5_vl": QWEN,
    "qwen3": QWEN,
    "qwen3_vl": QWEN,
}


def normalize_backbone(value: str) -> str:
    """Normalize CLI aliases to ``internvl`` or ``qwen``."""
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == "auto":
        return normalized
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(("auto",) + SUPPORTED_BACKBONES)
        raise ValueError(f"unsupported backbone {value!r}; choose one of: {choices}") from exc


def _flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _flatten_strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _flatten_strings(item)


def _classify_text(
    values: Iterable[str], generic_sa2va_is_native: bool = False
) -> Optional[str]:
    text = " ".join(values).lower().replace("-", "_")
    qwen_markers = (
        "sa2vachatmodelqwen",
        "sa2vachatconfigqwen",
        "qwen2_5_vl",
        "qwen2.5_vl",
        "qwen2_vl",
        "qwen3_vl",
        "qwen_vl",
    )
    if any(marker in text for marker in qwen_markers):
        return QWEN
    # Within this repository every Sa2VA build that is not the processor-driven
    # Qwen flavour uses the native InternVL/Phi wrapper. Keep this after the
    # Qwen check so ordinary names such as Sa2VA-1B/4B resolve correctly.
    native_markers = ("internvl", "internlm", "phi_2", "phi2", "phi-2")
    if any(marker in text for marker in native_markers):
        return INTERNVL
    if generic_sa2va_is_native and "sa2va" in text:
        return INTERNVL
    return None


def infer_backbone_from_config(model_path: str) -> Optional[str]:
    """Inspect a local Hugging Face ``config.json`` when one is available."""
    config_path = Path(model_path).expanduser() / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    # A generic ``sa2va`` model_type alone does not distinguish the Qwen build;
    # let the checkpoint path provide the final fallback in that case.
    return _classify_text(_flatten_strings(config))


def infer_backbone_from_model(model: Any) -> Optional[str]:
    """Inspect an already-loaded model/config without importing Transformers."""
    values = [type(model).__name__]
    config = getattr(model, "config", None)
    if config is not None:
        values.append(type(config).__name__)
        for name in ("model_type", "architectures", "auto_map"):
            values.extend(_flatten_strings(getattr(config, name, None)))
    return _classify_text(values, generic_sa2va_is_native=True)


def resolve_backbone(
    model_path: str,
    requested: str = "auto",
    loaded_model: Any = None,
) -> str:
    """Resolve a backbone, preferring explicit choice and model metadata.

    The path-name check is intentionally last.  It preserves convenient use
    with ordinary checkpoint directories while local ``config.json`` metadata
    remains authoritative when present.
    """
    requested = normalize_backbone(requested)
    if requested != "auto":
        return requested

    detected = infer_backbone_from_model(loaded_model) if loaded_model is not None else None
    detected = detected or infer_backbone_from_config(model_path)
    detected = detected or _classify_text(
        [str(model_path)], generic_sa2va_is_native=True
    )
    if detected is None:
        raise ValueError(
            "could not infer the Sa2VA backbone; pass --backbone internvl or "
            "--backbone qwen explicitly"
        )
    return detected
