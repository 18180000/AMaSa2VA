from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FusionConfig:
    alpha: float = 0.5
    cosine_c0: float = 0.80
    cosine_c1: float = 0.95
    route_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if self.cosine_c1 <= self.cosine_c0:
            raise ValueError("cosine_c1 must exceed cosine_c0")
        if self.route_threshold is not None and not 0 <= self.route_threshold <= 1:
            raise ValueError("route_threshold must be in [0, 1]")


def fuse_and_route(
    adapted_prompt: torch.Tensor,
    original_prompt: torch.Tensor,
    config: FusionConfig,
) -> Tuple[torch.Tensor, float, bool]:
    """Apply manuscript equations (6)-(8) and (11)."""
    adapted = adapted_prompt.reshape(-1, adapted_prompt.shape[-1])
    original = original_prompt.reshape(-1, original_prompt.shape[-1])
    adapted_mean = adapted.mean(dim=0)
    original_mean = original.mean(dim=0)
    similarity = F.cosine_similarity(adapted_mean[None], original_mean[None], dim=1)[0]
    gate = torch.clamp(
        (similarity - config.cosine_c0) / (config.cosine_c1 - config.cosine_c0), 0, 1
    )
    mixed = config.alpha * adapted_prompt + (1 - config.alpha) * original_prompt
    final = gate * mixed + (1 - gate) * original_prompt
    route_memory = config.route_threshold is None or float(gate) >= config.route_threshold
    return (final if route_memory else original_prompt), float(gate), route_memory
