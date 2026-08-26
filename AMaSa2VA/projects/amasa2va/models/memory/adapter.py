from __future__ import annotations

import torch
from torch import nn


class PaperPromptAdapter(nn.Module):
    """Trainable prompt-space adapter for aggregated evidence and current visual context."""

    def __init__(self, dim: int = 256, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, evidence: torch.Tensor, current_visual: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([evidence, current_visual], dim=-1))


def load_paper_adapter(checkpoint_path: str, device: torch.device) -> PaperPromptAdapter:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    expected_first = (1024, 512)
    actual_first = tuple(state_dict.get("net.0.weight", torch.empty(0)).shape)
    if actual_first != expected_first:
        raise RuntimeError(
            "adapter checkpoint is incompatible with the paper-aligned adapter: "
            f"expected net.0.weight={expected_first}, got {actual_first}. "
            "The recovered legacy checkpoint consumes 16 flattened memory slots and must be retrained."
        )
    model = PaperPromptAdapter().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
