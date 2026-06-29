"""Event head used by the public EVD video reference implementation."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .utils import group_count, validate_video_tensor


class EventHead3D(nn.Module):
    """Predict spatiotemporal event activity from a video update field.

    This is the public U-Net-compatible version of the paper's token-aligned
    event head. It accepts dense video tensors shaped ``[B, C, T, H, W]`` and
    returns one event logit/activity value per frame and spatial location.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        detach_input: bool = False,
        max_groups: int = 8,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

        hidden = int(hidden_channels or min(128, in_channels))
        if hidden <= 0:
            raise ValueError("hidden_channels must be positive")

        self.in_channels = int(in_channels)
        self.hidden_channels = hidden
        self.detach_input = detach_input

        self.net = nn.Sequential(
            nn.Conv3d(self.in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(group_count(hidden, max_groups), hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, kernel_size=1),
            nn.GroupNorm(group_count(hidden, max_groups), hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, 1, kernel_size=1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the final projection near zero for stable opt-in use."""

        final = self.net[-1]
        if isinstance(final, nn.Conv3d):
            nn.init.normal_(final.weight, mean=0.0, std=1.0e-5)
            nn.init.zeros_(final.bias)

    def forward(self, update_or_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(event_logits, event_activity)`` for ``[B, C, T, H, W]`` input."""

        validate_video_tensor(update_or_features, "update_or_features", channels=self.in_channels)
        x = update_or_features.detach() if self.detach_input else update_or_features
        event_logits = self.net(x)
        event_activity = torch.sigmoid(event_logits)
        return event_logits, event_activity

    def infer_activity(self, update_or_features: torch.Tensor) -> torch.Tensor:
        """Return only sigmoid event activity for inference-time gating."""

        _, event_activity = self.forward(update_or_features)
        return event_activity
