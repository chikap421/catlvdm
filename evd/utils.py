"""Small tensor utilities shared by the EVD reference modules."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch


def validate_video_tensor(
    tensor: torch.Tensor,
    name: str = "tensor",
    channels: Optional[int] = None,
) -> None:
    """Validate a video tensor with shape ``[B, C, T, H, W]``."""

    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 5:
        raise ValueError(f"{name} must have shape [B, C, T, H, W], got {tuple(tensor.shape)}")
    if channels is not None and tensor.shape[1] != channels:
        raise ValueError(f"{name} must have {channels} channels, got {tensor.shape[1]}")


def group_count(channels: int, max_groups: int = 8) -> int:
    """Choose a GroupNorm group count that divides ``channels``."""

    if channels <= 0:
        raise ValueError("channels must be positive")
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def parse_token_to_video_shape(
    token_to_video: Union[Dict[str, Any], Tuple[int, int, int]],
) -> Tuple[int, int, int]:
    """Read ``(T, H, W)`` from a tuple or a small reshape-info dictionary."""

    if isinstance(token_to_video, dict):
        t = token_to_video.get("T", token_to_video.get("t"))
        h = token_to_video.get("H", token_to_video.get("h"))
        w = token_to_video.get("W", token_to_video.get("w"))
        if t is None or h is None or w is None:
            raise ValueError("token_to_video dict must contain T/H/W or t/h/w")
        return int(t), int(h), int(w)
    if len(token_to_video) != 3:
        raise ValueError("token_to_video tuple must be (T, H, W)")
    t, h, w = token_to_video
    return int(t), int(h), int(w)
