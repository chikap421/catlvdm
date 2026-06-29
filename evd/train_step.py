"""Minimal training-step helper for adding EVD losses to a base objective."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .losses import evd_total_loss
from .utils import validate_video_tensor


def evd_training_step(
    pred_update: torch.Tensor,
    event_head: nn.Module,
    base_loss: Optional[torch.Tensor] = None,
    target_update: Optional[torch.Tensor] = None,
    pred_update_tp: Optional[torch.Tensor] = None,
    activity_tp: Optional[torch.Tensor] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute event activity and add EVD losses to a base training loss.

    If ``base_loss`` is omitted and ``target_update`` is provided, a simple MSE
    base loss is used. This keeps the public helper runnable without depending
    on a particular diffusion training loop.
    """

    validate_video_tensor(pred_update, "pred_update")
    event_logits, event_activity = event_head(pred_update)
    if base_loss is None:
        if target_update is None:
            base_loss = pred_update.new_zeros(())
        else:
            validate_video_tensor(target_update, "target_update")
            base_loss = F.mse_loss(pred_update, target_update)

    total, components = evd_total_loss(
        base_loss=base_loss,
        pred_update=pred_update,
        activity=event_activity,
        pred_update_tp=pred_update_tp,
        activity_tp=activity_tp,
        weights=weights,
    )
    result: Dict[str, torch.Tensor] = {
        "loss": total,
        "event_logits": event_logits,
        "event_activity": event_activity,
    }
    result.update(components)
    return result
