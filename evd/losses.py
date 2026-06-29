"""Training losses for the compact EVD reference implementation."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch

from .utils import validate_video_tensor


def event_realization_loss(pred_update: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
    """Penalize update energy where event activity is low."""

    validate_video_tensor(pred_update, "pred_update")
    validate_video_tensor(activity, "activity", channels=1)
    return (((1.0 - activity) * pred_update) ** 2).mean()


def event_consistency_loss(
    pred_update_t: torch.Tensor,
    activity_t: torch.Tensor,
    pred_update_tp: torch.Tensor,
    activity_tp: torch.Tensor,
) -> torch.Tensor:
    """Keep activity-weighted update fields consistent across paired steps."""

    validate_video_tensor(pred_update_t, "pred_update_t")
    validate_video_tensor(activity_t, "activity_t", channels=1)
    validate_video_tensor(pred_update_tp, "pred_update_tp")
    validate_video_tensor(activity_tp, "activity_tp", channels=1)
    return ((activity_t * pred_update_t - activity_tp * pred_update_tp) ** 2).mean()


def event_ordering_loss(
    pred_update: torch.Tensor,
    activity: torch.Tensor,
    tau_on: float = 0.62,
    tau_off: float = 0.38,
) -> torch.Tensor:
    """Suppress update energy before activity crosses the event thresholds."""

    validate_video_tensor(pred_update, "pred_update")
    validate_video_tensor(activity, "activity", channels=1)
    below_on = (activity < tau_on).to(dtype=pred_update.dtype)
    below_off = (activity < tau_off).to(dtype=pred_update.dtype)
    return ((below_on + below_off) * pred_update.square()).mean()


def _read_weight(weights: Optional[Mapping[str, float]], key: str, default: float) -> float:
    if weights is None:
        return default
    short_key = key.replace("lambda_", "")
    return float(weights.get(key, weights.get(short_key, default)))


def evd_total_loss(
    base_loss: torch.Tensor,
    pred_update: torch.Tensor,
    activity: torch.Tensor,
    pred_update_tp: Optional[torch.Tensor] = None,
    activity_tp: Optional[torch.Tensor] = None,
    weights: Optional[Mapping[str, float]] = None,
    lambda_real: float = 0.12,
    lambda_cons: float = 0.08,
    lambda_order: float = 0.03,
    tau_on: float = 0.62,
    tau_off: float = 0.38,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Combine a base diffusion loss with the public EVD reference losses."""

    lambda_real = _read_weight(weights, "lambda_real", lambda_real)
    lambda_cons = _read_weight(weights, "lambda_cons", lambda_cons)
    lambda_order = _read_weight(weights, "lambda_order", lambda_order)

    real = event_realization_loss(pred_update, activity)
    order = event_ordering_loss(pred_update, activity, tau_on=tau_on, tau_off=tau_off)
    if pred_update_tp is not None and activity_tp is not None:
        cons = event_consistency_loss(pred_update, activity, pred_update_tp, activity_tp)
    else:
        cons = pred_update.new_zeros(())

    total = base_loss + lambda_real * real + lambda_cons * cons + lambda_order * order
    components = {
        "base": base_loss.detach(),
        "event_realization": real,
        "event_consistency": cons,
        "event_ordering": order,
        "total": total,
    }
    return total, components
