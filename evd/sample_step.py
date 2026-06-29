"""Sampling helpers for applying EVD to predicted update fields."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn

from .gating import (
    EVDGateState,
    make_evd_gate,
)
from .utils import validate_video_tensor


def apply_evd_to_update(
    pred_update: torch.Tensor,
    event_activity: torch.Tensor,
    gate_state: Optional[EVDGateState] = None,
    t: Optional[Union[float, torch.Tensor]] = None,
    beta: float = 12.0,
    tau_on: float = 0.62,
    tau_off: float = 0.38,
    t_star: float = 0.60,
    smooth_kernel: int = 3,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Gate a predicted update/noise/velocity field with EVD activity."""

    validate_video_tensor(pred_update, "pred_update")
    validate_video_tensor(event_activity, "event_activity", channels=1)

    prev_gate = gate_state.prev_gate if gate_state is not None else None
    final_gate, binary_gate, diagnostics = make_evd_gate(
        event_activity,
        prev_gate=prev_gate,
        beta=beta,
        tau_on=tau_on,
        tau_off=tau_off,
        t=t,
        t_star=t_star,
        smooth_kernel=smooth_kernel,
    )
    if gate_state is not None:
        gate_state.update(binary_gate)
    return final_gate * pred_update, diagnostics


def _event_head_forward(
    event_head: nn.Module,
    update: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    output = event_head(update)
    if isinstance(output, tuple):
        if len(output) < 2:
            raise ValueError("event_head tuple output must contain logits and activity")
        return output[0], output[1]
    return None, output


def evd_guided_update(
    cond_update: torch.Tensor,
    uncond_update: torch.Tensor,
    cfg_scale: float,
    event_head: nn.Module,
    gate_state: Optional[EVDGateState],
    t: Optional[Union[float, torch.Tensor]],
    beta: float = 12.0,
    tau_on: float = 0.62,
    tau_off: float = 0.38,
    t_star: float = 0.60,
    smooth_kernel: int = 3,
    activity_source: str = "cond",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply classifier-free guidance, then gate the guided update with EVD."""

    validate_video_tensor(cond_update, "cond_update")
    validate_video_tensor(uncond_update, "uncond_update")
    cfg_update = (1.0 + cfg_scale) * cond_update - cfg_scale * uncond_update

    if activity_source == "cond":
        activity_input = cond_update
    elif activity_source == "cfg":
        activity_input = cfg_update
    elif activity_source == "uncond":
        activity_input = uncond_update
    else:
        raise ValueError("activity_source must be one of: cond, cfg, uncond")

    event_logits, event_activity = _event_head_forward(event_head, activity_input)
    gated_update, diagnostics = apply_evd_to_update(
        cfg_update,
        event_activity,
        gate_state=gate_state,
        t=t,
        beta=beta,
        tau_on=tau_on,
        tau_off=tau_off,
        t_star=t_star,
        smooth_kernel=smooth_kernel,
    )
    if event_logits is not None:
        diagnostics["event_logits"] = event_logits
    return gated_update, diagnostics
