"""Inference-time EVD gates with smoothing, hysteresis, and scheduling."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .utils import validate_video_tensor


def spatial_smooth_3d(activity: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Smooth event activity over ``H/W`` independently for each frame.

    The input and output shape is ``[B, 1, T, H, W]``. Time is not pooled by
    default, matching the EVD sampler's per-frame event map.
    """

    validate_video_tensor(activity, "activity", channels=1)
    if kernel_size <= 1:
        return activity
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd so output shape is preserved")

    pad = kernel_size // 2
    padded = F.pad(activity, (pad, pad, pad, pad, 0, 0), mode="replicate")
    return F.avg_pool3d(padded, kernel_size=(1, kernel_size, kernel_size), stride=1)


def soft_gate(
    activity: torch.Tensor,
    beta: float = 12.0,
    tau_on: float = 0.62,
    tau_off: float = 0.38,
) -> torch.Tensor:
    """Compute the differentiable soft event gate."""

    center = (tau_on + tau_off) / 2.0
    return torch.sigmoid(beta * (activity - center))


def hysteresis_gate(
    activity: torch.Tensor,
    prev_gate: Optional[torch.Tensor] = None,
    tau_on: float = 0.62,
    tau_off: float = 0.38,
) -> torch.Tensor:
    """Apply binary EVD hysteresis.

    Activity turns on above ``tau_on``, turns off below ``tau_off``, and keeps
    the previous binary state inside the threshold band.
    """

    validate_video_tensor(activity, "activity", channels=1)
    if prev_gate is None:
        prev_gate = torch.zeros_like(activity)
    else:
        prev_gate = prev_gate.to(device=activity.device, dtype=activity.dtype)
        if prev_gate.shape != activity.shape:
            prev_gate = torch.broadcast_to(prev_gate, activity.shape)

    ones = torch.ones_like(activity)
    zeros = torch.zeros_like(activity)
    return torch.where(activity >= tau_on, ones, torch.where(activity <= tau_off, zeros, prev_gate))


def scheduled_gate(
    gate: torch.Tensor,
    t: Union[float, torch.Tensor],
    t_star: float = 0.60,
) -> torch.Tensor:
    """Relax the event gate toward one late in sampling.

    ``rho(t)=1`` for ``t <= t_star`` and linearly decays to ``0`` at ``t=1``.
    The returned gate is ``rho * gate + (1-rho) * 1``.
    """

    t_tensor = torch.as_tensor(t, device=gate.device, dtype=gate.dtype)
    if t_tensor.ndim == 1:
        t_tensor = t_tensor.view(-1, 1, 1, 1, 1)
    elif t_tensor.ndim > 1 and t_tensor.ndim < gate.ndim:
        t_tensor = t_tensor.reshape(*t_tensor.shape, *([1] * (gate.ndim - t_tensor.ndim)))

    denom = max(1.0 - float(t_star), 1.0e-6)
    rho = torch.where(
        t_tensor <= t_star,
        torch.ones_like(t_tensor),
        1.0 - (t_tensor - float(t_star)) / denom,
    ).clamp(0.0, 1.0)
    return rho * gate + (1.0 - rho) * torch.ones_like(gate)


class EVDGateState:
    """Hold the previous binary hysteresis gate across sampling steps."""

    def __init__(self, prev_gate: Optional[torch.Tensor] = None) -> None:
        self.prev_gate = prev_gate

    def reset(self) -> None:
        """Clear the stored hysteresis state."""

        self.prev_gate = None

    def update(self, binary_gate: torch.Tensor) -> torch.Tensor:
        """Store and return a detached binary gate."""

        self.prev_gate = binary_gate.detach()
        return self.prev_gate

    def make_gate(
        self,
        activity: torch.Tensor,
        beta: float = 12.0,
        tau_on: float = 0.62,
        tau_off: float = 0.38,
        t: Optional[Union[float, torch.Tensor]] = None,
        t_star: float = 0.60,
        smooth_kernel: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute a gate using the stored previous hysteresis state."""

        final_gate, binary_gate, diagnostics = make_evd_gate(
            activity=activity,
            prev_gate=self.prev_gate,
            beta=beta,
            tau_on=tau_on,
            tau_off=tau_off,
            t=t,
            t_star=t_star,
            smooth_kernel=smooth_kernel,
        )
        self.update(binary_gate)
        return final_gate, binary_gate, diagnostics


def make_evd_gate(
    activity: torch.Tensor,
    prev_gate: Optional[torch.Tensor] = None,
    beta: float = 12.0,
    tau_on: float = 0.62,
    tau_off: float = 0.38,
    t: Optional[Union[float, torch.Tensor]] = None,
    t_star: float = 0.60,
    smooth_kernel: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Create the final EVD gate, binary state, and diagnostics."""

    smoothed = spatial_smooth_3d(activity, kernel_size=smooth_kernel)
    differentiable_gate = soft_gate(smoothed, beta=beta, tau_on=tau_on, tau_off=tau_off)
    binary_gate = hysteresis_gate(smoothed, prev_gate=prev_gate, tau_on=tau_on, tau_off=tau_off)
    final_gate = differentiable_gate * binary_gate
    if t is not None:
        final_gate = scheduled_gate(final_gate, t=t, t_star=t_star)
    diagnostics = {
        "activity": activity,
        "smoothed_activity": smoothed,
        "soft_gate": differentiable_gate,
        "binary_gate": binary_gate,
        "final_gate": final_gate,
    }
    return final_gate, binary_gate, diagnostics
