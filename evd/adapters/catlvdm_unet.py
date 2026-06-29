"""CAT-LVDM / 3D U-Net adapter for Event-Driven Video Generation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..event_head import EventHead3D
from ..gating import EVDGateState
from ..losses import evd_total_loss
from ..sample_step import apply_evd_to_update, evd_guided_update
from ..utils import validate_video_tensor


class CATLVDMEVDAdapter(nn.Module):
    """Wrap a CAT-LVDM/3D-U-Net model with optional EVD training and sampling.

    The base model is expected to return a video-shaped update/noise/velocity
    prediction with shape ``[B, C, T, H, W]``. EVD is disabled unless
    ``enable_evd=True``.
    """

    def __init__(
        self,
        base_model: nn.Module,
        event_head: Optional[EventHead3D] = None,
        in_channels: Optional[int] = None,
        enable_evd: bool = True,
        detach_event_input: bool = False,
        loss_weights: Optional[Mapping[str, float]] = None,
        beta: float = 12.0,
        tau_on: float = 0.62,
        tau_off: float = 0.38,
        t_star: float = 0.60,
        smooth_kernel: int = 3,
        cfg_scale: float = 4.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.enable_evd = enable_evd
        self.detach_event_input = detach_event_input
        self.loss_weights = dict(loss_weights or {})
        self.beta = beta
        self.tau_on = tau_on
        self.tau_off = tau_off
        self.t_star = t_star
        self.smooth_kernel = smooth_kernel
        self.cfg_scale = cfg_scale
        self.gate_state = EVDGateState()

        inferred_channels = in_channels
        if inferred_channels is None:
            inferred_channels = getattr(base_model, "out_dim", None)
        self.event_head = event_head
        if self.event_head is None and inferred_channels is not None:
            self.event_head = EventHead3D(
                in_channels=int(inferred_channels),
                detach_input=detach_event_input,
            )

    def _call_base_model(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if y is None:
            return self.base_model(x, t, **kwargs)
        return self.base_model(x, t, y=y, **kwargs)

    def _ensure_event_head(self, pred_update: torch.Tensor) -> EventHead3D:
        if self.event_head is None:
            self.event_head = EventHead3D(
                in_channels=int(pred_update.shape[1]),
                detach_input=self.detach_event_input,
            )
        first_param = next(self.event_head.parameters())
        if first_param.device != pred_update.device or first_param.dtype != pred_update.dtype:
            self.event_head.to(device=pred_update.device, dtype=pred_update.dtype)
        return self.event_head

    def _predict_event_activity(
        self,
        pred_update: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        event_head = self._ensure_event_head(pred_update)
        return event_head(pred_update)

    def forward_train(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        base_target: Optional[torch.Tensor] = None,
        base_loss: Optional[torch.Tensor] = None,
        pred_update: Optional[torch.Tensor] = None,
        pred_update_tp: Optional[torch.Tensor] = None,
        activity_tp: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a training step and return update, event maps, losses, diagnostics."""

        if pred_update is None:
            pred_update = self._call_base_model(x, t, y=y, **kwargs)
        validate_video_tensor(pred_update, "pred_update")

        if base_loss is None and base_target is not None:
            validate_video_tensor(base_target, "base_target")
            base_loss = F.mse_loss(pred_update, base_target)

        if not self.enable_evd:
            losses: Dict[str, torch.Tensor] = {}
            if base_loss is not None:
                losses = {"base": base_loss, "total": base_loss}
            return {
                "pred_update": pred_update,
                "event_logits": None,
                "event_activity": None,
                "losses": losses,
                "diagnostics": {"enabled": False},
            }

        event_logits, event_activity = self._predict_event_activity(pred_update)
        losses = {}
        if base_loss is not None:
            total, components = evd_total_loss(
                base_loss=base_loss,
                pred_update=pred_update,
                activity=event_activity,
                pred_update_tp=pred_update_tp,
                activity_tp=activity_tp,
                weights=self.loss_weights,
                tau_on=self.tau_on,
                tau_off=self.tau_off,
            )
            losses = dict(components)
            losses["total"] = total

        return {
            "pred_update": pred_update,
            "event_logits": event_logits,
            "event_activity": event_activity,
            "losses": losses,
            "diagnostics": {
                "enabled": True,
                "activity_mean": event_activity.detach().mean(),
            },
        }

    def forward_sample(
        self,
        x: Optional[torch.Tensor] = None,
        t: Optional[Union[float, torch.Tensor]] = None,
        y: Optional[torch.Tensor] = None,
        pred_update: Optional[torch.Tensor] = None,
        cond_update: Optional[torch.Tensor] = None,
        uncond_update: Optional[torch.Tensor] = None,
        cfg_scale: Optional[float] = None,
        gate_state: Optional[EVDGateState] = None,
        activity_source: str = "cond",
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run EVD-gated sampling from a base output or precomputed update."""

        state = gate_state if gate_state is not None else self.gate_state
        if cond_update is not None and uncond_update is not None:
            validate_video_tensor(cond_update, "cond_update")
            validate_video_tensor(uncond_update, "uncond_update")
            if not self.enable_evd:
                scale = self.cfg_scale if cfg_scale is None else cfg_scale
                update = (1.0 + scale) * cond_update - scale * uncond_update
                return update, {"enabled": torch.tensor(False)}
            self._ensure_event_head(cond_update)
            return evd_guided_update(
                cond_update=cond_update,
                uncond_update=uncond_update,
                cfg_scale=self.cfg_scale if cfg_scale is None else cfg_scale,
                event_head=self.event_head,
                gate_state=state,
                t=t,
                beta=self.beta,
                tau_on=self.tau_on,
                tau_off=self.tau_off,
                t_star=self.t_star,
                smooth_kernel=self.smooth_kernel,
                activity_source=activity_source,
            )

        if pred_update is None:
            if x is None or t is None:
                raise ValueError("x and t are required when pred_update is not provided")
            pred_update = self._call_base_model(x, t, y=y, **kwargs)
        validate_video_tensor(pred_update, "pred_update")

        if not self.enable_evd:
            return pred_update, {"enabled": torch.tensor(False)}

        event_logits, event_activity = self._predict_event_activity(pred_update)
        gated_update, diagnostics = apply_evd_to_update(
            pred_update,
            event_activity,
            gate_state=state,
            t=t,
            beta=self.beta,
            tau_on=self.tau_on,
            tau_off=self.tau_off,
            t_star=self.t_star,
            smooth_kernel=self.smooth_kernel,
        )
        diagnostics["event_logits"] = event_logits
        return gated_update, diagnostics

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Route to train or sample mode while preserving disabled base behavior."""

        if not self.enable_evd:
            evd_only_keys = {
                "base_target",
                "base_loss",
                "pred_update",
                "pred_update_tp",
                "activity_tp",
                "gate_state",
                "cfg_scale",
                "activity_source",
            }
            base_kwargs = {key: value for key, value in kwargs.items() if key not in evd_only_keys}
            return self._call_base_model(x, t, y=y, **base_kwargs)
        if mode == "sample" or (mode is None and not self.training):
            return self.forward_sample(x=x, t=t, y=y, **kwargs)
        return self.forward_train(x=x, t=t, y=y, **kwargs)
