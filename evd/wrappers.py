"""Reference wrappers for CAT-LVDM/3D U-Net and generic video DiT pipelines."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .event_head import EventHead3D
from .gating import EVDGateState
from .losses import evd_total_loss
from .sample_step import apply_evd_to_update
from .utils import parse_token_to_video_shape, validate_video_tensor


class CATLVDMEVDWrapper(nn.Module):
    """Attach EVD to a CAT-LVDM/3D-U-Net-style video generator.

    The wrapped model is called normally. Its output is treated as the
    update/noise/velocity field. With ``enable_evd=False``, this wrapper returns
    the base model output unchanged.
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
        self.gate_state = EVDGateState()

        inferred_channels = in_channels
        if inferred_channels is None:
            inferred_channels = getattr(base_model, "out_dim", None)
        self.event_head = event_head
        if self.event_head is None and inferred_channels is not None and enable_evd:
            self.event_head = EventHead3D(
                in_channels=int(inferred_channels),
                detach_input=detach_event_input,
            )

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

    def forward(
        self,
        *args: Any,
        enable_evd: Optional[bool] = None,
        base_loss: Optional[torch.Tensor] = None,
        target_update: Optional[torch.Tensor] = None,
        pred_update_tp: Optional[torch.Tensor] = None,
        activity_tp: Optional[torch.Tensor] = None,
        inference: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        gate_state: Optional[EVDGateState] = None,
        evd_t: Optional[Union[float, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run the base model and optionally apply EVD training or sampling logic."""

        pred_update = self.base_model(*args, **kwargs)
        validate_video_tensor(pred_update, "base_model output")

        active = self.enable_evd if enable_evd is None else enable_evd
        if not active:
            return pred_update

        event_head = self._ensure_event_head(pred_update)
        event_logits, event_activity = event_head(pred_update)
        is_inference = (not self.training) if inference is None else inference
        if return_dict is None:
            return_dict = not is_inference

        if is_inference:
            state = gate_state if gate_state is not None else self.gate_state
            gated_update, diagnostics = apply_evd_to_update(
                pred_update,
                event_activity,
                gate_state=state,
                t=evd_t,
                beta=self.beta,
                tau_on=self.tau_on,
                tau_off=self.tau_off,
                t_star=self.t_star,
                smooth_kernel=self.smooth_kernel,
            )
            if not return_dict:
                return gated_update
            return {
                "pred_update": pred_update,
                "gated_update": gated_update,
                "event_logits": event_logits,
                "event_activity": event_activity,
                "evd_diagnostics": diagnostics,
            }

        result: Dict[str, torch.Tensor] = {
            "pred_update": pred_update,
            "event_logits": event_logits,
            "event_activity": event_activity,
        }
        if base_loss is None and target_update is not None:
            validate_video_tensor(target_update, "target_update")
            base_loss = F.mse_loss(pred_update, target_update)
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
            result["loss"] = total
            result["evd_losses"] = components
        return result if return_dict else pred_update


class GenericDiTEVDAdapter(nn.Module):
    """Backbone-agnostic EVD adapter for conventional video DiT pipelines.

    Use this adapter when a DiT exposes final token features; otherwise use the
    update-field event head.
    """

    def __init__(
        self,
        update_channels: Optional[int] = None,
        token_dim: Optional[int] = None,
        detach_input: bool = False,
    ) -> None:
        super().__init__()
        self.detach_input = detach_input
        self.update_head = EventHead3D(update_channels, detach_input=detach_input) if update_channels else None
        self.token_head = self._make_token_head(token_dim) if token_dim else None

    @staticmethod
    def _make_token_head(token_dim: int) -> nn.Module:
        head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))
        nn.init.normal_(head[-1].weight, mean=0.0, std=1.0e-5)
        nn.init.zeros_(head[-1].bias)
        return head

    def _ensure_update_head(self, update: torch.Tensor) -> EventHead3D:
        if self.update_head is None:
            self.update_head = EventHead3D(update.shape[1], detach_input=self.detach_input)
        self.update_head.to(device=update.device, dtype=update.dtype)
        return self.update_head

    def _ensure_token_head(self, token_features: torch.Tensor) -> nn.Module:
        if self.token_head is None:
            self.token_head = self._make_token_head(token_features.shape[-1])
        self.token_head.to(device=token_features.device, dtype=token_features.dtype)
        return self.token_head

    def predict_event_activity(
        self,
        token_features: Optional[torch.Tensor] = None,
        token_to_video: Optional[Union[Dict[str, Any], Tuple[int, int, int]]] = None,
        predicted_update: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[Optional[torch.Tensor], torch.Tensor]]:
        """Predict activity from DiT tokens or directly from a video update field."""

        if token_features is not None:
            if token_features.ndim != 3:
                raise ValueError("token_features must have shape [B, N, D]")
            if token_to_video is None:
                raise ValueError("token_to_video is required with token_features")
            t, h, w = parse_token_to_video_shape(token_to_video)
            batch, tokens, _ = token_features.shape
            if tokens != t * h * w:
                raise ValueError(f"token count {tokens} does not match T*H*W={t*h*w}")
            x = token_features.detach() if self.detach_input else token_features
            token_head = self._ensure_token_head(x)
            token_logits = token_head(x)
            activity = torch.sigmoid(token_logits).view(batch, t, h, w, 1)
            activity = activity.permute(0, 4, 1, 2, 3).contiguous()
            logits = token_logits.view(batch, t, h, w, 1).permute(0, 4, 1, 2, 3).contiguous()
            return (logits, activity) if return_logits else activity

        if predicted_update is None:
            raise ValueError("Provide token_features or predicted_update")
        validate_video_tensor(predicted_update, "predicted_update")
        update_head = self._ensure_update_head(predicted_update)
        logits, activity = update_head(predicted_update)
        return (logits, activity) if return_logits else activity

    def gate_update(
        self,
        predicted_update: torch.Tensor,
        event_activity: Optional[torch.Tensor] = None,
        token_features: Optional[torch.Tensor] = None,
        token_to_video: Optional[Union[Dict[str, Any], Tuple[int, int, int]]] = None,
        gate_state: Optional[EVDGateState] = None,
        t: Optional[Union[float, torch.Tensor]] = None,
        beta: float = 12.0,
        tau_on: float = 0.62,
        tau_off: float = 0.38,
        t_star: float = 0.60,
        smooth_kernel: int = 3,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Gate a DiT-predicted video update using token or update-field activity."""

        logits: Optional[torch.Tensor] = None
        if event_activity is None:
            logits, event_activity = self.predict_event_activity(
                token_features=token_features,
                token_to_video=token_to_video,
                predicted_update=predicted_update if token_features is None else None,
                return_logits=True,
            )
        gated_update, diagnostics = apply_evd_to_update(
            predicted_update,
            event_activity,
            gate_state=gate_state,
            t=t,
            beta=beta,
            tau_on=tau_on,
            tau_off=tau_off,
            t_star=t_star,
            smooth_kernel=smooth_kernel,
        )
        if logits is not None:
            diagnostics["event_logits"] = logits
        return gated_update, diagnostics
