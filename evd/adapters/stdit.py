"""STDiT/Open-Sora-style DiT adapter for EVD."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from ..event_head import EventHead3D, EventHeadTokens
from ..gating import EVDGateState
from ..sample_step import apply_evd_to_update
from ..utils import parse_token_to_video_shape, validate_video_tensor


class STDiTEVDAdapter(nn.Module):
    """Dependency-free EVD adapter for video DiT/STDiT-style backbones.

    This adapter targets STDiT/Open-Sora-style video DiT backbones that expose
    final spatiotemporal token features. It is dependency-free and can be wired
    into any DiT that returns token features and a latent update field.
    """

    def __init__(
        self,
        token_dim: Optional[int] = None,
        update_channels: Optional[int] = None,
        token_hidden_dim: Optional[int] = None,
        detach_input: bool = False,
        enable_evd: bool = True,
        beta: float = 12.0,
        tau_on: float = 0.62,
        tau_off: float = 0.38,
        t_star: float = 0.60,
        smooth_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.detach_input = detach_input
        self.enable_evd = enable_evd
        self.beta = beta
        self.tau_on = tau_on
        self.tau_off = tau_off
        self.t_star = t_star
        self.smooth_kernel = smooth_kernel
        self.token_hidden_dim = token_hidden_dim
        self.gate_state = EVDGateState()
        self.token_event_head = (
            EventHeadTokens(token_dim, hidden_dim=token_hidden_dim, detach_input=detach_input)
            if token_dim is not None
            else None
        )
        self.update_event_head = (
            EventHead3D(update_channels, detach_input=detach_input)
            if update_channels is not None
            else None
        )

    def _ensure_token_head(self, token_features: torch.Tensor) -> EventHeadTokens:
        if self.token_event_head is None:
            self.token_event_head = EventHeadTokens(
                token_dim=token_features.shape[-1],
                hidden_dim=self.token_hidden_dim,
                detach_input=self.detach_input,
            )
        self.token_event_head.to(device=token_features.device, dtype=token_features.dtype)
        return self.token_event_head

    def _ensure_update_head(self, update: torch.Tensor) -> EventHead3D:
        if self.update_event_head is None:
            self.update_event_head = EventHead3D(update.shape[1], detach_input=self.detach_input)
        self.update_event_head.to(device=update.device, dtype=update.dtype)
        return self.update_event_head

    @staticmethod
    def _tokens_to_video(
        token_values: torch.Tensor,
        grid_shape: Union[Dict[str, Any], Tuple[int, int, int]],
    ) -> torch.Tensor:
        t, h, w = parse_token_to_video_shape(grid_shape)
        batch, tokens, channels = token_values.shape
        if channels != 1:
            raise ValueError("token event values must have one channel")
        expected_tokens = t * h * w
        if tokens != expected_tokens:
            raise ValueError(f"token count {tokens} does not match T*H*W={expected_tokens}")
        return token_values.view(batch, t, h, w, 1).permute(0, 4, 1, 2, 3).contiguous()

    def predict_event_activity_from_tokens(
        self,
        token_features: torch.Tensor,
        grid_shape: Union[Dict[str, Any], Tuple[int, int, int]],
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict ``[B, 1, T, H, W]`` event activity from ``[B, N, D]`` tokens."""

        if token_features.ndim != 3:
            raise ValueError("token_features must have shape [B, N, D]")
        token_head = self._ensure_token_head(token_features)
        token_logits, token_activity = token_head(token_features)
        logits = self._tokens_to_video(token_logits, grid_shape)
        activity = self._tokens_to_video(token_activity, grid_shape)
        return (logits, activity) if return_logits else activity

    def predict_event_activity_from_update(
        self,
        update: torch.Tensor,
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Fallback event prediction when a DiT only exposes the update field."""

        validate_video_tensor(update, "update")
        update_head = self._ensure_update_head(update)
        logits, activity = update_head(update)
        return (logits, activity) if return_logits else activity

    def gate_update(
        self,
        update: torch.Tensor,
        event_activity: torch.Tensor,
        t: Optional[Union[float, torch.Tensor]],
        gate_state: Optional[EVDGateState] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Gate a DiT update field with event activity."""

        validate_video_tensor(update, "update")
        validate_video_tensor(event_activity, "event_activity", channels=1)
        if not self.enable_evd:
            return update, {"enabled": torch.tensor(False, device=update.device)}
        state = gate_state if gate_state is not None else self.gate_state
        return apply_evd_to_update(
            update,
            event_activity,
            gate_state=state,
            t=t,
            beta=self.beta,
            tau_on=self.tau_on,
            tau_off=self.tau_off,
            t_star=self.t_star,
            smooth_kernel=self.smooth_kernel,
        )

    def forward_from_dit_outputs(
        self,
        update: torch.Tensor,
        token_features: Optional[torch.Tensor] = None,
        grid_shape: Optional[Union[Dict[str, Any], Tuple[int, int, int]]] = None,
        t: Optional[Union[float, torch.Tensor]] = None,
        gate_state: Optional[EVDGateState] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Gate a DiT output using token activity when available."""

        if not self.enable_evd:
            return update, {"enabled": torch.tensor(False, device=update.device)}
        if token_features is not None:
            if grid_shape is None:
                raise ValueError("grid_shape is required with token_features")
            event_logits, event_activity = self.predict_event_activity_from_tokens(
                token_features,
                grid_shape,
                return_logits=True,
            )
            source = "tokens"
        else:
            event_logits, event_activity = self.predict_event_activity_from_update(
                update,
                return_logits=True,
            )
            source = "update"

        gated_update, diagnostics = self.gate_update(
            update=update,
            event_activity=event_activity,
            t=t,
            gate_state=gate_state,
        )
        diagnostics["event_logits"] = event_logits
        diagnostics["event_source"] = source
        return gated_update, diagnostics
