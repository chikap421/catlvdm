"""Backward-compatible wrapper names for EVD adapters."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch

from .adapters import CATLVDMEVDAdapter, STDiTEVDAdapter
from .gating import EVDGateState


class CATLVDMEVDWrapper(CATLVDMEVDAdapter):
    """Compatibility alias for the CAT-LVDM EVD adapter."""


class GenericDiTEVDAdapter(STDiTEVDAdapter):
    """Compatibility shim for the earlier generic DiT adapter name."""

    def predict_event_activity(
        self,
        token_features: Optional[torch.Tensor] = None,
        token_to_video: Optional[Union[Dict[str, Any], Tuple[int, int, int]]] = None,
        predicted_update: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Any:
        if token_features is not None:
            if token_to_video is None:
                raise ValueError("token_to_video is required with token_features")
            return self.predict_event_activity_from_tokens(
                token_features,
                grid_shape=token_to_video,
                return_logits=return_logits,
            )
        if predicted_update is None:
            raise ValueError("Provide token_features or predicted_update")
        return self.predict_event_activity_from_update(predicted_update, return_logits=return_logits)

    def gate_update(
        self,
        predicted_update: torch.Tensor,
        event_activity: Optional[torch.Tensor] = None,
        token_features: Optional[torch.Tensor] = None,
        token_to_video: Optional[Union[Dict[str, Any], Tuple[int, int, int]]] = None,
        gate_state: Optional[EVDGateState] = None,
        t: Optional[Union[float, torch.Tensor]] = None,
        **_: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if event_activity is None:
            logits, event_activity = self.predict_event_activity(
                token_features=token_features,
                token_to_video=token_to_video,
                predicted_update=predicted_update if token_features is None else None,
                return_logits=True,
            )
        else:
            logits = None
        gated_update, diagnostics = super().gate_update(predicted_update, event_activity, t=t, gate_state=gate_state)
        if logits is not None:
            diagnostics["event_logits"] = logits
        return gated_update, diagnostics
