"""Compact Event-Driven Video Generation reference implementation."""

from .adapters import CATLVDMEVDAdapter, STDiTEVDAdapter
from .config import EVDConfig
from .event_head import EventHead3D, EventHeadTokens
from .gating import (
    EVDGateState,
    hysteresis_gate,
    make_evd_gate,
    scheduled_gate,
    soft_gate,
    spatial_smooth_3d,
)
from .losses import (
    evd_total_loss,
    event_consistency_loss,
    event_ordering_loss,
    event_realization_loss,
)
from .sample_step import apply_evd_to_update, evd_guided_update
from .wrappers import CATLVDMEVDWrapper, GenericDiTEVDAdapter

__all__ = [
    "CATLVDMEVDWrapper",
    "CATLVDMEVDAdapter",
    "EVDConfig",
    "EVDGateState",
    "EventHead3D",
    "EventHeadTokens",
    "GenericDiTEVDAdapter",
    "STDiTEVDAdapter",
    "apply_evd_to_update",
    "evd_guided_update",
    "evd_total_loss",
    "event_consistency_loss",
    "event_ordering_loss",
    "event_realization_loss",
    "hysteresis_gate",
    "make_evd_gate",
    "scheduled_gate",
    "soft_gate",
    "spatial_smooth_3d",
]
