"""Configuration helpers for the EVD reference implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass
class EVDConfig:
    """Small typed container for public EVD defaults."""

    enable: bool = True
    beta: float = 12.0
    tau_on: float = 0.62
    tau_off: float = 0.38
    t_star: float = 0.60
    lambda_real: float = 0.12
    lambda_cons: float = 0.08
    lambda_order: float = 0.03
    event_dropout: float = 0.25
    smooth_kernel: int = 3
    cfg_scale: float = 4.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EVDConfig":
        """Build a config from either a flat dict or an ``evd`` block."""

        data = values.get("evd", values)
        if "enable_evd" in data and "enable" not in data:
            data = dict(data)
            data["enable"] = data["enable_evd"]
        valid = {field: data[field] for field in cls.__dataclass_fields__ if field in data}
        return cls(**valid)

    def loss_weights(self) -> Dict[str, float]:
        """Return the loss-weight subset expected by ``evd_total_loss``."""

        return {
            "lambda_real": self.lambda_real,
            "lambda_cons": self.lambda_cons,
            "lambda_order": self.lambda_order,
        }
