import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evd import EVDGateState, EventHead3D, evd_guided_update


def main() -> None:
    torch.manual_seed(11)
    cond_update = torch.randn(2, 4, 4, 16, 16)
    uncond_update = torch.randn(2, 4, 4, 16, 16)
    event_head = EventHead3D(in_channels=4)
    gate_state = EVDGateState()

    gated_update, diagnostics = evd_guided_update(
        cond_update,
        uncond_update,
        cfg_scale=4.0,
        event_head=event_head,
        gate_state=gate_state,
        t=0.80,
    )

    final_gate = diagnostics["final_gate"]
    print(f"gated update shape: {tuple(gated_update.shape)}")
    print(f"gate min/max: {float(final_gate.min()):.6f}/{float(final_gate.max()):.6f}")


if __name__ == "__main__":
    main()
