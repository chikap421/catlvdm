import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evd import EVDGateState
from evd.adapters import CATLVDMEVDAdapter


class TinyVideoUNet(nn.Module):
    out_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Conv3d(4, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y=None) -> torch.Tensor:
        del t, y
        return self.net(x)


def main() -> None:
    torch.manual_seed(11)
    x = torch.randn(2, 4, 4, 16, 16)
    t = torch.full((2,), 0.80)
    model = CATLVDMEVDAdapter(TinyVideoUNet(), enable_evd=True, in_channels=4)
    model.eval()
    gate_state = EVDGateState()

    gated_update, diagnostics = model.forward_sample(
        x=x,
        t=t,
        gate_state=gate_state,
    )

    final_gate = diagnostics["final_gate"]
    print(f"gated update shape: {tuple(gated_update.shape)}")
    print(f"gate min/max: {float(final_gate.min()):.6f}/{float(final_gate.max()):.6f}")


if __name__ == "__main__":
    main()
