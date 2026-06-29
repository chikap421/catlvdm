import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    torch.manual_seed(7)
    x = torch.randn(2, 4, 4, 16, 16)
    t = torch.randint(0, 1000, (2,))
    target = torch.randn_like(x)
    model = CATLVDMEVDAdapter(TinyVideoUNet(), enable_evd=True, in_channels=4)

    out = model.forward_train(x, t, base_target=target)
    out["losses"]["total"].backward()

    print(f"pred update shape: {tuple(out['pred_update'].shape)}")
    for name, value in out["losses"].items():
        print(f"{name}: {float(value.detach()):.6f}")


if __name__ == "__main__":
    main()
