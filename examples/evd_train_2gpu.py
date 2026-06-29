import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evd.adapters import CATLVDMEVDAdapter
from evd.distributed import all_reduce_mean, cleanup_distributed, get_rank, get_world_size, init_distributed


class TinyVideoUNet(nn.Module):
    out_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Conv3d(4, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y=None) -> torch.Tensor:
        del t, y
        return self.net(x)


def main() -> None:
    device = init_distributed()
    try:
        torch.manual_seed(2026 + get_rank())
        model = CATLVDMEVDAdapter(TinyVideoUNet(), enable_evd=True, in_channels=4).to(device)
        if device.type == "cuda":
            ddp_model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
            )
        else:
            ddp_model = DistributedDataParallel(model)

        x = torch.randn(1, 4, 2, 8, 8, device=device)
        t = torch.randint(0, 1000, (1,), device=device)
        target = torch.randn_like(x)

        out = ddp_model(x, t, base_target=target)
        loss = out["losses"]["total"]
        loss.backward()

        reduced_loss = all_reduce_mean(loss.detach().clone())
        if get_rank() == 0:
            print(
                "2-GPU integration check passed. "
                f"world_size={get_world_size()} loss={float(reduced_loss):.6f}"
            )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
