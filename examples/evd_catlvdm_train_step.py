import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evd import EventHead3D, evd_total_loss


def main() -> None:
    torch.manual_seed(7)
    pred_update = torch.randn(2, 4, 4, 16, 16, requires_grad=True)
    event_head = EventHead3D(in_channels=4)

    _, activity = event_head(pred_update)
    base_loss = pred_update.square().mean()
    total_loss, components = evd_total_loss(base_loss, pred_update, activity)
    total_loss.backward()

    for name, value in components.items():
        print(f"{name}: {float(value.detach()):.6f}")


if __name__ == "__main__":
    main()
