import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evd.adapters import STDiTEVDAdapter


def main() -> None:
    torch.manual_seed(19)
    batch, channels, frames, height, width = 2, 4, 4, 16, 16
    hidden_size = 32
    update = torch.randn(batch, channels, frames, height, width)
    final_tokens = torch.randn(batch, frames * height * width, hidden_size)

    evd = STDiTEVDAdapter(token_dim=hidden_size, update_channels=channels)
    gated_update, diagnostics = evd.forward_from_dit_outputs(
        update=update,
        token_features=final_tokens,
        grid_shape=(frames, height, width),
        t=0.80,
    )

    print(f"gated update shape: {tuple(gated_update.shape)}")
    print(f"event source: {diagnostics['event_source']}")
    print(f"gate mean: {float(diagnostics['final_gate'].mean()):.6f}")


if __name__ == "__main__":
    main()
