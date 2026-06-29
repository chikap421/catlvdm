import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evd import EVDGateState, GenericDiTEVDAdapter


def main() -> None:
    torch.manual_seed(19)
    update = torch.randn(2, 4, 4, 16, 16)
    token_features = torch.randn(2, 4 * 16 * 16, 32)

    adapter = GenericDiTEVDAdapter(update_channels=4, token_dim=32)
    activity = adapter.predict_event_activity(
        token_features=token_features,
        token_to_video=(4, 16, 16),
    )
    gated_update, diagnostics = adapter.gate_update(
        update,
        event_activity=activity,
        gate_state=EVDGateState(),
        t=0.80,
    )

    print(f"activity shape: {tuple(activity.shape)}")
    print(f"gated update shape: {tuple(gated_update.shape)}")
    print(f"gate mean: {float(diagnostics['final_gate'].mean()):.6f}")


if __name__ == "__main__":
    main()
