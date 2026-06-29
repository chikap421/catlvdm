import torch
from torch import nn

from evd import EVDGateState
from evd.adapters import CATLVDMEVDAdapter, STDiTEVDAdapter


class TinyVideoModel(nn.Module):
    out_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv3d(4, 4, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y=None) -> torch.Tensor:
        del t, y
        return self.proj(x)


def test_catlvdm_adapter_disabled_returns_base_output() -> None:
    x = torch.randn(2, 4, 3, 8, 8)
    t = torch.randint(0, 1000, (2,))
    base = TinyVideoModel()
    adapter = CATLVDMEVDAdapter(base, enable_evd=False, in_channels=4)

    expected = base(x, t)
    actual = adapter(x, t)

    assert torch.allclose(actual, expected)


def test_catlvdm_adapter_train_and_sample_shapes() -> None:
    x = torch.randn(2, 4, 3, 8, 8)
    t = torch.randint(0, 1000, (2,))
    target = torch.randn_like(x)
    adapter = CATLVDMEVDAdapter(TinyVideoModel(), enable_evd=True, in_channels=4)

    out = adapter.forward_train(x, t, base_target=target)
    gated, diagnostics = adapter.forward_sample(
        pred_update=out["pred_update"],
        t=0.75,
        gate_state=EVDGateState(),
    )

    assert out["pred_update"].shape == x.shape
    assert out["event_activity"].shape == (2, 1, 3, 8, 8)
    assert out["losses"]["total"].ndim == 0
    assert gated.shape == x.shape
    assert diagnostics["final_gate"].shape == (2, 1, 3, 8, 8)


def test_stdit_adapter_token_path_shapes() -> None:
    adapter = STDiTEVDAdapter(token_dim=16, update_channels=4)
    update = torch.randn(2, 4, 3, 4, 5)
    tokens = torch.randn(2, 3 * 4 * 5, 16)

    activity = adapter.predict_event_activity_from_tokens(tokens, grid_shape=(3, 4, 5))
    gated, diagnostics = adapter.forward_from_dit_outputs(
        update=update,
        token_features=tokens,
        grid_shape=(3, 4, 5),
        t=0.80,
    )

    assert activity.shape == (2, 1, 3, 4, 5)
    assert gated.shape == update.shape
    assert diagnostics["event_source"] == "tokens"


def test_stdit_adapter_update_fallback_shapes() -> None:
    adapter = STDiTEVDAdapter(update_channels=4)
    update = torch.randn(2, 4, 3, 4, 5)

    gated, diagnostics = adapter.forward_from_dit_outputs(update=update, t=0.80)

    assert gated.shape == update.shape
    assert diagnostics["event_source"] == "update"
