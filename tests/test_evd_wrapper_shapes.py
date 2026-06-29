import torch
from torch import nn

from evd import CATLVDMEVDWrapper, EventHead3D, GenericDiTEVDAdapter, apply_evd_to_update


class TinyVideoModel(nn.Module):
    out_dim = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * x


def test_event_head_3d_shapes() -> None:
    x = torch.randn(2, 4, 3, 8, 8)
    head = EventHead3D(in_channels=4)

    logits, activity = head(x)

    assert logits.shape == (2, 1, 3, 8, 8)
    assert activity.shape == (2, 1, 3, 8, 8)
    assert torch.all(activity >= 0.0)
    assert torch.all(activity <= 1.0)


def test_apply_evd_to_update_preserves_shape() -> None:
    pred_update = torch.randn(2, 4, 3, 8, 8)
    activity = torch.rand(2, 1, 3, 8, 8)

    gated_update, diagnostics = apply_evd_to_update(pred_update, activity, t=0.75)

    assert gated_update.shape == pred_update.shape
    assert diagnostics["final_gate"].shape == activity.shape


def test_catlvdm_wrapper_preserves_disabled_behavior() -> None:
    x = torch.randn(2, 4, 3, 8, 8)
    wrapper = CATLVDMEVDWrapper(TinyVideoModel(), enable_evd=False)

    out = wrapper(x)

    assert torch.allclose(out, 2.0 * x)


def test_catlvdm_wrapper_training_outputs_shapes() -> None:
    x = torch.randn(2, 4, 3, 8, 8)
    wrapper = CATLVDMEVDWrapper(TinyVideoModel(), enable_evd=True)
    wrapper.train()

    out = wrapper(x, target_update=torch.zeros_like(x))

    assert out["pred_update"].shape == x.shape
    assert out["event_logits"].shape == (2, 1, 3, 8, 8)
    assert out["event_activity"].shape == (2, 1, 3, 8, 8)
    assert out["loss"].ndim == 0


def test_generic_dit_adapter_token_activity_shapes() -> None:
    adapter = GenericDiTEVDAdapter(token_dim=16)
    token_features = torch.randn(2, 3 * 4 * 5, 16)

    activity = adapter.predict_event_activity(token_features, token_to_video=(3, 4, 5))

    assert activity.shape == (2, 1, 3, 4, 5)
