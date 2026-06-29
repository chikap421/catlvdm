import torch

from evd import (
    evd_total_loss,
    event_consistency_loss,
    event_ordering_loss,
    event_realization_loss,
)


def test_evd_losses_are_finite_scalars() -> None:
    pred_update = torch.randn(2, 4, 3, 8, 8, requires_grad=True)
    activity = torch.sigmoid(torch.randn(2, 1, 3, 8, 8, requires_grad=True))
    pred_update_tp = torch.randn(2, 4, 3, 8, 8, requires_grad=True)
    activity_tp = torch.sigmoid(torch.randn(2, 1, 3, 8, 8, requires_grad=True))

    losses = [
        event_realization_loss(pred_update, activity),
        event_consistency_loss(pred_update, activity, pred_update_tp, activity_tp),
        event_ordering_loss(pred_update, activity),
    ]

    for loss in losses:
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_evd_total_loss_backward_works() -> None:
    pred_update = torch.randn(2, 4, 3, 8, 8, requires_grad=True)
    activity = torch.sigmoid(torch.randn(2, 1, 3, 8, 8, requires_grad=True))
    base_loss = pred_update.square().mean()

    total, components = evd_total_loss(base_loss, pred_update, activity)
    total.backward()

    assert total.ndim == 0
    assert torch.isfinite(total)
    assert pred_update.grad is not None
    assert set(components) >= {
        "base",
        "event_realization",
        "event_consistency",
        "event_ordering",
        "total",
    }
