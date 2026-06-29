import torch

from evd import hysteresis_gate, make_evd_gate, scheduled_gate


def test_gate_outputs_are_in_unit_interval() -> None:
    activity = torch.rand(2, 1, 3, 8, 8)
    final_gate, binary_gate = make_evd_gate(activity, t=0.5)

    assert torch.all(final_gate >= 0.0)
    assert torch.all(final_gate <= 1.0)
    assert torch.all((binary_gate == 0.0) | (binary_gate == 1.0))


def test_hysteresis_keeps_previous_state_inside_threshold_band() -> None:
    activity = torch.full((1, 1, 1, 1, 2), 0.5)
    prev_gate = torch.tensor([[[[[1.0, 0.0]]]]])

    gate = hysteresis_gate(activity, prev_gate=prev_gate, tau_on=0.62, tau_off=0.38)

    assert torch.equal(gate, prev_gate)


def test_scheduled_gate_tends_toward_ones() -> None:
    gate = torch.zeros(1, 1, 1, 2, 2)

    early = scheduled_gate(gate, t=0.60, t_star=0.60)
    late = scheduled_gate(gate, t=1.0, t_star=0.60)

    assert torch.allclose(early, gate)
    assert torch.allclose(late, torch.ones_like(gate))
    assert late.mean() > early.mean()
