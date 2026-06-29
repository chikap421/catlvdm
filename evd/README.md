# Event-Driven Video Generation (EVD)

This directory contains the reference implementation of Event-Driven Video Generation (EVD), accepted to ECCV 2026. EVD adds three components to a video generator: an event head, event-grounded training losses, and event-gated sampling with smoothing, hysteresis, and an early-step schedule.

The implementation supports:

- CAT-LVDM / 3D U-Net backbones via `CATLVDMEVDAdapter`
- video DiT / STDiT-style backbones via `STDiTEVDAdapter`
- CPU, single-GPU, and 2-GPU integration checks

Run from the repository root:

```bash
python examples/evd_catlvdm_train_step.py
python examples/evd_catlvdm_inference_step.py
python examples/evd_stdit_adapter.py
torchrun --nproc_per_node=2 examples/evd_train_2gpu.py
python -m pytest tests/test_evd_* -q
```

Use EVD in training:

```python
from evd.adapters import CATLVDMEVDAdapter

model = CATLVDMEVDAdapter(base_model, enable_evd=True)
out = model.forward_train(x, t, y=y, base_target=target)
loss = out["losses"]["total"]
loss.backward()
```

Use EVD in sampling:

```python
gated_update, info = model.forward_sample(x=x, t=t, y=y, gate_state=gate_state)
```

For DiT/STDiT-style backbones:

```python
from evd.adapters import STDiTEVDAdapter

evd = STDiTEVDAdapter(token_dim=hidden_size)
gated_update, info = evd.forward_from_dit_outputs(
    update=update,
    token_features=final_tokens,
    grid_shape=(T, H, W),
    t=t,
)
```

This code documents and supports the EVD mechanism. It does not include private training data, private checkpoints, or the full-scale DiT-30B training stack used in the paper.
