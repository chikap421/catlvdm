# Event-Driven Video Generation (EVD) Reference

This directory contains a compact, readable reference implementation of Event-Driven Video Generation (EVD). It is designed to be usable as a standalone project or as a small package embedded inside CAT-LVDM.

EVD can be attached to an existing video generator by adding:

1. an event head,
2. event-grounded losses during training,
3. event-gated update modification during sampling.

This implementation provides a CAT-LVDM/3D-U-Net wrapper and a generic DiT adapter skeleton. It shows how to wire EVD into training and inference without releasing private training data, private weights, or full-scale private experiments.

## Contents

- `event_head.py`: lightweight 3D event head for video update fields.
- `gating.py`: spatial smoothing, soft activation, hysteresis, and sampler scheduling.
- `losses.py`: public reference EVD training losses.
- `sample_step.py`: inference-time update gating and CFG helper.
- `train_step.py`: minimal training-step helper.
- `wrappers.py`: CAT-LVDM wrapper and generic DiT adapter.
- `utils.py`: shared tensor validation and reshape helpers.

## Usage

From the CAT-LVDM repository root:

```bash
python examples/evd_catlvdm_train_step.py
python examples/evd_catlvdm_inference_step.py
python examples/evd_generic_dit_adapter.py
python -m pytest tests/test_evd_* -q
```

Minimal training-side use:

```python
import torch

from evd import EventHead3D, evd_total_loss

pred_update = torch.randn(2, 4, 4, 16, 16, requires_grad=True)
event_head = EventHead3D(in_channels=4)

_, activity = event_head(pred_update)
base_loss = pred_update.square().mean()
loss, components = evd_total_loss(base_loss, pred_update, activity)
loss.backward()
```

Minimal inference-side use:

```python
import torch

from evd import EVDGateState, EventHead3D, evd_guided_update

cond_update = torch.randn(2, 4, 4, 16, 16)
uncond_update = torch.randn(2, 4, 4, 16, 16)
event_head = EventHead3D(in_channels=4)

gated_update, diagnostics = evd_guided_update(
    cond_update,
    uncond_update,
    cfg_scale=4.0,
    event_head=event_head,
    gate_state=EVDGateState(),
    t=0.80,
)
```

## Standalone Extraction

The `evd/` package only depends on PyTorch. To use it in another video generation codebase, copy this directory into that project, then instantiate `EventHead3D` on a predicted update/noise/velocity tensor shaped `[B, C, T, H, W]`.

For DiT-style pipelines, use `GenericDiTEVDAdapter` when final token features are available. Otherwise, use the update-field event head directly.

## Scope

This is public reference code for the EVD mechanism: event activity prediction, event-grounded losses, and event-gated sampling. It is not a release of private training data or private weights, and it is not intended to reproduce every full-scale paper number by itself.
