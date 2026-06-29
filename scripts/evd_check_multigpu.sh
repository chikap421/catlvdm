#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$(python - <<'PY'
import torch

count = torch.cuda.device_count()
print(count if count > 0 else 1)
PY
)"
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" examples/evd_train_multigpu.py
