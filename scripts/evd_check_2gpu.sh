#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
torchrun --nproc_per_node="${NPROC_PER_NODE}" examples/evd_train_2gpu.py
