"""Lightweight distributed helpers for EVD integration checks."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


def distributed_is_initialized() -> bool:
    """Return whether ``torch.distributed`` is available and initialized."""

    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the distributed rank, or zero outside distributed execution."""

    return dist.get_rank() if distributed_is_initialized() else 0


def get_world_size() -> int:
    """Return world size, or one outside distributed execution."""

    return dist.get_world_size() if distributed_is_initialized() else 1


def local_rank() -> int:
    """Return ``LOCAL_RANK`` from ``torchrun`` with a safe default."""

    return int(os.environ.get("LOCAL_RANK", "0"))


def pick_device() -> torch.device:
    """Choose a CUDA device when enough GPUs are visible, else CPU."""

    rank = local_rank()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        torch.cuda.set_device(rank)
        return torch.device("cuda", rank)
    return torch.device("cpu")


def init_distributed(backend: Optional[str] = None) -> torch.device:
    """Initialize ``torch.distributed`` and return the selected device."""

    device = pick_device()
    if not distributed_is_initialized():
        selected_backend = backend or ("nccl" if device.type == "cuda" else "gloo")
        dist.init_process_group(backend=selected_backend)
    return device


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across ranks when distributed is active."""

    if distributed_is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / get_world_size()
    return value


def cleanup_distributed() -> None:
    """Destroy the default process group if this process initialized it."""

    if distributed_is_initialized():
        dist.destroy_process_group()
