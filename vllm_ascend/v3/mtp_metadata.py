"""Experimental metadata correction for rejected padded MTP rows."""

from __future__ import annotations

import copy
from typing import Any

import torch


def accepted_draft_metadata(metadata: Any, rejected: torch.Tensor, batch_size: int) -> Any:
    """Copy next-step lengths after excluding rejected target rows."""
    if rejected.ndim != 1 or batch_size < 1 or len(rejected) < batch_size:
        raise ValueError("invalid rejected-token counts")
    rejected_cpu = rejected[:batch_size].detach().to(device="cpu", dtype=torch.int64)
    if (rejected_cpu < 0).any():
        raise ValueError("rejected-token counts cannot be negative")
    original = getattr(metadata, "seq_lens", None)
    if not isinstance(original, torch.Tensor) or original.ndim != 1 or len(original) < batch_size:
        raise ValueError("missing device sequence lengths")
    cpu_fields = ("seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound")
    present_cpu_fields = [name for name in cpu_fields if getattr(metadata, name, None) is not None]
    if not present_cpu_fields:
        raise ValueError("missing CPU sequence-length mirror")
    result = copy.copy(metadata)
    result.seq_lens = original.clone()
    result.seq_lens[:batch_size] -= rejected_cpu.to(device=original.device, dtype=original.dtype)
    for name in present_cpu_fields:
        value = getattr(metadata, name)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or len(value) < batch_size:
            raise ValueError(f"invalid sequence-length mirror: {name}")
        if (value[:batch_size].to(device="cpu", dtype=torch.int64) <= rejected_cpu).any():
            raise ValueError(f"rejected rows exceed sequence length: {name}")
        corrected = value.clone()
        corrected[:batch_size] -= rejected_cpu.to(device=value.device, dtype=value.dtype)
        setattr(result, name, corrected)
    return result
