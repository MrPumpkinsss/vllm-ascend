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


def prefix_mtp_metadata(metadata: Any, prefix_rows: int, total_rows: int) -> Any:
    """Return a one-request K/2 metadata view for an outer fused 2K batch."""
    if prefix_rows < 2 or prefix_rows >= total_rows:
        raise ValueError("prefix rows must be a proper fused prefix")
    query = getattr(metadata, "query_start_loc", None)
    query_cpu = getattr(metadata, "query_start_loc_cpu", None)
    seq_lens = getattr(metadata, "seq_lens", None)
    if (
        not isinstance(query, torch.Tensor)
        or not isinstance(query_cpu, torch.Tensor)
        or not isinstance(seq_lens, torch.Tensor)
        or query_cpu.numel() < 2
        or seq_lens.ndim != 1
        or seq_lens.numel() < 1
    ):
        raise ValueError("missing one-request speculative attention metadata")
    result = copy.copy(metadata)
    device = query.device
    result.query_start_loc = torch.tensor(
        [0, prefix_rows], dtype=query.dtype, device=device
    )
    result.query_start_loc_cpu = torch.tensor(
        [0, prefix_rows], dtype=query_cpu.dtype, device=query_cpu.device
    )
    rejected = total_rows - prefix_rows
    result.seq_lens = seq_lens.clone()
    result.seq_lens[:1] -= rejected
    for name in ("seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound"):
        value = getattr(metadata, name, None)
        if isinstance(value, torch.Tensor) and value.ndim == 1 and value.numel():
            copied = value.clone()
            copied[:1] -= rejected
            setattr(result, name, copied)
    slot_mapping = getattr(metadata, "slot_mapping", None)
    if isinstance(slot_mapping, torch.Tensor) and slot_mapping.numel() >= total_rows:
        result.slot_mapping = slot_mapping.reshape(-1)[:prefix_rows]
    state_indices = getattr(metadata, "spec_state_indices_tensor", None)
    if isinstance(state_indices, torch.Tensor) and state_indices.ndim == 2:
        result.spec_state_indices_tensor = state_indices[:, :prefix_rows]
    positions = getattr(metadata, "positions", None)
    if isinstance(positions, torch.Tensor) and positions.numel() >= total_rows:
        result.positions = positions.reshape(-1)[:prefix_rows]
    positions_cpu = getattr(metadata, "positions_cpu", None)
    if isinstance(positions_cpu, torch.Tensor) and positions_cpu.numel() >= total_rows:
        result.positions_cpu = positions_cpu.reshape(-1)[:prefix_rows]
    result.num_reqs = 1
    result.num_actual_tokens = prefix_rows
    result.num_input_tokens = prefix_rows
    result.max_query_len = prefix_rows
    result.decode_token_per_req = prefix_rows
    result.actual_seq_lengths_q = list(
        getattr(metadata, "actual_seq_lengths_q", [])[:prefix_rows]
    )
    return result
