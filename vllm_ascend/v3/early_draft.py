"""Helpers for a single-request, non-committing early MTP experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_ascend.v3.state_snapshot import IndexedStateSnapshot, UnsupportedStateSnapshot
from vllm_ascend.v3.verification import greedy_matching_drafts


@dataclass(frozen=True)
class EarlyDraft:
    accepted_drafts: int
    root: int
    drafts: tuple[int, ...]
    snapshot_regions: int
    snapshot_ms: float
    draft_ms: float
    restore_ms: float


def greedy_frontier(logits: torch.Tensor, draft_ids: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Return the guessed accepted length and next root without committing it."""
    if logits.ndim != 2 or draft_ids.ndim != 1 or logits.shape[0] != len(draft_ids) + 1:
        raise ValueError("expected K logits and K-1 draft tokens")
    accepted = greedy_matching_drafts(logits, draft_ids, len(draft_ids))
    return accepted, logits[accepted].argmax().reshape(1).to(dtype=torch.int32)


def snapshot_draft_cache(
    forward_context: dict[str, Any], layer_names: list[str], block_table: torch.Tensor
) -> IndexedStateSnapshot:
    """Preserve every allocated draft-cache page named by one request's table."""
    if not layer_names or not isinstance(block_table, torch.Tensor) or block_table.ndim != 2:
        raise UnsupportedStateSnapshot("missing draft layers or block table")
    if block_table.shape[0] != 1:
        raise UnsupportedStateSnapshot("early MTP requires a single request")
    indices = block_table[0].reshape(-1)
    regions: list[tuple[torch.Tensor, torch.Tensor]] = []
    labels: list[str] = []
    for layer_name in layer_names:
        layer = forward_context.get(layer_name)
        cache = getattr(layer, "kv_cache", None)
        if not isinstance(cache, (list, tuple)) or not cache:
            raise UnsupportedStateSnapshot(f"unsupported draft cache for {layer_name}")
        if not all(isinstance(tensor, torch.Tensor) and tensor.ndim >= 2 for tensor in cache):
            raise UnsupportedStateSnapshot(f"unsupported draft cache tensors for {layer_name}")
        for index, tensor in enumerate(cache):
            regions.append((tensor, indices))
            labels.append(f"{layer_name}:{index}")
    return IndexedStateSnapshot(regions, labels)


def compare_early_draft(early: EarlyDraft, sampled: list[int], drafts: list[int]) -> dict[str, Any]:
    """Compare provisional and authoritative paths without promoting either state."""
    if not sampled:
        raise ValueError("authoritative sampled tokens must include a root")
    prefix = 0
    for predicted, actual in zip(early.drafts, drafts):
        if predicted != actual:
            break
        prefix += 1
    same_frontier = early.accepted_drafts == len(sampled) - 1 and early.root == sampled[-1]
    return {
        "predicted_accepted": early.accepted_drafts,
        "exact_accepted": len(sampled) - 1,
        "same_frontier": same_frontier,
        "draft_prefix": prefix,
        "same_chain": same_frontier and list(early.drafts) == drafts,
        "snapshot_regions": early.snapshot_regions,
        "snapshot_ms": early.snapshot_ms,
        "draft_ms": early.draft_ms,
        "restore_ms": early.restore_ms,
    }
