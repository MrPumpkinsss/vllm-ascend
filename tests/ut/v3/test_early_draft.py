from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.v3.early_draft import EarlyDraft, compare_early_draft, greedy_frontier, snapshot_draft_cache
from vllm_ascend.v3.state_snapshot import UnsupportedStateSnapshot


def test_greedy_frontier_handles_rejected_and_full_draft():
    logits = torch.tensor([[0, 5, 1], [4, 0, 1], [0, 1, 3]], dtype=torch.float32)
    accepted, root = greedy_frontier(logits, torch.tensor([1, 2]))
    assert accepted == 1
    assert root.tolist() == [0]
    accepted, root = greedy_frontier(logits, torch.tensor([1, 0]))
    assert accepted == 2
    assert root.tolist() == [2]
    with pytest.raises(ValueError):
        greedy_frontier(logits[:2], torch.tensor([1, 0]))


def test_draft_cache_snapshot_restores_all_allocated_pages():
    key = torch.arange(30, dtype=torch.float32).reshape(5, 2, 3)
    value = key.clone()
    original_key, original_value = key.clone(), value.clone()
    context = {"draft": SimpleNamespace(kv_cache=[key, value])}
    snapshot = snapshot_draft_cache(context, ["draft"], torch.tensor([[1, 3, 3, -1]]))
    key[1] = -1
    value[3] = -1
    key[2] = -7
    snapshot.restore()
    assert snapshot.matches()
    assert snapshot.region_count == 2
    assert torch.equal(key[1], original_key[1])
    assert torch.equal(value, original_value)
    assert (key[2] == -7).all()


def test_draft_cache_snapshot_rejects_unsupported_layout():
    cache = torch.zeros((3, 2, 2))
    context = {"draft": SimpleNamespace(kv_cache=[cache])}
    with pytest.raises(UnsupportedStateSnapshot):
        snapshot_draft_cache(context, ["missing"], torch.tensor([[1]]))
    with pytest.raises(UnsupportedStateSnapshot):
        snapshot_draft_cache(context, ["draft"], torch.tensor([[4]]))
    with pytest.raises(UnsupportedStateSnapshot):
        snapshot_draft_cache(context, ["draft"], torch.tensor([[1], [2]]))


def test_compare_early_draft_requires_matching_frontier_and_chain():
    early = EarlyDraft(1, 4, (5, 6, 7), 2, 1.0, 2.0, 3.0)
    assert compare_early_draft(early, [8, 4], [5, 6, 7])["same_chain"]
    assert not compare_early_draft(early, [8, 9, 4], [5, 6, 7])["same_chain"]
    mismatch = compare_early_draft(early, [8, 4], [5, 9, 7])
    assert mismatch["same_frontier"] and mismatch["draft_prefix"] == 1
    assert not mismatch["same_chain"]
    with pytest.raises(ValueError):
        compare_early_draft(early, [], [5, 6, 7])
