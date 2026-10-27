from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.v3.state_snapshot import (
    UnsupportedStateSnapshot,
    snapshot_attention_committed_state,
    snapshot_attention_state,
    snapshot_attention_suffix_state,
    snapshot_gdn_committed_state,
    snapshot_gdn_prefix,
    tensors_bitwise_equal,
)


def test_snapshot_restores_gdn_and_attention_pages_without_rebinding():
    conv = torch.arange(30, dtype=torch.float32).reshape(5, 2, 3)
    recurrent = torch.arange(40, dtype=torch.float32).reshape(5, 2, 4)
    key = torch.arange(80, dtype=torch.float32).reshape(5, 4, 4)
    value = key.clone() + 100
    originals = [tensor.clone() for tensor in (conv, recurrent, key, value)]
    pointers = [tensor.data_ptr() for tensor in (conv, recurrent, key, value)]
    context = {
        "gdn": SimpleNamespace(kv_cache=[conv, recurrent]),
        "attention": SimpleNamespace(kv_cache=(key, value)),
    }
    metadata = {
        "gdn": SimpleNamespace(spec_state_indices_tensor=torch.tensor([[1, 3, -1]])),
        "attention": SimpleNamespace(slot_mapping=torch.tensor([4, 7, 12, -1])),
    }
    snapshot = snapshot_attention_state(context, metadata)
    for tensor in (conv, recurrent, key, value):
        tensor[1] = -10
        tensor[3] = -20
    assert not snapshot.matches()
    snapshot.restore()
    assert snapshot.matches()
    assert snapshot.region_count == 4
    assert all(torch.equal(tensor, original) for tensor, original in zip((conv, recurrent, key, value), originals))
    assert [tensor.data_ptr() for tensor in (conv, recurrent, key, value)] == pointers


def test_snapshot_rejects_unknown_or_out_of_range_layout():
    cache = torch.zeros(3, 2, 4)
    context = {"layer": SimpleNamespace(kv_cache=[cache])}
    with pytest.raises(UnsupportedStateSnapshot):
        snapshot_attention_state(context, {"layer": SimpleNamespace()})
    with pytest.raises(UnsupportedStateSnapshot):
        snapshot_attention_state(context, {"layer": SimpleNamespace(spec_state_indices_tensor=torch.tensor([5]))})
    with pytest.raises(UnsupportedStateSnapshot):
        snapshot_attention_state(context, {"layer": SimpleNamespace(spec_state_indices_tensor=torch.tensor([-1]))})


def test_gdn_prefix_comparison_ignores_later_pages():
    cache = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2)
    context = {"layer": SimpleNamespace(kv_cache=[cache])}
    metadata = {"layer": SimpleNamespace(spec_state_indices_tensor=torch.tensor([[1, 2, 3, 4]]))}
    snapshot = snapshot_gdn_prefix(context, metadata, 2)
    cache[3] = -1
    assert snapshot.matches()
    cache[2] = -1
    assert not snapshot.matches()
    assert snapshot.differences()[0]["pages"] == [2]


def test_gdn_committed_state_ignores_unaccepted_conv_and_recurrent_slots():
    conv = torch.zeros((6, 10, 2))
    recurrent = torch.zeros((6, 2, 2))
    layer = SimpleNamespace(kv_cache=[conv, recurrent], conv1d=SimpleNamespace(weight=torch.zeros((2, 1, 4))))
    metadata = {"layer": SimpleNamespace(spec_state_indices_tensor=torch.tensor([[1, 2, 3, 4]]))}
    snapshot = snapshot_gdn_committed_state({"layer": layer}, metadata, consumed_end=2)
    conv[1, 5:] = 1
    recurrent[3] = 1
    assert snapshot.matches()
    conv[1, 2] = 1
    assert not snapshot.matches()


def test_attention_committed_state_ignores_unaccepted_slots_in_same_page():
    key = torch.zeros((4, 4, 2))
    value = torch.zeros((4, 4, 2))
    context = {"attention": SimpleNamespace(kv_cache=(key, value))}
    metadata = {"attention": SimpleNamespace(slot_mapping=torch.tensor([7, 8, 9, 10]))}
    snapshot = snapshot_attention_committed_state(context, metadata, consumed_end=2)
    key[2, 1] = 1
    value[1, 2] = 1
    assert snapshot.matches()
    value[2, 0] = 1
    assert not snapshot.matches()
    assert snapshot.differences()[0]["region"] == "attention:1:page[2]"


def test_attention_committed_state_rejects_invalid_slots():
    context = {"attention": SimpleNamespace(kv_cache=(torch.zeros((2, 4, 2)),))}
    for slots in (torch.tensor([-1]), torch.tensor([8])):
        with pytest.raises(UnsupportedStateSnapshot):
            snapshot_attention_committed_state(
                context, {"attention": SimpleNamespace(slot_mapping=slots)}, consumed_end=1
            )


def test_bitwise_tensor_comparison_distinguishes_signed_zero_and_nan():
    original = torch.tensor([0.0, float("nan")], dtype=torch.bfloat16)
    equal = original.clone()
    assert tensors_bitwise_equal(original, equal)
    equal[0] = -0.0
    assert not tensors_bitwise_equal(original, equal)
    assert not tensors_bitwise_equal(original, original.float())


def test_suffix_snapshot_restores_only_paged_suffix_slots():
    key = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2).clone()
    value = key + 100
    context = {"page": SimpleNamespace(kv_cache=(key, value))}
    metadata = {"page": SimpleNamespace(slot_mapping=torch.tensor([0, 1, 2, 3]))}
    snapshot = snapshot_attention_suffix_state(context, metadata, suffix_start=2)
    original_key, original_value = key.clone(), value.clone()
    key[1, 1] = -1000
    value[1, 1] = -2000
    snapshot.restore()
    assert torch.equal(key, original_key)
    assert torch.equal(value, original_value)


def test_gdn_state_is_snapshotted_whole_for_fused_suffix():
    recurrent = torch.tensor([[1.0], [2.0]])
    context = {"gdn": SimpleNamespace(kv_cache=(torch.zeros(2, 1, 1), recurrent))}
    metadata = {"gdn": SimpleNamespace(spec_state_indices_tensor=torch.tensor([[0, 1]]))}
    snapshot = snapshot_attention_suffix_state(context, metadata, suffix_start=1)
    recurrent[:] = -1
    snapshot.restore()
    assert torch.equal(recurrent, torch.tensor([[1.0], [2.0]]))
