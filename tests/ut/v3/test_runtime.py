from __future__ import annotations

from vllm_ascend.v3.runtime import (
    EventRecorder,
    NetworkAwarePrefixPolicy,
    PrefixPolicy,
    ReadyFlowController,
    SingleOwnerTransactionalState,
    TrajectoryKey,
    V3EventKind,
)


def test_network_aware_policy_uses_only_previous_outcomes():
    policy = NetworkAwarePrefixPolicy(bandwidth_mbps=10, rtt_ms=0, retry_ms=0, min_history=4)
    sizes = [5000] * 8
    assert policy.choose(sizes).prefix_length == 8
    for consumed in (2, 3, 2, 3):
        policy.observe(consumed, predicted_end=3)
    decision = policy.choose(sizes, predicted_end=3)
    assert decision.prefix_length == 4
    assert decision.history_source == "predicted"
    assert decision.miss_probability[4] == 1 / 6
    assert policy.choose(sizes, predicted_end=7).history_source == "global"


def test_network_aware_policy_avoids_repair_with_large_rtt():
    policy = NetworkAwarePrefixPolicy(bandwidth_mbps=100, rtt_ms=20, retry_ms=5, min_history=4)
    for consumed in (2, 8, 6, 8):
        policy.observe(consumed)
    assert policy.choose([5000] * 8).prefix_length == 8


def test_network_aware_policy_rejects_invalid_observations():
    import pytest

    policy = NetworkAwarePrefixPolicy(bandwidth_mbps=10, rtt_ms=1, retry_ms=0)
    with pytest.raises(ValueError):
        policy.observe(0)
    with pytest.raises(ValueError):
        policy.choose([1] * 7)
    with pytest.raises(ValueError):
        NetworkAwarePrefixPolicy(bandwidth_mbps=0, rtt_ms=1, retry_ms=0)


def test_prefix_policy_uses_history_and_nack():
    policy = PrefixPolicy()
    assert policy.request_length() == 8
    for consumed in (2, 3, 2):
        assert policy.observe_request(policy.request_length(), consumed) is True
    assert policy.request_length() == 4
    assert policy.observe_request(4, 5) is False
    assert policy.stats()["suffix_requests"] == 1


def test_sots_never_promotes_approximate_state():
    recorder = EventRecorder()
    sots = SingleOwnerTransactionalState(recorder)
    trajectory = TrajectoryKey(
        model_version="qwen35",
        cut_point=32,
        dtype="bf16",
        k=8,
        batch_composition="r0",
        fused_or_separate="separate",
        kernel_id="acl1",
        execution_mode="vllm",
        activation_codec_id="nvfp4-v1",
    )
    epoch = sots.begin(trajectory)
    assert sots.may_promote_approximate_state() is False
    sots.anchor_ready(epoch, [1, 2, 3])
    try:
        sots.commit(epoch, 0, [1], 2, 1)
        raise AssertionError("expected exact-ready failure")
    except RuntimeError:
        pass
    sots.exact_ready(epoch)
    record = sots.commit(epoch, 2, [1, 2, 3], 4, 3)
    assert record.boundary_version == 0
    assert any(event.kind == V3EventKind.COMMIT for event in recorder.events)


def test_ready_flow_depth_one():
    recorder = EventRecorder()
    flow = ReadyFlowController(recorder)
    assert flow.admit(0, "HIGH_CONFIDENCE", True, False)
    assert not flow.admit(1, "HIGH_CONFIDENCE", True, False)
    flow.resolve(0, False)
    assert flow.admit(1, "HIGH_CONFIDENCE", True, False)
