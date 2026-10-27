from __future__ import annotations

from vllm_ascend.v3.runtime import (
    EventRecorder,
    PrefixPolicy,
    ReadyFlowController,
    SingleOwnerTransactionalState,
    TrajectoryKey,
    V3EventKind,
)


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
