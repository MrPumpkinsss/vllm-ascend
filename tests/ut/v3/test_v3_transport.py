from __future__ import annotations

import os
import socket
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm_ascend.v3.fast_huffman import NativeHuffmanCodec
from vllm_ascend.v3.nvfp4_codec import NVFP4CodecConfig, NVFP4ExactCodec
from vllm_ascend.v3.protocol import recv_frame, send_frame
from vllm_ascend.v3.runtime import NetworkAwarePrefixPolicy
from vllm_ascend.v3.transport import BoundaryDecision, V3TransportConfig, VerdictAdaptiveTransport


def _codec(directory: Path) -> NVFP4ExactCodec:
    # ctypes keeps loaded Windows DLLs mapped; keep the shared library outside
    # pytest's temporary directory so directory cleanup remains deterministic.
    os.environ.setdefault(
        "VLLM_ASCEND_V3_CODEC_CACHE",
        str(Path(tempfile.gettempdir()) / "vllm-ascend-v3-test-codec"),
    )
    generator = torch.Generator().manual_seed(17)
    calibration = torch.randn((8, 32), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    codec = NVFP4ExactCodec(NVFP4CodecConfig(hidden_size=32))
    codec.fit([calibration]).save(directory / "tables.npz")
    return NVFP4ExactCodec(NVFP4CodecConfig(hidden_size=32, calibration_path=str(directory / "tables.npz")))


def test_nvfp4_rows_are_exact_and_independent(tmp_path):
    codec = _codec(tmp_path)
    generator = torch.Generator().manual_seed(23)
    tensor = torch.randn((8, 32), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    anchor, residuals, prediction, _ = codec.encode_rows(tensor)
    exact, approximate = codec.decode_rows(tensor.shape, tensor.device, anchor, residuals)
    assert torch.equal(exact, tensor)
    assert not torch.equal(approximate, tensor)
    exact_prefix, _ = codec.decode_rows(tensor.shape, tensor.device, anchor, residuals[:4])
    assert torch.equal(exact_prefix[:4], tensor[:4])


def test_nvfp4_roundtrip_preserves_special_bf16_codes(tmp_path):
    codec = _codec(tmp_path)
    patterns = np.array(
        [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0xFFC2, 0x7F7F, 0xFF7F], dtype=np.uint16
    )
    codes = np.tile(patterns, 32).reshape(8, 32)
    tensor = torch.from_numpy(codes.copy()).view(torch.bfloat16)
    with np.errstate(invalid="ignore", divide="ignore"):
        anchor, residuals, _, _ = codec.encode_rows(tensor)
        decoded, _ = codec.decode_rows(tensor.shape, tensor.device, anchor, residuals)
    assert torch.equal(decoded.contiguous().view(torch.uint16), tensor.contiguous().view(torch.uint16))


def test_frame_roundtrip():
    left, right = socket.socketpair()
    try:
        send_frame(left, "ANCHOR", {"x": 1}, b"payload")
        frame = recv_frame(right)
        assert frame.kind == "ANCHOR"
        assert frame.header == {"x": 1}
        assert frame.body == b"payload"
    finally:
        left.close()
        right.close()


def test_native_huffman_context_offsets_escape_and_truncation():
    tables = [
        ["0", "10", "11" + "0" * 15],
        ["0", "1" * 16 + "0", "1" * 17],
    ]
    codec = NativeHuffmanCodec(tables, 17)
    if not codec.available:
        pytest.skip("native compiler unavailable")
    contexts = np.array([0, 1, 1], dtype=np.int64)
    symbols = np.array([2, 1, 2], dtype=np.uint16)
    exact = np.array([0x1234, 0, 0xABCD], dtype=np.uint16)
    bits = tables[0][2] + format(0x1234, "016b") + tables[1][1] + tables[1][2] + format(0xABCD, "016b")
    bits += "0" * (-len(bits) % 8)
    expected = int(bits, 2).to_bytes(len(bits) // 8, "big")
    payload = codec.encode(contexts, symbols, exact).tobytes()
    assert payload == expected
    decoded, escapes, _ = codec.decode_symbols(payload, contexts)
    assert np.array_equal(decoded, symbols)
    assert escapes[0] == 0x1234 and escapes[2] == 0xABCD
    with pytest.raises(RuntimeError):
        codec.decode_symbols(b"\xff\xff", np.array([1], dtype=np.int64))


def test_transport_full_and_prefix_repair(tmp_path):
    _codec(tmp_path)
    generator = torch.Generator().manual_seed(31)
    tensor = torch.randn((8, 32), generator=generator, dtype=torch.float32).to(torch.bfloat16)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    endpoint = f"127.0.0.1:{port}"
    receiver = VerdictAdaptiveTransport(
        "127.0.0.1",
        port,
        endpoint.split(":")[0],
        port,
        sender=False,
        calibration_path=str(tmp_path / "tables.npz"),
        adaptive=True,
        default_prefix_length=4,
    )
    sender = VerdictAdaptiveTransport(
        "127.0.0.1",
        port,
        "127.0.0.1",
        port,
        sender=True,
        calibration_path=str(tmp_path / "tables.npz"),
        adaptive=True,
        default_prefix_length=4,
    )
    received: list[torch.Tensor] = []
    ready = threading.Event()

    def receiver_main():
        def choose_prefix(approximate):
            assert approximate["hidden_states"].shape == (8, 32)
            ready.set()
            return 4

        def verdict(exact_prefix, prefix_length):
            # Rows 0..3 must be exact; demand the suffix to repair rows 4..7.
            assert torch.equal(exact_prefix["hidden_states"][:prefix_length], tensor[:prefix_length])
            return 8

        result = receiver.recv_tensor_dict(choose_prefix, verdict)
        received.append(result["hidden_states"])

    thread = threading.Thread(target=receiver_main)
    thread.start()
    sender.send_tensor_dict({"hidden_states": tensor, "epoch": 7})
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert received and torch.equal(received[0], tensor)
    assert sender.stats()["suffix_requests"] == 1
    assert receiver.stats()["suffix_requests"] == 1


def test_prefix_repair_sends_only_missing_rows(tmp_path):
    codec = _codec(tmp_path)
    tensor = torch.randn((8, 2, 32), generator=torch.Generator().manual_seed(91)).to(torch.bfloat16)
    _, payloads, _, _ = codec.encode_rows(tensor)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    kwargs = dict(
        listen_host="127.0.0.1", listen_port=port, connect_host="127.0.0.1", connect_port=port,
        calibration_path=str(tmp_path / "tables.npz"), adaptive=True, default_prefix_length=4,
    )
    sender = VerdictAdaptiveTransport(sender=True, **kwargs)
    receiver = VerdictAdaptiveTransport(sender=False, **kwargs)
    seen: list[int] = []
    received: list[torch.Tensor] = []

    def receive():
        def verdict(result, length):
            assert torch.equal(result["hidden_states"][:length], tensor[:length])
            seen.append(length)
            return {4: 6, 6: 8, 8: 8}[length]

        received.append(receiver.recv_tensor_dict(verdict_callback=verdict)["hidden_states"])

    thread = threading.Thread(target=receive)
    thread.start()
    sender.send_tensor_dict({"hidden_states": tensor})
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert seen == [4, 6, 8]
    assert torch.equal(received[0], tensor)
    assert sender.residual_wire_bytes == sum(map(len, payloads))
    assert receiver.residual_wire_bytes == sender.residual_wire_bytes


def test_adaptive_without_verdict_defaults_to_full_exact(tmp_path):
    _codec(tmp_path)
    tensor = torch.randn((8, 32), generator=torch.Generator().manual_seed(99)).to(torch.bfloat16)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    kwargs = dict(
        listen_host="127.0.0.1", listen_port=port, connect_host="127.0.0.1", connect_port=port,
        calibration_path=str(tmp_path / "tables.npz"), adaptive=True, default_prefix_length=4,
    )
    sender = VerdictAdaptiveTransport(sender=True, **kwargs)
    receiver = VerdictAdaptiveTransport(sender=False, **kwargs)
    received: list[torch.Tensor] = []
    thread = threading.Thread(target=lambda: received.append(receiver.recv_tensor_dict()["hidden_states"]))
    thread.start()
    sender.send_tensor_dict({"hidden_states": tensor})
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert torch.equal(received[0], tensor)


def test_network_policy_drives_verified_transport_without_future_verdicts(tmp_path):
    _codec(tmp_path)
    tensors = [torch.randn((8, 32), generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)
               for seed in (201, 202, 203, 204)]
    needed_rows = [2, 2, 8, 2]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    kwargs = dict(
        listen_host="127.0.0.1", listen_port=port, connect_host="127.0.0.1", connect_port=port,
        calibration_path=str(tmp_path / "tables.npz"), adaptive=True, default_prefix_length=4,
    )
    sender = VerdictAdaptiveTransport(sender=True, **kwargs)
    receiver = VerdictAdaptiveTransport(sender=False, **kwargs)
    policy = NetworkAwarePrefixPolicy(bandwidth_mbps=10, rtt_ms=0, retry_ms=0, min_history=2)
    selected: list[int] = []
    errors: list[Exception] = []

    def receive():
        try:
            for tensor, needed in zip(tensors, needed_rows):
                def verdict(result, prefix, expected=tensor, required=needed):
                    assert torch.equal(result["hidden_states"][:prefix], expected[:prefix])
                    return required

                result = receiver.recv_tensor_dict(
                    approximate_callback=lambda _: BoundaryDecision(4, {"predicted_end": 2}),
                    verdict_callback=verdict,
                    policy=policy,
                )
                assert torch.equal(result["hidden_states"][:needed], tensor[:needed])
                selected.append(receiver.policy_last_prefix)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=receive)
    thread.start()
    for tensor in tensors:
        sender.send_tensor_dict({"hidden_states": tensor})
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert not errors
    assert selected == [8, 8, 4, 4]
    assert sender.suffix_requests == 1
    assert list(policy.history) == needed_rows


def test_pending_boundary_requires_repair_before_finish(tmp_path):
    _codec(tmp_path)
    tensor = torch.randn((8, 32), generator=torch.Generator().manual_seed(301)).to(torch.bfloat16)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    kwargs = dict(
        listen_host="127.0.0.1", listen_port=port, connect_host="127.0.0.1", connect_port=port,
        calibration_path=str(tmp_path / "tables.npz"), adaptive=True, default_prefix_length=4,
    )
    sender = VerdictAdaptiveTransport(sender=True, **kwargs)
    receiver = VerdictAdaptiveTransport(sender=False, **kwargs)
    errors: list[Exception] = []

    def receive():
        try:
            boundary = receiver.begin_boundary()
            with pytest.raises(ValueError):
                boundary.finish(0)
            partial = boundary.request(4)["hidden_states"]
            assert torch.equal(partial[:4], tensor[:4])
            assert torch.count_nonzero(partial[4:]) == 0
            with pytest.raises(ValueError):
                boundary.finish(5)
            full = boundary.request(8)["hidden_states"]
            assert torch.equal(full, tensor)
            boundary.finish(8)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=receive)
    thread.start()
    sender.send_tensor_dict({"hidden_states": tensor})
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert not errors
    assert sender.suffix_requests == receiver.suffix_requests == 1


def test_split_request_allows_work_before_residual_receive(tmp_path, monkeypatch):
    _codec(tmp_path)
    tensor = torch.randn((8, 32), generator=torch.Generator().manual_seed(302)).to(torch.bfloat16)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    kwargs = dict(
        listen_host="127.0.0.1", listen_port=port, connect_host="127.0.0.1", connect_port=port,
        calibration_path=str(tmp_path / "tables.npz"), adaptive=True, default_prefix_length=4,
    )
    sender = VerdictAdaptiveTransport(sender=True, provisional_feedback=True, **kwargs)
    receiver = VerdictAdaptiveTransport(sender=False, **kwargs)
    sender_received_request = threading.Event()
    proposal_received = threading.Event()
    residual_send_started = threading.Event()
    release_residual_send = threading.Event()
    proposals: list[dict] = []

    def receive_proposal(proposal):
        proposals.append(proposal)
        proposal_received.set()

    sender.provisional_draft_callback = receive_proposal
    encode = sender._encode_residual_rows
    send_residual = sender._send_residual_frame

    def record_request(boundary, rows):
        sender_received_request.set()
        return encode(boundary, rows)

    def delayed_send(sock, header, body):
        residual_send_started.set()
        if not release_residual_send.wait(timeout=20):
            raise TimeoutError("residual send was not released")
        send_residual(sock, header, body)

    monkeypatch.setattr(sender, "_encode_residual_rows", record_request)
    monkeypatch.setattr(sender, "_send_residual_frame", delayed_send)
    errors: list[Exception] = []

    def send():
        try:
            sender.send_tensor_dict({"hidden_states": tensor})
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=send)
    thread.start()
    boundary = receiver.begin_boundary()
    try:
        boundary.start_request(4)
        with pytest.raises(ValueError):
            boundary.start_request(8)
        with pytest.raises(ValueError):
            boundary.finish(0)
        assert sender_received_request.wait(timeout=20)
        assert residual_send_started.wait(timeout=20)
        with pytest.raises(ValueError):
            boundary.send_provisional_draft(8, 42, (1, 2, 3, 4, 5, 6, 7))
        boundary.send_provisional_draft(2, 42, (1, 2, 3, 4, 5, 6, 7))
        with pytest.raises(ValueError):
            boundary.send_provisional_draft(2, 42, (1, 2, 3, 4, 5, 6, 7))
        assert proposal_received.wait(timeout=20)
        assert thread.is_alive()
        assert not release_residual_send.is_set()
        release_residual_send.set()
        partial = boundary.complete_request()["hidden_states"]
        assert torch.equal(partial[:4], tensor[:4])
        with pytest.raises(ValueError):
            boundary.complete_request()
        boundary.start_request(8)
        full = boundary.complete_request()["hidden_states"]
        assert torch.equal(full, tensor)
        boundary.finish(8)
    finally:
        release_residual_send.set()
        if not boundary.closed:
            boundary.abort()
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert not errors
    assert sender.suffix_requests == receiver.suffix_requests == 1
    assert len(proposals) == 1
    assert proposals[0]["accepted_drafts"] == 2
    assert proposals[0]["root"] == 42
    assert proposals[0]["drafts"] == [1, 2, 3, 4, 5, 6, 7]
    assert sender.last_provisional_draft == proposals[0]


def test_duplex_residual_send_failure_unblocks_receiver(tmp_path, monkeypatch):
    _codec(tmp_path)
    tensor = torch.randn((8, 32), generator=torch.Generator().manual_seed(303)).to(torch.bfloat16)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    kwargs = dict(
        listen_host="127.0.0.1", listen_port=port, connect_host="127.0.0.1", connect_port=port,
        calibration_path=str(tmp_path / "tables.npz"), adaptive=True,
    )
    sender = VerdictAdaptiveTransport(sender=True, provisional_feedback=True, **kwargs)
    receiver = VerdictAdaptiveTransport(sender=False, **kwargs)
    errors: list[Exception] = []

    def fail_send(sock, header, body):
        raise OSError("injected residual send failure")

    def send():
        try:
            sender.send_tensor_dict({"hidden_states": tensor})
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(sender, "_send_residual_frame", fail_send)
    thread = threading.Thread(target=send)
    thread.start()
    boundary = receiver.begin_boundary()
    boundary.start_request(4)
    with pytest.raises(OSError):
        boundary.complete_request()
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert errors and boundary.closed


@pytest.mark.parametrize(
    "change",
    [
        {"boundary_id": 12},
        {"accepted_drafts": 8},
        {"root": -1},
        {"drafts": [1, 2]},
        {"drafts": [1, 2, 3, 4, 5, 6, -1]},
    ],
)
def test_provisional_draft_rejects_stale_or_invalid_metadata(change):
    proposal = {"version": 1, "boundary_id": 11, "accepted_drafts": 2, "root": 42, "drafts": list(range(7))}
    proposal.update(change)
    with pytest.raises(ValueError):
        VerdictAdaptiveTransport._validate_provisional_draft(proposal, 8, 11)


def test_background_frames_remain_ordered(tmp_path, monkeypatch):
    _codec(tmp_path)
    tensors = [torch.randn((8, 32), generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)
               for seed in (101, 102)]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    kwargs = dict(
        listen_host="127.0.0.1", listen_port=port, connect_host="127.0.0.1", connect_port=port,
        calibration_path=str(tmp_path / "tables.npz"), adaptive=False,
    )
    sender = VerdictAdaptiveTransport(sender=True, **kwargs)
    receiver = VerdictAdaptiveTransport(sender=False, **kwargs)
    send_residual = sender._send_full_residual_background

    def delayed_send(entries, semantic):
        time.sleep(0.05)
        send_residual(entries, semantic)

    monkeypatch.setattr(sender, "_send_full_residual_background", delayed_send)
    received: list[torch.Tensor] = []

    def receive():
        for _ in tensors:
            received.append(receiver.recv_tensor_dict()["hidden_states"])

    thread = threading.Thread(target=receive)
    thread.start()
    for tensor in tensors:
        sender.send_tensor_dict({"hidden_states": tensor})
    sender.flush()
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert len(received) == len(tensors)
    assert all(torch.equal(left, right) for left, right in zip(received, tensors))


def test_background_send_error_is_propagated(tmp_path, monkeypatch):
    _codec(tmp_path)
    sender = VerdictAdaptiveTransport(
        "127.0.0.1", 1, "127.0.0.1", 1, sender=True, calibration_path=str(tmp_path / "tables.npz")
    )
    left, right = socket.socketpair()
    sender._socket = left

    def fail(entries, semantic):
        raise ValueError("test failure")

    monkeypatch.setattr(sender, "_send_full_residual_background", fail)
    try:
        sender.send_tensor_dict({"epoch": 1})
        with pytest.raises(RuntimeError, match="background residual sender failed"):
            sender.flush()
        assert recv_frame(right).kind == "ANCHOR"
        with pytest.raises(ConnectionError):
            recv_frame(right)
    finally:
        left.close()
        right.close()


def test_verdict_callback_error_unblocks_sender(tmp_path):
    _codec(tmp_path)
    tensor = torch.randn((8, 32), generator=torch.Generator().manual_seed(103)).to(torch.bfloat16)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    kwargs = dict(
        listen_host="127.0.0.1", listen_port=port, connect_host="127.0.0.1", connect_port=port,
        calibration_path=str(tmp_path / "tables.npz"), adaptive=True, default_prefix_length=4,
    )
    sender = VerdictAdaptiveTransport(sender=True, **kwargs)
    receiver = VerdictAdaptiveTransport(sender=False, **kwargs)
    errors: list[Exception] = []

    def receive():
        def fail(result, prefix):
            raise ValueError("verdict failed")

        try:
            receiver.recv_tensor_dict(verdict_callback=fail)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=receive)
    thread.start()
    with pytest.raises((ConnectionError, OSError)):
        sender.send_tensor_dict({"hidden_states": tensor})
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], ValueError)


def test_additional_config_is_spawn_safe():
    config = V3TransportConfig.from_vllm_config(
        {
            "v3_verdict_adaptive": {
                "enabled": True,
                "calibration": "/tmp/tables.npz",
                "listen": "127.0.0.1:2",
                "connect": "127.0.0.1:2",
                "prefix_repair": True,
                "prefix_length": 6,
                "experimental_dv_mtp_shadow": True,
                "experimental_fused_vv_dv": True,
            }
        }
    )
    assert config.enabled
    assert config.calibration_path == "/tmp/tables.npz"
    assert config.adaptive
    assert config.prefix_length == 6
    assert config.provisional_feedback
    assert config.fused_vv_dv
    assert not V3TransportConfig.from_vllm_config({}).provisional_feedback
    assert not V3TransportConfig.from_vllm_config({}).fused_vv_dv


def test_transport_full_residual_is_exact(tmp_path):
    _codec(tmp_path)
    generator = torch.Generator().manual_seed(41)
    tensor = torch.randn((8, 32), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    receiver = VerdictAdaptiveTransport(
        "127.0.0.1",
        port,
        "127.0.0.1",
        port,
        sender=False,
        calibration_path=str(tmp_path / "tables.npz"),
        adaptive=False,
    )
    sender = VerdictAdaptiveTransport(
        "127.0.0.1",
        port,
        "127.0.0.1",
        port,
        sender=True,
        calibration_path=str(tmp_path / "tables.npz"),
        adaptive=False,
    )
    received: list[torch.Tensor] = []

    def receiver_main():
        received.append(receiver.recv_tensor_dict()["hidden_states"])

    thread = threading.Thread(target=receiver_main)
    thread.start()
    sender.send_tensor_dict({"hidden_states": tensor})
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert torch.equal(received[0], tensor)
    assert sender.stats()["suffix_requests"] == 0


def test_transport_raw_bf16_mode(tmp_path):
    _codec(tmp_path)
    generator = torch.Generator().manual_seed(53)
    tensor = torch.randn((8, 32), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    receiver = VerdictAdaptiveTransport(
        "127.0.0.1", port, "127.0.0.1", port, sender=False,
        calibration_path=str(tmp_path / "tables.npz"), codec_mode="raw",
    )
    sender = VerdictAdaptiveTransport(
        "127.0.0.1", port, "127.0.0.1", port, sender=True,
        calibration_path=str(tmp_path / "tables.npz"), codec_mode="raw",
    )
    received: list[torch.Tensor] = []

    def receiver_main():
        received.append(receiver.recv_tensor_dict()["hidden_states"])

    thread = threading.Thread(target=receiver_main)
    thread.start()
    sender.send_tensor_dict({"hidden_states": tensor})
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert torch.equal(received[0], tensor)

def test_greedy_frontier_limits_fused_first_block():
    from vllm_ascend.v3.early_draft import greedy_frontier

    logits = torch.zeros((16, 8), dtype=torch.float32)
    drafts = torch.arange(1, 16, dtype=torch.int32)
    # Only the first seven comparisons match; the fused suffix is ignored.
    for row, token in enumerate(drafts[:7]):
        logits[row, token] = 8.0
    accepted, root = greedy_frontier(logits, drafts, limit=7)
    assert accepted == 7
    assert int(root) == 0


def test_fused_boundary_keeps_approximate_suffix(tmp_path):
    _codec(tmp_path)
    tensor = torch.randn((16, 32), generator=torch.Generator().manual_seed(401)).to(torch.bfloat16)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    kwargs = dict(
        listen_host="127.0.0.1", listen_port=port, connect_host="127.0.0.1", connect_port=port,
        calibration_path=str(tmp_path / "tables.npz"), adaptive=True, default_prefix_length=8,
    )
    sender = VerdictAdaptiveTransport(sender=True, provisional_feedback=True, **kwargs)
    receiver = VerdictAdaptiveTransport(sender=False, **kwargs)
    errors: list[Exception] = []

    def receive():
        try:
            boundary = receiver.begin_boundary()
            boundary.start_request(8)
            boundary.send_provisional_draft(7, 42, tuple(range(1, 8)))
            result = boundary.complete_request(keep_approximate_suffix=True)["hidden_states"]
            assert torch.equal(result[:8], tensor[:8])
            assert torch.count_nonzero(result[8:]) > 0
            boundary.finish(8)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=receive)
    thread.start()
    sender.send_tensor_dict({"hidden_states": tensor})
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert not errors
    assert sender.stats()["provisional_drafts_received"] == 1
