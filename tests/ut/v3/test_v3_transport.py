from __future__ import annotations

import os
import socket
import tempfile
import threading
from pathlib import Path

import torch

from vllm_ascend.v3.nvfp4_codec import NVFP4CodecConfig, NVFP4ExactCodec
from vllm_ascend.v3.protocol import recv_frame, send_frame
from vllm_ascend.v3.transport import V3TransportConfig, VerdictAdaptiveTransport


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
            }
        }
    )
    assert config.enabled
    assert config.calibration_path == "/tmp/tables.npz"
    assert config.adaptive
    assert config.prefix_length == 6


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
