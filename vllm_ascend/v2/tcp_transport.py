from __future__ import annotations

import io
import os
import pickle
import socket
import time
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import logger

from .semantic_codec import SemanticBoundaryCodec, SemanticCodecConfig


@dataclass
class _SemanticEntry:
    shape: tuple[int, ...]
    dtype: str
    device: str
    anchor_bytes: int
    residual_bytes: int
    escape_rate: float


@dataclass
class _RawEntry:
    shape: tuple[int, ...]
    dtype: str
    device: str
    size: int


class SemanticPPTransport:
    """TCP transport for one experimental PP2 V2 activation boundary.

    This path is deliberately separate from HCCL.  It sends the real anchor and
    exact residual byte counts, and reconstructs BF16 before the next PP stage.
    It does not yet publish the anchor before waiting for the residual; that
    optimistic DV scheduling step is intentionally left for the V2 scheduler
    integration.
    """

    def __init__(self, listen_host: str, listen_port: int, connect_host: str, connect_port: int, *, sender: bool, calibration_path: str):
        self.listen_host, self.listen_port = listen_host, listen_port
        self.connect_host, self.connect_port = connect_host, connect_port
        self.sender = sender
        self.codec = SemanticBoundaryCodec(
            SemanticCodecConfig(calibration_path=calibration_path)
        )
        self._socket: socket.socket | None = None
        self.wire_bytes = 0
        self.bf16_reference_bytes = 0
        self.messages = 0
        self.encode_seconds = 0.0
        self.decode_seconds = 0.0

    @classmethod
    def from_env(cls, *, sender: bool) -> "SemanticPPTransport | None":
        if os.environ.get("VLLM_ASCEND_V2_SEMANTIC_PP", "0") != "1":
            return None
        listen = os.environ.get("VLLM_ASCEND_V2_LISTEN", "127.0.0.1:29600")
        connect = os.environ.get("VLLM_ASCEND_V2_CONNECT", "127.0.0.1:29600")
        calibration = os.environ.get("VLLM_ASCEND_V2_CALIBRATION", "")
        if not calibration:
            raise ValueError("VLLM_ASCEND_V2_SEMANTIC_PP=1 requires VLLM_ASCEND_V2_CALIBRATION")
        listen_host, listen_port_text = listen.rsplit(":", 1)
        connect_host, connect_port_text = connect.rsplit(":", 1)
        logger.info(
            "V2 semantic PP transport enabled: role=%s listen=%s connect=%s calibration=%s",
            "sender" if sender else "receiver", listen, connect, calibration,
        )
        return cls(
            listen_host,
            int(listen_port_text),
            connect_host,
            int(connect_port_text),
            sender=sender,
            calibration_path=calibration,
        )

    def _ensure_socket(self) -> socket.socket:
        if self._socket is not None:
            return self._socket
        if self.sender:
            deadline = time.monotonic() + 120.0
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                try:
                    sock = socket.create_connection((self.connect_host, self.connect_port), timeout=2)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self._socket = sock
                    return sock
                except OSError as exc:
                    last_error = exc
                    time.sleep(0.25)
            raise ConnectionError(f"V2 semantic sender cannot connect to {self.connect_host}:{self.connect_port}: {last_error}")
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.listen_host, self.listen_port))
        server.listen(1)
        logger.info("V2 semantic receiver waiting at %s:%s", self.listen_host, self.listen_port)
        sock, address = server.accept()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server.close()
        self._socket = sock
        logger.info("V2 semantic receiver connected from %s:%s", *address)
        return sock

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = sock.recv(size - len(chunks))
            if not chunk:
                raise ConnectionError("V2 semantic peer closed")
            chunks.extend(chunk)
        return bytes(chunks)

    @staticmethod
    def _torch_save(value: Any) -> bytes:
        buffer = io.BytesIO()
        torch.save(value, buffer)
        return buffer.getvalue()

    @staticmethod
    def _torch_load(payload: bytes) -> Any:
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)

    def send_tensor_dict(self, tensor_dict: dict[str, Any]) -> None:
        started = time.perf_counter()
        entries: list[tuple[str, str, Any]] = []
        payloads: list[bytes] = []
        semantic_bytes = 0
        reference_bytes = 0
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16 and value.shape[-1] == self.codec.config.hidden_size:
                anchor_payload, residual_payload, escape_rate = self.codec.encode(key, value)
                entry = _SemanticEntry(tuple(value.shape), str(value.dtype).removeprefix("torch."), str(value.device), len(anchor_payload), len(residual_payload), escape_rate)
                entries.append((key, "semantic", entry))
                payloads.extend((anchor_payload, residual_payload))
                semantic_bytes += len(anchor_payload) + len(residual_payload)
                reference_bytes += value.numel() * 2
            elif isinstance(value, torch.Tensor):
                payload = self._torch_save(value.detach().cpu())
                entry = _RawEntry(tuple(value.shape), str(value.dtype).removeprefix("torch."), str(value.device), len(payload))
                entries.append((key, "raw", entry))
                payloads.append(payload)
                semantic_bytes += len(payload)
                reference_bytes += value.numel() * value.element_size()
            else:
                entries.append((key, "object", value))
        header = pickle.dumps({"version": 1, "entries": entries}, protocol=pickle.HIGHEST_PROTOCOL)
        body = b"".join(payloads)
        message = len(header).to_bytes(8, "little") + len(body).to_bytes(8, "little") + header + body
        sock = self._ensure_socket()
        sock.sendall(message)
        self.messages += 1
        self.wire_bytes += semantic_bytes
        self.bf16_reference_bytes += reference_bytes
        self.encode_seconds += time.perf_counter() - started

    def recv_tensor_dict(self) -> dict[str, Any]:
        started = time.perf_counter()
        sock = self._ensure_socket()
        header_size = int.from_bytes(self._recv_exact(sock, 8), "little")
        body_size = int.from_bytes(self._recv_exact(sock, 8), "little")
        header = pickle.loads(self._recv_exact(sock, header_size))
        body = self._recv_exact(sock, body_size)
        if header.get("version") != 1:
            raise ValueError("unsupported V2 semantic transport header")
        result: dict[str, Any] = {}
        offset = 0
        for key, kind, entry in header["entries"]:
            if kind == "object":
                result[key] = entry
                continue
            if kind == "semantic":
                anchor = body[offset : offset + entry.anchor_bytes]
                offset += entry.anchor_bytes
                residual = body[offset : offset + entry.residual_bytes]
                offset += entry.residual_bytes
                result[key] = self.codec.decode(
                    torch.Size(entry.shape), getattr(torch, entry.dtype), torch.device(entry.device), anchor, residual
                )
                self.wire_bytes += entry.anchor_bytes + entry.residual_bytes
                self.bf16_reference_bytes += int(torch.tensor(entry.shape).prod().item()) * 2
            else:
                payload = body[offset : offset + entry.size]
                offset += entry.size
                tensor = self._torch_load(payload)
                result[key] = tensor.to(device=torch.device(entry.device), dtype=getattr(torch, entry.dtype))
                self.wire_bytes += entry.size
                self.bf16_reference_bytes += tensor.numel() * tensor.element_size()
        self.messages += 1
        self.decode_seconds += time.perf_counter() - started
        return result

    def stats(self) -> dict[str, float | int]:
        return {
            "messages": self.messages,
            "wire_bytes": self.wire_bytes,
            "bf16_reference_bytes": self.bf16_reference_bytes,
            "reduction": 1.0 - self.wire_bytes / self.bf16_reference_bytes if self.bf16_reference_bytes else 0.0,
            "encode_seconds": self.encode_seconds,
            "decode_seconds": self.decode_seconds,
        }
