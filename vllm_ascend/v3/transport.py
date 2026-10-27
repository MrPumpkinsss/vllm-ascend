from __future__ import annotations

# ruff: noqa: E402

"""Two-wave V3 PP2 transport: anchor/DV first, exact residual/VV second.

The transport deliberately exposes anchor arrival as a separate event.  A
receiver-side callback runs after approximate decode but before residual wait,
which is the hook needed by an engine scheduler for approximate DV work.  The
stock vLLM worker currently uses the safe full-residual policy; adaptive
Prefix-Repair is exercised by the unit tests until scheduler state ownership is
wired into the target model runner.
"""

import io
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .nvfp4_codec import NVFP4CodecConfig, NVFP4ExactCodec
from .protocol import recv_frame, send_frame

LOGGER = logging.getLogger(__name__)


@dataclass
class V3TransportConfig:
    enabled: bool = False
    listen: str = "127.0.0.1:29600"
    connect: str = "127.0.0.1:29600"
    calibration_path: str = ""
    adaptive: bool = False
    prefix_length: int = 8
    runtime_trace: str = ""

    @classmethod
    def from_env(cls) -> V3TransportConfig:
        return cls(
            enabled=os.environ.get("VLLM_ASCEND_V3_VERDICT_ADAPTIVE", "0") == "1",
            listen=os.environ.get("VLLM_ASCEND_V3_LISTEN", "127.0.0.1:29600"),
            connect=os.environ.get("VLLM_ASCEND_V3_CONNECT", "127.0.0.1:29600"),
            calibration_path=os.environ.get("VLLM_ASCEND_V3_CALIBRATION", ""),
            adaptive=os.environ.get("VLLM_ASCEND_V3_PREFIX_REPAIR", "0") == "1",
            prefix_length=int(os.environ.get("VLLM_ASCEND_V3_PREFIX_LENGTH", "8")),
        )

    @classmethod
    def from_vllm_config(cls, additional_config: dict[str, Any] | None) -> V3TransportConfig:
        """Read the spawn-safe vLLM `--additional-config` representation.

        Environment variables are convenient for single-process probes, but
        vLLM multiprocess workers do not necessarily preserve every parent
        VLLM_* variable.  The JSON config travels inside VllmConfig and is the
        serving path.
        """
        config = cls.from_env()
        raw = (additional_config or {}).get("v3_verdict_adaptive")
        if not isinstance(raw, dict):
            return config
        if "enabled" in raw:
            config.enabled = bool(raw["enabled"])
        config.listen = str(raw.get("listen", config.listen))
        config.connect = str(raw.get("connect", config.connect))
        config.calibration_path = str(raw.get("calibration", config.calibration_path))
        config.adaptive = bool(raw.get("prefix_repair", config.adaptive))
        config.prefix_length = int(raw.get("prefix_length", config.prefix_length))
        config.runtime_trace = str(raw.get("runtime_trace", config.runtime_trace))
        return config


@dataclass
class _SemanticBoundary:
    key: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    anchor: bytes
    exact: torch.Tensor
    prediction: torch.Tensor
    anchor_code: Any
    residuals: list[bytes] = field(default_factory=list)
    encoded_rows: int = 0
    escape_rate: float = 0.0


@dataclass
class _AnchorEntry:
    shape: tuple[int, ...]
    dtype: str
    device: str
    anchor_bytes: int
    rows: int
    escape_rate: float


@dataclass
class _RawEntry:
    shape: tuple[int, ...]
    dtype: str
    device: str
    size: int


@dataclass
class BoundaryDecision:
    """Receiver decision returned by an approximate-DV callback."""

    prefix_length: int
    metadata: dict[str, Any] = field(default_factory=dict)


class VerdictAdaptiveTransport:
    """TCP transport for one PP2 V3 activation boundary."""

    def __init__(
        self,
        listen_host: str,
        listen_port: int,
        connect_host: str,
        connect_port: int,
        *,
        sender: bool,
        calibration_path: str,
        adaptive: bool = False,
        default_prefix_length: int = 8,
    ):
        self.listen_host, self.listen_port = listen_host, listen_port
        self.connect_host, self.connect_port = connect_host, connect_port
        self.sender = sender
        self.adaptive = adaptive
        self.default_prefix_length = default_prefix_length
        if not calibration_path:
            raise ValueError("V3 transport requires VLLM_ASCEND_V3_CALIBRATION")
        hidden_size = NVFP4ExactCodec.read_hidden_size(calibration_path)
        self.codec: NVFP4ExactCodec = NVFP4ExactCodec(
            NVFP4CodecConfig(hidden_size=hidden_size, calibration_path=calibration_path)
        )
        self._socket: socket.socket | None = None
        self.boundaries = 0
        self.anchor_wire_bytes = 0
        self.residual_wire_bytes = 0
        self.bf16_reference_bytes = 0
        self.suffix_requests = 0
        self.encode_seconds = 0.0
        self.decode_seconds = 0.0
        self.approximate_seconds = 0.0

    @classmethod
    def from_config(cls, config: V3TransportConfig, *, sender: bool) -> VerdictAdaptiveTransport | None:
        if not config.enabled:
            return None
        listen_host, listen_port = config.listen.rsplit(":", 1)
        connect_host, connect_port = config.connect.rsplit(":", 1)
        LOGGER.info(
            "V3 transport enabled: role=%s listen=%s connect=%s adaptive=%s calibration=%s",
            "sender" if sender else "receiver",
            config.listen,
            config.connect,
            config.adaptive,
            config.calibration_path,
        )
        return cls(
            listen_host,
            int(listen_port),
            connect_host,
            int(connect_port),
            sender=sender,
            calibration_path=config.calibration_path,
            adaptive=config.adaptive,
            default_prefix_length=config.prefix_length,
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
            raise ConnectionError(f"V3 sender cannot connect to {self.connect_host}:{self.connect_port}: {last_error}")
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.listen_host, self.listen_port))
        server.listen(1)
        LOGGER.info("V3 receiver waiting at %s:%s", self.listen_host, self.listen_port)
        sock, address = server.accept()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server.close()
        self._socket = sock
        LOGGER.info("V3 receiver connected from %s:%s", *address)
        return sock

    @staticmethod
    def _torch_save(value: Any) -> bytes:
        buffer = io.BytesIO()
        torch.save(value, buffer)
        return buffer.getvalue()

    @staticmethod
    def _torch_load(payload: bytes) -> Any:
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)

    def _is_semantic(self, value: Any) -> bool:
        return (
            isinstance(value, torch.Tensor)
            and value.dtype == torch.bfloat16
            and value.shape[-1] == self.codec.config.hidden_size
        )

    def _prepare_boundary(self, tensor_dict: dict[str, Any]):
        """Encode only anchor values before the first wire wave.

        Residual rows are intentionally not encoded yet: stage A sends the
        anchor immediately and encodes requested exact rows while stage B can
        consume the approximate operand.
        """
        entries: list[tuple[str, str, Any]] = []
        semantic: dict[str, _SemanticBoundary] = {}
        anchor_payloads: list[bytes] = []
        anchor_bytes = 0
        reference_bytes = 0
        for key, value in tensor_dict.items():
            if self._is_semantic(value):
                exact = value.detach()
                anchor, anchor_code = self.codec.encode_anchor(exact)
                prediction, decoded_code = self.codec.decode_anchor(anchor, exact.shape, exact.device)
                boundary = _SemanticBoundary(
                    key,
                    tuple(exact.shape),
                    str(exact.dtype).removeprefix("torch."),
                    str(exact.device),
                    anchor,
                    exact.cpu(),
                    prediction.cpu(),
                    decoded_code,
                )
                semantic[key] = boundary
                entry = _AnchorEntry(boundary.shape, boundary.dtype, boundary.device, len(anchor), exact.shape[0], 0.0)
                entries.append((key, "semantic", entry))
                anchor_payloads.append(anchor)
                anchor_bytes += len(anchor)
                reference_bytes += exact.numel() * 2
            elif isinstance(value, torch.Tensor):
                payload = self._torch_save(value.detach().cpu())
                entry = _RawEntry(
                    tuple(value.shape), str(value.dtype).removeprefix("torch."), str(value.device), len(payload)
                )
                entries.append((key, "raw", entry))
                anchor_payloads.append(payload)
                anchor_bytes += len(payload)
                reference_bytes += value.numel() * value.element_size()
            else:
                entries.append((key, "object", value))
        return entries, semantic, b"".join(anchor_payloads), anchor_bytes, reference_bytes

    def _encode_residual_rows(self, boundary: _SemanticBoundary, rows: int) -> list[bytes]:
        rows = min(max(rows, boundary.encoded_rows), int(boundary.exact.shape[0]))
        while boundary.encoded_rows < rows:
            row = boundary.encoded_rows
            prediction = boundary.prediction[row : row + 1].to(boundary.exact.device)
            exact = boundary.exact[row : row + 1].to(boundary.exact.device)
            code = boundary.anchor_code[row : row + 1]
            payload, escape_rate = self.codec.encode_residual(prediction, exact, code)
            boundary.residuals.append(payload)
            boundary.encoded_rows += 1
            boundary.escape_rate = escape_rate
        return boundary.residuals[:rows]

    def send_tensor_dict(self, tensor_dict: dict[str, Any]) -> None:
        started = time.perf_counter()
        entries, semantic, anchor_body, anchor_bytes, reference_bytes = self._prepare_boundary(tensor_dict)
        sock = self._ensure_socket()
        send_frame(sock, "ANCHOR", {"version": 1, "entries": entries, "adaptive": self.adaptive}, anchor_body)
        self.boundaries += 1
        self.anchor_wire_bytes += anchor_bytes
        self.bf16_reference_bytes += reference_bytes

        if self.adaptive:
            request = recv_frame(sock)
            if request.kind != "RESIDUAL_REQUEST":
                raise RuntimeError(f"V3 sender expected RESIDUAL_REQUEST, got {request.kind}")
            requested = int(request.header["prefix_length"])
        else:
            requested = max((entry.rows for _, kind, entry in entries if kind == "semantic"), default=0)
            if requested == 0:
                requested = 0

        while True:
            selected: list[tuple[str, _SemanticBoundary, list[bytes]]] = []
            residual_payloads: list[bytes] = []
            residual_bytes = 0
            for key, kind, entry in entries:
                if kind != "semantic":
                    continue
                boundary = semantic[key]
                end = min(max(requested, 0), int(boundary.exact.shape[0]))
                payloads = self._encode_residual_rows(boundary, end)
                selected.append((key, boundary, payloads))
                residual_payloads.extend(payloads)
                residual_bytes += sum(map(len, payloads))
            residual_entries = [
                (key, {"start_row": 0, "rows": len(payloads), "lengths": [len(payload) for payload in payloads]})
                for key, _, payloads in selected
            ]
            send_frame(sock, "RESIDUAL", {"version": 1, "entries": residual_entries}, b"".join(residual_payloads))
            self.residual_wire_bytes += residual_bytes
            if not self.adaptive:
                break
            verdict = recv_frame(sock)
            if verdict.kind == "BOUNDARY_DONE":
                break
            if verdict.kind != "SUFFIX_REQUEST":
                raise RuntimeError(f"V3 sender expected SUFFIX_REQUEST/DONE, got {verdict.kind}")
            requested = int(verdict.header["prefix_length"])
            self.suffix_requests += 1

        self.encode_seconds += time.perf_counter() - started

    def recv_anchor(self) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        sock = self._ensure_socket()
        frame = recv_frame(sock)
        if frame.kind != "ANCHOR":
            raise RuntimeError(f"V3 receiver expected ANCHOR, got {frame.kind}")
        if frame.header.get("version") != 1:
            raise RuntimeError("unsupported V3 boundary version")
        result: dict[str, Any] = {}
        semantic_entries: dict[str, _AnchorEntry] = {}
        anchor_codes: dict[str, Any] = {}
        anchor_bytes = 0
        reference_bytes = 0
        offset = 0
        for key, kind, entry in frame.header["entries"]:
            if kind == "object":
                result[key] = entry
                continue
            if kind == "semantic":
                payload = frame.body[offset : offset + entry.anchor_bytes]
                offset += entry.anchor_bytes
                prediction, anchor_code = self.codec.decode_anchor(
                    payload, torch.Size(entry.shape), torch.device(entry.device)
                )
                result[key] = prediction
                semantic_entries[key] = entry
                anchor_codes[key] = anchor_code
                anchor_bytes += entry.anchor_bytes
                reference_bytes += int(torch.tensor(entry.shape).prod().item()) * 2
            else:
                payload = frame.body[offset : offset + entry.size]
                offset += entry.size
                tensor = self._torch_load(payload)
                result[key] = tensor.to(device=torch.device(entry.device), dtype=getattr(torch, entry.dtype))
                anchor_bytes += entry.size
                reference_bytes += tensor.numel() * tensor.element_size()
        if offset != len(frame.body):
            raise RuntimeError("trailing bytes in V3 anchor frame")
        self.boundaries += 1
        self.anchor_wire_bytes += anchor_bytes
        self.bf16_reference_bytes += reference_bytes
        metadata = {
            "adaptive": bool(frame.header.get("adaptive", False)),
            "entries": semantic_entries,
            "anchor_codes": anchor_codes,
        }
        self.decode_seconds += time.perf_counter() - started
        return result, metadata

    def _request_residual(
        self, prefix_length: int, metadata: dict[str, Any], *, kind: str = "RESIDUAL_REQUEST"
    ) -> tuple[dict[str, Any], bytes]:
        sock = self._ensure_socket()
        lengths: dict[str, int] = {key: min(prefix_length, entry.rows) for key, entry in metadata["entries"].items()}
        send_frame(sock, kind, {"version": 1, "prefix_length": prefix_length, "lengths": lengths})
        frame = recv_frame(sock)
        if frame.kind != "RESIDUAL":
            raise RuntimeError(f"V3 receiver expected RESIDUAL, got {frame.kind}")
        return frame.header, frame.body

    def _decode_residual(
        self,
        approximate: dict[str, torch.Tensor],
        metadata: dict[str, Any],
        residual_header: dict[str, Any],
        body: bytes,
    ) -> dict[str, Any]:
        result = dict(approximate)
        residual_bytes = 0
        offset = 0
        started = time.perf_counter()
        for key, entry in dict(residual_header["entries"]).items():
            anchor_code = metadata["anchor_codes"][key]
            prediction = approximate[key]
            rows = int(entry["rows"])
            lengths = entry["lengths"]
            payloads: list[bytes] = []
            for length in lengths:
                payload = body[offset : offset + int(length)]
                offset += int(length)
                payloads.append(payload)
            if len(payloads) != rows:
                raise RuntimeError("V3 residual row-count mismatch")
            if rows:
                # The wire residual is row-independent; decode into a copy without mutating approximate DV state.
                tensor = prediction.clone()
                for row, payload in enumerate(payloads):
                    tensor[row : row + 1] = self.codec.decode_residual(
                        prediction[row : row + 1],
                        anchor_code[row : row + 1],
                        payload,
                        tensor[row : row + 1].shape,
                        prediction.device,
                    )
            else:
                tensor = prediction
            result[key] = tensor
            residual_bytes += sum(map(len, payloads))
        if offset != len(body):
            raise RuntimeError("trailing bytes in V3 residual frame")
        self.residual_wire_bytes += residual_bytes
        self.decode_seconds += time.perf_counter() - started
        return result

    def recv_tensor_dict(
        self,
        approximate_callback: Callable[[dict[str, Any]], BoundaryDecision | int | None] | None = None,
        verdict_callback: Callable[[dict[str, Any], int], int] | None = None,
    ) -> dict[str, Any]:
        approximate, metadata = self.recv_anchor()
        approximate_started = time.perf_counter()
        decision: BoundaryDecision | int | None = None
        if approximate_callback is not None:
            decision = approximate_callback(approximate)
        self.approximate_seconds += time.perf_counter() - approximate_started

        if metadata["adaptive"]:
            if isinstance(decision, BoundaryDecision):
                prefix_length = decision.prefix_length
            elif isinstance(decision, int):
                prefix_length = decision
            else:
                prefix_length = self.default_prefix_length
        else:
            first = next(iter(metadata["entries"].values()), None)
            prefix_length = first.rows if first is not None else 0

        if metadata["adaptive"]:
            residual_header, residual_body = self._request_residual(prefix_length, metadata)
        else:
            # A non-adaptive sender streams residual immediately; do not inject a
            # control request that it will never consume.
            residual_header, residual_body = self.recv_residual_frame()
        result = self._decode_residual(approximate, metadata, residual_header, residual_body)
        sock = self._ensure_socket()
        if metadata["adaptive"] and verdict_callback is not None:
            consumed_end = int(verdict_callback(result, prefix_length))
            if consumed_end > prefix_length:
                self.suffix_requests += 1
                residual_header, residual_body = self._request_residual(consumed_end, metadata, kind="SUFFIX_REQUEST")
                result = self._decode_residual(approximate, metadata, residual_header, residual_body)
        if metadata["adaptive"]:
            send_frame(sock, "BOUNDARY_DONE", {"version": 1})
        return result

    def recv_residual_frame(self) -> tuple[dict[str, Any], bytes]:
        sock = self._ensure_socket()
        frame = recv_frame(sock)
        if frame.kind != "RESIDUAL":
            raise RuntimeError(f"V3 expected RESIDUAL, got {frame.kind}")
        return frame.header, frame.body

    def stats(self) -> dict[str, float | int]:
        wire = self.anchor_wire_bytes + self.residual_wire_bytes
        return {
            "boundaries": self.boundaries,
            "wire_bytes": wire,
            "anchor_wire_bytes": self.anchor_wire_bytes,
            "residual_wire_bytes": self.residual_wire_bytes,
            "bf16_reference_bytes": self.bf16_reference_bytes,
            "reduction": 1.0 - wire / self.bf16_reference_bytes if self.bf16_reference_bytes else 0.0,
            "suffix_requests": self.suffix_requests,
            "encode_seconds": self.encode_seconds,
            "decode_seconds": self.decode_seconds,
            "approximate_callback_seconds": self.approximate_seconds,
        }
