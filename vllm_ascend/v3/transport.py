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

import contextlib
import io
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from .nvfp4_codec import NVFP4CodecConfig, NVFP4ExactCodec
from .protocol import recv_frame, send_frame
from .runtime import NetworkAwarePrefixPolicy

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
    codec: str = "nvfp4"
    engine_prefix_vv: bool = False

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
        config.codec = str(raw.get("codec", config.codec)).lower()
        config.engine_prefix_vv = bool(raw.get("engine_prefix_vv", config.engine_prefix_vv))
        if config.codec not in {"nvfp4", "raw"}:
            raise ValueError(f"unsupported V3 codec {config.codec}")
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
    encoding: str = "nvfp4"


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


class PendingBoundary:
    """Hold one adaptive boundary until exact verification authorizes commit."""

    def __init__(self, transport: VerdictAdaptiveTransport, approximate: dict[str, Any], metadata: dict[str, Any]):
        self.transport = transport
        self.approximate = approximate
        self.metadata = metadata
        self.tensors = dict(approximate)
        self.exact_rows = 0
        self.total_rows = max((entry.rows for entry in metadata["entries"].values()), default=0)
        self.row_bytes: list[int | None] = [None] * self.total_rows
        self.requested = False
        self.closed = False

    def request(self, prefix_length: int) -> dict[str, Any]:
        empty_initial = not self.requested and self.total_rows == 0 and prefix_length == 0
        if self.closed or (not empty_initial and prefix_length <= self.exact_rows) or prefix_length > self.total_rows:
            raise ValueError("invalid or closed V3 boundary request")
        kind = "RESIDUAL_REQUEST" if not self.requested else "SUFFIX_REQUEST"
        try:
            header, body = self.transport._request_residual(prefix_length, self.metadata, kind=kind)
            self.tensors = self.transport._decode_residual(
                self.approximate, self.metadata, header, body, previous=self.tensors
            )
            self.transport._validate_decoded_prefix(self.metadata, prefix_length)
            sizes = [0] * self.total_rows
            counts = [0] * self.total_rows
            for _, entry in header["entries"]:
                for index, size in enumerate(entry["lengths"], start=int(entry["start_row"])):
                    sizes[index] += int(size)
                    counts[index] += 1
            for index, count in enumerate(counts):
                if count == len(self.metadata["entries"]):
                    self.row_bytes[index] = sizes[index]
            for key, entry in self.metadata["entries"].items():
                if prefix_length < entry.rows:
                    self.tensors[key][prefix_length:].zero_()
            if self.requested:
                self.transport.suffix_requests += 1
            self.requested = True
            self.exact_rows = prefix_length
            return self.tensors
        except BaseException:
            self.abort()
            raise

    def finish(self, consumed_end: int) -> None:
        if self.closed or not self.requested or not 0 <= consumed_end <= self.exact_rows:
            raise ValueError("cannot finish V3 boundary beyond the exact prefix")
        try:
            send_frame(self.transport._ensure_socket(), "BOUNDARY_DONE", {"version": 1})
            self.closed = True
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        self.closed = True
        self.transport._shutdown_socket()


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
        codec_mode: str = "nvfp4",
    ):
        self.listen_host, self.listen_port = listen_host, listen_port
        self.connect_host, self.connect_port = connect_host, connect_port
        self.sender = sender
        self.adaptive = adaptive
        self.default_prefix_length = default_prefix_length
        self.codec_mode = codec_mode
        if not calibration_path:
            raise ValueError("V3 transport requires VLLM_ASCEND_V3_CALIBRATION")
        hidden_size = NVFP4ExactCodec.read_hidden_size(calibration_path)
        self.codec: NVFP4ExactCodec = NVFP4ExactCodec(
            NVFP4CodecConfig(hidden_size=hidden_size, calibration_path=calibration_path)
        )
        self._socket: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._background_sender: threading.Thread | None = None
        self._background_error: BaseException | None = None
        self.boundaries = 0
        self.anchor_wire_bytes = 0
        self.residual_wire_bytes = 0
        self.bf16_reference_bytes = 0
        self.suffix_requests = 0
        self.encode_seconds = 0.0
        self.decode_seconds = 0.0
        self.approximate_seconds = 0.0
        self.policy_decisions = 0
        self.policy_last_prefix = 0
        self._policy_row_bytes: list[int] | None = None

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
            codec_mode=config.codec,
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
                    sock.settimeout(None)
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
                if self.codec_mode == "raw":
                    # High-bandwidth compatibility mode carries exact BF16 only.
                    # The anchor wave remains a protocol event but has no NVFP4
                    # payload; the exact residual reconstructs every row bitwise.
                    exact_cpu = exact.cpu()
                    anchor = b""
                    anchor_code = np.zeros(
                        (int(np.prod(exact.shape[:-1], dtype=np.int64)), 0), dtype=np.uint8
                    )
                    prediction = torch.zeros_like(exact_cpu)
                else:
                    anchor, anchor_code = self.codec.encode_anchor(exact)
                    prediction, anchor_code = self.codec.decode_anchor(
                        anchor, exact.shape, exact.device, output_device="cpu"
                    )
                    exact_cpu = exact.cpu()
                boundary = _SemanticBoundary(
                    key,
                    tuple(exact.shape),
                    str(exact.dtype).removeprefix("torch."),
                    str(exact.device),
                    anchor,
                    exact_cpu,
                    prediction,
                    anchor_code,
                )
                semantic[key] = boundary
                if self.codec_mode == "raw":
                    boundary.encoding = "raw"
                    boundary.residuals = self.codec.encode_raw_rows(exact)
                    boundary.encoded_rows = int(exact.shape[0])
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
        if boundary.encoded_rows < rows:
            begin = boundary.encoded_rows
            prediction = boundary.prediction[begin:rows].to(boundary.exact.device)
            exact = boundary.exact[begin:rows].to(boundary.exact.device)
            row_tokens = int(np.prod(boundary.exact.shape[1:-1], dtype=np.int64))
            code = boundary.anchor_code[begin * row_tokens : rows * row_tokens]
            payloads, escape_rate = self.codec.encode_residual_rows(prediction, exact, code)
            boundary.residuals.extend(payloads)
            boundary.encoded_rows = rows
            boundary.escape_rate = escape_rate
        return boundary.residuals[:rows]

    def _send_full_residual_background(
        self,
        entries: list[tuple[str, str, Any]],
        semantic: dict[str, _SemanticBoundary],
    ) -> None:
        started = time.perf_counter()
        selected: list[tuple[str, _SemanticBoundary, list[bytes]]] = []
        residual_payloads: list[bytes] = []
        residual_bytes = 0
        for key, kind, _entry in entries:
            if kind != "semantic":
                continue
            boundary = semantic[key]
            payloads = self._encode_residual_rows(boundary, int(boundary.exact.shape[0]))
            selected.append((key, boundary, payloads))
            residual_payloads.extend(payloads)
            residual_bytes += sum(map(len, payloads))
        residual_entries = [
            (
                key,
                {
                    "start_row": 0,
                    "rows": len(payloads),
                    "lengths": [len(payload) for payload in payloads],
                },
            )
            for key, _, payloads in selected
        ]
        with self._send_lock:
            send_frame(
                self._ensure_socket(),
                "RESIDUAL",
                {"version": 1, "entries": residual_entries},
                b"".join(residual_payloads),
            )
        self.residual_wire_bytes += residual_bytes
        self.encode_seconds += time.perf_counter() - started

    def _run_background_sender(
        self,
        entries: list[tuple[str, str, Any]],
        semantic: dict[str, _SemanticBoundary],
    ) -> None:
        try:
            self._send_full_residual_background(entries, semantic)
        except BaseException as exc:
            self._background_error = exc
            self._shutdown_socket()

    def _shutdown_socket(self) -> None:
        if self._socket is not None:
            with contextlib.suppress(OSError):
                self._socket.shutdown(socket.SHUT_RDWR)

    def flush(self) -> None:
        """Wait for the preceding residual and propagate asynchronous failures."""
        if self._background_sender is not None:
            self._background_sender.join()
            self._background_sender = None
        if self._background_error is not None:
            error, self._background_error = self._background_error, None
            raise RuntimeError("V3 background residual sender failed") from error

    def send_tensor_dict(self, tensor_dict: dict[str, Any]) -> None:
        try:
            self._send_tensor_dict(tensor_dict)
        except BaseException:
            self._shutdown_socket()
            raise

    def _send_tensor_dict(self, tensor_dict: dict[str, Any]) -> None:
        self.flush()
        started = time.perf_counter()
        entries, semantic, anchor_body, anchor_bytes, reference_bytes = self._prepare_boundary(tensor_dict)
        sock = self._ensure_socket()
        with self._send_lock:
            send_frame(
                sock,
                "ANCHOR",
                {
                    "version": 1,
                    "entries": entries,
                    "adaptive": self.adaptive,
                    "codec": self.codec_mode,
                },
                anchor_body,
            )
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

        if not self.adaptive:
            thread = threading.Thread(
                target=self._run_background_sender,
                args=(entries, semantic),
                daemon=True,
            )
            self._background_sender = thread
            thread.start()
            return

        max_rows = max((entry.rows for _, kind, entry in entries if kind == "semantic"), default=0)
        if requested < 0 or requested > max_rows or (max_rows and requested == 0):
            raise ValueError(f"invalid V3 residual prefix length {requested} for {max_rows} rows")
        sent_rows = {key: 0 for key in semantic}
        while True:
            selected: list[tuple[str, int, list[bytes]]] = []
            residual_payloads: list[bytes] = []
            residual_bytes = 0
            for key, kind, entry in entries:
                if kind != "semantic":
                    continue
                boundary = semantic[key]
                end = min(max(requested, 0), int(boundary.exact.shape[0]))
                start = sent_rows[key]
                payloads = self._encode_residual_rows(boundary, end)[start:end]
                selected.append((key, start, payloads))
                sent_rows[key] = end
                residual_payloads.extend(payloads)
                residual_bytes += sum(map(len, payloads))
            residual_entries = [
                (
                    key,
                    {
                        "start_row": start,
                        "rows": len(payloads),
                        "lengths": [len(payload) for payload in payloads],
                    },
                )
                for key, start, payloads in selected
            ]
            send_frame(
                sock,
                "RESIDUAL",
                {"version": 1, "entries": residual_entries},
                b"".join(residual_payloads),
            )
            self.residual_wire_bytes += residual_bytes
            verdict = recv_frame(sock)
            if verdict.kind == "BOUNDARY_DONE":
                break
            if verdict.kind != "SUFFIX_REQUEST":
                raise RuntimeError(
                    f"V3 sender expected SUFFIX_REQUEST/DONE, got {verdict.kind}"
                )
            next_requested = int(verdict.header["prefix_length"])
            if next_requested <= requested or next_requested > max_rows:
                raise ValueError(f"invalid V3 suffix request {next_requested} after {requested}")
            requested = next_requested
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
                target = torch.device(entry.device)
                if frame.header.get("codec") == "raw":
                    prediction = torch.zeros(torch.Size(entry.shape), dtype=torch.bfloat16, device=target)
                    anchor_code = np.zeros(
                        (int(np.prod(entry.shape[:-1], dtype=np.int64)), 0), dtype=np.uint8
                    )
                else:
                    prediction, anchor_code = self.codec.decode_anchor(
                        payload, torch.Size(entry.shape), target
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
            "codec": str(frame.header.get("codec", "nvfp4")),
            "entries": semantic_entries,
            "anchor_codes": anchor_codes,
            "decoded_rows": {key: 0 for key in semantic_entries},
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
        previous: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = dict(approximate if previous is None else previous)
        residual_bytes = 0
        offset = 0
        started = time.perf_counter()
        if residual_header.get("version") != 1:
            raise RuntimeError("unsupported V3 residual version")
        entries = residual_header["entries"]
        if len(entries) != len(metadata["entries"]) or {key for key, _ in entries} != set(metadata["entries"]):
            raise RuntimeError("V3 residual entries do not match anchor")
        for key, entry in entries:
            anchor_code = metadata["anchor_codes"][key]
            prediction = approximate[key]
            start = int(entry["start_row"])
            rows = int(entry["rows"])
            lengths = entry["lengths"]
            if start < 0 or rows < 0 or start + rows > prediction.shape[0] or len(lengths) != rows:
                raise RuntimeError("invalid V3 residual row range")
            if start != metadata["decoded_rows"][key]:
                raise RuntimeError("V3 residual rows must be contiguous and non-overlapping")
            payloads: list[bytes] = []
            for length in lengths:
                size = int(length)
                if size < 0 or offset + size > len(body):
                    raise RuntimeError("invalid V3 residual payload length")
                payload = body[offset : offset + size]
                offset += size
                payloads.append(payload)
            tensor = result[key].clone()
            if rows:
                end = start + rows
                if metadata.get("codec") == "raw":
                    decoded = self.codec.decode_raw_rows(prediction[start:end], payloads)
                else:
                    row_tokens = int(np.prod(prediction.shape[1:-1], dtype=np.int64))
                    codes = anchor_code[start * row_tokens : end * row_tokens]
                    decoded = self.codec.decode_residual_rows(prediction[start:end], codes, payloads)
                tensor[start:end] = decoded.to(device=prediction.device, dtype=prediction.dtype)
            result[key] = tensor
            metadata["decoded_rows"][key] = start + rows
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
        policy: NetworkAwarePrefixPolicy | None = None,
    ) -> dict[str, Any]:
        """Receive a boundary; only callback-validated prefix rows may be consumed.

        Without a verdict callback the full tensor is always reconstructed.
        With a callback, unrequested suffix rows remain approximate and the
        caller must not use them to update committed model state.
        """
        try:
            return self._recv_tensor_dict(approximate_callback, verdict_callback, policy)
        except BaseException:
            self._shutdown_socket()
            raise

    def begin_boundary(self) -> PendingBoundary:
        """Receive the anchor without requesting residual or mutating model state."""
        try:
            approximate, metadata = self.recv_anchor()
            if not metadata["adaptive"]:
                raise RuntimeError("pending boundaries require adaptive transport")
            return PendingBoundary(self, approximate, metadata)
        except BaseException:
            self._shutdown_socket()
            raise

    def _recv_tensor_dict(
        self,
        approximate_callback: Callable[[dict[str, Any]], BoundaryDecision | int | None] | None = None,
        verdict_callback: Callable[[dict[str, Any], int], int] | None = None,
        policy: NetworkAwarePrefixPolicy | None = None,
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
            prefix_length = max((entry.rows for entry in metadata["entries"].values()), default=0)

        predicted_end = decision.metadata.get("predicted_end") if isinstance(decision, BoundaryDecision) else None
        use_policy = (
            metadata["adaptive"]
            and verdict_callback is not None
            and policy is not None
            and bool(metadata["entries"])
            and all(entry.rows == policy.k for entry in metadata["entries"].values())
        )
        if use_policy:
            assert policy is not None
            if self._policy_row_bytes is None:
                prefix_length = policy.k
            else:
                prefix_length = policy.choose(self._policy_row_bytes, predicted_end=predicted_end).prefix_length
            self.policy_decisions += 1
            self.policy_last_prefix = prefix_length
            observed_sizes = [0] * policy.k
            observed_counts = [0] * policy.k

            def record_sizes(header: dict[str, Any]) -> None:
                for _, entry in header["entries"]:
                    start = int(entry["start_row"])
                    for index, size in enumerate(entry["lengths"], start=start):
                        observed_sizes[index] += int(size)
                        observed_counts[index] += 1

        if metadata["adaptive"]:
            max_rows = max((entry.rows for entry in metadata["entries"].values()), default=0)
            if verdict_callback is None:
                prefix_length = max_rows
            if prefix_length < 0 or prefix_length > max_rows or (max_rows and prefix_length == 0):
                raise ValueError(f"invalid V3 residual prefix length {prefix_length} for {max_rows} rows")
            residual_header, residual_body = self._request_residual(prefix_length, metadata)
        else:
            # A non-adaptive sender streams residual immediately; do not inject a
            # control request that it will never consume.
            residual_header, residual_body = self.recv_residual_frame()
        result = self._decode_residual(approximate, metadata, residual_header, residual_body)
        self._validate_decoded_prefix(metadata, prefix_length)
        if use_policy:
            record_sizes(residual_header)
        sock = self._ensure_socket()
        if metadata["adaptive"] and verdict_callback is not None:
            while True:
                consumed_end = int(verdict_callback(result, prefix_length))
                if consumed_end < 0 or consumed_end > max_rows:
                    raise ValueError(f"invalid V3 consumed end {consumed_end}")
                if consumed_end <= prefix_length:
                    break
                self.suffix_requests += 1
                residual_header, residual_body = self._request_residual(consumed_end, metadata, kind="SUFFIX_REQUEST")
                result = self._decode_residual(approximate, metadata, residual_header, residual_body, previous=result)
                prefix_length = consumed_end
                self._validate_decoded_prefix(metadata, prefix_length)
                if use_policy:
                    record_sizes(residual_header)
            if use_policy:
                assert policy is not None
                policy.observe(consumed_end, predicted_end=predicted_end)
                if self._policy_row_bytes is None:
                    self._policy_row_bytes = [0] * policy.k
                for index, count in enumerate(observed_counts):
                    if count == len(metadata["entries"]):
                        self._policy_row_bytes[index] = observed_sizes[index]
        if metadata["adaptive"]:
            send_frame(sock, "BOUNDARY_DONE", {"version": 1})
        return result

    @staticmethod
    def _validate_decoded_prefix(metadata: dict[str, Any], prefix_length: int) -> None:
        for key, entry in metadata["entries"].items():
            if metadata["decoded_rows"][key] != min(prefix_length, entry.rows):
                raise RuntimeError("V3 residual response did not cover the requested prefix")

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
            "policy_decisions": self.policy_decisions,
            "policy_last_prefix": self.policy_last_prefix,
        }
