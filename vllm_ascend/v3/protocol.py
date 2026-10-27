from __future__ import annotations

# ruff: noqa: E402

"""Length-prefixed, CRC-checked control frames for V3 split inference."""

import pickle
import struct
import zlib
from dataclasses import dataclass
from typing import Any

MAGIC = b"V3F1"
FRAME_HEADER = struct.Struct("<4sIIIQ")


class FrameProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Frame:
    kind: str
    header: dict[str, Any]
    body: bytes = b""


def send_frame(sock, kind: str, header: dict[str, Any], body: bytes = b"") -> None:
    if not kind or not kind.isascii() or not kind.replace("_", "").isalnum():
        raise ValueError(f"invalid V3 frame kind: {kind!r}")
    encoded_kind = kind.encode("ascii")
    if len(encoded_kind) > 32:
        raise ValueError("V3 frame kind exceeds 32 bytes")
    header_bytes = pickle.dumps(header, protocol=pickle.HIGHEST_PROTOCOL)
    header_crc = zlib.crc32(header_bytes) & 0xFFFFFFFF
    body_crc = zlib.crc32(body) & 0xFFFFFFFF
    prefix = FRAME_HEADER.pack(MAGIC, len(encoded_kind), len(header_bytes), len(body), header_crc | body_crc << 32)
    # This is a small control path; sendall is intentional and preserves ordering.
    sock.sendall(prefix + encoded_kind + header_bytes + body)


def recv_exact(sock, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("V3 peer closed")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_frame(sock) -> Frame:
    prefix = recv_exact(sock, FRAME_HEADER.size)
    magic, kind_size, header_size, body_size, packed_crc = FRAME_HEADER.unpack(prefix)
    if magic != MAGIC:
        raise FrameProtocolError("invalid V3 frame magic")
    kind = recv_exact(sock, kind_size).decode("ascii")
    header_bytes = recv_exact(sock, header_size)
    body = recv_exact(sock, body_size)
    if (zlib.crc32(header_bytes) & 0xFFFFFFFF) | (zlib.crc32(body) & 0xFFFFFFFF) << 32 != packed_crc:
        raise FrameProtocolError("V3 frame CRC mismatch")
    try:
        header = pickle.loads(header_bytes)
    except Exception as exc:
        raise FrameProtocolError("invalid V3 frame header") from exc
    if not isinstance(header, dict):
        raise FrameProtocolError("V3 frame header must be a dict")
    return Frame(kind, header, body)
