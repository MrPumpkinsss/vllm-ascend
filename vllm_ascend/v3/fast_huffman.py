from __future__ import annotations

# ruff: noqa: E402

"""Optional native canonical-Huffman codec for the V3 exact residual."""

import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

WINDOW_BITS = 16
LONG_MARKER = np.uint32(0xFFFFFFFF)
LENGTH_SHIFT = np.uint32(20)


def _cache_dir() -> Path:
    override = os.environ.get("VLLM_ASCEND_V3_CODEC_CACHE")
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "vllm-ascend-v3-codec"


def _load_native():
    if os.environ.get("VLLM_ASCEND_V3_NATIVE_HUFFMAN", "1") == "0":
        return None
    source = Path(__file__).with_name("huffman_fast.cpp")
    library = _cache_dir() / "libv3_huffman.so"
    library.parent.mkdir(parents=True, exist_ok=True)
    if not library.exists() or source.stat().st_mtime > library.stat().st_mtime:
        temporary = library.with_name(f".{os.getpid()}-{library.name}.tmp.so")
        try:
            subprocess.run(
                ["g++", "-O3", "-fPIC", "-shared", str(source), "-o", str(temporary)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.replace(temporary, library)
        except (OSError, subprocess.CalledProcessError):
            temporary.unlink(missing_ok=True)
            return None
    try:
        lib = ctypes.CDLL(library)
        u32 = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        u16 = np.ctypeslib.ndpointer(dtype=np.uint16, flags="C_CONTIGUOUS")
        i64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        u64 = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
        lib.huff_encode.argtypes = [u32, u8, i64, u16, u16, ctypes.c_int64, u8, ctypes.c_int64, ctypes.c_uint16]
        lib.huff_encode.restype = ctypes.c_int64
        lib.huff_decode.argtypes = [
            u32,
            i64,
            u64,
            u64,
            u32,
            i64,
            ctypes.c_int64,
            u8,
            ctypes.c_int64,
            u16,
            u16,
            ctypes.c_int32,
            ctypes.c_uint16,
        ]
        lib.huff_decode.restype = ctypes.c_int64
        return lib
    except OSError:
        return None


class NativeHuffmanCodec:
    """Encoder/decoder using the same canonical wire format as Python fallback."""

    def __init__(self, tables: list[list[str]], max_code_length: int):
        if not (0 < max_code_length <= 32):
            raise ValueError(f"unsupported Huffman code length {max_code_length}")
        self.tables = tables
        self.max_code_length = max_code_length
        self.native = _load_native()
        self.escape_symbol = tables[0] and len(tables[0]) - 1
        rows = len(tables)
        self.code_values = np.zeros((rows, 1024), dtype=np.uint32)
        self.code_lengths = np.zeros((rows, 1024), dtype=np.uint8)
        self.luts = np.full((rows, 1 << WINDOW_BITS), LONG_MARKER, dtype=np.uint32)
        long_rows: list[list[tuple[int, int, int]]] = [[] for _ in tables]
        long_width = max(max_code_length, WINDOW_BITS)

        for context, codes in enumerate(tables):
            for symbol, code in enumerate(codes):
                length = len(code)
                if not length:
                    continue
                value = int(code, 2)
                self.code_values[context, symbol] = value
                self.code_lengths[context, symbol] = length
                if length <= WINDOW_BITS:
                    lo = value << (WINDOW_BITS - length)
                    hi = (value + 1) << (WINDOW_BITS - length)
                    self.luts[context, lo:hi] = (np.uint32(length) << LENGTH_SHIFT) | np.uint32(symbol)
                else:
                    lo = value << (long_width - length)
                    hi = (value + 1) << (long_width - length)
                    long_rows[context].append((lo, hi, (length << 20) | symbol))
                    # A canonical <=16-bit code cannot share this 16-bit prefix.
                    self.luts[context, value >> (length - WINDOW_BITS)] = LONG_MARKER

        max_count = max((len(row) for row in long_rows), default=0)
        self.long_offsets = np.zeros(len(tables) + 1, dtype=np.int64)
        self.long_starts = np.zeros(len(tables) * max_count, dtype=np.uint64)
        self.long_ends = np.zeros(len(tables) * max_count, dtype=np.uint64)
        self.long_values = np.zeros(len(tables) * max_count, dtype=np.uint32)
        for context, row in enumerate(long_rows):
            row.sort()
            self.long_offsets[context + 1] = self.long_offsets[context] + len(row)
            if not row:
                continue
            starts, ends, values = zip(*row)
            offset = context * max_count
            self.long_starts[offset : offset + len(row)] = starts
            self.long_ends[offset : offset + len(row)] = ends
            self.long_values[offset : offset + len(row)] = values
        self.long_width = long_width

    @property
    def available(self) -> bool:
        return self.native is not None

    def encode(self, context: np.ndarray, symbols: np.ndarray, exact_codes: np.ndarray) -> bytes:
        context = np.ascontiguousarray(context, dtype=np.int64)
        symbols = np.ascontiguousarray(symbols, dtype=np.uint16)
        exact_codes = np.ascontiguousarray(exact_codes, dtype=np.uint16)
        if not self.available:
            raise RuntimeError("native V3 Huffman encoder unavailable")
        capacity_bits = len(context) * (self.max_code_length + 16)
        output = np.zeros((capacity_bits + 7) // 8, dtype=np.uint8)
        bits = int(
            self.native.huff_encode(
                self.code_values,
                self.code_lengths,
                context,
                symbols,
                exact_codes,
                len(context),
                output,
                capacity_bits,
                self.escape_symbol,
            )
        )
        if bits < 0:
            raise RuntimeError(f"native V3 Huffman encoder error {bits}")
        return output[: (bits + 7) // 8].copy()

    def decode_symbols(self, payload: bytes, context: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        context = np.ascontiguousarray(context, dtype=np.int64)
        if not self.available:
            raise RuntimeError("native V3 Huffman decoder unavailable")
        data = np.zeros(len(payload) + 8, dtype=np.uint8)
        data[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
        decoded = np.empty(len(context), dtype=np.uint16)
        escapes = np.empty(len(context), dtype=np.uint16)
        position = int(
            self.native.huff_decode(
                self.luts,
                self.long_offsets,
                self.long_starts,
                self.long_ends,
                self.long_values,
                context,
                len(context),
                data,
                len(payload) * 8,
                decoded,
                escapes,
                self.long_width,
                self.escape_symbol,
            )
        )
        if position < 0:
            raise RuntimeError(f"native V3 Huffman decoder error {position}")
        return decoded, escapes, (-position) % 8
