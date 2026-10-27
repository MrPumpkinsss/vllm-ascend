from __future__ import annotations

# ruff: noqa: E402

"""Exact NVFP4 activation codec for the V3 split-boundary protocol.

The first wave is a signed e2m1 value plus one e4m3 scale per 16 channels
(4.5 bits/element).  The second wave is an FP4-context Huffman-coded signed
BF16-rank delta with a raw-BF16 escape.  Residuals can be encoded either as
one continuous stream or as independent first-dimension rows; the latter is
required by Prefix-Repair.  Every transmitted row reconstructs its original
BF16 code exactly.
"""

import heapq
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .fast_huffman import NativeHuffmanCodec

FP4_BLOCK = 16
FP4_POSITIVE = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP4_MIDPOINTS = np.asarray([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
MAX_RANK_DELTA = 255
NUM_SYMBOLS = 2 * MAX_RANK_DELTA + 2
ESCAPE_SYMBOL = NUM_SYMBOLS - 1


def _bf16_rank_tables() -> tuple[np.ndarray, np.ndarray]:
    codes = np.arange(65536, dtype=np.uint16)
    values = (codes.astype(np.uint32) << 16).view(np.float32)
    finite = np.isfinite(values)
    order = np.argsort(values[finite], kind="stable")
    finite_codes = codes[finite][order]
    rank = np.full(65536, -1, dtype=np.int32)
    rank[finite_codes] = np.arange(len(finite_codes), dtype=np.int32)
    return rank, finite_codes


RANK_TABLE, FINITE_CODES = _bf16_rank_tables()


@dataclass(frozen=True)
class NVFP4CodecConfig:
    hidden_size: int = 5120
    calibration_path: str | None = None


def _canonical_huffman(probability: np.ndarray) -> list[str]:
    weights = np.maximum(probability.astype(np.float64) * 1_000_000_000.0, 1.0)
    heap = [(int(weight), symbol, (symbol,)) for symbol, weight in enumerate(weights)]
    heapq.heapify(heap)
    lengths = [0] * len(probability)
    tiebreak = len(probability)
    while len(heap) > 1:
        weight1, _, leaves1 = heapq.heappop(heap)
        weight2, _, leaves2 = heapq.heappop(heap)
        for leaf in leaves1 + leaves2:
            lengths[leaf] += 1
        tiebreak += 1
        heapq.heappush(heap, (weight1 + weight2, tiebreak, leaves1 + leaves2))
    pairs = sorted((length, symbol) for symbol, length in enumerate(lengths) if length)
    codes: list[str] = [""] * len(probability)
    value, previous_length = 0, pairs[0][0] if pairs else 0
    for length, symbol in pairs:
        value <<= length - previous_length
        codes[symbol] = format(value, f"0{length}b")
        value += 1
        previous_length = length
    if codes and max(map(len, codes)) > 32:
        raise RuntimeError("canonical Huffman code exceeds 32 bits")
    return codes


class _BitWriter:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, bits: str) -> None:
        self.parts.append(bits)

    def bytes(self) -> bytes:
        joined = "".join(self.parts)
        joined += "0" * ((-len(joined)) % 8)
        return np.packbits(
            np.frombuffer(joined.encode("ascii"), np.uint8) - ord("0"),
            bitorder="big",
        ).tobytes()


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self.bits = np.unpackbits(np.frombuffer(payload, np.uint8), bitorder="big").tolist()
        self.position = 0

    def bit(self) -> str:
        if self.position >= len(self.bits):
            raise ValueError("NVFP4 residual bitstream underflow")
        value = self.bits[self.position]
        self.position += 1
        return str(value)

    def raw16(self) -> int:
        return int("".join(self.bit() for _ in range(16)), 2)


class NVFP4ExactCodec:
    """NVFP4 anchor plus exact, row-addressable BF16 residual."""

    @staticmethod
    def read_hidden_size(path: str | Path) -> int:
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
        return int(metadata["hidden_size"])

    def __init__(self, config: NVFP4CodecConfig | None = None):
        self.config = config or NVFP4CodecConfig()
        if self.config.hidden_size % FP4_BLOCK:
            raise ValueError("hidden_size must be divisible by 16")
        self.tables: list[list[str]] | None = None
        self.decoders: list[dict[str, int]] | None = None
        self.max_code_length = 0
        self.native_huffman: NativeHuffmanCodec | None = None
        if self.config.calibration_path:
            self.load(self.config.calibration_path)

    def _tokens(self, tensor: torch.Tensor) -> int:
        return int(np.prod(tensor.shape[:-1], dtype=np.int64))

    def _codes(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().contiguous().view(torch.uint16).numpy().astype(np.uint16)

    def encode_anchor(self, tensor: torch.Tensor) -> tuple[bytes, np.ndarray]:
        if tensor.dtype != torch.bfloat16 or tensor.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected BF16 [..., {self.config.hidden_size}], got {tuple(tensor.shape)} {tensor.dtype}"
            )
        n = self._tokens(tensor)
        blocks = self.config.hidden_size // FP4_BLOCK
        z = tensor.detach().cpu().float().numpy().reshape(n, blocks, FP4_BLOCK)
        amax = np.abs(z).max(axis=-1, keepdims=True)
        scale_exact = np.clip(amax / 6.0, 2.0**-9, 448.0).astype(np.float32)
        scale8 = torch.from_numpy(scale_exact).to(torch.float8_e4m3fn)
        scale = scale8.float().numpy()
        norm = z.astype(np.float32) / scale
        sign = np.sign(norm)
        index = np.searchsorted(FP4_MIDPOINTS, np.abs(norm), side="right")
        code = ((sign < 0).astype(np.uint8) << 3) | index.astype(np.uint8)
        # Two signed e2m1 codes form one byte.  This direct nibble pack avoids
        # the 4x temporary produced by unpackbits/packbits.
        flat_code = code.reshape(-1).astype(np.uint8)
        packed = (flat_code[0::2] << 4) | flat_code[1::2]
        payload = packed.tobytes()
        payload += scale8.view(torch.uint8).numpy().tobytes()
        expected = n * (self.config.hidden_size // 2 + blocks)
        if len(payload) != expected:
            raise AssertionError((len(payload), expected))
        return payload, code

    def decode_anchor(
        self,
        payload: bytes,
        shape: torch.Size,
        device: torch.device,
        *,
        output_device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, np.ndarray]:
        blocks = self.config.hidden_size // FP4_BLOCK
        per_token = self.config.hidden_size // 2 + blocks
        if len(payload) % per_token:
            raise ValueError(f"bad NVFP4 anchor length {len(payload)}")
        n = len(payload) // per_token
        expected_n = self._tokens(torch.empty(shape, dtype=torch.bfloat16))
        if n != expected_n:
            raise ValueError(f"anchor has {n} tokens, metadata expects {expected_n}")
        qbytes = n * self.config.hidden_size // 2
        packed = np.frombuffer(payload[:qbytes], np.uint8)
        code = np.empty(packed.size * 2, dtype=np.uint8)
        code[0::2] = packed >> 4
        code[1::2] = packed & np.uint8(0x0F)
        code = code.reshape(n, blocks, FP4_BLOCK)
        scale_bytes = np.frombuffer(payload[qbytes:], np.uint8).reshape(n, blocks, 1).copy()
        scale = torch.from_numpy(scale_bytes).view(torch.float8_e4m3fn).float().numpy()
        sign = np.where(code >= 8, -1.0, 1.0).astype(np.float32)
        q = sign * FP4_POSITIVE[(code & 7).astype(np.int8)]
        prediction = (q.astype(np.float32) * scale).reshape(n, self.config.hidden_size)
        tensor = torch.from_numpy(prediction.copy()).to(torch.bfloat16).reshape(shape)
        target = device if output_device is None else torch.device(output_device)
        return tensor.to(device=target), code

    def _symbols(self, prediction: torch.Tensor, exact: torch.Tensor):
        x = self._codes(exact).reshape(-1)
        p = self._codes(prediction).reshape(-1)
        rank_x, rank_p = RANK_TABLE[x], RANK_TABLE[p]
        delta = rank_x.astype(np.int64) - rank_p.astype(np.int64)
        symbols = np.full(len(delta), ESCAPE_SYMBOL, dtype=np.uint16)
        representable = (np.abs(delta) <= MAX_RANK_DELTA) & (rank_x >= 0) & (rank_p >= 0)
        symbols[representable] = (delta[representable] + MAX_RANK_DELTA).astype(np.uint16)
        return symbols, x, p, representable

    def fit(self, samples: Iterable[torch.Tensor]) -> NVFP4ExactCodec:
        counts = np.zeros((16, NUM_SYMBOLS), dtype=np.float64)
        elements = 0
        for sample in samples:
            sample = sample.detach().cpu().to(torch.bfloat16)
            anchor, code = self.encode_anchor(sample)
            prediction, decoded_code = self.decode_anchor(anchor, sample.shape, sample.device)
            symbols, _, _, _ = self._symbols(prediction, sample)
            context = decoded_code.reshape(-1).astype(np.int64)
            flat = context * NUM_SYMBOLS + symbols.astype(np.int64)
            counts += np.bincount(flat, minlength=16 * NUM_SYMBOLS).reshape(16, NUM_SYMBOLS)
            elements += len(symbols)
        if elements == 0:
            raise ValueError("NVFP4 calibration is empty")
        global_counts = counts.sum(axis=0)
        global_probability = (global_counts + 1.0 / NUM_SYMBOLS) / (global_counts.sum() + 1.0)
        beta = 64.0
        self.tables, self.decoders, self.max_code_length = [], [], 0
        for context in range(16):
            probability = (counts[context] + beta * global_probability) / (counts[context].sum() + beta)
            codes = _canonical_huffman(probability)
            self.max_code_length = max(self.max_code_length, max(map(len, codes)))
            self.tables.append(codes)
            self.decoders.append({code: symbol for symbol, code in enumerate(codes)})
        self.native_huffman = NativeHuffmanCodec(self.tables, self.max_code_length)
        return self

    def save(self, path: str | Path) -> None:
        if self.tables is None:
            raise RuntimeError("cannot save an unfitted NVFP4 codec")
        metadata = {
            "format": "vllm-ascend-v3-nvfp4",
            "hidden_size": self.config.hidden_size,
            "max_code_length": self.max_code_length,
        }
        np.savez_compressed(Path(path), tables=np.asarray(self.tables, dtype=str), metadata=json.dumps(metadata))

    def load(self, path: str | Path) -> NVFP4ExactCodec:
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("format") != "vllm-ascend-v3-nvfp4":
                raise ValueError("not a V3 NVFP4 calibration archive")
            if int(metadata["hidden_size"]) != self.config.hidden_size:
                raise ValueError("hidden size mismatch in NVFP4 calibration")
            tables = archive["tables"].astype(str)
        if tables.shape != (16, NUM_SYMBOLS):
            raise ValueError(f"bad NVFP4 table shape {tables.shape}")
        self.tables = tables.tolist()
        self.decoders = [{code: symbol for symbol, code in enumerate(row)} for row in self.tables]
        self.max_code_length = max(len(code) for row in self.tables for code in row)
        self.native_huffman = NativeHuffmanCodec(self.tables, self.max_code_length)
        return self

    def encode_residual(
        self, prediction: torch.Tensor, exact: torch.Tensor, anchor_code: np.ndarray
    ) -> tuple[bytes, float]:
        if self.tables is None:
            raise RuntimeError("NVFP4 residual coder is not calibrated")
        if prediction.shape != exact.shape:
            raise ValueError("prediction and exact tensors must have the same shape")
        symbols, exact_codes, _, representable = self._symbols(prediction, exact)
        context = anchor_code.reshape(-1).astype(np.int64)
        if len(context) != len(symbols):
            raise ValueError("anchor code count does not match residual tensor")
        if self.native_huffman is not None and self.native_huffman.available:
            payload = self.native_huffman.encode(context, symbols, exact_codes)
            return payload, float(1.0 - representable.mean())
        writer = _BitWriter()
        for ctx, symbol, represented, code in zip(context, symbols, representable, exact_codes):
            writer.write(self.tables[int(ctx)][int(symbol)])
            if not represented:
                writer.write(format(int(code), "016b"))
        return writer.bytes(), float(1.0 - representable.mean())

    def encode_residual_rows(
        self, prediction: torch.Tensor, exact: torch.Tensor, anchor_code: np.ndarray
    ) -> tuple[list[bytes], float]:
        if self.tables is None:
            raise RuntimeError("NVFP4 residual coder is not calibrated")
        if prediction.shape != exact.shape or prediction.shape[0] == 0:
            raise ValueError("prediction and exact row tensors must have the same non-empty shape")
        symbols, exact_codes, _, representable = self._symbols(prediction, exact)
        context = anchor_code.reshape(-1).astype(np.int64)
        if len(context) != len(symbols):
            raise ValueError("anchor code count does not match residual tensor")
        if self.native_huffman is not None and self.native_huffman.available:
            row_elements = int(np.prod(prediction.shape[1:], dtype=np.int64))
            payloads = self.native_huffman.encode_rows(context, symbols, exact_codes, row_elements)
            return payloads, float(1.0 - representable.mean())
        payloads: list[bytes] = []
        row_tokens = int(np.prod(prediction.shape[1:-1], dtype=np.int64))
        for row in range(prediction.shape[0]):
            payload, escape_rate = self.encode_residual(
                prediction[row : row + 1], exact[row : row + 1],
                anchor_code[row * row_tokens : (row + 1) * row_tokens],
            )
            payloads.append(payload)
        return payloads, float(1.0 - representable.mean())

    def decode_residual_rows(
        self, prediction: torch.Tensor, anchor_code: np.ndarray, payloads: list[bytes]
    ) -> torch.Tensor:
        if self.decoders is None:
            raise RuntimeError("NVFP4 residual coder is not calibrated")
        if len(payloads) > prediction.shape[0]:
            raise ValueError("more residual payloads than tensor rows")
        predicted_codes = self._codes(prediction).reshape(-1)
        context = anchor_code.reshape(-1).astype(np.int64)
        if len(context) != len(predicted_codes):
            raise ValueError("anchor code count does not match residual tensor")
        row_elements = int(np.prod(prediction.shape[1:], dtype=np.int64))
        decoded = np.empty(len(payloads) * row_elements, dtype=np.uint16)
        escapes = np.empty(len(payloads) * row_elements, dtype=np.uint16)
        if self.native_huffman is not None and self.native_huffman.available:
            for row, payload in enumerate(payloads):
                begin, end = row * row_elements, (row + 1) * row_elements
                decoded[begin:end], escapes[begin:end], _ = self.native_huffman.decode_symbols(
                    payload, context[begin:end]
                )
        else:
            for row, payload in enumerate(payloads):
                begin, end = row * row_elements, (row + 1) * row_elements
                reader = _BitReader(payload)
                for index in range(begin, end):
                    prefix, symbol, decoder = "", None, self.decoders[int(context[index])]
                    while symbol is None:
                        prefix += reader.bit()
                        symbol = decoder.get(prefix)
                        if len(prefix) > self.max_code_length:
                            raise ValueError("invalid NVFP4 residual Huffman prefix")
                    decoded[index] = symbol
                    if symbol == ESCAPE_SYMBOL:
                        escapes[index] = reader.raw16()
        rank = RANK_TABLE[predicted_codes[: len(decoded)]].astype(np.int32)
        rank += decoded.astype(np.int32) - MAX_RANK_DELTA
        escaped = decoded == ESCAPE_SYMBOL
        bounded = (~escaped) & (rank >= 0) & (rank < len(FINITE_CODES))
        if not bool(np.all(escaped | bounded)):
            raise ValueError("decoded BF16 rank is out of range")
        output = np.empty(len(prediction.reshape(-1)), dtype=np.uint16)
        output[: len(decoded)][escaped] = escapes[escaped]
        output[: len(decoded)][~escaped] = FINITE_CODES[rank[~escaped]]
        # Untouched suffix rows retain their deterministic anchor prediction.
        output[len(decoded) :] = predicted_codes[len(decoded) :]
        return torch.from_numpy(output.copy()).view(torch.bfloat16).reshape(prediction.shape).to(
            device=prediction.device, dtype=torch.bfloat16
        )

    def decode_residual(
        self, prediction: torch.Tensor, anchor_code: np.ndarray, payload: bytes, shape: torch.Size, device: torch.device
    ) -> torch.Tensor:
        if self.decoders is None:
            raise RuntimeError("NVFP4 residual coder is not calibrated")
        predicted_codes = self._codes(prediction).reshape(-1)
        context = anchor_code.reshape(-1).astype(np.int64)
        if len(context) != len(predicted_codes):
            raise ValueError("anchor code count does not match residual tensor")
        if self.native_huffman is not None and self.native_huffman.available:
            decoded, escape_codes, _ = self.native_huffman.decode_symbols(payload, context)
            rank = RANK_TABLE[predicted_codes].astype(np.int32) + decoded.astype(np.int32) - MAX_RANK_DELTA
            escaped = decoded == ESCAPE_SYMBOL
            bounded = (~escaped) & (rank >= 0) & (rank < len(FINITE_CODES))
            if not bool(np.all(escaped | bounded)):
                raise ValueError("decoded BF16 rank is out of range")
            output = np.empty(len(context), dtype=np.uint16)
            output[escaped] = escape_codes[escaped]
            output[~escaped] = FINITE_CODES[rank[~escaped]]
        else:
            reader = _BitReader(payload)
            output = np.empty(len(context), dtype=np.uint16)
            for index in range(len(context)):
                prefix, symbol, decoder = "", None, self.decoders[int(context[index])]
                while symbol is None:
                    prefix += reader.bit()
                    symbol = decoder.get(prefix)
                    if len(prefix) > self.max_code_length:
                        raise ValueError("invalid NVFP4 residual Huffman prefix")
                if symbol == ESCAPE_SYMBOL:
                    output[index] = reader.raw16()
                else:
                    rank = int(RANK_TABLE[int(predicted_codes[index])]) + int(symbol) - MAX_RANK_DELTA
                    if not 0 <= rank < len(FINITE_CODES):
                        raise ValueError("decoded BF16 rank is out of range")
                    output[index] = FINITE_CODES[rank]
        tensor = torch.from_numpy(output.copy()).view(torch.bfloat16).reshape(shape)
        return tensor.to(device=device, dtype=torch.bfloat16)

    @staticmethod
    def encode_raw_rows(tensor: torch.Tensor) -> list[bytes]:
        raw = tensor.detach().cpu().contiguous().view(torch.uint16).numpy()
        return [row.tobytes() for row in raw.reshape(raw.shape[0], -1)]

    @staticmethod
    def decode_raw_rows(
        prediction: torch.Tensor, payloads: list[bytes]
    ) -> torch.Tensor:
        if not payloads:
            return prediction
        rows = len(payloads)
        raw = np.frombuffer(b"".join(payloads), dtype=np.uint16).copy()
        expected = int(np.prod(prediction.shape[1:], dtype=np.int64))
        if raw.size != rows * expected:
            raise ValueError("bad raw BF16 residual size")
        exact = torch.from_numpy(raw).view(torch.bfloat16).reshape(
            rows, *prediction.shape[1:]
        )
        result = prediction.clone()
        result[:rows] = exact.to(device=prediction.device, dtype=prediction.dtype)
        return result

    def encode(self, tensor: torch.Tensor) -> tuple[bytes, bytes, torch.Tensor, float]:
        anchor, code = self.encode_anchor(tensor)
        prediction, decoded_code = self.decode_anchor(anchor, tensor.shape, tensor.device)
        residual, escape_rate = self.encode_residual(prediction, tensor, decoded_code)
        return anchor, residual, prediction, escape_rate

    def decode(self, shape: torch.Size, device: torch.device, anchor: bytes, residual: bytes) -> torch.Tensor:
        prediction, code = self.decode_anchor(anchor, shape, device)
        return self.decode_residual(prediction, code, residual, shape, device)

    def encode_rows(
        self, tensor: torch.Tensor, rows: range | list[int] | None = None
    ) -> tuple[bytes, list[bytes], torch.Tensor, float]:
        """Encode all anchor rows and independent residual streams for selected rows."""
        anchor, code = self.encode_anchor(tensor)
        prediction, decoded_code = self.decode_anchor(anchor, tensor.shape, tensor.device)
        n = tensor.shape[0]
        selected = list(range(n)) if rows is None else list(rows)
        residuals: list[bytes] = []
        escapes: list[float] = []
        row_tokens = int(np.prod(tensor.shape[1:-1], dtype=np.int64))
        for row in selected:
            residual, escape = self.encode_residual(
                prediction[row : row + 1], tensor[row : row + 1],
                decoded_code[row * row_tokens : (row + 1) * row_tokens],
            )
            residuals.append(residual)
            escapes.append(escape)
        return anchor, residuals, prediction, float(np.mean(escapes or [0.0]))

    def decode_rows(
        self,
        shape: torch.Size,
        device: torch.device,
        anchor: bytes,
        residuals: list[bytes],
        *,
        dummy_suffix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact tensor for transmitted rows plus the approximate full tensor."""
        if len(residuals) > shape[0]:
            raise ValueError("more residual rows than tensor rows")
        prediction, code = self.decode_anchor(anchor, shape, device)
        result = prediction.clone()
        row_tokens = int(np.prod(shape[1:-1], dtype=np.int64))
        for row, payload in enumerate(residuals):
            result[row : row + 1] = self.decode_residual(
                prediction[row : row + 1], code[row * row_tokens : (row + 1) * row_tokens],
                payload, result[row : row + 1].shape, device,
            )
        if dummy_suffix is not None and len(residuals) < shape[0]:
            result[len(residuals) :] = dummy_suffix.to(device=device, dtype=result.dtype)
        return result, prediction


def fit_calibration_archive(archive_path: str | Path, output_path: str | Path) -> int:
    """Fit tables from an NPZ archive of BF16 uint16 codes, shape [K, hidden]."""
    samples: list[torch.Tensor] = []
    with np.load(Path(archive_path), allow_pickle=False) as archive:
        for key in archive.files:
            raw = archive[key]
            if raw.ndim == 1:
                raw = raw.reshape(-1, NVFP4CodecConfig().hidden_size)
            samples.append(
                torch.from_numpy(raw.astype(np.int16).copy())
                .view(torch.bfloat16)
                .reshape(raw.shape[0], NVFP4CodecConfig().hidden_size)
            )
    codec = NVFP4ExactCodec().fit(samples)
    codec.save(output_path)
    return sum(sample.numel() for sample in samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("output")
    arguments = parser.parse_args()
    count = fit_calibration_archive(arguments.archive, arguments.output)
    print(f"fitted {count} elements -> {arguments.output}")
