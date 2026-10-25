from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class SemanticCodecConfig:
    """Configuration for the V2 split-boundary activation codec."""

    hidden_size: int = 5120
    group_size: int = 128
    quant_max: int = 15
    max_rank_delta: int = 1023
    calibration_path: str | None = None


def _bf16_rank_tables() -> tuple[np.ndarray, np.ndarray]:
    codes = np.arange(65536, dtype=np.uint16)
    vals = (codes.astype(np.uint32) << 16).view(np.float32)
    finite = np.isfinite(vals)
    order = np.argsort(vals[finite], kind="stable")
    finite_codes = codes[finite][order]
    rank = np.full(65536, -1, dtype=np.int32)
    rank[finite_codes] = np.arange(len(finite_codes), dtype=np.int32)
    return rank, finite_codes


class HadamardGroupAnchor:
    """Hadamard transform plus per-token/group fp16 scales and int5 codes."""

    def __init__(self, config: SemanticCodecConfig = SemanticCodecConfig()):
        self.config = config
        assert config.hidden_size % config.group_size == 0
        assert config.hidden_size % 1024 == 0
        self._matrices: dict[tuple[str, int], torch.Tensor] = {}

    def _matrix(self, device: torch.device) -> torch.Tensor:
        n = 1024
        key = (str(device), n)
        if key not in self._matrices:
            matrix = np.asarray([[1.0]], dtype=np.float32)
            while matrix.shape[0] < n:
                matrix = np.block([[matrix, matrix], [matrix, -matrix]]).astype(np.float32)
            matrix /= math.sqrt(n)
            self._matrices[key] = torch.from_numpy(matrix).to(device=device)
        return self._matrices[key]

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        shape = tensor.shape
        flat = tensor.float().reshape(-1, self.config.hidden_size)
        n, width = flat.shape
        matrix = self._matrix(tensor.device)
        transformed = matrix @ flat.reshape(n, width // 1024, 1024, 1)
        return transformed.squeeze(-1).reshape(shape)

    def encode(self, tensor: torch.Tensor) -> tuple[bytes, torch.Tensor]:
        if tensor.dtype != torch.bfloat16 or tensor.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected BF16 [..., {self.config.hidden_size}], got {tuple(tensor.shape)} {tensor.dtype}"
            )
        old_shape = tensor.shape
        n = int(np.prod(old_shape[:-1]))
        z = self.forward(tensor).reshape(n, -1, self.config.group_size)
        scale = z.abs().amax(dim=-1, keepdim=True)
        scale = (scale / self.config.quant_max).clamp(min=1e-6).to(torch.float16).float()
        q = torch.clamp(torch.round(z / scale), -self.config.quant_max, self.config.quant_max)
        prediction = self.forward((q * scale).reshape(old_shape)).to(torch.bfloat16)

        q_np = q.detach().cpu().numpy().astype(np.int8)
        biased = (q_np.astype(np.uint16) + 16).astype(np.uint8)
        bits = np.unpackbits(biased[..., None], axis=-1, bitorder="big")[..., -5:]
        anchor = np.packbits(bits.reshape(-1), bitorder="big").tobytes()
        anchor += scale.detach().cpu().numpy().astype("<f2").tobytes()
        expected = n * self.config.hidden_size * 5 // 8 + n * (self.config.hidden_size // self.config.group_size) * 2
        assert len(anchor) == expected
        return anchor, prediction

    def decode(self, payload: bytes, shape: torch.Size, device: torch.device) -> torch.Tensor:
        n = int(np.prod(shape[:-1]))
        ngroup = self.config.hidden_size // self.config.group_size
        qbytes = len(payload) - n * ngroup * 2
        expected = n * self.config.hidden_size * 5 // 8 + n * ngroup * 2
        if len(payload) != expected:
            raise ValueError(f"bad anchor size {len(payload)} != {expected}")
        raw = np.unpackbits(np.frombuffer(payload[:qbytes], np.uint8), bitorder="big")
        if raw.shape[0] < n * self.config.hidden_size * 5:
            raise ValueError("truncated anchor bitstream")
        bits = raw[: n * self.config.hidden_size * 5].reshape(-1, 5)
        vals = np.zeros(bits.shape[0], dtype=np.uint8)
        for bit, weight in enumerate((16, 8, 4, 2, 1)):
            vals += bits[:, bit].astype(np.uint8) * weight
        q = vals.reshape(n, self.config.hidden_size).astype(np.int16) - 16
        scale_np = np.frombuffer(payload[qbytes:], dtype="<f2").reshape(n, ngroup, 1).copy()
        scale = torch.from_numpy(scale_np.astype(np.float32)).to(device=device)
        qt = torch.from_numpy(q.astype(np.float32)).to(device=device)
        z = (qt.reshape(n, ngroup, self.config.group_size) * scale).reshape(n, self.config.hidden_size)
        prediction = self.forward(z.reshape(shape)).to(torch.bfloat16)
        return prediction


class _BitWriter:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, bits: str) -> None:
        self.parts.append(bits)

    def bytes(self) -> bytes:
        joined = "".join(self.parts)
        joined += "0" * ((-len(joined)) % 8)
        return np.packbits(
            np.frombuffer(joined.encode("ascii"), np.uint8).astype(np.uint8) - ord("0"),
            bitorder="big",
        ).tobytes()


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self.bits = np.unpackbits(np.frombuffer(data, np.uint8), bitorder="big").tolist()
        self.pos = 0

    def bit(self) -> str:
        if self.pos >= len(self.bits):
            raise ValueError("semantic residual bitstream underflow")
        value = self.bits[self.pos]
        self.pos += 1
        return str(value)

    def raw16(self) -> int:
        return int("".join(self.bit() for _ in range(16)), 2)


def _canonical_huffman(probability: np.ndarray) -> list[str]:
    weights = np.maximum(probability.astype(np.float64) * 1_000_000_000.0, 1.0)
    heap = [(int(weight), symbol, (symbol,)) for symbol, weight in enumerate(weights)]
    heapq.heapify(heap)
    lengths = [0] * len(probability)
    while len(heap) > 1:
        weight1, _, leaves1 = heapq.heappop(heap)
        weight2, _, leaves2 = heapq.heappop(heap)
        for leaf in leaves1 + leaves2:
            lengths[leaf] += 1
        heapq.heappush(heap, (weight1 + weight2, len(probability), leaves1 + leaves2))
    if max(lengths) > 32:
        raise RuntimeError("canonical Huffman code exceeds 32 bits")
    pairs = sorted(
        ((length, symbol) for symbol, length in enumerate(lengths) if length),
        key=lambda pair: (pair[0], pair[1]),
    )
    codes: list[str] = [""] * len(probability)
    value, previous_length = 0, pairs[0][0] if pairs else 0
    for length, symbol in pairs:
        value <<= length - previous_length
        codes[symbol] = format(value, f"0{length}b")
        value += 1
        previous_length = length
    return codes


class ExactRankDeltaResidual:
    """Context-conditioned Huffman residual with an exact raw-BF16 escape."""

    def __init__(self, config: SemanticCodecConfig):
        self.config = config
        self.rank_table, self.finite_codes = _bf16_rank_tables()
        self.num_symbols = 2 * config.max_rank_delta + 2
        self.escape_symbol = self.num_symbols - 1
        self.tables: list[list[str]] | None = None
        self.decoders: list[dict[str, int]] | None = None
        if config.calibration_path:
            self.load(Path(config.calibration_path))

    @staticmethod
    def _codes(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().contiguous().view(torch.uint16).numpy().astype(np.uint16)

    def _symbols(self, prediction: torch.Tensor, exact: torch.Tensor):
        x = self._codes(exact).reshape(-1)
        p = self._codes(prediction).reshape(-1)
        rank_x, rank_p = self.rank_table[x], self.rank_table[p]
        delta = rank_x.astype(np.int64) - rank_p.astype(np.int64)
        symbols = np.full(len(delta), self.escape_symbol, dtype=np.uint16)
        representable = (np.abs(delta) <= self.config.max_rank_delta) & (rank_x >= 0) & (rank_p >= 0)
        symbols[representable] = (delta[representable] + self.config.max_rank_delta).astype(np.uint16)
        context = ((p.astype(np.uint32) >> 7) & 0xFF).astype(np.int64)
        return symbols, x, p, context, representable

    def fit(self, samples: Iterable[torch.Tensor]) -> "ExactRankDeltaResidual":
        anchor = HadamardGroupAnchor(self.config)
        counts = np.zeros((256, self.num_symbols), dtype=np.float64)
        elements = 0
        for sample in samples:
            sample = sample.detach().cpu().to(torch.bfloat16)
            _, prediction = anchor.encode(sample)
            symbols, _, predicted_codes, context, _ = self._symbols(prediction, sample)
            flat = context * self.num_symbols + symbols.astype(np.int64)
            counts += np.bincount(flat, minlength=256 * self.num_symbols).reshape(256, self.num_symbols)
            elements += len(symbols)
        if elements == 0:
            raise ValueError("semantic residual calibration is empty")
        global_counts = counts.sum(axis=0)
        global_probability = (global_counts + 1.0 / self.num_symbols) / (global_counts.sum() + 1.0)
        beta = 64.0
        self.tables, self.decoders = [], []
        for context in range(256):
            probability = (counts[context] + beta * global_probability) / (counts[context].sum() + beta)
            codes = _canonical_huffman(probability)
            self.tables.append(codes)
            self.decoders.append({code: symbol for symbol, code in enumerate(codes)})
        return self

    def save(self, path: str | Path) -> None:
        if self.tables is None:
            raise RuntimeError("cannot save an unfitted residual coder")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, tables=np.asarray(self.tables, dtype=str))

    def load(self, path: str | Path) -> "ExactRankDeltaResidual":
        with np.load(Path(path), allow_pickle=False) as archive:
            tables = archive["tables"].astype(str)
        if tables.shape != (256, self.num_symbols):
            raise ValueError(f"bad coder table shape {tables.shape}")
        self.tables = tables.tolist()
        self.decoders = [{code: symbol for symbol, code in enumerate(row)} for row in self.tables]
        return self

    def encode(self, prediction: torch.Tensor, exact: torch.Tensor) -> tuple[bytes, float]:
        if self.tables is None:
            raise RuntimeError("semantic residual coder is not calibrated")
        symbols, _, _, contexts, representable = self._symbols(prediction, exact)
        writer = _BitWriter()
        for context, symbol, represented, code in zip(contexts, symbols, representable, self._codes(exact).reshape(-1)):
            writer.write(self.tables[int(context)][int(symbol)])
            if not represented:
                writer.write(format(int(code), "016b"))
        return writer.bytes(), float(1.0 - representable.mean())

    def decode(self, prediction: torch.Tensor, payload: bytes) -> torch.Tensor:
        if self.decoders is None:
            raise RuntimeError("semantic residual coder is not calibrated")
        shape = prediction.shape
        predicted_codes = self._codes(prediction).reshape(-1)
        contexts = ((predicted_codes.astype(np.uint32) >> 7) & 0xFF).astype(np.int64)
        reader = _BitReader(payload)
        output = np.empty(len(predicted_codes), dtype=np.uint16)
        for index in range(len(predicted_codes)):
            prefix, symbol, decoder = "", None, self.decoders[int(contexts[index])]
            while symbol is None:
                prefix += reader.bit()
                symbol = decoder.get(prefix)
                if len(prefix) > 32:
                    raise ValueError("invalid semantic Huffman prefix")
            if symbol == self.escape_symbol:
                output[index] = reader.raw16()
            else:
                rank_prediction = int(self.rank_table[int(predicted_codes[index])])
                rank_exact = rank_prediction + symbol - self.config.max_rank_delta
                if not 0 <= rank_exact < len(self.finite_codes):
                    raise ValueError("decoded BF16 rank is out of range")
                output[index] = self.finite_codes[rank_exact]
        tensor = torch.from_numpy(output.copy()).view(torch.bfloat16).reshape(shape)
        return tensor.to(device=prediction.device, dtype=torch.bfloat16)


class SemanticBoundaryCodec:
    """Pack and bitwise-unpack every BF16 tensor at a V2 PP boundary."""

    def __init__(self, config: SemanticCodecConfig | None = None):
        self.config = config or SemanticCodecConfig()
        self.anchor = HadamardGroupAnchor(self.config)
        self.residual = ExactRankDeltaResidual(self.config)

    def encode(self, key: str, tensor: torch.Tensor) -> tuple[bytes, bytes, float]:
        anchor_payload, prediction = self.anchor.encode(tensor)
        residual_payload, escape_rate = self.residual.encode(prediction, tensor)
        return anchor_payload, residual_payload, escape_rate

    def decode(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        anchor_payload: bytes,
        residual_payload: bytes,
    ) -> torch.Tensor:
        if dtype != torch.bfloat16:
            raise TypeError("semantic decode requires BF16")
        prediction = self.anchor.decode(anchor_payload, shape, device)
        return self.residual.decode(prediction, residual_payload)


def fit_calibration_archive(archive_path: str | Path, output_path: str | Path) -> int:
    """Fit residual tables from a V2 activation-code NPZ archive."""
    with np.load(Path(archive_path), allow_pickle=False) as archive:
        samples = []
        for key in archive.files:
            codes = archive[key]
            expected = codes.size // SemanticCodecConfig().hidden_size
            samples.append(
                torch.from_numpy(codes.reshape(-1).astype(np.int16).copy())
                .view(torch.bfloat16)
                .reshape(expected, SemanticCodecConfig().hidden_size)
            )
        config = SemanticCodecConfig()
        ExactRankDeltaResidual(config).fit(samples).save(output_path)
        return sum(sample.numel() for sample in samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("output")
    args = parser.parse_args()
    count = fit_calibration_archive(args.archive, args.output)
    print(f"fit {count} elements -> {args.output}")
