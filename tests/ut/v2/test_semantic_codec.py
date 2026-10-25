from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from vllm_ascend.v2.semantic_codec import SemanticBoundaryCodec, SemanticCodecConfig


def test_semantic_boundary_is_bitwise_exact():
    generator = torch.Generator().manual_seed(7)
    calibration = torch.randn((4, 5120), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    target = torch.randn((3, 5120), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tables.npz"
        codec = SemanticBoundaryCodec(SemanticCodecConfig())
        codec.residual.fit([calibration]).save(path)
        # A transport-created codec loads the pre-shared tables from disk.
        SemanticBoundaryCodec(SemanticCodecConfig(calibration_path=str(path)))
        anchor, residual, _ = codec.encode("hidden_states", target)
        recovered = codec.decode(target.shape, target.dtype, target.device, anchor, residual)
        assert torch.equal(recovered, target)
        assert len(anchor) + len(residual) < target.numel() * 2
