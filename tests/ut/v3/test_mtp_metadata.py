from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.v3.mtp_metadata import accepted_draft_metadata


def test_padded_mtp_correction_preserves_original_and_separate_mirrors():
    lengths = torch.tensor([20, 100], dtype=torch.int32)
    mirror = lengths.clone()
    metadata = SimpleNamespace(
        seq_lens=lengths,
        seq_lens_cpu=mirror,
        _seq_lens_cpu=mirror,
        seq_lens_cpu_upper_bound=mirror,
    )
    result = accepted_draft_metadata(metadata, torch.tensor([5], dtype=torch.int32), batch_size=1)
    assert result.seq_lens.tolist() == [15, 100]
    assert result.seq_lens_cpu.tolist() == [15, 100]
    assert result._seq_lens_cpu.tolist() == [15, 100]
    assert result.seq_lens_cpu_upper_bound.tolist() == [15, 100]
    assert metadata.seq_lens.tolist() == [20, 100]
    assert metadata.seq_lens_cpu.tolist() == [20, 100]
    result.seq_lens_cpu[0] += 1
    assert result._seq_lens_cpu[0] == 15


def test_padded_mtp_correction_rejects_invalid_lengths():
    metadata = SimpleNamespace(seq_lens=torch.tensor([4]), seq_lens_cpu=torch.tensor([4]))
    for rejected in (torch.tensor([-1]), torch.tensor([4]), torch.tensor([])):
        with pytest.raises(ValueError):
            accepted_draft_metadata(metadata, rejected, batch_size=1)
