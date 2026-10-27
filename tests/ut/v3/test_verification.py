import pytest
import torch

from vllm_ascend.v3.verification import greedy_matching_drafts


def test_greedy_prefix_uses_first_rejection():
    logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 0.0], [1.0, 0.0, 4.0]])
    assert greedy_matching_drafts(logits, torch.tensor([1, 2, 2]), 3) == 1
    assert greedy_matching_drafts(logits, torch.tensor([1, 0, 2]), 2) == 2


def test_greedy_prefix_rejects_invalid_shape_and_limit():
    with pytest.raises(ValueError):
        greedy_matching_drafts(torch.zeros(2), torch.tensor([1]), 1)
    with pytest.raises(ValueError):
        greedy_matching_drafts(torch.zeros((2, 3)), torch.tensor([1]), 2)
