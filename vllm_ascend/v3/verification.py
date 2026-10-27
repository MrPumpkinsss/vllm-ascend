"""Greedy prefix verdicts for the experimental single-request V3 path."""

from __future__ import annotations

import torch


def greedy_matching_drafts(logits: torch.Tensor, draft_ids: torch.Tensor, limit: int) -> int:
    if logits.ndim != 2 or draft_ids.ndim != 1:
        raise ValueError("expected token-by-vocabulary logits and one draft chain")
    if limit < 0 or limit > len(draft_ids) or limit > logits.shape[0]:
        raise ValueError("invalid greedy prefix limit")
    predictions = logits[:limit].argmax(dim=-1).detach().cpu().tolist()
    candidates = draft_ids[:limit].detach().cpu().tolist()
    for index, (prediction, candidate) in enumerate(zip(predictions, candidates)):
        if prediction != candidate:
            return index
    return limit
