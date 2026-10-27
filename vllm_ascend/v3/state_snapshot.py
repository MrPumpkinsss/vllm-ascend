"""Indexed snapshots of the cache pages touched by one verification batch."""

from __future__ import annotations

from typing import Any

import torch


class UnsupportedStateSnapshot(RuntimeError):
    """The current attention/cache layout cannot be snapshotted safely."""


def tensors_bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare tensor representations, including signed zero and NaN payloads."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_bytes = left.detach().cpu().contiguous().view(torch.uint8)
    right_bytes = right.detach().cpu().contiguous().view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


class IndexedStateSnapshot:
    """Preserve touched KV, convolution and recurrent pages in their original storage."""

    def __init__(self, regions: list[tuple[torch.Tensor, torch.Tensor]], labels: list[str] | None = None):
        if not regions:
            raise UnsupportedStateSnapshot("no cache regions to snapshot")
        if labels is not None and len(labels) != len(regions):
            raise ValueError("cache-region labels must match the region count")
        self._labels = labels or [str(index) for index in range(len(regions))]
        self._copies: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for tensor, indices in regions:
            if tensor.ndim == 0 or indices.ndim != 1:
                raise UnsupportedStateSnapshot("unsupported cache or index shape")
            cpu_indices = indices.detach().to(device="cpu", dtype=torch.long)
            cpu_indices = torch.unique(cpu_indices[cpu_indices >= 0])
            if not cpu_indices.numel():
                raise UnsupportedStateSnapshot("no active cache page indices")
            if int(cpu_indices.max()) >= tensor.shape[0]:
                raise UnsupportedStateSnapshot("cache index is out of bounds")
            device_indices = cpu_indices.to(device=tensor.device)
            values = torch.index_select(tensor, 0, device_indices).clone()
            self._copies.append((tensor, device_indices, values))

    def restore(self) -> None:
        for tensor, indices, values in self._copies:
            tensor.index_copy_(0, indices, values)

    def matches(self) -> bool:
        for tensor, indices, values in self._copies:
            current = torch.index_select(tensor, 0, indices).detach().cpu().contiguous().view(torch.uint8)
            original = values.detach().cpu().contiguous().view(torch.uint8)
            if not torch.equal(current, original):
                return False
        return True

    def differences(self, limit: int = 4) -> list[dict[str, Any]]:
        differences: list[dict[str, Any]] = []
        for label, (tensor, indices, values) in zip(self._labels, self._copies):
            current = torch.index_select(tensor, 0, indices).detach().cpu().contiguous()
            original = values.detach().cpu().contiguous()
            changed = (current.view(torch.uint8) != original.view(torch.uint8)).reshape(len(indices), -1).any(dim=1)
            if changed.any():
                differences.append({
                    "region": label,
                    "pages": indices.detach().cpu()[changed].tolist(),
                    "max_abs": float((current.float() - original.float()).abs().max()),
                })
                if len(differences) >= limit:
                    break
        return differences

    @property
    def region_count(self) -> int:
        return len(self._copies)


def snapshot_attention_state(
    forward_context: dict[str, Any], attn_metadata: dict[str, Any]
) -> IndexedStateSnapshot:
    """Snapshot cache rows named by GDN state indices or paged-attention slots."""
    regions: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_name, metadata in attn_metadata.items():
        layer = forward_context.get(layer_name)
        if layer is None or not hasattr(layer, "kv_cache"):
            continue
        cache = layer.kv_cache
        if not isinstance(cache, (list, tuple)) or not cache:
            raise UnsupportedStateSnapshot(f"unsupported cache layout for {layer_name}")
        if not all(isinstance(tensor, torch.Tensor) and tensor.ndim >= 2 for tensor in cache):
            raise UnsupportedStateSnapshot(f"unsupported cache tensors for {layer_name}")
        state_indices = getattr(metadata, "spec_state_indices_tensor", None)
        if state_indices is not None:
            indices = state_indices.reshape(-1)
        else:
            slot_mapping = getattr(metadata, "slot_mapping", None)
            if slot_mapping is None:
                raise UnsupportedStateSnapshot(f"missing writable-slot metadata for {layer_name}")
            block_size = cache[0].shape[1]
            if block_size <= 0 or any(tensor.shape[1] != block_size for tensor in cache):
                raise UnsupportedStateSnapshot(f"unsupported paged-cache layout for {layer_name}")
            slots = slot_mapping.reshape(-1)
            indices = torch.where(slots >= 0, slots // block_size, -1)
        regions.extend((tensor, indices) for tensor in cache)
    return IndexedStateSnapshot(regions)


def snapshot_gdn_prefix(
    forward_context: dict[str, Any], attn_metadata: dict[str, Any], prefix_length: int
) -> IndexedStateSnapshot:
    """Capture recurrent and convolution pages associated with a valid prefix."""
    regions: list[tuple[torch.Tensor, torch.Tensor]] = []
    labels: list[str] = []
    for layer_name, metadata in attn_metadata.items():
        indices = getattr(metadata, "spec_state_indices_tensor", None)
        if indices is None:
            continue
        layer = forward_context.get(layer_name)
        cache = getattr(layer, "kv_cache", None)
        if not isinstance(cache, (list, tuple)) or indices.ndim != 2 or prefix_length > indices.shape[1]:
            raise UnsupportedStateSnapshot(f"unsupported GDN prefix state for {layer_name}")
        selected = indices[:, :prefix_length].reshape(-1)
        for index, tensor in enumerate(cache):
            regions.append((tensor, selected))
            labels.append(f"{layer_name}:{index}")
    return IndexedStateSnapshot(regions, labels)


def snapshot_gdn_committed_state(
    forward_context: dict[str, Any], attn_metadata: dict[str, Any], consumed_end: int
) -> IndexedStateSnapshot:
    """Capture the GDN conv window and recurrent page read after acceptance."""
    if consumed_end < 1:
        raise UnsupportedStateSnapshot("a committed GDN prefix must contain a token")
    regions: list[tuple[torch.Tensor, torch.Tensor]] = []
    labels: list[str] = []
    for layer_name, metadata in attn_metadata.items():
        indices = getattr(metadata, "spec_state_indices_tensor", None)
        if indices is None:
            continue
        layer = forward_context.get(layer_name)
        cache = getattr(layer, "kv_cache", None)
        convolution = getattr(layer, "conv1d", None)
        weight = getattr(convolution, "weight", None)
        if (
            not isinstance(cache, (list, tuple)) or len(cache) != 2
            or indices.ndim != 2 or indices.shape[0] != 1 or consumed_end > indices.shape[1]
            or not isinstance(weight, torch.Tensor)
        ):
            raise UnsupportedStateSnapshot(f"unsupported committed GDN state for {layer_name}")
        window = weight.shape[-1] - 1
        start = consumed_end - 1
        if window < 1 or cache[0].ndim != 3 or start + window > cache[0].shape[1]:
            raise UnsupportedStateSnapshot(f"unsupported GDN convolution window for {layer_name}")
        regions.append((cache[0][:, start : start + window], indices[:, 0]))
        labels.append(f"{layer_name}:conv[{start}:{start + window}]")
        regions.append((cache[1], indices[:, start]))
        labels.append(f"{layer_name}:recurrent[{start}]")
    return IndexedStateSnapshot(regions, labels)


def snapshot_attention_committed_state(
    forward_context: dict[str, Any], attn_metadata: dict[str, Any], consumed_end: int
) -> IndexedStateSnapshot:
    """Capture only the paged-attention slots belonging to a consumed prefix."""
    if consumed_end < 1:
        raise UnsupportedStateSnapshot("a committed attention prefix must contain a token")
    regions: list[tuple[torch.Tensor, torch.Tensor]] = []
    labels: list[str] = []
    for layer_name, metadata in attn_metadata.items():
        if getattr(metadata, "spec_state_indices_tensor", None) is not None:
            continue
        layer = forward_context.get(layer_name)
        cache = getattr(layer, "kv_cache", None)
        if layer is None or cache is None:
            continue
        slot_mapping = getattr(metadata, "slot_mapping", None)
        if (
            not isinstance(cache, (list, tuple)) or not cache
            or not isinstance(slot_mapping, torch.Tensor)
            or slot_mapping.numel() < consumed_end
            or not all(isinstance(tensor, torch.Tensor) and tensor.ndim >= 2 for tensor in cache)
        ):
            raise UnsupportedStateSnapshot(f"unsupported committed attention state for {layer_name}")
        slots = slot_mapping.reshape(-1)[:consumed_end].detach().to(device="cpu", dtype=torch.long)
        block_size = cache[0].shape[1]
        if block_size <= 0 or any(tensor.shape[1] != block_size for tensor in cache) or (slots < 0).any():
            raise UnsupportedStateSnapshot(f"unsupported committed attention slots for {layer_name}")
        pages = slots // block_size
        offsets = slots % block_size
        for page in torch.unique(pages).tolist():
            for index, tensor in enumerate(cache):
                if page >= tensor.shape[0]:
                    raise UnsupportedStateSnapshot(f"attention page is out of bounds for {layer_name}")
                regions.append((tensor[page], offsets[pages == page]))
                labels.append(f"{layer_name}:{index}:page[{page}]")
    return IndexedStateSnapshot(regions, labels)


def snapshot_attention_suffix_state(
    forward_context: dict[str, Any], attn_metadata: dict[str, Any], suffix_start: int
) -> IndexedStateSnapshot:
    """Snapshot GDN state plus only paged-attention slots written by a fused suffix."""
    if suffix_start < 1:
        raise UnsupportedStateSnapshot("a fused suffix must start after the committed prefix")
    regions: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_name, metadata in attn_metadata.items():
        layer = forward_context.get(layer_name)
        if layer is None or not hasattr(layer, "kv_cache"):
            continue
        cache = layer.kv_cache
        if not isinstance(cache, (list, tuple)) or not cache:
            raise UnsupportedStateSnapshot(f"unsupported cache layout for {layer_name}")
        if not all(isinstance(tensor, torch.Tensor) and tensor.ndim >= 2 for tensor in cache):
            raise UnsupportedStateSnapshot(f"unsupported cache tensors for {layer_name}")
        state_indices = getattr(metadata, "spec_state_indices_tensor", None)
        if state_indices is not None:
            indices = state_indices.reshape(-1)
            regions.extend((tensor, indices) for tensor in cache)
            continue
        slot_mapping = getattr(metadata, "slot_mapping", None)
        if not isinstance(slot_mapping, torch.Tensor):
            raise UnsupportedStateSnapshot(f"missing writable-slot metadata for {layer_name}")
        slots = slot_mapping.reshape(-1).detach().to(device="cpu", dtype=torch.long)
        if slots.numel() < suffix_start or (slots[suffix_start:] < 0).any():
            raise UnsupportedStateSnapshot(f"invalid fused suffix slots for {layer_name}")
        suffix_slots = slots[suffix_start:]
        block_size = cache[0].shape[1]
        if block_size <= 0 or any(tensor.shape[1] != block_size for tensor in cache):
            raise UnsupportedStateSnapshot(f"unsupported paged-cache layout for {layer_name}")
        pages = suffix_slots // block_size
        offsets = suffix_slots % block_size
        for page in torch.unique(pages).tolist():
            if page >= cache[0].shape[0]:
                raise UnsupportedStateSnapshot(f"attention page is out of bounds for {layer_name}")
            selected_offsets = offsets[pages == page].to(device=cache[0].device)
            for tensor in cache:
                regions.append((tensor[page], selected_offsets))
    return IndexedStateSnapshot(regions)
