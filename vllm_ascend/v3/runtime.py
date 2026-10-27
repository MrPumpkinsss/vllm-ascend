from __future__ import annotations

# ruff: noqa: E402

"""Runtime ledger for V3 Prefix-Repair, SOTS, and coarse Ready-Flow.

The module is deliberately transport/model agnostic.  A vLLM scheduler can use
it as the single source of truth for boundary events, verdict-adaptive residual
requests, and depth-one speculative leases without exposing approximate state
through the commit API.
"""


import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class V3EventKind(str, Enum):
    BOUNDARY_OPEN = "boundary_open"
    ANCHOR_READY = "anchor_ready"
    APPROXIMATE_DV_DONE = "approximate_dv_done"
    RESIDUAL_REQUEST = "residual_request"
    RESIDUAL_READY = "residual_ready"
    EXACT_DECODE_DONE = "exact_decode_done"
    VV_DONE = "vv_done"
    COMMIT = "commit"
    FLUSH = "flush"
    A_RUNAHEAD_ADMIT = "a_runahead_admit"
    A_RUNAHEAD_CANCEL = "a_runahead_cancel"


@dataclass(frozen=True)
class TrajectoryKey:
    model_version: str
    cut_point: int
    dtype: str
    k: int
    batch_composition: str
    fused_or_separate: str
    kernel_id: str
    execution_mode: str
    activation_codec_id: str

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass
class V3Event:
    epoch: int
    kind: V3EventKind
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommitRecord:
    epoch: int
    accepted_drafts: int
    committed_tokens: list[int]
    next_root: int
    consumed_end: int
    trajectory: str
    boundary_version: int


class EventRecorder:
    """Append-only JSONL/JSON event trace with bounded in-memory history."""

    def __init__(self, path: str | Path | None = None, keep_events: int = 100_000):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        else:
            self._handle = None
        self.events: deque[V3Event] = deque(maxlen=keep_events)

    def record(self, epoch: int, kind: V3EventKind, **data: Any) -> V3Event:
        event = V3Event(epoch=epoch, kind=kind, timestamp=time.perf_counter(), data=data)
        self.events.append(event)
        if self._handle is not None:
            self._handle.write(json.dumps(asdict(event), separators=(",", ":")) + "\n")
            self._handle.flush()
        return event

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def dump(self) -> list[dict[str, Any]]:
        return [asdict(event) for event in self.events]


@dataclass
class PrefixPolicy:
    """Verdict-adaptive residual prefix policy."""

    k: int = 8
    short_prefix: int = 4
    long_prefix: int = 6
    safety_window: int = 1
    min_short_streak: int = 3
    recent_consumed: deque[int] = field(default_factory=lambda: deque(maxlen=8))
    requests: int = 0
    suffix_requests: int = 0

    def observe(self, consumed_end: int) -> None:
        self.recent_consumed.append(int(consumed_end))

    def request_length(self, predicted_end: int | None = None) -> int:
        self.requests += 1
        predicted = int(predicted_end or 0)
        if predicted:
            return min(self.k, max(predicted + self.safety_window, self.short_prefix))
        if len(self.recent_consumed) >= self.min_short_streak:
            recent_max = max(self.recent_consumed)
            if recent_max + self.safety_window <= self.short_prefix:
                return self.short_prefix
            if recent_max + self.safety_window <= self.long_prefix:
                return self.long_prefix
        return self.k

    def observe_request(self, requested: int, needed: int) -> bool:
        needed = int(needed)
        if needed > requested:
            self.suffix_requests += 1
            self.observe(needed)
            return False
        self.observe(needed)
        return True

    def stats(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "suffix_requests": self.suffix_requests,
            "suffix_request_rate": self.suffix_requests / self.requests if self.requests else 0.0,
            "recent_consumed": list(self.recent_consumed),
        }


@dataclass(frozen=True)
class PrefixCostDecision:
    prefix_length: int
    expected_cost_ms: dict[int, float]
    miss_probability: dict[int, float]
    history_size: int
    history_source: str


@dataclass
class NetworkAwarePrefixPolicy:
    """Choose exact-prefix bytes from past VV outcomes, never from future verdicts."""

    bandwidth_mbps: float
    rtt_ms: float
    retry_ms: float
    k: int = 8
    candidate_prefixes: tuple[int, ...] = (4, 6, 8)
    min_history: int = 8
    history_size: int = 128
    prior_misses: float = 1.0
    prior_hits: float = 1.0
    history: deque[int] = field(default_factory=deque, init=False)
    predicted_history: dict[int, deque[int]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        values = (self.bandwidth_mbps, self.rtt_ms, self.retry_ms, self.prior_misses, self.prior_hits)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("network policy parameters must be finite")
        if self.bandwidth_mbps <= 0 or min(self.rtt_ms, self.retry_ms, self.prior_misses, self.prior_hits) < 0:
            raise ValueError("invalid network policy parameters")
        if self.k <= 0 or self.min_history <= 0 or self.history_size < self.min_history:
            raise ValueError("invalid policy history or K")
        prefixes = tuple(sorted(set(self.candidate_prefixes)))
        if not prefixes or prefixes[-1] != self.k or prefixes[0] < 1:
            raise ValueError("candidate prefixes must be in 1..K and include K")
        self.candidate_prefixes = prefixes
        self.history = deque(maxlen=self.history_size)

    def observe(self, consumed_end: int, predicted_end: int | None = None) -> None:
        if not 1 <= consumed_end <= self.k:
            raise ValueError("consumed end must be in 1..K")
        if predicted_end is not None and not 1 <= predicted_end <= self.k:
            raise ValueError("predicted end must be in 1..K")
        self.history.append(consumed_end)
        if predicted_end is not None:
            history = self.predicted_history.setdefault(predicted_end, deque(maxlen=self.history_size))
            history.append(consumed_end)

    def choose(self, row_bytes: list[int] | tuple[int, ...], predicted_end: int | None = None) -> PrefixCostDecision:
        if len(row_bytes) != self.k or any(size < 0 for size in row_bytes):
            raise ValueError("row byte estimates must contain K nonnegative values")
        if predicted_end is not None and not 1 <= predicted_end <= self.k:
            raise ValueError("predicted end must be in 1..K")
        history = self.predicted_history.get(predicted_end) if predicted_end is not None else None
        if history is not None and len(history) >= self.min_history:
            source = "predicted"
        else:
            history = self.history
            source = "global"
        if len(history) < self.min_history:
            return PrefixCostDecision(self.k, {}, {}, len(history), "cold")

        cost_ms: dict[int, float] = {}
        miss_probability: dict[int, float] = {}
        byte_ms = 8.0 / (self.bandwidth_mbps * 1000.0)
        for prefix in self.candidate_prefixes:
            if prefix == self.k:
                miss = 0.0
            else:
                misses = sum(consumed > prefix for consumed in history)
                miss = (misses + self.prior_misses) / (len(history) + self.prior_misses + self.prior_hits)
            initial_ms = sum(row_bytes[:prefix]) * byte_ms
            suffix_ms = sum(row_bytes[prefix:]) * byte_ms
            cost_ms[prefix] = initial_ms + miss * (suffix_ms + self.rtt_ms + self.retry_ms)
            miss_probability[prefix] = miss
        best = min(self.candidate_prefixes, key=lambda prefix: (cost_ms[prefix], -prefix))
        return PrefixCostDecision(best, cost_ms, miss_probability, len(history), source)


@dataclass
class _Epoch:
    epoch: int
    trajectory: str
    approximate_chain: list[int] | None = None
    approximate_state_ready: bool = False
    exact_state_ready: bool = False
    committed: bool = False


class SingleOwnerTransactionalState:
    """SOTS ledger enforcing B-side exact-VV single ownership.

    The class intentionally stores logical metadata only.  Numeric hidden/KV or
    recurrent state is never promoted from an approximate tensor; callers must
    associate numeric state with the returned trajectory key.
    """

    def __init__(self, recorder: EventRecorder | None = None):
        self.recorder = recorder
        self.epoch_counter = 0
        self.current: _Epoch | None = None
        self.committed_boundary_version = -1
        self.commits: list[CommitRecord] = []

    def begin(self, trajectory: TrajectoryKey) -> int:
        if self.current is not None and not self.current.committed:
            self.flush("superseded")
        epoch = self.epoch_counter
        self.epoch_counter += 1
        self.current = _Epoch(epoch=epoch, trajectory=trajectory.key())
        self._record(V3EventKind.BOUNDARY_OPEN, trajectory=trajectory.key())
        return epoch

    def anchor_ready(self, epoch: int, approximate_chain: list[int]) -> None:
        self._require(epoch)
        assert self.current is not None
        self.current.approximate_chain = list(approximate_chain)
        self.current.approximate_state_ready = True
        self._record(V3EventKind.ANCHOR_READY, approximate_chain=list(approximate_chain))

    def exact_ready(self, epoch: int) -> None:
        self._require(epoch)
        assert self.current is not None
        self.current.exact_state_ready = True
        self._record(V3EventKind.EXACT_DECODE_DONE)

    def commit(
        self,
        epoch: int,
        accepted_drafts: int,
        committed_tokens: list[int],
        next_root: int,
        consumed_end: int,
    ) -> CommitRecord:
        self._require(epoch)
        assert self.current is not None
        if not self.current.exact_state_ready:
            raise RuntimeError("cannot commit before exact VV state is ready")
        record = CommitRecord(
            epoch=epoch,
            accepted_drafts=int(accepted_drafts),
            committed_tokens=list(committed_tokens),
            next_root=int(next_root),
            consumed_end=int(consumed_end),
            trajectory=self.current.trajectory,
            boundary_version=self.committed_boundary_version + 1,
        )
        self.current.committed = True
        self.committed_boundary_version += 1
        self.commits.append(record)
        self._record(
            V3EventKind.COMMIT,
            accepted_drafts=record.accepted_drafts,
            consumed_end=record.consumed_end,
            next_root=record.next_root,
        )
        return record

    def flush(self, reason: str) -> None:
        if self.current is None or self.current.committed:
            return
        self._record(V3EventKind.FLUSH, reason=reason)
        self.current.committed = True

    def may_promote_approximate_state(self) -> bool:
        return False

    def _require(self, epoch: int) -> None:
        if self.current is None or self.current.epoch != epoch:
            raise RuntimeError(f"epoch {epoch} is not the active SOTS epoch")

    def _record(self, kind: V3EventKind, **data: Any) -> None:
        if self.recorder is None or self.current is None:
            return
        self.recorder.record(self.current.epoch, kind, **data)


class ReadyFlowController:
    """Depth-one, binary admission controller for A runahead and next anchor."""

    def __init__(self, recorder: EventRecorder | None = None, max_depth: int = 1):
        if max_depth != 1:
            raise ValueError("correctness-first V3 Ready-Flow permits depth one only")
        self.recorder = recorder
        self.active_epoch: int | None = None

    def admit(self, epoch: int, risk_bucket: str, resources_free: bool, rollback_recent: bool) -> bool:
        allowed = (
            self.active_epoch is None and risk_bucket == "HIGH_CONFIDENCE" and resources_free and not rollback_recent
        )
        if allowed:
            self.active_epoch = epoch
            if self.recorder:
                self.recorder.record(epoch, V3EventKind.A_RUNAHEAD_ADMIT, risk_bucket=risk_bucket)
        return allowed

    def resolve(self, epoch: int, committed: bool) -> None:
        if self.active_epoch != epoch:
            return
        self.active_epoch = None
        if self.recorder:
            kind = V3EventKind.COMMIT if committed else V3EventKind.A_RUNAHEAD_CANCEL
            self.recorder.record(epoch, kind)
