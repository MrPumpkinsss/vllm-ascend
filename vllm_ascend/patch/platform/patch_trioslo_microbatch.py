"""TrioSLO: PP decode-parity microbatching for the V1 engine loop.

Enabled by ``TRIO_PP_MICROBATCH=<pp_size>`` (wire-up in
``vllm_ascend/platform.py::check_and_update_config``; set it equal to
``--pipeline-parallel-size``).

Why this patch exists
---------------------
With PP > 1 on the V1 runner, ``Scheduler.schedule()`` drains every running
decode into a single ``SchedulerOutput``.  ``EngineCore`` keeps ``pp_size``
batches in flight (the batch queue), but the second in-flight batch is
decode-empty: each decode's next input token depends on a sample that is
still inside batch 1.  The pipeline therefore never holds two real
microbatches, and stage0/stage1 run serialized - on CodeLlama-34B/PP=2 the
measured decode floor (57.8 ms) is exactly the serial sum of the two
stages' weight reads.

What it does
------------
Decode requests are partitioned into ``pp`` parity groups; only group
``step % pp`` may decode in a given step.  Consecutive in-flight
``SchedulerOutput``s become disjoint, so the existing batch queue plus the
async P2P transfers actually overlap ``stage0(microbatch i+1)`` with
``stage1(microbatch i)``.

* Prefill chunks are never gated: prompt tokens have no dependency on an
  in-flight batch's sampled output, so chunked prefill may legitimately be
  scheduled in consecutive steps.
* A request still decodes at most once per ``pp`` steps - the data
  dependency forces that regardless - so per-request ITL cannot regress.
  The gains are prefill throughput (each slot carries ~half the decode
  riders, freeing token budget), TTFT, and ITL at high concurrency.
  The low-concurrency ITL floor is set by the per-stage weight-read time
  and does not move; this is a throughput/TTFT lever, not a per-request
  latency lever.
* On a parity match the gate records the engine-native hold
  (``request.next_decode_eligible_step = current_step + pp``), which lands
  exactly on the next parity-match step and guarantees the request can
  never appear in two in-flight microbatches even if parity state were
  lost.

Scope and non-goals
-------------------
* Patches the **vanilla** upstream ``Scheduler.schedule()`` (see
  ``patch_balance_schedule.py`` for why the module attribute cannot be
  trusted).  ``BalanceScheduler`` inherits the patched method; when balance
  scheduling is enabled it runs its own copied body instead, which is a DP
  feature this patch does not target.
* Speculative decoding is unsupported (the parity cadence assumes one
  sampled token per scheduled decode).
* Not idempotent-dangerous: re-importing applies at most once.
"""

import inspect
import textwrap
from typing import Any

import vllm.v1.core.sched.scheduler as _sched_mod
from vllm.logger import init_logger

logger = init_logger(__name__)

_APPLIED_ATTR = "_trio_mb_applied"

# The anchor is the engine's own (on V1 dormant) decode-cadence check inside
# the RUNNING loop of Scheduler.schedule().  It must occur exactly once in
# the dedented source of the installed vLLM, or the patch refuses to load -
# a vLLM upgrade must fail loudly at startup, never mis-schedule at runtime.
_ANCHOR = """\
        if self.current_step < request.next_decode_eligible_step:
            # V2+PP+async: enforce `pp_size` steps between same-req decodes
            # to match worker-side sampled-tokens broadcast slot ring cadence.
            req_index += 1
            continue
"""

_INSERT = """\
        if self.current_step < request.next_decode_eligible_step:
            # V2+PP+async: enforce `pp_size` steps between same-req decodes
            # to match worker-side sampled-tokens broadcast slot ring cadence.
            req_index += 1
            continue

        if self._trio_mb_gate(request):
            # TrioSLO PP microbatch: only parity group (step % pp) decodes
            # this step, so consecutive in-flight SchedulerOutputs are
            # disjoint and the batch queue (depth = pp_size) fills the PP
            # pipeline.  Prefill chunks pass the gate unconditionally.
            req_index += 1
            continue
"""


def vanilla_scheduler() -> type:
    """The REAL upstream Scheduler, whatever has replaced the module attr.

    ``patch_balance_schedule`` swaps ``_sched_mod.Scheduler`` for its
    ``BalanceScheduler`` subclass at engine-core init.  Subclassing or
    source-patching the swapped class is wrong: its ``schedule()`` is a
    copy of an older upstream body.  Prefer the exported original, then
    fall back to an MRO walk.
    """
    cls = getattr(_sched_mod, "VanillaScheduler", None)
    if cls is not None:
        return cls
    cls = _sched_mod.Scheduler
    if cls.__name__ == "Scheduler" and cls.__module__ == "vllm.v1.core.sched.scheduler":
        return cls
    for klass in cls.__mro__:
        if (
            klass.__name__ == "Scheduler"
            and klass.__module__ == "vllm.v1.core.sched.scheduler"
        ):
            return klass
    return cls


def _build_patched_schedule() -> Any:
    src = textwrap.dedent(inspect.getsource(vanilla_scheduler().schedule))
    n = src.count(_ANCHOR)
    if n != 1:
        raise RuntimeError(
            f"TrioSLO microbatch patch: anchor matched {n} times (expected 1). "
            "The installed vLLM's Scheduler.schedule() has changed shape; "
            "re-derive _ANCHOR in patch_trioslo_microbatch.py."
        )
    src = src.replace(_ANCHOR, _INSERT)
    ns: dict[str, Any] = dict(vars(_sched_mod))
    ns["__name__"] = "vllm_ascend.patch.platform.patch_trioslo_microbatch._schedule"
    exec(compile(src, "<trioslo-microbatch-schedule>", "exec"), ns)  # noqa: S102
    return ns["schedule"]


def _trio_mb_assign(self) -> int:
    """Pick the least-populated parity group for a newly-decoding request."""
    pp = self._trio_pp
    counts = [0] * pp
    parity_map = getattr(self, "_trio_mb_parity", None)
    if parity_map is None:
        parity_map = {}
    for r in self.running:
        p = parity_map.get(r.request_id)
        if p is not None and p < pp:
            counts[p] += 1
    return counts.index(min(counts))


def _trio_mb_gate(self, request) -> bool:
    """True = this request must sit out this step's microbatch."""
    pp = self._trio_pp
    if pp <= 1:
        return False
    computed = int(getattr(request, "num_computed_tokens", 0) or 0)
    prompt_len = int(getattr(request, "num_prompt_tokens", 0) or 0)
    if computed < prompt_len:
        return False  # prefill chunk: never gated
    parity_map = getattr(self, "_trio_mb_parity", None)
    if parity_map is None:
        parity_map = self._trio_mb_parity = {}
    rid = request.request_id
    par = parity_map.get(rid)
    if par is None:
        par = parity_map[rid] = _trio_mb_assign(self)
    if par != (self.current_step % pp):
        return True
    # Parity match: this decode is about to be scheduled (subject to the
    # usual budget checks).  Record the engine-native hold; it lands exactly
    # on the next parity-match step, so it can only reinforce the cadence.
    request.next_decode_eligible_step = self.current_step + pp
    return False


def apply(pp: int) -> None:
    """Patch the vanilla Scheduler with the parity gate (idempotent)."""
    cls = vanilla_scheduler()
    if int(getattr(cls, "_trio_pp", 1)) == pp and getattr(cls, _APPLIED_ATTR, False):
        return
    if pp <= 1:
        raise ValueError("apply() requires pp > 1; disable by not importing")
    cls._trio_pp = pp
    cls._trio_mb_gate = _trio_mb_gate
    cls._trio_mb_assign = _trio_mb_assign
    cls.schedule = _build_patched_schedule()
    setattr(cls, _APPLIED_ATTR, True)
    logger.info(
        "TrioSLO microbatch: patched vanilla Scheduler.schedule (pp=%d)", pp
    )


# Self-apply on import; platform.py guards the import behind the env check.
# Read the switch through the fork's envs module when running inside this
# fork, falling back to os.environ when this file is dropped onto a stock
# vllm-ascend install (whose envs dict predates TRIO_PP_MICROBATCH).
import os

_pp = 0
try:
    from vllm_ascend import envs as _envs

    _pp = int(_envs.TRIO_PP_MICROBATCH or 0)
except Exception:
    _pp = int(os.environ.get("TRIO_PP_MICROBATCH", "0") or 0)
if _pp > 1:
    apply(_pp)
