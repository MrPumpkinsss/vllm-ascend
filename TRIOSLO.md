# TrioSLO fork of vllm-ascend

This fork carries the **engine-side modifications** for the TrioSLO project:
SLO-aware batch scheduling (three SLO tiers) and PP pipeline microbatching for
CodeLlama-34B on 2x Ascend 910B3, based on **v0.23.0** (branch
`trioslo-v0.23.0`).

The RL controller, simulator, calibration and load generator live in the
separate `TrioSLO` project on the training box (`/home/m7zhang/code/TrioSLO`).
This repo is only the engine.

## Why a fork (and not runtime monkey-patching)

The first prototype patched `Scheduler.schedule()` at runtime by rewriting its
source through anchored edits (`--scheduler-cls trioslo.scheduler.TrioSLOScheduler`).
That works, but it broke against a behaviour of this very repo:

> `vllm_ascend/patch/platform/patch_balance_schedule.py` replaces
> `vllm.v1.core.sched.scheduler.Scheduler` with `BalanceScheduler`
> **globally at engine-core init**. `BalanceScheduler.schedule()` is a
> *copy* of an older upstream `schedule()`, so `inspect.getsource()` returns
> a different body than the pristine engine, and any source-anchored patch
> silently aims at the wrong code.

Engine modifications that must survive plugin patch ordering belong in this
fork, where they are plain source edits under version control. The runtime
shim in the TrioSLO project remains as the *deployment* path against the
installed wheel (see below).

## What lives here

| Change | File | Notes |
|---|---|---|
| Keep pristine `Scheduler` reachable | `patch/platform/patch_balance_schedule.py` | exports `VanillaScheduler` next to the global swap |
| PP decode-parity microbatching | `patch/platform/patch_trioslo_microbatch.py` | `TRIO_PP_MICROBATCH=2` enables it; patches the **vanilla** `Scheduler.schedule()` |

## The microbatch patch in one paragraph

With PP=2 on the V1 runner, `Scheduler.schedule()` drains every running decode
into one `SchedulerOutput`, so although `EngineCore` keeps `pp_size` batches in
flight (batch queue), the second in-flight batch is decode-empty: each decode's
next input token depends on a sample still inside batch 1. The pipeline never
holds two real microbatches and the stages run serialized — the measured 34B
decode floor (57.8 ms) is exactly the serial sum of the two stages' weight
reads. The patch partitions decode requests into `pp` parity groups and lets
only group `step % pp` decode each step, which makes consecutive in-flight
batches **disjoint** and lets the existing batch queue + async P2P actually
overlap stage0(microbatch i+1) with stage1(microbatch i). Prefill chunks are
never gated: prompt tokens have no dependency on an in-flight batch's sampled
output. A request still decodes at most once per `pp` steps — the data
dependency forces that anyway — so per-request ITL cannot get worse; the wins
are prefill throughput (each slot carries ~half the decode riders), TTFT, and
ITL at high concurrency.

Honest expectation setting (measured on this hardware): the ITL **floor at low
concurrency does not move** — it is the per-stage weight-read time and every
slot pays it regardless of how requests are split. Microbatching is a
throughput/TTFT lever, not a per-request-latency lever.

## Deployment model

The training box runs the stock `vllm-ascend==0.23.0` wheel in a shared venv
that must stay pristine. Until this fork is installed as a package of its own
(own venv or built wheel), the running engine gets these changes through the
TrioSLO runtime shim, which applies the *same* anchor edits to the live
vanilla scheduler and is kept in sync with `patch_trioslo_microbatch.py` by
the offline test suite in `TrioSLO/trioslo/tests/`.

## Branches

- `trioslo-v0.23.0` — working branch, based on upstream tag `v0.23.0`
- `upstream` remote points at `vllm-project/vllm-ascend`
