# V3 Experimental Split Inference

This branch is a research prototype for Qwen3.5-family PP2 inference with
seven MTP drafts and an eight-row verification batch. The launch script uses
the full exact-residual path; it does not enable prefix repair by default.

The opt-in engine prefix-verification path is **not production-safe**. A
16-request greedy comparison of P4 versus P8 in the default padded/full-graph
proposer found one output-token divergence. An unpadded, draft-eager comparison
matched in that sample but changed both execution settings. Matching reconstructed
activation bytes for selected rows does not establish equivalent outputs or
downstream MTP state. The `prefix_shadow_full` diagnostic commits the
full-residual result and compares the partial path from the same saved cache
snapshot. It is intended for debugging, not benchmarking.

The opt-in `experimental_fix_mtp_padding` diagnostic adjusts the next MTP
drafting steps' sequence lengths after rejected padded rows. In one 16-request
workload, corrected P4 and P8 token IDs and acceptance traces match in both
draft-eager and full-graph modes. The correction changes the original padded
trajectory, is restricted to one request without context parallelism,
synchronizes counts to the CPU, and has no general correctness or performance
guarantee.

The opt-in `experimental_dv_mtp_shadow` path runs a provisional native MTP
proposal from the approximate DV result. It snapshots and restores draft-cache
pages and sends the uncommitted chain back to the first PP stage while the
requested residual is in flight. The receiver exposes a boundary-tagged
callback, but does not run ahead or promote the proposal. This diagnostic
requires prefix verification, `experimental_fix_mtp_padding`, one greedy
request, and an eager padded MTP proposer. It synchronizes to the CPU and is
not a performance-ready implementation. The authoritative proposer still runs
after VV, so the diagnostic duplicates draft work. Its boundary identifier is
only valid for the current serialized single-request transport.

One Qwen3.5 27B PP2 greedy workload exercised 107 early drafts and delivered
all 107 to the first stage; 97 provisional next chains matched their later
authoritative chains. The 16 requests and their acceptance traces matched a
same-settings P4 control and a no-DV unpadded-eager reference. This is a
limited diagnostic, not a general correctness or network-speedup claim.

The A-side optimistic runahead and a real fused `VV(t)+DV(t+1)` engine batch
are not implemented. Existing codec, socket, and limited engine checks do not
constitute a general exactness or performance guarantee. The experimental
socket protocol uses Python pickle and must only be used between trusted peers.

Run unit tests for codec, transport, policy, verification, and state snapshots
with `pytest tests/ut/v3/` in a compatible vLLM-Ascend environment. NPU engine
validation additionally requires the model, calibration tables, and a PP2
deployment; set the model and calibration paths in `launch_pp2_mtp.sh` or
provide their corresponding environment overrides.
