# V3 Verdict-Adaptive Split Inference (experimental)

This branch contains the first correctness-first vLLM-Ascend PP2 implementation
of the V3 bridge for Qwen3.5-family 27B + native recurrent MTP.

## Components

- `vllm_ascend.v3.nvfp4_codec`
  - First wave: signed e2m1 FP4 value plus e4m3 scale per 16 channels (4.5 bits/element).
  - Second wave: FP4-context Huffman signed BF16-rank delta plus raw-BF16 escape.
  - Residuals are independent per first-dimension row, enabling consumed-prefix repair.
  - Every transmitted row reconstructs its original BF16 bit pattern exactly.
- `vllm_ascend.v3.protocol`
  - Length-prefixed, CRC-checked `ANCHOR`, `RESIDUAL_REQUEST`, `RESIDUAL`,
    `SUFFIX_REQUEST`, and `BOUNDARY_DONE` frames.
- `vllm_ascend.v3.transport`
  - Anchor arrival is a separate event from exact residual arrival.
  - A receiver callback can run approximate DV work while residual transfer is pending.
  - Selective suffix repair is covered by unit tests.
  - The stock vLLM worker deliberately requests all rows: vLLM does not yet expose
    a VV-verdict callback before returning the activation tensor to stage B.
- `vllm_ascend.patch.platform.patch_pp_mtp`
  - Fixes PP2+MTP correctness by serializing EngineCore scheduler batches for V3.
  - Upstream PP normally keeps a batch queue of depth PP; it can schedule stage A
    before draft token ids return from the stage-B MTP proposer. Stage A then embeds
    `-1` placeholders and corrupts target state. V3 forces one in-flight scheduler
    batch and disables async scheduling for this mode.

The safe serving path is therefore:

```text
stage A: prefix target -> NVFP4 anchor + full exact residual over TCP
stage B: recover BF16 bitwise -> target VV -> native MTP K=8 drafting
commit: only normal vLLM target verification output
```

This is an exact transport and scheduler correctness bridge, not yet an E2E
throughput claim. Approximate DV, SOTS state promotion, A runahead, fused
`VV(t)+DV(t+1)`, and engine Prefix-Repair remain scheduler work.

## Build calibration

From the V2 repository's E12 activation-code archive:

```bash
python -m vllm_ascend.v3.nvfp4_codec \
  /home/m7zhang/code/v2/results/e12_qwen35_cut32_codes_n23.npz \
  /home/m7zhang/code/v2/results/e17_nvfp4_v3_tables.npz
```

The archive contains BF16 codes stored as uint16 and is only 1.5 MiB; the
resulting Huffman table is about 28 KiB and is excluded from wire-byte claims.

## Launch

```bash
bash tools/v3/launch_pp2_mtp.sh
```

The script uses PP=2, TP=1, `num_speculative_tokens=7`, and root-inclusive
`K_verify=8`. Configuration travels through `--additional-config`, not unknown
`VLLM_*` environment variables, because vLLM multiprocess workers do not
preserve arbitrary parent environment entries.

For a quick request:

```bash
curl http://127.0.0.1:31003/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35-v3-mtp","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
```

## Measured status (Ascend 910B3, 2026-10-27)

- Codec validation: 23/23 E12 blocks recovered BF16 bitwise; wire was 1,396,445 B
  versus 1,884,160 B BF16 (25.89% reduction), excluding the pre-shared table.
- PP2/no-spec completed a real OpenAI request. First boundary: 76,213 B versus
  102,400 B BF16 (25.57% reduction).
- Correctness-first PP2+MTP now completes requests. Without scheduler
  serialization, sync mode produced corrupted repeated text and async mode hung;
  native HCCL PP2+MTP also failed, showing that the bug was upstream scheduler
  coordination rather than the exact codec.
- After serialization, a 32-token code prompt produced coherent output in 9.7s.
- The 4-task × 4-prompt HTTP probe completed 512 output tokens:
  - 110 draft rounds
  - mean accepted drafts 3.755
  - mean committed length including bonus 4.755
  - full 7/7 survival 32.73%
  - median request latency 8.05s
  - 4.01 sequential output tok/s in this correctness-first Python codec prototype

The acceptance length now matches the engine-level native-MTP semantics and is
no longer the low offline/reset-KV spectrum. Current throughput is intentionally
limited by serialized PP scheduling and the Python Huffman implementation; it is
not the final V3 performance target.
