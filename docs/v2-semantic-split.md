from pathlib import Path
p=Path('vllm_ascend/v2/semantic_codec.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''        config = SemanticCodecConfig(calibration_path=str(output_path))
        ExactRankDeltaResidual(config).fit(samples).save(output_path)
''','''        config = SemanticCodecConfig()
        ExactRankDeltaResidual(config).fit(samples).save(output_path)
''')
p.write_text(s,encoding='utf-8')

p=Path('tests/ut/v2/test_semantic_codec.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''        config = SemanticCodecConfig(calibration_path=str(path))
        codec = SemanticBoundaryCodec(config)
        codec.residual.fit([calibration]).save(path)
''','''        codec = SemanticBoundaryCodec(SemanticCodecConfig())
        codec.residual.fit([calibration]).save(path)
        # A transport-created codec loads the pre-shared tables from disk.
        SemanticBoundaryCodec(SemanticCodecConfig(calibration_path=str(path)))
''')
p.write_text(s,encoding='utf-8')
@'
# V2 semantic split boundary (experimental)

This branch contains the first vLLM Ascend port of the V2 activation path.

## What is implemented

- `vllm_ascend.v2.semantic_codec`
  - Hadamard-1024 semantic anchor: per-token/per-128-channel fp16 scale plus signed int5 codes.
  - Exact context-conditioned Huffman BF16 rank-delta residual with a raw-BF16 escape.
  - Bitwise BF16 reconstruction after the residual is received.
- `vllm_ascend.v2.tcp_transport`
  - An optional, real-wire PP2 replacement for the intermediate tensor send.
  - It sends both `hidden_states` and `residual` as semantic payloads and reconstructs them before stage B.
  - Byte counts include anchor plus residual, excluding the pre-shared Huffman table.
- `vllm_ascend/worker/worker.py`
  - Enables the transport with environment variables when `--pipeline-parallel-size 2`.

This is a correctness bridge, not yet the final optimistic DV scheduler: stage B
waits for the exact residual before its normal forward. Anchor-first DV and fused
`VV(t)+DV(t+1)` remain scheduler-level work.

## Build calibration tables

From the V2 repository's E12 activation-code archive:

```bash
python -m vllm_ascend.v2.semantic_codec \
  /home/m7zhang/code/v2/results/e12_qwen35_cut32_codes_n23.npz \
  /home/m7zhang/code/v2/results/e12_semantic_residual_tables.npz
```

## Run a PP2 vLLM engine with the boundary enabled

Set the same table path in both worker environments:

```bash
export VLLM_ASCEND_V2_SEMANTIC_PP=1
export VLLM_ASCEND_V2_CALIBRATION=/home/m7zhang/code/v2/results/e12_semantic_residual_tables.npz
export VLLM_ASCEND_V2_LISTEN=127.0.0.1:29600
export VLLM_ASCEND_V2_CONNECT=127.0.0.1:29600
```

Start vLLM with `--pipeline-parallel-size 2`. Stage A connects to stage B over
the configured TCP endpoint while retaining vLLM's normal PP scheduling and MTP.
