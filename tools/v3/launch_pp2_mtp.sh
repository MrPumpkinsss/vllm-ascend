#!/usr/bin/env bash
# Launch the correctness-first V3 PP2 + native-MTP bridge on Ascend 910B3.
# Usage: tools/v3/launch_pp2_mtp.sh [extra vllm-serve arguments...]
set -eo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export VLLM_USE_V1=1

MODEL=${V3_MODEL:-/home/m7zhang/models/Qwen3.8-27B}
SERVED_MODEL=${V3_SERVED_MODEL:-qwen35-v3-mtp}
CALIBRATION=${V3_CALIBRATION:-/home/m7zhang/code/v2/results/e17_nvfp4_v3_tables.npz}
API_PORT=${V3_API_PORT:-31003}
TRANSPORT_PORT=${V3_TRANSPORT_PORT:-29621}
V3_CONFIG=$(printf \
  '{"v3_verdict_adaptive":{"enabled":true,"calibration":"%s","listen":"127.0.0.1:%s","connect":"127.0.0.1:%s","prefix_repair":false,"prefix_length":8,"serialize_spec":true}}' \
  "$CALIBRATION" "$TRANSPORT_PORT" "$TRANSPORT_PORT")
SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":7}'

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --host 127.0.0.1 --port "$API_PORT" \
  --pipeline-parallel-size 2 --tensor-parallel-size 1 \
  --max-model-len 4096 --max-num-seqs 1 \
  --additional-config "$V3_CONFIG" \
  --speculative-config "$SPEC_CONFIG" \
  --hf-overrides '{"language_model_only":true}' \
  "$@"
