#!/usr/bin/env bash
set -euo pipefail
SERVE=$(cd "$(dirname "$0")" && pwd)
M=${MODEL_DIR:?set MODEL_DIR to the directory holding the model weights}

V41_ARGS=(--tokenizer-mode deepseek_v41 --tensor-parallel-size 8 --engram-config '{"cpu_offload":true}'
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice --mm-encoder-tp-mode data
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use_per_rank":68719476736,"lazy_offload":false}}')

case "${1:-}" in
  judge41)
    exec env VLLM_ENGINE_READY_TIMEOUT_S=3600 "${VLLM_0_31:?}" serve $M/deepseek-v4.1-flash \
      --served-model-name deepseek-v4.1-flash-judge --host 127.0.0.1 --port 3800 "${V41_ARGS[@]}" \
      --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95 --max-model-len 262144 \
      --reasoning-parser deepseek_v41 --max-cudagraph-capture-size 64 ;;
  deepseek-v41-flash)
    exec env VLLM_ENGINE_READY_TIMEOUT_S=3600 "${VLLM_0_31:?}" serve $M/deepseek-v4.1-flash \
      --served-model-name deepseek-v4.1-flash --host 127.0.0.1 --port 3800 "${V41_ARGS[@]}" \
      --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95 --max-model-len 131072 \
      --generation-config vllm --default-chat-template-kwargs '{"thinking":false}' --max-cudagraph-capture-size 64 ;;
esac

V031=(env VLLM_ENGINE_READY_TIMEOUT_S=3600 "${VLLM_0_31:-vllm}" serve)
V031_TEST=(--trust-remote-code --host 127.0.0.1 --port 3800 --tensor-parallel-size 8 --language-model-only
  --gpu-memory-utilization 0.8 --max-model-len 131072 --max-num-batched-tokens 8192 --max-num-seqs 128
  --enable-prefix-caching --enable-chunked-prefill
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
  --performance-mode throughput --optimization-level 3 --enable-auto-tool-choice --generation-config vllm
  --enable-flashinfer-autotune)
case "${1:-}" in
  ling3-flash-fin)
    exec "${V031[@]}" $M/Ling-3.0 --served-model-name Ling-3.0-flash-Fin --tool-call-parser ling3 \
      --default-chat-template-kwargs '{"enable_thinking":false}' "${V031_TEST[@]}" ;;
  minimax-m21-think|minimax-m25-think|minimax-m27-think)
    case $1 in
      minimax-m21-think) W=$M/MiniMax-M2.1 N=MiniMax-M2.1 ;;
      minimax-m25-think) W=$M/MiniMax-M2.5 N=MiniMax-M2.5 ;;
      minimax-m27-think) W=$M/MiniMax-M2.7 N=MiniMax-M2.7 ;;
    esac
    exec "${V031[@]}" $W --served-model-name $N --tool-call-parser minimax_m2 --reasoning-parser minimax_m2 \
      "${V031_TEST[@]}" ;;
  qwen36-35b-a3b)
    exec "${V031[@]}" $M/Qwen3.6-35B-A3B --served-model-name Qwen3.6-35B-A3B --tool-call-parser qwen3_xml \
      --default-chat-template-kwargs '{"enable_thinking":false}' "${V031_TEST[@]}" ;;
esac

PY=${PY_0_29:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
if [ -n "${CUDA_HOME:-}" ]; then
  export LD_LIBRARY_PATH=$CUDA_HOME/compat:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
  export PATH=$CUDA_HOME/bin:$PATH
fi
COMMON=(--trust-remote-code --port 3800 --tensor-parallel-size 8 --language-model-only
  --gpu-memory-utilization 0.8 --max-model-len 131072 --max-num-batched-tokens 8192
  --max-num-seqs 128 --enable-prefix-caching --enable-chunked-prefill
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
  --performance-mode throughput --optimization-level 3 --enable-auto-tool-choice)
AUTOTUNE=--enable-flashinfer-autotune
GENCFG=vllm

case "${1:-}" in
  qwen38-flash-next)
    exec "$PY" -m vllm.entrypoints.openai.api_server --model $M/Qwen3.8-Flash-Next \
      --served-model-name Qwen3.8-Flash-Next --port 3800 --tensor-parallel-size 8 --language-model-only \
      --gpu-memory-utilization 0.8 --max-model-len 262144 --max-num-batched-tokens 8192 --max-num-seqs 128 \
      --enable-prefix-caching --enable-chunked-prefill \
      --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
      --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' \
      --performance-mode throughput --optimization-level 3 --enable-flashinfer-autotune --generation-config vllm \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml --chat-template $SERVE/qwen38_flash_next.jinja ;;
  glm53-flash)
    PY=${PY_GLM:?}
    AUTOTUNE=--no-enable-flashinfer-autotune
    ARGS=(--model $M/GLM-5.3-Flash --served-model-name GLM-5.3-Flash
      --safetensors-load-strategy prefetch --tool-call-parser glm47
      --chat-template $SERVE/glm53_flash_nothink.jinja) ;;
  step35-flash)
    ARGS=(--model $M/Step-3.5-Flash --served-model-name Step-3.5-Flash
      --tool-call-parser step3p5 --chat-template $SERVE/step35_flash_nothink.jinja) ;;
  gpt-oss-120b|gpt-oss-20b)
    export TIKTOKEN_ENCODINGS_BASE=$M/$1
    GENCFG=$SERVE/gpt_oss_generation_config
    ARGS=(--model $M/$1 --served-model-name $1 --tool-call-parser openai
      --default-chat-template-kwargs '{"reasoning_effort":"low"}') ;;
  gemma4-31b)
    ARGS=(--model $M/gemma-4-31B-it --served-model-name gemma-4-31B-it --tool-call-parser gemma4
      --default-chat-template-kwargs '{"enable_thinking":false}') ;;
  qwen3-30b-a3b)
    ARGS=(--model $M/Qwen3-30B-A3B-Instruct-2507 --served-model-name Qwen3-30B-A3B-Instruct-2507
      --tool-call-parser hermes) ;;
  deepseek-v4-flash)
    export VLLM_DEEP_GEMM_WARMUP=full
    ARGS=(--model $M/deepseek-v4-flash --served-model-name deepseek-v4-flash
      --safetensors-load-strategy prefetch --kv-cache-dtype fp8 --block-size 256
      --tool-call-parser deepseek_v4 --default-chat-template-kwargs '{"thinking":false}') ;;
  qwen38-27b)
    ARGS=(--model $M/Qwen3.8-27B --served-model-name Qwen3.8-27B --tool-call-parser qwen3_xml
      --default-chat-template-kwargs '{"enable_thinking":false}') ;;
  *) echo "unknown model key: ${1:-}" >&2; exit 2 ;;
esac
exec "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}" "${COMMON[@]}" $AUTOTUNE --generation-config "$GENCFG"
