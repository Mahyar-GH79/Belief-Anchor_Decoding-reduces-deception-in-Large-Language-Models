#!/bin/bash
# Launch vLLM servers for local models.
# All models fit on a single RTX 5090. Run one at a time.
# Usage: bash launch_servers.sh [model_name]
#   model_name: llama | gemma | mistral | qwen32b | gemma2b | llama3b | qwen4b | qwen7b

set -e

# Model configurations
LLAMA_HF="meta-llama/Llama-3.1-8B-Instruct"
GEMMA_HF="google/gemma-2-9b-it"
MISTRAL_HF="mistralai/Mistral-Nemo-Instruct-2407"
QWEN_HF="Qwen/Qwen2.5-32B-Instruct-AWQ"
GEMMA2B_HF="google/gemma-2-2b-it"
LLAMA3B_HF="meta-llama/Llama-3.2-3B-Instruct"
QWEN4B_HF="Qwen/Qwen3-4B"
QWEN7B_HF="Qwen/Qwen2.5-7B-Instruct"
DEEPSEEK7B_HF="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PHI4MINI_HF="microsoft/Phi-4-mini-instruct"
GPTOSS20B_HF="openai/gpt-oss-20b"

LLAMA_PORT=8001
GEMMA_PORT=8002
QWEN_PORT=8003
MISTRAL_PORT=8004
GEMMA2B_PORT=8005
LLAMA3B_PORT=8006
QWEN4B_PORT=8007
QWEN7B_PORT=8008
DEEPSEEK7B_PORT=8009
PHI4MINI_PORT=8010
GPTOSS20B_PORT=8011

launch_llama() {
    echo "=== Launching Llama-3.1-8B on port $LLAMA_PORT (TP=1) ==="
    python3 -m vllm.entrypoints.openai.api_server \
        --model $LLAMA_HF \
        --port $LLAMA_PORT \
        --tensor-parallel-size 1 \
        --max-model-len 8192 \
        --dtype auto \
        --trust-remote-code &
    echo "Llama PID: $!"
}

launch_gemma() {
    echo "=== Launching Gemma-2-9B on port $GEMMA_PORT (TP=1) ==="
    python3 -m vllm.entrypoints.openai.api_server \
        --model $GEMMA_HF \
        --port $GEMMA_PORT \
        --tensor-parallel-size 1 \
        --max-model-len 8192 \
        --dtype auto \
        --trust-remote-code &
    echo "Gemma PID: $!"
}

launch_mistral() {
    echo "=== Launching Mistral-Nemo-12B on port $MISTRAL_PORT (TP=1) ==="
    python3 -m vllm.entrypoints.openai.api_server \
        --model $MISTRAL_HF \
        --port $MISTRAL_PORT \
        --tensor-parallel-size 1 \
        --max-model-len 8192 \
        --dtype auto \
        --trust-remote-code &
    echo "Mistral PID: $!"
}

launch_qwen32b() {
    echo "=== Launching Qwen2.5-32B-AWQ on port $QWEN_PORT (TP=1, 4-bit) ==="
    python3 -m vllm.entrypoints.openai.api_server \
        --model $QWEN_HF \
        --port $QWEN_PORT \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --dtype auto \
        --quantization awq \
        --trust-remote-code &
    echo "Qwen PID: $!"
}

launch_gemma2b() {
    echo "=== Launching Gemma-2-2B on port $GEMMA2B_PORT (TP=1) ==="
    python3 -m vllm.entrypoints.openai.api_server \
        --model $GEMMA2B_HF \
        --port $GEMMA2B_PORT \
        --tensor-parallel-size 1 \
        --max-model-len 8192 \
        --dtype auto \
        --trust-remote-code &
    echo "Gemma-2B PID: $!"
}

launch_llama3b() {
    echo "=== Launching Llama-3.2-3B on port $LLAMA3B_PORT (TP=1) ==="
    python3 -m vllm.entrypoints.openai.api_server \
        --model $LLAMA3B_HF \
        --port $LLAMA3B_PORT \
        --tensor-parallel-size 1 \
        --max-model-len 8192 \
        --dtype auto \
        --trust-remote-code &
    echo "Llama-3B PID: $!"
}

launch_qwen4b() {
    echo "=== Launching Qwen3-4B on port $QWEN4B_PORT (TP=1) ==="
    python3 -m vllm.entrypoints.openai.api_server \
        --model $QWEN4B_HF \
        --port $QWEN4B_PORT \
        --tensor-parallel-size 1 \
        --max-model-len 8192 \
        --dtype auto \
        --trust-remote-code &
    echo "Qwen3-4B PID: $!"
}

launch_qwen7b() {
    echo "=== Launching Qwen2.5-7B on port $QWEN7B_PORT (TP=1) ==="
    python3 -m vllm.entrypoints.openai.api_server \
        --model $QWEN7B_HF \
        --port $QWEN7B_PORT \
        --tensor-parallel-size 1 \
        --max-model-len 8192 \
        --dtype auto \
        --trust-remote-code &
    echo "Qwen2.5-7B PID: $!"
}

launch_deepseek7b() {
    echo "=== Launching DeepSeek-R1-Distill-Qwen-7B on port $DEEPSEEK7B_PORT (TP=1) ==="
    python3 -m vllm.entrypoints.openai.api_server \
        --model $DEEPSEEK7B_HF \
        --port $DEEPSEEK7B_PORT \
        --tensor-parallel-size 1 \
        --max-model-len 8192 \
        --dtype auto \
        --trust-remote-code &
    echo "DeepSeek-R1-7B PID: $!"
}

launch_phi4mini() {
    echo "=== Launching Phi-4-mini-instruct on port $PHI4MINI_PORT (TP=1) ==="
    python3 -m vllm.entrypoints.openai.api_server \
        --model $PHI4MINI_HF \
        --port $PHI4MINI_PORT \
        --tensor-parallel-size 1 \
        --max-model-len 8192 \
        --dtype auto \
        --trust-remote-code &
    echo "Phi-4-mini PID: $!"
}

launch_gptoss20b() {
    echo "=== Launching GPT-oss-20B on port $GPTOSS20B_PORT (TP=1) ==="
    echo "NOTE: GPT-oss-20B requires special vLLM build:"
    echo "  uv pip install --pre vllm==0.10.1+gptoss \\"
    echo "    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \\"
    echo "    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \\"
    echo "    --index-strategy unsafe-best-match"
    echo ""
    vllm serve $GPTOSS20B_HF \
        --port $GPTOSS20B_PORT \
        --tensor-parallel-size 1 \
        --max-model-len 8192 &
    echo "GPT-oss-20B PID: $!"
}

case "${1:-help}" in
    llama)    launch_llama ;;
    gemma)    launch_gemma ;;
    mistral)  launch_mistral ;;
    qwen32b)  launch_qwen32b ;;
    gemma2b)  launch_gemma2b ;;
    llama3b)  launch_llama3b ;;
    qwen4b)   launch_qwen4b ;;
    qwen7b)      launch_qwen7b ;;
    deepseek7b)  launch_deepseek7b ;;
    phi4mini)    launch_phi4mini ;;
    gptoss20b)   launch_gptoss20b ;;
    *)
        echo "Usage: $0 [llama|gemma|mistral|qwen32b|gemma2b|llama3b|qwen4b|qwen7b]"
        echo ""
        echo "Run one model at a time (each fits on a single RTX 5090)."
        echo "  llama   - Llama-3.1-8B-Instruct  (~16GB, port $LLAMA_PORT)"
        echo "  gemma   - Gemma-2-9B-it           (~18GB, port $GEMMA_PORT)"
        echo "  mistral - Mistral-Nemo-12B        (~24GB, port $MISTRAL_PORT)"
        echo "  qwen32b - Qwen2.5-32B-AWQ         (~18GB, port $QWEN_PORT)"
        echo "  gemma2b - Gemma-2-2B-it           (~5GB,  port $GEMMA2B_PORT)"
        echo "  llama3b - Llama-3.2-3B-Instruct   (~6GB,  port $LLAMA3B_PORT)"
        echo "  qwen4b  - Qwen3-4B                (~8GB,  port $QWEN4B_PORT)"
        echo "  qwen7b     - Qwen2.5-7B-Instruct            (~15GB, port $QWEN7B_PORT)"
        echo "  deepseek7b - DeepSeek-R1-Distill-Qwen-7B    (~14GB, port $DEEPSEEK7B_PORT)"
        echo "  phi4mini   - Phi-4-mini-instruct             (~8GB,  port $PHI4MINI_PORT)"
        echo "  gptoss20b  - GPT-oss-20B (OpenAI, MoE)       (~16GB, port $GPTOSS20B_PORT)"
        exit 1
        ;;
esac

echo ""
echo "Server launching in background. Wait ~60-90s, then check:"
echo "  curl http://localhost:${1:-8001}/v1/models"
