#!/bin/bash
# Run all conditions for a single model: server start → experiments → server stop.
#
# Usage:
#   bash run_model.sh llama
#   bash run_model.sh gemma
#   bash run_model.sh mistral
#   bash run_model.sh qwen32b
#   bash run_model.sh gemma2b
#   bash run_model.sh llama3b
#   bash run_model.sh qwen4b
#   bash run_model.sh qwen7b

set -e

case "${1}" in
    llama)
        MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
        SERVER_CMD="llama"
        PORT=8001
        ;;
    gemma)
        MODEL_NAME="gemma-2-9b-it"
        SERVER_CMD="gemma"
        PORT=8002
        ;;
    mistral)
        MODEL_NAME="Mistral-Nemo-Instruct-2407"
        SERVER_CMD="mistral"
        PORT=8004
        ;;
    qwen32b)
        MODEL_NAME="Qwen2.5-32B-Instruct"
        SERVER_CMD="qwen32b"
        PORT=8003
        ;;
    gemma2b)
        MODEL_NAME="gemma-2-2b-it"
        SERVER_CMD="gemma2b"
        PORT=8005
        ;;
    llama3b)
        MODEL_NAME="Llama-3.2-3B-Instruct"
        SERVER_CMD="llama3b"
        PORT=8006
        ;;
    qwen4b)
        MODEL_NAME="Qwen3-4B"
        SERVER_CMD="qwen4b"
        PORT=8007
        ;;
    qwen7b)
        MODEL_NAME="Qwen2.5-7B-Instruct"
        SERVER_CMD="qwen7b"
        PORT=8008
        ;;
    deepseek7b)
        MODEL_NAME="DeepSeek-R1-Distill-Qwen-7B"
        SERVER_CMD="deepseek7b"
        PORT=8009
        ;;
    phi4mini)
        MODEL_NAME="Phi-4-mini-instruct"
        SERVER_CMD="phi4mini"
        PORT=8010
        ;;
    gptoss20b)
        MODEL_NAME="GPT-oss-20B"
        SERVER_CMD="gptoss20b"
        PORT=8011
        ;;
    # --- Remote models (Together.ai API, no local server needed) ---
    llama70b)
        MODEL_NAME="Llama-3.3-70B-Instruct"
        REMOTE=true
        ;;
    deepseekv3)
        MODEL_NAME="DeepSeek-V3.1"
        REMOTE=true
        ;;
    qwen235b)
        MODEL_NAME="Qwen3-235B-A22B-Instruct"
        REMOTE=true
        ;;
    *)
        echo "Usage: $0 <model>"
        echo ""
        echo "Local models (starts vLLM server automatically):"
        echo "  llama | gemma | mistral | qwen32b"
        echo "  gemma2b | llama3b | qwen4b | qwen7b"
        echo "  deepseek7b | phi4mini | gptoss20b"
        echo ""
        echo "Remote models (Together.ai API, needs TOGETHER_API_KEY):"
        echo "  llama70b | deepseekv3 | qwen235b"
        exit 1
        ;;
esac

echo "============================================================"
echo "  Model: $MODEL_NAME"
if [ "${REMOTE:-false}" = "true" ]; then
    echo "  Mode:  REMOTE (Together.ai API)"
else
    echo "  Mode:  LOCAL (vLLM on port $PORT)"
fi
echo "============================================================"

# --- Local: start vLLM server ---
if [ "${REMOTE:-false}" != "true" ]; then
    pkill -f "port $PORT" 2>/dev/null || true
    sleep 2

    echo ""
    echo "[1/3] Starting vLLM server..."
    bash launch_servers.sh $SERVER_CMD

    echo "Waiting for server to be ready..."
    for i in $(seq 1 36); do
        sleep 5
        if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
            echo "Server ready after $((i*5))s"
            break
        fi
        if [ $i -eq 36 ]; then
            echo "ERROR: Server did not start within 180s"
            echo "Check logs or GPU memory"
            exit 1
        fi
    done

    SERVER_PID=$(lsof -ti :$PORT 2>/dev/null | head -1)
    echo "Server PID: $SERVER_PID"
else
    # Remote: verify API key is set
    if [ -z "$TOGETHER_API_KEY" ]; then
        echo ""
        echo "ERROR: TOGETHER_API_KEY is not set."
        echo "Run: export TOGETHER_API_KEY=your_key_here"
        exit 1
    fi
    echo ""
    echo "[1/3] Remote model — no local server needed."
fi

# --- Run all conditions ---
echo ""
echo "[2/3] Running experiments..."
CONDITIONS="baseline intervention_a intervention_b intervention_c intervention_ab"

FAILED=""
for COND in $CONDITIONS; do
    echo ""
    echo "────────────────────────────────────────"
    echo "  Condition: $COND"
    echo "────────────────────────────────────────"
    if ! python3 main.py run --model "$MODEL_NAME" --condition "$COND"; then
        echo "  WARNING: $COND failed, continuing to next condition..."
        FAILED="$FAILED $COND"
    fi
done

if [ -n "$FAILED" ]; then
    echo ""
    echo "WARNING: The following conditions failed:$FAILED"
    echo "Re-run them manually after fixing the issue."
fi

# --- Local: stop server ---
echo ""
if [ "${REMOTE:-false}" != "true" ]; then
    echo "[3/3] Stopping server..."
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
    else
        pkill -f "port $PORT" 2>/dev/null || true
    fi
else
    echo "[3/3] Remote model — nothing to stop."
fi

echo ""
echo "============================================================"
echo "  DONE: $MODEL_NAME — all conditions complete"
echo "============================================================"
echo ""
echo "Results in: results/$MODEL_NAME/"
ls -1 results/$MODEL_NAME/ 2>/dev/null | wc -l
echo "files saved."
