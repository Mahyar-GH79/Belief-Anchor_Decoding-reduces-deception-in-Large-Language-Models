#!/bin/bash
#
# Parallel runner for GPU 1, while runs/run_all.sh is still running on GPU 0.
#
# Order: Qwen-32B-AWQ first (slowest), then Phi-4-mini. Reasoning: those are
# last in the master script's queue, so they're the ones that benefit most
# from being done in parallel.
#
# All Python scripts are idempotent — when GPU 0 eventually iterates to the
# same model, it will skip every cell already on disk.
#
# Ports are offset by +100 from the master script to avoid vLLM port clashes
# on the same machine.
#
# Usage:  bash runs/run_parallel_gpu1.sh
#
set -e
set -o pipefail

PROJECT_ROOT="/mnt/data1/mahyar/TEMP_ICML2026"
cd "$PROJECT_ROOT"

mkdir -p logs

GPU=1
export CUDA_VISIBLE_DEVICES=$GPU

if [ -f "$PROJECT_ROOT/env/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/env/bin/activate"
    echo "[GPU $GPU] venv: $(which python3) ($(python3 -c 'import sys; print(sys.version.split()[0])'))"
else
    echo "ERROR: $PROJECT_ROOT/env/bin/activate not found." >&2
    exit 1
fi
python3 -c "import vllm" 2>/dev/null || { echo "ERROR: vllm not importable in venv." >&2; exit 1; }


# Ports offset by +100 from the master to avoid clashes
declare -a MODELS=(
    "Qwen2.5-32B-Instruct:8103:Qwen/Qwen2.5-32B-Instruct-AWQ:awq"
    "Phi-4-mini-instruct:8110:microsoft/Phi-4-mini-instruct:"
)


start_vllm() {
    local hf=$1 port=$2 quant=$3
    local extra=""
    [ -n "$quant" ] && extra="--quantization $quant"
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$hf" \
        --port "$port" \
        --tensor-parallel-size 1 \
        --max-model-len 8192 \
        --dtype auto \
        --trust-remote-code \
        $extra \
        > "logs/vllm_${port}.log" 2>&1 &
    echo $!
}

wait_ready() {
    local port=$1
    for i in $(seq 1 180); do
        if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    echo "ERROR: vLLM on port $port did not become ready in 900s." >&2
    return 1
}

stop_vllm() {
    local pid=$1 port=$2
    kill "$pid" 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server.*--port $port" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sleep 8
}


for entry in "${MODELS[@]}"; do
    IFS=":" read -r model port hf quant <<< "$entry"
    log_prefix="logs/${model}_gpu1"

    echo ""
    echo "=============================================================="
    echo "[GPU 1]  $model  (port $port)"
    echo "=============================================================="

    # Phase A: vLLM-dependent scripts
    echo "[GPU 1] --- Phase A: launching vLLM on port $port ---"
    pid=$(start_vllm "$hf" "$port" "$quant")
    echo "[GPU 1] vLLM PID: $pid (log: logs/vllm_${port}.log)"
    if ! wait_ready "$port"; then
        stop_vllm "$pid" "$port"
        echo "[GPU 1] Skipping $model (vLLM failed to start)."
        continue
    fi

    echo "[GPU 1] → intervention_c on Linked + LinkedReverse"
    python3 -m runs.run_intervention_c_linked --model "$model" --port "$port" \
        2>&1 | tee "${log_prefix}_c_linked.log"

    echo "[GPU 1] → compute-matched Sample-and-Vote"
    python3 -m runs.run_compute_matched_voting --model "$model" --port "$port" \
        2>&1 | tee "${log_prefix}_matched_voting.log"

    echo "[GPU 1] → sycophancy: baseline + intervention_c"
    python3 -m runs.run_sycophancy_prompt --model "$model" --port "$port" \
        2>&1 | tee "${log_prefix}_sycophancy_prompt.log"

    echo "[GPU 1] --- Phase A done; stopping vLLM ---"
    stop_vllm "$pid" "$port"

    # Phase B: HF-only BAD scripts
    echo "[GPU 1] --- Phase B: HF-based BAD ---"

    echo "[GPU 1] → BAD on Linked + LinkedReverse + BrokenReverse"
    python3 -m runs.run_bad_remaining_qtypes --model "$model" --device 0 \
        2>&1 | tee "${log_prefix}_bad_remaining.log"

    echo "[GPU 1] → sycophantic BAD on all 4 qtypes"
    python3 -m runs.run_sycophancy_bad --model "$model" --device 0 \
        2>&1 | tee "${log_prefix}_sycophancy_bad.log"

    echo "[GPU 1] $model done"
done

echo ""
echo "[GPU 1] ALL DONE. Bootstrap CIs will be computed by GPU 0's master script when it finishes."
