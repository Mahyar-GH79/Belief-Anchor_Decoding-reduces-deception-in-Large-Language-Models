#!/bin/bash
#
# Sequential pipeline for the EMNLP submission, GPU 0 only.
# For each of the 6 reported models:
#   Phase A: launch vLLM → run all vLLM-dependent scripts → kill vLLM
#   Phase B: run all HF-dependent BAD scripts (no vLLM)
# After all models: compute bootstrap CIs (CPU-only).
#
# Usage:  bash runs/run_all.sh [model1 model2 ...]
# (no args = all 6 models)
#
# Logs go to logs/{vllm_<port>.log,<model>_<task>.log}.
#
set -e
set -o pipefail

PROJECT_ROOT="/mnt/data1/mahyar/TEMP_ICML2026"
cd "$PROJECT_ROOT"

mkdir -p logs

GPU=1
export CUDA_VISIBLE_DEVICES=$GPU

# Activate the project's venv (vLLM 0.17.1, transformers, etc. live here).
if [ -f "$PROJECT_ROOT/env/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/env/bin/activate"
    echo "Activated venv: $(which python3) ($(python3 -c 'import sys; print(sys.version.split()[0])'))"
else
    echo "ERROR: $PROJECT_ROOT/env/bin/activate not found." >&2
    exit 1
fi
python3 -c "import vllm" 2>/dev/null || { echo "ERROR: vllm not importable in venv." >&2; exit 1; }

# Models: name : port : hf_id : (optional) vllm-quant-flag
declare -a MODELS=(
    "gemma-2-9b-it:8002:google/gemma-2-9b-it:"
    "Qwen2.5-7B-Instruct:8008:Qwen/Qwen2.5-7B-Instruct:"
    "Meta-Llama-3.1-8B-Instruct:8001:meta-llama/Llama-3.1-8B-Instruct:"
    "Mistral-Nemo-Instruct-2407:8004:mistralai/Mistral-Nemo-Instruct-2407:"
    "Phi-4-mini-instruct:8010:microsoft/Phi-4-mini-instruct:"
    "Qwen2.5-32B-Instruct:8003:Qwen/Qwen2.5-32B-Instruct-AWQ:awq"
)

# Optional CLI filter: only run the named subset.
if [ $# -gt 0 ]; then
    REQUESTED=("$@")
    declare -a FILTERED=()
    for entry in "${MODELS[@]}"; do
        IFS=":" read -r m _ _ _ <<< "$entry"
        for r in "${REQUESTED[@]}"; do
            [[ "$m" == "$r" ]] && FILTERED+=("$entry")
        done
    done
    MODELS=("${FILTERED[@]}")
    [ ${#MODELS[@]} -eq 0 ] && { echo "No matching models."; exit 1; }
fi


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
    local pid=$1
    kill "$pid" 2>/dev/null || true
    # vLLM children: best effort cleanup
    pkill -f "vllm.entrypoints.openai.api_server.*--port $2" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sleep 8
}


for entry in "${MODELS[@]}"; do
    IFS=":" read -r model port hf quant <<< "$entry"
    log_prefix="logs/${model}"

    echo ""
    echo "##############################################################"
    echo "###  $model  (GPU $GPU, port $port)"
    echo "##############################################################"

    # ── Phase A: vLLM-dependent scripts ────────────────────────────────────
    echo "--- Phase A: launching vLLM ---"
    pid=$(start_vllm "$hf" "$port" "$quant")
    echo "vLLM PID: $pid (log: logs/vllm_${port}.log)"
    if ! wait_ready "$port"; then
        stop_vllm "$pid" "$port"
        echo "Skipping $model (vLLM failed to start)."
        continue
    fi

    echo "→ intervention_c on Linked + LinkedReverse"
    python3 -m runs.run_intervention_c_linked --model "$model" --port "$port" \
        2>&1 | tee "${log_prefix}_c_linked.log"

    echo "→ compute-matched Sample-and-Vote (k = n+1) on all 4 qtypes"
    python3 -m runs.run_compute_matched_voting --model "$model" --port "$port" \
        2>&1 | tee "${log_prefix}_matched_voting.log"

    echo "→ sycophancy: baseline + intervention_c on all 4 qtypes"
    python3 -m runs.run_sycophancy_prompt --model "$model" --port "$port" \
        2>&1 | tee "${log_prefix}_sycophancy_prompt.log"

    echo "--- Phase A done; stopping vLLM ---"
    stop_vllm "$pid" "$port"

    # ── Phase B: HF-only BAD scripts ───────────────────────────────────────
    echo "--- Phase B: HF-based BAD ---"

    echo "→ BAD on Linked + LinkedReverse + BrokenReverse (best λ from summary.json)"
    python3 -m runs.run_bad_remaining_qtypes --model "$model" --device 0 \
        2>&1 | tee "${log_prefix}_bad_remaining.log"

    echo "→ sycophantic BAD on all 4 qtypes"
    python3 -m runs.run_sycophancy_bad --model "$model" --device 0 \
        2>&1 | tee "${log_prefix}_sycophancy_bad.log"

    echo "##############################################################"
    echo "###  $model done"
    echo "##############################################################"
done


# ── Final analysis (CPU only) ─────────────────────────────────────────────────
echo ""
echo "=== Bootstrap 95% CIs ==="
python3 -m runs.compute_bootstrap_cis 2>&1 | tee logs/bootstrap_cis.log

echo ""
echo "ALL DONE."
