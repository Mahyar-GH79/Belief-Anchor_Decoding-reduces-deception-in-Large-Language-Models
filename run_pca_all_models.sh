#!/usr/bin/env bash
# Run PCA embedding analysis for all models and all difficulty levels (n),
# using every available sample — matching the original paper's approach.
#
# Usage:
#   ./run_pca_all_models.sh        # GPU 0
#   ./run_pca_all_models.sh 1      # GPU 1

set -euo pipefail
cd "$(dirname "$0")"

DEVICE=${1:-0}
BATCH=4
OUT_DIR="plots/pca"

MODELS=(
    gemma-2-2b-it
    gemma-2-9b-it
    Qwen3-4B
    Qwen2.5-7B-Instruct
    Meta-Llama-3.1-8B-Instruct
    Mistral-Nemo-Instruct-2407
    Phi-4-mini-instruct
)

# All difficulty levels (200 questions each, all used)
N_VALUES=(5 10 20 40 80)

mkdir -p "${OUT_DIR}" logs

echo "=== PCA embedding analysis: GPU=${DEVICE}, all n, all samples ==="
echo "Models: ${MODELS[*]}"
echo "n values: ${N_VALUES[*]}"
echo "Output: ${OUT_DIR}"
echo

for model in "${MODELS[@]}"; do
    for n in "${N_VALUES[@]}"; do
        echo "--- Starting: ${model}  n=${n} ---"
        python -m analysis.pca_embeddings \
            --models "${model}" \
            --n "${n}" \
            --device "${DEVICE}" \
            --batch-size "${BATCH}" \
            --out-dir "${OUT_DIR}" \
            && echo "--- Done: ${model}  n=${n} ---" \
            || echo "--- FAILED: ${model}  n=${n} (continuing) ---"
        echo
    done
done

echo "=== All models done. Plots saved to ${OUT_DIR} ==="
