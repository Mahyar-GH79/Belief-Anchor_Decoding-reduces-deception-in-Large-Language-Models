#!/usr/bin/env bash
# Method 3: Hierarchical Consistency Annealing
# Usage: ./run_method3.sh [GPU_ID]   default GPU=0
set -euo pipefail
cd "$(dirname "$0")"
DEVICE=${1:-0}
mkdir -p logs
echo "=== Method 3: Hierarchical Consistency Annealing — GPU ${DEVICE} ==="
python -m methods.method3_hca --device "${DEVICE}" 2>&1 | tee logs/method3_gpu${DEVICE}.log
echo "Done."
