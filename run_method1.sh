#!/usr/bin/env bash
# Method 1: Activation Steering via Deception Subspace Projection
# Usage: ./run_method1.sh [GPU_ID]   default GPU=0
set -euo pipefail
cd "$(dirname "$0")"
DEVICE=${1:-0}
mkdir -p logs
echo "=== Method 1: Activation Steering — GPU ${DEVICE} ==="
python -m methods.method1_steering --device "${DEVICE}" 2>&1 | tee logs/method1_gpu${DEVICE}.log
echo "Done."
