#!/usr/bin/env bash
# Method 2: Probe-Guided Activation Steering
# Usage: ./run_method2.sh [GPU_ID]   default GPU=0
set -euo pipefail
cd "$(dirname "$0")"
DEVICE=${1:-0}
mkdir -p logs
echo "=== Method 2: Probe-Guided Steering — GPU ${DEVICE} ==="
python -m methods.method2_probe --device "${DEVICE}" 2>&1 | tee logs/method2_gpu${DEVICE}.log
echo "Done."
