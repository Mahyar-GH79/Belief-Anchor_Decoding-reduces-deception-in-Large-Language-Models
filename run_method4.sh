#!/usr/bin/env bash
# Method 4: Deception-Contrastive Decoding
# Usage: ./run_method4.sh [GPU_ID]   default GPU=0
set -euo pipefail
cd "$(dirname "$0")"
DEVICE=${1:-0}
mkdir -p logs
echo "=== Method 4: Deception-Contrastive Decoding — GPU ${DEVICE} ==="
python -m methods.method4_contrastive --device "${DEVICE}" 2>&1 | tee logs/method4_gpu${DEVICE}.log
echo "Done."
