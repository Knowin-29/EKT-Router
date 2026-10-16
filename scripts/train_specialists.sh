#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

GPUS="${GPUS:-0,1}"
for expert in spatial_reasoning spatiotemporal; do
  python "$EKT_ROOT/train/train_expert_swift.py" \
    --config "$EKT_ROOT/configs/train/${expert}.yaml" \
    --gpus "$GPUS"
done
