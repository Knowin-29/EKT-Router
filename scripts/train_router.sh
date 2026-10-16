#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

SIGNATURES="${SIGNATURES:?Set SIGNATURES to expert_signatures.json}"
TRAIN_JSONL="${TRAIN_JSONL:-$EKT_DATA_ROOT/router/train.jsonl}"
VERIFIER_JSONL="${VERIFIER_JSONL:-$EKT_DATA_ROOT/router/verifier.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$EKT_CHECKPOINT_ROOT/router}"
DEVICE="${DEVICE:-cuda}"

args=(
  --config "$EKT_ROOT/configs/train/router.yaml"
  --train-path "$TRAIN_JSONL"
  --signatures "$SIGNATURES"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
)
if [ -f "$VERIFIER_JSONL" ]; then
  args+=(--verifier-path "$VERIFIER_JSONL")
fi
python "$EKT_ROOT/train/train_utility_router.py" "${args[@]}"
