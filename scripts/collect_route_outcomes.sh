#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$EKT_ROOT"
python -m judge.label_generator --data-path "${INPUT_JSONL:?Set INPUT_JSONL}" \
  --output-path "${OUTPUT_JSONL:?Set OUTPUT_JSONL}" --backend "${BACKEND:-mock}" "$@"
