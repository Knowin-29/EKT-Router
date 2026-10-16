#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../scripts/_common.sh"

smoke_dir="$(mktemp -d)"
trap 'rm -rf "$smoke_dir"' EXIT

python "$EKT_ROOT/train/build_signatures.py" \
  --input "$EKT_ROOT/supplement/sample_data/diagnostic_outcomes.jsonl" \
  --output "$smoke_dir/expert_signatures.json"

python "$EKT_ROOT/train/train_utility_router.py" \
  --config "$EKT_ROOT/supplement/router_smoke.yaml" \
  --train-path "$EKT_ROOT/supplement/sample_data/route_pairs.jsonl" \
  --verifier-path "$EKT_ROOT/supplement/sample_data/verifier.jsonl" \
  --signatures "$smoke_dir/expert_signatures.json" \
  --output-dir "$smoke_dir/checkpoint" \
  --device cpu

test -s "$smoke_dir/checkpoint/knowinrouter.pt"
python "$EKT_ROOT/examples/bounded_routing_demo.py"
echo "Supplement smoke validation passed."
