#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

python -m pytest -q
python "$EKT_ROOT/examples/bounded_routing_demo.py"
