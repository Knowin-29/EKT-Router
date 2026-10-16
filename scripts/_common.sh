#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EKT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export EKT_ROOT
export PYTHONPATH="$EKT_ROOT:${PYTHONPATH:-}"

export EKT_DATA_ROOT="${EKT_DATA_ROOT:-$EKT_ROOT/data}"
export EKT_CHECKPOINT_ROOT="${EKT_CHECKPOINT_ROOT:-$EKT_ROOT/checkpoints}"
export EKT_RESULTS_ROOT="${EKT_RESULTS_ROOT:-$EKT_ROOT/results}"
export EKT_LOG_ROOT="${EKT_LOG_ROOT:-$EKT_ROOT/logs}"
