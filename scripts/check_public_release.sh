#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
cd "$EKT_ROOT"
python scripts/audit_release.py "$@"
