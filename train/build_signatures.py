"""Build outcome-derived expert signatures from a diagnostic outcome table."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.taxonomy import EXPERT_IDS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="diagnostic JSONL")
    parser.add_argument("--output", required=True, help="signature JSON")
    parser.add_argument(
        "--axes",
        default="general,spatial_reasoning,spatiotemporal",
        help="ordered comma-separated capability axes",
    )
    args = parser.parse_args()

    axes = tuple(value.strip() for value in args.axes.split(",") if value.strip())
    sums = defaultdict(float)
    counts = defaultdict(int)
    with Path(args.input).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            expert = str(row["expert"])
            axis = str(row["capability_axis"])
            if expert not in EXPERT_IDS or axis not in axes:
                continue
            key = (expert, axis)
            score = float(row["score"])
            lower, upper = float(row.get("score_min", 0.0)), float(row.get("score_max", 1.0))
            if not all(math.isfinite(v) for v in (score, lower, upper)) or upper <= lower or not lower <= score <= upper:
                raise ValueError("supply finite probe score bounds; scores must lie within them")
            sums[key] += (score - lower) / (upper - lower)
            counts[key] += 1

    missing = [
        f"{expert}:{axis}"
        for expert in EXPERT_IDS
        for axis in axes
        if counts[(expert, axis)] == 0
    ]
    if missing:
        raise ValueError("missing diagnostic cells: " + ", ".join(missing))

    signatures = {
        expert: [sums[(expert, axis)] / counts[(expert, axis)] for axis in axes]
        for expert in EXPERT_IDS
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(signatures, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
