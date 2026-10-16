"""Build paired utility targets from normalized offline route outcomes.

Use a media-group split before collecting/encoding data. This command never
infers scores or embeddings from answers, and never calls a model or a judge.
"""

import argparse
import itertools
import json
import math
from pathlib import Path


def pairs(record, epsilon=0.0, cost_weight=1.0):
    embedding = record["query_embedding"]
    if not embedding or not all(math.isfinite(float(v)) for v in embedding):
        raise ValueError("finite frozen query_embedding is required")
    outcomes = record["outcomes"]
    for row in outcomes:
        if not 0 <= float(row["score"]) <= 1:
            raise ValueError("normalize native scores to [0, 1] before pairing")
        if not math.isfinite(float(row["cost"])) or row["cost"] < 0:
            raise ValueError("finite non-negative acquisition cost is required")
    for left, right in itertools.combinations(outcomes, 2):
        delta = left["score"] - right["score"] - cost_weight * (left["cost"] - right["cost"])
        yield {
            "id": record["id"],
            "media_group": record["media_group"],
            "query_embedding": embedding,
            "left": {k: left[k] for k in ("expert", "action", "cost")},
            "right": {k: right[k] for k in ("expert", "action", "cost")},
            "preference": 0 if abs(delta) <= epsilon else (1 if delta > 0 else -1),
            "target_gain": delta,
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--verifier-output", help="optional typed packet supervision JSONL")
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--cost-weight", type=float, default=1.0)
    args = p.parse_args()
    if args.epsilon < 0 or args.cost_weight < 0:
        p.error("epsilon and cost-weight must be non-negative")
    verifier_rows = []
    with Path(args.output).open("x") as stream:
        for line in Path(args.input).read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                for outcome in record["outcomes"]:
                    action = outcome["action"]
                    if outcome.get("packets_valid") and (
                        action["knowledge"] != "none" or action["tool"] != "none"
                    ):
                        verifier_rows.append(
                            {
                                "id": record["id"],
                                "media_group": record["media_group"],
                                "query_embedding": record["query_embedding"],
                                "expert": outcome["expert"],
                                "action": action,
                                "packet_features": outcome["packet_features"],
                                "realized_utility": outcome["score"]
                                - args.cost_weight * outcome["cost"],
                            }
                        )
                for row in pairs(record, args.epsilon, args.cost_weight):
                    stream.write(json.dumps(row) + "\n")

    if args.verifier_output:
        with Path(args.verifier_output).open("x") as stream:
            stream.write("".join(json.dumps(row) + "\n" for row in verifier_rows))


if __name__ == "__main__":
    main()
