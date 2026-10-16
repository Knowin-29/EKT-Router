"""JSONL inference plus explicitly selected local scorer; no implicit API judge."""

import argparse
import json
from pathlib import Path
from modules.runtime import build_runtime, load_callable
from .protocol_manifest import sha256_file, sha256_json


def exact_match(question, reference, prediction):
    return float(str(reference).strip().casefold() == str(prediction).strip().casefold())


def run(input_path, output_path, config=None, scorer=None, benchmark="custom", split="test"):
    rows = [json.loads(line) for line in Path(input_path).read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("input must be non-empty with unique IDs")
    pipeline, provider, cfg = build_runtime(config)
    score_fn = load_callable(scorer) if scorer else None
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("prediction file exists; choose a new output")
    output.parent.mkdir(parents=True, exist_ok=True)
    scores = []
    # Failed runs remain visibly incomplete; never substitute empty/stub predictions.
    with output.open("x") as stream:
        for row in rows:
            question = row["question"]
            answer = pipeline.run_bounded(question, row.get("media"), provider)
            result = {
                "id": row["id"],
                "prediction": answer,
                "route": pipeline.last_route.as_decision(),
                "packets": {k: p.to_dict() for k, p in pipeline.last_packets.items()},
            }
            if score_fn:
                result["score"] = float(score_fn(question, row["reference"], answer))
                scores.append(result["score"])
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
    manifest = {
        "model": "KnowinRouter",
        "benchmark": benchmark,
        "split": split,
        "status": "completed_local_run",
        "paper_reproduction": False,
        "sample_count": len(rows),
        "input_sha256": sha256_file(input_path),
        "prediction_sha256": sha256_file(output),
        "config_sha256": sha256_json(cfg),
        "scorer": scorer,
        "metric_mean": sum(scores) / len(scores) if scores else None,
        "note": "Native split/scorer versions and checkpoint hashes must be audited before claiming protocol-matched reproduction.",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/global.yaml")
    p.add_argument(
        "--scorer",
        help="local module:function(question, reference, prediction), e.g. native benchmark scorer",
    )
    p.add_argument("--benchmark", default="custom")
    p.add_argument("--split", default="test")
    args = p.parse_args()
    print(
        json.dumps(
            run(args.input, args.output, args.config, args.scorer, args.benchmark, args.split),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
