"""Cosmos benchmark loader for routing evaluation."""

import json
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_DATA_DIR = "data/benchmarks/cosmos"
COSMOS_SUBSETS = ["agibot", "bridgev2", "robovqa", "robofail", "holoassist"]


def _build_prompt(sample: Dict) -> str:
    q = sample["question"]
    idx = sample["index2ans"]
    keys = sorted(idx.keys())
    lines = [f"{k}. {idx[k]}" for k in keys]
    options_text = "\n".join(lines)
    return (
        f"{q}\n\nOptions:\n{options_text}\n\n"
        "Answer with only the letter (e.g. A, B, C, or D):"
    )


def load_cosmos_samples(
    data_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    **kwargs,
) -> List[Dict]:
    """Load Cosmos benchmark samples and construct routing-ready prompts."""
    data_dir = Path(data_dir or DEFAULT_DATA_DIR)

    samples = []
    for subset in COSMOS_SUBSETS:
        json_path = data_dir / subset / f"{subset}_benchmark_qa_pairs.json"
        if not json_path.exists():
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raw = [raw]

        for i, item in enumerate(raw):
            video_rel = item.get("video", "")
            qa = item.get("qa_pairs", {})
            question = qa.get("question", "")
            index2ans = qa.get("index2ans", {})
            answer = qa.get("answer", "")

            if not video_rel or not question or not answer:
                continue

            video_path = str(data_dir / subset / video_rel)

            prompt_sample = {
                "question": question,
                "index2ans": index2ans,
            }
            prompt = _build_prompt(prompt_sample)

            samples.append({
                "sample_id": f"{subset}_{i}",
                "benchmark": "cosmos",
                "prompt": prompt,
                "media_type": "video",
                "media_paths": [video_path],
            })

            if max_samples and len(samples) >= max_samples:
                return samples

    return samples
