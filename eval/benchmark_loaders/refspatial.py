"""RefSpatial-Bench benchmark loader for routing evaluation."""

import json
import os
from typing import Dict, List, Optional


DEFAULT_DATA_DIR = "data/benchmarks/refspatial"

REFSPATIAL_SPLITS = ["Location", "Placement", "Unseen"]

CUSTOM_PROMPT = "Your answer should contain exactly ONE point and is formatted as follows [[x1, y1]]."


def load_refspatial_samples(
    data_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    **kwargs,
) -> List[Dict]:
    """Load RefSpatial-Bench samples and construct routing-ready prompts."""
    data_dir = data_dir or DEFAULT_DATA_DIR

    samples = []
    for split_name in REFSPATIAL_SPLITS:
        split_dir = os.path.join(data_dir, split_name)
        if not os.path.exists(split_dir):
            split_dir = os.path.join(data_dir, split_name.lower())
        question_file = os.path.join(split_dir, "question.json")
        if not os.path.exists(question_file):
            continue

        with open(question_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            qid = item.get("id", item.get("question_id", ""))
            prompt_text = item.get("prompt", "")

            full_prompt = f"{prompt_text} {CUSTOM_PROMPT}"

            rgb_path = item.get("rgb_path", item.get("image", ""))
            image_path = os.path.join(split_dir, rgb_path) if rgb_path else ""

            media_paths = [image_path] if image_path and os.path.exists(image_path) else []

            samples.append({
                "sample_id": f"refspatial_{split_name}_{qid}",
                "benchmark": "refspatial",
                "prompt": full_prompt,
                "media_type": "image",
                "media_paths": media_paths,
            })

            if max_samples and len(samples) >= max_samples:
                return samples

    return samples
