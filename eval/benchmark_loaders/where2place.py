"""Where2Place benchmark loader for routing evaluation."""

import json
import os
import re
from typing import Dict, List, Optional


DEFAULT_DATA_DIR = "data/benchmarks/where2place"

ORIGINAL_SUFFIX_PATTERN = (
    r"Your answer should be formatted as a list of tuples.*?"
    r"the normalized pixel locations of the points in the image\."
)
CUSTOM_PROMPT = "Your answer should contain exactly ONE point and is formatted as follows [[x1, y1]]."


def load_where2place_samples(
    data_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    **kwargs,
) -> List[Dict]:
    """Load Where2Place samples and construct routing-ready prompts."""
    data_dir = data_dir or DEFAULT_DATA_DIR
    jsonl_path = os.path.join(data_dir, "point_questions.jsonl")
    images_dir = os.path.join(data_dir, "images")

    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            qid = item.get("question_id", i)
            image_file = item.get("image", "")
            text = item.get("text", "")

            prompt = re.sub(ORIGINAL_SUFFIX_PATTERN, CUSTOM_PROMPT, text, flags=re.DOTALL)

            image_path = os.path.join(images_dir, image_file) if image_file else ""
            if image_path and not os.path.exists(image_path):
                image_path = os.path.join(data_dir, image_file)

            media_paths = [image_path] if image_path and os.path.exists(image_path) else []

            samples.append({
                "sample_id": f"w2p_{qid}",
                "benchmark": "where2place",
                "prompt": prompt,
                "media_type": "image",
                "media_paths": media_paths,
            })

            if max_samples and len(samples) >= max_samples:
                break

    return samples
