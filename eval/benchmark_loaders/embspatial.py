"""EmbSpatial-Bench benchmark loader for routing evaluation."""

import base64
import json
import os
from typing import Dict, List, Optional

DEFAULT_DATA_PATH = (
    "data/benchmarks/embspatial/embspatial_bench.json"
)

MC_PROMPT_TEMPLATE = (
    "Look at the image and answer the following multiple-choice question. "
    "Choose exactly one option (A, B, C, or D).\n\n"
    "Question: {question}\n\n"
    "Options:\n"
    "A. {option_a}\n"
    "B. {option_b}\n"
    "C. {option_c}\n"
    "D. {option_d}\n\n"
    "Answer with only the letter (A, B, C, or D):"
)


def _decode_base64_to_tempfile(b64_string: str, tmp_dir: str, idx: int) -> str:
    """Decode base64 image and save to a temp file, return the path."""
    img_bytes = base64.b64decode(b64_string)
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"embspatial_{idx}.jpg")
    with open(path, "wb") as f:
        f.write(img_bytes)
    return path


def load_embspatial_samples(
    data_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    **kwargs,
) -> List[Dict]:
    """Load EmbSpatial-Bench samples and construct routing-ready prompts.

    Since images are base64-encoded inline, we store the base64 string
    in media_data for API mode, or decode to temp files for local mode.
    """
    data_path = data_dir or DEFAULT_DATA_PATH
    if os.path.isdir(data_path):
        data_path = os.path.join(data_path, "embspatial_bench.json")

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = []
    for item in dataset:
        question = item.get("question", "")
        options = item.get("answer_options", [])
        if len(options) < 4:
            continue

        prompt = MC_PROMPT_TEMPLATE.format(
            question=question,
            option_a=options[0],
            option_b=options[1],
            option_c=options[2],
            option_d=options[3],
        )

        question_id = item.get("question_id", len(samples))

        samples.append({
            "sample_id": str(question_id),
            "benchmark": "embspatial",
            "prompt": prompt,
            "media_type": "base64_image",
            "media_paths": [],
            "media_base64": item.get("image", ""),
        })

        if max_samples and len(samples) >= max_samples:
            break

    return samples
