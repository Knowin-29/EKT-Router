"""EA-Temporal benchmark loader for routing evaluation."""

import json
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_DATA_DIR = "data/benchmarks/ea-temporal"
MERGED_JSON = "ea_temporal_merged.json"


def _build_prompt(sample: Dict, num_images: int) -> str:
    """Build the prompt matching the original inference script's default versions."""
    context = (sample.get("context") or "").strip()
    task_type = sample.get("task_type")

    if task_type == "type2":
        return (
            "You are given a sequence of images.\n"
            "Each image is labeled as 'Figure i:' in the input.\n"
            f"There are {num_images} images provided, in the same order as "
            f"Figure 1..Figure {num_images}.\n"
            "Goal: output the chronological order of these figures from EARLIEST to LATEST.\n"
            "\n"
            "IMPORTANT:\n"
            "- The figures are SHUFFLED. Do NOT assume the input order is chronological.\n"
            '- Do NOT default to "1, 2" or "1, 2, 3, 4". '
            "Decide by comparing visual state changes across figures.\n"
            "\n"
            "STRICT FORMAT RULES:\n"
            f"- Output EXACTLY {num_images} integers.\n"
            f"- The integers must be a permutation of 1..{num_images} (each used exactly once).\n"
            "- Output numbers only separated by comma+space.\n"
            "- Do NOT add any other words, quotes, brackets, or explanations.\n"
            "- If the question text shows an example order, it is ONLY an example of format; "
            "DO NOT copy it.\n"
            "\n"
            "SELF-CHECK (silent): Before you answer, verify you used each number exactly once "
            "and the count equals the number of images.\n"
            "\n"
            f"Question:\n{context}\n"
            "\nAnswer:"
        )

    # type1 (v1 default)
    return (
        "You are given two images: Figure 1 is BEFORE, Figure 2 is AFTER.\n"
        "Describe the robot's action as a short verb phrase (<= 10 words).\n"
        "\n"
        "STRICT FORMAT RULES:\n"
        "- Output ONLY the short phrase.\n"
        "- Do NOT include quotes or extra explanation.\n"
        "- Focus only on what the robot did (action + object + destination if needed).\n"
        "\n"
        f"Question:\n{context}\n"
        "\nAnswer:"
    )


def load_ea_temporal_samples(
    data_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    **kwargs,
) -> List[Dict]:
    """Load EA-Temporal samples and construct routing-ready prompts."""
    data_dir = Path(data_dir or DEFAULT_DATA_DIR)
    json_path = data_dir / MERGED_JSON
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = data.get("data", data) if isinstance(data, dict) else []

    samples = []
    for item in data:
        task = item.get("task_instance", {})
        context = task.get("context", "")
        images_path = task.get("images_path", [])
        if not context or not images_path:
            continue

        resolved = []
        for p in images_path:
            p = p.lstrip("./")
            full = str(data_dir / p)
            resolved.append(full)

        task_type = "type2" if item.get("_response_is_list") else "type1"
        sample_data = {
            "context": context,
            "task_type": task_type,
        }

        prompt = _build_prompt(sample_data, num_images=len(resolved))

        samples.append({
            "sample_id": str(item.get("sample_id", len(samples))),
            "benchmark": "ea_temporal",
            "prompt": prompt,
            "media_type": "images",
            "media_paths": resolved,
        })

        if max_samples and len(samples) >= max_samples:
            break

    return samples
