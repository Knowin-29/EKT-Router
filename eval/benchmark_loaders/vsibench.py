"""VSIbench benchmark loader for routing evaluation.

Loads the VSI-Bench dataset from HuggingFace (nyu-visionx/VSI-Bench)
and constructs routing-ready prompts matching the lmms-eval task definition.
"""

import os
from typing import Dict, List, Optional


MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]

NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]


def _doc_to_text(doc: Dict) -> str:
    """Replicate vsibench_doc_to_text from lmms-eval utils.py."""
    question = doc["question"]
    pre_prompt = "These are frames of a video."
    question_type = doc.get("question_type", "")

    if question_type in NA_QUESTION_TYPES:
        post_prompt = "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif question_type in MCA_QUESTION_TYPES:
        options = doc.get("options", [])
        options_text = "Options:\n" + "\n".join(options)
        post_prompt = "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options_text, post_prompt])
    else:
        post_prompt = "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt


def load_vsibench_samples(
    data_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    **kwargs,
) -> List[Dict]:
    """Load VSI-Bench samples from HuggingFace dataset.

    Requires: pip install datasets
    Video files should be cached locally.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    cache_dir = data_dir or os.path.expanduser("~/.cache/huggingface/vsibench")

    dataset = load_dataset(
        "nyu-visionx/VSI-Bench",
        split="test",
        cache_dir=cache_dir if data_dir else None,
    )

    samples = []
    for i, doc in enumerate(dataset):
        prompt = _doc_to_text(doc)

        dataset_name = doc.get("dataset", "")
        scene_name = doc.get("scene_name", "")
        video_path = os.path.join(cache_dir, dataset_name, scene_name + ".mp4")

        if not os.path.exists(video_path):
            alt_paths = [
                os.path.join(cache_dir, "vsibench", dataset_name, scene_name + ".mp4"),
                os.path.join(
                    os.path.expanduser("~/.cache/huggingface"),
                    "vsibench", dataset_name, scene_name + ".mp4"
                ),
            ]
            video_path = next((p for p in alt_paths if os.path.exists(p)), video_path)

        samples.append({
            "sample_id": f"vsibench_{i}",
            "benchmark": "vsibench",
            "prompt": prompt,
            "media_type": "video",
            "media_paths": [video_path],
        })

        if max_samples and len(samples) >= max_samples:
            break

    return samples
