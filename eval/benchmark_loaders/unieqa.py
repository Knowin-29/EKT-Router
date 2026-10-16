"""UniEQA benchmark loader for routing evaluation."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_DATA_DIR = "data/benchmarks/unieqa"

VQA_PROMPT_TEMPLATE = (
    "Based on the image(s) provided, please answer the following question concisely.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _midpoint_segment_indices(start: int, end: int, num_segments: int) -> np.ndarray:
    seg_size = (end - start + 1) / num_segments
    return np.array(
        [int(start + (i + 0.5) * seg_size) for i in range(num_segments)]
    )


def _sample_sequence_midpoint(items: list, num_segments: int) -> list:
    if len(items) == 0:
        return []
    if len(items) <= num_segments:
        return list(items)
    max_frame = len(items) - 1
    indices = _midpoint_segment_indices(0, max_frame, num_segments)
    indices = np.clip(indices, 0, max_frame)
    return [items[int(i)] for i in indices.tolist()]


def _sample_frames_from_episode(episode_dir: str, num_frames: int = 8) -> List[str]:
    if not os.path.isdir(episode_dir):
        return []
    frames = sorted(Path(episode_dir).glob("*-rgb.png"))
    if not frames:
        frames = sorted(Path(episode_dir).glob("*.png"))
        frames = [f for f in frames if "depth" not in f.name.lower()]
    if not frames:
        frames = sorted(Path(episode_dir).glob("*.jpg"))
    if not frames:
        return []
    frame_paths = [str(p) for p in frames]
    return _sample_sequence_midpoint(frame_paths, num_frames)


def _resolve_image_paths(images_path: List[str], images_dir: str, num_frames: int = 8) -> List[str]:
    resolved = []
    for img_path in images_path:
        full_path = os.path.join(images_dir, img_path)
        if os.path.isdir(full_path):
            sampled = _sample_frames_from_episode(full_path, num_frames)
            resolved.extend(sampled)
        elif os.path.isfile(full_path):
            resolved.append(full_path)
        elif os.path.isfile(full_path + ".jpg"):
            resolved.append(full_path + ".jpg")
        elif os.path.isfile(full_path + ".png"):
            resolved.append(full_path + ".png")
    return resolved


def _find_all_data_files(base_dir: str) -> List[Tuple[str, str, str]]:
    data_files = []
    for root, dirs, files in os.walk(base_dir):
        if "data.json" in files:
            data_path = os.path.join(root, "data.json")
            images_dir = os.path.join(root, "images")
            if not os.path.exists(images_dir):
                parent_images = os.path.join(os.path.dirname(root), "images")
                if os.path.exists(parent_images):
                    images_dir = parent_images
                else:
                    part_match = root.split("/data/Part")
                    if len(part_match) > 1:
                        part_num = part_match[1].split("/")[0]
                        part_images = os.path.join(base_dir, "data", f"Part{part_num}", "images")
                        if os.path.exists(part_images):
                            images_dir = part_images
            rel_path = os.path.relpath(root, base_dir)
            part_name = rel_path.replace("/core", "").replace("\\core", "")
            data_files.append((data_path, images_dir, part_name))
    return data_files


def load_unieqa_samples(
    data_dir: Optional[str] = None,
    num_frames: int = 8,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Load UniEQA samples and construct routing-ready prompts."""
    data_dir = data_dir or DEFAULT_DATA_DIR
    data_files = _find_all_data_files(data_dir)
    if not data_files:
        raise FileNotFoundError(f"No data.json files found in {data_dir}")

    samples = []
    for data_path, images_dir, part_name in data_files:
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        metadata = data.get("metadata", {})
        dataset_name = metadata.get("dataset", "unknown")

        for sample in data.get("data", []):
            task_instance = sample.get("task_instance", {})
            question = task_instance.get("context", "")
            images_path = task_instance.get("images_path", [])
            sample_id = sample.get("sample_id", 0)

            resolved_images = _resolve_image_paths(images_path, images_dir, num_frames)
            if not resolved_images:
                continue

            prompt = VQA_PROMPT_TEMPLATE.format(question=question)

            if len(resolved_images) == 1:
                media_type = "image"
            else:
                media_type = "video_frames"

            samples.append({
                "sample_id": f"{part_name}_{dataset_name}_{sample_id}",
                "benchmark": "unieqa",
                "prompt": prompt,
                "media_type": media_type,
                "media_paths": resolved_images,
            })

            if max_samples and len(samples) >= max_samples:
                return samples

    return samples
