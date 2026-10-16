"""OpenEQA benchmark loader for routing evaluation."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


DEFAULT_DATA_DIR = "data/benchmarks/openeqa"

VQA_PROMPT_TEMPLATE = (
    "You are an intelligent question answering agent. "
    "You are shown video frames from an indoor environment.\n"
    "Based on what you see in the video frames, please answer the following "
    "question concisely.\n\n"
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


def _get_frames_dir(data_dir: str) -> str:
    if (os.path.exists(os.path.join(data_dir, "hm3d-v0")) or
            os.path.exists(os.path.join(data_dir, "scannet-v0"))):
        return data_dir
    frames_dir = os.path.join(data_dir, "open-eqa", "data", "frames")
    if os.path.exists(frames_dir):
        return frames_dir
    return data_dir


def _sample_frames(frames_dir: str, episode_history: str, num_frames: int = 32) -> List[str]:
    folder = Path(frames_dir) / episode_history
    if not folder.exists():
        return []
    frames = sorted(folder.glob("*-rgb.png"))
    if not frames:
        frames = sorted(folder.glob("*.png"))
        frames = [f for f in frames if "depth" not in f.name.lower()]
    if not frames:
        return []
    frame_paths = [str(p) for p in frames]
    return _sample_sequence_midpoint(frame_paths, num_frames)


def load_openeqa_samples(
    data_dir: Optional[str] = None,
    num_frames: int = 32,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Load OpenEQA samples and construct routing-ready prompts.

    Returns list of dicts with keys:
        sample_id, benchmark, prompt, media_type, media_paths
    """
    data_dir = data_dir or DEFAULT_DATA_DIR

    possible_paths = [
        os.path.join(data_dir, "open-eqa-v0.json"),
        os.path.join(data_dir, "open-eqa", "data", "open-eqa-v0.json"),
        os.path.join(data_dir, "data", "open-eqa-v0.json"),
    ]
    dataset_path = None
    for p in possible_paths:
        if os.path.exists(p):
            dataset_path = p
            break
    if dataset_path is None:
        raise FileNotFoundError(f"Could not find open-eqa-v0.json in: {possible_paths}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    frames_dir = _get_frames_dir(data_dir)

    samples = []
    for item in dataset:
        question_id = item.get("question_id", len(samples))
        question = item.get("question", "")
        episode_history = item.get("episode_history", "")

        frame_paths = _sample_frames(frames_dir, episode_history, num_frames)
        if not frame_paths:
            continue

        prompt = VQA_PROMPT_TEMPLATE.format(question=question)

        samples.append({
            "sample_id": str(question_id),
            "benchmark": "openeqa",
            "prompt": prompt,
            "media_type": "video_frames",
            "media_paths": frame_paths,
        })

        if max_samples and len(samples) >= max_samples:
            break

    return samples
