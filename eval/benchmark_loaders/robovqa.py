"""RoboVQA benchmark loader for routing evaluation."""

import json
import os
import re
from glob import glob
from typing import Dict, List, Optional


DEFAULT_DATA_DIR = "data/benchmarks/robovqa"

TAGS_RE = r"(</?[\w:]+>)"
TASK_RE = r"(<task[\:\w]*>)"


def _remove_tags(text: str) -> str:
    return re.sub(TAGS_RE, "", text)


def _extract_binary_answer(text: str) -> Optional[str]:
    binary_match = re.search(r"<PRED:BINARY>(.*?)</PRED:BINARY>", text, re.IGNORECASE)
    if binary_match:
        return binary_match.group(1).strip().lower()
    clean_text = _remove_tags(text).strip().lower()
    if clean_text in ("yes", "no"):
        return clean_text
    return None


def _parse_sample(sample: Dict) -> List[Dict]:
    text = sample.get("text", "")
    uid = sample.get("uid", "")
    video = sample.get("video", "")
    results = []

    task_splits = re.split(TASK_RE, text, flags=re.IGNORECASE)
    i = 1
    while i < len(task_splits) - 1:
        task_tag = task_splits[i].strip()
        task_content = task_splits[i + 1].strip() if i + 1 < len(task_splits) else ""
        task_type = task_tag.replace("<", "").replace(">", "").lower()
        is_binary = any(k in task_type.lower() for k in (
            ":discrete:true", ":discrete:false", ":binary"
        ))

        if "Q:" in task_content and "A:" in task_content:
            q_idx = task_content.find("Q:")
            a_idx = task_content.find("A:")
            if q_idx < a_idx:
                raw_question = _remove_tags(task_content[q_idx + 2:a_idx].strip()).strip()
                answer_raw = task_content[a_idx + 2:].strip()
                if is_binary:
                    answer = _extract_binary_answer(answer_raw)
                    if answer is None:
                        answer = _remove_tags(answer_raw).strip()
                else:
                    answer = _remove_tags(answer_raw).strip()

                pre_question = task_content[:q_idx].strip()
                context = ""
                goal = ""

                if "current goal is:" in pre_question.lower():
                    goal_match = re.search(
                        r"current goal is:\s*(.+?)(?:\.\s*last|\s*Q:|$)",
                        pre_question, re.IGNORECASE | re.DOTALL,
                    )
                    if goal_match:
                        goal = goal_match.group(1).strip().rstrip(".")
                    history_match = re.search(
                        r"last \d+ steps?:\s*(.+?)(?:\s*Q:|$)",
                        pre_question, re.IGNORECASE | re.DOTALL,
                    )
                    history = history_match.group(1).strip() if history_match else ""
                    if goal and history:
                        context = f"Goal: {goal}. Recent steps: {history}"
                    elif goal:
                        context = f"Goal: {goal}"
                elif pre_question and ("success" in task_type or "affordance:discriminative" in task_type):
                    goal = _remove_tags(pre_question).strip()
                    if goal:
                        context = f"Goal: {goal}"

                if context:
                    full_question = f"{context}\nQuestion: {raw_question}"
                else:
                    full_question = raw_question

                results.append({
                    "uid": uid,
                    "video": video,
                    "task_type": task_type,
                    "question": full_question,
                    "context": context,
                    "is_binary": is_binary,
                })
        i += 2
    return results


def load_robovqa_samples(
    data_dir: Optional[str] = None,
    split: str = "val",
    max_samples: Optional[int] = None,
    **kwargs,
) -> List[Dict]:
    """Load RoboVQA samples and construct routing-ready prompts."""
    data_dir = data_dir or DEFAULT_DATA_DIR
    json_dir = os.path.join(data_dir, "json", split)
    video_dir = os.path.join(data_dir, "videos")

    if not os.path.exists(json_dir):
        raise FileNotFoundError(f"Data directory not found: {json_dir}")

    json_files = sorted(glob(os.path.join(json_dir, "data-*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {json_dir}")

    raw_samples = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        raw_samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    available_videos = set(os.listdir(video_dir)) if os.path.exists(video_dir) else set()

    all_qa = []
    for raw in raw_samples:
        parsed = _parse_sample(raw)
        for qa in parsed:
            if qa["video"] in available_videos:
                all_qa.append(qa)

    samples = []
    for idx, qa in enumerate(all_qa):
        if qa.get("context"):
            prompt = f"Context: {qa['context']}\nQuestion: {qa['question']}\nAnswer:"
        else:
            prompt = f"Question: {qa['question']}\nAnswer:"

        video_path = os.path.join(video_dir, qa["video"])

        samples.append({
            "sample_id": f"{qa['uid']}_{idx}",
            "benchmark": "robovqa",
            "prompt": prompt,
            "media_type": "video",
            "media_paths": [video_path],
        })

        if max_samples and len(samples) >= max_samples:
            break

    return samples
