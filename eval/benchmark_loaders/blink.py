"""BLINK benchmark loader for routing evaluation."""

import base64
import io
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_DATA_DIR = "data/benchmarks/blink"

BLINK_TASKS = [
    "Multi-view_Reasoning",
    "Relative_Depth",
    "Spatial_Relation",
    "Object_Localization",
]


def _encode_image(img_data) -> str:
    if img_data is None:
        return ""
    if isinstance(img_data, dict) and "bytes" in img_data:
        pil_img = Image.open(io.BytesIO(img_data["bytes"]))
    elif isinstance(img_data, Image.Image):
        pil_img = img_data
    else:
        return ""

    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    if max(w, h) > 1024:
        scale = 1024 / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_blink_samples(
    data_dir: Optional[str] = None,
    split: str = "val",
    max_samples: Optional[int] = None,
    **kwargs,
) -> List[Dict]:
    """Load BLINK samples and construct routing-ready prompts."""
    data_dir = data_dir or DEFAULT_DATA_DIR

    samples = []
    for task_name in BLINK_TASKS:
        parquet_path = os.path.join(data_dir, task_name, f"{split}-00000-of-00001.parquet")
        if not os.path.exists(parquet_path):
            continue

        df = pd.read_parquet(parquet_path)

        for _, row in df.iterrows():
            idx = row.get("idx", "")
            prompt_text = row.get("prompt", "")
            question = row.get("question", "")
            choices = row.get("choices", "")

            full_prompt = prompt_text if prompt_text else question
            if choices is not None:
                if isinstance(choices, str) and choices:
                    full_prompt += f"\n{choices}"
                elif isinstance(choices, (list, np.ndarray)):
                    opts = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
                    full_prompt += f"\n{opts}"

            image_b64_list = []
            for img_key in ("image_1", "image_2", "image_3", "image_4"):
                img = row.get(img_key)
                if img is not None:
                    b64 = _encode_image(img)
                    if b64:
                        image_b64_list.append(b64)

            samples.append({
                "sample_id": f"blink_{task_name}_{idx}",
                "benchmark": "blink",
                "prompt": full_prompt,
                "media_type": "base64_images",
                "media_paths": [],
                "media_base64_list": image_b64_list,
            })

            if max_samples and len(samples) >= max_samples:
                return samples

    return samples
