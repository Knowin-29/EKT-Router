"""VABench-Point benchmark loader for routing evaluation."""

import base64
import io
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_DATA_PATH = (
    "data/benchmarks/vabench/vabench_point_dataset.parquet"
)

CUSTOM_PROMPT = "Your answer should contain exactly ONE point and is formatted as follows [[x1, y1]]."


def _encode_image(img_data) -> str:
    if img_data is None:
        return ""
    if isinstance(img_data, dict) and "bytes" in img_data:
        pil_img = Image.open(io.BytesIO(img_data["bytes"]))
    elif isinstance(img_data, bytes):
        pil_img = Image.open(io.BytesIO(img_data))
    elif isinstance(img_data, Image.Image):
        pil_img = img_data
    elif isinstance(img_data, np.ndarray):
        pil_img = Image.fromarray(img_data)
    elif isinstance(img_data, str):
        if os.path.exists(img_data):
            pil_img = Image.open(img_data)
        else:
            try:
                pil_img = Image.open(io.BytesIO(base64.b64decode(img_data)))
            except Exception:
                return ""
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


def load_vabench_point_samples(
    data_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    **kwargs,
) -> List[Dict]:
    """Load VABench-Point samples and construct routing-ready prompts."""
    parquet_path = data_dir or DEFAULT_DATA_PATH
    if os.path.isdir(parquet_path):
        parquet_path = os.path.join(parquet_path, "data", "vabench_point_dataset.parquet")

    df = pd.read_parquet(parquet_path)

    samples = []
    for i, row in df.iterrows():
        instruction = (
            row.get("problem")
            or row.get("instruction", "")
            or str(row.get("question", ""))
        ).strip()
        prompt = f"{instruction} {CUSTOM_PROMPT}"

        images = row.get("images", None)
        image_b64_list = []
        if images is not None:
            if isinstance(images, (list, np.ndarray)):
                for img in images:
                    b64 = _encode_image(img)
                    if b64:
                        image_b64_list.append(b64)
            else:
                b64 = _encode_image(images)
                if b64:
                    image_b64_list.append(b64)

        samples.append({
            "sample_id": f"vabench_{i}",
            "benchmark": "vabench_point",
            "prompt": prompt,
            "media_type": "base64_image" if len(image_b64_list) == 1 else "base64_images",
            "media_paths": [],
            "media_base64": image_b64_list[0] if len(image_b64_list) == 1 else "",
            "media_base64_list": image_b64_list,
        })

        if max_samples and len(samples) >= max_samples:
            break

    return samples
