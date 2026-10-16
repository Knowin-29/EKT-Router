"""CV-Bench benchmark loader for routing evaluation."""

import base64
import io
import os
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image


DEFAULT_DATA_DIR = "data/benchmarks/cv-bench"


def _encode_image_to_base64(image) -> str:
    if isinstance(image, dict) and "bytes" in image:
        pil_image = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        pil_image = image

    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def load_cv_bench_samples(
    data_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    **kwargs,
) -> List[Dict]:
    """Load CV-Bench samples and construct routing-ready prompts.

    The prompt field comes directly from the parquet dataset.
    Images are stored as base64 strings.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR

    df_2d = pd.read_parquet(os.path.join(data_dir, "test_2d.parquet"))
    df_3d = pd.read_parquet(os.path.join(data_dir, "test_3d.parquet"))
    df = pd.concat([df_2d, df_3d], ignore_index=True)

    samples = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        image = row["image"]
        idx = row["idx"]

        b64 = _encode_image_to_base64(image)

        samples.append({
            "sample_id": str(idx),
            "benchmark": "cv_bench",
            "prompt": prompt,
            "media_type": "base64_image",
            "media_paths": [],
            "media_base64": b64,
        })

        if max_samples and len(samples) >= max_samples:
            break

    return samples
