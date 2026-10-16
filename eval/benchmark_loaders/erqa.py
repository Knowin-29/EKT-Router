"""ERQA benchmark loader for routing evaluation.

ERQA uses TFRecord format. If tensorflow is not available, this loader
will raise an ImportError with installation instructions.
"""

import base64
import io
import os
from typing import Dict, List, Optional

from PIL import Image


DEFAULT_TFRECORD_PATH = "data/benchmarks/erqa/erqa.tfrecord"


def _pil_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_erqa_samples(
    data_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    **kwargs,
) -> List[Dict]:
    """Load ERQA samples from TFRecord and construct routing-ready prompts.

    The prompt is the question text from the TFRecord, with images interleaved
    at positions specified by visual_indices. For routing, we flatten this into
    a text prompt + list of base64 images.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "ERQA loader requires tensorflow. Install with: pip install tensorflow"
        )

    tfrecord_path = data_dir or DEFAULT_TFRECORD_PATH
    if os.path.isdir(tfrecord_path):
        tfrecord_path = os.path.join(tfrecord_path, "data", "erqa.tfrecord")

    feature_description = {
        "answer": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.VarLenFeature(tf.string),
        "question_type": tf.io.VarLenFeature(tf.string),
        "visual_indices": tf.io.VarLenFeature(tf.int64),
        "question": tf.io.FixedLenFeature([], tf.string),
    }

    def parse_example(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        parsed["visual_indices"] = tf.sparse.to_dense(parsed["visual_indices"])
        parsed["image/encoded"] = tf.sparse.to_dense(parsed["image/encoded"])
        parsed["question_type"] = tf.sparse.to_dense(parsed["question_type"])
        return parsed

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example)

    samples = []
    for i, example in enumerate(dataset):
        question = example["question"].numpy().decode("utf-8")
        images_encoded = example["image/encoded"].numpy()

        pil_images = []
        for img_enc in images_encoded:
            img_tensor = tf.io.decode_image(img_enc).numpy()
            pil_images.append(Image.fromarray(img_tensor))

        image_b64_list = [_pil_to_b64(img) for img in pil_images]

        samples.append({
            "sample_id": f"erqa_{i}",
            "benchmark": "erqa",
            "prompt": question,
            "media_type": "base64_image" if len(image_b64_list) == 1 else "base64_images",
            "media_paths": [],
            "media_base64": image_b64_list[0] if len(image_b64_list) == 1 else "",
            "media_base64_list": image_b64_list,
        })

        if max_samples and len(samples) >= max_samples:
            break

    return samples
