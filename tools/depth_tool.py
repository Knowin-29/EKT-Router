"""DepthAnything V2 wrapper.

- Loads a DepthAnything model from HuggingFace (or a local path).
- Produces a textualised summary (center-point depth, histogram over depth
  bands) that the routed LoRA expert can consume as early-fusion prompt
  context.
- Falls back to an empty string on any error so the pipeline keeps running.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _load_image(image) -> Optional["object"]:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None
    if image is None:
        return None
    if hasattr(image, "convert"):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        p = Path(image)
        if p.exists():
            return Image.open(p).convert("RGB")
    return None


class DepthAnythingTool:
    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        local_path: str = "",
        device: str = "cuda",
        dry_run: bool = False,
    ) -> None:
        self.model_id = local_path or model_id
        self.device = device
        self.dry_run = dry_run
        self._pipe = None
        self._failed = False

    def _load(self) -> bool:
        if self._pipe is not None:
            return True
        if self._failed or self.dry_run:
            return False
        try:
            from transformers import pipeline  # type: ignore

            self._pipe = pipeline(
                "depth-estimation", model=self.model_id, device=self.device
            )
            return True
        except Exception as e:
            logger.warning("DepthAnything load failed (%s): %s", self.model_id, e)
            self._failed = True
            return False

    def __call__(self, image, question: str = "") -> str:
        if self.dry_run:
            return "[depth:dry-run] relative_values=[0.2,0.5,0.8] (synthetic)"
        img = _load_image(image)
        if img is None or not self._load():
            return ""
        try:
            out = self._pipe(img)
            import numpy as np  # type: ignore

            depth = out.get("predicted_depth")
            if hasattr(depth, "numpy"):
                depth = depth.detach().cpu().numpy()
            depth = np.asarray(depth, dtype="float32").squeeze()
            if depth.ndim == 3:
                depth = depth.squeeze()
            h, w = depth.shape[:2]
            center = float(depth[h // 2, w // 2])
            bands = [float(np.quantile(depth, q)) for q in (0.1, 0.5, 0.9)]
            return (
                "[Relative inverse-depth summary; uncalibrated units]\n"
                f"center_depth={center:.4f}; "
                f"p10={bands[0]:.4f}; median={bands[1]:.4f}; p90={bands[2]:.4f}"
            )
        except Exception as e:
            logger.warning("DepthAnything inference failed: %s", e)
            return ""
