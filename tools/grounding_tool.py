"""GroundingDINO wrapper for visual grounding as textual context.

Prompts the model with the natural-language question (or a comma-separated
noun phrase if available) and returns the top-k boxes as a textual summary
that the routed LoRA expert can consume.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "in", "on", "to", "and", "or",
    "with", "what", "where", "which", "how", "many", "much",
}


def _image_to_pil(image):
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


def _extract_noun_phrase(question: str) -> str:
    q = re.sub(r"[^a-zA-Z0-9\s]", " ", question or "").lower()
    tokens = [t for t in q.split() if t and t not in _STOPWORDS]
    if not tokens:
        return question or ""
    return " . ".join(tokens[:5])


class GroundingDinoTool:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        local_path: str = "",
        device: str = "cuda",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        top_k: int = 5,
        dry_run: bool = False,
    ) -> None:
        self.model_id = local_path or model_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.top_k = top_k
        self.dry_run = dry_run
        self._processor = None
        self._model = None
        self._failed = False

    def _load(self) -> bool:
        if self._processor is not None:
            return True
        if self._failed or self.dry_run:
            return False
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoProcessor,
                AutoModelForZeroShotObjectDetection,
            )

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
            if self.device and torch.cuda.is_available():
                self._model = self._model.to(self.device)
            self._model.eval()
            return True
        except Exception as e:
            logger.warning("GroundingDINO load failed (%s): %s", self.model_id, e)
            self._failed = True
            return False

    def __call__(self, image, question: str = "") -> str:
        if self.dry_run:
            return "[ground:dry-run] object@[0.40,0.55,0.50,0.65] score=0.82"
        img = _image_to_pil(image)
        if img is None or not self._load():
            return ""
        try:
            import torch  # type: ignore

            prompt = _extract_noun_phrase(question) + " ."
            inputs = self._processor(images=img, text=prompt, return_tensors="pt")
            if self.device and hasattr(next(self._model.parameters()), "is_cuda"):
                inputs = {k: v.to(next(self._model.parameters()).device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            W, H = img.size
            target_sizes = torch.tensor([[H, W]])
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.get("input_ids") if isinstance(inputs, dict) else inputs["input_ids"],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes.to(outputs.logits.device),
            )
            if not results:
                return ""
            res = results[0]
            boxes = res.get("boxes")
            scores = res.get("scores")
            labels = res.get("text_labels", res.get("labels", []))
            if boxes is None or len(boxes) == 0:
                return ""
            lines: List[str] = ["[Grounding]"]
            order = scores.argsort(descending=True) if hasattr(scores, "argsort") else list(range(len(scores)))
            for rank, i in enumerate(list(order)[: self.top_k]):
                i = int(i)
                x0, y0, x1, y1 = [float(v) for v in boxes[i].tolist()]
                s = float(scores[i])
                lab = labels[i] if i < len(labels) else ""
                lines.append(
                    f"{rank+1}. '{lab}' score={s:.2f} "
                    f"bbox=[{x0/W:.2f},{y0/H:.2f},{x1/W:.2f},{y1/H:.2f}]"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.warning("GroundingDINO inference failed: %s", e)
            return ""
