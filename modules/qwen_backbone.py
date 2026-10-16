"""Local Qwen3-VL execution with explicit LoRA activation and no mock fallback."""

from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
from typing import Any


class QwenRoutedBackbone:
    def __init__(
        self,
        backbone_path,
        adapter_paths=None,
        device="cuda",
        torch_dtype="bfloat16",
        max_new_tokens=256,
        temperature=0.0,
        max_video_frames=32,
    ):
        self.backbone_path = str(backbone_path)
        self.adapter_paths = dict(adapter_paths or {})
        self.device, self.torch_dtype = device, torch_dtype
        self.max_new_tokens, self.temperature = int(max_new_tokens), float(temperature)
        self.max_video_frames = int(max_video_frames)
        if self.max_video_frames < 1 or self.max_new_tokens < 1:
            raise ValueError("media and generation budgets must be positive")
        self._model = self._processor = None
        self._active_adapter = "backbone"
        self._ready = False

    def _load(self):
        if self._ready:
            return True
        if not Path(self.backbone_path).is_dir():
            raise FileNotFoundError("real inference requires an existing local Qwen3-VL checkpoint")
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.backbone_path, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.backbone_path,
            torch_dtype=getattr(torch, self.torch_dtype),
            device_map=self.device,
            local_files_only=True,
        )
        # Preserve the requested adapter across lazy loading.
        if self.adapter_paths:
            from peft import PeftModel

            for index, (name, path) in enumerate(self.adapter_paths.items()):
                if not Path(path).is_dir():
                    raise FileNotFoundError(f"missing local adapter: {name}")
                if index == 0:
                    model = PeftModel.from_pretrained(
                        model, path, adapter_name=name, is_trainable=False, local_files_only=True
                    )
                else:
                    model.load_adapter(
                        path, adapter_name=name, is_trainable=False, local_files_only=True
                    )
        self._model = model.eval()
        self._model.requires_grad_(False)
        if self._active_adapter != "backbone":
            self._model.set_adapter(self._active_adapter)
        self._ready = True
        return True

    def set_adapter(self, name):
        if name != "backbone" and name not in self.adapter_paths:
            raise ValueError(f"expert adapter is not configured: {name}")
        if self._ready and name != "backbone":
            self._model.set_adapter(name)
        self._active_adapter = name

    def load_adapter(self, path, name):
        if not Path(path).is_dir():
            raise FileNotFoundError(f"missing local adapter: {name}")
        if self._ready:
            self._model.load_adapter(
                path, adapter_name=name, is_trainable=False, local_files_only=True
            )
        self.adapter_paths[name] = str(path)

    def _build_messages(self, prompt, image):
        content = []
        if isinstance(image, list):
            # Uniform frame subsampling is deterministic and bounded.
            n = min(len(image), self.max_video_frames)
            indices = [round(i * (len(image) - 1) / max(n - 1, 1)) for i in range(n)]
            content.extend({"type": "image", "image": image[i]} for i in indices)
        elif image is not None:
            video = isinstance(image, (str, Path)) and Path(image).suffix.lower() in {
                ".mp4",
                ".avi",
                ".mov",
                ".webm",
                ".mkv",
            }
            if video:
                content.append(
                    {"type": "video", "video": str(image), "nframes": self.max_video_frames}
                )
            else:
                content.append(
                    {"type": "image", "image": str(image) if isinstance(image, Path) else image}
                )
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _inputs(self, prompt, image):
        from qwen_vl_utils import process_vision_info

        messages = self._build_messages(prompt, image)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages)
        batch = self._processor(
            text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
        )
        return {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in batch.items()}

    def generate_answer(self, prompt, image=None, adapter_id=0):
        self._load()  # errors propagate: a failed model is never a benchmark prediction
        import torch

        inputs = self._inputs(prompt, image)
        context = (
            self._model.disable_adapter()
            if self._active_adapter == "backbone" and self.adapter_paths
            else nullcontext()
        )
        kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0}
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature
        with context, torch.inference_mode():
            out = self._model.generate(**inputs, **kwargs)
        text = self._processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        if not text:
            raise RuntimeError("model produced an empty answer")
        return text


class FrozenQwenDescriptor(QwenRoutedBackbone):
    """Frozen 2B last-layer masked-mean descriptor; does not generate answers.

    The pooling convention must be recorded with any trained router checkpoint.
    This release supplies the convention, not the missing experimental checkpoint.
    """

    def __call__(self, question: str, image: Any = None):
        self._load()
        import torch

        inputs = self._inputs(question, image)
        with torch.inference_mode():
            out = self._model.model(**inputs, return_dict=True)
            hidden = out.last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1)
            return ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1))[0].cpu()
