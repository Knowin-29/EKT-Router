"""Tool module: routes perceptual-tool calls to depth / grounding backends."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ToolModule:
    def __init__(self, depth_tool=None, grounding_tool=None, dry_run: bool = False) -> None:
        self.depth_tool = depth_tool
        self.grounding_tool = grounding_tool
        self.dry_run = dry_run

    def __call__(self, image, question: str, mode: str) -> str:
        mode = (mode or "none").lower()
        if self.dry_run:
            return f"[tool:{mode}:dry-run]" if mode != "none" else ""
        try:
            if mode == "depth" and self.depth_tool is not None:
                return self.depth_tool(image, question) or ""
            if mode == "ground" and self.grounding_tool is not None:
                return self.grounding_tool(image, question) or ""
        except Exception as e:
            logger.warning("tool module failure (mode=%s): %s", mode, e)
        return ""
