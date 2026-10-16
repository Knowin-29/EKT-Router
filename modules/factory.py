"""Helpers for building pipeline components from ``configs/global.yaml``."""

from __future__ import annotations

from pathlib import Path
import os
import re
from typing import Any, Dict

import yaml

from .knowledge_module import KnowledgeModule
from .tool_module import ToolModule


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _local_path(value: Any) -> str:
    if value in (None, ""):
        return ""
    path = Path(str(value)).expanduser()
    return str(path if path.is_absolute() else _PROJECT_ROOT / path)


def load_global_config(path: str | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else _PROJECT_ROOT / "configs" / "global.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        text = f.read()
    def expand(match):
        name, default = match.group(1), match.group(2)
        value = os.environ.get(name, default)
        if value is None:
            raise ValueError(f"required configuration environment variable is missing: {name}")
        return value
    return yaml.safe_load(re.sub(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}", expand, text)) or {}


def build_knowledge(cfg: Dict[str, Any], dry_run: bool = False) -> KnowledgeModule:
    know = cfg.get("knowledge", {})
    return KnowledgeModule(
        template_db_path=_local_path(
            know.get("template_db", "knowledge/template_db.json")
        ),
        rag_index_dir=_local_path(know.get("rag_index_dir", "data/knowledge_index")),
        embedding_model_name=str(know.get("embedding_model", "BAAI/bge-small-en-v1.5")),
        top_n=int(know.get("top_k", 3)),
        dry_run=dry_run,
    )


def build_tool(cfg: Dict[str, Any], dry_run: bool = False) -> ToolModule:
    from tools.depth_tool import DepthAnythingTool
    from tools.grounding_tool import GroundingDinoTool

    tools_cfg = cfg.get("tools", {}) or {}
    depth_cfg = tools_cfg.get("depth", {}) or {}
    ground_cfg = tools_cfg.get("grounding", {}) or {}

    depth = DepthAnythingTool(
        model_id=str(depth_cfg.get("model_id", "depth-anything/Depth-Anything-V2-Small-hf")),
        local_path=_local_path(depth_cfg.get("local_path")),
        dry_run=dry_run,
    )
    ground = GroundingDinoTool(
        model_id=str(ground_cfg.get("model_id", "IDEA-Research/grounding-dino-tiny")),
        local_path=_local_path(ground_cfg.get("local_path")),
        box_threshold=float(ground_cfg.get("box_threshold", 0.3)),
        text_threshold=float(ground_cfg.get("text_threshold", 0.25)),
        dry_run=dry_run,
    )
    return ToolModule(depth_tool=depth, grounding_tool=ground, dry_run=dry_run)
