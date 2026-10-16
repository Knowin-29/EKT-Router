"""Public capability taxonomy used by EKT-Router.

The names describe deployment roles, not a hierarchy. Their outcome-derived
signatures may overlap and can reverse rank across diagnostic capability axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


EXPERT_IDS: Tuple[str, ...] = (
    "general",
    "spatial_reasoning",
    "spatiotemporal",
)


@dataclass(frozen=True)
class ExpertCapability:
    id: str
    index: int
    name: str
    description: str
    keywords: Tuple[str, ...]


EXPERTS: Dict[str, ExpertCapability] = {
    "general": ExpertCapability(
        id="general",
        index=0,
        name="General",
        description=(
            "Broad visual-language understanding, object and attribute recognition, "
            "and open-ended scene question answering."
        ),
        keywords=(
            "what color",
            "what is",
            "which object",
            "describe",
            "scene",
            "name",
            "type",
            "category",
        ),
    ),
    "spatial_reasoning": ExpertCapability(
        id="spatial_reasoning",
        index=1,
        name="Spatial Reasoning",
        description=(
            "Pointing, grounding, metric and relational spatial reasoning, and "
            "spatial affordance prediction."
        ),
        keywords=(
            "point",
            "locate",
            "position",
            "coordinate",
            "bounding box",
            "where is",
            "distance",
            "near",
            "left",
            "right",
            "behind",
            "front",
            "above",
            "below",
            "route",
            "path",
            "depth",
            "place",
        ),
    ),
    "spatiotemporal": ExpertCapability(
        id="spatiotemporal",
        index=2,
        name="Spatiotemporal",
        description=(
            "Temporal ordering, state transitions, action understanding, and "
            "reasoning over multi-frame or video observations."
        ),
        keywords=(
            "before",
            "after",
            "then",
            "sequence",
            "next",
            "previous",
            "temporal",
            "action",
            "step",
            "first",
            "last",
            "video",
        ),
    ),
}

KNOWLEDGE_MODES: Tuple[str, ...] = ("none", "template", "rag")
TOOL_MODES: Tuple[str, ...] = ("none", "depth", "ground", "ocr")


def expert_by_index(index: int) -> ExpertCapability:
    for expert in EXPERTS.values():
        if expert.index == index:
            return expert
    raise ValueError(f"unknown expert index: {index}")


def source_to_expert(source: str) -> str:
    """Map a public data source label to one of the three expert roles."""

    value = source.strip().lower()
    if any(token in value for token in ("video", "temporal", "sequence")):
        return "spatiotemporal"
    if any(token in value for token in ("spatial", "point", "ground", "depth")):
        return "spatial_reasoning"
    return "general"


def keyword_to_expert(question: str) -> str:
    """Deterministic fallback used only by smoke tests and diagnostics."""

    text = (question or "").lower()
    for expert_id in ("spatiotemporal", "spatial_reasoning"):
        if any(keyword in text for keyword in EXPERTS[expert_id].keywords):
            return expert_id
    return "general"


def router_system_prompt() -> str:
    lines = [
        "You route an embodied vision-language query to one capability expert,",
        "an optional knowledge source, and an optional perceptual tool.",
        "",
        "[Capability Experts]",
    ]
    for expert_id in EXPERT_IDS:
        expert = EXPERTS[expert_id]
        lines.append(f"- {expert.id}: {expert.description}")
    lines.extend(
        [
            "",
            "[Knowledge]",
            "- none: use no retrieved knowledge.",
            "- template: retrieve a structured reasoning prior.",
            "- rag: retrieve relevant scene or commonsense evidence.",
            "",
            "[Perceptual Tools]",
            "- none: invoke no perceptual tool.",
            "- depth: invoke monocular depth estimation.",
            "- ground: invoke open-vocabulary visual grounding.",
            "",
            "Return one JSON object with keys expert, knowledge, and tool.",
        ]
    )
    return "\n".join(lines)


def all_experts() -> List[ExpertCapability]:
    return [EXPERTS[key] for key in EXPERT_IDS]


# Compatibility aliases kept for the copied evaluation adapters. New code
# should use the expert-oriented names above.
LEVEL_IDS = EXPERT_IDS
LEVELS = EXPERTS
level_by_index = expert_by_index
keyword_to_level = keyword_to_expert
dir_to_level = source_to_expert
all_levels_as_list = all_experts
