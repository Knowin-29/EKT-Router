"""Candidate-generation adapter for route-outcome collection.

Applications may inject an initialized ``EKTRouterPipeline``. Without one,
the function returns a deterministic marker used only by smoke tests. A real
backend should preserve this signature:

    def generate_candidate(question: str, image: str,
                           adapter: int, knowledge: str, tool: str) -> str

which returns the textual answer produced by running the routed pipeline
(backbone + LoRA adapter + optional support modules) on the sample.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def generate_candidate(
    question: str,
    image: Any = None,
    adapter: int = 1,
    knowledge: str = "none",
    tool: str = "none",
    pipeline: Optional[Any] = None,
) -> str:
    """Delegate to an initialized routed pipeline.

    The production path injects a pre-built :class:`modules.pipeline.EKTRouterPipeline`
    via ``pipeline=``. The marker path never represents a benchmark result.
    """
    if pipeline is not None:
        try:
            return pipeline(
                question=question,
                image=image,
                override_decision={
                    "adapter": int(adapter),
                    "knowledge": knowledge,
                    "tool": tool,
                },
            )
        except Exception as e:
            logger.warning("pipeline candidate generation failed: %s", e)
    raise RuntimeError(
        "real candidate generation requires an initialized EKTRouterPipeline"
    )
