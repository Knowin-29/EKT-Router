"""CPU-only demonstration of bounded propose--verify--commit routing."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.evidence_contract import EvidenceChannel, EvidencePacket  # noqa: E402
from modules.pipeline import EKTRouterPipeline  # noqa: E402
from modules.utility_router import (  # noqa: E402
    ExpertSpec,
    InteractionAwareUtilityRouter,
    RouteAction,
    UtilityTerms,
)


class DemoBackend:
    def encode_query(self, question: str, image: Any = None) -> str:
        return question.lower()

    def score_route(
        self,
        query_state: str,
        expert: ExpertSpec,
        action: RouteAction,
    ) -> UtilityTerms:
        spatial_query = any(word in query_state for word in ("where", "between", "left"))
        competence = 1.0 + float(spatial_query and expert.name == "Spatial Reasoning")
        evidence = 0.4 * float(not action.is_null)
        interaction = 0.8 * float(
            expert.name == "Spatial Reasoning" and action.tool == "ground"
        )
        cost = 0.1 * (int(action.knowledge != "none") + int(action.tool != "none"))
        return UtilityTerms(competence, evidence, interaction, cost)

    def post_acquisition_adjustment(
        self,
        query_state: str,
        expert: ExpertSpec,
        action: RouteAction,
        packets: Mapping[EvidenceChannel, EvidencePacket],
    ) -> float:
        return sum(float(packet.metadata.get("adjustment", 0.0)) for packet in packets.values())


class DemoBackbone:
    def set_adapter(self, expert: str) -> None:
        self.expert = expert

    def generate_answer(self, prompt: str, image: Any, adapter_id: int) -> str:
        return f"expert={self.expert}; adapter={adapter_id}; prompt={prompt}"


class NoOpProvider:
    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return ""


def packet_provider(proposal: Any, question: str, image: Any):
    packets = {}
    if proposal.action.knowledge != "none":
        packets[EvidenceChannel.KNOWLEDGE] = EvidencePacket(
            channel=EvidenceChannel.KNOWLEDGE,
            action=proposal.action.knowledge,
            content="Use an object-centric spatial reference frame.",
            coverage=1.0,
            confidence=0.9,
            ambiguity=0.0,
            provenance=("demo-knowledge-v1",),
            metadata={"adjustment": 0.1},
        )
    if proposal.action.tool != "none":
        packets[EvidenceChannel.TOOL] = EvidencePacket(
            channel=EvidenceChannel.TOOL,
            action=proposal.action.tool,
            content="Detected mug centers at (0.33, 0.52) and (0.67, 0.51).",
            coverage=1.0,
            confidence=0.95,
            ambiguity=0.0,
            provenance=("demo-grounding-v1",),
            metadata={"adjustment": 0.2},
        )
    return packets


def main() -> None:
    print("MOCK CPU DEMO — synthetic signatures and packets; no model inference or benchmark score.")
    experts = (
        ExpertSpec("General", 0, "general", (0.8, 0.4, 0.3)),
        ExpertSpec("Spatial Reasoning", 1, "spatial_reasoning", (0.5, 0.9, 0.4)),
        ExpertSpec("Spatiotemporal", 2, "spatiotemporal", (0.5, 0.5, 0.9)),
    )
    router = InteractionAwareUtilityRouter(
        backend=DemoBackend(),
        experts=experts,
        knowledge_actions=("none", "template"),
        tool_actions=("none", "ground"),
        require_provenance=True,
    )
    pipeline = EKTRouterPipeline(
        backbone=DemoBackbone(),
        adapter_paths={"spatial_reasoning": "demo"},
        router_agent=router,
        knowledge_module=NoOpProvider(),
        tool_module=NoOpProvider(),
    )
    print(
        pipeline.run_bounded(
            "Identify the vacant space between the two mugs.",
            image=None,
            packet_provider=packet_provider,
        )
    )


if __name__ == "__main__":
    main()
