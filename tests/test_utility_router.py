from __future__ import annotations

import math
import unittest
from typing import Any, Mapping

from modules.evidence_contract import EvidenceChannel, EvidencePacket
from modules.pipeline import EKTRouterPipeline
from modules.utility_router import (
    ExpertSpec,
    InteractionAwareUtilityRouter,
    RouteAction,
    UtilityTerms,
)


class ToyBackend:
    def encode_query(self, question: str, image: Any = None) -> str:
        return question

    def score_route(
        self,
        query_state: str,
        expert: ExpertSpec,
        action: RouteAction,
    ) -> UtilityTerms:
        competence = {"General": 1.0, "Spatial Reasoning": 2.0}[expert.name]
        evidence = float(action.knowledge != "none") + float(action.tool != "none")
        interaction = 0.0
        if expert.name == "Spatial Reasoning" and action.knowledge != "none":
            interaction += 0.5
        if expert.name == "Spatial Reasoning" and action.tool != "none":
            interaction += 1.0
        return UtilityTerms(competence, evidence, interaction, cost=0.0)

    def post_acquisition_adjustment(
        self,
        query_state: str,
        expert: ExpertSpec,
        action: RouteAction,
        packets: Mapping[EvidenceChannel, EvidencePacket],
    ) -> float:
        return sum(float(packet.metadata.get("adjustment", 0.0)) for packet in packets.values())


def experts():
    return (
        ExpertSpec("General", 0, "general", (0.8, 0.2)),
        ExpertSpec("Spatial Reasoning", 1, "spatial_reasoning", (0.2, 0.9)),
    )


class EvidencePacketTest(unittest.TestCase):
    def test_invalid_non_null_packet_fails_closed(self):
        packet = EvidencePacket(
            channel=EvidenceChannel.TOOL,
            action="ground",
            content="one candidate",
            coverage=0.4,
            confidence=0.9,
            ambiguity=0.8,
            provenance=(),
        )
        validity = packet.validate(
            min_coverage=0.5,
            max_ambiguity=0.5,
            require_provenance=True,
        )
        self.assertFalse(validity.valid)
        self.assertIn("insufficient_coverage", validity.reasons)
        self.assertIn("high_ambiguity", validity.reasons)
        self.assertIn("missing_provenance", validity.reasons)

    def test_non_finite_packet_quality_fails_closed(self):
        fields = ("coverage", "confidence", "ambiguity", "cost")
        for field in fields:
            for value in (math.nan, math.inf, -math.inf):
                kwargs = {
                    "channel": EvidenceChannel.TOOL,
                    "action": "ground",
                    "content": "one candidate",
                    "coverage": 0.8,
                    "confidence": 0.8,
                    "ambiguity": 0.1,
                    "provenance": ("grounding:v1",),
                    "cost": 0.1,
                }
                kwargs[field] = value
                with self.subTest(field=field, value=value):
                    self.assertFalse(EvidencePacket(**kwargs).validate().valid)

    def test_invalid_packet_thresholds_are_rejected(self):
        packet = EvidencePacket(
            channel=EvidenceChannel.TOOL,
            action="ground",
            content="one candidate",
            provenance=("grounding:v1",),
        )
        for kwargs in (
            {"min_coverage": math.nan},
            {"min_confidence": -0.1},
            {"max_ambiguity": 1.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                packet.validate(**kwargs)


class UtilityRouterTest(unittest.TestCase):
    def setUp(self):
        self.router = InteractionAwareUtilityRouter(
            backend=ToyBackend(),
            experts=experts(),
            knowledge_actions=("none", "template"),
            tool_actions=("none", "ground"),
            commit_margin=0.2,
            min_coverage=0.5,
            min_confidence=0.5,
            max_ambiguity=0.5,
            require_provenance=True,
        )

    def test_interaction_changes_joint_proposal(self):
        proposal = self.router.propose("Where should the mug be placed?")
        self.assertEqual(proposal.expert.name, "Spatial Reasoning")
        self.assertEqual(proposal.action, RouteAction("template", "ground"))
        self.assertGreater(proposal.terms.interaction, 0.0)

    def test_invalid_router_thresholds_are_rejected(self):
        for kwargs in (
            {"commit_margin": -0.1},
            {"commit_margin": math.nan},
            {"min_coverage": 1.1},
            {"min_confidence": -0.1},
            {"max_ambiguity": math.inf},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                InteractionAwareUtilityRouter(
                    backend=ToyBackend(),
                    experts=experts(),
                    **kwargs,
                )

    def test_verifier_can_partially_fallback(self):
        proposal = self.router.propose("Where should the mug be placed?")
        knowledge = EvidencePacket(
            channel=EvidenceChannel.KNOWLEDGE,
            action="template",
            content="normalize the reference frame",
            coverage=1.0,
            confidence=0.9,
            ambiguity=0.0,
            provenance=("template:v2",),
            metadata={"adjustment": 0.2},
        )
        ambiguous_tool = EvidencePacket(
            channel=EvidenceChannel.TOOL,
            action="ground",
            content="two visually identical mugs",
            coverage=1.0,
            confidence=0.9,
            ambiguity=0.9,
            provenance=("grounding:call-1",),
        )
        committed = self.router.commit(
            proposal,
            {
                EvidenceChannel.KNOWLEDGE: knowledge,
                EvidenceChannel.TOOL: ambiguous_tool,
            },
        )
        self.assertEqual(committed.committed_action, RouteAction("template", "none"))
        self.assertIn("high_ambiguity", committed.rejected_reasons["tool"])
        self.assertGreater(committed.gain_over_null, 0.0)

    def test_predict_exposes_unverified_proposal(self):
        decision = self.router.predict("Where should the mug be placed?")
        for key in ("adapter", "expert", "knowledge", "tool", "confidence"):
            self.assertIn(key, decision)
        self.assertEqual(decision["adapter"], 1)
        self.assertTrue(decision["verification_pending"])

    def test_pipeline_consumes_verified_packet_without_reacquisition(self):
        proposal = self.router.propose("Where should the mug be placed?")
        knowledge = EvidencePacket(
            channel=EvidenceChannel.KNOWLEDGE,
            action="template",
            content="verified knowledge packet",
            coverage=1.0,
            confidence=0.9,
            ambiguity=0.0,
            provenance=("template:v2",),
        )
        invalid_tool = EvidencePacket(
            channel=EvidenceChannel.TOOL,
            action="ground",
            content="ambiguous candidates",
            coverage=1.0,
            confidence=0.9,
            ambiguity=0.9,
            provenance=("grounding:call-1",),
        )
        packets = {
            EvidenceChannel.KNOWLEDGE: knowledge,
            EvidenceChannel.TOOL: invalid_tool,
        }
        committed = self.router.commit(proposal, packets)

        class Backbone:
            def __init__(self):
                self.prompt = ""
                self.adapter_id = None

            def set_adapter(self, level_id):
                self.level_id = level_id

            def generate_answer(self, prompt, image, adapter_id):
                self.prompt = prompt
                self.adapter_id = adapter_id
                return "answer"

        class ForbiddenModule:
            def __call__(self, *args, **kwargs):
                raise AssertionError("verified execution must not reacquire evidence")

        backbone = Backbone()
        pipeline = EKTRouterPipeline(
            backbone=backbone,
            adapter_paths={},
            router_agent=self.router,
            knowledge_module=ForbiddenModule(),
            tool_module=ForbiddenModule(),
        )
        answer = pipeline.run_with_committed_route(
            "Where should the mug be placed?", None, committed, packets
        )
        self.assertEqual(answer, "answer")
        self.assertIn("verified knowledge packet", backbone.prompt)
        self.assertNotIn("ambiguous candidates", backbone.prompt)


if __name__ == "__main__":
    unittest.main()
