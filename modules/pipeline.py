"""One-pass answer execution and bounded propose--verify--commit inference."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Mapping, Optional

from .evidence_contract import EvidenceChannel, EvidencePacket
from .taxonomy import EXPERTS, expert_by_index
from .utility_router import packet_for

logger = logging.getLogger(__name__)


class EKTRouterPipeline:
    """Combine a frozen answer backbone with EKT routing components.

    ``run_bounded`` is the paper-facing path: it proposes one complete route,
    performs at most one acquisition round, verifies packet subsets, fixes the
    selected expert, and generates one answer. ``__call__`` remains useful for
    expert-only controls and routers that do not acquire external evidence.
    """

    def __init__(
        self,
        backbone: Any,
        adapter_paths: Mapping[str, str],
        router_agent: Any,
        knowledge_module: Any,
        tool_module: Any,
    ) -> None:
        self.backbone = backbone
        self.adapter_paths = dict(adapter_paths)
        self.router_agent = router_agent
        self.knowledge_module = knowledge_module
        self.tool_module = tool_module

    def __call__(
        self,
        question: str,
        image: Any = None,
        override_decision: Optional[Mapping[str, Any]] = None,
    ) -> str:
        decision = dict(override_decision or self.router_agent.predict(question, image))
        if decision.get("verification_pending"):
            raise ValueError("proposal requires run_bounded before answer generation")
        decision = self._normalize_decision(decision)
        return self._one_pass(decision, question, image)

    def run_bounded(
        self,
        question: str,
        image: Any,
        packet_provider: Callable[[Any, str, Any], Mapping[EvidenceChannel, EvidencePacket]],
    ) -> str:
        """Execute the bounded online protocol with exactly one proposal."""

        if not hasattr(self.router_agent, "propose") or not hasattr(self.router_agent, "commit"):
            raise TypeError("run_bounded requires a router with propose and commit methods")
        proposal = self.router_agent.propose(question, image)
        packets = {} if proposal.action.is_null else packet_provider(proposal, question, image)
        committed = self.router_agent.commit(proposal, packets)
        if committed.expert != proposal.expert:
            raise ValueError("verifier changed the selected expert")
        self.last_route, self.last_packets = committed, packets
        return self.run_with_committed_route(question, image, committed, packets)

    def _one_pass(
        self,
        decision: Mapping[str, Any],
        question: str,
        image: Any,
        evidence_override: Optional[Mapping[str, str]] = None,
    ) -> str:
        if evidence_override is None:
            knowledge_text = self.knowledge_module(question, decision["knowledge"])
            tool_text = "\n\n".join(
                self.tool_module(image, question, t)
                for t in decision["tool"].split("+")
                if t != "none"
            )
        else:
            knowledge_text = str(evidence_override.get("knowledge", ""))
            tool_text = str(evidence_override.get("tool", ""))

        prompt_parts = []
        if tool_text:
            prompt_parts.append(f"[Perceptual Evidence]\n{tool_text}")
        if knowledge_text:
            prompt_parts.append(f"[Retrieved Knowledge]\n{knowledge_text}")
        prompt_parts.append(f"[Question]\n{question}")
        return self.generate_with_expert("\n\n".join(prompt_parts), image, int(decision["adapter"]))

    def run_with_committed_route(
        self,
        question: str,
        image: Any,
        committed_route: Any,
        packets: Mapping[EvidenceChannel, EvidencePacket],
    ) -> str:
        """Generate once from the exact evidence packets accepted by verification."""

        decision = self._normalize_decision(committed_route.as_decision())
        evidence_override: Dict[str, str] = {}
        action_set = committed_route.committed_action
        channel_text = {"knowledge": [], "tool": []}
        for key, channel, action in action_set.components():
            packet = packet_for(packets, key, channel, action)
            if packet is None or packet.channel != channel or not packet.validate().valid:
                raise ValueError(f"committed {key} has no valid matching packet")
            channel_text[channel.value].append(packet.content)
        evidence_override = {k: "\n\n".join(v) for k, v in channel_text.items()}
        return self._one_pass(decision, question, image, evidence_override=evidence_override)

    def generate_with_expert(self, prompt: str, image: Any, adapter_id: int) -> str:
        expert = expert_by_index(int(adapter_id))
        self._activate_expert(expert.id)
        if hasattr(self.backbone, "generate_answer"):
            return self.backbone.generate_answer(
                prompt=prompt,
                image=image,
                adapter_id=expert.index,
            )
        raise TypeError("backbone must implement generate_answer; use an explicit mock for demos")

    @staticmethod
    def _normalize_decision(decision: Mapping[str, Any]) -> Dict[str, Any]:
        expert_id = str(decision.get("expert", "")).strip().lower()
        if expert_id not in EXPERTS:
            try:
                expert_id = expert_by_index(int(decision.get("adapter", 0))).id
            except (TypeError, ValueError):
                expert_id = "general"
        expert = EXPERTS[expert_id]
        return {
            "adapter": expert.index,
            "expert": expert.id,
            "knowledge": str(decision.get("knowledge", "none")).lower(),
            "tool": str(decision.get("tool", "none")).lower(),
            "confidence": float(decision.get("confidence", 1.0)),
        }

    def _activate_expert(self, expert_id: str) -> None:
        if not hasattr(self.backbone, "set_adapter"):
            raise TypeError("backbone must support explicit expert activation")
        # Never silently substitute a bare backbone or another loaded adapter.
        self.backbone.set_adapter(expert_id)


# Keep the historical import name for existing callers.
KnowinRouterPipeline = EKTRouterPipeline
