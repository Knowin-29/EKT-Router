"""Interaction-aware utility router for bounded EKT inference.

The orchestration in this file is model-agnostic.  A trained backend supplies
the pre-acquisition utility terms and post-acquisition adjustment; the router
enumerates the small typed action space, proposes one route, validates acquired
evidence, and commits a positive-utility subset.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from itertools import combinations, product
from typing import Any, Dict, Mapping, Protocol, Sequence, Tuple

from .evidence_contract import EvidenceChannel, EvidencePacket


@dataclass(frozen=True)
class ExpertSpec:
    """One named execution path and its outcome-derived signature."""

    name: str
    adapter_id: int
    checkpoint_key: str
    capability_signature: Tuple[float, ...]


@dataclass(frozen=True)
class RouteAction:
    knowledge: str = "none"
    tool: str = "none"

    def __post_init__(self):
        knowledge = self.knowledge.strip().lower()
        tools = tuple(
            sorted(
                set(
                    t.strip().lower()
                    for t in self.tool.split("+")
                    if t.strip() and t.strip().lower() != "none"
                )
            )
        )
        if not knowledge:
            raise ValueError("knowledge action cannot be empty")
        object.__setattr__(self, "knowledge", knowledge)
        object.__setattr__(self, "tool", "+".join(tools) or "none")

    @property
    def tools(self) -> Tuple[str, ...]:
        return () if self.tool == "none" else tuple(self.tool.split("+"))

    @property
    def is_null(self) -> bool:
        return self.knowledge == "none" and not self.tools

    def components(self):
        if self.knowledge != "none":
            yield "knowledge", EvidenceChannel.KNOWLEDGE, self.knowledge
        for tool in self.tools:
            yield "tool:" + tool, EvidenceChannel.TOOL, tool


def packet_for(packets, key, channel, action):
    """Accept per-provider keys; retain the old single-tool packet API."""
    packet = packets.get(key)
    if packet is None:
        packet = packets.get(channel)
    return packet if packet is not None and packet.action.strip().lower() == action else None


@dataclass(frozen=True)
class UtilityTerms:
    """The paper's B + G + Gamma - lambda C decomposition."""

    competence: float
    evidence: float
    interaction: float
    cost: float = 0.0

    @property
    def total(self) -> float:
        return (
            float(self.competence)
            + float(self.evidence)
            + float(self.interaction)
            - float(self.cost)
        )

    def to_dict(self) -> Dict[str, float]:
        row = asdict(self)
        row["total"] = self.total
        return row


class UtilityBackend(Protocol):
    """Interface implemented by the trained neural utility model."""

    def encode_query(self, question: str, image: Any = None) -> Any: ...

    def score_route(
        self,
        query_state: Any,
        expert: ExpertSpec,
        action: RouteAction,
    ) -> UtilityTerms: ...

    def post_acquisition_adjustment(
        self,
        query_state: Any,
        expert: ExpertSpec,
        action: RouteAction,
        packets: Mapping[EvidenceChannel, EvidencePacket],
    ) -> float: ...


@dataclass(frozen=True)
class RouteProposal:
    expert: ExpertSpec
    action: RouteAction
    terms: UtilityTerms
    confidence: float
    query_state: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class CommittedRoute:
    expert: ExpertSpec
    proposed_action: RouteAction
    committed_action: RouteAction
    pre_terms: UtilityTerms
    post_utility: float
    gain_over_null: float
    rejected_reasons: Mapping[str, Tuple[str, ...]]

    def as_decision(self) -> Dict[str, Any]:
        """Return a serializable decision consumed by ``EKTRouterPipeline``."""

        return {
            "adapter": self.expert.adapter_id,
            "expert": self.expert.checkpoint_key,
            "knowledge": self.committed_action.knowledge,
            "tool": self.committed_action.tool,
            "confidence": 1.0,
            "expert_name": self.expert.name,
            "proposed_knowledge": self.proposed_action.knowledge,
            "proposed_tool": self.proposed_action.tool,
            "utility_terms": self.pre_terms.to_dict(),
            "post_utility": float(self.post_utility),
            "gain_over_null": float(self.gain_over_null),
            "verification_rejections": {
                key: list(value) for key, value in self.rejected_reasons.items()
            },
        }


class InteractionAwareUtilityRouter:
    """Bounded propose--verify--commit router.

    Calling :meth:`predict` exposes the pre-acquisition proposal. Production
    code should call :meth:`propose`, acquire the
    proposed packets, then call :meth:`commit` before generation.
    """

    def __init__(
        self,
        *,
        backend: UtilityBackend,
        experts: Sequence[ExpertSpec],
        knowledge_actions: Sequence[str] = ("none", "template", "rag"),
        tool_actions: Sequence[str] = ("none", "depth", "ground", "ocr"),
        commit_margin: float = 0.0,
        min_coverage: float = 0.0,
        min_confidence: float = 0.0,
        max_ambiguity: float = 1.0,
        require_provenance: bool = True,
        tool_budget: float = 1.0,
        tool_costs: Mapping[str, float] | None = None,
        max_provider_calls: int | None = None,
    ) -> None:
        if not experts:
            raise ValueError("at least one expert is required")
        thresholds = (commit_margin, min_coverage, min_confidence, max_ambiguity)
        if not all(math.isfinite(float(value)) for value in thresholds):
            raise ValueError("router thresholds must be finite")
        if float(commit_margin) < 0.0:
            raise ValueError("commit_margin must be non-negative")
        if not 0.0 <= float(min_coverage) <= 1.0:
            raise ValueError("min_coverage must lie in [0, 1]")
        if not 0.0 <= float(min_confidence) <= 1.0:
            raise ValueError("min_confidence must lie in [0, 1]")
        if not 0.0 <= float(max_ambiguity) <= 1.0:
            raise ValueError("max_ambiguity must lie in [0, 1]")
        self.backend = backend
        self.experts = tuple(experts)
        self.knowledge_actions = self._normalize_actions(knowledge_actions)
        self.tool_actions = self._normalize_actions(tool_actions)
        self.commit_margin = float(commit_margin)
        self.min_coverage = float(min_coverage)
        self.min_confidence = float(min_confidence)
        self.max_ambiguity = float(max_ambiguity)
        self.require_provenance = bool(require_provenance)
        self.tool_budget = float(tool_budget)
        self.tool_costs = {
            t: float((tool_costs or {}).get(t, 1.0)) for t in self.tool_actions if t != "none"
        }
        if not math.isfinite(self.tool_budget) or self.tool_budget < 0:
            raise ValueError("tool_budget must be finite and non-negative")
        if any(not math.isfinite(v) or v < 0 for v in self.tool_costs.values()):
            raise ValueError("tool costs must be finite and non-negative")
        if max_provider_calls is not None and (
            isinstance(max_provider_calls, bool)
            or int(max_provider_calls) != max_provider_calls
            or max_provider_calls < 0
        ):
            raise ValueError("max_provider_calls must be a non-negative integer")
        self.max_provider_calls = max_provider_calls

    def feasible_actions(self):
        tools = tuple(t for t in self.tool_actions if t != "none")
        for size in range(len(tools) + 1):
            for subset in combinations(tools, size):
                if sum(self.tool_costs[t] for t in subset) > self.tool_budget + 1e-12:
                    continue
                for knowledge in self.knowledge_actions:
                    if (
                        self.max_provider_calls is not None
                        and size + int(knowledge != "none") > self.max_provider_calls
                    ):
                        continue
                    yield RouteAction(knowledge, "+".join(subset) or "none")

    @staticmethod
    def _normalize_actions(actions: Sequence[str]) -> Tuple[str, ...]:
        normalized = tuple(dict.fromkeys(str(a).strip().lower() for a in actions))
        if "none" not in normalized:
            normalized = ("none",) + normalized
        return normalized

    @staticmethod
    def _confidence(scores: Sequence[float], selected: float) -> float:
        if len(scores) <= 1:
            return 1.0
        anchor = max(scores)
        weights = [math.exp(max(-60.0, min(0.0, score - anchor))) for score in scores]
        denominator = sum(weights)
        numerator = math.exp(max(-60.0, min(0.0, selected - anchor)))
        return float(numerator / denominator) if denominator else 0.0

    def _score(
        self,
        query_state: Any,
        expert: ExpertSpec,
        action: RouteAction,
    ) -> UtilityTerms:
        terms = self.backend.score_route(query_state, expert, action)
        if not isinstance(terms, UtilityTerms):
            raise TypeError("backend.score_route must return UtilityTerms")
        if not math.isfinite(terms.total):
            raise ValueError("route utility must be finite")
        return terms

    def propose(self, question: str, image: Any = None) -> RouteProposal:
        query_state = self.backend.encode_query(question, image)
        candidates = []
        for expert, action in product(self.experts, tuple(self.feasible_actions())):
            knowledge, tool = action.knowledge, action.tool
            terms = self._score(query_state, expert, action)
            # Equal utilities favor fewer acquisitions, then stable lexical IDs.
            calls = int(knowledge != "none") + len(action.tools)
            candidates.append(
                (terms.total, -calls, expert.name, knowledge, tool, expert, action, terms)
            )
        best = max(candidates, key=lambda row: row[:5])
        scores = [row[0] for row in candidates]
        return RouteProposal(
            expert=best[5],
            action=best[6],
            terms=best[7],
            confidence=self._confidence(scores, best[0]),
            query_state=query_state,
        )

    def predict(self, question: str, image: Any = None) -> Dict[str, Any]:
        """Return the proposal without claiming that verification occurred.

        It does not claim that post-acquisition verification occurred.  The
        explicit flag makes proposal-only evaluation distinguishable in logs.
        """

        proposal = self.propose(question, image)
        return {
            "adapter": proposal.expert.adapter_id,
            "expert": proposal.expert.checkpoint_key,
            "knowledge": proposal.action.knowledge,
            "tool": proposal.action.tool,
            "confidence": proposal.confidence,
            "expert_name": proposal.expert.name,
            "utility_terms": proposal.terms.to_dict(),
            "verification_pending": not proposal.action.is_null,
            "router_version": "interaction_aware_v2_proposal_only",
        }

    def _candidate_actions(self, proposed: RouteAction) -> Tuple[RouteAction, ...]:
        knowledge = ("none", proposed.knowledge) if proposed.knowledge != "none" else ("none",)
        return tuple(
            RouteAction(k, "+".join(subset) or "none")
            for k in knowledge
            for size in range(len(proposed.tools) + 1)
            for subset in combinations(proposed.tools, size)
        )

    def commit(
        self,
        proposal: RouteProposal,
        packets: Mapping[EvidenceChannel, EvidencePacket],
    ) -> CommittedRoute:
        """Validate acquired evidence and commit its best positive-gain subset."""

        rejected: Dict[str, Tuple[str, ...]] = {}
        available = {}
        for key, channel, action in proposal.action.components():
            packet = packet_for(packets, key, channel, action)
            reasons = ()
            if packet is None:
                reasons = ("missing_or_mismatched_packet",)
            elif packet.channel != channel:
                reasons = ("channel_mismatch",)
            else:
                reasons = packet.validate(
                    min_coverage=self.min_coverage,
                    min_confidence=self.min_confidence,
                    max_ambiguity=self.max_ambiguity,
                    require_provenance=self.require_provenance,
                ).reasons
            if reasons:
                rejected[key] = reasons
                if channel == EvidenceChannel.TOOL and len(proposal.action.tools) == 1:
                    rejected["tool"] = reasons  # compatibility with single-tool audits
            else:
                available[key] = packet

        null_action = RouteAction()
        null_terms = self._score(proposal.query_state, proposal.expert, null_action)
        # U(b) already subtracts lambda*C(b). All subsets incurred C(a*).
        null_post = null_terms.total - (proposal.terms.cost - null_terms.cost)
        scored = []
        for action in self._candidate_actions(proposal.action):
            keys = tuple(key for key, _, _ in action.components())
            if any(key not in available for key in keys):
                continue
            selected_packets = {key: available[key] for key in keys}
            terms = self._score(proposal.query_state, proposal.expert, action)
            adjustment = (
                0.0
                if action.is_null
                else float(
                    self.backend.post_acquisition_adjustment(
                        proposal.query_state,
                        proposal.expert,
                        action,
                        selected_packets,
                    )
                )
            )
            if not math.isfinite(adjustment):
                raise ValueError("post-acquisition adjustment must be finite")
            post_utility = terms.total - (proposal.terms.cost - terms.cost) + adjustment
            calls = int(action.knowledge != "none") + len(action.tools)
            scored.append((post_utility, -calls, action.knowledge, action.tool, action, terms))
        best = max(scored, key=lambda row: row[:4])
        gain = float(best[0] - null_post)
        if gain <= self.commit_margin:
            committed, committed_terms, post_utility, gain = null_action, null_terms, null_post, 0.0
        else:
            committed, committed_terms, post_utility = best[4], best[5], float(best[0])
        return CommittedRoute(
            expert=proposal.expert,
            proposed_action=proposal.action,
            committed_action=committed,
            pre_terms=committed_terms,
            post_utility=post_utility,
            gain_over_null=gain,
            rejected_reasons=rejected,
        )


def paper_experts(
    signatures: Mapping[str, Sequence[float]],
) -> Tuple[ExpertSpec, ...]:
    """Build the three paper-facing execution paths from measured signatures."""

    aliases = (
        ("General", 0, "general"),
        ("Spatial Reasoning", 1, "spatial_reasoning"),
        ("Spatiotemporal", 2, "spatiotemporal"),
    )
    experts = []
    for name, adapter_id, level in aliases:
        key = level if level in signatures else name
        if key not in signatures:
            raise KeyError(f"missing capability signature for {name}")
        experts.append(
            ExpertSpec(
                name=name,
                adapter_id=adapter_id,
                checkpoint_key=level,
                capability_signature=tuple(float(v) for v in signatures[key]),
            )
        )
    return tuple(experts)


# Kept as an import alias for existing local checkpoints; public code should
# use ``paper_experts``.
legacy_three_path_experts = paper_experts
