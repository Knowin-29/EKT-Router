"""Typed evidence packets for interaction-aware EKT routing.

Some provider adapters return knowledge and tool outputs as plain strings. The
v2 router needs the content plus enough metadata to decide whether that output
should be committed to an expert.  This module defines the transport contract
without depending on a particular retrieval or vision-tool implementation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Tuple


class EvidenceChannel(str, Enum):
    """Evidence type used by the EKT action space."""

    KNOWLEDGE = "knowledge"
    TOOL = "tool"


@dataclass(frozen=True)
class EvidenceValidity:
    """Result of validating one acquired evidence packet."""

    valid: bool
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidencePacket:
    """Structured output returned by a knowledge or perception action.

    ``coverage`` estimates whether the entities or facts requested by the
    question were actually returned.  ``confidence`` is the provider's
    calibrated reliability, while ``ambiguity`` measures unresolved matches.
    All three values are normalized to ``[0, 1]``.  ``provenance`` must identify
    the retriever record, tool call, frame, or model version that produced the
    content.  A packet with ``action='none'`` is the explicit null packet.
    """

    channel: EvidenceChannel
    action: str
    content: str = ""
    coverage: float = 0.0
    confidence: float = 0.0
    ambiguity: float = 0.0
    provenance: Tuple[str, ...] = ()
    cost: float = 0.0
    schema_valid: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_null(self) -> bool:
        return self.action.strip().lower() in {"", "none", "null"}

    def validate(
        self,
        *,
        min_coverage: float = 0.0,
        min_confidence: float = 0.0,
        max_ambiguity: float = 1.0,
        require_provenance: bool = True,
    ) -> EvidenceValidity:
        """Validate packet structure and acquisition quality.

        Null packets are always valid because they preserve the original input.
        Non-null packets fail closed when required fields are malformed or do
        not clear the configured thresholds.
        """

        if self.is_null:
            return EvidenceValidity(valid=True)

        thresholds = (min_coverage, min_confidence, max_ambiguity)
        if not all(math.isfinite(float(value)) for value in thresholds):
            raise ValueError("evidence thresholds must be finite")
        if not 0.0 <= float(min_coverage) <= 1.0:
            raise ValueError("min_coverage must lie in [0, 1]")
        if not 0.0 <= float(min_confidence) <= 1.0:
            raise ValueError("min_confidence must lie in [0, 1]")
        if not 0.0 <= float(max_ambiguity) <= 1.0:
            raise ValueError("max_ambiguity must lie in [0, 1]")

        reasons = []
        if not self.schema_valid:
            reasons.append("invalid_schema")
        if not self.content.strip():
            reasons.append("empty_content")
        if not math.isfinite(float(self.coverage)):
            reasons.append("coverage_non_finite")
        elif not 0.0 <= float(self.coverage) <= 1.0:
            reasons.append("coverage_out_of_range")
        elif float(self.coverage) < float(min_coverage):
            reasons.append("insufficient_coverage")
        if not math.isfinite(float(self.confidence)):
            reasons.append("confidence_non_finite")
        elif not 0.0 <= float(self.confidence) <= 1.0:
            reasons.append("confidence_out_of_range")
        elif float(self.confidence) < float(min_confidence):
            reasons.append("low_confidence")
        if not math.isfinite(float(self.ambiguity)):
            reasons.append("ambiguity_non_finite")
        elif not 0.0 <= float(self.ambiguity) <= 1.0:
            reasons.append("ambiguity_out_of_range")
        elif float(self.ambiguity) > float(max_ambiguity):
            reasons.append("high_ambiguity")
        if not math.isfinite(float(self.cost)):
            reasons.append("cost_non_finite")
        elif float(self.cost) < 0.0:
            reasons.append("negative_cost")
        if require_provenance and (not self.provenance or not all(isinstance(p, str) and p.strip() for p in self.provenance)):
            reasons.append("missing_provenance")
        return EvidenceValidity(valid=not reasons, reasons=tuple(reasons))

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["channel"] = self.channel.value
        row["provenance"] = list(self.provenance)
        row["metadata"] = dict(self.metadata)
        return row

    @classmethod
    def null(cls, channel: EvidenceChannel) -> "EvidencePacket":
        return cls(channel=channel, action="none", schema_valid=True)

    @classmethod
    def from_provider_output(
        cls,
        *,
        channel: EvidenceChannel,
        action: str,
        content: str,
        provenance: Tuple[str, ...] = (),
        cost: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> "EvidencePacket":
        """Wrap a plain-text provider output without inventing quality scores.

        Non-empty provider outputs receive neutral quality values. They remain
        usable with permissive thresholds, but experiments should replace these
        defaults with measurements from the actual provider.
        """

        is_null = action.strip().lower() in {"", "none", "null"}
        return cls(
            channel=channel,
            action="none" if is_null else action,
            content="" if is_null else (content or ""),
            coverage=0.0 if is_null else 0.5,
            confidence=0.0 if is_null else 0.5,
            ambiguity=0.0 if is_null else 0.5,
            provenance=() if is_null else tuple(provenance),
            cost=0.0 if is_null else float(cost),
            schema_valid=True,
            metadata=dict(metadata or {}),
        )
