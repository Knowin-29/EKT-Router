"""Neural backend for the factorized EKT utility model.

This module implements the paper-facing decomposition

    U(x, e, a) = B(x, e) + G(x, a) + Gamma(x, e, a) - lambda C(a)

The query encoder is injected so the
same backend can consume cached embeddings during training and a frozen VLM
encoder during deployment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .evidence_contract import EvidenceChannel, EvidencePacket
from .utility_router import ExpertSpec, RouteAction, UtilityTerms


@dataclass(frozen=True)
class ActionEncoding:
    features: Tensor
    active: Tensor
    cost: float


class ActionVocabulary:
    """Deterministic typed encoding and cost registry for route actions."""

    def __init__(
        self,
        knowledge_actions: Sequence[str] = ("none", "template", "rag"),
        tool_actions: Sequence[str] = ("none", "depth", "ground", "ocr"),
        costs: Mapping[str, float] | None = None,
    ) -> None:
        self.knowledge_actions = self._normalize(knowledge_actions)
        self.tool_actions = self._normalize(tool_actions)
        self._knowledge_index = {
            action: index for index, action in enumerate(self.knowledge_actions)
        }
        self._tool_index = {action: index for index, action in enumerate(self.tool_actions)}
        self.costs = {str(key): float(value) for key, value in (costs or {}).items()}
        if any(value < 0.0 or not math.isfinite(value) for value in self.costs.values()):
            raise ValueError("action costs must be finite and non-negative")

    @staticmethod
    def _normalize(actions: Sequence[str]) -> Tuple[str, ...]:
        result = tuple(dict.fromkeys(str(action).strip().lower() for action in actions))
        if "none" not in result:
            result = ("none",) + result
        if not result:
            raise ValueError("action vocabulary cannot be empty")
        return result

    @property
    def feature_dim(self) -> int:
        # Two one-hot fields and two acquisition indicators.
        return len(self.knowledge_actions) + len(self.tool_actions) + 2

    def encode(
        self,
        action: RouteAction,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> ActionEncoding:
        knowledge = action.knowledge.strip().lower()
        if knowledge not in self._knowledge_index:
            raise KeyError(f"unknown knowledge action: {knowledge}")
        features = torch.zeros(self.feature_dim, device=device, dtype=dtype)
        # Null embedding is exactly zero. Tools use a multi-hot subset encoding.
        if knowledge != "none":
            features[self._knowledge_index[knowledge]] = 1.0
        for tool in action.tools:
            if tool not in self._tool_index:
                raise KeyError(f"unknown tool action: {tool}")
            features[len(self.knowledge_actions) + self._tool_index[tool]] = 1.0
        features[-2] = float(knowledge != "none")
        features[-1] = float(bool(action.tools))
        cost = self.costs.get(f"knowledge:{knowledge}", 0.0) if knowledge != "none" else 0.0
        cost += sum(self.costs.get(f"tool:{tool}", 0.0) for tool in action.tools)
        return ActionEncoding(
            features,
            torch.tensor(float(not action.is_null), device=device, dtype=dtype),
            float(cost),
        )


class FactorizedUtilityModel(nn.Module):
    """Low-rank interaction-aware utility scorer.

    ``B`` and ``G`` use separate networks. ``Gamma`` is a CP-style trilinear
    interaction over query, expert signature, and evidence action.  Disabling
    the interaction leaves the other two branches unchanged, which makes the
    paper's B+G versus B+G+Gamma ablation exact.
    """

    def __init__(
        self,
        query_dim: int,
        signature_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        interaction_rank: int = 64,
        include_interaction: bool = True,
    ) -> None:
        super().__init__()
        if min(query_dim, signature_dim, action_dim, hidden_dim, interaction_rank) <= 0:
            raise ValueError("all model dimensions must be positive")
        self.query_dim = int(query_dim)
        self.signature_dim = int(signature_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.interaction_rank = int(interaction_rank)
        self.include_interaction = bool(include_interaction)

        self.query_projection = nn.Sequential(
            nn.LayerNorm(query_dim), nn.Linear(query_dim, hidden_dim), nn.GELU()
        )
        self.expert_projection = nn.Sequential(
            nn.LayerNorm(signature_dim), nn.Linear(signature_dim, hidden_dim), nn.GELU()
        )
        self.action_projection = nn.Sequential(
            nn.LayerNorm(action_dim), nn.Linear(action_dim, hidden_dim), nn.GELU()
        )
        self.competence_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.query_interaction = nn.Linear(hidden_dim, interaction_rank, bias=False)
        self.expert_interaction = nn.Linear(hidden_dim, interaction_rank, bias=False)
        self.action_interaction = nn.Linear(hidden_dim, interaction_rank, bias=False)
        self.interaction_weight = nn.Parameter(torch.ones(interaction_rank))

    @staticmethod
    def _batch(value: Tensor) -> Tensor:
        return value.unsqueeze(0) if value.ndim == 1 else value

    def forward(
        self,
        query: Tensor,
        signature: Tensor,
        action: Tensor,
        action_active: Tensor | None = None,
        expert_bank: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        query = self._batch(query)
        signature = self._batch(signature)
        action = self._batch(action)
        if query.shape[-1] != self.query_dim:
            raise ValueError(f"expected query dim {self.query_dim}, got {query.shape[-1]}")
        if signature.shape[-1] != self.signature_dim:
            raise ValueError(
                f"expected signature dim {self.signature_dim}, got {signature.shape[-1]}"
            )
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"expected action dim {self.action_dim}, got {action.shape[-1]}")

        query_state = self.query_projection(query)
        expert_state = self.expert_projection(signature)
        action_state = self.action_projection(action)
        competence = self.competence_head(
            torch.cat([query_state, expert_state, query_state * expert_state], dim=-1)
        ).squeeze(-1)
        evidence = self.evidence_head(
            torch.cat([query_state, action_state, query_state * action_state], dim=-1)
        ).squeeze(-1)

        if action_active is None:
            active = torch.ones_like(evidence)
        else:
            active = action_active.to(device=evidence.device, dtype=evidence.dtype)
            if active.ndim == 0:
                active = active.unsqueeze(0)
            active = active.reshape(evidence.shape)
        evidence = evidence * active

        if self.include_interaction:
            query_factor = torch.tanh(self.query_interaction(query_state))
            expert_factor = torch.tanh(self.expert_interaction(expert_state))
            action_factor = torch.tanh(self.action_interaction(action_state))
            factors = query_factor * expert_factor * action_factor
            interaction = (factors * self.interaction_weight).sum(dim=-1) / math.sqrt(
                self.interaction_rank
            )
            if expert_bank is not None:
                bank = self._batch(expert_bank)
                if bank.shape[-1] != self.signature_dim:
                    raise ValueError(
                        f"expected expert bank dim {self.signature_dim}, got {bank.shape[-1]}"
                    )
                bank_state = self.expert_projection(bank)
                bank_factor = torch.tanh(self.expert_interaction(bank_state))
                bank_interactions = (
                    query_factor[:, None, :]
                    * action_factor[:, None, :]
                    * bank_factor[None, :, :]
                    * self.interaction_weight
                ).sum(dim=-1) / math.sqrt(self.interaction_rank)
                interaction = interaction - bank_interactions.mean(dim=1)
            interaction = interaction * active
        else:
            interaction = torch.zeros_like(competence)
        return {
            "competence": competence,
            "evidence": evidence,
            "interaction": interaction,
            "utility_without_cost": competence + evidence + interaction,
        }


class PacketQualityVerifier(nn.Module):
    """Learned post-acquisition adjustment over typed packet quality."""

    PACKET_DIM = 12  # six fields for Knowledge and six for Tool

    def __init__(
        self,
        query_dim: int,
        signature_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = query_dim + signature_dim + action_dim + self.PACKET_DIM
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query: Tensor, signature: Tensor, action: Tensor, packet: Tensor) -> Tensor:
        values = [
            value.unsqueeze(0) if value.ndim == 1 else value
            for value in (query, signature, action, packet)
        ]
        return self.network(torch.cat(values, dim=-1)).squeeze(-1)


class PairwiseUtilityLoss(nn.Module):
    """Bradley--Terry ranking with a small Huber gain-calibration term."""

    def __init__(self, calibration_weight: float = 0.1, huber_delta: float = 0.1) -> None:
        super().__init__()
        if calibration_weight < 0.0:
            raise ValueError("calibration_weight must be non-negative")
        if huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        self.calibration_weight = float(calibration_weight)
        self.huber_delta = float(huber_delta)

    def forward(
        self,
        left_utility: Tensor,
        right_utility: Tensor,
        preference: Tensor,
        target_gain: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute loss; preference is +1 (left), -1 (right), or 0 (tie)."""

        delta = left_utility - right_utility
        preference = preference.to(device=delta.device, dtype=delta.dtype)
        target_gain = target_gain.to(device=delta.device, dtype=delta.dtype)
        non_tie = preference.ne(0)
        if non_tie.any():
            ranking = F.softplus(-preference[non_tie] * delta[non_tie]).mean()
        else:
            ranking = delta.sum() * 0.0
        tie = ~non_tie
        if tie.any():
            tie_loss = F.smooth_l1_loss(
                delta[tie], torch.zeros_like(delta[tie]), beta=self.huber_delta
            )
        else:
            tie_loss = delta.sum() * 0.0
        calibration = F.huber_loss(delta, target_gain, delta=self.huber_delta, reduction="mean")
        total = ranking + tie_loss + self.calibration_weight * calibration
        return {
            "loss": total,
            "ranking": ranking,
            "tie": tie_loss,
            "calibration": calibration,
        }


class NeuralUtilityBackend:
    """Adapter from torch modules to :class:`UtilityBackend`."""

    def __init__(
        self,
        *,
        model: FactorizedUtilityModel,
        query_encoder: Callable[[str, Any], Tensor | Sequence[float]],
        action_vocabulary: ActionVocabulary,
        cost_weight: float = 1.0,
        interaction_scale: float = 1.0,
        verifier: PacketQualityVerifier | None = None,
        signature_bank: Sequence[Sequence[float]] | None = None,
    ) -> None:
        if cost_weight < 0.0 or not math.isfinite(cost_weight):
            raise ValueError("cost_weight must be finite and non-negative")
        if interaction_scale < 0.0 or not math.isfinite(interaction_scale):
            raise ValueError("interaction_scale must be finite and non-negative")
        if model.action_dim != action_vocabulary.feature_dim:
            raise ValueError("model action dimension does not match vocabulary")
        self.model = model
        self.query_encoder = query_encoder
        self.action_vocabulary = action_vocabulary
        self.cost_weight = float(cost_weight)
        # Selected only on held-out routing data.  Zero is an exact fail-closed
        # fallback to the unchanged B+G utility without altering the checkpoint.
        self.interaction_scale = float(interaction_scale)
        self.verifier = verifier
        self.signature_bank = (
            tuple(tuple(float(value) for value in row) for row in signature_bank)
            if signature_bank is not None
            else None
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def encode_query(self, question: str, image: Any = None) -> Tensor:
        encoded = self.query_encoder(question, image)
        tensor = torch.as_tensor(encoded, device=self.device, dtype=self.dtype).flatten()
        if tensor.numel() != self.model.query_dim:
            raise ValueError(
                f"query encoder returned {tensor.numel()} values; expected {self.model.query_dim}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError("query representation contains non-finite values")
        return tensor

    def _signature(self, expert: ExpertSpec) -> Tensor:
        tensor = torch.tensor(expert.capability_signature, device=self.device, dtype=self.dtype)
        if tensor.numel() != self.model.signature_dim:
            raise ValueError(
                f"expert signature has {tensor.numel()} values; expected {self.model.signature_dim}"
            )
        return tensor

    def score_route(
        self, query_state: Tensor, expert: ExpertSpec, action: RouteAction
    ) -> UtilityTerms:
        encoded = self.action_vocabulary.encode(action, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            output = self.model(
                query_state,
                self._signature(expert),
                encoded.features,
                encoded.active,
                expert_bank=(
                    torch.tensor(
                        self.signature_bank,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    if self.signature_bank is not None
                    else None
                ),
            )
        return UtilityTerms(
            competence=float(output["competence"].item()),
            evidence=float(output["evidence"].item()),
            interaction=self.interaction_scale * float(output["interaction"].item()),
            cost=self.cost_weight * encoded.cost,
        )

    @staticmethod
    def packet_features(
        packets: Mapping[EvidenceChannel, EvidencePacket],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        values = []
        # Permutation-invariant conservative pooling per channel; all retained
        # tool packets contribute, including a summed sunk acquisition cost.
        for channel in (EvidenceChannel.KNOWLEDGE, EvidenceChannel.TOOL):
            group = [p for p in packets.values() if p.channel == channel]
            if not group:
                values.extend((0.0,) * 6)
            else:
                values.extend(
                    (
                        1.0,
                        float(all(p.schema_valid for p in group)),
                        min(p.coverage for p in group),
                        min(p.confidence for p in group),
                        1.0 - max(p.ambiguity for p in group),
                        sum(p.cost for p in group),
                    )
                )
        return torch.tensor(values, device=device, dtype=dtype)

    def post_acquisition_adjustment(
        self,
        query_state: Tensor,
        expert: ExpertSpec,
        action: RouteAction,
        packets: Mapping[EvidenceChannel, EvidencePacket],
    ) -> float:
        if self.verifier is None or action.is_null:
            return 0.0
        encoded = self.action_vocabulary.encode(action, device=self.device, dtype=self.dtype)
        packet = self.packet_features(packets, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            adjustment = self.verifier(
                query_state,
                self._signature(expert),
                encoded.features,
                packet,
            )
        return float(adjustment.item())
