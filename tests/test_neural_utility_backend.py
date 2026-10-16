import unittest

import torch

from modules.neural_utility_backend import (
    ActionVocabulary,
    FactorizedUtilityModel,
    NeuralUtilityBackend,
    PacketQualityVerifier,
    PairwiseUtilityLoss,
)
from modules.utility_router import ExpertSpec, RouteAction


class FactorizedUtilityModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.vocabulary = ActionVocabulary(
            knowledge_actions=("none", "template"),
            tool_actions=("none", "ground"),
            costs={"knowledge:template": 0.1, "tool:ground": 0.2},
        )
        self.full = FactorizedUtilityModel(
            query_dim=5,
            signature_dim=3,
            action_dim=self.vocabulary.feature_dim,
            hidden_dim=8,
            interaction_rank=4,
            include_interaction=True,
        )

    def test_null_action_has_no_evidence_or_interaction(self):
        action = self.vocabulary.encode(RouteAction())
        output = self.full(
            torch.randn(5), torch.randn(3), action.features, action.active
        )
        self.assertEqual(float(output["evidence"].item()), 0.0)
        self.assertEqual(float(output["interaction"].item()), 0.0)

    def test_interaction_ablation_preserves_b_and_g(self):
        additive = FactorizedUtilityModel(
            query_dim=5,
            signature_dim=3,
            action_dim=self.vocabulary.feature_dim,
            hidden_dim=8,
            interaction_rank=4,
            include_interaction=False,
        )
        additive.load_state_dict(self.full.state_dict())
        action = self.vocabulary.encode(RouteAction("template", "ground"))
        inputs = (torch.randn(5), torch.randn(3), action.features, action.active)
        full_output = self.full(*inputs)
        additive_output = additive(*inputs)
        self.assertTrue(
            torch.allclose(full_output["competence"], additive_output["competence"])
        )
        self.assertTrue(torch.allclose(full_output["evidence"], additive_output["evidence"]))
        self.assertEqual(float(additive_output["interaction"].item()), 0.0)

    def test_interaction_is_centered_over_expert_bank(self):
        query = torch.randn(5)
        action = self.vocabulary.encode(RouteAction("template", "ground"))
        bank = torch.tensor(
            [[0.8, 0.2, 0.1], [0.3, 0.9, 0.4], [0.4, 0.5, 0.9]],
            dtype=torch.float32,
        )
        values = []
        for signature in bank:
            output = self.full(
                query,
                signature,
                action.features,
                action.active,
                expert_bank=bank,
            )
            values.append(output["interaction"])
        self.assertAlmostEqual(float(torch.stack(values).mean().item()), 0.0, places=6)

    def test_pairwise_objective_has_finite_gradients(self):
        loss_fn = PairwiseUtilityLoss(calibration_weight=0.1, huber_delta=0.2)
        left = torch.tensor([0.2, -0.1, 0.0], requires_grad=True)
        right = torch.tensor([-0.2, 0.3, 0.1], requires_grad=True)
        preference = torch.tensor([1.0, -1.0, 0.0])
        target_gain = torch.tensor([0.5, -0.4, 0.0])
        output = loss_fn(left, right, preference, target_gain)
        output["loss"].backward()
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(torch.isfinite(left.grad).all())
        self.assertTrue(torch.isfinite(right.grad).all())

    def test_backend_matches_router_protocol(self):
        verifier = PacketQualityVerifier(
            query_dim=5,
            signature_dim=3,
            action_dim=self.vocabulary.feature_dim,
            hidden_dim=8,
        )
        backend = NeuralUtilityBackend(
            model=self.full,
            query_encoder=lambda question, image: torch.arange(5, dtype=torch.float32),
            action_vocabulary=self.vocabulary,
            verifier=verifier,
        )
        expert = ExpertSpec(
            "Spatial Reasoning", 1, "spatial_reasoning", (0.2, 0.5, 0.9)
        )
        query = backend.encode_query("where")
        terms = backend.score_route(query, expert, RouteAction("template", "ground"))
        self.assertTrue(torch.isfinite(torch.tensor(terms.total)))
        self.assertAlmostEqual(terms.cost, 0.3, places=6)

    def test_zero_interaction_scale_exactly_recovers_additive_utility(self):
        def encoder(question, image):
            return torch.arange(5, dtype=torch.float32)
        full_backend = NeuralUtilityBackend(
            model=self.full,
            query_encoder=encoder,
            action_vocabulary=self.vocabulary,
            interaction_scale=1.0,
        )
        additive_backend = NeuralUtilityBackend(
            model=self.full,
            query_encoder=encoder,
            action_vocabulary=self.vocabulary,
            interaction_scale=0.0,
        )
        expert = ExpertSpec(
            "Spatial Reasoning", 1, "spatial_reasoning", (0.2, 0.5, 0.9)
        )
        action = RouteAction("template", "ground")
        query = full_backend.encode_query("where")
        full_terms = full_backend.score_route(query, expert, action)
        additive_terms = additive_backend.score_route(query, expert, action)
        self.assertEqual(additive_terms.interaction, 0.0)
        self.assertAlmostEqual(full_terms.competence, additive_terms.competence)
        self.assertAlmostEqual(full_terms.evidence, additive_terms.evidence)
        self.assertAlmostEqual(full_terms.cost, additive_terms.cost)

    def test_interaction_scale_rejects_unsafe_values(self):
        for value in (-0.1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                NeuralUtilityBackend(
                    model=self.full,
                    query_encoder=lambda question, image: torch.zeros(5),
                    action_vocabulary=self.vocabulary,
                    interaction_scale=value,
                )


if __name__ == "__main__":
    unittest.main()
