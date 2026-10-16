from dataclasses import replace
from types import SimpleNamespace
import pytest
from modules.evidence_contract import EvidenceChannel, EvidencePacket
from modules.utility_router import (
    ExpertSpec,
    InteractionAwareUtilityRouter,
    RouteAction,
    UtilityTerms,
)
from modules.pipeline import EKTRouterPipeline
from modules.providers import PacketProvider
from modules.qwen_backbone import QwenRoutedBackbone

EXPERT = ExpertSpec("Spatial Reasoning", 1, "spatial_reasoning", (0.2, 0.8, 0.4))


class Backend:
    def encode_query(self, question, image):
        return question

    def score_route(self, state, expert, action):
        n = len(action.tools)
        return UtilityTerms(1, 3 * n, 0, 2 * n)

    def post_acquisition_adjustment(self, state, expert, action, packets):
        return sum(p.metadata.get("adjustment", 0) for p in packets.values())


def packet(action, **changes):
    return replace(
        EvidencePacket(
            EvidenceChannel.TOOL,
            action,
            "observed " + action,
            coverage=1,
            confidence=0.8,
            provenance=("local:v1",),
            cost=2,
        ),
        **changes,
    )


def router(**kwargs):
    return InteractionAwareUtilityRouter(
        backend=Backend(),
        experts=(EXPERT,),
        knowledge_actions=("none",),
        tool_actions=("none", "depth", "ground", "ocr"),
        **kwargs,
    )


def test_budget_and_total_calls_constrain_joint_enumeration():
    r = router(tool_budget=2, tool_costs={"depth": 2, "ground": 1, "ocr": 1}, max_provider_calls=2)
    actions = tuple(r.feasible_actions())
    assert RouteAction(tool="ground+ocr") in actions
    assert RouteAction(tool="depth+ground") not in actions
    assert RouteAction() in actions
    assert len(r.propose("where").action.tools) <= 2


def test_negative_budget_is_rejected():
    with pytest.raises(ValueError):
        router(tool_budget=-1)


def test_multitool_rejection_is_independent():
    r = router(tool_budget=3)
    proposal = r.propose("where")
    result = r.commit(
        proposal,
        {
            "tool:depth": packet("depth"),
            "tool:ground": packet("ground", schema_valid=False),
            "tool:ocr": packet("ocr"),
        },
    )
    assert result.committed_action.tools == ("depth", "ocr")
    assert result.expert is proposal.expert
    assert "tool:ground" in result.rejected_reasons


def test_sunk_acquisition_cost_cancels_at_retention():
    r = router(tool_budget=1)
    proposal = r.propose("where")
    action = proposal.action.tool
    # Pre-acquisition net gain after correction is -1, but retained evidence is
    # worth +1 once its already paid cost of 2 is held fixed for every subset.
    result = r.commit(proposal, {"tool:" + action: packet(action, metadata={"adjustment": -2})})
    assert result.committed_action == proposal.action
    assert result.gain_over_null == pytest.approx(1)


def test_all_rejected_keeps_expert_input_and_answers_once():
    r = router(tool_budget=1)
    calls = {"generate": 0, "acquire": 0}

    class Backbone:
        def set_adapter(self, name):
            self.name = name

        def generate_answer(self, prompt, image, adapter_id):
            calls["generate"] += 1
            assert image is observation
            assert prompt == "[Question]\nwhere"
            assert self.name == "spatial_reasoning"
            return "one answer"

    def provider(proposal, question, image):
        calls["acquire"] += 1
        return {"tool:" + proposal.action.tool: packet(proposal.action.tool, schema_valid=False)}

    observation = object()
    pipe = EKTRouterPipeline(Backbone(), {}, r, None, None)
    assert pipe.run_bounded("where", observation, provider) == "one answer"
    assert calls == {"generate": 1, "acquire": 1}
    assert pipe.last_route.committed_action.is_null


def test_null_route_never_calls_provider():
    r = router(tool_budget=0)

    class Backbone:
        def set_adapter(self, name):
            pass

        def generate_answer(self, **kwargs):
            return "one"

    pipe = EKTRouterPipeline(Backbone(), {}, r, None, None)
    assert pipe.run_bounded("q", None, lambda *a: pytest.fail("provider called for null")) == "one"


def test_provider_called_once_and_failure_becomes_invalid_packet():
    counts = []

    def good(q, im):
        counts.append("depth")
        return packet("depth")

    def bad(q, im):
        counts.append("ground")
        raise RuntimeError("tool failure")

    provider = PacketProvider({"tool:depth": good, "tool:ground": bad})
    packets = provider(SimpleNamespace(action=RouteAction(tool="depth+ground")), "q", None)
    assert counts == ["depth", "ground"]
    assert not packets["tool:ground"].validate().valid


def test_real_backend_does_not_return_stub_on_missing_weights():
    model = QwenRoutedBackbone("missing-local-checkpoint")
    with pytest.raises(FileNotFoundError):
        model.generate_answer("q")
    with pytest.raises(ValueError):
        model.set_adapter("general")


def test_adapter_selection_survives_lazy_loading(monkeypatch, tmp_path):
    import sys

    events = []

    class Model:
        device = "cpu"

        def eval(self):
            return self

        def requires_grad_(self, value):
            return self

        def load_adapter(self, path, adapter_name, **kw):
            events.append(("load", adapter_name))

        def set_adapter(self, name):
            events.append(("select", name))

    base = Model()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
            Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *a, **k: base),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=lambda *a, **k: base)),
    )
    model = QwenRoutedBackbone(
        tmp_path, {"general": str(tmp_path), "spatial_reasoning": str(tmp_path)}
    )
    model.set_adapter("spatial_reasoning")
    model._load()
    assert model._active_adapter == "spatial_reasoning"
    assert events[-1] == ("select", "spatial_reasoning")


def test_frame_budget_is_deterministic():
    model = QwenRoutedBackbone("missing", max_video_frames=3)
    content = model._build_messages("q", list(range(7)))[0]["content"]
    assert [x["image"] for x in content if x["type"] == "image"] == [0, 3, 6]


def test_action_multihot_and_null_anchor():
    from modules.neural_utility_backend import ActionVocabulary

    vocab = ActionVocabulary(costs={"tool:depth": 0.2, "tool:ocr": 0.3})
    assert not vocab.encode(RouteAction()).features.any()
    assert vocab.encode(RouteAction(tool="ocr+depth")).cost == pytest.approx(0.5)


def test_paired_target_uses_normalized_outcomes():
    from train.build_supervision import pairs

    record = {
        "id": "x",
        "media_group": "scene",
        "query_embedding": [0.2],
        "outcomes": [
            {
                "expert": "general",
                "action": {"knowledge": "none", "tool": "none"},
                "cost": 0,
                "score": 0.5,
            },
            {
                "expert": "general",
                "action": {"knowledge": "none", "tool": "depth"},
                "cost": 0.1,
                "score": 0.8,
            },
        ],
    }
    row = list(pairs(record))[0]
    assert row["target_gain"] == pytest.approx(-0.2)
    assert row["preference"] == -1
    record["outcomes"][0]["score"] = 50
    with pytest.raises(ValueError):
        list(pairs(record))
