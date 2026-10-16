"""OFFLINE complete-route outcome collection. Never used for online verification.

Real mode explicitly requires local checkpoints and a local native scorer that
returns [0,1]. The command makes multiple answer generations per training probe.
"""

import argparse
import json
from pathlib import Path
from modules.factory import load_global_config, _local_path
from modules.runtime import load_callable
from modules.utility_router import (
    ExpertSpec,
    InteractionAwareUtilityRouter,
    RouteProposal,
    UtilityTerms,
)
from modules.taxonomy import EXPERT_IDS, EXPERTS


def collect(args):
    cfg = load_global_config(args.config)
    experts = [ExpertSpec(EXPERTS[e].name, EXPERTS[e].index, e, ()) for e in EXPERT_IDS]
    routing = cfg["router"]
    router = InteractionAwareUtilityRouter(
        backend=None,
        experts=experts,
        tool_actions=("none", "depth", "ground", "ocr"),
        tool_budget=routing.get("tool_budget", 2),
        tool_costs=routing.get("tool_costs"),
        max_provider_calls=routing.get("max_provider_calls", 3),
    )
    from modules.neural_utility_backend import ActionVocabulary, NeuralUtilityBackend
    import torch

    vocabulary = ActionVocabulary(costs=cfg.get("action_costs", {}))
    if args.backend == "real":
        if not args.scorer:
            raise ValueError(
                "real collection requires --scorer module:function returning normalized [0,1]"
            )
        from modules.qwen_backbone import QwenRoutedBackbone, FrozenQwenDescriptor
        from modules.pipeline import EKTRouterPipeline

        adapters = {e: _local_path(cfg["experts"][e]["adapter_path"]) for e in EXPERT_IDS}
        if any(not Path(p).is_dir() for p in adapters.values()):
            raise FileNotFoundError("all three local expert adapters are required")
        model = QwenRoutedBackbone(
            _local_path(cfg["paths"]["backbone"]),
            adapters,
            max_video_frames=cfg["evaluation"]["max_video_frames"],
        )
        descriptor = FrozenQwenDescriptor(_local_path(cfg["paths"]["descriptor"]))
        pipeline = EKTRouterPipeline(model, adapters, None, None, None)
        provider = load_callable(cfg["provider_factory"])(cfg)
        score_fn = load_callable(args.scorer)
    target = Path(args.output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        for line in Path(args.data_path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            question, media = row["question"], row.get("media")
            outcomes = []
            embedding = (
                descriptor(question, media).tolist()
                if args.backend == "real"
                else [0.1, 0.2, 0.3, 0.4]
            )
            for expert in experts:
                for action in router.feasible_actions():
                    cost = vocabulary.encode(action).cost
                    if args.backend == "real":
                        proposal = RouteProposal(
                            expert, action, UtilityTerms(0, 0, 0, cost), 1, None
                        )
                        packets = provider(proposal, question, media) if not action.is_null else {}
                        valid = all(p.validate().valid for p in packets.values())
                        # Paired outcomes use exactly this acquired packet set; no reacquisition.
                        evidence = {
                            "knowledge": "\n".join(
                                p.content
                                for p in packets.values()
                                if p.channel.value == "knowledge"
                            ),
                            "tool": "\n".join(
                                p.content for p in packets.values() if p.channel.value == "tool"
                            ),
                        }
                        answer = pipeline._one_pass(
                            {"adapter": expert.adapter_id},
                            question,
                            media,
                            evidence_override=evidence,
                        )
                        score = float(score_fn(question, row["reference"], answer))
                        if not 0 <= score <= 1:
                            raise ValueError(
                                "scorer must return a finite normalized score in [0,1]"
                            )
                        features = NeuralUtilityBackend.packet_features(
                            packets, device=torch.device("cpu"), dtype=torch.float32
                        ).tolist()
                    else:
                        answer, score, packets, valid, features = (
                            "[MOCK offline candidate]",
                            0.0,
                            {},
                            False,
                            [0.0] * 12,
                        )
                    outcomes.append(
                        {
                            "expert": expert.checkpoint_key,
                            "action": {"knowledge": action.knowledge, "tool": action.tool},
                            "cost": cost,
                            "score": score,
                            "candidate": answer,
                            "packets_valid": valid,
                            "packet_features": features,
                            "packets": {k: p.to_dict() for k, p in packets.items()},
                        }
                    )
            result = {
                "id": row["id"],
                "media_group": row["media_group"],
                "query_embedding": embedding,
                "backend": args.backend,
                "scorer": args.scorer,
                "outcomes": outcomes,
            }
            stream.write(json.dumps(result) + "\n")
            stream.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--config", default="configs/global.yaml")
    p.add_argument("--backend", choices=("mock", "real"), default="mock")
    p.add_argument("--scorer", help="local native scorer; never an implicit paid judge")
    collect(p.parse_args())


if __name__ == "__main__":
    main()
