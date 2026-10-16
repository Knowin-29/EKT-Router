"""Build the actual bounded runtime from local checkpoints and configuration."""

import importlib
from pathlib import Path
from .factory import load_global_config, _local_path
from .pipeline import EKTRouterPipeline
from .utility_router import InteractionAwareUtilityRouter, paper_experts


def load_callable(spec):
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


def load_router(path, query_encoder, router_config=None, device="cpu"):
    import torch
    from .neural_utility_backend import (
        ActionVocabulary,
        FactorizedUtilityModel,
        NeuralUtilityBackend,
        PacketQualityVerifier,
    )

    data = torch.load(path, map_location=device, weights_only=True)
    if data.get("format_version") != "knowinrouter/2":
        raise ValueError(
            "checkpoint predates the corrected action encoding; retrain using this release"
        )
    if data.get("synthetic_smoke"):
        raise ValueError("synthetic smoke checkpoints cannot be used as real models")
    cfg = data["config"]
    vocabulary = ActionVocabulary(
        data["knowledge_actions"], data["tool_actions"], data["action_costs"]
    )
    model = FactorizedUtilityModel(
        cfg["query_dim"],
        cfg["signature_dim"],
        vocabulary.feature_dim,
        cfg.get("hidden_dim", 256),
        cfg.get("interaction_rank", 64),
    ).to(device)
    model.load_state_dict(data["utility_model"])
    model.eval()
    if not data.get("verifier_trained"):
        raise ValueError("real bounded inference requires a trained Packet Verifier")
    verifier = PacketQualityVerifier(
        cfg["query_dim"],
        cfg["signature_dim"],
        vocabulary.feature_dim,
        cfg.get("verifier_hidden_dim", 128),
    ).to(device)
    verifier.load_state_dict(data["packet_verifier"])
    verifier.eval()
    experts = paper_experts(data["expert_signatures"])
    backend = NeuralUtilityBackend(
        model=model,
        query_encoder=query_encoder,
        action_vocabulary=vocabulary,
        cost_weight=data["cost_weight"],
        verifier=verifier,
        signature_bank=[e.capability_signature for e in experts],
    )
    settings = dict(data.get("operating_point", {}))
    if router_config:
        # Evaluation must use the frozen operating point, not silently override it.
        for key in settings:
            if key in router_config and router_config[key] != settings[key]:
                raise ValueError(
                    f"configuration differs from frozen checkpoint operating point: {key}"
                )
    return InteractionAwareUtilityRouter(
        backend=backend,
        experts=experts,
        knowledge_actions=vocabulary.knowledge_actions,
        tool_actions=vocabulary.tool_actions,
        **settings,
    )


def build_runtime(config_path=None):
    from .qwen_backbone import FrozenQwenDescriptor, QwenRoutedBackbone

    config = load_global_config(config_path)
    device = config.get("device", "cuda")
    adapters = {key: _local_path(value["adapter_path"]) for key, value in config["experts"].items()}
    from .taxonomy import EXPERT_IDS

    if set(adapters) != set(EXPERT_IDS) or any(not Path(p).is_dir() for p in adapters.values()):
        raise FileNotFoundError("configure all three local adapters, including CPT-trained General")
    paths = config["paths"]
    descriptor = FrozenQwenDescriptor(
        _local_path(paths["descriptor"]),
        device=device,
        max_video_frames=config["evaluation"]["max_video_frames"],
    )
    router = load_router(
        _local_path(config["router"]["checkpoint"]), descriptor, config["router"], device
    )
    backbone = QwenRoutedBackbone(
        _local_path(paths["backbone"]),
        adapters,
        device=device,
        max_new_tokens=config["evaluation"]["max_new_tokens"],
        max_video_frames=config["evaluation"]["max_video_frames"],
    )
    provider = load_callable(config["provider_factory"])(config)
    pipeline = EKTRouterPipeline(backbone, adapters, router, None, None)
    return pipeline, provider, config
