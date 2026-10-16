"""Train the factorized EKT utility model from paired route outcomes.

Expected JSONL fields are documented in the repository README. The trainer
never consumes benchmark identifiers or reference answers; query embeddings,
expert signatures, typed actions, and paired outcome differences are sufficient.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.neural_utility_backend import (  # noqa: E402
    ActionVocabulary,
    FactorizedUtilityModel,
    PacketQualityVerifier,
    PairwiseUtilityLoss,
)
from modules.utility_router import RouteAction  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class PairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query": torch.tensor([row["query_embedding"] for row in rows], dtype=torch.float32),
        "left": [row["left"] for row in rows],
        "right": [row["right"] for row in rows],
        "preference": torch.tensor([row["preference"] for row in rows], dtype=torch.float32),
        "target_gain": torch.tensor([row["target_gain"] for row in rows], dtype=torch.float32),
    }


def encode_routes(
    routes: Iterable[Mapping[str, Any]],
    signatures: Mapping[str, list[float]],
    vocabulary: ActionVocabulary,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    expert_rows, action_rows, active_rows, costs = [], [], [], []
    for route in routes:
        expert_rows.append(signatures[str(route["expert"])])
        action_row = route.get("action", {})
        encoded = vocabulary.encode(
            RouteAction(
                knowledge=str(action_row.get("knowledge", "none")),
                tool=str(action_row.get("tool", "none")),
            ),
            device=device,
        )
        action_rows.append(encoded.features)
        active_rows.append(encoded.active)
        costs.append(float(route.get("cost", encoded.cost)))
    return (
        torch.tensor(expert_rows, dtype=torch.float32, device=device),
        torch.stack(action_rows),
        torch.stack(active_rows),
        torch.tensor(costs, dtype=torch.float32, device=device),
    )


def route_utility(
    model: FactorizedUtilityModel,
    query: Tensor,
    routes: Iterable[Mapping[str, Any]],
    signatures: Mapping[str, list[float]],
    vocabulary: ActionVocabulary,
    cost_weight: float,
) -> Tensor:
    expert, action, active, cost = encode_routes(routes, signatures, vocabulary, query.device)
    expert_bank = torch.tensor(list(signatures.values()), dtype=torch.float32, device=query.device)
    output = model(query, expert, action, active, expert_bank=expert_bank)
    return output["utility_without_cost"] - cost_weight * cost


def train(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    random.seed(int(config.get("seed", 42)))
    torch.manual_seed(int(config.get("seed", 42)))

    signatures = json.loads(Path(args.signatures).read_text(encoding="utf-8"))
    rows = read_jsonl(Path(args.train_path or config["train_path"]))
    vocabulary = ActionVocabulary(
        knowledge_actions=config.get("knowledge_actions", ("none", "template", "rag")),
        tool_actions=config.get("tool_actions", ("none", "depth", "ground", "ocr")),
        costs=config.get("action_costs", {}),
    )
    validation_path = config.get("validation_path")
    validation_rows = read_jsonl(Path(validation_path)) if validation_path else []
    if validation_path and not validation_rows:
        raise ValueError("configured validation data are empty")
    if validation_rows:
        if any(not row.get("media_group") for row in rows + validation_rows):
            raise ValueError("media_group is required to audit held-out separation")
        if {row["media_group"] for row in rows} & {row["media_group"] for row in validation_rows}:
            raise ValueError("train/validation media groups overlap")
    if not rows:
        raise ValueError("training data are empty")
    query_dim = int(config.get("query_dim", 2048))
    signature_dim = int(config.get("signature_dim", 3))
    model = FactorizedUtilityModel(
        query_dim=query_dim,
        signature_dim=signature_dim,
        action_dim=vocabulary.feature_dim,
        hidden_dim=int(config.get("hidden_dim", 256)),
        interaction_rank=int(config.get("interaction_rank", 64)),
    )
    verifier = PacketQualityVerifier(
        query_dim=query_dim,
        signature_dim=signature_dim,
        action_dim=vocabulary.feature_dim,
        hidden_dim=int(config.get("verifier_hidden_dim", 128)),
    )
    device = torch.device(args.device)
    model.to(device)
    verifier.to(device)
    parameters = list(model.parameters()) + list(verifier.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    objective = PairwiseUtilityLoss(calibration_weight=args.calibration_weight)
    loader = DataLoader(
        PairDataset(rows),
        batch_size=int(config.get("batch_size", 256)),
        shuffle=True,
        collate_fn=collate,
    )
    verifier_rows = read_jsonl(Path(args.verifier_path)) if args.verifier_path else []
    if validation_rows:
        valid_groups = {row["media_group"] for row in validation_rows}
        if any(
            not row.get("media_group") or row["media_group"] in valid_groups
            for row in verifier_rows
        ):
            raise ValueError("verifier records need fitting-only media_group identifiers")
    best_validation = float("inf")
    best_state = None

    for epoch in range(int(config.get("epochs", 10))):
        model.train()
        verifier.train()
        running = 0.0
        for batch in loader:
            query = batch["query"].to(device)
            left = route_utility(
                model, query, batch["left"], signatures, vocabulary, args.cost_weight
            )
            right = route_utility(
                model, query, batch["right"], signatures, vocabulary, args.cost_weight
            )
            losses = objective(
                left,
                right,
                batch["preference"].to(device),
                batch["target_gain"].to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            running += float(losses["loss"].item())

        print(f"epoch={epoch + 1} pairwise_loss={running / max(len(loader), 1):.6f}")
        if validation_rows:
            model.eval()
            with torch.no_grad():
                batch = collate(validation_rows)
                query = batch["query"].to(device)
                val = objective(
                    route_utility(
                        model, query, batch["left"], signatures, vocabulary, args.cost_weight
                    ),
                    route_utility(
                        model, query, batch["right"], signatures, vocabulary, args.cost_weight
                    ),
                    batch["preference"].to(device),
                    batch["target_gain"].to(device),
                )["loss"].item()
            if val < best_validation:
                best_validation = val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    # Stage 2 freezes U_phi before fitting residual packet corrections.
    model.eval()
    model.requires_grad_(False)
    for _ in range(int(config.get("verifier_epochs", config.get("epochs", 10)))):
        for row in verifier_rows:
            query = torch.tensor(row["query_embedding"], dtype=torch.float32, device=device)
            expert = torch.tensor(signatures[row["expert"]], dtype=torch.float32, device=device)
            action = vocabulary.encode(RouteAction(**row["action"]), device=device)
            if not action.active.item():
                raise ValueError("null verifier correction is fixed at zero, not trained")
            packet = torch.tensor(row["packet_features"], dtype=torch.float32, device=device)
            if packet.numel() != 12 or not torch.isfinite(packet).all():
                raise ValueError("packet_features must contain 12 finite values")
            if "realized_utility" in row:
                with torch.no_grad():
                    predicted = route_utility(
                        model,
                        query,
                        [{"expert": row["expert"], "action": row["action"]}],
                        signatures,
                        vocabulary,
                        args.cost_weight,
                    )[0]
                target = torch.tensor(float(row["realized_utility"]), device=device) - predicted
            elif config.get("synthetic_smoke", False):
                target = torch.tensor(float(row["target_adjustment"]), device=device)
            else:
                raise ValueError(
                    "real verifier training requires realized_utility against the frozen router"
                )
            loss = F.smooth_l1_loss(
                verifier(query, expert, action.features, packet).squeeze(), target
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    output = Path(args.output_dir or config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": "knowinrouter/2",
            "verifier_trained": bool(verifier_rows),
            "synthetic_smoke": bool(config.get("synthetic_smoke", False)),
            "cost_weight": args.cost_weight,
            "operating_point": config.get("operating_point", {"tool_budget": 1.0}),
            "descriptor_pooling": "last-layer-masked-mean",
            "utility_model": model.state_dict(),
            "packet_verifier": verifier.state_dict(),
            "config": config,
            "expert_signatures": signatures,
            "knowledge_actions": vocabulary.knowledge_actions,
            "tool_actions": vocabulary.tool_actions,
            "action_costs": vocabulary.costs,
        },
        output / "knowinrouter.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/router.yaml")
    parser.add_argument("--train-path", default="")
    parser.add_argument("--verifier-path", default="")
    parser.add_argument("--signatures", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cost-weight", type=float, default=1.0)
    parser.add_argument("--calibration-weight", type=float, default=0.1)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
