"""Train one EKT specialist LoRA with MS-SWIFT.

The paper-aligned hyperparameters live in the public YAML files. Local model,
data, and output paths can be overridden with command-line arguments.

Usage
-----
    python train/train_expert_swift.py \
        --config configs/train/spatial_reasoning.yaml --gpus 0,1
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
from pathlib import Path

import yaml


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(os.path.expandvars(f.read())) or {}


def _fmt_flag(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_swift_cmd(cfg: dict, run_tag: str, extra_args: list[str]) -> list[str]:
    output_dir = Path(cfg["output_dir"]) / run_tag
    cmd = [
        "swift", str(cfg.get("task", "sft")),
        "--model", cfg["model_path"],
        "--dataset", cfg["data_path"],
        "--tuner_type", str(cfg.get("tuner_type", "lora")),
        "--torch_dtype", str(cfg.get("torch_dtype", "bfloat16")),
        "--num_train_epochs", str(cfg.get("num_train_epochs", 1)),
        "--per_device_train_batch_size", str(cfg.get("per_device_train_batch_size", 1)),
        "--gradient_accumulation_steps", str(cfg.get("gradient_accumulation_steps", 2)),
        "--learning_rate", str(cfg.get("learning_rate", 1.0e-4)),
        "--lora_rank", str(cfg.get("lora_rank", 128)),
        "--lora_alpha", str(cfg.get("lora_alpha", 256)),
        "--lora_dropout", str(cfg.get("lora_dropout", 0.05)),
        "--freeze_vit", _fmt_flag(cfg.get("freeze_vision_tower", True)),
        "--freeze_aligner", _fmt_flag(cfg.get("freeze_aligner", True)),
        "--warmup_ratio", str(cfg.get("warmup_ratio", 0.05)),
        "--weight_decay", str(cfg.get("weight_decay", 0.1)),
        "--lr_scheduler_type", str(cfg.get("lr_scheduler_type", "cosine")),
        "--gradient_checkpointing", _fmt_flag(cfg.get("gradient_checkpointing", True)),
        "--split_dataset_ratio", str(cfg.get("val_ratio", 0.001)),
        "--save_steps", str(cfg.get("save_steps", 500)),
        "--save_total_limit", str(cfg.get("save_total_limit", 3)),
        "--logging_steps", str(cfg.get("logging_steps", 10)),
        "--max_length", str(cfg.get("max_length", 32768)),
        "--output_dir", str(output_dir),
        "--report_to", str(cfg.get("report_to", "none")),
        "--seed", str(cfg.get("seed", 42)),
    ]
    target_modules = cfg.get("target_modules", "all-linear")
    if isinstance(target_modules, list):
        cmd += ["--target_modules", *[str(item) for item in target_modules]]
    else:
        cmd += ["--target_modules", str(target_modules)]
    if cfg.get("deepspeed"):
        cmd += ["--deepspeed", str(cfg["deepspeed"])]
    if cfg.get("max_steps"):
        cmd += ["--max_steps", str(cfg["max_steps"])]
    if cfg.get("model_type"):
        cmd += ["--model_type", str(cfg["model_type"])]
    cmd += list(extra_args)
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="specialist YAML configuration")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--run_tag", default="", help="override auto tag (date)")
    ap.add_argument("--dry_run", action="store_true", help="print command only")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="extra swift sft args after --")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = _load(cfg_path)
    expert = cfg.get("expert", cfg_path.stem)
    date = _dt.datetime.now().strftime("%Y%m%d")
    run_tag = args.run_tag or f"{date}_{expert}"

    extra = [a for a in args.extra if a != "--"]
    cmd = build_swift_cmd(cfg, run_tag, extra)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["WANDB_PROJECT"] = str(cfg.get("wandb_project", "KnowinRouter"))
    env["WANDB_NAME"] = f"specialist_{expert}_{run_tag}"
    env.setdefault("WANDB_MODE", "disabled")

    n_gpu = len([g for g in args.gpus.split(",") if g.strip()]) or 1
    env.setdefault("NPROC_PER_NODE", str(n_gpu))

    print("[train_expert_swift] run_tag =", run_tag)
    print("[train_expert_swift] CUDA_VISIBLE_DEVICES =", args.gpus)
    print("[train_expert_swift] WANDB_NAME =", env["WANDB_NAME"])
    print("[train_expert_swift] CMD:")
    print("  " + " ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
