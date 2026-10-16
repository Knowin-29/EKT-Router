<div align="center">

# EKT-Router: Expert–Evidence Alignment for Embodied Vision–Language Models

[Project](https://knowin-29.github.io/KnowinRouter.github.io/) · [Paper](https://knowin-29.github.io/KnowinRouter.github.io/assets/KnowinRouter.pdf) · [Data card](https://huggingface.co/datasets/Knowin-29/EKT-Router) · [Model card](https://huggingface.co/Knowin-29/EKT-Router) · [Collection](https://huggingface.co/collections/Knowin-29/ekt-router-6ad199395e1a1e54b4e1a179)

[![CI](https://github.com/Knowin-29/EKT-Router/actions/workflows/ci.yml/badge.svg)](https://github.com/Knowin-29/EKT-Router/actions/workflows/ci.yml)
[![MIT](https://img.shields.io/badge/code-MIT-147d82)](LICENSE)

</div>

EKT-Router jointly selects **one capability expert and the evidence it should receive**.
It is Knowin's first internal Mixture-of-LoRA (MoL) solution: three independent LoRA
experts share one Qwen3-VL-8B backbone. This describes an internal project milestone,
not a claim of academic priority. The source is derived from the supplied EKT-Router
implementation; the original MIT copyright and compatible Python APIs are retained.

**Release status:** code and a CPU mock demo are available. HF repositories contain
**Coming soon cards only**: no downloadable research data or model weights, and no
promised release date. This release has not reproduced the paper's experiments.

## 论文与工程对齐审计

| 状态 | 实际核验与处理 |
|---|---|
| 已实现 | 复用原工程的 outcome-derived signatures、因子化联合效用模型、Packet Verifier、单次回答管线、LoRA 包装器、训练代码与 13 个 benchmark loaders。 |
| 已修复 | 加入工具子集与显式预算；验证时固定已支付获取成本；General 必须有独立 CPT LoRA；修复首次加载时专家被覆盖的问题；真实推理失败直接报错；补齐本地推理/测评和离线监督入口；相对深度不再冒称米制距离。 |
| 已验证 | 原附件 19 项测试通过；发布版新增预算、独立拒绝、沉没成本、固定专家单次回答、零证据回退、适配器切换等回归测试。当前实际结果见 CI，不使用手写 passing 徽章。CPU 合成训练 smoke 可运行。 |
| 缺少证据 | 附件未提供真实专家/路由器权重、原始预测、实验日志、完整 split/scorer 版本和原始图表工程。61.4、74.7、+1.75 等只作对应协议的论文报告结果，不能视为本轮复现。 |

The original snapshot had no enforced tool-subset budget, subtracted avoidable
costs again during verification, used the bare backbone as General, could silently
return stub/empty predictions after real-model failures, and referenced a missing
evaluation runner. These are repaired in the inherited implementation.

## Method and invariants

- **Experts:** General (continued pretraining), Spatial Reasoning, Spatiotemporal;
  one independent LoRA adapter is active on the shared frozen Qwen3-VL-8B backbone.
- **Signatures:** normalized evidence-free diagnostic outcomes, projected by the
  learned router; expert names do not impose a capability hierarchy.
- **Joint proposal:** score `U = B + G + Gamma − lambda*C` over complete expert,
  optional knowledge, and budget-feasible tool subsets. Null evidence is feasible.
  `G(null)=Gamma(null)=0`; interactions are centered over the expert bank.
- **Acquire once:** each proposed provider runs at most once. Knowledge templates
  or retrieval and grounding, relative depth, or OCR return typed packets.
- **Verify with the expert fixed:** inspect schema, coverage, confidence, ambiguity
  and provenance. Score only subsets of acquired packets, holding the full
  acquisition cost fixed. Retain a subset only above the frozen gain margin.
- **Answer once:** no candidate answers, answer judge, reference answer, expert
  reselection or repeated tool loop in online verification. All rejected packets
  recover the same expert's evidence-free input. This is not a safety guarantee.

`modules.pipeline.EKTRouterPipeline` (compatible alias: `KnowinRouterPipeline`)
provides `run_bounded`. `modules.providers.PacketProvider` enforces one call per
selected provider and represents acquisition failures as invalid packets.

## Install and run without weights

Python 3.10+ is required. The following CPU path downloads Python dependencies,
not model weights, datasets, or paid API responses.

```bash
git clone https://github.com/Knowin-29/EKT-Router.git
cd EKT-Router
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python examples/bounded_routing_demo.py
python -m pytest -q
bash supplement/run_smoke.sh
```

The demo begins **MOCK CPU DEMO**. Its signatures, packet contents and coordinates
are synthetic; its backbone echoes the committed prompt. It is not model inference.
The smoke workflow fits tiny synthetic tensors on CPU, then deletes the temporary
checkpoint. Such checkpoints are marked synthetic and rejected by the real runner.

## Real inference and evaluation

Install `.[full]` only when you have the local model dependencies and checkpoints.
Copy `.env.example` to an ignored `.env`, edit it, and export the values:

```bash
python -m pip install -e '.[full]'
set -a
source .env
set +a
python -m eval.routed_runner --input data/evaluation.jsonl \
  --output results/run-001.jsonl --config configs/global.yaml
```

Each input row requires `id`, `question`, and `media` (local image/video path or
frame list). Add `reference` only for offline scoring. The router receives only
question/media, never the reference, sample ID, or benchmark identity. All three
expert adapters, the 2B descriptor, and a non-synthetic trained router/verifier
must exist locally. Missing assets fail with an error; no model download or mock
fallback is attempted. The same media and decoding budgets are used for all experts.

To score with a versioned native benchmark adapter, pass
`--scorer your_module:score --benchmark BENCHMARK --split SPLIT`.
The callable accepts `(question, reference, prediction)` and returns a score.
`eval.routed_runner:exact_match` is available for exact-match tasks only; it is
**not** a replacement for native OpenEQA/VSI-Bench/etc. evaluation.
The inherited loaders are format adapters, not proof of benchmark parity.

Predictions include committed routes and packet provenance. A completion manifest
records input/prediction/config hashes and the named scorer. This generic runner
does not certify protocol-matched reproduction. Use `eval/protocol_manifest.py`
to audit immutable checkpoint/scorer revisions, full coverage and protocol identity
before reporting such a result. An interrupted run has no completed manifest.

Provider configuration is in `configs/global.yaml`. Built-in local wrappers reuse
the original KnowledgeModule, Depth Anything V2 and Grounding DINO code; OCR uses
optional `.[ocr]` and an installed Tesseract executable. Missing providers return
invalid packets. Structural coverage and OCR confidence are **proxy features**, not
calibrated probabilities. The example retrieval implementation is dense/local;
the paper's exact hybrid index and provider calibration are not supplied. A custom
`KNOWINROUTER_PROVIDER_FACTORY=module:function` can supply versioned typed providers.
Its packet feature convention must match verifier training. Multi-frame built-in
perception uses the first frame and records frame 0; a task-specific video provider
must document its own bounded frame policy. No claim of paper-level tool parity is made.

## Train and construct supervision

These commands are entry points; real data and compute must be provided separately.
They are not executed by installation or CI. MS-SWIFT is optional (`pip install -e
'.[full,train]'`). The paper names a development SWIFT version whose immutable revision
is missing; current launchers target the v4 CLI and do not claim an exact recovered environment.

```bash
# Inspect specialist commands without launching training.
python train/train_expert_swift.py --config configs/train/spatial_reasoning.yaml --dry_run
python train/train_expert_swift.py --config configs/train/spatiotemporal.yaml --dry_run
python train/train_expert_swift.py --config configs/train/general.yaml --dry_run
# Remove --dry_run only when local checkpoints, licensed data and compute are ready.

python -m train.build_signatures --input data/router/diagnostic_outcomes.jsonl \
  --output data/router/expert_signatures.json

# OFFLINE: multiple route answers per probe; no implicit judge or paid service.
python -m judge.label_generator --backend real --config configs/global.yaml \
  --data-path data/router/probes-train.jsonl --output-path data/router/outcomes-train.jsonl \
  --scorer your_native_scorer:normalized_score
python -m train.build_supervision --input data/router/outcomes-train.jsonl \
  --output data/router/train.jsonl --verifier-output data/router/verifier.jsonl
# Collect/convert held-out media separately into data/router/validation.jsonl.
python -m train.train_utility_router --config configs/train/router.yaml \
  --signatures data/router/expert_signatures.json \
  --verifier-path data/router/verifier.jsonl --device cuda
```

Diagnostic rows require `expert`, `capability_axis`, `score`; non-[0,1] native scores
must include `score_min`/`score_max`. Missing cells or out-of-range scores are errors.
Offline probes require `id`, `media_group`, `question`, `media`, `reference`.
Paired records carry frozen query embeddings and complete routes; they never pass
answer keys to the router. A held-out validation file checks media-group separation
and selects the utility checkpoint. The verifier then fits realized-utility residuals
against that frozen model, rather than chasing a changing utility target. Cost/margin
settings must be chosen on held-out probes, copied into the training configuration,
and frozen before benchmark evaluation. `B` calibration and exact release operating
points still require genuine evidence-null outcomes; the tiny smoke is not calibration.

General CPT has a runnable `swift pt` launcher, but its corpus and original recipe
are missing. Its config explicitly contains **starting defaults**, not recovered
experiment settings. Spatial/video configs preserve the paper's rank 128, alpha 256,
dropout .05, two epochs, learning rate 1e-4, and frozen vision tower/aligner.

## Paper-reported results, with protocol boundaries

| Protocol | Paper report | Interpretation |
|---|---:|---|
| Q1: six embodied QA benchmarks, equal-weight native scores | 61.4 | Mixed reproduced/API/reported cross-system comparison in the manuscript; not this release's measurements. |
| Q4: ten general VLM benchmarks | 74.7 | Cross-system comparison; not a matched before/after retention experiment. |
| Q2: nine benchmarks, Knowledge and Tools disabled | 60.24 vs 58.49; +1.75 | Evidence-null expert-selection control; media-grouped 95% CI [1.25, 2.26], as reported. |
| Q3: separate frozen ablation protocol | 60.2 | Must not be merged with the Q1 61.4 result. |

The manuscript also reports losses against its General expert on **UniEQA −3.37**
and **OpenEQA −2.43**, despite a matched six-task mean gain of +1.78. These tradeoffs
are retained on the project page. Model capacity, tool availability, scorer and
media settings vary across cross-system comparisons. No new SOTA claim is made.

## Authors

Weilin Ruan¹², Yingjie Xu²³, Yangkai Wei², Bohan Zhou¹², Shuaibo Li¹,
Harold Haodong Chen³⁴, Kaiwen Zhou², Yinchuan Li², James Cheng¹ (corresponding author).

1. The Chinese University of Hong Kong
2. Knowin.AI
3. The Hong Kong University of Science and Technology (Guangzhou)
4. The Hong Kong University of Science and Technology

Contact: [rwlinno@gmail.com](mailto:rwlinno@gmail.com).

```bibtex
@misc{ruan2026ektrouter,
  title={EKT-Router: Expert--Evidence Alignment for Embodied Vision--Language Models},
  author={Ruan, Weilin and Xu, Yingjie and Wei, Yangkai and Zhou, Bohan and Li, Shuaibo and Chen, Harold Haodong and Zhou, Kaiwen and Li, Yinchuan and Cheng, James},
  year={2026},
  howpublished={Project preprint},
  url={https://knowin-29.github.io/KnowinRouter.github.io/}
}
```

No arXiv identifier is assigned by this repository. The downloadable source is
prepared for Overleaf/arXiv compilation; publication on arXiv is a separate action.

## Development and release

```bash
ruff check .
bash scripts/check_public_release.sh
bash scripts/build_supplement.sh
```

The audit scans tracked source for credentials, private endpoints, machine paths,
runtime artifacts and unexpected binaries; it does not reject public authors,
contact details or Chinese text. No private repository history was imported.
The code uses [MIT](LICENSE); see [NOTICE](NOTICE) for attribution and third-party
boundaries. Data, weights and institution trademarks are not relicensed as MIT.
