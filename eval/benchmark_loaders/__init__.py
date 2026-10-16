"""Unified entry point for the released EQA and grounding benchmark loaders."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


_BENCHMARK_MODULES: Dict[str, str] = {
    # EQA suite
    "openeqa": "openeqa:load_openeqa_samples",
    "unieqa": "unieqa:load_unieqa_samples",
    "robovqa": "robovqa:load_robovqa_samples",
    "embspatial": "embspatial:load_embspatial_samples",
    "cosmos": "cosmos:load_cosmos_samples",
    "ea_temporal": "ea_temporal:load_ea_temporal_samples",
    # Grounding / spatial
    "cv_bench": "cv_bench:load_cv_bench_samples",
    "vsibench": "vsibench:load_vsibench_samples",
    "where2place": "where2place:load_where2place_samples",
    "blink": "blink:load_blink_samples",
    "refspatial": "refspatial:load_refspatial_samples",
    "vabench_point": "vabench_point:load_vabench_point_samples",
    "erqa": "erqa:load_erqa_samples",
}

ALL_BENCHMARKS = list(_BENCHMARK_MODULES.keys())


def _get_loader(name: str) -> Optional[Callable]:
    spec = _BENCHMARK_MODULES.get(name)
    if spec is None:
        return None
    mod, attr = spec.split(":")

    try:
        m = importlib.import_module(f"{__name__}.{mod}")
        if hasattr(m, attr):
            return getattr(m, attr)
    except Exception as e:
        logger.warning("benchmark loader '%s' unavailable: %s", name, e)
    return None


def load_benchmark(name_or_path: str, max_samples: int = 0) -> Iterable[dict]:
    """Load samples by benchmark name (preferred) or raw data path."""
    p = Path(name_or_path)
    if p.exists():
        # delegate to routed_runner's simple JSON reader via this entrypoint
        import json

        if p.is_file() and p.suffix == ".jsonl":
            out: List[dict] = []
            with p.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    out.append(json.loads(line))
                    if max_samples and len(out) >= max_samples:
                        break
            return out
        if p.is_file() and p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data[:max_samples] if max_samples else data
    loader = _get_loader(name_or_path)
    if loader is None:
        raise ValueError(
            f"no benchmark loader for '{name_or_path}'. "
            "Provide the benchmark's public data root or a normalized JSONL path."
        )
    kwargs = {}
    if max_samples:
        kwargs["max_samples"] = max_samples
    try:
        return list(loader(**kwargs))
    except TypeError:
        data = list(loader())
        return data[:max_samples] if max_samples else data
