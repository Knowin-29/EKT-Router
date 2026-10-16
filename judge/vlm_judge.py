"""Qwen-VL judge client used for Stage 2 labeling.

Scoring modes
-------------
- ``heuristic``: no external call; reuses the numeric/string-overlap
  heuristics from :class:`judge.judge_eval.JudgeEvaluator`.
- ``openai_compat``: calls an OpenAI-compatible /v1/chat/completions endpoint
  (e.g. lmdeploy serving Qwen2.5-VL) with a concise 1-10 rubric.

The client is intentionally small so a versioned OpenAI-compatible evaluator
can be connected without coupling the release to a private service.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)


JUDGE_PROMPT = """You are evaluating an answer to an embodied question answering task.

Question: {question}
Reference Answer: {reference}
Candidate Answer: {candidate}

Rate the candidate answer on a 1-10 integer scale based on:
- Correctness: Does it match the reference answer?
- Completeness: Does it cover all requested aspects?
- Precision: For numerical answers, how close is it to the reference?

Output only a single integer score between 1 and 10."""


class VLMJudge:
    def __init__(
        self,
        api_url: Optional[str] = None,
        model: str = "qwen2.5-vl-72b-instruct",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_url = api_url or os.environ.get("EKT_JUDGE_URL", "")
        self.model = model
        self.api_key = api_key or os.environ.get("EKT_JUDGE_KEY", "EMPTY")
        self.timeout = timeout

    def score(self, question: str, reference: str, candidate: str) -> int:
        if not self.api_url:
            # heuristic fallback, same as JudgeEvaluator
            from .judge_eval import JudgeEvaluator

            return JudgeEvaluator().score(question, reference, candidate)
        prompt = JUDGE_PROMPT.format(question=question, reference=reference, candidate=candidate)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You output only integer scores."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 4,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            r = requests.post(
                self.api_url.rstrip("/") + "/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\d+", text)
            if not m:
                return 1
            return max(1, min(10, int(m.group(0))))
        except Exception as e:
            logger.warning("VLMJudge remote call failed: %s", e)
            return 1
