import re


JUDGE_PROMPT = """You are evaluating an answer to an embodied question answering task.

Question: {question}
Reference Answer: {reference}
Candidate Answer: {candidate}

Rate the candidate answer on a scale of 1-10 based on:
- Correctness: Does it match the reference answer?
- Completeness: Does it address all aspects of the question?
- Precision: For numerical answers, how close is it to the reference?

Output only a single integer score (1-10)."""


class JudgeEvaluator:
    def __init__(self, model="gpt-4o", api_key=None):
        self.model = model
        self.api_key = api_key

    def score(self, question: str, reference: str, candidate: str) -> int:
        ref = (reference or "").strip().lower()
        cand = (candidate or "").strip().lower()
        if not cand:
            return 1
        if cand == ref:
            return 10

        nums_ref = [float(x) for x in re.findall(r"-?\d+\.?\d*", ref)]
        nums_cand = [float(x) for x in re.findall(r"-?\d+\.?\d*", cand)]
        if nums_ref and nums_cand:
            r, c = nums_ref[0], nums_cand[0]
            err = abs(c - r) / (abs(r) + 1e-6)
            if err < 0.05:
                return 9
            if err < 0.15:
                return 8
            if err < 0.30:
                return 6
            return 4

        ref_set = set(ref.split())
        cand_set = set(cand.split())
        ov = len(ref_set & cand_set) / max(len(ref_set), 1)
        if ov > 0.7:
            return 8
        if ov > 0.4:
            return 6
        if ov > 0.2:
            return 4
        return 2
