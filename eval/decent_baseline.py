from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from models import AgentGeneration
from eval.baselines import _weighted_geometric_median

_N_CRITERIA: int = 5


def _evaluate_candidate(text: str, evaluator_id: int) -> np.ndarray:
    """Deterministic offline evaluator scoring one text across 5 criteria.

    Returns a (5,) array of scores in [0.0, 20.0] derived from a SHA-256 seed.
    Mockable in tests via monkeypatch on 'eval.decent_baseline._evaluate_candidate'.
    """
    seed_bytes = hashlib.sha256(f"{evaluator_id}:{text}".encode()).digest()
    seed_int = int.from_bytes(seed_bytes[:4], "big")
    rng = np.random.default_rng(seed_int)
    return rng.uniform(0.0, 20.0, size=_N_CRITERIA)


def run_decent_baseline(
    agents: List[AgentGeneration], num_evaluators: int = 5
) -> str:
    """DecentLLMs worker/evaluator scoring (Jo & Park) implemented offline.

    For each worker agent:
      1. Build an (num_evaluators, 5) score matrix via _evaluate_candidate.
      2. Compute the geometric median of that matrix (Weiszfeld, uniform weights).
      3. Sum the 5 robust-median components → scalar worker score.

    Returns the text of the highest-scoring worker.
    Tie-break: largest SHA-256 hex digest of the output_text.
    """
    if not agents:
        return ""

    scores: List[float] = []
    uniform_weights = np.ones(num_evaluators, dtype=np.float64)

    for gen in agents:
        score_matrix = np.array(
            [_evaluate_candidate(gen.output_text, eid) for eid in range(num_evaluators)],
            dtype=np.float64,
        )  # (num_evaluators, 5)
        robust_vector = _weighted_geometric_median(score_matrix, uniform_weights)  # (5,)
        scores.append(float(robust_vector.sum()))

    max_score = max(scores)
    candidates = [gen for gen, s in zip(agents, scores) if s == max_score]

    if len(candidates) == 1:
        return candidates[0].output_text

    return max(
        candidates,
        key=lambda g: hashlib.sha256(g.output_text.encode()).hexdigest(),
    ).output_text
