"""Module 2: Geometric median → nearest-centroid selection for multi-provider pools.

Reuses the geometric median and embedding logic from pipeline.aggregation directly.
No NLI, no weighted voting — geometric median nearest-centroid is the approach that
performed best under adversarial conditions in experiments (stage1_only condition).

Fallback: if fewer than 2 agents survive Module 1 filtering, fall back to
majority answer vote on the unfiltered pool to maintain liveness.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from models import AgentGeneration, ConsensusResult
from pipeline.aggregation import _embed, _geometric_median
from pipeline_multi.filter import GLOBAL_TAU_FALLBACK, filter_agents_multi


def _majority_vote_fallback(agents: List[AgentGeneration]) -> str:
    """Return the most common output_text.strip() across agents."""
    if not agents:
        return ""
    counts: Counter[str] = Counter(a.output_text.strip() for a in agents)
    return counts.most_common(1)[0][0]


def aggregate(
    generations: List[AgentGeneration],
    tau_by_provider: Optional[Dict[str, float]] = None,
) -> ConsensusResult:
    """Aggregate a multi-provider agent pool into a single consensus answer.

    Algorithm:
        1. Filter via per-provider TopKMass threshold (Module 1).
        2. If < 2 agents admitted: fall back to majority vote on unfiltered pool.
        3. Embed admitted outputs with sentence-transformers.
        4. Compute geometric median of embeddings.
        5. Return the admitted agent nearest to the geometric median.

    Args:
        generations: All agent outputs for one question (mixed providers).
        tau_by_provider: Per-provider threshold map from calibrate_tau().
                         Absent providers use GLOBAL_TAU_FALLBACK.

    Returns:
        ConsensusResult with final_answer, admitted_agents list, and
        is_low_confidence=False (nearest-centroid always produces an answer).
    """
    if not generations:
        return ConsensusResult(
            final_answer="",
            admitted_agents=[],
            is_low_confidence=False,
        )

    if len(generations) == 1:
        return ConsensusResult(
            final_answer=generations[0].output_text,
            admitted_agents=[generations[0].agent_id],
            is_low_confidence=False,
        )

    effective_tau = tau_by_provider or {}
    admitted = filter_agents_multi(generations, effective_tau)

    # Liveness fallback: if filter is too aggressive, use unfiltered pool
    if len(admitted) < 2:
        fallback_answer = _majority_vote_fallback(generations)
        return ConsensusResult(
            final_answer=fallback_answer,
            admitted_agents=[g.agent_id for g in generations],
            is_low_confidence=True,
        )

    texts = [g.output_text for g in admitted]
    embeddings = _embed(texts)                            # (N, D)
    median = _geometric_median(embeddings)                # (D,)
    dists = np.linalg.norm(embeddings - median, axis=1)  # (N,)
    nearest_idx = int(np.argmin(dists))

    return ConsensusResult(
        final_answer=admitted[nearest_idx].output_text,
        admitted_agents=[g.agent_id for g in admitted],
        is_low_confidence=False,
    )
