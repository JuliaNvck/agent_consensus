"""Module 1: Per-provider TopKMass reliability filter.

Calibrates a confidence threshold (τ) separately for each provider using the
25th-percentile of TopKMass scores across that provider's agents.  Agents below
their provider's τ are excluded before geometric-median aggregation.

Agents without a provider tag, or whose provider is absent from tau_by_provider,
fall back to GLOBAL_TAU_FALLBACK.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from models import AgentGeneration
from pipeline.filter import (
    _agent_stats,
    _compute_topk_mass_trajectory,
)

GLOBAL_TAU_FALLBACK: float = 0.5
_CALIBRATION_PERCENTILE: float = 25.0


def _score(agent: AgentGeneration) -> Optional[float]:
    """Return the mean TopKMass score for an agent, or None if logprobs are empty."""
    if not agent.token_logprobs:
        return None
    trajectory = _compute_topk_mass_trajectory(agent.token_logprobs)
    mean_score, _ = _agent_stats(trajectory)
    return mean_score


def calibrate_tau(agents: List[AgentGeneration]) -> Dict[str, float]:
    """Compute per-provider τ thresholds using the 25th-percentile of TopKMass scores.

    Agents without logprobs are skipped.  Providers represented in the pool are
    returned in the result dict; providers absent from the pool are not included
    (callers should fall back to GLOBAL_TAU_FALLBACK for missing keys).
    """
    scores_by_provider: Dict[str, List[float]] = {}
    for agent in agents:
        provider = agent.provider or "__unknown__"
        s = _score(agent)
        if s is None:
            continue
        scores_by_provider.setdefault(provider, []).append(s)

    tau: Dict[str, float] = {}
    for provider, scores in scores_by_provider.items():
        tau[provider] = float(np.percentile(scores, _CALIBRATION_PERCENTILE))
    return tau


def filter_agents_multi(
    agents: List[AgentGeneration],
    tau_by_provider: Dict[str, float],
) -> List[AgentGeneration]:
    """Admit agents whose mean TopKMass score >= their provider's τ threshold.

    Agents with empty logprobs are always excluded.
    Agents whose provider is not in tau_by_provider use GLOBAL_TAU_FALLBACK.
    """
    admitted: List[AgentGeneration] = []
    for agent in agents:
        s = _score(agent)
        if s is None:
            continue
        provider = agent.provider or "__unknown__"
        threshold = tau_by_provider.get(provider, GLOBAL_TAU_FALLBACK)
        if s >= threshold:
            admitted.append(agent)
    return admitted
