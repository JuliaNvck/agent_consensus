from __future__ import annotations

from collections import Counter
from typing import List

import numpy as np
from scipy.optimize import minimize

from models import AgentGeneration
from pipeline.filter import _compute_topk_mass_trajectory
from pipeline import aggregation as _aggregation


def majority_voting(agents: List[AgentGeneration]) -> str:
    """Return the most frequent output_text among agents."""
    if not agents:
        return ""
    return Counter(g.output_text for g in agents).most_common(1)[0][0]


def _weighted_geometric_median(
    embeddings: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Weighted geometric median via L-BFGS-B.

    Minimises f(y) = Σ w_i · ‖x_i − y‖₂ with analytic gradient.
    Distance denominators clamped to 1e-10 to handle duplicate embeddings.
    """
    def _obj(y: np.ndarray) -> float:
        return float(np.sum(weights * np.linalg.norm(embeddings - y, axis=1)))

    def _grad(y: np.ndarray) -> np.ndarray:
        diffs = embeddings - y                                      # (N, D)
        dists = np.linalg.norm(diffs, axis=1, keepdims=True)       # (N, 1)
        dists = np.maximum(dists, 1e-10)
        return -np.sum(weights[:, None] * diffs / dists, axis=0)   # (D,)

    x0 = np.average(embeddings, axis=0, weights=weights)
    result = minimize(_obj, x0, jac=_grad, method="L-BFGS-B")
    return result.x


def soft_weighted_geometric_median(agents: List[AgentGeneration]) -> str:
    """Weighted Weiszfeld geometric median using per-agent TopKMass score as weight.

    Weight for each agent = mean of its TopKMass trajectory (0.0 for empty logprobs).
    Falls back to uniform weights when all weights are zero (e.g. all F1 agents).
    Returns the text of the agent nearest to the weighted median.
    """
    if not agents:
        return ""
    if len(agents) == 1:
        return agents[0].output_text

    # Compute per-agent TopKMass mean as continuous weight
    raw_weights: List[float] = []
    for gen in agents:
        if not gen.token_logprobs:
            raw_weights.append(0.0)
        else:
            traj = _compute_topk_mass_trajectory(gen.token_logprobs)
            raw_weights.append(float(traj.mean()))

    weights = np.array(raw_weights, dtype=np.float64)
    if weights.sum() < 1e-15:
        weights = np.ones(len(agents), dtype=np.float64)

    texts = [gen.output_text for gen in agents]
    # Use module-attribute lookup so conftest monkeypatch on pipeline.aggregation._embed applies
    embeddings = _aggregation._embed(texts)                         # (N, D)
    median = _weighted_geometric_median(embeddings, weights)        # (D,)
    dists = np.linalg.norm(embeddings - median, axis=1)            # (N,)
    return texts[int(np.argmin(dists))]
