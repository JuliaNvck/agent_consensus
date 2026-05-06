from typing import List, Tuple

import numpy as np

from models import AgentGeneration

WINDOW_SIZE: int = 64
_TOP_K: int = 5  # number of logprob entries per token position


def _compute_topk_mass_trajectory(
    token_logprobs: List[float], w: int = WINDOW_SIZE
) -> np.ndarray:
    """Compute the causal sliding-window TopKMass trajectory.

    token_logprobs: flat list of top-5 logprobs per token position, length 5*T.
    Returns an ndarray of shape (T,) where entry i is:
        TopKMass(i) = (1 / window_size) * sum(topk_probs[max(0, i-w+1) : i+1])

    Raises ValueError if len(token_logprobs) is not a multiple of 5.
    """
    n = len(token_logprobs)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n % _TOP_K != 0:
        raise ValueError(
            f"token_logprobs length {n} is not a multiple of 5 "
            f"(expected 5 entries per token position)."
        )

    T = n // _TOP_K
    arr = np.array(token_logprobs, dtype=np.float64)
    probs = np.exp(arr.reshape(T, _TOP_K))        # (T, 5)
    topk_per_pos: np.ndarray = probs.sum(axis=1)  # (T,) — sum of top-5 at each position

    # Causal sliding window mean via prefix sums — O(T)
    cumsum = np.empty(T + 1, dtype=np.float64)
    cumsum[0] = 0.0
    np.cumsum(topk_per_pos, out=cumsum[1:])

    trajectory = np.empty(T, dtype=np.float64)
    for i in range(T):
        start = max(0, i - w + 1)
        window_len = i - start + 1
        trajectory[i] = (cumsum[i + 1] - cumsum[start]) / window_len

    return trajectory


def _agent_stats(
    trajectory: np.ndarray, warmup: int = WINDOW_SIZE
) -> Tuple[float, float]:
    """Return (mean, variance) of the stable (post-warmup) TopKMass trajectory.

    Skips the first `warmup` positions where the sliding window is partially
    populated, making the score length-invariant across output lengths.
    Falls back to the full trajectory if the output is shorter than warmup.
    """
    stable = trajectory[warmup:] if len(trajectory) > warmup else trajectory
    return float(stable.mean()), float(stable.var())


async def filter_agents(
    generations: List[AgentGeneration], tau: float = 0.0
) -> List[AgentGeneration]:
    """Module 1 reliability filter: admit agents whose mean TopKMass score >= tau.

    Agents with empty token_logprobs (e.g. F1_crash) are always dropped.
    Returns the subset of input objects (same instances, not copies).
    """
    admitted: List[AgentGeneration] = []
    for gen in generations:
        if not gen.token_logprobs:
            continue
        trajectory = _compute_topk_mass_trajectory(gen.token_logprobs)
        mean_score, _ = _agent_stats(trajectory)
        if mean_score >= tau:
            admitted.append(gen)
    return admitted
