"""
TDD tests for pipeline/filter.py — Module 1 TopKMass reliability filter.

token_logprobs layout: 5 consecutive logprobs per token position (flat list).
  len(token_logprobs) == 5 * T  for an output of T tokens.
"""

import math
from typing import List

import numpy as np
import pytest

from models import AgentGeneration
from pipeline.filter import (
    WINDOW_SIZE,
    _agent_stats,
    _compute_topk_mass_trajectory,
    filter_agents,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_logprobs(topk_sums: List[float]) -> List[float]:
    """Build a flat logprob list from desired top-5 sum per position.

    Each position gets 5 equal probabilities that sum to topk_sums[i].
    Uses a floor of 1e-300 to avoid log(0).
    """
    logprobs: List[float] = []
    for s in topk_sums:
        p = max(s / 5.0, 1e-300)
        logprobs.extend([math.log(p)] * 5)
    return logprobs


def _make_gen(
    agent_id: str,
    topk_sums: List[float],
    is_faulty: bool = False,
    fault_type: str | None = None,
) -> AgentGeneration:
    return AgentGeneration(
        agent_id=agent_id,
        output_text="output",
        token_logprobs=_make_logprobs(topk_sums),
        is_faulty=is_faulty,
        fault_type=fault_type,
    )


# ---------------------------------------------------------------------------
# Unit: _compute_topk_mass_trajectory
# ---------------------------------------------------------------------------

class TestComputeTopkMassTrajectory:
    def test_single_token_full_mass(self) -> None:
        """T=1: only one position, top-5 sum = 1.0 → trajectory = [1.0]."""
        traj = _compute_topk_mass_trajectory(_make_logprobs([1.0]))
        assert traj.shape == (1,)
        assert pytest.approx(traj[0], abs=1e-9) == 1.0

    def test_uniform_low_confidence_flat_trajectory(self) -> None:
        """All positions equal → trajectory is constant, variance = 0."""
        sums = [0.5] * 10
        traj = _compute_topk_mass_trajectory(_make_logprobs(sums))
        assert traj.shape == (10,)
        np.testing.assert_allclose(traj, 0.5, atol=1e-9)

    def test_highly_confident_score_near_one(self) -> None:
        """Top-5 probs per position sum to ≈1.0 → trajectory ≈ 1.0."""
        # logprobs: one near-0, four very negative
        per_pos = [0.0, -100.0, -100.0, -100.0, -100.0]
        logprobs = per_pos * 20
        traj = _compute_topk_mass_trajectory(logprobs)
        np.testing.assert_allclose(traj, 1.0, atol=1e-6)

    def test_growing_window_before_w(self) -> None:
        """T=3 < W=64: all three positions use growing prefix windows."""
        # topk_per_pos = [0.8, 0.6, 1.0]
        # trajectory[0] = 0.8
        # trajectory[1] = (0.8 + 0.6) / 2 = 0.7
        # trajectory[2] = (0.8 + 0.6 + 1.0) / 3 = 0.8
        sums = [0.8, 0.6, 1.0]
        traj = _compute_topk_mass_trajectory(_make_logprobs(sums))
        assert traj.shape == (3,)
        assert pytest.approx(traj[0], abs=1e-9) == 0.8
        assert pytest.approx(traj[1], abs=1e-9) == 0.7
        assert pytest.approx(traj[2], abs=1e-9) == 0.8

    def test_window_exactly_w_size(self) -> None:
        """T=W=64: last position uses the first and only full window."""
        sums = [0.9] * 64
        traj = _compute_topk_mass_trajectory(_make_logprobs(sums), w=64)
        assert traj.shape == (64,)
        np.testing.assert_allclose(traj, 0.9, atol=1e-9)

    def test_sliding_window_evicts_old_token(self) -> None:
        """Spike at position 0 should disappear from the trajectory once the
        window slides past it (at position W).

        topk_per_pos = [1.0, 0.0, 0.0, ..., 0.0]  (T = W+1 = 65, W=64)

        trajectory[0]  = 1.0      (window = [0])
        trajectory[1]  = 0.5      (window = [0,1])
        trajectory[63] = 1/64     (window = [0..63], spike still inside)
        trajectory[64] = 0.0      (window = [1..64], spike evicted)
        """
        W = 64
        T = W + 1
        sums = [1.0] + [0.0] * W
        traj = _compute_topk_mass_trajectory(_make_logprobs(sums), w=W)

        assert traj.shape == (T,)
        assert pytest.approx(traj[0], abs=1e-9) == 1.0
        assert pytest.approx(traj[1], abs=1e-9) == 0.5
        assert pytest.approx(traj[63], abs=1e-9) == 1.0 / 64
        assert pytest.approx(traj[64], abs=1e-9) == 0.0

    def test_longer_sequence_beyond_w(self) -> None:
        """T=128, W=64: positions ≥ 63 all use full windows of size 64."""
        sums = [0.75] * 128
        traj = _compute_topk_mass_trajectory(_make_logprobs(sums), w=64)
        assert traj.shape == (128,)
        np.testing.assert_allclose(traj, 0.75, atol=1e-9)

    def test_empty_logprobs_returns_empty(self) -> None:
        traj = _compute_topk_mass_trajectory([])
        assert traj.shape == (0,)

    def test_raises_on_non_multiple_of_five(self) -> None:
        """logprobs length must be 5*T; anything else is a data contract violation."""
        with pytest.raises(ValueError, match="multiple of 5"):
            _compute_topk_mass_trajectory([0.0, -1.0, -2.0])


# ---------------------------------------------------------------------------
# Unit: _agent_stats
# ---------------------------------------------------------------------------

class TestAgentStats:
    def test_constant_trajectory_zero_variance(self) -> None:
        traj = np.full(64, 0.8)
        mean, var = _agent_stats(traj)
        assert pytest.approx(mean, abs=1e-12) == 0.8
        assert pytest.approx(var, abs=1e-12) == 0.0

    def test_known_mean_and_variance(self) -> None:
        """T=4, W=4.

        topk_per_pos = [0.6, 0.9, 0.3, 0.8]
        trajectory:
          [0] = 0.6
          [1] = (0.6+0.9)/2 = 0.75
          [2] = (0.6+0.9+0.3)/3 = 0.6
          [3] = (0.6+0.9+0.3+0.8)/4 = 0.65

        mean = (0.6+0.75+0.6+0.65)/4 = 2.6/4 = 0.65
        var  = mean of squared deviations
             = ((-0.05)^2 + (0.10)^2 + (-0.05)^2 + (0.0)^2) / 4
             = (0.0025 + 0.01 + 0.0025 + 0.0) / 4
             = 0.015 / 4 = 0.00375
        """
        sums = [0.6, 0.9, 0.3, 0.8]
        traj = _compute_topk_mass_trajectory(_make_logprobs(sums), w=4)
        mean, var = _agent_stats(traj)

        assert pytest.approx(mean, abs=1e-9) == 0.65
        assert pytest.approx(var, abs=1e-9) == 0.00375

    def test_two_element_trajectory(self) -> None:
        """Minimal non-trivial case: trajectory = [0.4, 0.8]
        mean = 0.6, var = 0.04.
        """
        traj = np.array([0.4, 0.8])
        mean, var = _agent_stats(traj)
        assert pytest.approx(mean, abs=1e-12) == 0.6
        assert pytest.approx(var, abs=1e-12) == 0.04


# ---------------------------------------------------------------------------
# Integration: filter_agents
# ---------------------------------------------------------------------------

class TestFilterAgents:
    @pytest.mark.asyncio
    async def test_admits_agent_above_tau(self) -> None:
        gen = _make_gen("a0", [0.9] * 10)
        result = await filter_agents([gen], tau=0.5)
        assert len(result) == 1
        assert result[0].agent_id == "a0"

    @pytest.mark.asyncio
    async def test_drops_agent_below_tau(self) -> None:
        gen = _make_gen("a0", [0.2] * 10)
        result = await filter_agents([gen], tau=0.5)
        assert result == []

    @pytest.mark.asyncio
    async def test_admits_at_exact_tau_boundary(self) -> None:
        """score == tau → admitted (>=)."""
        # All sums = 0.5 → constant trajectory → mean score = 0.5
        gen = _make_gen("a0", [0.5] * 20)
        result = await filter_agents([gen], tau=0.5)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_tau_zero_admits_all_nonempty(self) -> None:
        gens = [_make_gen(f"a{i}", [0.1 * i + 0.1] * 5) for i in range(5)]
        result = await filter_agents(gens, tau=0.0)
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_empty_logprobs_drops_agent(self) -> None:
        gen = AgentGeneration(
            agent_id="empty",
            output_text="",
            token_logprobs=[],
            is_faulty=True,
            fault_type="F1_crash",
        )
        result = await filter_agents([gen], tau=0.0)
        assert result == []

    @pytest.mark.asyncio
    async def test_mixed_agents_only_high_scorers_pass(self) -> None:
        """5 agents: top-2 have mean ≈ 0.9, bottom-3 have mean ≈ 0.2.
        tau=0.5 should admit exactly 2.
        """
        high = [_make_gen(f"h{i}", [0.9] * 10) for i in range(2)]
        low = [_make_gen(f"l{i}", [0.2] * 10) for i in range(3)]
        result = await filter_agents(high + low, tau=0.5)
        admitted_ids = {g.agent_id for g in result}
        assert admitted_ids == {"h0", "h1"}

    @pytest.mark.asyncio
    async def test_output_type_is_list_of_agent_generation(self) -> None:
        gens = [_make_gen("a0", [0.8] * 5)]
        result = await filter_agents(gens, tau=0.0)
        assert isinstance(result, list)
        assert isinstance(result[0], AgentGeneration)

    @pytest.mark.asyncio
    async def test_dataclass_identity_preserved(self) -> None:
        """Admitted objects must be the exact same instances, not copies."""
        gen = _make_gen("a0", [0.8] * 5)
        result = await filter_agents([gen], tau=0.0)
        assert result[0] is gen

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self) -> None:
        result = await filter_agents([], tau=0.5)
        assert result == []
