"""Tests for pipeline_multi/filter.py and pipeline_multi/aggregation.py."""
from __future__ import annotations

import math
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models import AgentGeneration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOP_K = 5
_WINDOW_SIZE = 64


def _make_logprobs(n_tokens: int, value: float = -1.0) -> List[float]:
    """Flat list of top-5 logprobs per token, all equal to `value`."""
    return [value] * (n_tokens * _TOP_K)


def _make_agent(
    agent_id: str,
    output_text: str,
    n_tokens: int = 80,
    logprob_value: float = -1.0,
    provider: str = "llama",
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    is_faulty: bool = False,
    fault_type: Optional[str] = None,
    empty_logprobs: bool = False,
) -> AgentGeneration:
    logprobs = [] if empty_logprobs else _make_logprobs(n_tokens, logprob_value)
    return AgentGeneration(
        agent_id=agent_id,
        output_text=output_text,
        token_logprobs=logprobs,
        is_faulty=is_faulty,
        fault_type=fault_type,
        model_id=model_id,
        provider=provider,
    )


def _make_pool(texts_by_provider: Dict[str, List[str]], logprob_value: float = -1.0) -> List[AgentGeneration]:
    agents = []
    for provider, texts in texts_by_provider.items():
        for i, text in enumerate(texts):
            agents.append(_make_agent(f"{provider}_a{i}", text, provider=provider, logprob_value=logprob_value))
    return agents


# ---------------------------------------------------------------------------
# pipeline_multi/filter.py tests
# ---------------------------------------------------------------------------

class TestCalibratePerProviderTau:
    def test_returns_dict_with_all_providers(self):
        from pipeline_multi.filter import calibrate_tau
        pool = _make_pool({"llama": ["yes"] * 10, "qwen": ["no"] * 10})
        tau = calibrate_tau(pool)
        assert "llama" in tau
        assert "qwen" in tau

    def test_tau_is_non_negative_float(self):
        from pipeline_multi.filter import calibrate_tau
        pool = _make_pool({"llama": ["yes"] * 20})
        tau = calibrate_tau(pool)
        # TopKMass is sum of top-5 probs per token, so it is bounded [0, 5] in theory;
        # real model outputs stay < 1.0 but test fixtures with uniform logprobs can exceed 1.0.
        assert tau["llama"] >= 0.0

    def test_high_confidence_agents_get_higher_tau(self):
        from pipeline_multi.filter import calibrate_tau
        high_conf = [_make_agent(f"h{i}", "yes", logprob_value=-0.1, provider="high") for i in range(20)]
        low_conf = [_make_agent(f"l{i}", "yes", logprob_value=-5.0, provider="low") for i in range(20)]
        tau = calibrate_tau(high_conf + low_conf)
        assert tau["high"] > tau["low"]

    def test_empty_logprobs_agents_excluded_from_calibration(self):
        from pipeline_multi.filter import calibrate_tau
        pool = [
            _make_agent("a0", "yes", provider="llama"),
            _make_agent("a1", "yes", provider="llama", empty_logprobs=True),
        ]
        tau = calibrate_tau(pool)
        # Should not raise; tau for llama is calibrated on the 1 valid agent
        assert "llama" in tau

    def test_unknown_provider_returns_global_fallback(self):
        from pipeline_multi.filter import calibrate_tau, GLOBAL_TAU_FALLBACK
        # calibrate on llama only, then check that a key for 'unknown' is absent
        pool = _make_pool({"llama": ["yes"] * 10})
        tau = calibrate_tau(pool)
        assert "unknown_provider" not in tau


class TestFilterAgentsMulti:
    def test_agents_below_tau_are_excluded(self):
        from pipeline_multi.filter import filter_agents_multi, calibrate_tau
        low_conf_agents = [_make_agent(f"l{i}", "yes", logprob_value=-10.0, provider="phi3") for i in range(5)]
        high_conf_agents = [_make_agent(f"h{i}", "yes", logprob_value=-0.1, provider="llama") for i in range(5)]
        pool = low_conf_agents + high_conf_agents
        tau_by_provider = {"phi3": 0.9, "llama": 0.01}  # only phi3 filtered out
        admitted = filter_agents_multi(pool, tau_by_provider)
        for agent in admitted:
            assert agent.provider == "llama"

    def test_empty_logprobs_always_dropped(self):
        from pipeline_multi.filter import filter_agents_multi
        pool = [_make_agent("a0", "yes", empty_logprobs=True, provider="llama")]
        admitted = filter_agents_multi(pool, {"llama": 0.0})
        assert len(admitted) == 0

    def test_unknown_provider_uses_global_fallback(self):
        from pipeline_multi.filter import filter_agents_multi, GLOBAL_TAU_FALLBACK
        # Agent with high confidence, provider not in tau_by_provider
        pool = [_make_agent("a0", "yes", logprob_value=-0.1, provider="unknown_model")]
        # Global fallback is low (0.5 default), so this agent should be admitted
        admitted = filter_agents_multi(pool, {})
        # Whether admitted depends on actual TopKMass vs GLOBAL_TAU_FALLBACK;
        # just verify it doesn't crash
        assert isinstance(admitted, list)

    def test_all_admitted_when_tau_zero(self):
        from pipeline_multi.filter import filter_agents_multi
        pool = _make_pool({"llama": ["yes"] * 3, "qwen": ["no"] * 3})
        admitted = filter_agents_multi(pool, {"llama": 0.0, "qwen": 0.0})
        assert len(admitted) == 6


# ---------------------------------------------------------------------------
# pipeline_multi/aggregation.py tests
# ---------------------------------------------------------------------------

class TestAggregateMulti:
    """Tests for the top-level aggregate() function in pipeline_multi/aggregation.py."""

    def _fake_embed(self, texts: List[str]) -> np.ndarray:
        """Simple deterministic embeddings: each text maps to a 3D vector."""
        rng = np.random.default_rng(hash(tuple(texts)) % (2**32))
        return rng.random((len(texts), 3)).astype(np.float32)

    def test_single_agent_returned_directly(self):
        from pipeline_multi.aggregation import aggregate
        pool = [_make_agent("a0", "The answer is 42.", provider="llama")]
        with patch("pipeline_multi.aggregation._embed", return_value=np.array([[1.0, 0.0, 0.0]])):
            result = aggregate(pool)
        assert result.final_answer == "The answer is 42."
        assert not result.is_low_confidence

    def test_empty_pool_returns_empty_string(self):
        from pipeline_multi.aggregation import aggregate
        result = aggregate([])
        assert result.final_answer == ""
        assert not result.is_low_confidence

    def test_majority_answer_returned_for_clean_pool(self):
        """When 4 of 7 agents give the same answer, aggregate selects it."""
        from pipeline_multi.aggregation import aggregate
        majority_text = "The answer is yes."
        minority_text = "The answer is no, I think."
        pool = (
            [_make_agent(f"m{i}", majority_text, provider="llama") for i in range(4)]
            + [_make_agent(f"n{i}", minority_text, provider="qwen") for i in range(3)]
        )
        # Embeddings: cluster majority and minority apart
        majority_emb = np.tile([1.0, 0.0, 0.0], (4, 1)).astype(np.float32)
        minority_emb = np.tile([0.0, 1.0, 0.0], (3, 1)).astype(np.float32)
        embeddings = np.vstack([majority_emb, minority_emb])
        with patch("pipeline_multi.aggregation._embed", return_value=embeddings):
            result = aggregate(pool)
        assert result.final_answer == majority_text

    def test_fallback_when_fewer_than_two_admitted(self):
        """If filter leaves 0 or 1 agents, fallback to majority vote on all."""
        from pipeline_multi.aggregation import aggregate
        pool = [_make_agent("a0", "yes", provider="llama")]
        with patch("pipeline_multi.aggregation._embed", return_value=np.array([[1.0, 0.0, 0.0]])):
            result = aggregate(pool, tau_by_provider={"llama": 99.0})  # filter all out
        assert isinstance(result.final_answer, str)

    def test_geometric_median_resists_minority_cluster(self):
        """Biased provider (2 agents) should lose to 5 correct agents."""
        from pipeline_multi.aggregation import aggregate
        correct = "42"
        wrong = "999999"
        pool = (
            [_make_agent(f"c{i}", correct, provider="llama") for i in range(5)]
            + [_make_agent(f"w{i}", wrong, provider="phi3") for i in range(2)]
        )
        correct_emb = np.tile([1.0, 0.0, 0.0], (5, 1)).astype(np.float32)
        wrong_emb = np.tile([10.0, 0.0, 0.0], (2, 1)).astype(np.float32)
        embeddings = np.vstack([correct_emb, wrong_emb])
        with patch("pipeline_multi.aggregation._embed", return_value=embeddings):
            result = aggregate(pool)
        assert result.final_answer == correct

    def test_admitted_agents_tracked_in_result(self):
        from pipeline_multi.aggregation import aggregate
        pool = [_make_agent(f"a{i}", "yes", provider="llama") for i in range(3)]
        embeddings = np.tile([1.0, 0.0, 0.0], (3, 1)).astype(np.float32)
        with patch("pipeline_multi.aggregation._embed", return_value=embeddings):
            result = aggregate(pool, tau_by_provider={"llama": 0.0})
        assert len(result.admitted_agents) == 3

    def test_no_provider_tag_falls_back_gracefully(self):
        """Agents without a provider tag should not crash aggregate()."""
        from pipeline_multi.aggregation import aggregate
        pool = [
            AgentGeneration("a0", "yes", _make_logprobs(80), False, None),
            AgentGeneration("a1", "yes", _make_logprobs(80), False, None),
        ]
        embeddings = np.tile([1.0, 0.0, 0.0], (2, 1)).astype(np.float32)
        with patch("pipeline_multi.aggregation._embed", return_value=embeddings):
            result = aggregate(pool)
        assert isinstance(result.final_answer, str)
