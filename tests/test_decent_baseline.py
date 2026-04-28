from __future__ import annotations

import hashlib
from typing import List

import numpy as np
import pytest

from models import AgentGeneration
from eval.decent_baseline import _evaluate_candidate, run_decent_baseline


def _make_gen(agent_id: str, output_text: str) -> AgentGeneration:
    return AgentGeneration(
        agent_id=agent_id,
        output_text=output_text,
        token_logprobs=[-0.5, -1.0, -1.5, -2.0, -2.5] * 4,
        is_faulty=False,
        fault_type=None,
    )


class TestEvaluateCandidate:
    def test_returns_ndarray_of_shape_5(self):
        result = _evaluate_candidate("hello", 0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

    def test_values_in_range_0_to_20(self):
        result = _evaluate_candidate("test text", 3)
        assert all(0.0 <= v <= 20.0 for v in result)

    def test_deterministic(self):
        r1 = _evaluate_candidate("same text", 2)
        r2 = _evaluate_candidate("same text", 2)
        np.testing.assert_array_equal(r1, r2)

    def test_different_evaluator_ids_differ(self):
        r0 = _evaluate_candidate("same text", 0)
        r1 = _evaluate_candidate("same text", 1)
        assert not np.array_equal(r0, r1)

    def test_different_texts_differ(self):
        ra = _evaluate_candidate("alpha", 0)
        rb = _evaluate_candidate("beta", 0)
        assert not np.array_equal(ra, rb)


class TestRunDecentBaseline:
    def test_empty_returns_empty_string(self):
        assert run_decent_baseline([]) == ""

    def test_single_agent_returns_its_text(self):
        gen = _make_gen("a0", "only answer")
        assert run_decent_baseline([gen]) == "only answer"

    def test_highest_scoring_agent_selected(self, monkeypatch):
        score_map = {"high_text": np.full(5, 10.0), "low_text": np.full(5, 1.0)}
        monkeypatch.setattr(
            "eval.decent_baseline._evaluate_candidate",
            lambda text, eid: score_map[text],
        )
        agents = [_make_gen("a0", "high_text"), _make_gen("a1", "low_text")]
        assert run_decent_baseline(agents) == "high_text"

    def test_tie_broken_by_sha256_hash(self, monkeypatch):
        monkeypatch.setattr(
            "eval.decent_baseline._evaluate_candidate",
            lambda text, eid: np.full(5, 5.0),
        )
        text_a = "aaaa"
        text_b = "bbbb"
        expected = max(
            [text_a, text_b],
            key=lambda t: hashlib.sha256(t.encode()).hexdigest(),
        )
        agents = [_make_gen("a0", text_a), _make_gen("a1", text_b)]
        assert run_decent_baseline(agents) == expected

    def test_gm_computed_per_worker(self, monkeypatch):
        calls: List[tuple] = []

        def mock_eval(text: str, eid: int) -> np.ndarray:
            calls.append((text, eid))
            return np.ones(5)

        monkeypatch.setattr("eval.decent_baseline._evaluate_candidate", mock_eval)
        agents = [_make_gen(f"a{i}", f"text_{i}") for i in range(3)]
        run_decent_baseline(agents, num_evaluators=5)
        assert len(calls) == 3 * 5

    def test_custom_num_evaluators(self, monkeypatch):
        calls: List[tuple] = []

        def mock_eval(text: str, eid: int) -> np.ndarray:
            calls.append((text, eid))
            return np.ones(5)

        monkeypatch.setattr("eval.decent_baseline._evaluate_candidate", mock_eval)
        agents = [_make_gen("a0", "t0"), _make_gen("a1", "t1")]
        run_decent_baseline(agents, num_evaluators=3)
        assert len(calls) == 2 * 3

    def test_return_type_is_str(self, monkeypatch):
        monkeypatch.setattr(
            "eval.decent_baseline._evaluate_candidate",
            lambda text, eid: np.ones(5),
        )
        agents = [_make_gen("a0", "answer_a"), _make_gen("a1", "answer_b")]
        result = run_decent_baseline(agents)
        assert isinstance(result, str)
