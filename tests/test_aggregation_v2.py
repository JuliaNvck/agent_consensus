"""
Tests for pipeline_v2/aggregation.py — distance-weighted majority vote.

No NLI model anywhere. All embedding I/O is mocked — no weights downloaded,
no GPU required. Tests cover:
  - _geometric_median (same implementation as v1, verified again here)
  - _infer_is_numeric (new: GSM8K vs StrategyQA heuristic)
  - _extract_answer_key (new: answer extraction without ground_truth)
  - aggregate (new: weighted voting instead of NLI-based selection)
"""

from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from models import AgentGeneration
from pipeline_v2.aggregation import (
    _extract_answer_key,
    _geometric_median,
    _infer_is_numeric,
    aggregate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gen(agent_id: str, output_text: str, is_faulty: bool = False) -> AgentGeneration:
    return AgentGeneration(
        agent_id=agent_id,
        output_text=output_text,
        token_logprobs=[-0.1, -1.0, -2.0, -3.0, -4.0] * 4,
        is_faulty=is_faulty,
        fault_type=None,
    )


# ---------------------------------------------------------------------------
# Unit: _geometric_median (identical math to v1 — re-verified)
# ---------------------------------------------------------------------------

class TestGeometricMedian:
    def test_symmetric_three_collinear_points(self) -> None:
        """Geometric median of [0,0],[1,0],[2,0] is exactly [1,0]."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        med = _geometric_median(pts)
        np.testing.assert_allclose(med, [1.0, 0.0], atol=1e-5)

    def test_outlier_resistance(self) -> None:
        """4 pts near origin + 1 far outlier: median << arithmetic mean."""
        pts = np.array([
            [0.0, 0.0], [0.1, 0.0], [-0.1, 0.0], [0.0, 0.1],
            [100.0, 0.0],
        ])
        med = _geometric_median(pts)
        mean = pts.mean(axis=0)
        assert abs(med[0]) < abs(mean[0])
        assert abs(med[0]) < 1.0

    def test_single_point_is_fixed_point(self) -> None:
        pts = np.array([[3.0, 7.0]])
        med = _geometric_median(pts)
        np.testing.assert_allclose(med, [3.0, 7.0], atol=1e-5)

    def test_two_identical_points_numerically_stable(self) -> None:
        """Two identical points must not cause division-by-zero."""
        pts = np.array([[1.0, 1.0], [1.0, 1.0]])
        med = _geometric_median(pts)
        np.testing.assert_allclose(med, [1.0, 1.0], atol=1e-5)

    def test_majority_cluster_wins_4_vs_3(self) -> None:
        """4 pts at [1,0] vs 3 pts at [-1,0]: median must be at [1,0]."""
        pts = np.array([[1.0, 0.0]] * 4 + [[-1.0, 0.0]] * 3)
        med = _geometric_median(pts)
        np.testing.assert_allclose(med, [1.0, 0.0], atol=1e-4)


# ---------------------------------------------------------------------------
# Unit: _infer_is_numeric
# ---------------------------------------------------------------------------

class TestInferIsNumeric:
    def test_gsm8k_style_many_numbers(self) -> None:
        """Outputs with long computation traces → numeric."""
        texts = [
            "She has 12 apples. She buys 8 more. Total = 12 + 8 = 20.",
            "Step 1: 45 / 5 = 9. Step 2: 9 * 3 = 27. The answer is 27.",
            "First 100, then subtract 37, giving 63.",
        ]
        assert _infer_is_numeric(texts) is True

    def test_strategyqa_style_yes_no(self) -> None:
        """Simple yes/no reasoning outputs → not numeric."""
        texts = [
            "Yes, penguins are birds.",
            "No, the moon is not a planet.",
            "Yes, this is correct because of the definition.",
        ]
        assert _infer_is_numeric(texts) is False

    def test_mixed_majority_numeric(self) -> None:
        """5 numeric, 2 yes/no → numeric wins (>50%)."""
        texts = [
            "The answer is 42 + 8 = 50.",
            "Compute: 100 - 37 - 5 = 58.",
            "Result: 3 * 4 * 5 = 60.",
            "After 15 steps, total is 7.",
            "Sum: 1 + 2 + 3 + 4 + 5 = 15.",
            "Yes, it works.",
            "No.",
        ]
        assert _infer_is_numeric(texts) is True

    def test_empty_texts_defaults_false(self) -> None:
        """All empty outputs (F1 crash) → not numeric (no numbers)."""
        texts = ["", "", ""]
        assert _infer_is_numeric(texts) is False


# ---------------------------------------------------------------------------
# Unit: _extract_answer_key
# ---------------------------------------------------------------------------

class TestExtractAnswerKey:
    def test_yes_from_yes_no_mode(self) -> None:
        assert _extract_answer_key("Yes, this is correct.", is_numeric=False) == "yes"

    def test_no_from_yes_no_mode(self) -> None:
        assert _extract_answer_key("No, it is not the case.", is_numeric=False) == "no"

    def test_yes_no_mode_empty_text_returns_empty(self) -> None:
        assert _extract_answer_key("", is_numeric=False) == ""

    def test_yes_no_mode_no_match_returns_empty(self) -> None:
        assert _extract_answer_key("The computation is 3 + 4.", is_numeric=False) == ""

    def test_numeric_last_number(self) -> None:
        """Returns the last number in the text."""
        assert _extract_answer_key("Step 1: 10. Step 2: 5. Answer: 42.", is_numeric=True) == "42"

    def test_numeric_strips_dollar_comma(self) -> None:
        assert _extract_answer_key("The answer is $1,000.", is_numeric=True) == "1000"

    def test_numeric_empty_text_returns_empty(self) -> None:
        assert _extract_answer_key("", is_numeric=True) == ""

    def test_numeric_no_numbers_returns_empty(self) -> None:
        assert _extract_answer_key("Yes, the answer is yes.", is_numeric=True) == ""


# ---------------------------------------------------------------------------
# Integration: aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    @pytest.mark.asyncio
    async def test_empty_list_returns_empty_string(self) -> None:
        text, is_low = await aggregate([])
        assert text == ""
        assert is_low is False

    @pytest.mark.asyncio
    async def test_single_agent_bypasses_embedding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With one agent, _embed must never be called."""
        embed_mock = MagicMock()
        monkeypatch.setattr("pipeline_v2.aggregation._embed", embed_mock)

        gen = _make_gen("a0", "The answer is yes.")
        text, is_low = await aggregate([gen])

        assert text == "The answer is yes."
        assert is_low is False
        embed_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_return_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0]])
        monkeypatch.setattr("pipeline_v2.aggregation._embed", lambda _: embeddings)
        gens = [_make_gen("a0", "yes"), _make_gen("a1", "no")]
        result = await aggregate(gens)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], bool)

    @pytest.mark.asyncio
    async def test_unanimous_agreement_returns_that_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All agents agree → return that answer, no low-confidence."""
        # All embeddings at origin → all weights equal → unanimous vote
        embeddings = np.zeros((4, 2))
        monkeypatch.setattr("pipeline_v2.aggregation._embed", lambda _: embeddings)

        gens = [_make_gen(f"a{i}", "Yes, the answer is yes.") for i in range(4)]
        text, is_low = await aggregate(gens)

        assert _extract_answer_key(text, is_numeric=False) == "yes"
        assert is_low is False

    @pytest.mark.asyncio
    async def test_majority_wins_with_equal_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """5 correct (yes), 2 wrong (no), all equidistant → majority wins."""
        # All embeddings identical → all distances = 0 → weights all 1.0
        embeddings = np.zeros((7, 2))
        monkeypatch.setattr("pipeline_v2.aggregation._embed", lambda _: embeddings)

        gens = (
            [_make_gen(f"a{i}", "Yes, the sky is blue.") for i in range(5)]
            + [_make_gen(f"b{i}", "No, it is not.") for i in range(2)]
        )
        text, is_low = await aggregate(gens)

        assert _extract_answer_key(text, is_numeric=False) == "yes"
        assert is_low is False

    @pytest.mark.asyncio
    async def test_crash_agents_downweighted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F1 crash agents (empty text, far embeddings) must not override the majority.

        Setup: 4 clean agents at [1,0] say 'yes'; 3 crash agents at [10,0] say ''.
        Geometric median sits at [1,0] (majority cluster).
        Crash agents are 9 units away → w ≈ 0.1; clean agents at dist=0 → w=1.0.
        Weighted vote: 'yes' wins overwhelmingly.
        """
        clean_emb = np.array([[1.0, 0.0]] * 4)
        crash_emb = np.array([[10.0, 0.0]] * 3)
        embeddings = np.vstack([clean_emb, crash_emb])
        monkeypatch.setattr("pipeline_v2.aggregation._embed", lambda _: embeddings)

        clean_gens = [_make_gen(f"c{i}", "Yes, that is correct.") for i in range(4)]
        crash_gens = [_make_gen(f"f{i}", "", is_faulty=True) for i in range(3)]
        gens = clean_gens + crash_gens

        text, is_low = await aggregate(gens)
        assert _extract_answer_key(text, is_numeric=False) == "yes"

    @pytest.mark.asyncio
    async def test_byzantine_minority_downweighted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Byzantine minority (wrong answer, outlier embeddings) must not win.

        Setup: 4 correct (yes) at [1,0]; 3 Byzantine (no) at [-1,0].
        Geometric median is at [1,0] (4 > 3).
        'yes' agents: dist=0 → w=1.0. 'no' agents: dist=2 → w≈0.1.
        'yes' total weight >> 'no' total weight.
        """
        correct_emb = np.array([[1.0, 0.0]] * 4)
        wrong_emb = np.array([[-1.0, 0.0]] * 3)
        embeddings = np.vstack([correct_emb, wrong_emb])
        monkeypatch.setattr("pipeline_v2.aggregation._embed", lambda _: embeddings)

        correct_gens = [_make_gen(f"c{i}", "Yes, the answer is yes.") for i in range(4)]
        wrong_gens = [_make_gen(f"f{i}", "No, the answer is no.", is_faulty=True) for i in range(3)]
        gens = correct_gens + wrong_gens

        text, is_low = await aggregate(gens)
        assert _extract_answer_key(text, is_numeric=False) == "yes"

    @pytest.mark.asyncio
    async def test_numeric_mode_weighted_vote(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GSM8K-style: 4 correct (42) at [1,0]; 3 wrong (99) at [-1,0]."""
        correct_emb = np.array([[1.0, 0.0]] * 4)
        wrong_emb = np.array([[-1.0, 0.0]] * 3)
        embeddings = np.vstack([correct_emb, wrong_emb])
        monkeypatch.setattr("pipeline_v2.aggregation._embed", lambda _: embeddings)

        correct_gens = [
            _make_gen(f"c{i}", f"After step 1: 10 + 32 = 42. The answer is 42.")
            for i in range(4)
        ]
        wrong_gens = [
            _make_gen(f"f{i}", f"After step 1: 50 + 49 = 99. The answer is 99.", is_faulty=True)
            for i in range(3)
        ]
        gens = correct_gens + wrong_gens

        text, is_low = await aggregate(gens)
        assert _extract_answer_key(text, is_numeric=True) == "42"

    @pytest.mark.asyncio
    async def test_low_confidence_when_no_answer_has_majority_weight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """3-way split: no answer gets >= 50% of total weight → is_low_confidence=True.

        5 agents equally spaced (equal weights): 2 say 'yes', 2 say 'no', 1 says ''.
        Best answer ('yes' or 'no') gets 2/5 = 0.40 of weight < 0.50.
        """
        embeddings = np.zeros((5, 2))  # all at origin → equal weights
        monkeypatch.setattr("pipeline_v2.aggregation._embed", lambda _: embeddings)

        gens = [
            _make_gen("a0", "Yes, it is."),
            _make_gen("a1", "Yes, I agree."),
            _make_gen("a2", "No, it is not."),
            _make_gen("a3", "No, I disagree."),
            _make_gen("a4", ""),  # empty → key=""
        ]
        _, is_low = await aggregate(gens)
        assert is_low is True

    @pytest.mark.asyncio
    async def test_high_confidence_when_clear_winner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """4 agree, 1 disagrees, all equal weights → winner gets 4/5 = 80% → not low."""
        embeddings = np.zeros((5, 2))
        monkeypatch.setattr("pipeline_v2.aggregation._embed", lambda _: embeddings)

        gens = [
            _make_gen("a0", "Yes, it is."),
            _make_gen("a1", "Yes, I agree."),
            _make_gen("a2", "Yes, correct."),
            _make_gen("a3", "Yes, for sure."),
            _make_gen("a4", "No, it is not."),
        ]
        _, is_low = await aggregate(gens)
        assert is_low is False

    @pytest.mark.asyncio
    async def test_weight_formula_proportional_to_proximity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that the nearer agent's answer wins against a farther one even 1-vs-1.

        2 agents: 'yes' at distance 0.5, 'no' at distance 2.0.
        mean_d = (0.5 + 2.0) / 2 = 1.25
        w(yes) = exp(-0.5/1.25) = exp(-0.4) ≈ 0.67
        w(no)  = exp(-2.0/1.25) = exp(-1.6) ≈ 0.20
        'yes' wins.
        """
        # Median at [0,0]. 'yes' at dist=0.5, 'no' at dist=2.0
        embeddings = np.array([[0.5, 0.0], [2.0, 0.0]])
        # _geometric_median of these two → anywhere between, L-BFGS-B returns [0.5,0]
        # or [1.25, 0]. Patch it explicitly so dist calc is deterministic.
        monkeypatch.setattr("pipeline_v2.aggregation._embed", lambda _: embeddings)
        monkeypatch.setattr(
            "pipeline_v2.aggregation._geometric_median",
            lambda _: np.array([0.0, 0.0]),
        )

        gens = [
            _make_gen("a0", "Yes, the answer is yes."),
            _make_gen("a1", "No, the answer is no."),
        ]
        text, _ = await aggregate(gens)
        assert _extract_answer_key(text, is_numeric=False) == "yes"
