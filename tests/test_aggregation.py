"""
TDD tests for pipeline/aggregation.py — Module 2 Robust Semantic Aggregation.

Stage 1: geometric median of embeddings via scipy.optimize (Weiszfeld / L-BFGS-B).
Stage 2: bidirectional NLI entailment in a single batched tensor forward pass.

All model I/O is mocked — no weights are downloaded, no GPU required.
"""

from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from models import AgentGeneration
from pipeline.aggregation import (
    _batched_entailment,
    _geometric_median,
    aggregate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gen(agent_id: str, output_text: str) -> AgentGeneration:
    return AgentGeneration(
        agent_id=agent_id,
        output_text=output_text,
        token_logprobs=[-0.1, -1.0, -2.0, -3.0, -4.0] * 4,
        is_faulty=False,
        fault_type=None,
    )


def _nli_mock(
    logits: torch.Tensor,
) -> tuple[MagicMock, MagicMock]:
    """Return (tokenizer_mock, model_mock) wired up for _batched_entailment."""
    tok = MagicMock()
    tok.return_value = {
        "input_ids": torch.zeros(2, 4, dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
    }

    model = MagicMock()
    model.config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    model.return_value.logits = logits
    return tok, model


# ---------------------------------------------------------------------------
# Unit: _geometric_median
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
        # Geometric median x-coord must be much closer to 0 than the mean (~20)
        assert abs(med[0]) < abs(mean[0])
        assert abs(med[0]) < 1.0

    def test_single_point_is_fixed_point(self) -> None:
        """A single embedding is its own geometric median."""
        pts = np.array([[3.0, 7.0]])
        med = _geometric_median(pts)
        np.testing.assert_allclose(med, [3.0, 7.0], atol=1e-5)

    def test_two_identical_points_numerically_stable(self) -> None:
        """Two identical points must not cause division-by-zero."""
        pts = np.array([[1.0, 1.0], [1.0, 1.0]])
        med = _geometric_median(pts)
        np.testing.assert_allclose(med, [1.0, 1.0], atol=1e-5)


# ---------------------------------------------------------------------------
# Unit: _batched_entailment
# ---------------------------------------------------------------------------

class TestBatchedEntailment:
    def test_single_forward_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The NLI model must be called exactly once (batch_size=2, not twice)."""
        logits = torch.tensor([[0.0, 0.9, 0.1], [0.0, 0.9, 0.1]])
        tok, model = _nli_mock(logits)
        monkeypatch.setattr("pipeline.aggregation._get_nli_model", lambda: (tok, model))

        _batched_entailment("hello", "world")

        assert model.call_count == 1, "Expected exactly one forward pass"
        assert tok.call_count == 1, "Expected exactly one tokenizer call"
        # Tokenizer must receive exactly 2 pairs
        passed_pairs = tok.call_args[0][0]
        assert len(passed_pairs) == 2

    def test_pair_order_is_bidirectional(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pairs passed to tokenizer must be [A,B] and [B,A]."""
        logits = torch.tensor([[0.0, 0.9, 0.1], [0.0, 0.9, 0.1]])
        tok, model = _nli_mock(logits)
        monkeypatch.setattr("pipeline.aggregation._get_nli_model", lambda: (tok, model))

        _batched_entailment("AAA", "BBB")

        pairs = tok.call_args[0][0]
        assert pairs[0] == ["AAA", "BBB"]
        assert pairs[1] == ["BBB", "AAA"]

    def test_both_directions_entail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both argmax=1 (entailment) → (True, True)."""
        logits = torch.tensor([[0.0, 0.9, 0.1], [0.0, 0.9, 0.1]])
        tok, model = _nli_mock(logits)
        monkeypatch.setattr("pipeline.aggregation._get_nli_model", lambda: (tok, model))

        a_ok, b_ok = _batched_entailment("A", "B")
        assert a_ok is True
        assert b_ok is True

    def test_one_direction_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """First pair → contradiction (argmax=0), second → entailment (argmax=1)."""
        logits = torch.tensor([[0.8, 0.1, 0.1], [0.0, 0.9, 0.1]])
        tok, model = _nli_mock(logits)
        monkeypatch.setattr("pipeline.aggregation._get_nli_model", lambda: (tok, model))

        a_ok, b_ok = _batched_entailment("A", "B")
        assert a_ok is False
        assert b_ok is True

    def test_neither_direction_entails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both pairs → contradiction (argmax=0) → (False, False)."""
        logits = torch.tensor([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])
        tok, model = _nli_mock(logits)
        monkeypatch.setattr("pipeline.aggregation._get_nli_model", lambda: (tok, model))

        a_ok, b_ok = _batched_entailment("A", "B")
        assert a_ok is False
        assert b_ok is False


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
    async def test_single_agent_bypasses_both_stages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With one agent, embedding and entailment should never be called."""
        embed_mock = MagicMock()
        entail_mock = MagicMock()
        monkeypatch.setattr("pipeline.aggregation._embed", embed_mock)
        monkeypatch.setattr("pipeline.aggregation._batched_entailment", entail_mock)

        gen = _make_gen("a0", "only answer")
        text, is_low = await aggregate([gen])

        assert text == "only answer"
        assert is_low is False
        embed_mock.assert_not_called()
        entail_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_nearest_centroid_candidate_selected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """3 agents: two near origin, one far outlier.

        Embeddings: agent_0→[0,0], agent_1→[0.1,0], agent_2→[10,0].
        Geometric median of 3 collinear points ≈ [0.1,0].
        Nearest to [0.1,0] is agent_1 → result must be "text_b".
        """
        embeddings = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]])
        monkeypatch.setattr("pipeline.aggregation._embed", lambda _: embeddings)
        monkeypatch.setattr(
            "pipeline.aggregation._batched_entailment", lambda a, b: (True, True)
        )

        gens = [_make_gen("a0", "text_a"), _make_gen("a1", "text_b"), _make_gen("a2", "text_c")]
        text, is_low = await aggregate(gens)
        assert text == "text_b"
        assert is_low is False

    @pytest.mark.asyncio
    async def test_nli_failure_nearest_selects_next_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nearest candidate fails NLI → next candidate in distance order is selected.

        Embeddings: agent_0→[0,0], agent_1→[0.1,0], agent_2→[10,0].
        Median ≈ [0.1,0]; nearest=text_b (dist=0), reference=text_a (dist=0.1).
        First call (text_b vs text_a) fails → second call (text_a vs text_a) passes.
        """
        embeddings = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]])
        monkeypatch.setattr("pipeline.aggregation._embed", lambda _: embeddings)

        calls: list = []

        def _fail_then_pass(a: str, b: str) -> tuple[bool, bool]:
            calls.append((a, b))
            return (False, False) if len(calls) == 1 else (True, True)

        monkeypatch.setattr("pipeline.aggregation._batched_entailment", _fail_then_pass)

        gens = [_make_gen("a0", "text_a"), _make_gen("a1", "text_b"), _make_gen("a2", "text_c")]
        text, is_low = await aggregate(gens)
        assert text == "text_a"  # second nearest selected after nearest failed NLI
        assert is_low is False
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_all_nli_fail_returns_nearest_with_low_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All candidates fail NLI → nearest returned with is_low_confidence=True."""
        embeddings = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]])
        monkeypatch.setattr("pipeline.aggregation._embed", lambda _: embeddings)
        monkeypatch.setattr(
            "pipeline.aggregation._batched_entailment", lambda a, b: (False, False)
        )

        gens = [_make_gen("a0", "text_a"), _make_gen("a1", "text_b"), _make_gen("a2", "text_c")]
        text, is_low = await aggregate(gens)
        assert text == "text_b"  # nearest (dist=0) returned as fallback
        assert is_low is True

    @pytest.mark.asyncio
    async def test_two_agents_entailment_uses_both_texts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With 2 agents, entailment must be called with the two agent texts."""
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0]])
        monkeypatch.setattr("pipeline.aggregation._embed", lambda _: embeddings)

        captured: list[tuple[str, str]] = []

        def _capture(a: str, b: str) -> tuple[bool, bool]:
            captured.append((a, b))
            return (True, True)

        monkeypatch.setattr("pipeline.aggregation._batched_entailment", _capture)

        gens = [_make_gen("a0", "alpha"), _make_gen("a1", "beta")]
        text, is_low = await aggregate(gens)

        assert isinstance(text, str)
        assert is_low is False
        assert len(captured) == 1
        # Both agent texts must appear in the entailment call
        seen_texts = set(captured[0])
        assert "alpha" in seen_texts
        assert "beta" in seen_texts

    @pytest.mark.asyncio
    async def test_return_type_is_tuple_str_bool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0]])
        monkeypatch.setattr("pipeline.aggregation._embed", lambda _: embeddings)
        monkeypatch.setattr(
            "pipeline.aggregation._batched_entailment", lambda a, b: (True, True)
        )
        gens = [_make_gen("a0", "x"), _make_gen("a1", "y")]
        result = await aggregate(gens)
        assert isinstance(result, tuple)
        assert isinstance(result[0], str)
        assert isinstance(result[1], bool)
