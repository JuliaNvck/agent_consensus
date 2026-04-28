"""
TDD tests for eval/baselines.py and eval/runner.py — Evaluation Harness.

Baselines:
  majority_voting              — most frequent output_text.
  soft_weighted_geometric_median — TopKMass-weighted Weiszfeld median → nearest text.

Runner:
  load_cache     — JSON cache → List[(ground_truth, List[AgentGeneration])].
  run_experiment_1 — iterates (N, beta, fault_type) × 4 conditions → DataFrame + CSV.

All heavy model calls (_embed, _batched_entailment) are suppressed by the autouse fixture
in conftest.py. Per-test control uses monkeypatch.setattr on pipeline.aggregation._embed.
"""

from __future__ import annotations

import json
from typing import List

import numpy as np
import pandas as pd
import pytest

from models import AgentGeneration
from eval.baselines import majority_voting, soft_weighted_geometric_median
from eval.runner import load_cache, run_experiment_1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gen(
    agent_id: str,
    output_text: str,
    logprobs: List[float] | None = None,
    is_faulty: bool = False,
    fault_type: str | None = None,
) -> AgentGeneration:
    if logprobs is None:
        logprobs = [-0.5, -1.0, -1.5, -2.0, -2.5] * 4  # T=4, clean-ish weight
    return AgentGeneration(
        agent_id=agent_id,
        output_text=output_text,
        token_logprobs=logprobs,
        is_faulty=is_faulty,
        fault_type=fault_type,
    )


def _make_cache_dict(
    ground_truth: str = "correct answer",
    n_agents: int = 7,
    output_text: str | None = None,
) -> dict:
    """Single-question cache with `n_agents` clean agents all outputting `output_text`."""
    text = output_text if output_text is not None else ground_truth
    return {
        "questions": [
            {
                "question_id": "q0",
                "ground_truth": ground_truth,
                "generations": [
                    {
                        "agent_id": f"a{i}",
                        "output_text": text,
                        "token_logprobs": [-0.5, -1.0, -1.5, -2.0, -2.5] * 4,
                        "is_faulty": False,
                        "fault_type": None,
                    }
                    for i in range(n_agents)
                ],
            }
        ]
    }


def _write_cache(tmp_path, data: dict) -> str:
    p = tmp_path / "cache.json"
    p.write_text(json.dumps(data))
    return str(p)


# ---------------------------------------------------------------------------
# Class 1: majority_voting
# ---------------------------------------------------------------------------

class TestMajorityVoting:
    def test_empty_returns_empty_string(self) -> None:
        assert majority_voting([]) == ""

    def test_single_agent(self) -> None:
        gen = _make_gen("a0", "the answer")
        assert majority_voting([gen]) == "the answer"

    def test_clear_majority(self) -> None:
        gens = [
            _make_gen("a0", "correct"),
            _make_gen("a1", "correct"),
            _make_gen("a2", "wrong"),
        ]
        assert majority_voting(gens) == "correct"

    def test_all_same_text(self) -> None:
        gens = [_make_gen(f"a{i}", "unanimous") for i in range(5)]
        assert majority_voting(gens) == "unanimous"

    def test_tie_returns_a_string(self) -> None:
        gens = [
            _make_gen("a0", "alpha"),
            _make_gen("a1", "alpha"),
            _make_gen("a2", "beta"),
            _make_gen("a3", "beta"),
        ]
        result = majority_voting(gens)
        assert isinstance(result, str)
        assert result in ("alpha", "beta")

    def test_supermajority_five_agents(self) -> None:
        gens = [
            _make_gen("a0", "right"),
            _make_gen("a1", "right"),
            _make_gen("a2", "right"),
            _make_gen("a3", "wrong"),
            _make_gen("a4", "other"),
        ]
        assert majority_voting(gens) == "right"


# ---------------------------------------------------------------------------
# Class 2: soft_weighted_geometric_median
# ---------------------------------------------------------------------------

class TestSoftWeightedGeometricMedian:
    def test_empty_returns_empty_string(self) -> None:
        assert soft_weighted_geometric_median([]) == ""

    def test_single_agent_returns_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Single agent: embedding should never be called."""
        called = []
        monkeypatch.setattr(
            "pipeline.aggregation._embed",
            lambda texts: called.append(texts) or np.zeros((len(texts), 2)),
        )
        gen = _make_gen("a0", "solo answer")
        result = soft_weighted_geometric_median([gen])
        assert result == "solo answer"
        assert len(called) == 0, "_embed must not be called for a single agent"

    def test_high_weight_cluster_selected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two agents near origin (high TopKMass weight) + one far outlier (low weight).

        Weighted median should be pulled toward the origin cluster.
        logprob=-0.02  → TopKMass ≈ 4.90 (high weight)
        logprob=-10.0  → TopKMass ≈ 2.3e-4 (low weight)
        """
        embeddings = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]])
        monkeypatch.setattr("pipeline.aggregation._embed", lambda _: embeddings)

        high_lp = [-0.02] * (4 * 5)   # high weight
        low_lp = [-10.0] * (4 * 5)    # low weight
        gens = [
            _make_gen("a0", "text_a", high_lp),
            _make_gen("a1", "text_b", high_lp),
            _make_gen("a2", "text_c", low_lp),
        ]
        result = soft_weighted_geometric_median(gens)
        assert result in ("text_a", "text_b"), (
            f"Expected cluster text (text_a or text_b), got {result!r}"
        )

    def test_zero_weight_fallback_to_uniform(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All agents with empty logprobs (F1-like) get weight 0 → uniform fallback, no crash."""
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0]])
        monkeypatch.setattr("pipeline.aggregation._embed", lambda _: embeddings)

        gens = [
            _make_gen("a0", "text_a", []),  # weight = 0
            _make_gen("a1", "text_b", []),  # weight = 0
        ]
        result = soft_weighted_geometric_median(gens)
        assert isinstance(result, str)
        assert result in ("text_a", "text_b")

    def test_return_type_is_str(self, monkeypatch: pytest.MonkeyPatch) -> None:
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0]])
        monkeypatch.setattr("pipeline.aggregation._embed", lambda _: embeddings)

        gens = [_make_gen("a0", "x"), _make_gen("a1", "y")]
        result = soft_weighted_geometric_median(gens)
        assert isinstance(result, str)

    def test_weighted_median_resists_outlier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """4 agents near origin (high weight) + 1 far outlier (low weight).

        Result must come from the near-origin cluster.
        """
        embeddings = np.array([
            [0.0, 0.0], [0.05, 0.0], [-0.05, 0.0], [0.0, 0.05],
            [50.0, 0.0],
        ])
        monkeypatch.setattr("pipeline.aggregation._embed", lambda _: embeddings)

        high_lp = [-0.02] * 20
        low_lp = [-10.0] * 20
        gens = [
            _make_gen("a0", "near_0", high_lp),
            _make_gen("a1", "near_1", high_lp),
            _make_gen("a2", "near_2", high_lp),
            _make_gen("a3", "near_3", high_lp),
            _make_gen("a4", "far",    low_lp),
        ]
        result = soft_weighted_geometric_median(gens)
        assert result.startswith("near_"), f"Expected near-origin text, got {result!r}"


# ---------------------------------------------------------------------------
# Class 3: load_cache
# ---------------------------------------------------------------------------

class TestLoadCache:
    def test_returns_list(self, tmp_path: pytest.TempPathFactory) -> None:
        path = _write_cache(tmp_path, _make_cache_dict())
        result = load_cache(path)
        assert isinstance(result, list)

    def test_single_question_length(self, tmp_path: pytest.TempPathFactory) -> None:
        path = _write_cache(tmp_path, _make_cache_dict())
        result = load_cache(path)
        assert len(result) == 1

    def test_ground_truth_field(self, tmp_path: pytest.TempPathFactory) -> None:
        path = _write_cache(tmp_path, _make_cache_dict(ground_truth="my answer"))
        ground_truth, _ = load_cache(path)[0]
        assert ground_truth == "my answer"

    def test_generations_are_agent_generation_objects(self, tmp_path: pytest.TempPathFactory) -> None:
        path = _write_cache(tmp_path, _make_cache_dict(n_agents=3))
        _, generations = load_cache(path)[0]
        assert all(isinstance(g, AgentGeneration) for g in generations)
        assert len(generations) == 3

    def test_agent_generation_fields(self, tmp_path: pytest.TempPathFactory) -> None:
        path = _write_cache(tmp_path, _make_cache_dict(n_agents=1))
        _, generations = load_cache(path)[0]
        g = generations[0]
        assert g.agent_id == "a0"
        assert isinstance(g.output_text, str)
        assert isinstance(g.token_logprobs, list)
        assert g.is_faulty is False

    def test_fault_type_null_parsed_as_none(self, tmp_path: pytest.TempPathFactory) -> None:
        path = _write_cache(tmp_path, _make_cache_dict(n_agents=1))
        _, generations = load_cache(path)[0]
        assert generations[0].fault_type is None

    def test_multiple_questions(self, tmp_path: pytest.TempPathFactory) -> None:
        data = {
            "questions": [
                {
                    "question_id": f"q{i}",
                    "ground_truth": f"answer_{i}",
                    "generations": [
                        {
                            "agent_id": "a0",
                            "output_text": f"answer_{i}",
                            "token_logprobs": [-0.5] * 5,
                            "is_faulty": False,
                            "fault_type": None,
                        }
                    ],
                }
                for i in range(3)
            ]
        }
        path = _write_cache(tmp_path, data)
        result = load_cache(path)
        assert len(result) == 3
        assert result[0][0] == "answer_0"
        assert result[2][0] == "answer_2"


# ---------------------------------------------------------------------------
# Class 4: run_experiment_1
# ---------------------------------------------------------------------------

class TestRunExperiment1:
    """All tests use minimal grids and mock _embed / _batched_entailment.

    conftest.py autouse already stubs pipeline.aggregation._embed → zeros((N,2))
    and pipeline.aggregation._batched_entailment → (True, True).
    Per-test monkeypatches override as needed.
    """

    @pytest.mark.asyncio
    async def test_returns_dataframe(self, tmp_path: pytest.TempPathFactory) -> None:
        cache = _write_cache(tmp_path, _make_cache_dict())
        out = str(tmp_path / "results" / "out.csv")

        df = await run_experiment_1(
            cache, out, n_values=[5], beta_values=[0.0], fault_types=["F1"]
        )
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_dataframe_columns(self, tmp_path: pytest.TempPathFactory) -> None:
        cache = _write_cache(tmp_path, _make_cache_dict())
        out = str(tmp_path / "results" / "out.csv")

        df = await run_experiment_1(
            cache, out, n_values=[5], beta_values=[0.0], fault_types=["F1"]
        )
        expected = {
            "condition", "n_agents", "beta", "fault_type",
            "accuracy", "admission_rate", "fallback_frequency",
        }
        assert set(df.columns) == expected

    @pytest.mark.asyncio
    async def test_csv_written(self, tmp_path: pytest.TempPathFactory) -> None:
        import os
        cache = _write_cache(tmp_path, _make_cache_dict())
        out = str(tmp_path / "results" / "out.csv")

        await run_experiment_1(
            cache, out, n_values=[5], beta_values=[0.0], fault_types=["F1"]
        )
        assert os.path.exists(out)

    @pytest.mark.asyncio
    async def test_row_count_matches_grid(self, tmp_path: pytest.TempPathFactory) -> None:
        """4 conditions × 2 N × 2 beta × 2 fault_types = 32 rows."""
        cache = _write_cache(tmp_path, _make_cache_dict())
        out = str(tmp_path / "results" / "out.csv")

        df = await run_experiment_1(
            cache, out,
            n_values=[5, 7],
            beta_values=[0.0, 0.30],
            fault_types=["F1", "F2"],
        )
        assert len(df) == 4 * 2 * 2 * 2  # 32

    @pytest.mark.asyncio
    async def test_baseline_perfect_accuracy_clean_agents(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """beta=0, no faults: all agents output the ground truth → baseline accuracy=1.0."""
        cache = _write_cache(tmp_path, _make_cache_dict(ground_truth="correct", n_agents=7))
        out = str(tmp_path / "results" / "out.csv")

        df = await run_experiment_1(
            cache, out, n_values=[5], beta_values=[0.0], fault_types=["F1"]
        )
        baseline_row = df[df["condition"] == "baseline"]
        assert len(baseline_row) == 1
        assert float(baseline_row["accuracy"].iloc[0]) == 1.0

    @pytest.mark.asyncio
    async def test_admission_rate_in_range(self, tmp_path: pytest.TempPathFactory) -> None:
        cache = _write_cache(tmp_path, _make_cache_dict())
        out = str(tmp_path / "results" / "out.csv")

        df = await run_experiment_1(
            cache, out, n_values=[5], beta_values=[0.0, 0.45], fault_types=["F2"]
        )
        assert (df["admission_rate"] >= 0.0).all()
        assert (df["admission_rate"] <= 1.0).all()

    @pytest.mark.asyncio
    async def test_fallback_frequency_in_range(self, tmp_path: pytest.TempPathFactory) -> None:
        cache = _write_cache(tmp_path, _make_cache_dict())
        out = str(tmp_path / "results" / "out.csv")

        df = await run_experiment_1(
            cache, out, n_values=[5], beta_values=[0.0, 0.45], fault_types=["F1"]
        )
        assert (df["fallback_frequency"] >= 0.0).all()
        assert (df["fallback_frequency"] <= 1.0).all()

    @pytest.mark.asyncio
    async def test_f1_full_beta_triggers_fallback_for_filtered_conditions(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """beta=1.0, fault_type=F1: all agents crash → empty admitted pool → liveness fallback.

        hard_only and full_system must show fallback_frequency=1.0.
        baseline and soft_weighting never fall back (no filter).
        """
        cache = _write_cache(tmp_path, _make_cache_dict(n_agents=7))
        out = str(tmp_path / "results" / "out.csv")

        df = await run_experiment_1(
            cache, out, n_values=[5], beta_values=[1.0], fault_types=["F1"]
        )
        for cond in ("hard_only", "full_system"):
            row = df[df["condition"] == cond]
            assert float(row["fallback_frequency"].iloc[0]) == 1.0, (
                f"Expected fallback_frequency=1.0 for {cond}"
            )

        for cond in ("baseline", "soft_weighting"):
            row = df[df["condition"] == cond]
            assert float(row["fallback_frequency"].iloc[0]) == 0.0, (
                f"Expected fallback_frequency=0.0 for {cond}"
            )

    @pytest.mark.asyncio
    async def test_baseline_admission_rate_always_one(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """baseline and soft_weighting admit all agents (no filter) → admission_rate=1.0."""
        cache = _write_cache(tmp_path, _make_cache_dict())
        out = str(tmp_path / "results" / "out.csv")

        df = await run_experiment_1(
            cache, out, n_values=[5], beta_values=[0.45], fault_types=["F3"]
        )
        for cond in ("baseline", "soft_weighting"):
            row = df[df["condition"] == cond]
            assert float(row["admission_rate"].iloc[0]) == 1.0

    @pytest.mark.asyncio
    async def test_conditions_all_present(self, tmp_path: pytest.TempPathFactory) -> None:
        """All four condition labels must appear in the output."""
        cache = _write_cache(tmp_path, _make_cache_dict())
        out = str(tmp_path / "results" / "out.csv")

        df = await run_experiment_1(
            cache, out, n_values=[5], beta_values=[0.0], fault_types=["F1"]
        )
        assert set(df["condition"]) == {"baseline", "soft_weighting", "hard_only", "full_system"}
