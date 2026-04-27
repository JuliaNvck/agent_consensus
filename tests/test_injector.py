"""
TDD tests for faults/injector.py — deterministic fault injection layer.

Fault types:
  F1 (Crash)     — empty text + empty logprobs; always dropped by Module 1.
  F2 (Byzantine) — adversarial text + peaked logprobs; intentionally PASSES Module 1.
  F3 (Drifter)   — off-task text + flat logprobs; intentionally FAILS Module 1.

Spoofing math (verified via pipeline.filter._compute_topk_mass_trajectory):
  F2: logprob = -0.02 → 5×exp(-0.02) ≈ 4.90 per position  → mean TopKMass ≈ 4.90
  F3: logprob = -10.0 → 5×exp(-10)   ≈ 2.3e-4 per position → mean TopKMass ≈ 2.3e-4
"""

import math
from typing import List

import pytest

from models import AgentGeneration
from faults.injector import inject_faults
from pipeline.filter import _compute_topk_mass_trajectory, filter_agents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clean_gens(n: int, T: int = 20) -> List[AgentGeneration]:
    """Clean agents with T token positions (len(logprobs) == 5*T)."""
    return [
        AgentGeneration(
            agent_id=f"a{i}",
            output_text=f"clean answer {i}",
            token_logprobs=[-0.5, -1.0, -1.5, -2.0, -2.5] * T,
            is_faulty=False,
            fault_type=None,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Structure & determinism
# ---------------------------------------------------------------------------

class TestInjectFaultsStructure:
    def test_exact_fault_count_30pct(self) -> None:
        result = inject_faults(_make_clean_gens(10), beta=0.30, fault_type="F1", seed=42)
        assert sum(g.is_faulty for g in result) == math.floor(10 * 0.30)  # 3

    def test_exact_fault_count_15pct(self) -> None:
        result = inject_faults(_make_clean_gens(10), beta=0.15, fault_type="F1", seed=42)
        assert sum(g.is_faulty for g in result) == math.floor(10 * 0.15)  # 1

    def test_exact_fault_count_rounding(self) -> None:
        # floor(7 * 0.45) = floor(3.15) = 3
        result = inject_faults(_make_clean_gens(7), beta=0.45, fault_type="F1", seed=42)
        assert sum(g.is_faulty for g in result) == math.floor(7 * 0.45)

    def test_exact_fault_count_45pct_10agents(self) -> None:
        # floor(10 * 0.45) = 4
        result = inject_faults(_make_clean_gens(10), beta=0.45, fault_type="F2", seed=42)
        assert sum(g.is_faulty for g in result) == math.floor(10 * 0.45)

    def test_zero_beta_no_faults(self) -> None:
        result = inject_faults(_make_clean_gens(10), beta=0.0, fault_type="F1", seed=42)
        assert all(not g.is_faulty for g in result)

    def test_full_beta_all_faults(self) -> None:
        result = inject_faults(_make_clean_gens(5), beta=1.0, fault_type="F1", seed=42)
        assert all(g.is_faulty for g in result)

    def test_output_list_length_preserved(self) -> None:
        gens = _make_clean_gens(10)
        result = inject_faults(gens, beta=0.30, fault_type="F2", seed=42)
        assert len(result) == len(gens)

    def test_agent_ids_preserved_in_order(self) -> None:
        gens = _make_clean_gens(10)
        result = inject_faults(gens, beta=0.30, fault_type="F2", seed=42)
        assert [g.agent_id for g in result] == [g.agent_id for g in gens]

    def test_deterministic_same_seed_same_ids(self) -> None:
        gens = _make_clean_gens(20)
        r1 = inject_faults(gens, beta=0.30, fault_type="F1", seed=42)
        r2 = inject_faults(gens, beta=0.30, fault_type="F1", seed=42)
        ids1 = frozenset(g.agent_id for g in r1 if g.is_faulty)
        ids2 = frozenset(g.agent_id for g in r2 if g.is_faulty)
        assert ids1 == ids2

    def test_different_seeds_produce_different_ids(self) -> None:
        # N=20, beta=0.50 → 10 faults; P(collision) ≈ 1/C(20,10) < 0.001%
        gens = _make_clean_gens(20)
        r1 = inject_faults(gens, beta=0.50, fault_type="F1", seed=42)
        r2 = inject_faults(gens, beta=0.50, fault_type="F1", seed=99)
        ids1 = frozenset(g.agent_id for g in r1 if g.is_faulty)
        ids2 = frozenset(g.agent_id for g in r2 if g.is_faulty)
        assert ids1 != ids2

    def test_originals_not_mutated(self) -> None:
        gens = _make_clean_gens(10)
        orig_texts = [g.output_text for g in gens]
        inject_faults(gens, beta=0.50, fault_type="F2", seed=42)
        assert [g.output_text for g in gens] == orig_texts

    def test_invalid_beta_raises(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            inject_faults(_make_clean_gens(5), beta=1.5, fault_type="F1", seed=0)

    def test_negative_beta_raises(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            inject_faults(_make_clean_gens(5), beta=-0.1, fault_type="F1", seed=0)

    def test_invalid_fault_type_raises(self) -> None:
        with pytest.raises(ValueError, match="fault_type"):
            inject_faults(_make_clean_gens(5), beta=0.2, fault_type="X", seed=0)


# ---------------------------------------------------------------------------
# F1 — Crash fault
# ---------------------------------------------------------------------------

class TestF1Fault:
    def test_f1_empty_output_text(self) -> None:
        result = inject_faults(_make_clean_gens(5), beta=1.0, fault_type="F1", seed=0)
        assert all(g.output_text == "" for g in result)

    def test_f1_empty_logprobs(self) -> None:
        result = inject_faults(_make_clean_gens(5), beta=1.0, fault_type="F1", seed=0)
        assert all(g.token_logprobs == [] for g in result)

    def test_f1_fault_type_label(self) -> None:
        result = inject_faults(_make_clean_gens(5), beta=1.0, fault_type="F1", seed=0)
        assert all(g.fault_type == "F1_crash" for g in result)

    def test_f1_is_faulty_flag(self) -> None:
        result = inject_faults(_make_clean_gens(5), beta=1.0, fault_type="F1", seed=0)
        assert all(g.is_faulty for g in result)

    @pytest.mark.asyncio
    async def test_f1_always_dropped_by_filter(self) -> None:
        """F1 agents have empty logprobs and are unconditionally dropped by Module 1."""
        result = inject_faults(_make_clean_gens(5), beta=1.0, fault_type="F1", seed=0)
        admitted = await filter_agents(result, tau=0.0)
        assert len(admitted) == 0


# ---------------------------------------------------------------------------
# F2 — Byzantine fault (must PASS Module 1)
# ---------------------------------------------------------------------------

class TestF2Fault:
    def test_f2_output_text_replaced(self) -> None:
        gens = _make_clean_gens(3)
        result = inject_faults(gens, beta=1.0, fault_type="F2", seed=0)
        clean_texts = {g.output_text for g in gens}
        assert all(g.output_text not in clean_texts for g in result)

    def test_f2_output_text_nonempty(self) -> None:
        result = inject_faults(_make_clean_gens(3), beta=1.0, fault_type="F2", seed=0)
        assert all(g.output_text != "" for g in result)

    def test_f2_fault_type_label(self) -> None:
        result = inject_faults(_make_clean_gens(3), beta=1.0, fault_type="F2", seed=0)
        assert all(g.fault_type == "F2_byzantine" for g in result)

    def test_f2_is_faulty_flag(self) -> None:
        result = inject_faults(_make_clean_gens(3), beta=1.0, fault_type="F2", seed=0)
        assert all(g.is_faulty for g in result)

    def test_f2_logprobs_length_multiple_of_5(self) -> None:
        result = inject_faults(_make_clean_gens(3), beta=1.0, fault_type="F2", seed=0)
        assert all(len(g.token_logprobs) % 5 == 0 for g in result)

    def test_f2_logprobs_produce_high_topk_mass(self) -> None:
        """F2 mean TopKMass must exceed 2.0 — well clear of any realistic tau."""
        result = inject_faults(_make_clean_gens(3), beta=1.0, fault_type="F2", seed=0)
        for gen in result:
            traj = _compute_topk_mass_trajectory(gen.token_logprobs)
            assert float(traj.mean()) > 2.0

    @pytest.mark.asyncio
    async def test_f2_passes_module1_filter(self) -> None:
        """F2 agents must pass filter_agents at tau=1.0."""
        result = inject_faults(_make_clean_gens(5), beta=1.0, fault_type="F2", seed=0)
        admitted = await filter_agents(result, tau=1.0)
        assert len(admitted) == len(result)


# ---------------------------------------------------------------------------
# F3 — Drifter fault (must FAIL Module 1)
# ---------------------------------------------------------------------------

class TestF3Fault:
    def test_f3_output_text_replaced(self) -> None:
        gens = _make_clean_gens(3)
        result = inject_faults(gens, beta=1.0, fault_type="F3", seed=0)
        clean_texts = {g.output_text for g in gens}
        assert all(g.output_text not in clean_texts for g in result)

    def test_f3_output_text_nonempty(self) -> None:
        result = inject_faults(_make_clean_gens(3), beta=1.0, fault_type="F3", seed=0)
        assert all(g.output_text != "" for g in result)

    def test_f3_fault_type_label(self) -> None:
        result = inject_faults(_make_clean_gens(3), beta=1.0, fault_type="F3", seed=0)
        assert all(g.fault_type == "F3_drifter" for g in result)

    def test_f3_is_faulty_flag(self) -> None:
        result = inject_faults(_make_clean_gens(3), beta=1.0, fault_type="F3", seed=0)
        assert all(g.is_faulty for g in result)

    def test_f3_logprobs_length_multiple_of_5(self) -> None:
        result = inject_faults(_make_clean_gens(3), beta=1.0, fault_type="F3", seed=0)
        assert all(len(g.token_logprobs) % 5 == 0 for g in result)

    def test_f3_logprobs_produce_low_topk_mass(self) -> None:
        """F3 mean TopKMass must be < 0.01 — well below any realistic tau."""
        result = inject_faults(_make_clean_gens(3), beta=1.0, fault_type="F3", seed=0)
        for gen in result:
            traj = _compute_topk_mass_trajectory(gen.token_logprobs)
            assert float(traj.mean()) < 0.01

    @pytest.mark.asyncio
    async def test_f3_fails_module1_filter(self) -> None:
        """F3 agents must fail filter_agents at tau=0.01 (10× above F3 score)."""
        result = inject_faults(_make_clean_gens(5), beta=1.0, fault_type="F3", seed=0)
        admitted = await filter_agents(result, tau=0.01)
        assert len(admitted) == 0


# ---------------------------------------------------------------------------
# Mixed fault type
# ---------------------------------------------------------------------------

class TestMixFault:
    def test_mix_exact_fault_count(self) -> None:
        result = inject_faults(_make_clean_gens(20), beta=0.30, fault_type="mix", seed=42)
        assert sum(g.is_faulty for g in result) == math.floor(20 * 0.30)

    def test_mix_all_fault_types_are_valid(self) -> None:
        result = inject_faults(_make_clean_gens(20), beta=0.50, fault_type="mix", seed=42)
        valid = {"F1_crash", "F2_byzantine", "F3_drifter"}
        for gen in result:
            if gen.is_faulty:
                assert gen.fault_type in valid

    def test_mix_deterministic_seeding(self) -> None:
        gens = _make_clean_gens(20)
        r1 = inject_faults(gens, beta=0.30, fault_type="mix", seed=42)
        r2 = inject_faults(gens, beta=0.30, fault_type="mix", seed=42)
        # Same agent IDs and same fault_type assignments
        pairs1 = [(g.agent_id, g.fault_type) for g in r1]
        pairs2 = [(g.agent_id, g.fault_type) for g in r2]
        assert pairs1 == pairs2
