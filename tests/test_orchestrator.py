import asyncio
from typing import List

import pytest

from coordination.orchestrator import Orchestrator
from models import AgentGeneration, ConsensusResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gen(
    agent_id: str,
    fault_type: str | None = None,
    output_text: str = "answer",
) -> AgentGeneration:
    return AgentGeneration(
        agent_id=agent_id,
        output_text=output_text,
        token_logprobs=[-0.1, -1.0, -2.0, -3.0, -4.0] * 4,  # 20 entries = 4 token positions
        is_faulty=fault_type is not None,
        fault_type=fault_type,
    )


def _healthy_pool(n: int) -> List[AgentGeneration]:
    return [_make_gen(f"agent_{i}") for i in range(n)]


# ---------------------------------------------------------------------------
# Liveness check unit tests (synchronous, no I/O)
# ---------------------------------------------------------------------------

class TestLivenessCheck:
    def test_sufficient_admitted_returns_false(self) -> None:
        orch = Orchestrator(f=2)
        admitted = _healthy_pool(5)   # == 2*2+1
        available = _healthy_pool(7)
        pool, low_conf = orch._liveness_check(admitted, available)
        assert not low_conf
        assert pool is admitted

    def test_insufficient_admitted_triggers_fallback(self) -> None:
        orch = Orchestrator(f=2)
        admitted = _healthy_pool(4)   # < 2*2+1=5
        available = _healthy_pool(7)
        pool, low_conf = orch._liveness_check(admitted, available)
        assert low_conf
        assert pool is available

    def test_zero_admitted_triggers_fallback(self) -> None:
        orch = Orchestrator(f=1)
        pool, low_conf = orch._liveness_check([], _healthy_pool(3))
        assert low_conf

    def test_f0_threshold_is_one(self) -> None:
        orch = Orchestrator(f=0)
        admitted = _healthy_pool(1)
        pool, low_conf = orch._liveness_check(admitted, admitted)
        assert not low_conf


# ---------------------------------------------------------------------------
# End-to-end async integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_normal_round_no_fallback() -> None:
    """5 healthy agents + 1 crash + 1 byzantine with f=2 → no fallback."""
    generations = [
        *_healthy_pool(5),
        _make_gen("crash_0", fault_type="F1_crash"),
        _make_gen("byzantine_0", fault_type="F2_byzantine", output_text="wrong answer"),
    ]
    orch = Orchestrator(f=2, agent_timeout=0.6)
    result: ConsensusResult = await orch.run(generations)

    assert not result.is_low_confidence
    assert "crash_0" not in result.admitted_agents
    assert len(result.admitted_agents) >= 2 * 2 + 1


@pytest.mark.asyncio
async def test_fallback_triggered_when_too_few_respond() -> None:
    """Only 2 healthy agents survive (f=2 → threshold 5) → fallback fires."""
    generations = [
        *[_make_gen(f"agent_{i}") for i in range(2)],
        *[_make_gen(f"crash_{i}", fault_type="F1_crash") for i in range(5)],
    ]
    orch = Orchestrator(f=2, agent_timeout=0.6)
    result: ConsensusResult = await orch.run(generations)

    assert result.is_low_confidence
    # Admitted pool must be all responding (non-crashed) agents
    assert set(result.admitted_agents) == {"agent_0", "agent_1"}


@pytest.mark.asyncio
async def test_all_crash_returns_empty_answer() -> None:
    generations = [_make_gen(f"crash_{i}", fault_type="F1_crash") for i in range(4)]
    orch = Orchestrator(f=1, agent_timeout=0.3)
    result: ConsensusResult = await orch.run(generations)

    assert result.is_low_confidence
    assert result.final_answer == ""
    assert result.admitted_agents == []


@pytest.mark.asyncio
async def test_result_type_contract() -> None:
    orch = Orchestrator(f=1, agent_timeout=0.6)
    result = await orch.run(_healthy_pool(4))

    assert isinstance(result, ConsensusResult)
    assert isinstance(result.final_answer, str)
    assert isinstance(result.admitted_agents, list)
    assert isinstance(result.is_low_confidence, bool)
