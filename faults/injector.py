from __future__ import annotations

import math
import random
from typing import List

from models import AgentGeneration

# F2 (Byzantine): peaked logprobs → 5×exp(-0.02) ≈ 4.90 per position → HIGH TopKMass
_F2_LOGPROB: float = -0.02

# F3 (Drifter): flat logprobs → 5×exp(-10.0) ≈ 2.3e-4 per position → LOW TopKMass
_F3_LOGPROB: float = -10.0

_F2_TEXT: str = "WRONG: The answer is definitively incorrect."
_F3_TEXT: str = (
    "Historically, the migration patterns of Arctic terns have fascinated "
    "ornithologists due to their remarkable navigational precision."
)

_DEFAULT_TOKEN_COUNT: int = 20
_VALID_FAULT_TYPES = frozenset({"F1", "F2", "F3", "mix"})


def inject_faults(
    generations: List[AgentGeneration],
    beta: float,
    fault_type: str,
    seed: int,
) -> List[AgentGeneration]:
    """Deterministically inject faults into a fraction beta of agents.

    Returns a new list; original AgentGeneration objects are never mutated.
    Clean agents are returned as-is (same object reference). Faulty agents are
    new objects with is_faulty=True and the appropriate fault_type label.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0.0, 1.0], got {beta}")
    if fault_type not in _VALID_FAULT_TYPES:
        raise ValueError(
            f"fault_type must be one of {sorted(_VALID_FAULT_TYPES)}, got {fault_type!r}"
        )

    n = len(generations)
    n_faults = math.floor(n * beta)
    rng = random.Random(seed)

    fault_indices = frozenset(rng.sample(range(n), n_faults)) if n_faults > 0 else frozenset()

    result: List[AgentGeneration] = []
    for i, gen in enumerate(generations):
        if i not in fault_indices:
            result.append(gen)
        else:
            ft = rng.choice(("F1", "F2", "F3")) if fault_type == "mix" else fault_type
            result.append(_apply_fault(gen, ft))
    return result


def _apply_fault(gen: AgentGeneration, fault_type: str) -> AgentGeneration:
    """Return a new AgentGeneration with the requested fault applied."""
    if fault_type == "F1":
        return AgentGeneration(
            agent_id=gen.agent_id,
            output_text="",
            token_logprobs=[],
            is_faulty=True,
            fault_type="F1_crash",
        )

    T = len(gen.token_logprobs) // 5 or _DEFAULT_TOKEN_COUNT

    if fault_type == "F2":
        return AgentGeneration(
            agent_id=gen.agent_id,
            output_text=_F2_TEXT,
            token_logprobs=[_F2_LOGPROB] * (T * 5),
            is_faulty=True,
            fault_type="F2_byzantine",
        )

    if fault_type == "F3":
        return AgentGeneration(
            agent_id=gen.agent_id,
            output_text=_F3_TEXT,
            token_logprobs=[_F3_LOGPROB] * (T * 5),
            is_faulty=True,
            fault_type="F3_drifter",
        )

    raise ValueError(f"Unknown fault type: {fault_type!r}")  # unreachable
