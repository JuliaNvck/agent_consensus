from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from models import AgentGeneration
from pipeline.filter import filter_agents
from pipeline.aggregation import aggregate
from faults.injector import inject_faults
from eval.baselines import majority_voting, soft_weighted_geometric_median

_CONDITIONS: List[str] = ["baseline", "soft_weighting", "hard_only", "full_system"]
_N_VALUES: List[int] = [5, 7]
_BETA_VALUES: List[float] = [0.0, 0.15, 0.30, 0.45]
_FAULT_TYPES: List[str] = ["F1", "F2", "F3", "mix"]
_DEFAULT_TAU: float = 1.0
_DEFAULT_SEED: int = 42


def load_cache(filepath: str) -> List[Tuple[str, List[AgentGeneration]]]:
    """Load a JSON generation cache.

    Returns a list of (ground_truth, List[AgentGeneration]) pairs, one per question.
    """
    with open(filepath) as fh:
        data = json.load(fh)

    result: List[Tuple[str, List[AgentGeneration]]] = []
    for q in data["questions"]:
        ground_truth: str = q["ground_truth"]
        generations = [
            AgentGeneration(
                agent_id=g["agent_id"],
                output_text=g["output_text"],
                token_logprobs=g["token_logprobs"],
                is_faulty=g["is_faulty"],
                fault_type=g.get("fault_type"),
            )
            for g in q["generations"]
        ]
        result.append((ground_truth, generations))
    return result


async def _run_condition(
    agents: List[AgentGeneration],
    condition: str,
    tau: float,
    f: int,
) -> Tuple[str, int, bool]:
    """Run one ablation condition on a single question's agent pool.

    Returns (final_answer, n_admitted, is_low_confidence).
    baseline / soft_weighting: no filter, always admit all agents.
    hard_only / full_system: Module 1 filter with liveness fallback (BFT threshold 2f+1).
    """
    n = len(agents)

    if condition == "baseline":
        return majority_voting(agents), n, False

    if condition == "soft_weighting":
        return soft_weighted_geometric_median(agents), n, False

    # Module 1 filter + liveness fallback (same logic as Orchestrator._liveness_check)
    admitted = await filter_agents(agents, tau)
    if len(admitted) < 2 * f + 1:
        admitted = agents
        is_low = True
    else:
        is_low = False

    if condition == "hard_only":
        answer = majority_voting(admitted)
    else:  # full_system
        answer = await aggregate(admitted)

    return answer, len(admitted), is_low


async def run_experiment_1(
    cache_filepath: str,
    output_filepath: str = "results/experiment_1.csv",
    tau: float = _DEFAULT_TAU,
    seed: int = _DEFAULT_SEED,
    n_values: List[int] = _N_VALUES,
    beta_values: List[float] = _BETA_VALUES,
    fault_types: List[str] = _FAULT_TYPES,
) -> pd.DataFrame:
    """Ablation study over (N, beta, fault_type) × 4 pipeline conditions.

    For each combination, runs all 4 conditions on every cached question,
    computes accuracy / admission_rate / fallback_frequency, and writes a CSV.

    Args:
        cache_filepath: Path to the Phase 1 JSON generation cache.
        output_filepath: Destination for the results CSV.
        tau: Module 1 reliability filter threshold.
        seed: RNG seed forwarded to inject_faults for determinism.
        n_values: Pool sizes to evaluate (subset of cached agents per question).
        beta_values: Fault fractions to evaluate.
        fault_types: Fault type labels to evaluate.

    Returns:
        DataFrame with columns: condition, n_agents, beta, fault_type,
        accuracy, admission_rate, fallback_frequency.
    """
    questions = load_cache(cache_filepath)
    rows: List[Dict] = []

    for n in n_values:
        f = (n - 1) // 3  # BFT: N = 3f+1 → f = (N-1)//3
        for beta in beta_values:
            for fault_type in fault_types:
                # Per-condition accumulators over all questions
                accum: Dict[str, Dict[str, List]] = {
                    c: {"correct": [], "admission_rates": [], "fallbacks": []}
                    for c in _CONDITIONS
                }

                for ground_truth, clean_gens in questions:
                    pool = clean_gens[:n]
                    faulty = inject_faults(pool, beta=beta, fault_type=fault_type, seed=seed)

                    for condition in _CONDITIONS:
                        answer, n_admitted, is_low = await _run_condition(
                            faulty, condition, tau, f
                        )
                        accum[condition]["correct"].append(
                            answer.strip() == ground_truth.strip()
                        )
                        accum[condition]["admission_rates"].append(
                            n_admitted / n if n > 0 else 0.0
                        )
                        accum[condition]["fallbacks"].append(is_low)

                for condition in _CONDITIONS:
                    d = accum[condition]
                    rows.append(
                        {
                            "condition": condition,
                            "n_agents": n,
                            "beta": beta,
                            "fault_type": fault_type,
                            "accuracy": float(np.mean(d["correct"])),
                            "admission_rate": float(np.mean(d["admission_rates"])),
                            "fallback_frequency": float(np.mean(d["fallbacks"])),
                        }
                    )

    df = pd.DataFrame(
        rows,
        columns=[
            "condition", "n_agents", "beta", "fault_type",
            "accuracy", "admission_rate", "fallback_frequency",
        ],
    )

    out_dir = os.path.dirname(output_filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_filepath, index=False)
    return df
