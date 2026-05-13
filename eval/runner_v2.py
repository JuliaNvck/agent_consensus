"""Experiment 1 runner — final system (strict extraction + stage1_only).

Two fixes vs earlier runners:
  1. baseline and hard_only use answer_majority_voting_strict: faulty agents whose
     output has no parseable answer (F1 crash, F2 wrong-format text, F3 off-topic)
     return None and are excluded from the vote instead of injecting garbage keys
     that previously won pluralities at high beta.
  2. full_system uses geometric median nearest-centroid (stage1_only) instead of
     the distance-weighted majority vote from pipeline_v2, which regressed under
     adversarial conditions (fragmented correct votes vs concentrated wrong cluster).
"""

from __future__ import annotations

import asyncio
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eval.answer_extraction import answer_majority_voting_strict
from eval.answer_extraction import extract_answer as _extract_answer
from eval.baselines import soft_weighted_geometric_median
from eval.io import load_cache
from faults.injector import inject_faults
from models import AgentGeneration
from pipeline.filter import _agent_stats, _compute_topk_mass_trajectory, filter_agents
from pipeline.stage1 import nearest_centroid_text


def calibrate_tau(
    questions: List[Tuple[str, List[AgentGeneration]]],
    percentile: float = 5.0,
) -> float:
    """Return the `percentile`-th percentile of stable-region mean TopKMass across all agents.

    Implements the design doc §3.2 calibration rule: τ = 5th percentile of clean-agent
    scores on a dev slice. Uses the post-warmup stable region of each trajectory
    (positions ≥ W=64) so the score is length-invariant across output lengths.
    Falls back to _DEFAULT_TAU if no agents have non-empty logprobs.
    """
    scores: List[float] = []
    for _, gens in questions:
        for gen in gens:
            if not gen.token_logprobs:
                continue
            traj = _compute_topk_mass_trajectory(gen.token_logprobs)
            if len(traj) > 0:
                mean_score, _ = _agent_stats(traj)
                scores.append(mean_score)
    if not scores:
        return _DEFAULT_TAU
    scores.sort()
    idx = max(0, min(int(len(scores) * percentile / 100.0), len(scores) - 1))
    return scores[idx]


_CONDITIONS: List[str] = ["baseline", "soft_weighting", "hard_only", "full_system"]
_N_VALUES: List[int] = [5, 7]
_BETA_VALUES: List[float] = [0.0, 0.15, 0.30, 0.45]
_FAULT_TYPES: List[str] = ["F1", "F2", "F3", "mix"]
_DEFAULT_TAU: float = 1.0
_DEFAULT_SEED: int = 42


async def _run_condition(
    agents: List[AgentGeneration],
    condition: str,
    tau: float,
    f: int,
    ground_truth: str,
) -> Tuple[str, int, bool]:
    """Run one ablation condition on a single question's agent pool.

    Returns (final_answer, n_admitted, is_low_confidence).
    baseline / soft_weighting: no filter, always admit all agents.
    hard_only / full_system: Module 1 filter with liveness fallback (BFT threshold 2f+1).

    baseline and hard_only use strict extraction: agents with no parseable answer
    format are excluded from the vote (None vote), preventing garbage-key plurality.
    full_system uses geometric median nearest-centroid (stage1_only).
    """
    n = len(agents)

    if condition == "baseline":
        return answer_majority_voting_strict(agents, ground_truth), n, False

    if condition == "soft_weighting":
        return soft_weighted_geometric_median(agents), n, False

    admitted = await filter_agents(agents, tau)
    if len(admitted) < 2 * f + 1:
        admitted = agents
        is_low = True
    else:
        is_low = False

    if condition == "hard_only":
        answer = answer_majority_voting_strict(admitted, ground_truth)
    else:  # full_system — geometric median nearest-centroid
        answer = nearest_centroid_text(admitted)
        is_low = is_low  # nearest-centroid always produces an answer

    return answer, len(admitted), is_low


async def run_experiment_1(
    cache_filepath: str,
    output_filepath: str = "results/experiment_1_v3.csv",
    tau: Optional[float] = None,
    seed: int = _DEFAULT_SEED,
    n_values: List[int] = _N_VALUES,
    beta_values: List[float] = _BETA_VALUES,
    fault_types: List[str] = _FAULT_TYPES,
    dev_fraction: float = 0.2,
    n_questions: Optional[int] = None,
) -> pd.DataFrame:
    """Ablation study over (N, beta, fault_type) × 4 pipeline conditions.

    Final Exp 1 runner: strict answer extraction for baseline/hard_only and
    geometric median nearest-centroid selection for full_system.

    Returns:
        DataFrame with columns: condition, n_agents, beta, fault_type,
        accuracy, admission_rate, fallback_frequency.
    """
    import random as _random

    all_questions = load_cache(cache_filepath)
    if n_questions is not None:
        all_questions = all_questions[:n_questions]
    _rng = _random.Random(seed)
    _rng.shuffle(all_questions)
    dev_n = max(0, int(len(all_questions) * dev_fraction))
    if tau is None:
        tau = calibrate_tau(all_questions[:dev_n] if dev_n > 0 else all_questions)
    questions = all_questions[dev_n:] if dev_n < len(all_questions) else all_questions
    print(
        f"  Calibrated τ={tau:.4f} on {dev_n} dev questions, "
        f"evaluating on {len(questions)} questions."
    )
    rows: List[Dict] = []

    for n in n_values:
        f = (n - 1) // 3
        for beta in beta_values:
            for fault_type in fault_types:
                accum: Dict[str, Dict[str, List]] = {
                    c: {"correct": [], "admission_rates": [], "fallbacks": []}
                    for c in _CONDITIONS
                }

                for ground_truth, clean_gens in questions:
                    pool = clean_gens[:n]
                    faulty = inject_faults(
                        pool, beta=beta, fault_type=fault_type, seed=seed
                    )

                    for condition in _CONDITIONS:
                        answer, n_admitted, is_low = await _run_condition(
                            faulty, condition, tau, f, ground_truth
                        )
                        accum[condition]["correct"].append(
                            _extract_answer(answer, ground_truth)
                            == ground_truth.strip()
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
            "condition",
            "n_agents",
            "beta",
            "fault_type",
            "accuracy",
            "admission_rate",
            "fallback_frequency",
        ],
    )

    out_dir = os.path.dirname(output_filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_filepath, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 1 (v2): ablation grid.")
    parser.add_argument("--cache", required=True, help="Path to generation cache JSON.")
    parser.add_argument("--output", required=True, help="Destination CSV.")
    parser.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help="Limit to the first N questions (for quick smoke tests).",
    )
    parser.add_argument(
        "--include-n1",
        action="store_true",
        help="Add N=1 (single-agent) to the evaluation grid.",
    )
    args = parser.parse_args()

    n_values: List[int] = ([1] if args.include_n1 else []) + list(_N_VALUES)
    asyncio.run(
        run_experiment_1(
            cache_filepath=args.cache,
            output_filepath=args.output,
            n_values=n_values,
            n_questions=args.n_questions,
        )
    )
