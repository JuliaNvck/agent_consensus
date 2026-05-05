from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models import AgentGeneration
from pipeline.filter import filter_agents, _compute_topk_mass_trajectory
from pipeline.aggregation import aggregate
from faults.injector import inject_faults
from eval.baselines import answer_majority_voting, majority_voting, soft_weighted_geometric_median

def _extract_answer(output_text: str, ground_truth: str) -> str:
    """Extract a comparable answer token from a chain-of-thought output.

    StrategyQA (GT ∈ {"yes","no"}): returns first yes/no found (case-insensitive).
    GSM8K (GT is a number string): returns the last number-like token, stripping $ and commas.
    Fallback: returns output_text.strip() unchanged (preserves exact-match for synthetic GTs).
    """
    gt = ground_truth.strip().lower()
    if gt in {"yes", "no"}:
        m = re.search(r"\b(yes|no)\b", output_text.lower())
        return m.group(1) if m else output_text.strip()
    numbers = re.findall(r"\$?[\d,]+", output_text)
    if numbers:
        return numbers[-1].replace("$", "").replace(",", "")
    return output_text.strip()


def calibrate_tau(
    questions: List[Tuple[str, List[AgentGeneration]]],
    percentile: float = 10.0,
) -> float:
    """Return the `percentile`-th percentile of mean TopKMass across all agents.

    Implements the design doc §3.2 calibration rule: τ = 10th percentile of clean-agent
    scores on a dev slice. Call this on the uninjected cache before running experiments.
    Falls back to _DEFAULT_TAU if no agents have non-empty logprobs.
    """
    scores: List[float] = []
    for _, gens in questions:
        for gen in gens:
            if not gen.token_logprobs:
                continue
            traj = _compute_topk_mass_trajectory(gen.token_logprobs)
            if len(traj) > 0:
                scores.append(float(traj.mean()))
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
    ground_truth: str,
) -> Tuple[str, int, bool]:
    """Run one ablation condition on a single question's agent pool.

    Returns (final_answer, n_admitted, is_low_confidence).
    baseline / soft_weighting: no filter, always admit all agents.
    hard_only / full_system: Module 1 filter with liveness fallback (BFT threshold 2f+1).
    """
    n = len(agents)

    if condition == "baseline":
        return answer_majority_voting(agents, ground_truth), n, False

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
        answer = answer_majority_voting(admitted, ground_truth)
    else:  # full_system
        answer, nli_low = await aggregate(admitted)
        is_low = is_low or nli_low

    return answer, len(admitted), is_low


async def run_experiment_1(
    cache_filepath: str,
    output_filepath: str = "results/experiment_1.csv",
    tau: Optional[float] = None,
    seed: int = _DEFAULT_SEED,
    n_values: List[int] = _N_VALUES,
    beta_values: List[float] = _BETA_VALUES,
    fault_types: List[str] = _FAULT_TYPES,
    dev_fraction: float = 0.1,
    n_questions: Optional[int] = None,
) -> pd.DataFrame:
    """Ablation study over (N, beta, fault_type) × 4 pipeline conditions.

    For each combination, runs all 4 conditions on every cached question,
    computes accuracy / admission_rate / fallback_frequency, and writes a CSV.

    Args:
        cache_filepath: Path to the Phase 1 JSON generation cache.
        output_filepath: Destination for the results CSV.
        tau: Module 1 reliability filter threshold (auto-calibrated on dev slice if None).
        seed: RNG seed forwarded to inject_faults for determinism.
        n_values: Pool sizes to evaluate (subset of cached agents per question).
        beta_values: Fault fractions to evaluate.
        fault_types: Fault type labels to evaluate.
        dev_fraction: Fraction of questions used for τ calibration (default 0.1).

    Returns:
        DataFrame with columns: condition, n_agents, beta, fault_type,
        accuracy, admission_rate, fallback_frequency.
    """
    all_questions = load_cache(cache_filepath)
    if n_questions is not None:
        all_questions = all_questions[:n_questions]
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
                            faulty, condition, tau, f, ground_truth
                        )
                        accum[condition]["correct"].append(
                            _extract_answer(answer, ground_truth) == ground_truth.strip()
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 1: ablation grid.")
    parser.add_argument("--cache", required=True, help="Path to generation cache JSON.")
    parser.add_argument(
        "--output", required=True, help="Destination CSV (e.g. results/experiment_1_llama_answer_vote.csv)."
    )
    parser.add_argument(
        "--n-questions", type=int, default=None,
        help="Limit to the first N questions (for quick smoke tests).",
    )
    parser.add_argument(
        "--include-n1", action="store_true",
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
