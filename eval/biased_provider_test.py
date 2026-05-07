#!/usr/bin/env python3
"""Experiment B: Biased Provider Stress Test.

Tests whether geometric median nearest-centroid aggregation absorbs a
systematically wrong provider, compared to extracted-answer majority vote.

Setup:
  - Input: cache_mixed.json (2-2-2-1 split from 4 providers, no fault injection)
  - Default biased provider: mistral (2 agents, observed 50% accuracy vs ~67% for others)
  - Secondary biased provider: phi3 (1 agent, 60% accuracy — run with --biased-provider phi3)
  - Bias types:
      natural  — use biased-provider answers as-is (real systematic error)
      injected — replace all biased-provider agents with F1 crash faults
  - Conditions compared:
      no_{provider}        — remove biased-provider agents before aggregation (oracle)
      majority_vote_all    — extracted-answer plurality vote over all 7 agents
      geometric_median_all — our system: filter + geometric median nearest-centroid

CPU only — no vllm, no torch imports.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eval.baselines import answer_majority_voting
from eval.runner_multi import load_mixed_cache, _extract_answer
from models import AgentGeneration
from pipeline_multi.aggregation import aggregate
from pipeline_multi.filter import calibrate_tau


_BIASED_PROVIDER: str = "mistral"
_DEFAULT_SEED: int = 42

_BIAS_TYPES = ["natural", "injected"]


def _inject_bias(agents: List[AgentGeneration], provider: str) -> List[AgentGeneration]:
    """Replace all agents from `provider` with F1 crash faults (empty output, no logprobs)."""
    result: List[AgentGeneration] = []
    for g in agents:
        if g.provider == provider:
            result.append(AgentGeneration(
                agent_id=g.agent_id,
                output_text="",
                token_logprobs=[],
                is_faulty=True,
                fault_type="F1_crash",
                model_id=g.model_id,
                provider=g.provider,
            ))
        else:
            result.append(g)
    return result


async def run_experiment_b(
    cache_filepath: str,
    output_filepath: str = "results/exp_multi_b_biased_provider.csv",
    biased_provider: str = _BIASED_PROVIDER,
    n_questions: Optional[int] = None,
    seed: int = _DEFAULT_SEED,
) -> pd.DataFrame:
    """Biased provider stress test.

    Returns a DataFrame with columns:
        biased_provider, condition, bias_type, accuracy, fallback_freq
    """
    import random as _random

    questions = load_mixed_cache(cache_filepath)
    if n_questions is not None:
        questions = questions[:n_questions]

    rng = _random.Random(seed)
    rng.shuffle(questions)
    dev_n = max(0, int(len(questions) * 0.2))
    dev_qs = questions[:dev_n] if dev_n > 0 else questions
    eval_qs = questions[dev_n:] if dev_n < len(questions) else questions

    all_dev_agents: List[AgentGeneration] = [g for _, gens in dev_qs for g in gens]
    tau_by_provider = calibrate_tau(all_dev_agents)

    oracle_cond = f"no_{biased_provider}"
    conditions = [oracle_cond, "majority_vote_all", "geometric_median_all"]

    print(f"Biased provider: {biased_provider!r}")
    print(f"Per-provider τ: {tau_by_provider}")
    print(f"Calibrated on {len(dev_qs)} questions, evaluating on {len(eval_qs)}.")

    accum: Dict[Tuple[str, str], Dict[str, List]] = {
        (cond, bt): {"correct": [], "fallback": []}
        for cond in conditions
        for bt in _BIAS_TYPES
    }

    for q_idx, (ground_truth, gens) in enumerate(eval_qs):
        if q_idx % 20 == 0:
            print(f"  Question {q_idx + 1}/{len(eval_qs)}...")

        gt_clean = ground_truth.strip().lower()
        is_bool = gt_clean in {"yes", "no"}

        for bias_type in _BIAS_TYPES:
            pool = gens if bias_type == "natural" else _inject_bias(gens, biased_provider)

            # Oracle: remove biased provider entirely
            pool_no_biased = [g for g in pool if g.provider != biased_provider]
            if pool_no_biased:
                result = aggregate(pool_no_biased, tau_by_provider)
                pred = _extract_answer(result.final_answer, ground_truth)
                correct = (pred == gt_clean) if is_bool else (pred == ground_truth.strip())
                accum[(oracle_cond, bias_type)]["correct"].append(correct)
                accum[(oracle_cond, bias_type)]["fallback"].append(result.is_low_confidence)

            # Extracted-answer majority vote over all agents (fair apples-to-apples comparison)
            mv_answer = answer_majority_voting(pool, ground_truth)
            pred = _extract_answer(mv_answer, ground_truth)
            correct = (pred == gt_clean) if is_bool else (pred == ground_truth.strip())
            accum[("majority_vote_all", bias_type)]["correct"].append(correct)
            accum[("majority_vote_all", bias_type)]["fallback"].append(False)

            # Geometric median over all agents (our system)
            result = aggregate(pool, tau_by_provider)
            pred = _extract_answer(result.final_answer, ground_truth)
            correct = (pred == gt_clean) if is_bool else (pred == ground_truth.strip())
            accum[("geometric_median_all", bias_type)]["correct"].append(correct)
            accum[("geometric_median_all", bias_type)]["fallback"].append(result.is_low_confidence)

    rows = []
    for bias_type in _BIAS_TYPES:
        for cond in conditions:
            d = accum[(cond, bias_type)]
            if not d["correct"]:
                continue
            rows.append({
                "biased_provider": biased_provider,
                "condition": cond,
                "bias_type": bias_type,
                "accuracy": float(np.mean(d["correct"])),
                "fallback_freq": float(np.mean(d["fallback"])),
            })

    df = pd.DataFrame(rows, columns=["biased_provider", "condition", "bias_type", "accuracy", "fallback_freq"])

    print("\nResults:")
    for _, row in df.iterrows():
        print(
            f"  [{row['bias_type']:9s}] [{row['condition']:22s}]  "
            f"acc={row['accuracy']:.1%}  fallback={row['fallback_freq']:.1%}"
        )

    out_dir = os.path.dirname(output_filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"Saved → {output_filepath}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment B: Biased provider stress test."
    )
    parser.add_argument("--cache", required=True, help="Path to mixed cache JSON.")
    parser.add_argument(
        "--output", default="results/exp_multi_b_biased_provider.csv",
        help="Destination CSV.",
    )
    parser.add_argument(
        "--biased-provider", default=_BIASED_PROVIDER,
        help=f"Provider to designate as biased (default: {_BIASED_PROVIDER!r}).",
    )
    parser.add_argument("--n-questions", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_experiment_b(args.cache, args.output, args.biased_provider, args.n_questions))
