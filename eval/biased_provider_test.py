#!/usr/bin/env python3
"""Experiment B: Biased Provider Stress Test.

Tests whether geometric median nearest-centroid aggregation absorbs a
systematically wrong provider, compared to naive majority vote.

Setup:
  - Input: cache_mixed.json (2-2-2-1 split from 4 providers, no fault injection)
  - Biased provider: phi3 (smallest model, naturally weakest — 1 agent per question)
  - Bias types:
      natural  — use phi3 answers as-is (real systematic error)
      injected — replace all phi3 agents with F1 crash faults (zero logprobs, empty output)
  - Conditions compared:
      no_phi3              — remove phi3 agents before aggregation (oracle upper bound)
      majority_vote_all    — naive plurality vote on output_text (all 7 agents)
      geometric_median_all — our system: filter + geometric median nearest-centroid (all 7 agents)

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

from eval.runner_multi import load_mixed_cache, _extract_answer
from models import AgentGeneration
from pipeline_multi.aggregation import aggregate
from pipeline_multi.filter import calibrate_tau


_BIASED_PROVIDER: str = "phi3"
_DEFAULT_SEED: int = 42

_CONDITIONS = ["no_phi3", "majority_vote_all", "geometric_median_all"]
_BIAS_TYPES = ["natural", "injected"]


def _majority_vote_raw(agents: List[AgentGeneration]) -> str:
    """Plurality vote on raw output_text."""
    if not agents:
        return ""
    from collections import Counter
    return Counter(g.output_text.strip() for g in agents).most_common(1)[0][0]


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
        condition, bias_type, accuracy, fallback_freq
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

    print(f"Biased provider: {biased_provider!r}")
    print(f"Per-provider τ: {tau_by_provider}")
    print(f"Calibrated on {len(dev_qs)} questions, evaluating on {len(eval_qs)}.")

    accum: Dict[Tuple[str, str], Dict[str, List]] = {
        (cond, bt): {"correct": [], "fallback": []}
        for cond in _CONDITIONS
        for bt in _BIAS_TYPES
    }

    for q_idx, (ground_truth, gens) in enumerate(eval_qs):
        if q_idx % 20 == 0:
            print(f"  Question {q_idx + 1}/{len(eval_qs)}...")

        gt_clean = ground_truth.strip().lower()
        is_bool = gt_clean in {"yes", "no"}

        for bias_type in _BIAS_TYPES:
            if bias_type == "natural":
                pool = gens
            else:  # injected
                pool = _inject_bias(gens, biased_provider)

            # Condition 1: no_phi3 (oracle)
            pool_no_biased = [g for g in pool if g.provider != biased_provider]
            if pool_no_biased:
                result = aggregate(pool_no_biased, tau_by_provider)
                pred = _extract_answer(result.final_answer, ground_truth)
                correct = (pred == gt_clean) if is_bool else (pred == ground_truth.strip())
                accum[("no_phi3", bias_type)]["correct"].append(correct)
                accum[("no_phi3", bias_type)]["fallback"].append(result.is_low_confidence)

            # Condition 2: majority_vote_all
            raw_answer = _majority_vote_raw(pool)
            pred = _extract_answer(raw_answer, ground_truth)
            correct = (pred == gt_clean) if is_bool else (pred == ground_truth.strip())
            accum[("majority_vote_all", bias_type)]["correct"].append(correct)
            accum[("majority_vote_all", bias_type)]["fallback"].append(False)

            # Condition 3: geometric_median_all
            result = aggregate(pool, tau_by_provider)
            pred = _extract_answer(result.final_answer, ground_truth)
            correct = (pred == gt_clean) if is_bool else (pred == ground_truth.strip())
            accum[("geometric_median_all", bias_type)]["correct"].append(correct)
            accum[("geometric_median_all", bias_type)]["fallback"].append(result.is_low_confidence)

    rows = []
    for bias_type in _BIAS_TYPES:
        for cond in _CONDITIONS:
            d = accum[(cond, bias_type)]
            if not d["correct"]:
                continue
            rows.append({
                "condition": cond,
                "bias_type": bias_type,
                "accuracy": float(np.mean(d["correct"])),
                "fallback_freq": float(np.mean(d["fallback"])),
            })

    df = pd.DataFrame(rows, columns=["condition", "bias_type", "accuracy", "fallback_freq"])

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
