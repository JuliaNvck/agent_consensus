#!/usr/bin/env python3
"""Experiment A: Multi-Provider Diversity — does a mixed-provider ensemble beat each
individual provider on its own?

Setup:
  - Input: cache_mixed.json produced by scripts/mix_caches.py (2-2-2-1 split from
    4 providers, no fault injection)
  - Conditions:
      single_llama / single_qwen / single_mistral / single_phi3 — each provider's
      agents passed through geometric median nearest-centroid in isolation
      multi_provider — all 7 mixed agents aggregated together
  - Module 1: per-provider τ calibration (25th-percentile TopKMass)
  - Module 2: geometric median nearest-centroid (pipeline_multi.aggregation)

CPU only — no vllm, no torch imports.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models import AgentGeneration
from pipeline_multi.aggregation import aggregate
from pipeline_multi.filter import calibrate_tau


_DEFAULT_SEED: int = 42


def _extract_answer(output_text: str, ground_truth: str) -> str:
    gt = ground_truth.strip().lower()
    if gt in {"yes", "no"}:
        m = re.search(r"\b(yes|no)\b", output_text.lower())
        return m.group(1) if m else output_text.strip()
    numbers = re.findall(r"\$?[\d,]+", output_text)
    if numbers:
        return numbers[-1].replace("$", "").replace(",", "")
    return output_text.strip()


def load_mixed_cache(filepath: str) -> List[Tuple[str, List[AgentGeneration]]]:
    """Load a mixed-provider cache JSON.

    Returns list of (ground_truth, List[AgentGeneration]) pairs.
    Preserves model_id and provider fields from each generation record.
    """
    with open(filepath) as fh:
        data = json.load(fh)
    result: List[Tuple[str, List[AgentGeneration]]] = []
    for q in data["questions"]:
        gens = [
            AgentGeneration(
                agent_id=g["agent_id"],
                output_text=g["output_text"],
                token_logprobs=g.get("token_logprobs") or [],
                is_faulty=g.get("is_faulty", False),
                fault_type=g.get("fault_type"),
                model_id=g.get("model_id"),
                provider=g.get("provider"),
            )
            for g in q["generations"]
        ]
        result.append((q["ground_truth"], gens))
    return result


def _providers_present(questions: List[Tuple[str, List[AgentGeneration]]]) -> List[str]:
    providers: set[str] = set()
    for _, gens in questions:
        for g in gens:
            if g.provider:
                providers.add(g.provider)
    return sorted(providers)


async def run_experiment_a(
    cache_filepath: str,
    output_filepath: str = "results/exp_multi_a_diversity.csv",
    n_questions: Optional[int] = None,
    seed: int = _DEFAULT_SEED,
) -> pd.DataFrame:
    """Provider diversity ablation: individual provider vs multi-provider consensus.

    Returns a DataFrame with columns:
        condition, accuracy, low_confidence_freq, n_admitted_mean
    """
    import random as _random

    questions = load_mixed_cache(cache_filepath)
    if n_questions is not None:
        questions = questions[:n_questions]

    # Shuffle and split: 20% dev for τ calibration, 80% eval
    rng = _random.Random(seed)
    rng.shuffle(questions)
    dev_n = max(0, int(len(questions) * 0.2))
    dev_qs = questions[:dev_n] if dev_n > 0 else questions
    eval_qs = questions[dev_n:] if dev_n < len(questions) else questions

    # Calibrate per-provider τ on dev set (flat list of all agents)
    all_dev_agents: List[AgentGeneration] = [g for _, gens in dev_qs for g in gens]
    tau_by_provider = calibrate_tau(all_dev_agents)
    providers = _providers_present(questions)

    print(f"Providers detected: {providers}")
    print(f"Per-provider τ: {tau_by_provider}")
    print(f"Calibrated on {len(dev_qs)} questions, evaluating on {len(eval_qs)}.")

    conditions = [f"single_{p}" for p in providers] + ["multi_provider"]
    accum: Dict[str, Dict[str, List]] = {
        c: {"correct": [], "low_conf": [], "n_admitted": []}
        for c in conditions
    }

    for q_idx, (ground_truth, gens) in enumerate(eval_qs):
        if q_idx % 20 == 0:
            print(f"  Question {q_idx + 1}/{len(eval_qs)}...")

        # Single-provider conditions: filter to only that provider's agents
        for provider in providers:
            provider_agents = [g for g in gens if g.provider == provider]
            if not provider_agents:
                continue
            result = aggregate(provider_agents, tau_by_provider)
            predicted = _extract_answer(result.final_answer, ground_truth)
            correct = predicted == ground_truth.strip().lower() if ground_truth.strip().lower() in {"yes", "no"} else predicted == ground_truth.strip()
            cond = f"single_{provider}"
            accum[cond]["correct"].append(correct)
            accum[cond]["low_conf"].append(result.is_low_confidence)
            accum[cond]["n_admitted"].append(len(result.admitted_agents))

        # Multi-provider: all agents together
        result = aggregate(gens, tau_by_provider)
        predicted = _extract_answer(result.final_answer, ground_truth)
        correct = predicted == ground_truth.strip().lower() if ground_truth.strip().lower() in {"yes", "no"} else predicted == ground_truth.strip()
        accum["multi_provider"]["correct"].append(correct)
        accum["multi_provider"]["low_conf"].append(result.is_low_confidence)
        accum["multi_provider"]["n_admitted"].append(len(result.admitted_agents))

    rows = []
    for cond in conditions:
        d = accum[cond]
        if not d["correct"]:
            continue
        rows.append({
            "condition": cond,
            "accuracy": float(np.mean(d["correct"])),
            "low_confidence_freq": float(np.mean(d["low_conf"])),
            "n_admitted_mean": float(np.mean(d["n_admitted"])),
        })

    df = pd.DataFrame(rows, columns=["condition", "accuracy", "low_confidence_freq", "n_admitted_mean"])

    print("\nResults:")
    for _, row in df.iterrows():
        print(f"  [{row['condition']:20s}]  acc={row['accuracy']:.1%}  low_conf={row['low_confidence_freq']:.1%}  n_admitted={row['n_admitted_mean']:.1f}")

    best_single = df[df["condition"] != "multi_provider"]["accuracy"].max()
    multi_acc = df[df["condition"] == "multi_provider"]["accuracy"].values[0]
    print(f"\n  Gap (multi - best_single): {multi_acc - best_single:+.1%}")

    out_dir = os.path.dirname(output_filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"Saved → {output_filepath}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment A: Multi-provider diversity benchmark."
    )
    parser.add_argument("--cache", required=True, help="Path to mixed cache JSON.")
    parser.add_argument(
        "--output", default="results/exp_multi_a_diversity.csv",
        help="Destination CSV.",
    )
    parser.add_argument("--n-questions", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_experiment_a(args.cache, args.output, args.n_questions))
