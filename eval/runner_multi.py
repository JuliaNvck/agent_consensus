#!/usr/bin/env python3
"""Experiment A: Multi-Provider Diversity — does a mixed-provider ensemble beat each
individual provider on its own?

Setup:
  - Input: cache_mixed.json (multi-provider) + individual provider caches (7 agents each)
  - Conditions:
      single_{provider} — all 7 agents from that provider's individual cache
      multi_provider    — all 7 mixed agents from cache_mixed.json
  - Module 1: per-provider τ calibration (25th-percentile TopKMass) on mixed cache dev set
  - Module 2: geometric median nearest-centroid (pipeline_multi.aggregation)

Individual caches give 7 agents per question per provider — a fair baseline.
The previous approach of slicing 2 agents from the mixed pool caused 20-26% Module 1
fallback rates because losing even 1 of 2 agents hits the liveness threshold.

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


def load_provider_cache(filepath: str, provider: str) -> Dict[str, Tuple[str, List[AgentGeneration]]]:
    """Load an individual provider cache, keyed by question_id.

    Retrofits provider tag into agents that lack one (legacy cache format).
    Returns {question_id: (ground_truth, List[AgentGeneration])}.
    """
    with open(filepath) as fh:
        data = json.load(fh)
    result: Dict[str, Tuple[str, List[AgentGeneration]]] = {}
    for q in data["questions"]:
        gens = [
            AgentGeneration(
                agent_id=g["agent_id"],
                output_text=g["output_text"],
                token_logprobs=g.get("token_logprobs") or [],
                is_faulty=g.get("is_faulty", False),
                fault_type=g.get("fault_type"),
                model_id=g.get("model_id"),
                provider=g.get("provider") or provider,
            )
            for g in q["generations"]
        ]
        result[q["question_id"]] = (q["ground_truth"], gens)
    return result


def _parse_provider_cache_spec(spec: str) -> Tuple[str, str]:
    """Parse 'path/to/cache.json:provider_tag' → (path, tag)."""
    parts = spec.rsplit(":", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"--provider-caches entry must be 'path:provider_tag', got: {spec!r}")
    return parts[0], parts[1]


def _providers_present(questions: List[Tuple[str, List[AgentGeneration]]]) -> List[str]:
    providers: set[str] = set()
    for _, gens in questions:
        for g in gens:
            if g.provider:
                providers.add(g.provider)
    return sorted(providers)


async def run_experiment_a(
    cache_filepath: str,
    provider_cache_specs: Optional[List[str]] = None,
    output_filepath: str = "results/exp_multi_a_diversity.csv",
    n_questions: Optional[int] = None,
    seed: int = _DEFAULT_SEED,
) -> pd.DataFrame:
    """Provider diversity ablation: individual provider (7 agents) vs multi-provider consensus.

    For single-provider baselines, loads the full 7-agent individual cache so that
    Module 1 has enough agents to filter without hitting the liveness fallback.

    Args:
        cache_filepath: Path to cache_mixed.json (used for multi_provider condition
                        and for per-provider τ calibration).
        provider_cache_specs: List of 'path:provider' strings for individual caches,
                              e.g. ['cache_llma.json:llama', 'cache_qwen.json:qwen'].
                              If omitted, single-provider conditions are skipped.

    Returns a DataFrame with columns:
        condition, accuracy, low_confidence_freq, n_admitted_mean
    """
    import random as _random

    questions = load_mixed_cache(cache_filepath)
    if n_questions is not None:
        questions = questions[:n_questions]

    # Build question_id → index map for aligning individual caches
    mixed_ids = [None] * len(questions)  # we need question_ids from the JSON
    with open(cache_filepath) as fh:
        import json as _json
        raw = _json.load(fh)
    mixed_id_list = [q["question_id"] for q in raw["questions"]]
    if n_questions is not None:
        mixed_id_list = mixed_id_list[:n_questions]

    # Shuffle and split: 20% dev for τ calibration, 80% eval
    rng = _random.Random(seed)
    indexed = list(zip(mixed_id_list, questions))
    rng.shuffle(indexed)
    dev_n = max(0, int(len(indexed) * 0.2))
    dev_indexed = indexed[:dev_n] if dev_n > 0 else indexed
    eval_indexed = indexed[dev_n:] if dev_n < len(indexed) else indexed

    # Calibrate per-provider τ on mixed-cache dev set
    all_dev_agents: List[AgentGeneration] = [g for _, (_, gens) in dev_indexed for g in gens]
    tau_by_provider = calibrate_tau(all_dev_agents)

    # Load individual provider caches
    provider_caches: Dict[str, Dict[str, Tuple[str, List[AgentGeneration]]]] = {}
    if provider_cache_specs:
        for spec in provider_cache_specs:
            path, provider = _parse_provider_cache_spec(spec)
            provider_caches[provider] = load_provider_cache(path, provider)

    providers_with_cache = sorted(provider_caches.keys())
    providers_in_mixed = _providers_present(questions)

    print(f"Providers in mixed cache: {providers_in_mixed}")
    print(f"Individual caches loaded: {providers_with_cache}")
    print(f"Per-provider τ: {tau_by_provider}")
    print(f"Calibrated on {len(dev_indexed)} questions, evaluating on {len(eval_indexed)}.")

    conditions = [f"single_{p}" for p in providers_with_cache] + ["multi_provider"]
    accum: Dict[str, Dict[str, List]] = {
        c: {"correct": [], "low_conf": [], "n_admitted": []}
        for c in conditions
    }

    for q_idx, (qid, (ground_truth, gens)) in enumerate(eval_indexed):
        if q_idx % 20 == 0:
            print(f"  Question {q_idx + 1}/{len(eval_indexed)}...")

        gt_clean = ground_truth.strip().lower()
        is_bool = gt_clean in {"yes", "no"}

        # Single-provider baselines: full 7-agent pool from individual cache
        for provider, pcache in provider_caches.items():
            if qid not in pcache:
                continue
            _, provider_gens = pcache[qid]
            result = aggregate(provider_gens, tau_by_provider)
            pred = _extract_answer(result.final_answer, ground_truth)
            correct = (pred == gt_clean) if is_bool else (pred == ground_truth.strip())
            cond = f"single_{provider}"
            accum[cond]["correct"].append(correct)
            accum[cond]["low_conf"].append(result.is_low_confidence)
            accum[cond]["n_admitted"].append(len(result.admitted_agents))

        # Multi-provider: all 7 mixed agents
        result = aggregate(gens, tau_by_provider)
        pred = _extract_answer(result.final_answer, ground_truth)
        correct = (pred == gt_clean) if is_bool else (pred == ground_truth.strip())
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
        print(
            f"  [{row['condition']:20s}]  acc={row['accuracy']:.1%}"
            f"  low_conf={row['low_confidence_freq']:.1%}  n_admitted={row['n_admitted_mean']:.1f}"
        )

    if len(df) > 1:
        singles = df[df["condition"] != "multi_provider"]
        if not singles.empty:
            best_single = singles["accuracy"].max()
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
        "--provider-caches", nargs="+", metavar="PATH:PROVIDER",
        help="Individual provider caches for single-provider baselines (7 agents each). "
             "Format: 'cache_llma.json:llama cache_qwen.json:qwen ...'",
    )
    parser.add_argument(
        "--output", default="results/exp_multi_a_diversity.csv",
        help="Destination CSV.",
    )
    parser.add_argument("--n-questions", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_experiment_a(
        args.cache,
        provider_cache_specs=args.provider_caches,
        output_filepath=args.output,
        n_questions=args.n_questions,
    ))
