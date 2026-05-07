#!/usr/bin/env python3
"""Experiment 3 (v2): Adversarial Coordination Stress Test — System V2.

Identical to eval/adversarial_test.py except:
  - full_pipeline uses pipeline_v2.aggregation.aggregate (distance-weighted
    majority vote) instead of NLI-based selection.
  - stage1_only continues to use nearest-centroid selection, so the comparison
    between stage1_only and full_pipeline now measures the added value of
    weighted voting over simple nearest-centroid.
  - Column "nli_fallback_frequency" renamed to "low_confidence_frequency" since
    low-confidence is now signalled by the weighted-vote threshold, not NLI.
  - Condition label updated: "Full Pipeline (Weighted Vote)" replaces
    "Full Pipeline (Stage1 + NLI)".

Fixed parameters: N=7 agents, f=2 Byzantine, β≈28.6%.
CPU only — no vllm, no torch imports.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval.baselines import answer_majority_voting, majority_voting
from eval.runner_v2 import _extract_answer, calibrate_tau, load_cache
from faults.injector import _F2_TEXT, _F2_LOGPROBS_PER_TOKEN, _F3_TEXT
from models import AgentGeneration
from pipeline_v2.aggregation import _embed, _geometric_median, aggregate  # ← v2
from pipeline.filter import filter_agents

_N: int = 7
_F: int = 2
_DEFAULT_TOKEN_COUNT: int = 20

_COORD_LOGPROBS_PER_TOKEN: List[float] = [
    math.log(0.95), math.log(0.02), math.log(0.015),
    math.log(0.010), math.log(0.005),
]
_MAX_ADV_LOGPROBS_PER_TOKEN: List[float] = [
    math.log(0.95), math.log(0.02), math.log(0.015),
    math.log(0.010), math.log(0.005),
]

_COORDINATION_DEGREES = ["uncoordinated", "coordinated", "maximally_adversarial"]
_PIPELINE_CONDITIONS = ["majority_answer_vote", "stage1_only", "full_pipeline"]

_COORD_LABELS = {
    "uncoordinated": "Uncoordinated",
    "coordinated": "Coordinated",
    "maximally_adversarial": "Max. Adversarial",
}
_COND_LABELS = {
    "majority_answer_vote": "Answer Majority Vote",
    "stage1_only": "Stage 1 Only (Nearest-Centroid)",
    "full_pipeline": "Full Pipeline (Weighted Vote)",  # ← updated: no NLI
}
_COND_COLORS = {
    "majority_answer_vote": "#d62728",
    "stage1_only": "#1f77b4",
    "full_pipeline": "#2ca02c",
}


def _derive_wrong_answer(ground_truth: str) -> str:
    gt = ground_truth.strip().lower()
    if gt == "yes":
        return "no"
    if gt == "no":
        return "yes"
    digits = re.sub(r"[$,]", "", gt)
    if digits.isdigit():
        return str(int(digits) + 7)
    return "unknown"


def _build_adversarial_pool(
    clean_gens: List[AgentGeneration],
    ground_truth: str,
    coordination: str,
) -> Tuple[List[AgentGeneration], List[bool]]:
    """Build N=7 pool with f=2 Byzantine agents at the given coordination level."""
    pool = clean_gens[:_N]
    wrong = _derive_wrong_answer(ground_truth)
    wrong_text = f"The answer is {wrong}."

    result: List[AgentGeneration] = []
    faulty_mask: List[bool] = []

    for i, gen in enumerate(pool):
        if i >= _F:
            result.append(gen)
            faulty_mask.append(False)
            continue

        T = len(gen.token_logprobs) // 5 or _DEFAULT_TOKEN_COUNT

        if coordination == "uncoordinated":
            text = _F2_TEXT if i == 0 else _F3_TEXT
            logprobs_per_token = _COORD_LOGPROBS_PER_TOKEN
        elif coordination == "coordinated":
            text = wrong_text
            logprobs_per_token = _COORD_LOGPROBS_PER_TOKEN
        else:  # maximally_adversarial
            text = wrong_text
            logprobs_per_token = _MAX_ADV_LOGPROBS_PER_TOKEN

        result.append(AgentGeneration(
            agent_id=gen.agent_id,
            output_text=text,
            token_logprobs=logprobs_per_token * T,
            is_faulty=True,
            fault_type="F2_byzantine",
        ))
        faulty_mask.append(True)

    return result, faulty_mask


async def _aggregate_stage1_only(admitted: List[AgentGeneration]) -> str:
    """Nearest-centroid selection: geometric median → return closest output text."""
    if not admitted:
        return ""
    if len(admitted) == 1:
        return admitted[0].output_text
    texts = [g.output_text for g in admitted]
    embs = _embed(texts)
    median = _geometric_median(embs)
    dists = np.linalg.norm(embs - median, axis=1)
    return texts[int(np.argmin(dists))]


async def _run_pipeline_condition(
    agents: List[AgentGeneration],
    condition: str,
    tau: float,
    ground_truth: str,
) -> Tuple[str, bool, bool]:
    """Run one pipeline condition on a single question's agent pool.

    Returns (final_answer, is_liveness_fallback, is_low_confidence).
    majority_answer_vote: no filter, answer-extracted plurality.
    stage1_only / full_pipeline: Module 1 filter + liveness fallback.
    full_pipeline uses v2 weighted-vote aggregate.
    """
    if condition == "majority_answer_vote":
        return answer_majority_voting(agents, ground_truth), False, False

    admitted = await filter_agents(agents, tau)
    is_liveness = len(admitted) < 2 * _F + 1
    if is_liveness:
        admitted = agents

    if condition == "stage1_only":
        answer = await _aggregate_stage1_only(admitted)
        return answer, is_liveness, False
    else:  # full_pipeline — v2 weighted vote
        answer, is_vote_low = await aggregate(admitted)
        return answer, is_liveness, is_vote_low


def _compute_centroid_shift(
    agents: List[AgentGeneration],
    faulty_mask: List[bool],
) -> Dict[str, float]:
    """Measure how far arithmetic mean and geometric median drift from the clean cluster."""
    texts = [g.output_text for g in agents]
    embs = _embed(texts)

    clean_idx = [i for i, f in enumerate(faulty_mask) if not f]
    if not clean_idx:
        return {"dist_mean": 0.0, "dist_gm": 0.0, "delta": 0.0}

    clean_centroid = embs[clean_idx].mean(axis=0)
    mean_centroid = embs.mean(axis=0)
    gm_centroid = _geometric_median(embs)

    dist_mean = float(np.linalg.norm(mean_centroid - clean_centroid))
    dist_gm = float(np.linalg.norm(gm_centroid - clean_centroid))
    return {"dist_mean": dist_mean, "dist_gm": dist_gm, "delta": dist_mean - dist_gm}


async def run_experiment_3(
    cache_path: str,
    output_dir: str = "results",
    n_questions: Optional[int] = None,
) -> pd.DataFrame:
    """Adversarial stress test (v2): 3 coordination degrees × 3 pipeline conditions.

    Saves:
        <output_dir>/experiment_3_adversarial_v2.csv
        <output_dir>/experiment_3_adversarial_v2.png
    """
    print(f"Loading cache: {cache_path}")
    questions = load_cache(cache_path)
    if n_questions is not None:
        questions = questions[:n_questions]
    print(f"  {len(questions)} questions loaded.")

    import random as _random
    _random.Random(42).shuffle(questions)
    dev_n = max(0, int(len(questions) * 0.2))
    tau = calibrate_tau(questions[:dev_n] if dev_n > 0 else questions)
    questions = questions[dev_n:] if dev_n < len(questions) else questions
    print(f"  Calibrated τ = {tau:.4f} on {dev_n} dev questions, evaluating on {len(questions)}")

    acc: Dict[str, Dict[str, Dict[str, List]]] = {
        coord: {cond: {"correct": [], "fallback": [], "low_confidence": []} for cond in _PIPELINE_CONDITIONS}
        for coord in _COORDINATION_DEGREES
    }
    shifts: Dict[str, Dict[str, List[float]]] = {
        coord: {"dist_mean": [], "dist_gm": [], "delta": []}
        for coord in _COORDINATION_DEGREES
    }

    n_skipped = 0
    for q_idx, (ground_truth, clean_gens) in enumerate(questions):
        if len(clean_gens) < _N:
            n_skipped += 1
            continue

        if q_idx % 20 == 0:
            print(f"  Question {q_idx + 1}/{len(questions)}...")

        for coordination in _COORDINATION_DEGREES:
            agents, faulty_mask = _build_adversarial_pool(clean_gens, ground_truth, coordination)

            shift = _compute_centroid_shift(agents, faulty_mask)
            shifts[coordination]["dist_mean"].append(shift["dist_mean"])
            shifts[coordination]["dist_gm"].append(shift["dist_gm"])
            shifts[coordination]["delta"].append(shift["delta"])

            for condition in _PIPELINE_CONDITIONS:
                answer, is_liveness, is_low = await _run_pipeline_condition(
                    agents, condition, tau, ground_truth
                )
                is_correct = _extract_answer(answer, ground_truth) == ground_truth.strip()
                acc[coordination][condition]["correct"].append(is_correct)
                acc[coordination][condition]["fallback"].append(is_liveness)
                acc[coordination][condition]["low_confidence"].append(is_low)

    if n_skipped:
        print(f"  Skipped {n_skipped} questions with fewer than {_N} agents.")

    rows = []
    for coord in _COORDINATION_DEGREES:
        dist_mean_avg = float(np.mean(shifts[coord]["dist_mean"])) if shifts[coord]["dist_mean"] else 0.0
        dist_gm_avg = float(np.mean(shifts[coord]["dist_gm"])) if shifts[coord]["dist_gm"] else 0.0
        delta_avg = float(np.mean(shifts[coord]["delta"])) if shifts[coord]["delta"] else 0.0
        for cond in _PIPELINE_CONDITIONS:
            d = acc[coord][cond]
            rows.append({
                "coordination": coord,
                "pipeline_condition": cond,
                "accuracy": float(np.mean(d["correct"])) if d["correct"] else 0.0,
                "fallback_frequency": float(np.mean(d["fallback"])) if d["fallback"] else 0.0,
                "low_confidence_frequency": float(np.mean(d["low_confidence"])) if d["low_confidence"] else 0.0,
                "centroid_shift_mean": dist_mean_avg,
                "centroid_shift_gm": dist_gm_avg,
                "centroid_shift_delta": delta_avg,
            })

    df = pd.DataFrame(rows, columns=[
        "coordination", "pipeline_condition", "accuracy", "fallback_frequency",
        "low_confidence_frequency", "centroid_shift_mean", "centroid_shift_gm", "centroid_shift_delta",
    ])

    print("\nResults summary:")
    for _, row in df.iterrows():
        print(
            f"  [{row['coordination']:25s}] {row['pipeline_condition']:25s}"
            f"  acc={row['accuracy']:.1%}  fallback={row['fallback_frequency']:.1%}"
        )

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "experiment_3_adversarial_v2.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")

    png_path = os.path.join(output_dir, "experiment_3_adversarial_v2.png")
    plot_experiment_3(df, png_path)
    return df


def plot_experiment_3(df: pd.DataFrame, output_path: str) -> None:
    """Two-panel figure: accuracy vs coordination, centroid shift."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(_COORDINATION_DEGREES))
    x_labels = [_COORD_LABELS[c] for c in _COORDINATION_DEGREES]
    n_conds = len(_PIPELINE_CONDITIONS)
    bar_w = 0.7 / n_conds

    ax = axes[0]
    for j, cond in enumerate(_PIPELINE_CONDITIONS):
        offset = (j - (n_conds - 1) / 2) * bar_w
        y = [
            df.loc[
                (df["coordination"] == coord) & (df["pipeline_condition"] == cond),
                "accuracy",
            ].values[0]
            for coord in _COORDINATION_DEGREES
        ]
        bars = ax.bar(
            x + offset, y, bar_w * 0.9,
            label=_COND_LABELS[cond],
            color=_COND_COLORS[cond],
            alpha=0.85,
        )
        for bar, val in zip(bars, y):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.0%}",
                ha="center", va="bottom", fontsize=7.5,
                color=_COND_COLORS[cond], fontweight="bold",
            )

    ax.set_title("Accuracy vs. Coordination Degree (System V2)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Byzantine Coordination Level", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    shift_rows = df.drop_duplicates("coordination").set_index("coordination")
    width = 0.3

    for j, (col, label, color) in enumerate([
        ("centroid_shift_mean", "Arithmetic Mean", "#aec7e8"),
        ("centroid_shift_gm", "Geometric Median", "#2ca02c"),
    ]):
        offset = (j - 0.5) * width
        y = [shift_rows.loc[coord, col] for coord in _COORDINATION_DEGREES]
        bars = ax.bar(x + offset, y, width * 0.9, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, y):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="black",
            )

    for i, coord in enumerate(_COORDINATION_DEGREES):
        delta = shift_rows.loc[coord, "centroid_shift_delta"]
        max_h = max(
            shift_rows.loc[coord, "centroid_shift_mean"],
            shift_rows.loc[coord, "centroid_shift_gm"],
        )
        ax.text(
            x[i], max_h + 0.012,
            f"Δ={delta:.3f}",
            ha="center", va="bottom", fontsize=8.5,
            color="#2ca02c", fontweight="bold",
        )

    ax.set_title("Centroid Shift: Mean vs. Geometric Median", fontsize=13, fontweight="bold")
    ax.set_xlabel("Byzantine Coordination Level", fontsize=11)
    ax.set_ylabel("Distance to Clean Centroid  (↓ = more robust)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Experiment 3 V2: Adversarial Coordination Stress Test  (N={_N}, f={_F}, β≈28.6%)",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 3 (v2): Adversarial coordination stress test."
    )
    parser.add_argument("--cache", default="cache.json")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--n-questions", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_experiment_3(args.cache, args.output_dir, args.n_questions))
