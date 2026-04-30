#!/usr/bin/env python3
"""Experiment 3: Adversarial Coordination Stress Test.

Tests pipeline resilience under three Byzantine coordination degrees:
  - uncoordinated:       f agents produce semantically distinct wrong answers
  - coordinated:         f agents produce the exact same wrong answer
  - maximally_adversarial: same wrong answer + max-confidence spoofed logprobs (-0.01)

Fixed parameters: N=7 agents, f=2 Byzantine, β≈28.6%.

Compares Full Pipeline vs. Stage 1 Only (geometric median, no NLI) vs. Majority Voting.
Also measures centroid shift: how much the geometric median resists being dragged toward
the adversarial cluster compared to the arithmetic mean.

CPU only — no vllm, no torch imports.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval.baselines import majority_voting
from eval.runner import _extract_answer, calibrate_tau, load_cache
from faults.injector import _F2_TEXT, _F3_TEXT
from models import AgentGeneration
from pipeline.aggregation import _embed, _geometric_median, aggregate
from pipeline.filter import filter_agents

_N: int = 7
_F: int = 2
_COORD_LOGPROB: float = -0.02
_MAX_ADV_LOGPROB: float = -0.01
_DEFAULT_TOKEN_COUNT: int = 20

_COORDINATION_DEGREES = ["uncoordinated", "coordinated", "maximally_adversarial"]
_PIPELINE_CONDITIONS = ["majority_voting", "stage1_only", "full_pipeline"]

_COORD_LABELS = {
    "uncoordinated": "Uncoordinated",
    "coordinated": "Coordinated",
    "maximally_adversarial": "Max. Adversarial",
}
_COND_LABELS = {
    "majority_voting": "Majority Voting",
    "stage1_only": "Stage 1 Only (Geom. Median)",
    "full_pipeline": "Full Pipeline (+ NLI)",
}
_COND_COLORS = {
    "majority_voting": "#d62728",
    "stage1_only": "#1f77b4",
    "full_pipeline": "#2ca02c",
}


def _derive_wrong_answer(ground_truth: str) -> str:
    """Derive an adversarial wrong answer opposite to the ground truth."""
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
    """Build N=7 pool with f=2 Byzantine agents at the given coordination level.

    Faults agents at indices 0 and 1 (deterministic, no seed needed).
    Returns (agents, faulty_mask) where True marks Byzantine agents.

    Coordination semantics:
      uncoordinated       — Agent 0 gets _F2_TEXT, Agent 1 gets _F3_TEXT (different clusters).
                            Both use -0.02 logprobs so both pass Module 1.
      coordinated         — Both get the same wrong answer text, -0.02 logprobs.
      maximally_adversarial — Same text, but logprobs = -0.01 (even higher spoofed confidence).
    """
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
            logprob = _COORD_LOGPROB
        elif coordination == "coordinated":
            text = wrong_text
            logprob = _COORD_LOGPROB
        else:  # maximally_adversarial
            text = wrong_text
            logprob = _MAX_ADV_LOGPROB

        result.append(AgentGeneration(
            agent_id=gen.agent_id,
            output_text=text,
            token_logprobs=[logprob] * (T * 5),
            is_faulty=True,
            fault_type="F2_byzantine",
        ))
        faulty_mask.append(True)

    return result, faulty_mask


async def _aggregate_stage1_only(admitted: List[AgentGeneration]) -> str:
    """Geometric median nearest-neighbor selection without NLI Stage 2."""
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
) -> Tuple[str, bool]:
    """Run one pipeline condition on a single question's agent pool.

    Returns (final_answer, is_low_confidence).
    majority_voting: no filter, plurality answer.
    stage1_only / full_pipeline: Module 1 filter + liveness fallback.
    """
    if condition == "majority_voting":
        return majority_voting(agents), False

    admitted = await filter_agents(agents, tau)
    if len(admitted) < 2 * _F + 1:
        admitted = agents
        is_low = True
    else:
        is_low = False

    if condition == "stage1_only":
        answer = await _aggregate_stage1_only(admitted)
    else:  # full_pipeline
        answer = await aggregate(admitted)

    return answer, is_low


def _compute_centroid_shift(
    agents: List[AgentGeneration],
    faulty_mask: List[bool],
) -> Dict[str, float]:
    """Measure how far arithmetic mean and geometric median drift from the clean cluster.

    Returns dist_mean, dist_gm (distance to clean centroid), and delta = dist_mean - dist_gm.
    Positive delta means the geometric median stays closer to the clean cluster (more robust).
    """
    texts = [g.output_text for g in agents]
    embs = _embed(texts)  # (N, D)

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
    """Adversarial stress test: 3 coordination degrees × 3 pipeline conditions.

    Saves:
        <output_dir>/experiment_3_adversarial.csv — per-condition accuracy + centroid shift
        <output_dir>/experiment_3_adversarial.png — 2-panel figure
    Returns the results DataFrame.
    """
    print(f"Loading cache: {cache_path}")
    questions = load_cache(cache_path)
    if n_questions is not None:
        questions = questions[:n_questions]
    print(f"  {len(questions)} questions loaded.")

    tau = calibrate_tau(questions)
    print(f"  Calibrated τ = {tau:.4f}")

    acc: Dict[str, Dict[str, Dict[str, List]]] = {
        coord: {cond: {"correct": [], "fallback": []} for cond in _PIPELINE_CONDITIONS}
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
                answer, is_low = await _run_pipeline_condition(agents, condition, tau)
                is_correct = _extract_answer(answer, ground_truth) == ground_truth.strip()
                acc[coordination][condition]["correct"].append(is_correct)
                acc[coordination][condition]["fallback"].append(is_low)

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
                "centroid_shift_mean": dist_mean_avg,
                "centroid_shift_gm": dist_gm_avg,
                "centroid_shift_delta": delta_avg,
            })

    df = pd.DataFrame(rows, columns=[
        "coordination", "pipeline_condition", "accuracy", "fallback_frequency",
        "centroid_shift_mean", "centroid_shift_gm", "centroid_shift_delta",
    ])

    print("\nResults summary:")
    for _, row in df.iterrows():
        print(
            f"  [{row['coordination']:25s}] {row['pipeline_condition']:25s}"
            f"  acc={row['accuracy']:.1%}  fallback={row['fallback_frequency']:.1%}"
        )

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "experiment_3_adversarial.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")

    png_path = os.path.join(output_dir, "experiment_3_adversarial.png")
    plot_experiment_3(df, png_path)
    return df


def plot_experiment_3(df: pd.DataFrame, output_path: str) -> None:
    """Two-panel figure: accuracy vs coordination, centroid shift."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(_COORDINATION_DEGREES))
    x_labels = [_COORD_LABELS[c] for c in _COORDINATION_DEGREES]
    n_conds = len(_PIPELINE_CONDITIONS)
    bar_w = 0.7 / n_conds

    # ── Panel A: Accuracy vs. Coordination Degree ─────────────────────────────
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

    ax.set_title("Accuracy vs. Coordination Degree", fontsize=13, fontweight="bold")
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

    # ── Panel B: Centroid Shift ────────────────────────────────────────────────
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
        f"Experiment 3: Adversarial Coordination Stress Test  (N={_N}, f={_F}, β≈28.6%)",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 3: Adversarial coordination stress test."
    )
    parser.add_argument(
        "--cache", default="cache.json",
        help="Path to the generation cache (default: cache.json).",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory for output CSV and PNG (default: results/).",
    )
    parser.add_argument(
        "--n-questions", type=int, default=None,
        help="Limit to the first N questions (for quick testing).",
    )
    args = parser.parse_args()
    asyncio.run(run_experiment_3(args.cache, args.output_dir, args.n_questions))
