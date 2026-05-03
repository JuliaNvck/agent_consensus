#!/usr/bin/env python3
"""Experiment 2: Signal Quality Analysis.

Computes three per-agent generation signals (TopKMass, Token Entropy, Logprob Variance),
labels each generation as correct/incorrect against ground truth, then produces ROC/AUC
and Precision-Recall curves to validate that TopKMass is the best correctness predictor.

Optionally injects F2/F3 faults to additionally measure fault detectability.

CPU only — no vllm, no torch imports.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from eval.runner import _extract_answer, load_cache
from faults.injector import inject_faults
from models import AgentGeneration
from pipeline.filter import _compute_topk_mass_trajectory

_SIGNALS = ["topk_mass", "neg_entropy", "neg_logprob_var"]
_SIGNAL_LABELS = {
    "topk_mass": "TopKMass",
    "neg_entropy": "−Entropy",
    "neg_logprob_var": "−Logprob Var",
}
_COLORS = {
    "topk_mass": "#2ca02c",
    "neg_entropy": "#1f77b4",
    "neg_logprob_var": "#ff7f0e",
}


def _mean_token_entropy(token_logprobs: List[float]) -> float:
    """Approximate mean token entropy from top-5 logprobs per position.

    H_i = -sum(exp(lp) * lp) for the top-5 logprobs at position i.
    Uses raw (unnormalized) top-5 probs — a standard approximation when the full
    vocabulary distribution is unavailable.
    """
    arr = np.array(token_logprobs, dtype=np.float64).reshape(-1, 5)
    probs = np.exp(arr)                        # (T, 5) unnormalized top-5 probs
    H_per_pos = -np.sum(probs * arr, axis=1)  # (T,) = -sum(p * log(p)) per position
    return float(H_per_pos.mean())


def _logprob_variance(token_logprobs: List[float]) -> float:
    """Variance of per-position mean logprob across the token sequence.

    Captures volatility in generation confidence token-to-token.
    """
    arr = np.array(token_logprobs, dtype=np.float64).reshape(-1, 5)
    per_pos_mean = arr.mean(axis=1)  # (T,) mean logprob per position
    return float(per_pos_mean.var())


def compute_signals(gen: AgentGeneration) -> Optional[Dict[str, float]]:
    """Compute all three signals for one AgentGeneration.

    Returns None if token_logprobs is empty (e.g. F1_crash agents).
    All returned signals are oriented so that higher = more confident = more likely correct:
    - topk_mass:       higher is better (raw mean TopKMass)
    - neg_entropy:     higher is better (negated entropy — low entropy = confident)
    - neg_logprob_var: higher is better (negated variance — low variance = stable)
    """
    if not gen.token_logprobs:
        return None
    traj = _compute_topk_mass_trajectory(gen.token_logprobs)
    if len(traj) == 0:
        return None
    return {
        "topk_mass": float(traj.mean()),
        "neg_entropy": -_mean_token_entropy(gen.token_logprobs),
        "neg_logprob_var": -_logprob_variance(gen.token_logprobs),
    }


def analyze_cache(
    cache_path: str,
    include_faults: bool = False,
    beta: float = 0.3,
    fault_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load a generation cache and compute signals + correctness for every agent.

    When include_faults=True, also injects faults for each type in fault_types and
    includes those rows. Clean rows are labeled fault_type="clean", is_faulty=False.

    Returns a DataFrame with columns:
        question_id, topk_mass, neg_entropy, neg_logprob_var, is_correct,
        fault_type, is_faulty
    One row per agent generation with non-empty token_logprobs.
    """
    import json

    questions = load_cache(cache_path)

    with open(cache_path) as fh:
        raw = json.load(fh)
    qid_list = [q["question_id"] for q in raw["questions"]]

    if fault_types is None:
        fault_types = ["F2", "F3"]

    rows: List[Dict] = []
    for idx, (ground_truth, gens) in enumerate(questions):
        question_id = qid_list[idx]

        # Clean agents
        for gen in gens:
            signals = compute_signals(gen)
            if signals is None:
                continue
            is_correct = (
                _extract_answer(gen.output_text, ground_truth) == ground_truth.strip()
            )
            rows.append({
                "question_id": question_id,
                **signals,
                "is_correct": bool(is_correct),
                "fault_type": "clean",
                "is_faulty": False,
            })

        # Injected-fault agents (optional)
        if include_faults:
            for ft in fault_types:
                injected = inject_faults(gens, beta=beta, fault_type=ft, seed=42)
                for gen in injected:
                    if not gen.is_faulty:
                        continue
                    signals = compute_signals(gen)
                    if signals is None:
                        continue
                    is_correct = (
                        _extract_answer(gen.output_text, ground_truth) == ground_truth.strip()
                    )
                    rows.append({
                        "question_id": question_id,
                        **signals,
                        "is_correct": bool(is_correct),
                        "fault_type": ft,
                        "is_faulty": True,
                    })

    return pd.DataFrame(
        rows,
        columns=[
            "question_id", "topk_mass", "neg_entropy", "neg_logprob_var",
            "is_correct", "fault_type", "is_faulty",
        ],
    )


def plot_signals(df: pd.DataFrame, output_path: str) -> None:
    """Generate the Experiment 2 figure and save to output_path.

    Panel A: ROC curves for all three signals (correctness prediction).
    Panel B: Scatter of TopKMass score vs. correctness (jittered).
    Panel C: Precision-Recall curves for all three signals (correctness prediction).
    Panel D (if faults injected): ROC + PR for fault detection via TopKMass.
    """
    has_faults = "is_faulty" in df.columns and df["is_faulty"].any()
    n_panels = 4 if has_faults else 3
    fig_w = 5 * n_panels
    y_true = df["is_correct"].astype(int).values

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 5))

    # ── Panel A: ROC Curves ────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance")
    for sig in _SIGNALS:
        scores = df[sig].values
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(
            fpr, tpr,
            color=_COLORS[sig],
            linewidth=2,
            label=f"{_SIGNAL_LABELS[sig]} (AUC={auc:.3f})",
        )
    ax.set_title("ROC Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle="--", alpha=0.4)

    # ── Panel B: TopKMass Score vs. Correctness ────────────────────────────────
    ax = axes[1]
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.07, 0.07, size=len(df))
    y_jittered = df["is_correct"].astype(float).values + jitter

    incorrect_mask = ~df["is_correct"].values
    correct_mask = df["is_correct"].values
    ax.scatter(
        df.loc[incorrect_mask, "topk_mass"], y_jittered[incorrect_mask],
        alpha=0.2, s=8, color="#d62728", label="Incorrect",
    )
    ax.scatter(
        df.loc[correct_mask, "topk_mass"], y_jittered[correct_mask],
        alpha=0.2, s=8, color="#2ca02c", label="Correct",
    )

    # Median lines per class
    for is_correct, color in [(False, "#d62728"), (True, "#2ca02c")]:
        med = df.loc[df["is_correct"] == is_correct, "topk_mass"].median()
        y_pos = 0.0 if not is_correct else 1.0
        ax.axvline(med, ymin=(y_pos - 0.1 + 0.5) / 1.14, ymax=(y_pos + 0.1 + 0.5) / 1.14,
                   color=color, linewidth=2.5, alpha=0.9)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("TopKMass vs. Correctness", fontsize=13, fontweight="bold")
    ax.set_xlabel("TopKMass Score (mean)", fontsize=11)
    ax.set_ylabel("Correct (1) / Incorrect (0)", fontsize=11)
    ax.set_ylim(-0.25, 1.25)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Incorrect", "Correct"])
    ax.legend(fontsize=9, loc="center right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # ── Panel C: Precision-Recall Curves ──────────────────────────────────────
    ax = axes[2]
    baseline_pr = y_true.mean()
    ax.axhline(
        baseline_pr, color="gray", linestyle="--", linewidth=1,
        label=f"Random ({baseline_pr:.3f})",
    )
    for sig in _SIGNALS:
        scores = df[sig].values
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(
            rec, prec,
            color=_COLORS[sig],
            linewidth=2,
            label=f"{_SIGNAL_LABELS[sig]} (AP={ap:.3f})",
        )
    ax.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle="--", alpha=0.4)

    # ── Panel D: Fault Detection (only when injected faults present) ──────────
    if has_faults:
        ax = axes[3]
        faulty_true = df["is_faulty"].astype(int).values
        scores = df["topk_mass"].values

        # For fault detection, lower TopKMass → more likely faulty, so invert the score
        fpr, tpr, _ = roc_curve(faulty_true, -scores)
        auc = roc_auc_score(faulty_true, -scores)
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance")
        ax.plot(fpr, tpr, color=_COLORS["topk_mass"], linewidth=2,
                label=f"TopKMass (AUC={auc:.3f})")
        ax.set_title("Fault Detection ROC\n(−TopKMass as detector)", fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(linestyle="--", alpha=0.4)

    title = "Signal Quality Analysis: TopKMass vs. Entropy vs. Logprob Variance"
    if has_faults:
        title += "  (+Fault Detection)"
    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close()


def run_experiment_2(
    cache_path: str,
    output_dir: str = "results",
    include_faults: bool = False,
    beta: float = 0.3,
    fault_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Analyze a generation cache and produce signal quality outputs.

    Saves:
        <output_dir>/experiment_2_signals.csv  — per-agent signal DataFrame
        <output_dir>/experiment_2_signals.png  — figure (3 panels clean, 4 panels with faults)
    Returns the DataFrame.
    """
    print(f"Analyzing {cache_path}...")
    df = analyze_cache(cache_path, include_faults=include_faults, beta=beta,
                       fault_types=fault_types)
    n_clean = int((df["fault_type"] == "clean").sum())
    n_faulty = int(df["is_faulty"].sum()) if include_faults else 0
    print(f"  {n_clean} clean generations | {df['is_correct'].sum()} correct "
          f"({df.loc[df['fault_type']=='clean','is_correct'].mean():.1%})")
    if include_faults:
        print(f"  {n_faulty} injected-fault generations included")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "experiment_2_signals.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    png_path = os.path.join(output_dir, "experiment_2_signals.png")
    plot_signals(df, png_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 2: Signal quality analysis on a generation cache."
    )
    parser.add_argument(
        "--cache", default="cache.json",
        help="Path to the Phase 1 JSON generation cache (default: cache.json).",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory for output CSV and PNG (default: results/).",
    )
    parser.add_argument(
        "--include-faults", action="store_true",
        help="Also inject faults and include fault detection analysis.",
    )
    parser.add_argument(
        "--beta", type=float, default=0.3,
        help="Fault injection fraction (default: 0.3).",
    )
    parser.add_argument(
        "--fault-types", nargs="+", default=["F2", "F3"],
        metavar="FAULT_TYPE",
        help="Fault types to inject (default: F2 F3).",
    )
    args = parser.parse_args()
    run_experiment_2(
        args.cache, args.output_dir,
        include_faults=args.include_faults,
        beta=args.beta,
        fault_types=args.fault_types,
    )
