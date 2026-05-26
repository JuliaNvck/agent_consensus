#!/usr/bin/env python3
"""Generate conference-ready figures for main_acl.tex.

This script writes only into paper/latex/figures_acl/. It reads existing
canonical CSVs and caches, and it does not modify result CSVs or cache JSONs.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures_acl"

sys.path.insert(0, str(ROOT))
from eval.signal_quality import analyze_cache  # noqa: E402

BETAS = [0.0, 0.15, 0.30, 0.45]
CONDITIONS = ["baseline", "soft_weighting", "hard_only", "full_system"]
COND_LABELS = {
    "baseline": "Self-consistency",
    "soft_weighting": "Soft-weighted",
    "hard_only": "TopKMass + vote",
    "full_system": "TopKMass + GM",
}
COLORS = {
    "baseline": "#d62728",
    "soft_weighting": "#ff7f0e",
    "hard_only": "#1f77b4",
    "full_system": "#2ca02c",
}
STYLES = {
    "baseline": "--",
    "soft_weighting": ":",
    "hard_only": "-.",
    "full_system": "-",
}
MARKERS = {
    "baseline": "s",
    "soft_weighting": "^",
    "hard_only": "D",
    "full_system": "o",
}
SIGNALS = ["topk_mass", "neg_entropy", "neg_logprob_var"]
SIGNAL_LABELS = {
    "topk_mass": "TopKMass",
    "neg_entropy": "-Entropy",
    "neg_logprob_var": "-Logprob Var",
}
SIGNAL_COLORS = {
    "topk_mass": "#2ca02c",
    "neg_entropy": "#1f77b4",
    "neg_logprob_var": "#ff7f0e",
}


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.035)
    print(f"saved {OUT / (stem + '.pdf')}")
    plt.close(fig)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _mean(vals: Iterable[float]) -> float:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0


def _exp1_table(path: Path) -> tuple[dict[str, dict[float, float]], float]:
    rows = _read_rows(path)
    n1 = _mean(float(r["accuracy"]) for r in rows if int(float(r["n_agents"])) == 1)
    table: dict[str, dict[float, float]] = {}
    for cond in CONDITIONS:
        by_beta: dict[float, list[float]] = defaultdict(list)
        for r in rows:
            if r["condition"] != cond or int(float(r["n_agents"])) == 1:
                continue
            by_beta[float(r["beta"])].append(float(r["accuracy"]))
        table[cond] = {b: _mean(v) for b, v in by_beta.items()}
    return table, n1


def plot_accuracy(show_n1: bool, stem: str) -> None:
    llama, llama_n1 = _exp1_table(ROOT / "results/experiment_1_llama_v4.csv")
    qwen, qwen_n1 = _exp1_table(ROOT / "results/experiment_1_qwen_v4.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharey=True)
    x_ticks = [0, 15, 30, 45]
    for ax, table, n1, title in [
        (axes[0], llama, llama_n1, "LLaMA 3.1 8B"),
        (axes[1], qwen, qwen_n1, "Qwen2.5 7B"),
    ]:
        if show_n1:
            ax.axhline(n1, color="0.45", linestyle=":", linewidth=1.1, label=f"N=1 ({n1:.2f})")
        for cond in CONDITIONS:
            ax.plot(
                x_ticks,
                [table[cond][b] for b in BETAS],
                color=COLORS[cond],
                linestyle=STYLES[cond],
                marker=MARKERS[cond],
                linewidth=1.55,
                markersize=4.2,
                label=COND_LABELS[cond],
            )
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel(r"Fault fraction $\beta$ (%)", fontsize=8.5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{v}%" for v in x_ticks], fontsize=7.5)
        ax.set_ylim(0.5, 0.82)
        ax.set_yticks(np.arange(0.5, 0.85, 0.1))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Accuracy", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5 if show_n1 else 4, fontsize=7.5, frameon=False)
    fig.subplots_adjust(bottom=0.27, left=0.08, right=0.995, top=0.86, wspace=0.12)
    _save(fig, stem)


def plot_fault_breakdown() -> None:
    fault_types = ["F1", "F2", "F3", "mix"]
    fault_labels = ["F1", "F2", "F3", "Mix"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharey=True)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(CONDITIONS))
    for ax, path, title in [
        (axes[0], ROOT / "results/experiment_1_llama_v4.csv", "LLaMA 3.1 8B"),
        (axes[1], ROOT / "results/experiment_1_qwen_v4.csv", "Qwen2.5 7B"),
    ]:
        rows = _read_rows(path)
        x = np.arange(len(fault_types))
        for offset, cond in zip(offsets, CONDITIONS):
            values = []
            for ft in fault_types:
                vals = [
                    float(r["accuracy"])
                    for r in rows
                    if r["condition"] == cond
                    and float(r["beta"]) == 0.45
                    and r["fault_type"] == ft
                    and int(float(r["n_agents"])) != 1
                ]
                values.append(_mean(vals))
            bars = ax.bar(x + offset, values, width, label=COND_LABELS[cond], color=COLORS[cond], alpha=0.9)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.008,
                    f"{val:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=5.7,
                    color=COLORS[cond],
                    fontweight="bold",
                )
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Fault type", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(fault_labels, fontsize=7.5)
        ax.set_ylim(0.5, 0.82)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Accuracy", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.4, frameon=False)
    fig.subplots_adjust(bottom=0.25, left=0.08, right=0.995, top=0.86, wspace=0.12)
    _save(fig, "fault_type_breakdown_acl")


def _signal_df(cache: str):
    return analyze_cache(str(ROOT / cache))


def plot_signal_panels(cache: str, stem: str, title: str) -> None:
    df = _signal_df(cache)
    y_true = df["is_correct"].astype(int).values
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="0.55", linestyle="--", linewidth=0.8, label="Chance")
    for sig in SIGNALS:
        scores = df[sig].values
        auc = roc_auc_score(y_true, scores)
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, color=SIGNAL_COLORS[sig], linewidth=1.3, label=f"{SIGNAL_LABELS[sig]} ({auc:.3f})")
    ax.set_title("ROC", fontsize=8.8, fontweight="bold")
    ax.set_xlabel("False positive rate", fontsize=7.5)
    ax.set_ylabel("True positive rate", fontsize=7.5)
    ax.tick_params(labelsize=6.8)
    ax.legend(
        fontsize=5.6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.4,
    )
    ax.grid(linestyle="--", linewidth=0.45, alpha=0.3)

    ax = axes[1]
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.055, 0.055, size=len(df))
    correct = df["is_correct"].values.astype(bool)
    ax.scatter(df.loc[~correct, "topk_mass"], jitter[~correct], s=5, color="#d62728", alpha=0.35, label="Incorrect")
    ax.scatter(df.loc[correct, "topk_mass"], 1 + jitter[correct], s=5, color="#2ca02c", alpha=0.35, label="Correct")
    for is_correct, color in [(False, "#d62728"), (True, "#2ca02c")]:
        med = df.loc[df["is_correct"] == is_correct, "topk_mass"].median()
        y = 0 if not is_correct else 1
        ax.vlines(med, y - 0.13, y + 0.13, colors=color, linewidth=1.4)
    ax.set_title("TopKMass vs. correctness", fontsize=8.8, fontweight="bold")
    ax.set_xlabel("Stable-region TopKMass", fontsize=7.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Incorrect", "Correct"], fontsize=6.8)
    ax.tick_params(axis="x", labelsize=6.8)
    ax.set_ylim(-0.25, 1.25)
    ax.grid(axis="x", linestyle="--", linewidth=0.45, alpha=0.3)
    ax.legend(fontsize=5.8, loc="center right", frameon=False)

    ax = axes[2]
    baseline = float(np.mean(y_true))
    ax.axhline(baseline, color="0.55", linestyle="--", linewidth=0.8, label=f"Random ({baseline:.3f})")
    for sig in SIGNALS:
        scores = df[sig].values
        ap = average_precision_score(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.plot(recall, precision, color=SIGNAL_COLORS[sig], linewidth=1.3, label=f"{SIGNAL_LABELS[sig]} ({ap:.3f})")
    ax.set_title("Precision-recall", fontsize=8.8, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=7.5)
    ax.set_ylabel("Precision", fontsize=7.5)
    ax.tick_params(labelsize=6.8)
    ax.legend(
        fontsize=5.6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.4,
    )
    ax.grid(linestyle="--", linewidth=0.45, alpha=0.3)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(title, fontsize=9.5, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.28, top=0.79, wspace=0.32)
    _save(fig, stem)


def plot_topkmass_distribution() -> None:
    df = _signal_df("cache_llma.json")
    correct = df["is_correct"].values.astype(bool)
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    bins = np.linspace(0.78, 1.0, 32)
    ax.hist(df.loc[~correct, "topk_mass"], bins=bins, alpha=0.55, color="#d62728", label=f"Incorrect (n={(~correct).sum()})")
    ax.hist(df.loc[correct, "topk_mass"], bins=bins, alpha=0.55, color="#2ca02c", label=f"Correct (n={correct.sum()})")
    for is_correct, color in [(False, "#d62728"), (True, "#2ca02c")]:
        med = df.loc[df["is_correct"] == is_correct, "topk_mass"].median()
        ax.axvline(med, color=color, linestyle="--", linewidth=1.2)
    ax.text(
        0.06,
        0.63,
        "Fault scores off scale:\nF1=0, F3 approx. 2.3e-4",
        transform=ax.transAxes,
        fontsize=5.9,
        color="0.35",
    )
    ax.set_title("LLaMA TopKMass distribution", fontsize=8.8, fontweight="bold")
    ax.set_xlabel("Stable-region TopKMass", fontsize=7.5)
    ax.set_ylabel("Count", fontsize=7.5)
    ax.tick_params(labelsize=6.8)
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.2, top=0.88)
    _save(fig, "topkmass_distribution_acl")


def plot_exp3() -> None:
    coord = ["uncoordinated", "coordinated", "maximally_adversarial"]
    labels = ["Uncoord.", "Coord.", "Max-conf. coord."]
    specs = [
        ("majority_answer_vote", "#d62728", "--", "s", "Majority vote"),
        ("stage1_only", "#2ca02c", "-", "o", "GM selector"),
        ("full_pipeline", "#9467bd", ":", "^", "Weighted vote"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.75))
    for col, (csv_path, model) in enumerate([
        (ROOT / "results/exp3_llama/experiment_3_adversarial_v2.csv", "LLaMA 3.1 8B"),
        (ROOT / "results/exp3_qwen/experiment_3_adversarial_v2.csv", "Qwen2.5 7B"),
    ]):
        rows = _read_rows(csv_path)
        ax = axes[0, col]
        x = np.arange(len(coord))
        for cond, color, ls, marker, label in specs:
            vals = [
                _mean(float(r["accuracy"]) for r in rows if r["pipeline_condition"] == cond and r["coordination"] == c)
                for c in coord
            ]
            ax.plot(x, vals, color=color, linestyle=ls, marker=marker, linewidth=1.4, markersize=4, label=label)
            for xi, yi in zip(x, vals):
                ax.text(xi, yi + 0.008, f"{yi:.1%}", ha="center", fontsize=5.8, color=color, fontweight="bold")
        ax.set_title(f"{model}: accuracy", fontsize=8.8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.8)
        ax.set_ylim(0.4, 0.76)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(axis="y", labelsize=6.8)
        ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == 0:
            ax.set_ylabel("Accuracy", fontsize=7.5)

        ax = axes[1, col]
        shift = {}
        for c in coord:
            row = next(r for r in rows if r["coordination"] == c and r["pipeline_condition"] == "stage1_only")
            shift[c] = (float(row["centroid_shift_mean"]), float(row["centroid_shift_gm"]), float(row["centroid_shift_delta"]))
        width = 0.34
        mean_vals = [shift[c][0] for c in coord]
        gm_vals = [shift[c][1] for c in coord]
        ax.bar(x - width / 2, mean_vals, width, color="#9ecae1", label="Arithmetic mean")
        ax.bar(x + width / 2, gm_vals, width, color="#74c476", label="Geometric median")
        for xi, c in zip(x, coord):
            ax.text(xi, max(shift[c][0], shift[c][1]) + 0.012, f"Δ={shift[c][2]:.3f}", ha="center", fontsize=5.8)
        ax.set_title(f"{model}: centroid shift", fontsize=8.8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.8)
        ax.set_ylim(0, 0.42)
        ax.tick_params(axis="y", labelsize=6.8)
        ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == 0:
            ax.set_ylabel("Distance to honest centroid", fontsize=7.5)
    line_handles, line_labels = axes[0, 0].get_legend_handles_labels()
    bar_handles = [
        Patch(facecolor="#9ecae1", label="Arithmetic mean"),
        Patch(facecolor="#74c476", label="Geometric median"),
    ]
    fig.legend(
        line_handles + bar_handles,
        line_labels + ["Arithmetic mean", "Geometric median"],
        loc="lower center",
        ncol=5,
        fontsize=6.8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.14, top=0.94, wspace=0.18, hspace=0.42)
    _save(fig, "adversarial_coordination_acl")


def main() -> None:
    plot_accuracy(False, "accuracy_vs_beta_no_n1_acl")
    plot_accuracy(True, "accuracy_vs_beta_acl")
    plot_fault_breakdown()
    plot_signal_panels("cache_llma.json", "experiment_2_signals_llama_acl", "Experiment 2: LLaMA signal quality")
    plot_signal_panels("cache_qwen.json", "experiment_2_signals_qwen_acl", "Experiment 2: Qwen signal quality")
    plot_topkmass_distribution()
    plot_exp3()


if __name__ == "__main__":
    main()
