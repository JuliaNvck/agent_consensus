#!/usr/bin/env python3
"""Generate paper figures from experiment_1 CSVs."""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
_FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

_BETAS = [0.0, 0.15, 0.30, 0.45]
_CONDITIONS = ["baseline", "soft_weighting", "hard_only", "full_system"]
_CONDITION_LABELS = {
    "baseline": "Self-Consistency (Baseline)",
    "soft_weighting": "Soft-Weighted SC",
    "hard_only": "Hard Filter + Majority",
    "full_system": "Full System (Proposed)",
}
_COLORS = {
    "baseline": "#d62728",
    "soft_weighting": "#ff7f0e",
    "hard_only": "#1f77b4",
    "full_system": "#2ca02c",
}
_STYLES = {
    "baseline": "--",
    "soft_weighting": ":",
    "hard_only": "-.",
    "full_system": "-",
}
_MARKERS = {
    "baseline": "s",
    "soft_weighting": "^",
    "hard_only": "D",
    "full_system": "o",
}


def _load(filepath: str) -> tuple[dict, float]:
    """Returns (accuracy_table, n1_accuracy) where accuracy_table[condition][beta] = mean accuracy."""
    rows = []
    with open(filepath) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) if k not in ("condition", "fault_type") else v
                         for k, v in row.items()})

    # N=1 single-agent accuracy (flat across all beta)
    n1_vals = [r["accuracy"] for r in rows if r["n_agents"] == 1.0]
    n1_acc = sum(n1_vals) / len(n1_vals) if n1_vals else 0.0

    # Accuracy by condition × beta, averaged over N∈{5,7} and all fault types
    table: dict[str, dict[float, float]] = {}
    for cond in _CONDITIONS:
        beta_acc: dict[float, list] = defaultdict(list)
        for r in rows:
            if r["condition"] == cond and r["n_agents"] != 1.0:
                beta_acc[r["beta"]].append(r["accuracy"])
        table[cond] = {b: sum(v) / len(v) for b, v in beta_acc.items()}

    return table, n1_acc


def plot_accuracy_vs_beta(
    llama_csv: str,
    qwen_csv: str,
    output_path: str,
    show_n1: bool = True,
) -> None:
    llama_table, llama_n1 = _load(llama_csv)
    qwen_table, qwen_n1 = _load(qwen_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    x = np.array(_BETAS)
    x_ticks = [0, 15, 30, 45]

    for ax, table, n1_acc, title in [
        (axes[0], llama_table, llama_n1, "LLaMA 3.1 8B Instruct"),
        (axes[1], qwen_table, qwen_n1, "Qwen2.5 7B Instruct"),
    ]:
        if show_n1:
            ax.axhline(n1_acc, color="gray", linestyle=":", linewidth=1.2,
                       label=f"Single Agent (N=1, {n1_acc:.2f})")

        for cond in _CONDITIONS:
            y = [table[cond].get(b, 0.0) for b in _BETAS]
            ax.plot(
                x_ticks, y,
                color=_COLORS[cond],
                linestyle=_STYLES[cond],
                marker=_MARKERS[cond],
                linewidth=2,
                markersize=7,
                label=_CONDITION_LABELS[cond],
            )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Fault Fraction β (%)", fontsize=11)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{v}%" for v in x_ticks])
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Accuracy", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Accuracy vs. Fault Fraction: BFT Consensus Pipeline", fontsize=14, y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close()


def plot_fault_type_breakdown(
    llama_csv: str,
    qwen_csv: str,
    output_path: str,
    beta: float = 0.45,
) -> None:
    """Grouped bar chart showing all 4 conditions by fault type at a given beta."""
    fault_types = ["F1", "F2", "F3", "mix"]
    cond_specs = [
        ("baseline",      _COLORS["baseline"],      "Self-Consistency (Baseline)"),
        ("soft_weighting", _COLORS["soft_weighting"], "Soft-Weighted SC"),
        ("hard_only",     _COLORS["hard_only"],      "Hard Filter + Majority"),
        ("full_system",   _COLORS["full_system"],    "Full System (stage1_only)"),
    ]
    n_conds = len(cond_specs)
    width = 0.18
    offsets = np.linspace(-(n_conds - 1) / 2 * width, (n_conds - 1) / 2 * width, n_conds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

    for ax, csvfile, title in [
        (axes[0], llama_csv, "LLaMA 3.1 8B Instruct"),
        (axes[1], qwen_csv, "Qwen2.5 7B Instruct"),
    ]:
        rows = []
        with open(csvfile) as f:
            for row in csv.DictReader(f):
                rows.append({k: float(v) if k not in ("condition", "fault_type") else v
                             for k, v in row.items()})

        x = np.arange(len(fault_types))

        for offset, (cond, color, label) in zip(offsets, cond_specs):
            y = []
            for ft in fault_types:
                vals = [r["accuracy"] for r in rows
                        if r["condition"] == cond and r["beta"] == beta
                        and r["fault_type"] == ft and r["n_agents"] != 1.0]
                y.append(sum(vals) / len(vals) if vals else 0.0)
            bars = ax.bar(x + offset, y, width, label=label, color=color, alpha=0.85)
            for bar, val in zip(bars, y):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                        f"{val:.0%}", ha="center", va="bottom", fontsize=7.5,
                        color=color, fontweight="bold")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Fault Type", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(["F1 (Crash)", "F2 (Byzantine)", "F3 (Drifter)", "Mix"])
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Accuracy", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Accuracy by Fault Type at β=45%", fontsize=14, y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close()


def plot_adversarial_combined(
    llama_csv: str,
    qwen_csv: str,
    output_path: str,
) -> None:
    """Two-panel figure: accuracy by coordination degree (LLaMA left, Qwen right)."""
    coord_degrees = ["uncoordinated", "coordinated", "maximally_adversarial"]
    coord_labels = ["Uncoordinated", "Coordinated", "Max Adversarial"]
    pipeline_specs = [
        ("majority_answer_vote", "#d62728", "--", "s", "Majority Vote"),
        ("stage1_only",          "#2ca02c", "-",  "o", "Stage1-Only (Geometric Median)"),
        ("full_pipeline",        "#9467bd", ":",  "^", "Full Pipeline (NLI)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    x = np.arange(len(coord_degrees))

    for ax, csvfile, title in [
        (axes[0], llama_csv, "LLaMA 3.1 8B Instruct"),
        (axes[1], qwen_csv,  "Qwen2.5 7B Instruct"),
    ]:
        rows = []
        with open(csvfile) as f:
            for row in csv.DictReader(f):
                rows.append(row)

        for cond, color, ls, marker, label in pipeline_specs:
            y = []
            for cd in coord_degrees:
                vals = [float(r["accuracy"]) for r in rows
                        if r["pipeline_condition"] == cond and r["coordination"] == cd]
                y.append(sum(vals) / len(vals) if vals else 0.0)
            ax.plot(x, y, color=color, linestyle=ls, marker=marker,
                    linewidth=2, markersize=8, label=label)
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:.1%}", (xi, yi), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8.5,
                            color=color, fontweight="bold")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(coord_labels)
        ax.set_ylim(0.4, 0.85)
        ax.set_yticks(np.arange(0.4, 0.9, 0.05))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].set_xlabel("Coordination Degree", fontsize=11)
    axes[1].set_xlabel("Coordination Degree", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Experiment 3: Accuracy Under Adversarial Coordination", fontsize=14, y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close()


if __name__ == "__main__":
    llama_csv = os.path.join(_RESULTS_DIR, "experiment_1_llama_v4.csv")
    qwen_csv = os.path.join(_RESULTS_DIR, "experiment_1_qwen_v4.csv")

    # With single-agent reference line (supplementary / appendix)
    plot_accuracy_vs_beta(
        llama_csv, qwen_csv,
        os.path.join(_FIGURES_DIR, "accuracy_vs_beta.png"),
        show_n1=True,
    )
    plot_fault_type_breakdown(
        llama_csv, qwen_csv,
        os.path.join(_FIGURES_DIR, "fault_type_breakdown.png"),
    )

    # Without single-agent reference line (main paper body)
    plot_accuracy_vs_beta(
        llama_csv, qwen_csv,
        os.path.join(_FIGURES_DIR, "accuracy_vs_beta_no_n1.png"),
        show_n1=False,
    )

    # Experiment 3: adversarial coordination (combined LLaMA + Qwen panel)
    plot_adversarial_combined(
        os.path.join(_RESULTS_DIR, "exp3_llama", "experiment_3_adversarial_v2.csv"),
        os.path.join(_RESULTS_DIR, "exp3_qwen",  "experiment_3_adversarial_v2.csv"),
        os.path.join(_FIGURES_DIR, "adversarial_coordination.png"),
    )
