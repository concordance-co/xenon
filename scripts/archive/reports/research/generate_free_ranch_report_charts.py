from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/brockelmore/concordance/xenon")
OUT_DIR = ROOT / "docs" / "reports" / "assets" / "free_ranch_research_sweep"
SYNTH_PATH = ROOT / "data" / "analysis_results" / "synthetic_policy_policy_algebra_v1_results.json"
REAL_PATH = ROOT / "data" / "analysis_results" / "research_rerun_kickoff_v2" / "results.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#fcfbf8",
            "figure.facecolor": "#fcfbf8",
            "savefig.facecolor": "#fcfbf8",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.titleweight": "bold",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def candidate_ranking_chart() -> None:
    labels = [
        "Preference vs Permission Algebra",
        "Strategy Priority Compliance",
        "Direct Block-Type Latent Valence",
        "Per-Slider Settings Semantics",
        "Hold / Portfolio Gating",
        "Triggered Sell / Immediate Action",
        "Observe Taxonomy",
        "Sell-Side Asymmetry",
        "Token Priors vs Abstract Reasoning",
        "Memory Inertia",
    ]
    scores = list(range(10, 0, -1))
    y = np.arange(len(labels))
    colors = ["#b33a2a"] + ["#d7a26a"] * 2 + ["#8aa1b1"] * 3 + ["#c9c3b8"] * 4

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.barh(y, scores, color=colors, edgecolor="#444444", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Research Priority Rank (higher is better)")
    ax.set_title("Free-Ranch Candidate Ranking")
    for yi, score in zip(y, scores):
        ax.text(score + 0.1, yi, f"#{11 - score}", va="center", fontsize=9, color="#333333")
    ax.set_xlim(0, 11.5)
    _save(fig, "candidate_ranking.png")


def synthetic_dataset_chart() -> None:
    families = ["permission_grid", "strategy_override_grid", "risk_gate_grid"]
    counts = [48, 48, 24]
    colors = ["#b33a2a", "#d98f4e", "#5d7fa3"]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bars = ax.bar(families, counts, color=colors, edgecolor="#444444", linewidth=0.7)
    ax.set_title("Synthetic Policy-Algebra Dataset")
    ax.set_ylabel("Prompt Count")
    ax.set_ylim(0, 56)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 1, str(count), ha="center", va="bottom")
    _save(fig, "synthetic_dataset.png")


def decomposition_chart(synth: dict) -> None:
    summary = synth["summary"]
    labels = [
        "Market Best\n(row_mean)",
        "Permission Mode\n(active_settings)",
        "Action Type\n(active_settings)",
        "Policy Best Asset\n(last_token)",
    ]
    values = [
        summary["market_best_asset_probe"]["hit_at_1"],
        summary["permission_mode_classifier"]["accuracy"],
        summary["expected_action_type_classifier"]["accuracy"],
        summary["policy_best_asset_classifier"]["accuracy"],
    ]
    colors = ["#2e7d32", "#5d7fa3", "#8b6fb8", "#b33a2a"]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(labels, values, color=colors, edgecolor="#444444", linewidth=0.7)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Held-Out Accuracy / Hit@1")
    ax.set_title("Synthetic Decomposition Summary")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    _save(fig, "synthetic_decomposition.png")


def policy_best_curve_chart(synth: dict) -> None:
    rows = synth["tick_classification"]["policy_best_asset"]
    by_section: dict[str, list[dict]] = {}
    for row in rows:
        by_section.setdefault(row["section_key"], []).append(row)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    palette = {"active_settings_eos": "#5d7fa3", "last_token": "#b33a2a"}
    labels = {"active_settings_eos": "active_settings_eos", "last_token": "last_token"}
    for section, srows in by_section.items():
        srows = sorted(srows, key=lambda r: r["layer"])
        ax.plot(
            [r["layer"] for r in srows],
            [r["accuracy"] for r in srows],
            label=labels[section],
            color=palette[section],
            linewidth=2.2,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Held-Out Accuracy")
    ax.set_ylim(0.2, 1.02)
    ax.set_title("Where Policy-Adjusted Asset Choice Sharpens")
    ax.legend(frameon=False)
    _save(fig, "policy_best_asset_curves.png")


def invariance_chart(synth: dict) -> None:
    inv = synth["summary"]["repeated_invariance"]
    labels = ["Permission", "Strategy", "Risk"]
    means = [
        inv["permission_top_symbol_invariance"]["mean"],
        inv["strategy_top_symbol_invariance"]["mean"],
        inv["risk_pair_policy_accuracy"]["mean"],
    ]
    stds = [
        inv["permission_top_symbol_invariance"]["std"],
        inv["strategy_top_symbol_invariance"]["std"],
        inv["risk_pair_policy_accuracy"]["std"],
    ]
    colors = ["#2e7d32", "#3b8d5b", "#d98f4e"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors, edgecolor="#444444", linewidth=0.7)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Repeated-Split Mean")
    ax.set_title("Repeated-Split Invariance Summary")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.03, f"{mean:.3f}", ha="center", va="bottom")
    _save(fig, "synthetic_invariance.png")


def real_validation_chart(real: dict) -> None:
    blocked = real["summary"]["blocked_valence"]
    settings = real["summary"]["settings_twist"]
    labels = [
        "Blocked Reveals",
        "Settings Flips",
        "Strong Shifts",
        "Symbol Reranks",
    ]
    numerators = [3, 17, 13, 0]
    denominators = [blocked["n_pairs"], settings["n_triplets"], settings["n_triplets"], 154]
    values = [n / d for n, d in zip(numerators, denominators)]
    colors = ["#d98f4e", "#5d7fa3", "#8b6fb8", "#2e7d32"]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bars = ax.bar(labels, values, color=colors, edgecolor="#444444", linewidth=0.7)
    ax.set_ylim(0, 0.22)
    ax.set_ylabel("Rate")
    ax.set_title("Real-Data Validation Snapshot")
    for bar, n, d in zip(bars, numerators, denominators):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{n}/{d}", ha="center", va="bottom")
    _save(fig, "real_validation.png")


def main() -> None:
    _style()
    synth = _load_json(SYNTH_PATH)
    real = _load_json(REAL_PATH)
    candidate_ranking_chart()
    synthetic_dataset_chart()
    decomposition_chart(synth)
    policy_best_curve_chart(synth)
    invariance_chart(synth)
    real_validation_chart(real)
    print(str(OUT_DIR))


if __name__ == "__main__":
    main()
