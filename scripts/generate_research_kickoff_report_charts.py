from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AUDIT_PATH = Path("data/analysis_results/research_kickoff/audit.json")
OUTPUT_DIR = Path("data/report_assets/research_kickoff")

NAVY = "#16324F"
TEAL = "#2E6A69"
GOLD = "#CA9440"
ROSE = "#B56662"
SLATE = "#5E6F82"
GRID = "#D6DEE3"
CHARCOAL = "#21313F"
SAND = "#D9C3A5"


def _load() -> dict:
    return json.loads(AUDIT_PATH.read_text())


def _setup_axes(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)


def roadmap_scores_chart(audit: dict) -> Path:
    path = OUTPUT_DIR / "roadmap_scores.png"
    tracks = sorted(audit["roadmap"], key=lambda row: row["rank"])
    labels = [track["title"] for track in tracks]
    values = [track["score"] for track in tracks]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10.4, 5.4), dpi=180)
    fig.patch.set_facecolor("white")
    bars = ax.barh(y, values, color=[NAVY, TEAL, GOLD, ROSE, SAND], edgecolor="none", height=0.62)
    _setup_axes(ax, grid_axis="x")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 10)
    ax.set_xlabel("Priority score")
    ax.set_title("Ranked research tracks", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    for bar, value in zip(bars, values, strict=True):
        ax.text(value + 0.08, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", ha="left", fontsize=9, color=SLATE)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def candidate_pools_chart(audit: dict) -> Path:
    path = OUTPUT_DIR / "candidate_pools.png"
    counts = audit["candidate_pool_counts"]
    labels = [
        "Blocked observe",
        "Policy tension",
        "Buy trades",
        "Sell trades",
    ]
    values = [
        counts["blocked_observe_candidates"],
        counts["policy_tension_candidates"],
        counts["buy_candidates"],
        counts["sell_candidates"],
    ]

    fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    bars = ax.bar(labels, values, color=[ROSE, TEAL, NAVY, GOLD], edgecolor="none", width=0.64)
    _setup_axes(ax, grid_axis="y")
    ax.set_ylabel("Candidate rows")
    ax.set_title("Live candidate pools for the top-ranked tracks", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    ax.set_ylim(0, max(values) * 1.16)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.012, f"{value:,}", ha="center", va="bottom", fontsize=9, color=SLATE)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def blocked_reasons_chart(audit: dict) -> Path:
    path = OUTPUT_DIR / "blocked_reasons.png"
    items = list(audit["blocked_reason_counts"].items())
    labels = [key.replace("_", " ") for key, _ in items]
    values = [value for _, value in items]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10.2, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    bars = ax.barh(y, values, color=ROSE, edgecolor="none", height=0.62)
    _setup_axes(ax, grid_axis="x")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Rows")
    ax.set_title("Blocked-observe pool is dominated by strategy-driven cases", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    for bar, value in zip(bars, values, strict=True):
        ax.text(value + max(values) * 0.01, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", ha="left", fontsize=9, color=SLATE)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def policy_regimes_chart(audit: dict) -> Path:
    path = OUTPUT_DIR / "policy_regimes.png"
    items = list(audit["policy_risk_activity_counts"].items())[:10]
    labels = [key for key, _ in items]
    values = [value for _, value in items]

    fig, ax = plt.subplots(figsize=(10.6, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    bars = ax.bar(labels, values, color=TEAL, edgecolor="none", width=0.64)
    _setup_axes(ax, grid_axis="y")
    ax.set_ylabel("Rows")
    ax.set_title("Policy-tension candidates cluster in a few extreme settings cells", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    ax.set_ylim(0, max(values) * 1.18)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.012, f"{value:,}", ha="center", va="bottom", fontsize=8, color=SLATE, rotation=90)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def kickoff_manifest_chart(audit: dict) -> Path:
    path = OUTPUT_DIR / "kickoff_manifests.png"
    blocked = audit["kickoff_manifest_summary"]["blocked_valence"]["cohort_counts"]
    settings = audit["kickoff_manifest_summary"]["settings_twist"]["cohort_counts"]
    labels = ["Blocked-valence kickoff", "Settings-twist kickoff"]
    blocked_vals = [blocked.get("blocked_observe", 0), settings.get("blocked_observe", 0)]
    policy_vals = [0, settings.get("policy_tension_observe", 0)]
    buy_vals = [0, settings.get("buy", 0)]
    sell_vals = [0, settings.get("sell", 0)]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.bar(x, blocked_vals, color=ROSE, edgecolor="none", width=0.6, label="Blocked observe")
    ax.bar(x, policy_vals, bottom=blocked_vals, color=TEAL, edgecolor="none", width=0.6, label="Policy tension observe")
    ax.bar(x, buy_vals, bottom=np.array(blocked_vals) + np.array(policy_vals), color=NAVY, edgecolor="none", width=0.6, label="Buy refs")
    ax.bar(x, sell_vals, bottom=np.array(blocked_vals) + np.array(policy_vals) + np.array(buy_vals), color=GOLD, edgecolor="none", width=0.6, label="Sell refs")
    _setup_axes(ax, grid_axis="y")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Rows")
    ax.set_title("Kickoff manifests built from the live pool", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    totals = np.array(blocked_vals) + np.array(policy_vals) + np.array(buy_vals) + np.array(sell_vals)
    ax.set_ylim(0, max(totals) * 1.18)
    for xi, total in zip(x, totals, strict=True):
        ax.text(xi, total + max(totals) * 0.02, f"{int(total)}", ha="center", va="bottom", fontsize=9, color=SLATE)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = _load()
    outputs = {
        "roadmap_scores": str(roadmap_scores_chart(audit)),
        "candidate_pools": str(candidate_pools_chart(audit)),
        "blocked_reasons": str(blocked_reasons_chart(audit)),
        "policy_regimes": str(policy_regimes_chart(audit)),
        "kickoff_manifests": str(kickoff_manifest_chart(audit)),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
