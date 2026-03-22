from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "reports" / "assets" / "actionability_affordance"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load(name: str) -> dict:
    return json.loads((ROOT / "data" / "analysis_results" / name).read_text())


V1 = _load("synthetic_policy_actionability_algebra_v1_results.json")
V2 = _load("synthetic_policy_actionability_algebra_v2_results.json")
V3 = _load("synthetic_policy_actionability_algebra_v3_results.json")
V4 = _load("synthetic_policy_actionability_algebra_v4_results.json")
V4A = _load("synthetic_policy_actionability_algebra_v4_affordances_results.json")


def _metric(summary: dict, key: str) -> float:
    return float(summary[key]["balanced_accuracy"])


def make_phase_progression() -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    phases = ["v1", "v2", "v3", "v4"]
    x = np.arange(len(phases))
    width = 0.23

    permission = [_metric(d["summary"], "permission_mode_classifier") for d in [V1, V2, V3, V4]]
    action = [_metric(d["summary"], "expected_action_type_classifier") for d in [V1, V2, V3, V4]]
    policy = [_metric(d["summary"], "policy_best_asset_classifier") for d in [V1, V2, V3, V4]]

    ax.bar(x - width, permission, width, label="permission_mode", color="#b33a2a")
    ax.bar(x, action, width, label="expected_action_type", color="#58677c")
    ax.bar(x + width, policy, width, label="policy_best_asset", color="#b2a26b")

    ax.set_xticks(x, phases)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Fused downstream targets across prompt hardening")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()

    path = OUT_DIR / "phase_progression.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_v4_factorization() -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = [
        "permission_mode",
        "expected_action_type",
        "policy_best_asset",
        "observe_vs_act",
        "can_buy",
        "can_sell",
    ]
    values = [
        _metric(V4A["summary"], "permission_mode_classifier"),
        _metric(V4A["summary"], "expected_action_type_classifier"),
        _metric(V4A["summary"], "policy_best_asset_classifier"),
        _metric(V4A["summary"], "observe_vs_act_classifier"),
        _metric(V4A["summary"], "can_buy_classifier"),
        _metric(V4A["summary"], "can_sell_classifier"),
    ]
    colors = ["#b33a2a", "#6a7f96", "#a89b6d", "#3c7d5b", "#2f6f86", "#684c8d"]
    ypos = np.arange(len(labels))

    ax.barh(ypos, values, color=colors)
    ax.set_yticks(ypos, labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Balanced accuracy")
    ax.set_title("`v4` improves when permission is decomposed into affordance bits")
    ax.grid(axis="x", alpha=0.2)
    for y, value in zip(ypos, values):
        ax.text(value + 0.015, y, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()

    path = OUT_DIR / "v4_factorization.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_section_summary() -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    items = [
        ("market_best_asset", V4A["summary"]["market_best_asset_probe"]["row_key"], V4A["summary"]["market_best_asset_probe"]["layer"], 1.0),
        ("observe_vs_act", V4A["summary"]["observe_vs_act_classifier"]["section_key"], V4A["summary"]["observe_vs_act_classifier"]["layer"], _metric(V4A["summary"], "observe_vs_act_classifier")),
        ("can_buy", V4A["summary"]["can_buy_classifier"]["section_key"], V4A["summary"]["can_buy_classifier"]["layer"], _metric(V4A["summary"], "can_buy_classifier")),
        ("can_sell", V4A["summary"]["can_sell_classifier"]["section_key"], V4A["summary"]["can_sell_classifier"]["layer"], _metric(V4A["summary"], "can_sell_classifier")),
    ]
    labels = [name for name, _, _, _ in items]
    values = [value for _, _, _, value in items]
    colors = ["#2b5d8a", "#3c7d5b", "#2f6f86", "#684c8d"]
    bars = ax.bar(labels, values, color=colors)

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Balanced accuracy / AUROC")
    ax.set_title("Best section-local reads in `actionability_algebra_v4`")
    ax.grid(axis="y", alpha=0.2)
    for bar, (_, section, layer, value) in zip(bars, items):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{section}\nL{layer}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()

    path = OUT_DIR / "section_summary.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    outputs = {
        "phase_progression": str(make_phase_progression()),
        "v4_factorization": str(make_v4_factorization()),
        "section_summary": str(make_section_summary()),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
