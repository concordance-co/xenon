from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from projects.DX_TERMINAL.prompt_confusion.paths import prompt_confusion_root

ROOT = prompt_confusion_root(__file__)
OUT_DIR = ROOT / "phase_12" / "reports" / "dx_terminal_brief_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_layers(path: Path):
    data = json.loads(path.read_text())
    return sorted(data["layers"], key=lambda row: row["layer"])


def series_from_result(path: Path, label: str):
    rows = load_layers(path)
    return {
        "label": label,
        "x": [row["layer"] for row in rows],
        "y": [row["auroc"] for row in rows],
    }


def make_line_chart(series_list, title: str, subtitle: str, out_name: str, ylim=(0.48, 1.01)):
    plt.figure(figsize=(8.6, 4.8))
    colors = ["#295E9E", "#AF3E2F", "#2E7D55", "#7A4FB3", "#CC7A00"]
    for color, series in zip(colors, series_list):
        plt.plot(series["x"], series["y"], marker="o", linewidth=2.2, markersize=4.5, label=series["label"], color=color)

    plt.title(title, fontsize=14, weight="bold")
    plt.suptitle(subtitle, y=0.94, fontsize=9, color="#5A6772")
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.ylim(*ylim)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    representative_family_curves = [
        series_from_result(
            ROOT / "phase_09" / "reports" / "marshalls_battery" / "results" / "probe_conflict_strict_combined_trade_size.json",
            "trade_size",
        ),
        series_from_result(
            ROOT / "phase_10" / "reports" / "pipelines_v2" / "report_e38adf78048f_09ef704a" / "results" / "conflict_probe_strategy_holdout_results.json",
            "risk_preference",
        ),
        series_from_result(
            ROOT / "phase_12" / "reports" / "pipelines_v2" / "report_27dde9ff9c93_8aa7cdc8" / "results" / "conflict_probe_strategy_holdout_results.json",
            "diversification_preference",
        ),
    ]
    make_line_chart(
        representative_family_curves,
        "Representative Within-Family AUROC By Layer",
        "trade_size (strict combined), risk_preference (strategy holdout), diversification_preference (strategy holdout)",
        "family_within_auroc_by_layer.png",
    )

    strict_family_curves = [
        series_from_result(
            ROOT / "phase_09" / "reports" / "marshalls_battery" / "results" / "probe_conflict_strict_combined_trade_size.json",
            "trade_size strict",
        ),
        series_from_result(
            ROOT / "phase_10" / "reports" / "marshalls_strict" / "report_dc1f818293bf_ef5e1312" / "results" / "probe_conflict_strict_combined_holdout_results.json",
            "risk_preference strict",
        ),
        series_from_result(
            ROOT / "phase_12" / "reports" / "marshalls_strict" / "report_2e138f7be318_bd39d1fb" / "results" / "probe_conflict_strict_combined_holdout_results.json",
            "diversification_preference strict",
        ),
    ]
    make_line_chart(
        strict_family_curves,
        "Strict Holdout AUROC By Layer",
        "Harder lexical novelty remains well above chance across the validated family set",
        "strict_family_auroc_by_layer.png",
    )

    joint_prompt_curves = [
        series_from_result(
            ROOT / "phase_11" / "reports" / "pipelines_v2" / "report_6a8a0904734f_4ddbff0e" / "results" / "size_conflict_probe_results.json",
            "size_conflict_present",
        ),
        series_from_result(
            ROOT / "phase_11" / "reports" / "pipelines_v2" / "report_6a8a0904734f_4ddbff0e" / "results" / "risk_conflict_probe_results.json",
            "risk_conflict_present",
        ),
        series_from_result(
            ROOT / "phase_11" / "reports" / "pipelines_v2" / "report_6a8a0904734f_4ddbff0e" / "results" / "any_conflict_probe_results.json",
            "any_conflict_present",
        ),
        series_from_result(
            ROOT / "phase_11" / "reports" / "pipelines_v2" / "report_6a8a0904734f_4ddbff0e" / "results" / "double_conflict_probe_results.json",
            "double_conflict_present",
        ),
    ]
    make_line_chart(
        joint_prompt_curves,
        "Phase 11 Joint-Prompt AUROC By Layer",
        "Multiple conflict readouts stay strong in the same forward pass",
        "phase11_joint_prompt_auroc_by_layer.png",
    )


if __name__ == "__main__":
    main()
