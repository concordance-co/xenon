from __future__ import annotations

import json
import statistics as stats
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AXIS_RESULTS = Path(
    "data/analysis_results/synthetic_market_axis_decomposition/phase17_market_axis_decomposition_v1/results.json"
)
DISCOVERY_RESULTS = Path(
    "data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/results.json"
)
ASSET_DIR = Path("data/report_assets/synthetic_market_phase17_axis_decomposition")


def _label(text: str, *, width: int = 15) -> str:
    return "\n".join(textwrap.wrap(text.replace("_", " "), width=width))


def _top5_summary(state_payload: dict[str, dict]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    top5_values: list[float] = []
    participation_values: list[float] = []
    for layer_str, layer_data in state_payload.items():
        ev = [float(x) for x in layer_data["explained_variance_ratio"]]
        cumulative = []
        running = 0.0
        for value in ev:
            running += value
            cumulative.append(float(running))
        top5 = float(cumulative[-1]) if cumulative else 0.0
        pr = float(layer_data["participation_ratio_top_components"])
        top5_values.append(top5)
        participation_values.append(pr)
        rows.append({
            "layer": int(layer_str),
            "top5_cumulative": top5,
            "cumulative": cumulative,
            "participation_ratio": pr,
        })
    rows.sort(key=lambda row: int(row["layer"]))
    top_layers = sorted(rows, key=lambda row: float(row["top5_cumulative"]), reverse=True)[:5]
    selected_layers = {}
    for layer in [1, 4, 35, 40, 42]:
        match = next((row for row in rows if int(row["layer"]) == layer), None)
        if match is not None:
            selected_layers[str(layer)] = match
    return {
        "rows": rows,
        "top5_mean": float(stats.mean(top5_values)),
        "top5_min": float(min(top5_values)),
        "top5_max": float(max(top5_values)),
        "participation_mean": float(stats.mean(participation_values)),
        "participation_min": float(min(participation_values)),
        "participation_max": float(max(participation_values)),
        "top_layers": top_layers,
        "selected_layers": selected_layers,
    }


def _build_summary(axis_results: dict, discovery_results: dict) -> dict:
    summary: dict[str, object] = {
        "n_prompts": int(axis_results["n_prompts"]),
        "n_visible_features": int(len(axis_results["visible_feature_names"])),
        "n_nuisance_features": int(len(axis_results["nuisance_feature_names"])),
        "leader": axis_results["targets"]["leader_axis"],
        "dispersion": axis_results["targets"]["dispersion_axis"],
        "subspace": {
            "market_mean": _top5_summary(discovery_results["states"]["market_mean"]),
            "market_eos": _top5_summary(discovery_results["states"]["market_eos"]),
        },
    }
    return summary


def _plot_axis_panels(summary: dict) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    for row_idx, axis_key in enumerate(["leader", "dispersion"]):
        payload = summary[axis_key]

        top_single = payload["top_single_features"][:8]
        ax = axes[row_idx, 0]
        labels = [_label(row["feature"]) for row in top_single]
        values = [float(row["cv_r2"]) for row in top_single]
        ax.barh(labels[::-1], values[::-1], color="#b33a2a")
        ax.set_title(f"{axis_key.replace('_', ' ').title()}: top single features")
        ax.set_xlabel("Cross-validated $R^2$")

        ax = axes[row_idx, 1]
        family_rows = payload["metric_family_group_ridge_cv_r2"][:7]
        labels = [_label(row["group"]) for row in family_rows]
        values = [float(row["cv_r2"]) for row in family_rows]
        ax.bar(labels, values, color="#4f6d7a")
        ax.set_title(f"{axis_key.replace('_', ' ').title()}: metric-family fits")
        ax.set_ylabel("Ridge CV $R^2$")
        ax.tick_params(axis="x", labelsize=8)

        ax = axes[row_idx, 2]
        aggregate_rows = payload["aggregate_group_ridge_cv_r2"][:8]
        labels = [_label(row["group"], width=12) for row in aggregate_rows]
        values = [float(row["cv_r2"]) for row in aggregate_rows]
        ax.bar(labels, values, color="#c87533")
        ax.set_title(f"{axis_key.replace('_', ' ').title()}: aggregate-type fits")
        ax.set_ylabel("Ridge CV $R^2$")
        ax.tick_params(axis="x", labelsize=8)

    fig.savefig(ASSET_DIR / "phase17_axis_panels.png", dpi=220)
    plt.close(fig)


def _plot_subspace_summary(summary: dict) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for col_idx, state_key in enumerate(["market_mean", "market_eos"]):
        payload = summary["subspace"][state_key]
        layers = [int(row["layer"]) for row in payload["rows"]]
        top5 = [float(row["top5_cumulative"]) for row in payload["rows"]]
        pr = [float(row["participation_ratio"]) for row in payload["rows"]]

        ax = axes[0, col_idx]
        ax.plot(layers, top5, color="#b33a2a" if state_key == "market_mean" else "#4f6d7a", lw=2.2)
        ax.set_title(f"{state_key}: top-5 cumulative variance")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Top-5 variance")
        for layer in [1, 4, 35, 40, 42]:
            if str(layer) in payload["selected_layers"]:
                row = payload["selected_layers"][str(layer)]
                ax.scatter([layer], [row["top5_cumulative"]], color="black", s=16, zorder=3)

        ax = axes[1, col_idx]
        ax.plot(layers, pr, color="#c87533" if state_key == "market_mean" else "#89a0b0", lw=2.2)
        ax.set_title(f"{state_key}: participation ratio of top 5 PCs")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Participation ratio")
        for layer in [1, 4, 35, 40, 42]:
            if str(layer) in payload["selected_layers"]:
                row = payload["selected_layers"][str(layer)]
                ax.scatter([layer], [row["participation_ratio"]], color="black", s=16, zorder=3)

    fig.savefig(ASSET_DIR / "phase17_subspace_summary.png", dpi=220)
    plt.close(fig)


def build_assets() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    axis_results = json.loads(AXIS_RESULTS.read_text())
    discovery_results = json.loads(DISCOVERY_RESULTS.read_text())
    summary = _build_summary(axis_results, discovery_results)
    (ASSET_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot_axis_panels(summary)
    _plot_subspace_summary(summary)


if __name__ == "__main__":
    build_assets()
