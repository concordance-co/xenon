"""Generate combined report assets for the Phase 16 + 17 unified report.

Pulls from:
- Phase 16 context-order analysis results
- Phase 16 cross-basis results
- Phase 17 axis decomposition results
- Phase 15 residualized discovery results (for subspace summary)

Outputs to: data/report_assets/synthetic_market_phase16_17_combined/
"""
from __future__ import annotations

import json
import statistics as stats
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Data paths ──────────────────────────────────────────────────────
PHASE16_RESULTS = Path(
    "data/analysis_results/synthetic_market_context_order/phase16_context_order_v1/results.json"
)
PHASE16_CROSS_RESULTS = Path(
    "data/analysis_results/synthetic_market_context_order/phase16_context_order_v1/market_mean_cross_basis_v1_results.json"
)
PHASE17_AXIS_RESULTS = Path(
    "data/analysis_results/synthetic_market_axis_decomposition/phase17_market_axis_decomposition_v1/results.json"
)
DISCOVERY_RESULTS = Path(
    "data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/results.json"
)
ASSET_DIR = Path("data/report_assets/synthetic_market_phase16_17_combined")

# ── Colors ──────────────────────────────────────────────────────────
C_RED = "#b33a2a"
C_AMBER = "#c87533"
C_TEAL = "#4f6d7a"
C_OLIVE = "#7a8b5b"
C_SLATE = "#89a0b0"
C_DARK = "#222222"


def _label(text: str, *, width: int = 15) -> str:
    return "\n".join(textwrap.wrap(text.replace("_", " "), width=width))


def _sorted_layers(payload: dict[str, dict]) -> list[int]:
    return sorted(int(k) for k in payload.keys())


def _series(payload: dict[str, dict], key: str) -> tuple[list[int], list[float]]:
    layers = _sorted_layers(payload)
    return layers, [float(payload[str(l)][key]) for l in layers]


def _best_gap_layer(payload: dict[str, dict]) -> tuple[int, dict]:
    layer, row = max(payload.items(), key=lambda kv: kv[1]["perception_gap"])
    return int(layer), row


# ── Chart 1: Methodology overview ───────────────────────────────────
def plot_methodology_overview(p16: dict, p17: dict) -> None:
    """Three-panel overview: discovery → decomposition → context-order."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(14, 4.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel 1: Discovery basis — show top-5 cumulative variance for market_mean
    ax = fig.add_subplot(gs[0])
    disc = json.loads(DISCOVERY_RESULTS.read_text())
    mm_state = disc["states"]["market_mean"]
    layers_all = sorted(int(k) for k in mm_state.keys())
    ev_ratios = []
    for l in layers_all:
        ev = [float(x) for x in mm_state[str(l)]["explained_variance_ratio"]]
        ev_ratios.append(sum(ev))
    ax.plot(layers_all, ev_ratios, color=C_RED, lw=2.2)
    ax.set_title("Step 1: Discovery Basis\n(Phase 15 residualized PCA)", fontsize=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Top-5 cumulative variance")
    ax.scatter([4, 35], [ev_ratios[layers_all.index(4)], ev_ratios[layers_all.index(35)]],
               color=C_DARK, s=30, zorder=4)
    ax.annotate("L4: leader", (4, ev_ratios[layers_all.index(4)]),
                textcoords="offset points", xytext=(8, -14), fontsize=8)
    ax.annotate("L35: dispersion", (35, ev_ratios[layers_all.index(35)]),
                textcoords="offset points", xytext=(-60, 10), fontsize=8)

    # Panel 2: Axis decomposition — top single features for leader and dispersion
    ax = fig.add_subplot(gs[1])
    leader = p17["targets"]["leader_axis"]
    disp = p17["targets"]["dispersion_axis"]
    leader_top = leader["top_single_features"][:5]
    disp_top = disp["top_single_features"][:5]
    labels_l = [_label(r["feature"], width=18) for r in leader_top]
    vals_l = [float(r["cv_r2"]) for r in leader_top]
    labels_d = [_label(r["feature"], width=18) for r in disp_top]
    vals_d = [float(r["cv_r2"]) for r in disp_top]
    y_pos = np.arange(5)
    width = 0.38
    ax.barh(y_pos + width / 2, vals_l[::-1], width, color=C_RED, label="Leader axis")
    ax.barh(y_pos - width / 2, vals_d[::-1], width, color=C_TEAL, label="Dispersion axis")
    combined_labels = []
    for i in range(5):
        combined_labels.append(f"{labels_l[4 - i]}  /  {labels_d[4 - i]}")
    ax.set_yticks(y_pos, [f"#{5-i}" for i in range(5)], fontsize=8)
    ax.set_title("Step 2: Axis Decomposition\n(Phase 17 feature readout)", fontsize=10)
    ax.set_xlabel("Cross-validated R²")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # Panel 3: Context-order — A vs B, A vs C perception curves for risk market_eos
    ax = fig.add_subplot(gs[2])
    groups = p16["groups"]
    risk_eos = groups["risk"]["state_results"]["market_eos"]
    layers, ab_cos = _series(risk_eos, "ab_cosine_mean")
    _, ac_cos = _series(risk_eos, "ac_cosine_mean")
    ax.plot(layers, ab_cos, color=C_AMBER, lw=2.3, label="A vs B (after)")
    ax.plot(layers, ac_cos, color=C_RED, lw=2.3, label="A vs C (before)")
    best_layer, best_row = _best_gap_layer(risk_eos)
    ax.scatter([best_layer], [best_row["ac_cosine_mean"]], color=C_DARK, s=28, zorder=4)
    ax.annotate(f"L{best_layer}\nGap {best_row['perception_gap']:.3f}",
                (best_layer, best_row["ac_cosine_mean"]),
                textcoords="offset points", xytext=(6, -22), fontsize=8)
    ax.set_title("Step 3: Context-Order Test\n(Phase 16 perception warp)", fontsize=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean cosine")
    ax.set_ylim(0.92, 1.001)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    fig.savefig(ASSET_DIR / "combined_methodology_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# ── Chart 2: Axis decomposition panels (from Phase 17) ─────────────
def plot_axis_decomposition(p17: dict) -> None:
    """2x3 panel: single features, metric families, aggregate types for leader & dispersion."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    for row_idx, axis_key in enumerate(["leader_axis", "dispersion_axis"]):
        payload = p17["targets"][axis_key]
        display_name = axis_key.replace("_axis", "").title()

        # Single features
        top_single = payload["top_single_features"][:8]
        ax = axes[row_idx, 0]
        labels = [_label(r["feature"]) for r in top_single]
        values = [float(r["cv_r2"]) for r in top_single]
        ax.barh(labels[::-1], values[::-1], color=C_RED)
        ax.set_title(f"{display_name}: top single features")
        ax.set_xlabel("Cross-validated R²")

        # Metric families
        ax = axes[row_idx, 1]
        family_rows = payload["metric_family_group_ridge_cv_r2"][:7]
        labels = [_label(r["group"]) for r in family_rows]
        values = [float(r["cv_r2"]) for r in family_rows]
        ax.bar(labels, values, color=C_TEAL)
        ax.set_title(f"{display_name}: metric-family fits")
        ax.set_ylabel("Ridge CV R²")
        ax.tick_params(axis="x", labelsize=8)

        # Aggregate types
        ax = axes[row_idx, 2]
        agg_rows = payload["aggregate_group_ridge_cv_r2"][:8]
        labels = [_label(r["group"], width=12) for r in agg_rows]
        values = [float(r["cv_r2"]) for r in agg_rows]
        ax.bar(labels, values, color=C_AMBER)
        ax.set_title(f"{display_name}: aggregate-type fits")
        ax.set_ylabel("Ridge CV R²")
        ax.tick_params(axis="x", labelsize=8)

    fig.savefig(ASSET_DIR / "combined_axis_decomposition.png", dpi=220)
    plt.close(fig)


# ── Chart 3: Perception curves (from Phase 16) ─────────────────────
def plot_perception_curves(p16: dict) -> None:
    """2x2 panel: risk/affordance × market_mean/market_eos perception curves."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)

    specs = [
        ("risk", "market_mean", "Risk: market_mean"),
        ("risk", "market_eos", "Risk: corrected market_eos"),
        ("affordance", "market_mean", "Affordance: market_mean"),
        ("affordance", "market_eos", "Affordance: corrected market_eos"),
    ]
    for ax, (group, state, title) in zip(axes.flat, specs, strict=True):
        payload = p16["groups"][group]["state_results"][state]
        layers, ab_cos = _series(payload, "ab_cosine_mean")
        _, ac_cos = _series(payload, "ac_cosine_mean")
        ax.plot(layers, ab_cos, color=C_AMBER, lw=2.3, label="A vs B (context after market)")
        ax.plot(layers, ac_cos, color=C_RED, lw=2.3, label="A vs C (context before market)")
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean cosine")
        ax.set_ylim(0.92, 1.001)
        if state == "market_eos":
            best_layer, best_row = _best_gap_layer(payload)
            ax.scatter([best_layer], [best_row["ac_cosine_mean"]], color=C_DARK, s=28, zorder=4)
            ax.annotate(
                f"L{best_layer}\nGap {best_row['perception_gap']:.3f}",
                (best_layer, best_row["ac_cosine_mean"]),
                textcoords="offset points", xytext=(6, -22), fontsize=8,
            )
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(ASSET_DIR / "combined_perception_curves.png", dpi=220)
    plt.close(fig)


# ── Chart 4: Basis shift + cross-basis (from Phase 16) ─────────────
def plot_basis_shift(p16: dict, p16_cross: dict) -> None:
    """2x2 panel: native eos basis vs market_mean basis for risk and affordance."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), constrained_layout=True)

    for row_idx, group_name in enumerate(["risk", "affordance"]):
        native_payload = p16["groups"][group_name]["state_results"]["market_eos"]
        cross_payload = p16_cross["groups"][group_name]["state_results"]["market_eos"]
        best_layer, native_row = _best_gap_layer(native_payload)
        cross_row = cross_payload[str(best_layer)]["cross_basis_projection"]

        for col_idx, (title, payload_row) in enumerate([
            ("Phase 15 market_eos basis", native_row),
            ("Phase 15 market_mean basis", cross_row),
        ]):
            ax = axes[row_idx, col_idx]
            shifts = payload_row["pc_shift"]
            features = [r["feature"] or f"PC {i + 1}" for i, r in enumerate(shifts)]
            ab_vals = [r["ab_abs_mean"] for r in shifts]
            ac_vals = [r["ac_abs_mean"] for r in shifts]
            xpos = np.arange(len(features))
            width = 0.34
            ax.bar(xpos - width / 2, ab_vals, width, color=C_AMBER, label="A → B")
            ax.bar(xpos + width / 2, ac_vals, width, color=C_RED, label="A → C")
            ax.set_xticks(xpos, [f.replace("_", "\n") for f in features])
            ax.set_ylabel("Mean abs shift")
            ax.set_title(f"{group_name.title()} L{best_layer}: {title}")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(ASSET_DIR / "combined_basis_shift.png", dpi=220)
    plt.close(fig)


# ── Chart 5: Downstream integration (from Phase 16) ────────────────
def plot_integration(p16: dict) -> None:
    """1x2 panel: B vs C downstream convergence for risk and affordance."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)

    states = [
        ("last_token", "Last token", C_RED),
        ("active_settings_eos", "Settings EOS", C_AMBER),
        ("portfolio_eos", "Portfolio EOS", C_TEAL),
        ("constraints_eos", "Constraints EOS", C_OLIVE),
    ]
    for ax, group_name in zip(axes, ["risk", "affordance"], strict=True):
        for state_key, label, color in states:
            payload = p16["groups"][group_name]["integration_results"][state_key]
            layers, bc_cos = _series(payload, "bc_cosine_mean")
            ax.plot(layers, bc_cos, lw=2.0, color=color, label=label)
        ax.set_title(f"{group_name.title()}: B vs C downstream convergence")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean cosine")
        ax.set_ylim(0.92, 1.001)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(ASSET_DIR / "combined_integration_curves.png", dpi=220)
    plt.close(fig)


# ── Chart 6: Subspace dimensionality (from Phase 15 discovery) ─────
def plot_subspace_summary(discovery: dict) -> None:
    """2x2 panel: top-5 cumulative variance and participation ratio for both states."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for col_idx, (state_key, color_top, color_pr) in enumerate([
        ("market_mean", C_RED, C_AMBER),
        ("market_eos", C_TEAL, C_SLATE),
    ]):
        state_data = discovery["states"][state_key]
        layers = sorted(int(k) for k in state_data.keys())
        top5 = []
        pr_vals = []
        for l in layers:
            ev = [float(x) for x in state_data[str(l)]["explained_variance_ratio"]]
            top5.append(sum(ev))
            pr_vals.append(float(state_data[str(l)]["participation_ratio_top_components"]))

        ax = axes[0, col_idx]
        ax.plot(layers, top5, color=color_top, lw=2.2)
        ax.set_title(f"{state_key}: top-5 cumulative variance")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Top-5 variance")
        for mark_l in [1, 4, 35, 40, 42]:
            if mark_l in layers:
                idx = layers.index(mark_l)
                ax.scatter([mark_l], [top5[idx]], color="black", s=16, zorder=3)

        ax = axes[1, col_idx]
        ax.plot(layers, pr_vals, color=color_pr, lw=2.2)
        ax.set_title(f"{state_key}: participation ratio of top 5 PCs")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Participation ratio")
        for mark_l in [1, 4, 35, 40, 42]:
            if mark_l in layers:
                idx = layers.index(mark_l)
                ax.scatter([mark_l], [pr_vals[idx]], color="black", s=16, zorder=3)

    fig.savefig(ASSET_DIR / "combined_subspace_summary.png", dpi=220)
    plt.close(fig)


# ── Build summary JSON for typst data binding ──────────────────────
def build_summary_json(p16: dict, p16_cross: dict, p17: dict, discovery: dict) -> dict:
    """Build a combined summary JSON for the typst report to consume."""
    leader = p17["targets"]["leader_axis"]
    dispersion = p17["targets"]["dispersion_axis"]

    # Phase 16 key numbers
    risk_eos = p16["groups"]["risk"]["state_results"]["market_eos"]
    aff_eos = p16["groups"]["affordance"]["state_results"]["market_eos"]
    risk_best_layer, risk_best = _best_gap_layer(risk_eos)
    aff_best_layer, aff_best = _best_gap_layer(aff_eos)

    # Subspace summary
    def _top5_summary(state_data: dict) -> dict:
        top5_values = []
        pr_values = []
        for layer_str, ld in state_data.items():
            ev = [float(x) for x in ld["explained_variance_ratio"]]
            top5_values.append(sum(ev))
            pr_values.append(float(ld["participation_ratio_top_components"]))
        return {
            "top5_mean": round(stats.mean(top5_values), 3),
            "top5_max": round(max(top5_values), 3),
            "participation_mean": round(stats.mean(pr_values), 3),
        }

    return {
        "leader": {
            "best_single_feature": leader["best_single_feature"]["feature"],
            "best_single_cv_r2": round(leader["best_single_feature"]["cv_r2"], 3),
            "best_pair_features": leader["best_pair_quadratic"]["features"],
            "best_pair_cv_r2": round(leader["best_pair_quadratic"]["cv_r2"], 3),
        },
        "dispersion": {
            "best_single_feature": dispersion["best_single_feature"]["feature"],
            "best_single_cv_r2": round(dispersion["best_single_feature"]["cv_r2"], 3),
            "best_pair_features": dispersion["best_pair_quadratic"]["features"],
            "best_pair_cv_r2": round(dispersion["best_pair_quadratic"]["cv_r2"], 3),
        },
        "context_order": {
            "risk_best_layer": risk_best_layer,
            "risk_ab_cosine": round(risk_best["ab_cosine_mean"], 3),
            "risk_ac_cosine": round(risk_best["ac_cosine_mean"], 3),
            "risk_gap": round(risk_best["perception_gap"], 3),
            "aff_best_layer": aff_best_layer,
            "aff_ab_cosine": round(aff_best["ab_cosine_mean"], 3),
            "aff_ac_cosine": round(aff_best["ac_cosine_mean"], 3),
            "aff_gap": round(aff_best["perception_gap"], 3),
        },
        "subspace": {
            "market_mean": _top5_summary(discovery["states"]["market_mean"]),
            "market_eos": _top5_summary(discovery["states"]["market_eos"]),
        },
    }


# ── Main ────────────────────────────────────────────────────────────
def build_assets() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    p16 = json.loads(PHASE16_RESULTS.read_text())
    p16_cross = json.loads(PHASE16_CROSS_RESULTS.read_text())
    p17 = json.loads(PHASE17_AXIS_RESULTS.read_text())
    discovery = json.loads(DISCOVERY_RESULTS.read_text())

    summary = build_summary_json(p16, p16_cross, p17, discovery)
    (ASSET_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_methodology_overview(p16, p17)
    plot_axis_decomposition(p17)
    plot_perception_curves(p16)
    plot_basis_shift(p16, p16_cross)
    plot_integration(p16)
    plot_subspace_summary(discovery)

    print(f"Assets written to {ASSET_DIR}")


if __name__ == "__main__":
    build_assets()
