from __future__ import annotations

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.synthetic_market.synthetic_market_behavior_analysis import (
    SyntheticMarketBehaviorAnalysisConfig,
    run_synthetic_market_behavior_analysis,
)

ASSET_DIR = ROOT / "data" / "report_assets" / "synthetic_market_phase20_paired_robustness"

RUNS = {
    "leader": {
        "label": "Leader",
        "pair_metric": "vol_1h_max",
        "target_layer": 4,
        "results": Path("/tmp/phase20_market_behavior_leader_l4top4_v1_results.json"),
        "metadata": Path("/tmp/phase20_market_behavior_leader_l4top4_v1_metadata.parquet"),
    },
    "dispersion": {
        "label": "Dispersion",
        "pair_metric": "pct_1h_mad",
        "target_layer": 35,
        "results": Path("/tmp/phase20_market_behavior_dispersion_l35top4_v1_results.json"),
        "metadata": Path("/tmp/phase20_market_behavior_dispersion_l35top4_v1_metadata.parquet"),
    },
}

PAIR_MODES = ("denoise", "noise")
LAMBDAS = ("0.5", "1.0", "1.5")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _min(values: list[float]) -> float | None:
    if not values:
        return None
    return float(min(values))


def _max(values: list[float]) -> float | None:
    if not values:
        return None
    return float(max(values))


def _parse_patch_stats(raw: Any) -> list[dict[str, Any]]:
    if raw in (None, "", "{}"):
        return []
    payload = raw
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if isinstance(payload, dict):
        return [value for value in payload.values() if isinstance(value, dict)]
    if isinstance(payload, list):
        return [value for value in payload if isinstance(value, dict)]
    return []


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path / "metadata.parquet")


def _analyze_cell_pair(
    baseline_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        baseline_dir = root / "baseline"
        intervention_dir = root / "intervention"
        output_dir = root / "output"
        _write_rows(baseline_dir, baseline_rows)
        _write_rows(intervention_dir, intervention_rows)
        return run_synthetic_market_behavior_analysis(
            SyntheticMarketBehaviorAnalysisConfig(
                baseline_dir=baseline_dir,
                intervention_dir=intervention_dir,
                output_dir=output_dir,
                bootstrap_samples=400,
            )
        )


def _aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "tool_token_change_rate",
        "tool_name_change_rate",
        "mean_generated_token_count_delta",
        "mean_tool_spend_pct_delta",
        "mean_patch_delta_norm_std",
        "mean_selected_proj_norm_before",
        "mean_patch_mean_norm_ratio",
        "mean_patch_mean_std_norm_ratio",
        "patch_applied_rate",
        "patch_skipped_rate",
    ]
    out: dict[str, Any] = {"n": len(results)}
    for metric in metrics:
        values = [float(result[metric]) for result in results if result.get(metric) is not None]
        out[metric] = _mean(values)
        out[f"{metric}_min"] = _min(values)
        out[f"{metric}_max"] = _max(values)
    return out


def _cell_kind(cell_id: str) -> str:
    if "project_out" in cell_id:
        return "project_out"
    if "random_control" in cell_id:
        return "random_control"
    if "baseline" in cell_id:
        return "baseline"
    raise ValueError(f"Unknown cell kind for {cell_id}")


def _cell_pair_mode(cell_id: str) -> str:
    if "_denoise" in cell_id:
        return "denoise"
    if "_noise" in cell_id:
        return "noise"
    raise ValueError(f"Unknown pair mode for {cell_id}")


def _cell_lambda(cell_id: str) -> str | None:
    if "lam_0p5" in cell_id:
        return "0.5"
    if "lam_1p5" in cell_id:
        return "1.5"
    if "lam_1" in cell_id:
        return "1.0"
    return None


def _patch_health(rows: list[dict[str, Any]]) -> dict[str, Any]:
    patch_entries: list[dict[str, Any]] = []
    for row in rows:
        patch_entries.extend(_parse_patch_stats(row.get("patch_stats_json")))
    applied = sum(1 for entry in patch_entries if entry.get("status") != "skipped")
    skipped = sum(1 for entry in patch_entries if entry.get("status") == "skipped")
    coverage = [
        float(entry["coverage_fraction"])
        for entry in patch_entries
        if entry.get("coverage_fraction") is not None
    ]
    mean_std_ratio = [
        float(entry["mean_std_norm_after"]) / float(entry["mean_std_norm_before"])
        for entry in patch_entries
        if entry.get("mean_std_norm_before") not in (None, 0, 0.0)
        and entry.get("mean_std_norm_after") is not None
    ]
    mean_norm_ratio = [
        float(entry["mean_norm_after"]) / float(entry["mean_norm_before"])
        for entry in patch_entries
        if entry.get("mean_norm_before") not in (None, 0, 0.0)
        and entry.get("mean_norm_after") is not None
    ]
    return {
        "patch_entries": len(patch_entries),
        "patch_applied_rate": float(applied / len(patch_entries)) if patch_entries else None,
        "patch_skipped_rate": float(skipped / len(patch_entries)) if patch_entries else None,
        "mean_coverage_fraction": _mean(coverage),
        "mean_patch_mean_std_norm_ratio": _mean(mean_std_ratio),
        "mean_patch_mean_norm_ratio": _mean(mean_norm_ratio),
    }


def _plot_metric_curves(summary: dict[str, Any], *, metric: str, ylabel: str, filename: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.8), sharex=True)
    hypothesis_order = ["leader", "dispersion"]
    pair_mode_order = ["denoise", "noise"]
    colors = {"project_out": "#b33a2a", "random_control": "#5a5a5a"}
    xvals = [0.5, 1.0, 1.5]

    for row_idx, hypothesis in enumerate(hypothesis_order):
        for col_idx, pair_mode in enumerate(pair_mode_order):
            ax = axes[row_idx][col_idx]
            hypothesis_data = summary["hypotheses"][hypothesis]["pair_modes"][pair_mode]
            project_series = []
            control_mean_series = []
            control_min_series = []
            control_max_series = []
            for lam in LAMBDAS:
                project_series.append(float(hypothesis_data["project_out"][lam][metric]))
                control_mean_series.append(float(hypothesis_data["random_control"][lam][metric]))
                control_min_series.append(float(hypothesis_data["random_control"][lam][f"{metric}_min"]))
                control_max_series.append(float(hypothesis_data["random_control"][lam][f"{metric}_max"]))
            ax.plot(xvals, project_series, marker="o", color=colors["project_out"], label="project_out")
            ax.plot(
                xvals,
                control_mean_series,
                marker="o",
                color=colors["random_control"],
                linestyle="--",
                label="random_control mean",
            )
            ax.fill_between(xvals, control_min_series, control_max_series, color="#bdbdbd", alpha=0.25)
            ax.set_title(f"{summary['hypotheses'][hypothesis]['label']} · {pair_mode}", fontsize=9)
            ax.set_xticks(xvals, ["0.5", "1.0", "1.5"])
            ax.tick_params(axis="both", labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_lambda_one_gaps(summary: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.2))
    order = [
        ("leader", "denoise"),
        ("leader", "noise"),
        ("dispersion", "denoise"),
        ("dispersion", "noise"),
    ]
    labels = ["Leader denoise", "Leader noise", "Dispersion denoise", "Dispersion noise"]
    tool_vals = []
    gen_vals = []
    for hypothesis, pair_mode in order:
        headline = summary["hypotheses"][hypothesis]["pair_modes"][pair_mode]["headline_lambda_1"]
        tool_vals.append(float(headline["tool_token_selectivity_gap"]))
        gen_vals.append(float(headline["generated_token_delta_selectivity_gap"]) / 1000.0)
    xs = list(range(len(order)))
    width = 0.34
    ax.bar([x - width / 2 for x in xs], tool_vals, width=width, color="#b33a2a", label="tool-token gap")
    ax.bar([x + width / 2 for x in xs], gen_vals, width=width, color="#5a5a5a", label="token-delta gap / 1000")
    ax.axhline(0.0, color="#777", linewidth=0.8)
    ax.set_xticks(xs, labels)
    ax.tick_params(axis="x", labelrotation=18, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("Random-control minus project-out")
    ax.set_title("Selectivity at λ = 1.0")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "phase20_lambda1_selectivity_gaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    hypotheses: dict[str, Any] = {}
    per_cell_results: dict[str, Any] = {}
    lambda_one_rows: list[dict[str, Any]] = []
    overall_tool_gap_wins = 0
    overall_gen_gap_wins = 0
    total_gap_comparisons = 0
    source_metrics_available = False

    for hypothesis_key, spec in RUNS.items():
        rows = _load_parquet(spec["metadata"])
        by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_cell[str(row["matrix_cell_id"])].append(row)

        baseline_cells = {
            pair_mode: next(cell_id for cell_id in by_cell if f"baseline_{pair_mode}" in cell_id)
            for pair_mode in PAIR_MODES
        }

        pair_mode_summary: dict[str, Any] = {}
        for pair_mode in PAIR_MODES:
            baseline_rows = by_cell[baseline_cells[pair_mode]]
            project_cells = sorted(
                cell_id for cell_id in by_cell if "project_out" in cell_id and f"_{pair_mode}_" in cell_id
            )
            random_cells = sorted(
                cell_id for cell_id in by_cell if "random_control" in cell_id and f"_{pair_mode}_" in cell_id
            )

            project_results: dict[str, Any] = {}
            random_results_by_lambda: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for cell_id in project_cells + random_cells:
                result = _analyze_cell_pair(baseline_rows, by_cell[cell_id])
                per_cell_results[cell_id] = result
                source_metrics_available = source_metrics_available or (
                    result.get("source_tool_token_restoration_rate") is not None
                    or result.get("source_tool_name_restoration_rate") is not None
                )
                lam = _cell_lambda(cell_id)
                assert lam is not None
                if "project_out" in cell_id:
                    project_results[lam] = result
                else:
                    random_results_by_lambda[lam].append(result)

            random_aggregates = {
                lam: _aggregate_results(random_results_by_lambda[lam]) for lam in LAMBDAS
            }
            project_trimmed = {
                lam: {
                    key: value
                    for key, value in project_results[lam].items()
                    if key
                    in {
                        "tool_token_change_rate",
                        "tool_name_change_rate",
                        "mean_generated_token_count_delta",
                        "mean_tool_spend_pct_delta",
                        "mean_patch_delta_norm_std",
                        "mean_selected_proj_norm_before",
                        "mean_patch_mean_norm_ratio",
                        "mean_patch_mean_std_norm_ratio",
                        "patch_applied_rate",
                        "patch_skipped_rate",
                        "rows_with_patch_stats",
                    }
                }
                for lam in LAMBDAS
            }

            lambda_one = {
                "tool_token_selectivity_gap": (
                    float(random_aggregates["1.0"]["tool_token_change_rate"])
                    - float(project_trimmed["1.0"]["tool_token_change_rate"])
                ),
                "tool_name_selectivity_gap": (
                    float(random_aggregates["1.0"]["tool_name_change_rate"])
                    - float(project_trimmed["1.0"]["tool_name_change_rate"])
                ),
                "generated_token_delta_selectivity_gap": (
                    float(random_aggregates["1.0"]["mean_generated_token_count_delta"])
                    - float(project_trimmed["1.0"]["mean_generated_token_count_delta"])
                ),
            }
            lambda_one_rows.append(
                {
                    "hypothesis": spec["label"],
                    "pair_mode": pair_mode,
                    "project_tool_token_change_rate": float(project_trimmed["1.0"]["tool_token_change_rate"]),
                    "control_tool_token_change_rate": float(random_aggregates["1.0"]["tool_token_change_rate"]),
                    "project_generated_token_delta": float(project_trimmed["1.0"]["mean_generated_token_count_delta"]),
                    "control_generated_token_delta": float(random_aggregates["1.0"]["mean_generated_token_count_delta"]),
                    **lambda_one,
                }
            )

            for lam in LAMBDAS:
                total_gap_comparisons += 1
                if float(project_trimmed[lam]["tool_token_change_rate"]) < float(
                    random_aggregates[lam]["tool_token_change_rate"]
                ):
                    overall_tool_gap_wins += 1
                if float(project_trimmed[lam]["mean_generated_token_count_delta"]) < float(
                    random_aggregates[lam]["mean_generated_token_count_delta"]
                ):
                    overall_gen_gap_wins += 1

            pair_mode_summary[pair_mode] = {
                "baseline_cell_id": baseline_cells[pair_mode],
                "project_out": project_trimmed,
                "random_control": random_aggregates,
                "headline_lambda_1": lambda_one,
            }

        hypotheses[hypothesis_key] = {
            "label": spec["label"],
            "pair_metric": spec["pair_metric"],
            "target_layer": spec["target_layer"],
            "result_summary": _load_json(spec["results"]),
            "patch": _patch_health(rows),
            "pair_modes": pair_mode_summary,
        }

    overall_patch_entries = sum(int(hypotheses[key]["patch"]["patch_entries"]) for key in hypotheses)
    overall_applied = sum(
        float(hypotheses[key]["patch"]["patch_applied_rate"]) * float(hypotheses[key]["patch"]["patch_entries"])
        for key in hypotheses
    )
    overall_skipped = sum(
        float(hypotheses[key]["patch"]["patch_skipped_rate"]) * float(hypotheses[key]["patch"]["patch_entries"])
        for key in hypotheses
    )

    methodology_scorecard = [
        {"item": "Matched paired prompt cohorts (denoise + noise)", "status": "meets"},
        {"item": "Lambda sweep over intervention strength", "status": "meets"},
        {"item": "Matched orthogonal random-direction controls", "status": "meets"},
        {"item": "Exact span targeting with per-row patch diagnostics", "status": "meets"},
        {
            "item": "Source-behavior restoration metrics from generated source outputs",
            "status": "partial" if source_metrics_available else "missing",
        },
        {"item": "Neighboring-component / neighboring-PC sweep", "status": "missing"},
        {"item": "Subspace-size sweep", "status": "missing"},
        {"item": "Path-level validation", "status": "missing"},
        {"item": "Norm/distribution diagnostics", "status": "partial"},
    ]

    summary = {
        "date": "28 March 2026",
        "model": "Qwen/Qwen3-30B-A3B",
        "phase_name": "phase20_paired_robustness",
        "sample": {
            "hypothesis_count": len(hypotheses),
            "rows_per_cell": 32,
            "cell_count_per_hypothesis": 20,
            "pair_modes": list(PAIR_MODES),
            "lambda_sweep": [0.5, 1.0, 1.5],
            "random_control_seeds": [11, 17],
            "batch_size": 32,
            "max_tokens": 15000,
        },
        "methodology": {
            "scorecard": methodology_scorecard,
            "source_metrics_available": source_metrics_available,
            "main_read": (
                "Phase 20 upgrades the earlier patching work with paired denoise/noise cohorts, "
                "a lambda sweep, and matched orthogonal random controls, but it still does not "
                "provide full source-restoration metrics because source behaviors were not generated."
            ),
        },
        "overall": {
            "tool_token_selectivity_gap_wins": overall_tool_gap_wins,
            "generated_token_selectivity_gap_wins": overall_gen_gap_wins,
            "total_gap_comparisons": total_gap_comparisons,
            "overall_patch_entries": overall_patch_entries,
            "overall_patch_applied_rate": overall_applied / overall_patch_entries if overall_patch_entries else None,
            "overall_patch_skipped_rate": overall_skipped / overall_patch_entries if overall_patch_entries else None,
        },
        "patch": {
            "leader": hypotheses["leader"]["patch"],
            "dispersion": hypotheses["dispersion"]["patch"],
        },
        "hypotheses": hypotheses,
        "lambda_one_rows": lambda_one_rows,
    }

    _plot_metric_curves(
        summary,
        metric="tool_token_change_rate",
        ylabel="Tool-token change rate",
        filename="phase20_tool_token_change_curves.png",
    )
    _plot_metric_curves(
        summary,
        metric="mean_generated_token_count_delta",
        ylabel="Mean generated-token delta",
        filename="phase20_generated_token_delta_curves.png",
    )
    _plot_metric_curves(
        summary,
        metric="mean_patch_delta_norm_std",
        ylabel="Mean patch delta norm (std space)",
        filename="phase20_patch_delta_norm_curves.png",
    )
    _plot_lambda_one_gaps(summary)

    (ASSET_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (ASSET_DIR / "per_cell_results.json").write_text(json.dumps(per_cell_results, indent=2))


if __name__ == "__main__":
    main()
