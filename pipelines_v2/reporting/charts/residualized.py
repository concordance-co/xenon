"""Residualized-probe chart and table generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipelines_v2.reporting.chart_style import display_metric, display_name, metric_value, new_figure, save_figure


def render(*, step_name: str, step_slug: str, result: dict[str, Any], report_root: Path) -> dict[str, Any]:
    layers = sorted(
        [dict(item) for item in result.get("layers", []) if isinstance(item, dict)],
        key=lambda item: int(item.get("layer", 0)),
    )
    rows = [_flatten_layer(item) for item in layers]
    asset_dir = report_root / "assets" / step_slug
    figures: list[dict[str, Any]] = []

    if _has_metric(rows, "raw_balanced_accuracy") or _has_metric(rows, "residualized_balanced_accuracy"):
        _plot_raw_vs_residualized(
            rows=rows,
            metric="balanced_accuracy",
            step_name=step_name,
            output_path=asset_dir / "balanced_accuracy_raw_vs_residualized.png",
        )
        figures.append(
            {
                "path": f"assets/{step_slug}/balanced_accuracy_raw_vs_residualized.png",
                "chart_kind": "residualized_metric_comparison",
                "title": "Balanced accuracy raw vs residualized",
                "caption": f"Raw and residualized balanced accuracy by layer for step {step_name}.",
                "primary": False,
            }
        )

    if _has_metric(rows, "raw_auroc") or _has_metric(rows, "residualized_auroc"):
        _plot_raw_vs_residualized(
            rows=rows,
            metric="auroc",
            step_name=step_name,
            output_path=asset_dir / "auroc_raw_vs_residualized.png",
        )
        figures.append(
            {
                "path": f"assets/{step_slug}/auroc_raw_vs_residualized.png",
                "chart_kind": "residualized_metric_comparison",
                "title": "AUROC raw vs residualized",
                "caption": f"Raw and residualized AUROC by layer for step {step_name}.",
                "primary": False,
            }
        )

    _plot_delta(
        rows=rows,
        step_name=step_name,
        output_path=asset_dir / "delta_raw_minus_null.png",
    )
    figures.append(
        {
            "path": f"assets/{step_slug}/delta_raw_minus_null.png",
            "chart_kind": "residualized_delta_by_layer",
            "title": "Delta raw minus residualized",
            "caption": f"Per-layer delta between raw and residualized probe metrics for step {step_name}.",
            "primary": False,
        }
    )

    _plot_nuisance_accuracy(
        rows=rows,
        step_name=step_name,
        output_path=asset_dir / "nuisance_accuracy.png",
    )
    figures.append(
        {
            "path": f"assets/{step_slug}/nuisance_accuracy.png",
            "chart_kind": "residualized_nuisance_accuracy",
            "title": "Nuisance accuracy",
            "caption": f"Nuisance training-fit accuracy before and after residualization for step {step_name}.",
            "primary": False,
        }
    )

    primary_path = (
        f"assets/{step_slug}/auroc_raw_vs_residualized.png"
        if any(item["path"] == f"assets/{step_slug}/auroc_raw_vs_residualized.png" for item in figures)
        else f"assets/{step_slug}/balanced_accuracy_raw_vs_residualized.png"
    )
    for item in figures:
        item["primary"] = item["path"] == primary_path

    headline_metrics = _headline_metrics(rows=rows, summary=dict(result.get("summary", {})))
    return {
        "result_kind": "residualized_probe_result",
        "figures": figures,
        "table": {
            "step_name": step_name,
            "result_kind": "residualized_probe_result",
            "columns": sorted({key for row in rows for key in row}),
            "summary": dict(result.get("summary", {})),
            "headline_metrics": headline_metrics,
            "rows": rows,
        },
        "headline_metrics": headline_metrics,
    }


def _flatten_layer(layer: dict[str, Any]) -> dict[str, Any]:
    row = {
        "layer": int(layer.get("layer", 0)),
        "nuisance_accuracy_raw_training_fit": layer.get("nuisance_accuracy_raw_training_fit"),
        "nuisance_accuracy_on_null_training_fit": layer.get("nuisance_accuracy_on_null_training_fit"),
        "nuisance_subspace_rank": layer.get("family_subspace_rank"),
    }
    raw_probe = dict(layer.get("raw_probe", {}))
    residualized_probe = dict(layer.get("residualized_probe", {}))
    delta = dict(layer.get("delta_raw_minus_null", {}))
    for key, value in raw_probe.items():
        row[f"raw_{key}"] = value
    for key, value in residualized_probe.items():
        row[f"residualized_{key}"] = value
    for key, value in delta.items():
        row[f"delta_raw_minus_null_{key}"] = value
    return row


def _plot_raw_vs_residualized(
    *,
    rows: list[dict[str, Any]],
    metric: str,
    step_name: str,
    output_path: Path,
) -> None:
    x_values = [int(row["layer"]) for row in rows]
    raw_key = f"raw_{metric}"
    residualized_key = f"residualized_{metric}"
    fig, ax = new_figure()
    ax.plot(x_values, [metric_value(row, raw_key) for row in rows], marker="o", label="Raw")
    ax.plot(x_values, [metric_value(row, residualized_key) for row in rows], marker="o", label="Residualized")
    ax.set_xlabel("Layer")
    ax.set_ylabel(display_metric(metric))
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{display_metric(metric)} raw vs residualized: {display_name(step_name)}")
    ax.legend(loc="best")
    save_figure(fig, output_path)


def _plot_delta(*, rows: list[dict[str, Any]], step_name: str, output_path: Path) -> None:
    x_values = [int(row["layer"]) for row in rows]
    fig, ax = new_figure()
    plotted = False
    for metric in ("balanced_accuracy", "auroc"):
        key = f"delta_raw_minus_null_{metric}"
        if not _has_metric(rows, key):
            continue
        ax.plot(x_values, [metric_value(row, key) for row in rows], marker="o", label=display_metric(metric))
        plotted = True
    if not plotted:
        ax.plot(x_values, [0.0 for _ in rows], marker="o", label="Delta")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Raw - Residualized")
    ax.set_title(f"Residualization delta by layer: {display_name(step_name)}")
    ax.axhline(0.0, color="#6b7280", linewidth=1.0, linestyle="--")
    ax.legend(loc="best")
    save_figure(fig, output_path)


def _plot_nuisance_accuracy(*, rows: list[dict[str, Any]], step_name: str, output_path: Path) -> None:
    x_values = [int(row["layer"]) for row in rows]
    fig, ax = new_figure()
    ax.plot(
        x_values,
        [metric_value(row, "nuisance_accuracy_raw_training_fit") for row in rows],
        marker="o",
        label="Raw training fit",
    )
    ax.plot(
        x_values,
        [metric_value(row, "nuisance_accuracy_on_null_training_fit") for row in rows],
        marker="o",
        label="Null-space training fit",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Nuisance accuracy by layer: {display_name(step_name)}")
    ax.legend(loc="best")
    save_figure(fig, output_path)


def _has_metric(rows: list[dict[str, Any]], key: str) -> bool:
    return any(metric_value(row, key) is not None for row in rows)


def _headline_metrics(*, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_count": summary.get("layer_count"),
        "example_count": summary.get("example_count"),
        "best_raw_balanced_accuracy": _best_row(rows, "raw_balanced_accuracy"),
        "best_residualized_balanced_accuracy": _best_row(rows, "residualized_balanced_accuracy"),
        "max_delta_raw_minus_null_balanced_accuracy": _best_row(rows, "delta_raw_minus_null_balanced_accuracy"),
    }


def _best_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    ranked = [(metric_value(row, key), int(row["layer"])) for row in rows if metric_value(row, key) is not None]
    if not ranked:
        return None
    value, layer = max(ranked, key=lambda item: item[0])
    return {"layer": layer, "value": value}
