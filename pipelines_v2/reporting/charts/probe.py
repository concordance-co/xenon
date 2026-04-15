"""Probe-result chart and table generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipelines_v2.reporting.chart_style import display_metric, display_name, metric_value, new_figure, save_figure

_PROBE_METRICS = ("balanced_accuracy", "accuracy", "auroc", "selectivity")


def render(*, step_name: str, step_slug: str, result: dict[str, Any], report_root: Path) -> dict[str, Any]:
    layers = sorted(
        [dict(item) for item in result.get("layers", []) if isinstance(item, dict)],
        key=lambda item: int(item.get("layer", 0)),
    )
    rows = []
    for layer in layers:
        row = {"layer": int(layer.get("layer", 0))}
        for key, value in layer.items():
            if key == "layer":
                continue
            row[str(key)] = value
        rows.append(row)

    figures: list[dict[str, Any]] = []
    asset_dir = report_root / "assets" / step_slug
    metric_to_filename = {
        "balanced_accuracy": "balanced_accuracy_by_layer.png",
        "accuracy": "accuracy_by_layer.png",
        "auroc": "auroc_by_layer.png",
    }

    for metric, filename in metric_to_filename.items():
        if not _has_metric(rows, metric):
            continue
        _plot_metric_by_layer(
            rows=rows,
            metric=metric,
            step_name=step_name,
            output_path=asset_dir / filename,
        )
        figures.append(
            {
                "path": f"assets/{step_slug}/{filename}",
                "chart_kind": "probe_metric_by_layer",
                "title": f"{display_metric(metric)} by layer",
                "caption": f"{display_metric(metric)} across captured layers for probe step {step_name}.",
                "primary": False,
            }
        )

    available_metrics = [metric for metric in _PROBE_METRICS if _has_metric(rows, metric)]
    if len(available_metrics) > 1:
        _plot_combined_metrics(
            rows=rows,
            metrics=available_metrics,
            step_name=step_name,
            output_path=asset_dir / "probe_metrics_by_layer.png",
        )
        figures.append(
            {
                "path": f"assets/{step_slug}/probe_metrics_by_layer.png",
                "chart_kind": "probe_metrics_by_layer",
                "title": "Probe metrics by layer",
                "caption": f"Available probe metrics across captured layers for step {step_name}.",
                "primary": False,
            }
        )

    primary_path = None
    for candidate in (
        f"assets/{step_slug}/balanced_accuracy_by_layer.png",
        f"assets/{step_slug}/accuracy_by_layer.png",
        f"assets/{step_slug}/auroc_by_layer.png",
    ):
        if any(item["path"] == candidate for item in figures):
            primary_path = candidate
            break
    for item in figures:
        item["primary"] = item["path"] == primary_path

    summary = dict(result.get("summary", {}))
    headline_metrics = {
        "best_layer": summary.get("best_layer"),
        "best_metric": summary.get("best_metric"),
        "best_value": summary.get("best_value"),
        "example_count": summary.get("example_count"),
        "group_count": summary.get("group_count"),
        "split_mode": summary.get("split_mode"),
    }
    return {
        "result_kind": "probe_result",
        "figures": figures,
        "table": {
            "step_name": step_name,
            "result_kind": "probe_result",
            "columns": _table_columns(rows),
            "summary": summary,
            "headline_metrics": headline_metrics,
            "rows": rows,
        },
        "headline_metrics": headline_metrics,
    }


def _has_metric(rows: list[dict[str, Any]], metric: str) -> bool:
    return any(metric_value(row, metric) is not None for row in rows)


def _plot_metric_by_layer(
    *,
    rows: list[dict[str, Any]],
    metric: str,
    step_name: str,
    output_path: Path,
) -> None:
    x_values = [int(row["layer"]) for row in rows]
    y_values = [metric_value(row, metric) for row in rows]
    fig, ax = new_figure()
    ax.plot(x_values, y_values, marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel(display_metric(metric))
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{display_metric(metric)} by layer: {display_name(step_name)}")
    save_figure(fig, output_path)


def _plot_combined_metrics(
    *,
    rows: list[dict[str, Any]],
    metrics: list[str],
    step_name: str,
    output_path: Path,
) -> None:
    x_values = [int(row["layer"]) for row in rows]
    fig, ax = new_figure()
    for metric in metrics:
        y_values = [metric_value(row, metric) for row in rows]
        ax.plot(x_values, y_values, marker="o", label=display_metric(metric))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Probe metrics by layer: {display_name(step_name)}")
    ax.legend(loc="best")
    save_figure(fig, output_path)


def _table_columns(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({key for row in rows for key in row})
