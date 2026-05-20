"""Probe-result chart and table generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipelines_v2.reporting.chart_style import (
    best_point,
    display_metric,
    display_name,
    format_layer,
    format_stat,
    header_legend,
    highlight_point,
    metric_value,
    new_figure,
    plot_series,
    save_figure,
    style_axes,
    value_limits,
)

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
    best_layer, best_value = best_point(x_values, y_values)
    fig, ax = new_figure(
        title=display_name(step_name),
        subtitle="PROBE",
        metric_label=f"BEST {display_metric(metric).upper()}",
        metric_value=format_stat(best_value),
        right_label="BEST LAYER",
        right_value=format_layer(best_layer),
    )
    plot_series(ax, x_values, y_values)
    if best_layer is not None and best_value is not None:
        highlight_point(ax, best_layer, best_value)
    style_axes(
        ax,
        xlabel="Layer",
        ylabel=display_metric(metric),
        x_values=x_values,
        layer_axis=True,
        metric_axis=True,
        y_limits=value_limits(y_values),
    )
    save_figure(fig, output_path)


def _plot_combined_metrics(
    *,
    rows: list[dict[str, Any]],
    metrics: list[str],
    step_name: str,
    output_path: Path,
) -> None:
    x_values = [int(row["layer"]) for row in rows]
    fig, ax = new_figure(
        title=display_name(step_name),
        subtitle="PROBE",
        detail="Metric overview across captured layers",
    )
    plotted_values: list[float] = []
    for metric in metrics:
        y_values = [metric_value(row, metric) for row in rows]
        plot_series(ax, x_values, y_values, label=display_metric(metric))
        plotted_values.extend(float(value) for value in y_values if value is not None)
    style_axes(
        ax,
        xlabel="Layer",
        ylabel="Metric Value",
        x_values=x_values,
        layer_axis=True,
        metric_axis=True,
        y_limits=value_limits(plotted_values),
    )
    header_legend(ax, ncol=min(2, len(metrics)))
    save_figure(fig, output_path)


def _table_columns(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({key for row in rows for key in row})
