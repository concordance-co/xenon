"""Residualized-probe chart and table generation."""

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
    horizontal_reference,
    metric_value,
    new_figure,
    plot_series,
    save_figure,
    style_axes,
    value_limits,
)


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
    raw_values = [metric_value(row, raw_key) for row in rows]
    residualized_values = [metric_value(row, residualized_key) for row in rows]
    raw_best_layer, raw_best_value = best_point(x_values, raw_values)
    residualized_best_layer, residualized_best_value = best_point(x_values, residualized_values)
    if (residualized_best_value or float("-inf")) > (raw_best_value or float("-inf")):
        best_layer, best_value = residualized_best_layer, residualized_best_value
    else:
        best_layer, best_value = raw_best_layer, raw_best_value
    fig, ax = new_figure(
        title=display_name(step_name),
        subtitle="RESIDUALIZED PROBE",
        metric_label=f"BEST {display_metric(metric).upper()}",
        metric_value=format_stat(best_value),
        right_label="BEST LAYER",
        right_value=format_layer(best_layer),
    )
    raw_line = plot_series(ax, x_values, raw_values, label="Raw")
    residualized_line = plot_series(ax, x_values, residualized_values, label="Residualized")
    if best_layer is not None and best_value is not None:
        highlight_color = residualized_line.get_color() if best_value == residualized_best_value else raw_line.get_color()
        highlight_point(ax, best_layer, best_value, color=highlight_color)
    style_axes(
        ax,
        xlabel="Layer",
        ylabel=display_metric(metric),
        x_values=x_values,
        layer_axis=True,
        metric_axis=True,
        y_limits=value_limits([*raw_values, *residualized_values]),
    )
    header_legend(ax, ncol=2)
    save_figure(fig, output_path)


def _plot_delta(*, rows: list[dict[str, Any]], step_name: str, output_path: Path) -> None:
    x_values = [int(row["layer"]) for row in rows]
    plotted_pairs: list[tuple[int, float]] = []
    fig, ax = new_figure(
        title=display_name(step_name),
        subtitle="RESIDUALIZED PROBE",
        detail="Raw minus residualized delta across captured layers",
    )
    plotted = False
    for metric in ("balanced_accuracy", "auroc"):
        key = f"delta_raw_minus_null_{metric}"
        if not _has_metric(rows, key):
            continue
        y_values = [metric_value(row, key) for row in rows]
        plot_series(ax, x_values, y_values, label=display_metric(metric))
        plotted_pairs.extend((layer, float(value)) for layer, value in zip(x_values, y_values, strict=False) if value is not None)
        plotted = True
    if not plotted:
        zero_values = [0.0 for _ in rows]
        plot_series(ax, x_values, zero_values, label="Delta")
        plotted_pairs.extend((layer, 0.0) for layer in x_values)
    max_abs = max((abs(value) for _, value in plotted_pairs), default=0.0)
    delta_limit = max(0.05, max_abs + 0.05)
    style_axes(
        ax,
        xlabel="Layer",
        ylabel="Raw - Residualized",
        x_values=x_values,
        layer_axis=True,
        y_limits=(-delta_limit, delta_limit),
    )
    horizontal_reference(ax, 0.0)
    header_legend(ax)
    save_figure(fig, output_path)


def _plot_nuisance_accuracy(*, rows: list[dict[str, Any]], step_name: str, output_path: Path) -> None:
    x_values = [int(row["layer"]) for row in rows]
    raw_values = [metric_value(row, "nuisance_accuracy_raw_training_fit") for row in rows]
    null_values = [metric_value(row, "nuisance_accuracy_on_null_training_fit") for row in rows]
    best_layer, best_value = best_point(x_values, raw_values)
    fig, ax = new_figure(
        title=display_name(step_name),
        subtitle="RESIDUALIZED PROBE",
        metric_label="BEST RAW TRAINING FIT",
        metric_value=format_stat(best_value),
        right_label="BEST LAYER",
        right_value=format_layer(best_layer),
    )
    raw_line = plot_series(ax, x_values, raw_values, label="Raw training fit")
    plot_series(ax, x_values, null_values, label="Null-space training fit")
    if best_layer is not None and best_value is not None:
        highlight_point(ax, best_layer, best_value, color=raw_line.get_color())
    style_axes(
        ax,
        xlabel="Layer",
        ylabel="Balanced Accuracy",
        x_values=x_values,
        layer_axis=True,
        metric_axis=True,
        y_limits=value_limits([*raw_values, *null_values]),
    )
    header_legend(ax)
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
