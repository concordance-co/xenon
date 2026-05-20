"""Text-baseline chart and table generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from pipelines_v2.reporting.chart_style import (
    annotate_bars,
    best_named_value,
    best_series_point,
    display_metric,
    display_name,
    format_regularization,
    format_stat,
    header_legend,
    highlight_point,
    metric_value,
    new_figure,
    plot_series,
    save_figure,
    style_axes,
    style_bars,
    value_limits,
)

_METRICS = ("balanced_accuracy", "auroc")


def render(*, step_name: str, step_slug: str, result: dict[str, Any], report_root: Path) -> dict[str, Any]:
    summary = dict(result.get("summary", {}))
    mode = str(result.get("mode") or summary.get("mode") or "grouped_cv")
    results_payload = dict(result.get("results", {}))
    model = result.get("model")
    asset_dir = report_root / "assets" / step_slug

    rows: list[dict[str, Any]] = []
    cross_series = {metric: defaultdict(list) for metric in _METRICS}
    split_series: dict[str, dict[str, dict[str, float | None]]] = {}
    sweep_series = {metric: defaultdict(list) for metric in _METRICS}
    grouped_metrics = dict(results_payload.get("grouped_cv", {})) if isinstance(results_payload.get("grouped_cv"), Mapping) else {}

    within = results_payload.get("within_cohort_baseline")
    if isinstance(within, Mapping):
        for cohort, payload in within.items():
            if not isinstance(payload, Mapping):
                continue
            rows.append(
                {
                    "row_kind": "within_baseline",
                    "model": model,
                    "cohort": str(cohort),
                    "direction": None,
                    "split_name": None,
                    "C": payload.get("C"),
                    "balanced_accuracy": None,
                    "auroc": None,
                    "within_baseline_balanced_accuracy": payload.get("balanced_accuracy"),
                    "within_baseline_auroc": payload.get("auroc"),
                    "cross_transfer_balanced_accuracy": None,
                    "cross_transfer_auroc": None,
                    "transfer_delta_balanced_accuracy": None,
                }
            )

    transfers = results_payload.get("cross_cohort_transfer")
    if isinstance(transfers, Mapping):
        for direction, payload in transfers.items():
            baseline = dict(within.get(_test_cohort(str(direction)), {})) if isinstance(within, Mapping) else {}
            if isinstance(payload, Mapping) and "regularization_sweep" in payload:
                for sweep in _iter_sweep(payload):
                    rows.append(_transfer_row(direction=str(direction), baseline=baseline, payload=sweep, model=model))
                    for metric in _METRICS:
                        sweep_series[metric][str(direction)].append((float(sweep.get("C", 0.0)), metric_value(sweep, metric)))
                continue
            if not isinstance(payload, Mapping):
                continue
            rows.append(_transfer_row(direction=str(direction), baseline=baseline, payload=payload, model=model))
            for metric in _METRICS:
                cross_series[metric][str(direction)] = metric_value(payload, metric)

    split_results = results_payload.get("split_results")
    if isinstance(split_results, Mapping):
        for split_name, payload in split_results.items():
            _flatten_split_rows(
                rows=rows,
                split_series=split_series,
                sweep_series=sweep_series,
                split_name=str(split_name),
                payload=payload,
                model=str(model) if model is not None else None,
            )

    if isinstance(results_payload.get("regularization_sweep"), list):
        for sweep in _iter_sweep(results_payload):
            rows.append(
                {
                    "row_kind": "grouped_cv_regularization",
                    "model": model,
                    "cohort": None,
                    "direction": None,
                    "split_name": None,
                    "C": sweep.get("C"),
                    "balanced_accuracy": sweep.get("balanced_accuracy"),
                    "auroc": sweep.get("auroc"),
                    "within_baseline_balanced_accuracy": None,
                    "within_baseline_auroc": None,
                    "cross_transfer_balanced_accuracy": None,
                    "cross_transfer_auroc": None,
                    "transfer_delta_balanced_accuracy": None,
                }
            )
            for metric in _METRICS:
                sweep_series[metric]["grouped_cv"].append((float(sweep.get("C", 0.0)), metric_value(sweep, metric)))

    figures: list[dict[str, Any]] = []
    if grouped_metrics:
        rows.append(
            {
                "row_kind": "grouped_cv",
                "model": model,
                "cohort": None,
                "direction": None,
                "split_name": None,
                "C": grouped_metrics.get("C"),
                "balanced_accuracy": grouped_metrics.get("balanced_accuracy"),
                "auroc": grouped_metrics.get("auroc"),
                "within_baseline_balanced_accuracy": None,
                "within_baseline_auroc": None,
                "cross_transfer_balanced_accuracy": None,
                "cross_transfer_auroc": None,
                "transfer_delta_balanced_accuracy": None,
            }
        )
        _plot_grouped_cv(grouped_metrics=grouped_metrics, step_name=step_name, output_path=asset_dir / "grouped_cv_metrics.png")
        figures.append(
            {
                "path": f"assets/{step_slug}/grouped_cv_metrics.png",
                "chart_kind": "text_grouped_cv_metrics",
                "title": "Grouped CV metrics",
                "caption": f"Grouped cross-validation text-baseline metrics for step {step_name}.",
                "primary": False,
            }
        )

    for metric in _METRICS:
        if cross_series[metric]:
            filename = f"{metric}_cross_cohort.png"
            _plot_bar_series(
                series_map=cross_series[metric],
                step_name=step_name,
                detail=f"Cross cohort \u00b7 {display_metric(metric)}",
                ylabel=display_metric(metric),
                output_path=asset_dir / filename,
            )
            figures.append(
                {
                    "path": f"assets/{step_slug}/{filename}",
                    "chart_kind": "text_cross_cohort_metric",
                    "title": f"{display_metric(metric)} cross cohort",
                    "caption": f"{display_metric(metric)} for text cross-cohort transfer in step {step_name}.",
                    "primary": False,
                }
            )

    for split_name, metric_map in sorted(split_series.items()):
        for metric in _METRICS:
            if metric not in metric_map or not metric_map[metric]:
                continue
            filename = f"{split_name}_{metric}.png"
            _plot_named_values(
                values=metric_map[metric],
                step_name=step_name,
                detail=f"{display_name(split_name)} \u00b7 {display_metric(metric)}",
                ylabel=display_metric(metric),
                output_path=asset_dir / filename,
            )
            figures.append(
                {
                    "path": f"assets/{step_slug}/{filename}",
                    "chart_kind": "text_split_metric",
                    "title": f"{display_metric(metric)} {display_name(split_name)}",
                    "caption": f"{display_metric(metric)} for text split {split_name} in step {step_name}.",
                    "primary": False,
                }
            )

    for metric in _METRICS:
        if sweep_series[metric]:
            filename = f"regularization_sweep_{metric}.png"
            _plot_sweep_series(
                series_map=sweep_series[metric],
                step_name=step_name,
                detail=f"Regularization sweep \u00b7 {display_metric(metric)}",
                ylabel=display_metric(metric),
                output_path=asset_dir / filename,
            )
            figures.append(
                {
                    "path": f"assets/{step_slug}/{filename}",
                    "chart_kind": "text_regularization_sweep",
                    "title": f"Regularization sweep {display_metric(metric)}",
                    "caption": f"{display_metric(metric)} across regularization values for text-baseline step {step_name}.",
                    "primary": False,
                }
            )

    primary_path = _primary_path(mode=mode, step_slug=step_slug, split_names=[str(name) for name in summary.get("split_names", [])], figures=figures)
    for figure in figures:
        figure["primary"] = figure["path"] == primary_path

    headline_metrics = {
        "mode": mode,
        "model": model,
        "example_count": summary.get("example_count"),
        "best_cross_transfer_balanced_accuracy": _best_row(rows, "cross_transfer_balanced_accuracy"),
        "best_split_balanced_accuracy": _best_row(rows, "balanced_accuracy"),
    }
    return {
        "result_kind": "text_baseline_result",
        "figures": figures,
        "table": {
            "step_name": step_name,
            "result_kind": "text_baseline_result",
            "columns": sorted({key for row in rows for key in row}),
            "summary": summary,
            "headline_metrics": headline_metrics,
            "rows": rows,
        },
        "headline_metrics": headline_metrics,
    }


def _transfer_row(*, direction: str, baseline: Mapping[str, Any], payload: Mapping[str, Any], model: Any) -> dict[str, Any]:
    return {
        "row_kind": "cross_transfer",
        "model": model,
        "cohort": _test_cohort(direction),
        "direction": direction,
        "split_name": None,
        "C": payload.get("C"),
        "balanced_accuracy": None,
        "auroc": None,
        "within_baseline_balanced_accuracy": baseline.get("balanced_accuracy"),
        "within_baseline_auroc": baseline.get("auroc"),
        "cross_transfer_balanced_accuracy": payload.get("balanced_accuracy"),
        "cross_transfer_auroc": payload.get("auroc"),
        "transfer_delta_balanced_accuracy": payload.get("transfer_delta_vs_test_within"),
    }


def _flatten_split_rows(
    *,
    rows: list[dict[str, Any]],
    split_series: dict[str, dict[str, dict[str, float | None]]],
    sweep_series: dict[str, dict[str, list[tuple[float, float | None]]]],
    split_name: str,
    payload: Any,
    model: str | None,
) -> None:
    if isinstance(payload, Mapping) and "regularization_sweep" in payload:
        for sweep in _iter_sweep(payload):
            rows.append(_split_row(split_name=split_name, cohort=None, payload=sweep, model=model))
            for metric in _METRICS:
                sweep_series[metric][split_name].append((float(sweep.get("C", 0.0)), metric_value(sweep, metric)))
        return
    if _is_metric_payload(payload):
        metric_payload = dict(payload)
        rows.append(_split_row(split_name=split_name, cohort=None, payload=metric_payload, model=model))
        split_series[split_name] = {
            metric: {"all": metric_value(metric_payload, metric)}
            for metric in _METRICS
            if metric_value(metric_payload, metric) is not None
        }
        return
    if not isinstance(payload, Mapping):
        return
    values: dict[str, dict[str, float | None]] = {metric: {} for metric in _METRICS}
    for cohort, cohort_payload in payload.items():
        if isinstance(cohort_payload, Mapping) and "regularization_sweep" in cohort_payload:
            for sweep in _iter_sweep(cohort_payload):
                rows.append(_split_row(split_name=split_name, cohort=str(cohort), payload=sweep, model=model))
                for metric in _METRICS:
                    sweep_series[metric][f"{split_name}:{cohort}"].append((float(sweep.get("C", 0.0)), metric_value(sweep, metric)))
            continue
        if not _is_metric_payload(cohort_payload):
            continue
        metric_payload = dict(cohort_payload)
        rows.append(_split_row(split_name=split_name, cohort=str(cohort), payload=metric_payload, model=model))
        for metric in _METRICS:
            if metric_value(metric_payload, metric) is not None:
                values[metric][str(cohort)] = metric_value(metric_payload, metric)
    if any(values[metric] for metric in _METRICS):
        split_series[split_name] = {metric: values[metric] for metric in _METRICS if values[metric]}


def _split_row(*, split_name: str, cohort: str | None, payload: Mapping[str, Any], model: str | None) -> dict[str, Any]:
    return {
        "row_kind": "split_holdout",
        "model": model,
        "cohort": cohort,
        "direction": None,
        "split_name": split_name,
        "C": payload.get("C"),
        "balanced_accuracy": payload.get("balanced_accuracy"),
        "auroc": payload.get("auroc"),
        "within_baseline_balanced_accuracy": None,
        "within_baseline_auroc": None,
        "cross_transfer_balanced_accuracy": None,
        "cross_transfer_auroc": None,
        "transfer_delta_balanced_accuracy": None,
    }


def _plot_grouped_cv(*, grouped_metrics: Mapping[str, Any], step_name: str, output_path: Path) -> None:
    plotted = {
        metric: metric_value(dict(grouped_metrics), metric)
        for metric in _METRICS
        if metric_value(dict(grouped_metrics), metric) is not None
    }
    _plot_named_values(
        values=plotted,
        step_name=step_name,
        detail="Grouped CV metrics",
        ylabel="Metric Value",
        output_path=output_path,
    )


def _plot_bar_series(*, series_map: Mapping[str, float | None], step_name: str, detail: str, ylabel: str, output_path: Path) -> None:
    _plot_named_values(values=series_map, step_name=step_name, detail=detail, ylabel=ylabel, output_path=output_path)


def _plot_named_values(
    *,
    values: Mapping[str, float | None],
    step_name: str,
    detail: str,
    ylabel: str,
    output_path: Path,
) -> None:
    labels = list(values)
    best_label, best_value = best_named_value(values)
    fig, ax = new_figure(
        title=display_name(step_name),
        subtitle="TEXT BASELINE",
        metric_label=detail.upper(),
        metric_value=format_stat(best_value),
        right_label="BEST",
        right_value=display_name(best_label) if best_label is not None else "n/a",
    )
    heights = [values[label] for label in labels]
    bar_width = 0.55 if len(labels) == 1 else 0.72
    bars = ax.bar(range(len(labels)), heights, width=bar_width)
    style_bars(ax, bars)
    rotation = 0 if len(labels) == 1 else 25
    ha = "center" if len(labels) == 1 else "right"
    ax.set_xticks(range(len(labels)), [display_name(label) for label in labels], rotation=rotation, ha=ha)
    if len(labels) == 1:
        ax.set_xlim(-0.45, 0.45)
    metric_values = [float(value) for value in heights if value is not None]
    y_limits = (0.0, 1.10) if _is_metric_scale(metric_values) else _bar_limits(metric_values)
    style_axes(ax, ylabel=ylabel, metric_axis=_is_metric_scale(metric_values), y_limits=y_limits)
    annotate_bars(ax, bars, digits=2 if ylabel == "Metric Value" else 3)
    save_figure(fig, output_path)


def _plot_sweep_series(
    *,
    series_map: Mapping[str, list[tuple[float, float | None]]],
    step_name: str,
    detail: str,
    ylabel: str,
    output_path: Path,
) -> None:
    best_label, best_x, best_value = best_series_point(series_map)
    fig, ax = new_figure(
        title=display_name(step_name),
        subtitle="TEXT BASELINE",
        metric_label=detail.upper(),
        metric_value=format_stat(best_value),
        right_label="BEST C",
        right_value=format_regularization(best_x),
    )
    line_by_label = {}
    x_values: list[float] = []
    plotted_values: list[float] = []
    for label, pairs in sorted(series_map.items()):
        ordered = sorted(pairs, key=lambda item: item[0])
        xs = [item[0] for item in ordered]
        ys = [item[1] for item in ordered]
        line_by_label[label] = plot_series(ax, xs, ys, label=display_name(label))
        x_values.extend(xs)
        plotted_values.extend(float(value) for value in ys if value is not None)
    if best_label is not None and best_x is not None and best_value is not None:
        highlight_point(ax, best_x, best_value, color=line_by_label[best_label].get_color())
    style_axes(
        ax,
        xlabel="Regularization C",
        ylabel=ylabel,
        metric_axis=_is_metric_scale(plotted_values),
        y_limits=value_limits(plotted_values) if plotted_values else _bar_limits(plotted_values),
        xscale="log" if x_values and all(value > 0.0 for value in x_values) else None,
    )
    header_legend(ax, ncol=2 if len(series_map) > 2 else 1)
    save_figure(fig, output_path)


def _is_metric_scale(values: list[float]) -> bool:
    return bool(values) and min(values) >= 0.0 and max(values) <= 1.05


def _bar_limits(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    lower = min(0.0, min(values))
    upper = max(values)
    padding = max((upper - lower) * 0.12, 0.08)
    return (lower, upper + padding)


def _iter_sweep(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = payload.get("regularization_sweep")
    if not isinstance(values, list):
        return []
    return [dict(item) for item in values if isinstance(item, Mapping)]


def _is_metric_payload(payload: Any) -> bool:
    return isinstance(payload, Mapping) and any(key in payload for key in ("balanced_accuracy", "auroc", "accuracy", "split_mode"))


def _test_cohort(direction: str) -> str:
    if "_to_" not in direction:
        return direction
    return direction.split("_to_", 1)[1]


def _primary_path(*, mode: str, step_slug: str, split_names: list[str], figures: list[dict[str, Any]]) -> str | None:
    candidates = []
    if mode == "grouped_cv":
        candidates.append(f"assets/{step_slug}/grouped_cv_metrics.png")
    elif mode == "cross_cohort_transfer":
        candidates.append(f"assets/{step_slug}/balanced_accuracy_cross_cohort.png")
    elif mode == "split_holdout" and split_names:
        candidates.append(f"assets/{step_slug}/{split_names[0]}_balanced_accuracy.png")
    candidates.append(f"assets/{step_slug}/regularization_sweep_balanced_accuracy.png")
    for candidate in candidates:
        if any(item["path"] == candidate for item in figures):
            return candidate
    return None


def _best_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    ranked = []
    for row in rows:
        value = metric_value(row, key)
        if value is None:
            continue
        ranked.append((value, row))
    if not ranked:
        return None
    value, row = max(ranked, key=lambda item: item[0])
    return {
        "cohort": row.get("cohort"),
        "direction": row.get("direction"),
        "split_name": row.get("split_name"),
        "value": value,
    }
