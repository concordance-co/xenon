"""Transfer-probe chart and table generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from pipelines_v2.reporting.chart_style import (
    best_series_point,
    display_metric,
    display_name,
    format_layer,
    format_regularization,
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

_METRICS = ("balanced_accuracy", "auroc")


def render(*, step_name: str, step_slug: str, result: dict[str, Any], report_root: Path) -> dict[str, Any]:
    layers = sorted(
        [dict(item) for item in result.get("layers", []) if isinstance(item, dict)],
        key=lambda item: int(item.get("layer", 0)),
    )
    summary = dict(result.get("summary", {}))
    mode = str(summary.get("mode") or "cross_cohort_transfer")
    asset_dir = report_root / "assets" / step_slug

    rows: list[dict[str, Any]] = []
    cross_metric_series = {metric: defaultdict(list) for metric in _METRICS}
    delta_series: dict[str, list[tuple[int, float | None]]] = defaultdict(list)
    direction_series: dict[str, list[tuple[int, float | None]]] = defaultdict(list)
    split_metric_series: dict[str, dict[str, dict[str, list[tuple[int, float | None]]]]] = defaultdict(
        lambda: {metric: defaultdict(list) for metric in _METRICS}
    )
    sweep_series = {metric: defaultdict(list) for metric in _METRICS}

    for layer_payload in layers:
        layer = int(layer_payload.get("layer", 0))
        within = layer_payload.get("within_cohort_baseline")
        if isinstance(within, Mapping):
            for cohort, metrics in within.items():
                if not isinstance(metrics, Mapping):
                    continue
                rows.append(
                    {
                        "row_kind": "within_baseline",
                        "layer": layer,
                        "cohort": str(cohort),
                        "direction": None,
                        "split_name": None,
                        "C": None,
                        "balanced_accuracy": None,
                        "auroc": None,
                        "within_baseline_balanced_accuracy": metrics.get("balanced_accuracy"),
                        "within_baseline_auroc": metrics.get("auroc"),
                        "cross_transfer_balanced_accuracy": None,
                        "cross_transfer_auroc": None,
                        "transfer_delta_balanced_accuracy": None,
                        "direction_similarity": None,
                    }
                )

        transfers = layer_payload.get("cross_cohort_transfer")
        if isinstance(transfers, Mapping):
            for direction, payload in transfers.items():
                test_cohort = _test_cohort(str(direction))
                baseline = dict(within.get(test_cohort, {})) if isinstance(within, Mapping) and test_cohort in within else {}
                if isinstance(payload, Mapping) and "regularization_sweep" in payload:
                    for sweep in _iter_sweep(payload):
                        row = _transfer_row(
                            layer=layer,
                            direction=str(direction),
                            split_name=None,
                            baseline=baseline,
                            payload=sweep,
                        )
                        row["row_kind"] = "cross_transfer_regularization"
                        rows.append(row)
                        for metric in _METRICS:
                            sweep_series[metric][f"{direction} @ L{layer}"].append(
                                (float(sweep.get("C", 0.0)), metric_value(sweep, metric))
                            )
                    continue
                if not isinstance(payload, Mapping):
                    continue
                row = _transfer_row(
                    layer=layer,
                    direction=str(direction),
                    split_name=None,
                    baseline=baseline,
                    payload=payload,
                )
                rows.append(row)
                for metric in _METRICS:
                    cross_metric_series[metric][str(direction)].append((layer, metric_value(payload, metric)))
                delta_series[str(direction)].append((layer, metric_value(payload, "transfer_delta_vs_test_within")))

        similarities = layer_payload.get("direction_similarity")
        if isinstance(similarities, Mapping):
            for name, value in similarities.items():
                rows.append(
                    {
                        "row_kind": "direction_similarity",
                        "layer": layer,
                        "cohort": None,
                        "direction": str(name),
                        "split_name": None,
                        "C": None,
                        "balanced_accuracy": None,
                        "auroc": None,
                        "within_baseline_balanced_accuracy": None,
                        "within_baseline_auroc": None,
                        "cross_transfer_balanced_accuracy": None,
                        "cross_transfer_auroc": None,
                        "transfer_delta_balanced_accuracy": None,
                        "direction_similarity": value,
                    }
                )
                direction_series[str(name)].append((layer, float(value) if value is not None else None))

        split_results = layer_payload.get("split_results")
        if isinstance(split_results, Mapping):
            for split_name, payload in split_results.items():
                _flatten_split_payload(
                    rows=rows,
                    split_metric_series=split_metric_series,
                    sweep_series=sweep_series,
                    layer=layer,
                    split_name=str(split_name),
                    payload=payload,
                )

    figures: list[dict[str, Any]] = []
    for metric in _METRICS:
        if any(series for series in cross_metric_series[metric].values()):
            filename = f"{metric}_cross_cohort.png"
            _plot_series_by_layer(
                series_map=cross_metric_series[metric],
                step_name=step_name,
                detail=f"Cross cohort \u00b7 {display_metric(metric)}",
                ylabel=display_metric(metric),
                metric_axis=True,
                output_path=asset_dir / filename,
            )
            figures.append(
                {
                    "path": f"assets/{step_slug}/{filename}",
                    "chart_kind": "transfer_cross_cohort_metric",
                    "title": f"{display_metric(metric)} cross cohort",
                    "caption": f"{display_metric(metric)} across layers for cross-cohort transfer in step {step_name}.",
                    "primary": False,
                }
            )

    if any(series for series in delta_series.values()):
        _plot_series_by_layer(
            series_map=delta_series,
            step_name=step_name,
            detail="Transfer delta vs test-cohort within baseline",
            ylabel="Transfer Delta vs Test Within",
            output_path=asset_dir / "transfer_delta_balanced_accuracy.png",
        )
        figures.append(
            {
                "path": f"assets/{step_slug}/transfer_delta_balanced_accuracy.png",
                "chart_kind": "transfer_delta_by_layer",
                "title": "Transfer delta vs within baseline",
                "caption": f"Balanced-accuracy transfer delta against the test-cohort within baseline for step {step_name}.",
                "primary": False,
            }
        )

    if any(series for series in direction_series.values()):
        _plot_series_by_layer(
            series_map=direction_series,
            step_name=step_name,
            detail="Direction similarity across layers",
            ylabel="Cosine Similarity",
            metric_axis=True,
            output_path=asset_dir / "direction_similarity.png",
        )
        figures.append(
            {
                "path": f"assets/{step_slug}/direction_similarity.png",
                "chart_kind": "transfer_direction_similarity",
                "title": "Direction similarity",
                "caption": f"Per-layer direction similarity for transfer comparisons in step {step_name}.",
                "primary": False,
            }
        )

    for split_name, metric_series in sorted(split_metric_series.items()):
        for metric in _METRICS:
            series = metric_series.get(metric, {})
            if not series:
                continue
            filename = f"{split_name}_{metric}.png"
            _plot_series_by_layer(
                series_map=series,
                step_name=step_name,
                detail=f"{display_name(split_name)} \u00b7 {display_metric(metric)}",
                ylabel=display_metric(metric),
                metric_axis=True,
                output_path=asset_dir / filename,
            )
            figures.append(
                {
                    "path": f"assets/{step_slug}/{filename}",
                    "chart_kind": "transfer_split_metric",
                    "title": f"{display_metric(metric)} {display_name(split_name)}",
                    "caption": f"{display_metric(metric)} across layers for split {split_name} in step {step_name}.",
                    "primary": False,
                }
            )

    for metric in _METRICS:
        if any(series for series in sweep_series[metric].values()):
            filename = f"regularization_sweep_{metric}.png"
            _plot_sweep_series(
                series_map=sweep_series[metric],
                step_name=step_name,
                detail=f"Regularization sweep \u00b7 {display_metric(metric)}",
                ylabel=display_metric(metric),
                metric_axis=True,
                output_path=asset_dir / filename,
            )
            figures.append(
                {
                    "path": f"assets/{step_slug}/{filename}",
                    "chart_kind": "transfer_regularization_sweep",
                    "title": f"Regularization sweep {display_metric(metric)}",
                    "caption": f"{display_metric(metric)} across regularization values for transfer step {step_name}.",
                    "primary": False,
                }
            )

    primary_path = _primary_path(
        mode=mode,
        figures=figures,
        split_names=[str(name) for name in summary.get("split_names", [])],
        step_slug=step_slug,
        has_sweep=any(series for series in sweep_series["balanced_accuracy"].values()),
    )
    for figure in figures:
        figure["primary"] = figure["path"] == primary_path

    headline_metrics = {
        "mode": mode,
        "cohort_count": summary.get("cohort_count"),
        "layer_count": summary.get("layer_count"),
        "regularization": summary.get("regularization"),
        "best_cross_transfer_balanced_accuracy": _best_row(rows, "cross_transfer_balanced_accuracy", row_kind="cross_transfer"),
        "best_split_balanced_accuracy": _best_row(rows, "balanced_accuracy", row_kind="split_holdout"),
        "max_direction_similarity": _best_row(rows, "direction_similarity", row_kind="direction_similarity"),
    }
    return {
        "result_kind": "transfer_probe_result",
        "figures": figures,
        "table": {
            "step_name": step_name,
            "result_kind": "transfer_probe_result",
            "columns": sorted({key for row in rows for key in row}),
            "summary": summary,
            "headline_metrics": headline_metrics,
            "rows": rows,
        },
        "headline_metrics": headline_metrics,
    }


def _test_cohort(direction: str) -> str:
    if "_to_" not in direction:
        return direction
    return direction.split("_to_", 1)[1]


def _transfer_row(
    *,
    layer: int,
    direction: str,
    split_name: str | None,
    baseline: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "row_kind": "cross_transfer",
        "layer": layer,
        "cohort": _test_cohort(direction),
        "direction": direction,
        "split_name": split_name,
        "C": payload.get("C"),
        "balanced_accuracy": None,
        "auroc": None,
        "within_baseline_balanced_accuracy": baseline.get("balanced_accuracy"),
        "within_baseline_auroc": baseline.get("auroc"),
        "cross_transfer_balanced_accuracy": payload.get("balanced_accuracy"),
        "cross_transfer_auroc": payload.get("auroc"),
        "transfer_delta_balanced_accuracy": payload.get("transfer_delta_vs_test_within"),
        "direction_similarity": None,
    }


def _flatten_split_payload(
    *,
    rows: list[dict[str, Any]],
    split_metric_series: dict[str, dict[str, dict[str, list[tuple[int, float | None]]]]],
    sweep_series: dict[str, dict[str, list[tuple[float, float | None]]]],
    layer: int,
    split_name: str,
    payload: Any,
) -> None:
    if isinstance(payload, Mapping) and "regularization_sweep" in payload:
        _append_split_sweep_rows(
            rows=rows,
            sweep_series=sweep_series,
            layer=layer,
            split_name=split_name,
            cohort=None,
            sweep_payload=payload,
        )
        return
    if _is_metric_payload(payload):
        metric_payload = dict(payload)
        rows.append(
            {
                "row_kind": "split_holdout",
                "layer": layer,
                "cohort": None,
                "direction": None,
                "split_name": split_name,
                "C": metric_payload.get("C"),
                "balanced_accuracy": metric_payload.get("balanced_accuracy"),
                "auroc": metric_payload.get("auroc"),
                "within_baseline_balanced_accuracy": None,
                "within_baseline_auroc": None,
                "cross_transfer_balanced_accuracy": None,
                "cross_transfer_auroc": None,
                "transfer_delta_balanced_accuracy": None,
                "direction_similarity": None,
            }
        )
        for metric in _METRICS:
            split_metric_series[split_name][metric]["all"].append((layer, metric_value(metric_payload, metric)))
        return
    if not isinstance(payload, Mapping):
        return
    for cohort, cohort_payload in payload.items():
        if isinstance(cohort_payload, Mapping) and "regularization_sweep" in cohort_payload:
            _append_split_sweep_rows(
                rows=rows,
                sweep_series=sweep_series,
                layer=layer,
                split_name=split_name,
                cohort=str(cohort),
                sweep_payload=cohort_payload,
            )
            continue
        if not _is_metric_payload(cohort_payload):
            continue
        metric_payload = dict(cohort_payload)
        rows.append(
            {
                "row_kind": "split_holdout",
                "layer": layer,
                "cohort": str(cohort),
                "direction": None,
                "split_name": split_name,
                "C": metric_payload.get("C"),
                "balanced_accuracy": metric_payload.get("balanced_accuracy"),
                "auroc": metric_payload.get("auroc"),
                "within_baseline_balanced_accuracy": None,
                "within_baseline_auroc": None,
                "cross_transfer_balanced_accuracy": None,
                "cross_transfer_auroc": None,
                "transfer_delta_balanced_accuracy": None,
                "direction_similarity": None,
            }
        )
        label = str(cohort)
        for metric in _METRICS:
            split_metric_series[split_name][metric][label].append((layer, metric_value(metric_payload, metric)))


def _append_split_sweep_rows(
    *,
    rows: list[dict[str, Any]],
    sweep_series: dict[str, dict[str, list[tuple[float, float | None]]]],
    layer: int,
    split_name: str,
    cohort: str | None,
    sweep_payload: Mapping[str, Any],
) -> None:
    label = split_name if cohort is None else f"{split_name}:{cohort}"
    for sweep in _iter_sweep(sweep_payload):
        rows.append(
            {
                "row_kind": "split_holdout_regularization",
                "layer": layer,
                "cohort": cohort,
                "direction": None,
                "split_name": split_name,
                "C": sweep.get("C"),
                "balanced_accuracy": sweep.get("balanced_accuracy"),
                "auroc": sweep.get("auroc"),
                "within_baseline_balanced_accuracy": None,
                "within_baseline_auroc": None,
                "cross_transfer_balanced_accuracy": None,
                "cross_transfer_auroc": None,
                "transfer_delta_balanced_accuracy": None,
                "direction_similarity": None,
            }
        )
        for metric in _METRICS:
            sweep_series[metric][f"{label} @ L{layer}"].append((float(sweep.get("C", 0.0)), metric_value(sweep, metric)))


def _iter_sweep(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = payload.get("regularization_sweep")
    if not isinstance(values, list):
        return []
    return [dict(item) for item in values if isinstance(item, Mapping)]


def _is_metric_payload(payload: Any) -> bool:
    return isinstance(payload, Mapping) and any(key in payload for key in ("balanced_accuracy", "auroc", "accuracy", "split_mode"))


def _plot_series_by_layer(
    *,
    series_map: Mapping[str, list[tuple[int, float | None]]],
    step_name: str,
    detail: str,
    ylabel: str,
    metric_axis: bool = False,
    output_path: Path,
) -> None:
    best_label, best_x, best_value = best_series_point(series_map)
    fig, ax = new_figure(
        title=display_name(step_name),
        subtitle="TRANSFER PROBE",
        metric_label=detail.upper(),
        metric_value=format_stat(best_value),
        right_label="BEST LAYER",
        right_value=format_layer(best_x),
    )
    line_by_label = {}
    x_values: list[int] = []
    plotted_values: list[float] = []
    for label, pairs in sorted(series_map.items()):
        ordered = sorted(pairs, key=lambda item: item[0])
        xs = [item[0] for item in ordered]
        ys = [item[1] for item in ordered]
        line_by_label[label] = plot_series(
            ax,
            xs,
            ys,
            label=display_name(label),
            dashed=("within" in label or "baseline" in label),
        )
        x_values.extend(xs)
        plotted_values.extend(float(value) for value in ys if value is not None)
    if best_label is not None and best_x is not None and best_value is not None:
        highlight_point(ax, best_x, best_value, color=line_by_label[best_label].get_color())
    y_limits = value_limits(plotted_values)
    style_axes(
        ax,
        xlabel="Layer",
        ylabel=ylabel,
        x_values=x_values,
        layer_axis=True,
        metric_axis=metric_axis,
        y_limits=y_limits,
    )
    header_legend(ax, ncol=2 if len(series_map) > 2 else 1)
    save_figure(fig, output_path)


def _plot_sweep_series(
    *,
    series_map: Mapping[str, list[tuple[float, float | None]]],
    step_name: str,
    detail: str,
    ylabel: str,
    metric_axis: bool = False,
    output_path: Path,
) -> None:
    best_label, best_x, best_value = best_series_point(series_map)
    fig, ax = new_figure(
        title=display_name(step_name),
        subtitle="TRANSFER PROBE",
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
        line_by_label[label] = plot_series(
            ax,
            xs,
            ys,
            label=display_name(label),
            dashed=("within" in label or "baseline" in label),
        )
        x_values.extend(xs)
        plotted_values.extend(float(value) for value in ys if value is not None)
    if best_label is not None and best_x is not None and best_value is not None:
        highlight_point(ax, best_x, best_value, color=line_by_label[best_label].get_color())
    use_log_scale = all(value > 0.0 for value in x_values)
    y_limits = value_limits(plotted_values)
    style_axes(
        ax,
        xlabel="Regularization C",
        ylabel=ylabel,
        metric_axis=metric_axis,
        y_limits=y_limits,
        xscale="log" if use_log_scale else None,
    )
    header_legend(ax, ncol=2 if len(series_map) > 2 else 1)
    save_figure(fig, output_path)


def _primary_path(
    *,
    mode: str,
    figures: list[dict[str, Any]],
    split_names: list[str],
    step_slug: str,
    has_sweep: bool,
) -> str | None:
    if mode == "cross_cohort_transfer":
        candidate = f"assets/{step_slug}/balanced_accuracy_cross_cohort.png"
        if any(item["path"] == candidate for item in figures):
            return candidate
    if mode == "split_holdout" and split_names:
        candidate = f"assets/{step_slug}/{split_names[0]}_balanced_accuracy.png"
        if any(item["path"] == candidate for item in figures):
            return candidate
    if has_sweep:
        candidate = f"assets/{step_slug}/regularization_sweep_balanced_accuracy.png"
        if any(item["path"] == candidate for item in figures):
            return candidate
    return None


def _best_row(rows: list[dict[str, Any]], key: str, *, row_kind: str) -> dict[str, Any] | None:
    ranked = []
    for row in rows:
        if row.get("row_kind") != row_kind:
            continue
        value = metric_value(row, key)
        if value is None:
            continue
        ranked.append((value, row))
    if not ranked:
        return None
    value, row = max(ranked, key=lambda item: item[0])
    return {
        "layer": row.get("layer"),
        "direction": row.get("direction"),
        "cohort": row.get("cohort"),
        "split_name": row.get("split_name"),
        "value": value,
    }
