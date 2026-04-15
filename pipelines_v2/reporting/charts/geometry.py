"""Geometry-result chart and table generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pipelines_v2.reporting.chart_style import (
    categorical_colors,
    display_name,
    first_two_components,
    metric_value,
    new_figure,
    save_figure,
    slugify,
)


def render(*, step_name: str, step_slug: str, result: dict[str, Any], report_root: Path) -> dict[str, Any]:
    layers = sorted(
        [dict(item) for item in result.get("layers", []) if isinstance(item, dict)],
        key=lambda item: int(item.get("layer", 0)),
    )
    rows = [_table_row(item) for item in layers]
    asset_dir = report_root / "assets" / step_slug
    figures: list[dict[str, Any]] = []

    for layer in layers:
        layer_index = int(layer.get("layer", 0))
        color_by = layer.get("color_by")
        labels = layer.get("labels")
        if isinstance(color_by, dict) and color_by:
            for color_key, values in color_by.items():
                filename = f"layer{layer_index}_{slugify(str(color_key))}.png"
                _plot_scatter(
                    layer=layer,
                    label_values=list(values),
                    title=f"Layer {layer_index} by {display_name(str(color_key))}",
                    output_path=asset_dir / filename,
                )
                figures.append(
                    {
                        "path": f"assets/{step_slug}/{filename}",
                        "chart_kind": "geometry_scatter",
                        "title": f"Layer {layer_index} by {display_name(str(color_key))}",
                        "caption": f"Two-component geometry projection for layer {layer_index} colored by {color_key}.",
                        "primary": False,
                    }
                )
        elif isinstance(labels, list) and labels:
            filename = f"layer{layer_index}_label.png"
            _plot_scatter(
                layer=layer,
                label_values=list(labels),
                title=f"Layer {layer_index} by label",
                output_path=asset_dir / filename,
            )
            figures.append(
                {
                    "path": f"assets/{step_slug}/{filename}",
                    "chart_kind": "geometry_scatter",
                    "title": f"Layer {layer_index} by label",
                    "caption": f"Two-component geometry projection for layer {layer_index} colored by label.",
                    "primary": False,
                }
            )

    if str(result.get("method")).lower() == "pca" and any(layer.get("explained_variance_ratio") for layer in layers):
        _plot_explained_variance(
            layers=layers,
            step_name=step_name,
            output_path=asset_dir / "explained_variance_by_layer.png",
        )
        figures.append(
            {
                "path": f"assets/{step_slug}/explained_variance_by_layer.png",
                "chart_kind": "geometry_explained_variance",
                "title": "Explained variance by layer",
                "caption": f"Per-component explained variance across layers for geometry step {step_name}.",
                "primary": False,
            }
        )

    headline_metrics = {
        "method": result.get("method"),
        "layer_count": len(layers),
        "example_count": (result.get("summary") or {}).get("example_count"),
        "available_color_keys": sorted({key for row in rows for key in row.get("available_color_keys", [])}),
    }
    return {
        "result_kind": "geometry_result",
        "figures": figures,
        "table": {
            "step_name": step_name,
            "result_kind": "geometry_result",
            "columns": sorted({key for row in rows for key in row}),
            "summary": dict(result.get("summary", {})),
            "headline_metrics": headline_metrics,
            "rows": rows,
        },
        "headline_metrics": headline_metrics,
    }


def _table_row(layer: dict[str, Any]) -> dict[str, Any]:
    color_by = layer.get("color_by")
    labels = layer.get("labels")
    explained_variance = layer.get("explained_variance_ratio")
    return {
        "layer": int(layer.get("layer", 0)),
        "component_count": layer.get("component_count"),
        "explained_variance_ratio": list(explained_variance) if isinstance(explained_variance, list) else explained_variance,
        "example_count": layer.get("example_count"),
        "selected_example_keys_count": len(layer.get("selected_example_keys", [])),
        "available_color_keys": sorted(color_by) if isinstance(color_by, dict) else [],
        "has_label": isinstance(labels, list) and bool(labels),
    }


def _plot_scatter(*, layer: dict[str, Any], label_values: list[Any], title: str, output_path: Path) -> None:
    x_values, y_values = first_two_components(layer.get("components", []))
    fig, ax = new_figure(figsize=(7.0, 5.4))
    label_strings = [str(value) for value in label_values]
    unique_labels = sorted(set(label_strings))
    colors = categorical_colors(len(unique_labels))
    for label, color in zip(unique_labels, colors, strict=False):
        mask = np.asarray([item == label for item in label_strings], dtype=bool)
        ax.scatter(x_values[mask], y_values[mask], label=label, s=28, alpha=0.8, color=color)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    if unique_labels:
        ax.legend(loc="best", fontsize=8)
    save_figure(fig, output_path)


def _plot_explained_variance(*, layers: list[dict[str, Any]], step_name: str, output_path: Path) -> None:
    x_values = [int(layer.get("layer", 0)) for layer in layers]
    max_components = max(len(layer.get("explained_variance_ratio") or []) for layer in layers)
    fig, ax = new_figure()
    for component_index in range(max_components):
        y_values = []
        for layer in layers:
            explained_variance = list(layer.get("explained_variance_ratio") or [])
            y_values.append(float(explained_variance[component_index]) if component_index < len(explained_variance) else None)
        ax.plot(x_values, y_values, marker="o", label=f"Component {component_index + 1}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Explained variance by layer: {display_name(step_name)}")
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, output_path)
