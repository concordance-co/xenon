"""Shared chart styling for report-side asset generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

_STYLE_APPLIED = False

_METRIC_TITLES = {
    "accuracy": "Accuracy",
    "auroc": "AUROC",
    "balanced_accuracy": "Balanced Accuracy",
    "selectivity": "Selectivity",
}


def ensure_style() -> None:
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.edgecolor": "#364152",
            "axes.facecolor": "#ffffff",
            "axes.grid": True,
            "axes.labelcolor": "#111827",
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.facecolor": "#ffffff",
            "font.size": 10.0,
            "grid.alpha": 0.24,
            "grid.color": "#9ca3af",
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "savefig.facecolor": "#ffffff",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
        }
    )
    _STYLE_APPLIED = True


def new_figure(*, figsize: tuple[float, float] = (7.2, 4.2)) -> tuple[Any, Any]:
    ensure_style()
    return plt.subplots(figsize=figsize)


def save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def display_name(value: str | None) -> str:
    if value is None:
        return "Unnamed"
    text = str(value).strip().replace("_", " ")
    if not text:
        return "Unnamed"
    return " ".join(token[:1].upper() + token[1:] for token in text.split())


def display_metric(metric: str) -> str:
    return _METRIC_TITLES.get(str(metric), display_name(str(metric)))


def slugify(value: str | None) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._-")
    return text or "item"


def categorical_colors(count: int) -> list[Any]:
    if count <= 0:
        return []
    cmap_name = "tab10" if count <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)
    return [cmap(index / max(count - 1, 1)) for index in range(count)]


def metric_value(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    return float(value)


def first_two_components(components: list[list[float]] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(components, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    x_values = matrix[:, 0]
    if matrix.shape[1] >= 2:
        y_values = matrix[:, 1]
    else:
        y_values = np.zeros(matrix.shape[0], dtype=np.float64)
    return x_values, y_values
