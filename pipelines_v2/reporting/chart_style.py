"""Shared chart styling for report-side asset generation."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

_STYLE_APPLIED_THEME: str | None = None
_DEFAULT_THEME = "light"
_CARD_RECT = (0.04, 0.035, 0.92, 0.925)
_CARD_TITLE_X = 0.075
_CARD_PLOT_RECT = (0.145, 0.13, 0.79, 0.60)
_CARD_HERO_PLOT_RECT = (0.145, 0.13, 0.79, 0.44)

_METRIC_TITLES = {
    "accuracy": "Accuracy",
    "auroc": "AUROC",
    "balanced_accuracy": "Balanced Accuracy",
    "selectivity": "Selectivity",
}


@dataclass(frozen=True)
class ChartTheme:
    name: str
    page_bg: str
    card_bg: str
    border: str
    text: str
    muted: str
    accent: str
    amber: str
    blue: str
    purple: str
    grid: str

    @property
    def cycle(self) -> list[str]:
        return [self.amber, self.blue, self.purple, self.accent]


_LIGHT_THEME = ChartTheme(
    name="light",
    page_bg="#eef1f5",
    card_bg="#ffffff",
    border="#dde2ea",
    text="#1a2236",
    muted="#6c7689",
    accent="#3a7bc4",
    amber="#b8722e",
    blue="#2c6fa5",
    purple="#6b5d9c",
    grid="#dde2ea",
)

_DARK_THEME = ChartTheme(
    name="dark",
    page_bg="#0d1320",
    card_bg="#161c2a",
    border="#222a3d",
    text="#e8ecf4",
    muted="#5d6b85",
    accent="#5e9cc8",
    amber="#d49758",
    blue="#5e9cc8",
    purple="#8b7fb5",
    grid="#222a3d",
)

_THEMES = {
    _LIGHT_THEME.name: _LIGHT_THEME,
    _DARK_THEME.name: _DARK_THEME,
}


def theme_colors(theme: str = _DEFAULT_THEME) -> ChartTheme:
    return _THEMES.get(str(theme), _LIGHT_THEME)


def ensure_style(theme: str = _DEFAULT_THEME) -> ChartTheme:
    global _STYLE_APPLIED_THEME
    colors = theme_colors(theme)
    if _STYLE_APPLIED_THEME == colors.name:
        return colors
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.edgecolor": colors.border,
            "axes.facecolor": colors.card_bg,
            "axes.grid": False,
            "axes.labelcolor": colors.muted,
            "axes.linewidth": 0.8,
            "axes.prop_cycle": cycler("color", colors.cycle),
            "axes.spines.bottom": False,
            "axes.spines.left": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.facecolor": colors.card_bg,
            "font.family": ["DejaVu Sans Mono", "monospace"],
            "font.size": 9.0,
            "grid.color": colors.grid,
            "grid.linewidth": 0.5,
            "legend.fontsize": 8.0,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.6,
            "savefig.facecolor": colors.card_bg,
            "text.color": colors.text,
            "xtick.color": colors.muted,
            "xtick.labelsize": 8.0,
            "xtick.major.size": 0.0,
            "ytick.color": colors.muted,
            "ytick.labelsize": 8.0,
            "ytick.major.size": 0.0,
        }
    )
    _STYLE_APPLIED_THEME = colors.name
    return colors


def new_figure(
    *,
    figsize: tuple[float, float] = (8.0, 5.0),
    title: str | None = None,
    subtitle: str | None = None,
    detail: str | None = None,
    metric_label: str | None = None,
    metric_value: str | None = None,
    right_label: str | None = None,
    right_value: str | None = None,
    theme: str = _DEFAULT_THEME,
    accent_color: str | None = None,
    plot_rect: tuple[float, float, float, float] | None = None,
) -> tuple[Any, Any]:
    colors = ensure_style(theme)
    fig = plt.figure(figsize=figsize, facecolor=colors.card_bg)
    _draw_card(fig=fig, colors=colors, accent_color=accent_color)

    title_text = _wrap_text(title or "Chart", width=34)
    title_line_count = title_text.count("\n") + 1 if title_text else 1
    subtitle_y = 0.852 - (0.052 * max(title_line_count - 1, 0))
    info_y = subtitle_y - 0.047 if subtitle else 0.818 - (0.040 * max(title_line_count - 1, 0))
    value_y = info_y - 0.095
    fig.text(
        _CARD_TITLE_X,
        0.905,
        title_text,
        color=colors.text,
        fontsize=14.0,
        fontweight="bold",
        family="DejaVu Sans Mono",
        va="top",
    )

    if subtitle:
        fig.text(
            _CARD_TITLE_X,
            subtitle_y,
            subtitle,
            color=accent_color or colors.accent,
            fontsize=8.6,
            family="DejaVu Sans Mono",
            va="top",
        )

    metric_value_artist = None
    if metric_value is not None:
        if metric_label:
            fig.text(
                _CARD_TITLE_X,
                info_y,
                metric_label,
                color=colors.muted,
                fontsize=8.0,
                family="DejaVu Sans Mono",
                va="top",
            )
        if right_label:
            fig.text(
                0.93,
                info_y,
                right_label,
                color=colors.muted,
                fontsize=8.0,
                family="DejaVu Sans Mono",
                ha="right",
                va="top",
            )
        metric_value_artist = fig.text(
            _CARD_TITLE_X,
            value_y,
            metric_value,
            color=colors.text,
            fontsize=28.0,
            family="DejaVu Sans",
            fontweight="light",
            va="top",
        )
        if right_value:
            fig.text(
                0.93,
                value_y + 0.030,
                right_value,
                color=colors.text,
                fontsize=13.0,
                family="DejaVu Sans",
                ha="right",
                va="top",
            )
    elif detail:
        fig.text(
            _CARD_TITLE_X,
            info_y,
            _wrap_text(detail, width=62),
            color=colors.muted,
            fontsize=8.0,
            family="DejaVu Sans Mono",
            va="top",
        )

    ax = fig.add_axes(list(plot_rect or (_CARD_HERO_PLOT_RECT if metric_value is not None else _CARD_PLOT_RECT)))
    ax.set_zorder(5)
    ax.patch.set_zorder(5)
    setattr(
        ax,
        "_xenon_header_layout",
        {
            "metric_value_artist": metric_value_artist,
        },
    )
    style_axes(ax, theme=theme)
    return fig, ax


def save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def style_axes(
    ax: Any,
    *,
    theme: str = _DEFAULT_THEME,
    xlabel: str | None = None,
    ylabel: str | None = None,
    x_values: Sequence[int | float] | None = None,
    layer_axis: bool = False,
    metric_axis: bool = False,
    y_limits: tuple[float, float] | None = None,
    xscale: str | None = None,
) -> None:
    colors = ensure_style(theme)
    ax.set_facecolor(colors.card_bg)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="major", length=4, width=0.7, color=colors.border, pad=3)
    ax.grid(False)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=colors.muted, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=colors.muted, labelpad=8)
    if xscale:
        ax.set_xscale(xscale)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if metric_axis:
        lower, upper = y_limits or ax.get_ylim()
        if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
            padding = max(abs(float(lower) if np.isfinite(lower) else 0.0) * 0.1, 0.05)
            lower = float(lower) - padding if np.isfinite(lower) else -padding
            upper = float(upper) + padding if np.isfinite(upper) else padding
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.2f}"))
    if layer_axis and x_values:
        layer_ticks = _sample_ticks([int(value) for value in x_values], desired=6)
        ax.set_xticks(layer_ticks)
        ax.set_xticklabels([format_layer(layer) for layer in layer_ticks], color=colors.muted, fontsize=8.0)


def value_limits(
    values: Sequence[float | None],
    *,
    padding_ratio: float = 0.12,
    minimum_padding: float = 0.05,
    include_zero: bool = False,
) -> tuple[float, float] | None:
    numeric_values: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if not np.isfinite(numeric):
            continue
        numeric_values.append(numeric)
    if not numeric_values:
        return None
    lower = min(numeric_values)
    upper = max(numeric_values)
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    if lower == upper:
        padding = max(abs(lower) * padding_ratio, minimum_padding)
        return (lower - padding, upper + padding)
    padding = max((upper - lower) * padding_ratio, minimum_padding)
    return (lower - padding, upper + padding)


def header_legend(
    ax: Any,
    *,
    theme: str = _DEFAULT_THEME,
    ncol: int = 1,
) -> Any | None:
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return None
    colors = ensure_style(theme)
    layout = getattr(ax, "_xenon_header_layout", {}) or {}
    metric_value_artist = layout.get("metric_value_artist")
    if metric_value_artist is not None:
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        bbox_display = metric_value_artist.get_window_extent(renderer=renderer)
        bbox_figure = bbox_display.transformed(ax.figure.transFigure.inverted())
        anchor_x = min(float(bbox_figure.x1) + 0.018, 0.72)
        anchor_y = float((bbox_figure.y0 + bbox_figure.y1) * 0.5)
        loc = "center left"
    else:
        position = ax.get_position()
        anchor_x = position.x1 - 0.01
        anchor_y = min(0.82, position.y1 + (0.13 if position.height <= 0.50 else 0.06))
        loc = "upper right"
    legend = ax.figure.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=(anchor_x, anchor_y),
        bbox_transform=ax.figure.transFigure,
        ncol=ncol,
        frameon=False,
        fontsize=8.0,
        handlelength=1.6,
        handletextpad=0.6,
        columnspacing=0.9,
    )
    for text in legend.get_texts():
        text.set_color(colors.muted)
        text.set_fontfamily("DejaVu Sans Mono")
    return legend


def plot_series(
    ax: Any,
    x_values: Sequence[float | int],
    y_values: Sequence[float | None],
    *,
    label: str | None = None,
    color: str | None = None,
    dashed: bool = False,
    theme: str = _DEFAULT_THEME,
) -> Any:
    colors = ensure_style(theme)
    line_style = (0, (3, 2)) if dashed else "-"
    plotted = ax.plot(
        x_values,
        y_values,
        linestyle=line_style,
        color=color,
        linewidth=1.45,
        marker="o",
        markersize=4.4,
        markeredgewidth=0.85,
        markeredgecolor=colors.card_bg,
        label=label,
        zorder=4,
    )
    return plotted[0]


def style_bars(ax: Any, bars: Any, *, theme: str = _DEFAULT_THEME) -> None:
    colors = categorical_colors(len(bars), theme=theme)
    for bar, color in zip(bars, colors, strict=False):
        bar.set_facecolor(color)
        bar.set_edgecolor("none")
        bar.set_alpha(0.95)


def annotate_bars(ax: Any, bars: Any, *, theme: str = _DEFAULT_THEME, digits: int = 3) -> None:
    colors = ensure_style(theme)
    upper = ax.get_ylim()[1]
    offset = max((upper - ax.get_ylim()[0]) * 0.02, 0.02)
    for bar in bars:
        height = float(bar.get_height())
        if not np.isfinite(height):
            continue
        ax.text(
            bar.get_x() + (bar.get_width() / 2.0),
            height + offset,
            format_stat(height, digits=digits),
            ha="center",
            va="bottom",
            color=colors.muted,
            fontsize=7.0,
            family="DejaVu Sans Mono",
        )


def highlight_point(ax: Any, x_value: float | int, y_value: float, *, color: str | None = None, theme: str = _DEFAULT_THEME) -> None:
    colors = ensure_style(theme)
    ax.axvline(
        x_value,
        color=colors.muted,
        linewidth=0.75,
        linestyle=(0, (1, 2)),
        alpha=0.75,
        zorder=2,
    )
    ax.plot(
        x_value,
        y_value,
        "o",
        color=color,
        markersize=8.5,
        markeredgecolor=colors.card_bg,
        markeredgewidth=1.8,
        zorder=6,
    )


def horizontal_reference(ax: Any, y_value: float, *, theme: str = _DEFAULT_THEME) -> None:
    colors = ensure_style(theme)
    ax.axhline(
        y_value,
        color=colors.muted,
        linewidth=0.85,
        linestyle=(0, (1, 2)),
        alpha=0.75,
        zorder=2,
    )


def format_stat(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def format_layer(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"L{int(value)}"


def format_regularization(value: float | None) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if number == 0.0:
        return "0"
    if abs(number) >= 1.0:
        return f"{number:.2f}".rstrip("0").rstrip(".")
    return f"{number:.3g}"


def best_point(x_values: Sequence[int | float], y_values: Sequence[float | None]) -> tuple[int | float | None, float | None]:
    ranked = []
    for x_value, y_value in zip(x_values, y_values, strict=False):
        if y_value is None:
            continue
        numeric = float(y_value)
        if not np.isfinite(numeric):
            continue
        ranked.append((numeric, x_value))
    if not ranked:
        return None, None
    value, x_value = max(ranked, key=lambda item: item[0])
    return x_value, value


def best_named_value(values: Mapping[str, float | None]) -> tuple[str | None, float | None]:
    ranked = []
    for label, value in values.items():
        if value is None:
            continue
        numeric = float(value)
        if not np.isfinite(numeric):
            continue
        ranked.append((numeric, label))
    if not ranked:
        return None, None
    value, label = max(ranked, key=lambda item: item[0])
    return label, value


def best_series_point(
    series_map: Mapping[str, Sequence[tuple[int | float, float | None]]],
) -> tuple[str | None, int | float | None, float | None]:
    ranked = []
    for label, pairs in series_map.items():
        for x_value, y_value in pairs:
            if y_value is None:
                continue
            numeric = float(y_value)
            if not np.isfinite(numeric):
                continue
            ranked.append((numeric, label, x_value))
    if not ranked:
        return None, None, None
    value, label, x_value = max(ranked, key=lambda item: item[0])
    return label, x_value, value


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


def categorical_colors(count: int, *, theme: str = _DEFAULT_THEME) -> list[Any]:
    if count <= 0:
        return []
    colors = theme_colors(theme)
    palette = list(colors.cycle)
    if count <= len(palette):
        return palette[:count]
    cmap_name = "tab10" if count <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)
    extra_needed = count - len(palette)
    extra = [cmap(index / max(extra_needed - 1, 1)) for index in range(extra_needed)]
    return palette + extra


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


def _draw_card(fig: Any, *, colors: ChartTheme, accent_color: str | None = None) -> None:
    del fig, colors, accent_color


def _sample_ticks(values: Sequence[int], *, desired: int) -> list[int]:
    ordered = sorted(set(values))
    if len(ordered) <= desired:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, desired).round().astype(int)
    return [ordered[index] for index in sorted(set(int(item) for item in positions))]


def _wrap_text(value: str, *, width: int) -> str:
    wrapped = textwrap.wrap(str(value), width=width)
    if not wrapped:
        return ""
    if len(wrapped) <= 2:
        return "\n".join(wrapped)
    return "\n".join([wrapped[0], textwrap.shorten(" ".join(wrapped[1:]), width=width, placeholder="...")])
