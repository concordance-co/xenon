"""Generation-result chart and prompt/response table generation."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from pipelines_v2.reporting.chart_style import (
    display_name,
    format_stat,
    new_figure,
    save_figure,
    style_axes,
    theme_colors,
)

_PREFERRED_LABEL_KEYS = (
    "label",
    "target",
    "case_id",
    "case_key",
    "condition",
    "family",
    "authority_cell",
    "expected_selected_source",
    "positive_authority_risk",
    "instruction_uptake_allowed",
    "domain",
    "outcome",
)

_TABLE_COLUMNS = [
    "example_key",
    "case_key",
    "finish_reason",
    "label_summary",
    "probe_scores",
    "span_names",
    "prompt",
    "response",
    "prompt_hash",
    "labels",
    "spans",
]


def render(*, step_name: str, step_slug: str, result: dict[str, Any], report_root: Path) -> dict[str, Any]:
    raw_rows = [dict(item) for item in result.get("rows", []) if isinstance(item, Mapping)]
    rows = [_inspection_row(item) for item in raw_rows]
    summary = dict(result.get("summary", {})) if isinstance(result.get("summary"), Mapping) else {}

    figures: list[dict[str, Any]] = []
    lengths = [_response_length(row) for row in rows]
    if any(length is not None for length in lengths):
        asset_dir = report_root / "assets" / step_slug
        filename = "response_lengths.png"
        _plot_response_lengths(
            rows=rows,
            step_name=step_name,
            output_path=asset_dir / filename,
        )
        figures.append(
            {
                "path": f"assets/{step_slug}/{filename}",
                "chart_kind": "generation_response_lengths",
                "title": "Generated response lengths",
                "caption": f"Generated response character counts for step {step_name}.",
                "primary": True,
            }
        )

    response_lengths = [length for length in lengths if length is not None]
    headline_metrics = {
        "example_count": summary.get("example_count", len(rows)),
        "completed_example_count": summary.get("completed_example_count"),
        "total_example_count": summary.get("total_example_count"),
        "partial": summary.get("partial"),
        "avg_response_chars": mean(response_lengths) if response_lengths else None,
    }
    return {
        "result_kind": "generation_run_result",
        "figures": figures,
        "table": {
            "step_name": step_name,
            "result_kind": "generation_run_result",
            "columns": _table_columns(rows),
            "summary": summary,
            "headline_metrics": headline_metrics,
            "rows": rows,
        },
        "headline_metrics": headline_metrics,
    }


def _inspection_row(row: Mapping[str, Any]) -> dict[str, Any]:
    example = _as_mapping(row.get("example"))
    labels = _as_mapping(example.get("labels"))
    metadata = _as_mapping(example.get("metadata"))
    spans = _span_rows(metadata.get("span_specs"))
    scores = _score_mapping(row)
    prompt = row.get("prompt", example.get("prompt"))
    response = _first_present(
        row,
        "generated_text",
        "response",
        "completion",
        "output",
        "text",
    )

    return {
        "example_key": str(row.get("example_key") or example.get("key") or ""),
        "case_key": _optional_str(example.get("case_key") or labels.get("case_key") or labels.get("case_id")),
        "finish_reason": _optional_str(row.get("finish_reason")),
        "label_summary": _label_summary(labels),
        "probe_scores": scores,
        "span_names": ", ".join(_span_name(span) for span in spans if _span_name(span)),
        "prompt": _prompt_text(prompt),
        "response": _stringify_text(response),
        "prompt_hash": _optional_str(example.get("prompt_hash") or row.get("prompt_hash")),
        "labels": dict(labels),
        "spans": spans,
    }


def _span_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            {
                "name": _optional_str(item.get("name")),
                "span_label": _optional_str(item.get("span_label")),
                "source_type": _optional_str(item.get("source_type")),
                "sender": _optional_str(item.get("sender")),
                "provenance": _optional_str(item.get("provenance")),
                "assigned_authority": _optional_str(item.get("assigned_authority")),
                "instruction_like": item.get("instruction_like"),
                "content_text": _stringify_text(item.get("content_text")),
            }
        )
    return rows


def _span_name(span: Mapping[str, Any]) -> str:
    return str(span.get("name") or span.get("span_label") or "").strip()


def _score_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("probe_scores", "probeScores", "scores", "metrics"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    if row.get("score") is not None:
        return {"score": row.get("score")}
    return {}


def _label_summary(labels: Mapping[str, Any]) -> str:
    if not labels:
        return ""
    selected: list[tuple[str, Any]] = []
    for key in _PREFERRED_LABEL_KEYS:
        if key in labels:
            selected.append((key, labels[key]))
    if not selected:
        for key, value in labels.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                selected.append((str(key), value))
            if len(selected) >= 5:
                break
    return " | ".join(f"{key}={_stringify_scalar(value)}" for key, value in selected[:8])


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: list[str] = []
        for item in prompt:
            if isinstance(item, Mapping):
                role = str(item.get("role") or "message")
                content = _stringify_text(item.get("content"))
                parts.append(f"{role}: {content}")
            else:
                parts.append(_stringify_text(item))
        return "\n\n".join(part for part in parts if part)
    return _stringify_text(prompt)


def _plot_response_lengths(*, rows: list[dict[str, Any]], step_name: str, output_path: Path) -> None:
    lengths = [len(str(row.get("response") or "")) for row in rows if row.get("response")]
    if not lengths:
        return
    avg_length = mean(lengths)
    fig, ax = new_figure(
        title=display_name(step_name),
        subtitle="GENERATION",
        metric_label="ROWS",
        metric_value=format_stat(float(len(lengths)), digits=0),
        right_label="AVG CHARS",
        right_value=format_stat(float(avg_length), digits=0),
    )
    colors = theme_colors()
    bins = min(24, max(4, int(math.sqrt(len(lengths)))))
    ax.hist(lengths, bins=bins, color=colors.blue, edgecolor=colors.border, linewidth=0.7)
    style_axes(ax, xlabel="Generated characters", ylabel="Rows")
    save_figure(fig, output_path)


def _table_columns(rows: list[dict[str, Any]]) -> list[str]:
    extra = sorted({key for row in rows for key in row if key not in set(_TABLE_COLUMNS)})
    return [column for column in _TABLE_COLUMNS if any(row.get(column) not in (None, "", {}, []) for row in rows)] + extra


def _response_length(row: Mapping[str, Any]) -> int | None:
    value = row.get("response")
    if isinstance(value, str) and value:
        return len(value)
    return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _stringify_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
