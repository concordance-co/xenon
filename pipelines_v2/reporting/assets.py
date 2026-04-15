"""Local report asset generation from copied direct-input results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from pipelines_v2.reporting.charts import geometry, probe, residualized, text, transfer

Renderer = Callable[[str, str, dict[str, Any], Path], dict[str, Any]]

_RENDERERS: dict[str, Renderer] = {
    "geometry_result": geometry.render,
    "probe_result": probe.render,
    "residualized_probe_result": residualized.render,
    "text_baseline_result": text.render,
    "transfer_probe_result": transfer.render,
}


def generate_report_assets(*, report_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    assets_dir = report_root / "assets"
    tables_dir = report_root / "tables"
    assets_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    payload_inputs = payload.get("inputs")
    if not isinstance(payload_inputs, list):
        payload_inputs = []

    figure_registry: dict[str, dict[str, Any]] = {}
    table_registry: dict[str, dict[str, Any]] = {}
    step_summaries: dict[str, dict[str, Any]] = {}
    unsupported_inputs: list[dict[str, Any]] = []

    for input_entry in payload_inputs:
        if not isinstance(input_entry, dict):
            continue
        downloaded_result_path = input_entry.get("downloaded_result_path")
        if not downloaded_result_path:
            continue
        result_path = Path(str(downloaded_result_path))
        if not result_path.exists():
            unsupported_inputs.append(
                {
                    "step_name": input_entry.get("name"),
                    "artifact_id": input_entry.get("artifact_id"),
                    "artifact_kind": input_entry.get("artifact_kind"),
                    "downloaded_result_path": str(result_path),
                    "reason": "missing_downloaded_result",
                }
            )
            continue
        result_payload = _read_json(result_path)
        result_kind = str(result_payload.get("kind") or "")
        renderer = _RENDERERS.get(result_kind)
        if renderer is None:
            unsupported_inputs.append(
                {
                    "step_name": input_entry.get("name"),
                    "artifact_id": input_entry.get("artifact_id"),
                    "artifact_kind": input_entry.get("artifact_kind"),
                    "downloaded_result_path": str(result_path),
                    "result_kind": result_kind or None,
                    "reason": "unsupported_result_kind",
                }
            )
            continue

        step_slug = _step_slug_from_result_path(result_path)
        step_name = str(input_entry.get("name") or step_slug)
        rendered = renderer(
            step_name=step_name,
            step_slug=step_slug,
            result=result_payload,
            report_root=report_root,
        )

        figures: list[dict[str, Any]] = []
        for figure in rendered.get("figures", []):
            figure_record = {
                "figure_id": f"{step_slug}/{Path(str(figure['path'])).stem}",
                "path": str(figure["path"]),
                "step_name": step_name,
                "result_kind": result_kind,
                "chart_kind": figure["chart_kind"],
                "title": figure["title"],
                "caption": figure["caption"],
                "primary": bool(figure.get("primary")),
            }
            figure_registry[figure_record["figure_id"]] = figure_record
            figures.append(figure_record)

        table_path = tables_dir / f"{step_slug}.json"
        table_relative_path = str(table_path.relative_to(report_root))
        table_payload = dict(rendered.get("table", {}))
        table_payload.setdefault("step_name", step_name)
        table_payload.setdefault("result_kind", result_kind)
        _write_json(table_path, table_payload)
        table_registry[step_slug] = {
            "path": table_relative_path,
            "step_name": step_name,
            "result_kind": result_kind,
        }

        headline_metrics = dict(rendered.get("headline_metrics", {}))
        input_entry["assets"] = figures
        input_entry["table_path"] = table_relative_path
        input_entry["headline_metrics"] = headline_metrics

        step_summary = {
            "kind": result_kind,
            "headline_metrics": headline_metrics,
            "table_path": table_relative_path,
        }
        primary_figure = next((figure for figure in figures if figure.get("primary")), None)
        if primary_figure is not None:
            step_summary["primary_figure_id"] = primary_figure["figure_id"]
        step_summaries[step_slug] = step_summary

    _run_template_postprocessor(
        template=str(payload.get("template") or "summary"),
        report_root=report_root,
        figure_registry=figure_registry,
        table_registry=table_registry,
        step_summaries=step_summaries,
    )

    manifest = {
        "figures": figure_registry,
        "tables": table_registry,
        "unsupported_inputs": unsupported_inputs,
    }
    manifest_path = assets_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    existing_summary = dict(payload.get("summary", {}))
    summary = {
        **existing_summary,
        "template": payload.get("template", existing_summary.get("template")),
        "input_count": len(payload_inputs),
        "example_count": existing_summary.get("example_count"),
        "figures": figure_registry,
        "tables": table_registry,
        "step_summaries": step_summaries,
    }
    payload["summary"] = summary
    return {
        "manifest_path": str(manifest_path),
        "assets_dir": str(assets_dir),
        "tables_dir": str(tables_dir),
        "manifest": manifest,
        "summary": summary,
    }


def _step_slug_from_result_path(result_path: Path) -> str:
    name = result_path.name
    if name.endswith("_results.json"):
        return name[: -len("_results.json")]
    if name.endswith(".json"):
        return name[:-5]
    return name


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _run_template_postprocessor(
    *,
    template: str,
    report_root: Path,
    figure_registry: dict[str, dict[str, Any]],
    table_registry: dict[str, dict[str, Any]],
    step_summaries: dict[str, dict[str, Any]],
) -> None:
    postprocessor = _TEMPLATE_POSTPROCESSORS.get(template)
    if postprocessor is None:
        return
    postprocessor(
        report_root=report_root,
        figure_registry=figure_registry,
        table_registry=table_registry,
        step_summaries=step_summaries,
    )


def _postprocess_summary_template(
    *,
    report_root: Path,
    figure_registry: dict[str, dict[str, Any]],
    table_registry: dict[str, dict[str, Any]],
    step_summaries: dict[str, dict[str, Any]],
) -> None:
    del report_root, figure_registry, table_registry, step_summaries


_TEMPLATE_POSTPROCESSORS: dict[str, Callable[..., None]] = {
    "summary": _postprocess_summary_template,
}
