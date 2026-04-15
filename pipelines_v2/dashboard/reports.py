"""Report artifact browsing.

Responsibilities:
- Resolve an artifact id to its on-disk report root (the directory created by
  ReportSpec when `output_dir` is set). Refuse any path that escapes that root.
- Read `report.json`, `summary.json`, and `assets/manifest.json` into typed
  response models.
- Enumerate `tables/*.json` summary metadata and `results/*.json` file
  descriptors (bytes, name) without reading them into memory.

This module never touches capture tensors or any artifact not rooted under
the report's own output directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pipelines_v2.dashboard.models import (
    ReportDetail,
    ReportFigure,
    ReportResult,
    ReportTableSummary,
)
from pipelines_v2.storage.artifacts import ArtifactManifest


class ReportUnavailable(RuntimeError):
    """Raised when an artifact is not a published report readable from disk."""


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def resolve_report_root(manifest: ArtifactManifest) -> Path:
    """Return the on-disk root directory for a report artifact.

    Uses `manifest.metadata.published_report.output_dir`, which the local
    report runner sets when materializing a report. Remote-only report
    artifacts (no local output) are not browsable by v1 of the dashboard.
    """
    if manifest.artifact_kind != "report":
        raise ReportUnavailable(
            f"artifact {manifest.artifact_id} is kind={manifest.artifact_kind}, not 'report'"
        )
    published = manifest.metadata.get("published_report") if isinstance(manifest.metadata, Mapping) else None
    if not isinstance(published, Mapping):
        raise ReportUnavailable(
            f"artifact {manifest.artifact_id} has no `published_report` metadata; "
            "the report likely wasn't materialized to a local output_dir."
        )
    output_dir = published.get("output_dir")
    if not output_dir:
        raise ReportUnavailable(
            f"artifact {manifest.artifact_id} published_report has no output_dir"
        )
    root = Path(str(output_dir)).resolve()
    if not root.is_dir():
        raise ReportUnavailable(f"report output_dir does not exist: {root}")
    return root


def build_report_detail(manifest: ArtifactManifest) -> ReportDetail:
    root = resolve_report_root(manifest)

    report = _read_json_optional(root / "report.json")
    summary = _read_json_optional(root / "summary.json")
    assets_manifest = _read_json_optional(root / "assets" / "manifest.json") or {}

    figures = _collect_figures(assets_manifest)
    tables = _collect_tables(assets_manifest, root)
    results = _collect_results(root / "results")
    headline = _extract_headline(summary or report)

    workflow_context = manifest.workflow_context if isinstance(manifest.workflow_context, Mapping) else {}
    return ReportDetail(
        artifact_id=manifest.artifact_id,
        artifact_kind=manifest.artifact_kind,
        run_id=str(workflow_context.get("run_id")) if workflow_context.get("run_id") else None,
        report=report,
        summary=summary,
        headline=headline,
        figures=figures,
        tables=tables,
        results=results,
        unsupported_inputs=list(assets_manifest.get("unsupported_inputs", []) or []),
    )


def safe_asset_path(root: Path, relative: str) -> Path:
    """Resolve `relative` against `root`, rejecting any path that escapes."""
    # Strip leading slashes so `/assets/foo.png` doesn't become absolute.
    cleaned = relative.lstrip("/\\")
    candidate = (root / cleaned).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        raise ReportUnavailable(f"asset path escapes report root: {relative}")
    if not candidate.is_file():
        raise ReportUnavailable(f"asset not found: {relative}")
    return candidate


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    return payload


def _collect_figures(assets_manifest: Mapping[str, Any]) -> list[ReportFigure]:
    figures = assets_manifest.get("figures") if isinstance(assets_manifest, Mapping) else None
    if not isinstance(figures, Mapping):
        return []
    out: list[ReportFigure] = []
    for figure_id, record in figures.items():
        if not isinstance(record, Mapping):
            continue
        out.append(
            ReportFigure(
                figure_id=str(figure_id),
                path=str(record.get("path", "")),
                step_name=_optional_str(record.get("step_name")),
                result_kind=_optional_str(record.get("result_kind")),
                chart_kind=_optional_str(record.get("chart_kind")),
                title=_optional_str(record.get("title")),
                caption=_optional_str(record.get("caption")),
                primary=bool(record.get("primary", False)),
            )
        )
    # Primary first, then alphabetical.
    out.sort(key=lambda f: (not f.primary, f.figure_id))
    return out


def _collect_tables(assets_manifest: Mapping[str, Any], root: Path) -> list[ReportTableSummary]:
    tables = assets_manifest.get("tables") if isinstance(assets_manifest, Mapping) else None
    if not isinstance(tables, Mapping):
        return []
    out: list[ReportTableSummary] = []
    for slug, record in tables.items():
        if not isinstance(record, Mapping):
            continue
        path_rel = str(record.get("path", ""))
        rows, columns = _table_shape(root / path_rel)
        out.append(
            ReportTableSummary(
                slug=str(slug),
                step_name=_optional_str(record.get("step_name")),
                result_kind=_optional_str(record.get("result_kind")),
                rows=rows,
                columns=columns,
                path=path_rel,
            )
        )
    out.sort(key=lambda t: t.slug)
    return out


def _table_shape(path: Path) -> tuple[int, list[str]]:
    payload = _read_json_optional(path)
    if not payload:
        return 0, []
    # Table JSON shapes observed from reporting.charts: various. Try a few.
    if isinstance(payload.get("rows"), list):
        rows = payload["rows"]
        columns = list(payload.get("columns", []))
        if not columns and rows and isinstance(rows[0], Mapping):
            columns = list(rows[0].keys())
        return len(rows), [str(c) for c in columns]
    if isinstance(payload.get("records"), list):
        records = payload["records"]
        columns = list(records[0].keys()) if records and isinstance(records[0], Mapping) else []
        return len(records), [str(c) for c in columns]
    return 0, []


def _collect_results(results_dir: Path) -> list[ReportResult]:
    if not results_dir.is_dir():
        return []
    out: list[ReportResult] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        step_name = path.stem.removesuffix("_results")
        out.append(
            ReportResult(
                name=path.name,
                path=f"results/{path.name}",
                step_name=step_name or None,
                bytes=size,
            )
        )
    return out


def _extract_headline(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    # Prefer an explicit `headline_metrics` block; fall back to `step_summaries`
    # flattened by slug -> headline_metrics.
    if isinstance(payload.get("headline_metrics"), Mapping):
        return dict(payload["headline_metrics"])
    step_summaries = payload.get("step_summaries") or payload.get("steps")
    if isinstance(step_summaries, Mapping):
        flat: dict[str, Any] = {}
        for slug, step in step_summaries.items():
            if not isinstance(step, Mapping):
                continue
            metrics = step.get("headline_metrics")
            if isinstance(metrics, Mapping) and metrics:
                flat[str(slug)] = dict(metrics)
        if flat:
            return flat
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
