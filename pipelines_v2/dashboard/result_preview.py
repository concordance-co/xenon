"""Read a step's result JSON from disk, when available.

Operation artifacts persist a `result` payload — usually at
`storage_refs["result"]` with `store == "local_path"` and a `path`. Modal-
produced artifacts live in a remote volume and aren't readable here; in that
case we fall back to the report's copied `results/{slug}_results.json` if
the run has a report artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pipelines_v2.dashboard.models import ResultPreview
from pipelines_v2.storage.artifacts import ArtifactManifest
from pipelines_v2.dashboard.reports import ReportUnavailable, resolve_report_root


RESULT_PREVIEW_MAX_BYTES = 1_500_000
RESULT_TABLE_PREVIEW_ROWS = 50
RESULT_PAYLOAD_PREVIEW_ITEMS = 50
RESULT_PAYLOAD_PREVIEW_DEPTH = 5


def local_result_path(manifest: ArtifactManifest | None) -> Path | None:
    if manifest is None:
        return None
    refs = manifest.storage_refs if isinstance(manifest.storage_refs, Mapping) else {}
    ref = refs.get("result")
    if not isinstance(ref, Mapping):
        return None
    if ref.get("store") not in {"local_path", "local"}:
        return None
    path_str = ref.get("path")
    if not path_str:
        return None
    path = Path(str(path_str))
    if not path.is_file():
        return None
    return path


def report_copied_result_path(
    *,
    report_manifest: ArtifactManifest | None,
    step_name: str,
) -> Path | None:
    """Fallback: look inside the published report's results/ for this step."""
    if report_manifest is None:
        return None
    published = (
        report_manifest.metadata.get("published_report")
        if isinstance(report_manifest.metadata, Mapping)
        else None
    )
    if not isinstance(published, Mapping):
        return None
    results_dir = published.get("results_dir")
    if not results_dir:
        return None
    root = Path(str(results_dir))
    if not root.is_dir():
        return None
    # Report runner copies each input step's result under a slug matching the
    # step_name. We try both `<step_name>_results.json` and `<step_name>.json`.
    for candidate in (root / f"{step_name}_results.json", root / f"{step_name}.json"):
        if candidate.is_file():
            return candidate
    return None


def read_result_payload(
    *,
    artifact_manifest: ArtifactManifest | None,
    report_manifest: ArtifactManifest | None,
    step_name: str,
) -> ResultPreview:
    path = local_result_path(artifact_manifest)
    if path is None:
        path = report_copied_result_path(
            report_manifest=report_manifest,
            step_name=step_name,
        )
    if path is None:
        if artifact_manifest is None:
            reason = "step has no recorded artifact"
        else:
            reason = (
                "result payload isn't available locally; this artifact stores "
                "its result in a remote volume (e.g. Modal)."
            )
        return ResultPreview(available=False, reason=reason)
    try:
        payload_bytes = path.stat().st_size
    except OSError:
        payload_bytes = None
    if payload_bytes is not None and payload_bytes > RESULT_PREVIEW_MAX_BYTES:
        truncation_reason = (
            "raw result preview skipped because "
            f"{path.name} is {payload_bytes:,} bytes "
            f"(dashboard limit {RESULT_PREVIEW_MAX_BYTES:,})"
        )
        report_headline, report_tables = _preview_from_report(
            report_manifest=report_manifest,
            step_name=step_name,
        )
        return ResultPreview(
            available=True,
            path=str(path),
            bytes=payload_bytes,
            payload={"__dashboard_notice__": truncation_reason},
            headline=report_headline,
            tables=report_tables,
            truncated=True,
            truncation_reason=truncation_reason,
        )
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        return ResultPreview(available=False, reason=f"failed to read {path}: {exc}")

    headline = _extract_headline(payload)
    tables = _extract_tables(payload)
    preview_payload, truncated = _preview_value(payload)
    truncation_reason = (
        "raw result preview was truncated for dashboard safety"
        if truncated
        else None
    )
    return ResultPreview(
        available=True,
        path=str(path),
        bytes=payload_bytes,
        payload=preview_payload if isinstance(preview_payload, dict) else {"value": preview_payload},
        headline=headline,
        tables=tables,
        truncated=truncated,
        truncation_reason=truncation_reason,
    )


def _extract_headline(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    for key in ("headline_metrics", "headline", "metrics"):
        v = payload.get(key)
        if isinstance(v, Mapping) and v:
            return dict(v)
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        scalars = {
            k: v
            for k, v in summary.items()
            if isinstance(v, (int, float, str, bool)) or v is None
        }
        if scalars:
            return scalars
    # Last resort: scalar top-level fields (skip the identity fields).
    skip = {"kind", "schema_version", "artifact_id", "created_at", "operation_spec_hash"}
    scalars = {
        k: v
        for k, v in payload.items()
        if k not in skip
        and (isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) <= 80))
    }
    if scalars:
        return scalars
    return None


def _extract_tables(payload: Any) -> list[dict[str, Any]]:
    """Find array-of-object-looking things worth rendering as tables."""
    if not isinstance(payload, Mapping):
        return []
    tables: list[dict[str, Any]] = []

    # Look for common table shapes at known keys first.
    for key in (
        "rows",
        "records",
        "folds",
        "per_class",
        "per_fold",
        "cases",
        "examples",
    ):
        v = payload.get(key)
        if _looks_like_rows(v):
            rows = v
            tables.append(_table_preview(name=key, rows=rows))

    # Also scan one level deep into `summary`, `results`, `per_*` etc.
    for key, v in payload.items():
        if key in {"rows", "records", "folds", "per_class", "per_fold", "cases", "examples"}:
            continue
        if isinstance(v, list) and _looks_like_rows(v):
            tables.append(_table_preview(name=key, rows=v))
        elif isinstance(v, Mapping):
            for inner_key, inner_v in v.items():
                if _looks_like_rows(inner_v):
                    tables.append(_table_preview(name=f"{key}.{inner_key}", rows=inner_v))

    # Deduplicate by name, keep first occurrence.
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for t in tables:
        if t["name"] in seen:
            continue
        seen.add(t["name"])
        out.append(t)
    return out


def _looks_like_rows(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    return isinstance(first, Mapping) and len(first) >= 1


def _table_preview(*, name: str, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    preview_rows = [dict(row) for row in rows[:RESULT_TABLE_PREVIEW_ROWS]]
    columns = [str(column) for column in preview_rows[0].keys()] if preview_rows else []
    return {
        "name": name,
        "rows": preview_rows,
        "columns": columns,
        "total_rows": len(rows),
        "truncated": len(rows) > len(preview_rows),
    }


def _preview_value(value: Any, *, depth: int = 0) -> tuple[Any, bool]:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value, False
    if depth >= RESULT_PAYLOAD_PREVIEW_DEPTH:
        return "<truncated: maximum preview depth reached>", True
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        truncated = False
        items = list(value.items())
        for index, (key, child) in enumerate(items):
            if index >= RESULT_PAYLOAD_PREVIEW_ITEMS:
                out["__dashboard_notice__"] = (
                    f"truncated {len(items) - RESULT_PAYLOAD_PREVIEW_ITEMS} additional keys"
                )
                truncated = True
                break
            preview_child, child_truncated = _preview_value(child, depth=depth + 1)
            out[str(key)] = preview_child
            truncated = truncated or child_truncated
        return out, truncated
    if isinstance(value, list):
        preview_items: list[Any] = []
        truncated = False
        for item in value[:RESULT_PAYLOAD_PREVIEW_ITEMS]:
            preview_item, item_truncated = _preview_value(item, depth=depth + 1)
            preview_items.append(preview_item)
            truncated = truncated or item_truncated
        if len(value) > RESULT_PAYLOAD_PREVIEW_ITEMS:
            preview_items.append(
                f"<truncated: {len(value) - RESULT_PAYLOAD_PREVIEW_ITEMS} additional items>"
            )
            truncated = True
        return preview_items, truncated
    return repr(value), True


def _preview_from_report(
    *,
    report_manifest: ArtifactManifest | None,
    step_name: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if report_manifest is None:
        return None, []
    try:
        root = resolve_report_root(report_manifest)
    except ReportUnavailable:
        return None, []
    summary = _read_json_optional(root / "summary.json") or _read_json_optional(root / "report.json")
    headline = _extract_step_headline(summary, step_name=step_name)
    assets_manifest = _read_json_optional(root / "assets" / "manifest.json") or {}
    tables = _preview_tables_from_report_manifest(
        root=root,
        assets_manifest=assets_manifest,
        step_name=step_name,
    )
    return headline, tables


def _extract_step_headline(
    payload: Mapping[str, Any] | None,
    *,
    step_name: str,
) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    step_summaries = payload.get("step_summaries") or payload.get("steps")
    if isinstance(step_summaries, Mapping):
        direct = step_summaries.get(step_name)
        if isinstance(direct, Mapping):
            metrics = direct.get("headline_metrics")
            if isinstance(metrics, Mapping) and metrics:
                return dict(metrics)
        for item in step_summaries.values():
            if not isinstance(item, Mapping):
                continue
            if item.get("step_name") != step_name:
                continue
            metrics = item.get("headline_metrics")
            if isinstance(metrics, Mapping) and metrics:
                return dict(metrics)
    return None


def _preview_tables_from_report_manifest(
    *,
    root: Path,
    assets_manifest: Mapping[str, Any],
    step_name: str,
) -> list[dict[str, Any]]:
    tables = assets_manifest.get("tables") if isinstance(assets_manifest, Mapping) else None
    if not isinstance(tables, Mapping):
        return []
    previews: list[dict[str, Any]] = []
    for slug, record in tables.items():
        if not isinstance(record, Mapping):
            continue
        if str(slug) != step_name and record.get("step_name") != step_name:
            continue
        rows = record.get("rows")
        columns = record.get("columns")
        path_rel = record.get("path")
        payload = _read_json_optional(root / str(path_rel)) if path_rel else None
        preview = _table_preview_from_payload(
            name=str(slug),
            payload=payload,
            total_rows=int(rows) if isinstance(rows, int) else None,
            columns=[str(column) for column in columns] if isinstance(columns, list) else None,
        )
        if preview is not None:
            previews.append(preview)
    return previews


def _table_preview_from_payload(
    *,
    name: str,
    payload: Mapping[str, Any] | None,
    total_rows: int | None,
    columns: list[str] | None,
) -> dict[str, Any] | None:
    if isinstance(payload, Mapping):
        rows = payload.get("rows")
        if _looks_like_rows(rows):
            preview = _table_preview(name=name, rows=rows)
            if total_rows is not None:
                preview["total_rows"] = total_rows
            if columns:
                preview["columns"] = columns
            return preview
        records = payload.get("records")
        if _looks_like_rows(records):
            preview = _table_preview(name=name, rows=records)
            if total_rows is not None:
                preview["total_rows"] = total_rows
            if columns:
                preview["columns"] = columns
            return preview
    if total_rows is None and not columns:
        return None
    return {
        "name": name,
        "rows": [],
        "columns": columns or [],
        "total_rows": total_rows,
        "truncated": bool(total_rows),
    }


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    return payload
