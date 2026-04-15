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


def local_result_path(manifest: ArtifactManifest | None) -> Path | None:
    if manifest is None:
        return None
    refs = manifest.storage_refs if isinstance(manifest.storage_refs, Mapping) else {}
    ref = refs.get("result")
    if not isinstance(ref, Mapping):
        return None
    if ref.get("store") != "local_path":
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
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        return ResultPreview(available=False, reason=f"failed to read {path}: {exc}")

    headline = _extract_headline(payload)
    tables = _extract_tables(payload)
    return ResultPreview(
        available=True,
        path=str(path),
        payload=payload if isinstance(payload, dict) else {"value": payload},
        headline=headline,
        tables=tables,
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
            tables.append({"name": key, "rows": list(v)})

    # Also scan one level deep into `summary`, `results`, `per_*` etc.
    for key, v in payload.items():
        if key in {"rows", "records", "folds", "per_class", "per_fold", "cases", "examples"}:
            continue
        if isinstance(v, list) and _looks_like_rows(v):
            tables.append({"name": key, "rows": list(v)})
        elif isinstance(v, Mapping):
            for inner_key, inner_v in v.items():
                if _looks_like_rows(inner_v):
                    tables.append({"name": f"{key}.{inner_key}", "rows": list(inner_v)})

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
