"""Normalization for /api/runs/{run_id}/steps/{step_name}.

Produces the step detail payload used by the inspector's Overview/Spec/Inputs/
Artifacts/Results tabs. Does NOT localize any capture tensors — only metadata
that is already present in the catalog.
"""

from __future__ import annotations

from typing import Any, Mapping  # noqa: F401

from pipelines_v2.dashboard.models import (
    ArtifactSummary,
    ResolvedDep,
    ResultSummary,
    StepDetail,
    StepSummary,
)
from pipelines_v2.dashboard.normalize import build_run_detail
from pipelines_v2.storage.artifacts import ArtifactManifest
from pipelines_v2.workflow.records import WorkflowRunRecord, WorkflowStepRecord
from pipelines_v2.workflow.specs import WorkflowSpec


def build_step_detail(
    *,
    run: WorkflowRunRecord,
    step_records: list[WorkflowStepRecord],
    target_step: str,
    artifact_manifest: ArtifactManifest | None,
    report_artifact_id: str | None = None,
) -> StepDetail:
    """Shape a single step's detail response."""
    run_detail = build_run_detail(run, step_records)
    target = next((s for s in run_detail.steps if s.step_name == target_step), None)
    if target is None:
        raise LookupError(f"Unknown step: {target_step}")

    # Pull the raw spec from the workflow payload for the normalized spec
    # summary + raw JSON viewer.
    raw_spec = _lookup_raw_spec(run.workflow_payload, target_step)

    # Resolve dependency edges (both directions) against the full step index
    by_name = {s.step_name: s for s in run_detail.steps}
    upstream = _resolve_deps(by_name, target.resolved_depends_on)
    downstream_names = [s.step_name for s in run_detail.steps if target_step in s.resolved_depends_on]
    downstream = _resolve_deps(by_name, downstream_names)

    artifact_summary = _summarize_manifest(artifact_manifest) if artifact_manifest is not None else None

    # Phase C keeps result_summary minimal — actual headline / tables parsing
    # lives behind the report endpoint (Phase F) because it requires reading
    # files from disk. Expose availability flag so the UI can link out.
    has_result_ref = artifact_summary is not None and _has_storage_ref(
        artifact_summary.storage_refs, "result"
    )
    result_summary: ResultSummary | None = None
    if artifact_summary is not None:
        result_summary = ResultSummary(headline=None, tables=[], raw_available=has_result_ref)

    has_capture = _has_capture_reachable(run, target_step)

    return StepDetail(
        step=target,
        spec=raw_spec,
        spec_summary=_summarize_spec(raw_spec),
        upstream=upstream,
        downstream=downstream,
        artifact=artifact_summary,
        result_summary=result_summary,
        report_artifact_id=report_artifact_id,
        has_prompt=has_capture,
        has_dataset=has_capture,
        has_labels=has_capture,
        has_artifact=artifact_summary is not None,
        # Show the Results tab when anything useful is reachable: a stored
        # result ref, a downstream report artifact (which copies results), or
        # a report_artifact_id on the step itself.
        has_results=has_result_ref or report_artifact_id is not None,
    )


def _has_capture_reachable(run: WorkflowRunRecord, target_step: str) -> bool:
    """True if the target step owns a dataset or has a capture ancestor.

    Walks the raw `workflow_payload` dict so it keeps working when
    `WorkflowSpec.from_dict` can't fully rehydrate (e.g. a run persisted by a
    remote process that uses op kinds the local interpreter hasn't imported).
    """
    payload = run.workflow_payload
    if not isinstance(payload, Mapping):
        return False
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return False

    # Index the raw payload: step_name -> (kind, depends_on, spec_dict).
    index: dict[str, tuple[str | None, tuple[str, ...], Mapping[str, Any]]] = {}
    for entry in steps:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        spec = entry.get("spec")
        kind = spec.get("kind") if isinstance(spec, Mapping) else None
        declared = entry.get("depends_on") or ()
        deps_from_declared = tuple(str(d) for d in declared if isinstance(d, str))
        deps_inferred = _collect_step_refs(spec if isinstance(spec, Mapping) else {})
        index[name] = (
            str(kind) if kind is not None else None,
            tuple(sorted(set(deps_from_declared) | deps_inferred)),
            spec if isinstance(spec, Mapping) else {},
        )

    if target_step not in index:
        return False

    # BFS over ancestors. Any capture on the closure makes the tab applicable.
    visited: set[str] = set()
    stack = [target_step]
    while stack:
        name = stack.pop()
        if name in visited or name not in index:
            continue
        visited.add(name)
        kind, deps, spec = index[name]
        if kind == "capture":
            return True
        # A step might embed a dataset directly without being tagged capture
        # (unlikely but defensive).
        if isinstance(spec.get("dataset"), Mapping):
            return True
        stack.extend(deps)
    return False


def _collect_step_refs(value: Any) -> set[str]:
    """Walk a raw spec dict and return referenced step names via StepRef-like
    payloads. Mirrors the ref shapes produced by `to_dict()` without importing
    them, so it works on payloads produced by unknown op kinds."""
    out: set[str] = set()
    if isinstance(value, Mapping):
        if "step" in value and isinstance(value.get("step"), str) and (
            "output" in value or "feature_name" in value or "label_name" in value
        ):
            out.add(str(value["step"]))
            return out
        for v in value.values():
            out.update(_collect_step_refs(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            out.update(_collect_step_refs(v))
    return out


def _lookup_raw_spec(workflow_payload: Mapping[str, Any], step_name: str) -> dict[str, Any]:
    steps = workflow_payload.get("steps", ()) if isinstance(workflow_payload, Mapping) else ()
    for step in steps:
        if isinstance(step, Mapping) and step.get("name") == step_name:
            spec = step.get("spec")
            if isinstance(spec, Mapping):
                return dict(spec)
    return {}


_NOISY_FIELDS = frozenset({"schema_version"})


def _summarize_spec(spec: Mapping[str, Any]) -> list[dict[str, str]]:
    """Produce a human-readable summary of a spec dict.

    Skips empty collections, Nones, and noisy housekeeping fields. Renders
    references (step refs, dataset label/case refs, token selectors,
    pooling) in their constructor-like short form so the summary reads like
    the Python that built the spec.
    """
    items: list[dict[str, str]] = []
    if "kind" in spec:
        items.append({"label": "kind", "value": str(spec["kind"])})
    for key, value in spec.items():
        if key in {"kind", *_NOISY_FIELDS}:
            continue
        if _is_trivial(value):
            continue
        items.append({"label": key, "value": _format_value(value)})
    return items


def _is_trivial(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value == "":
        return True
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return True
    if isinstance(value, Mapping) and len(value) == 0:
        return True
    return False


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value if len(value) <= 160 else value[:157] + "…"
    if isinstance(value, Mapping):
        return _format_mapping(value)
    if isinstance(value, (list, tuple)):
        return _format_sequence(list(value))
    return type(value).__name__


def _format_sequence(items: list[Any]) -> str:
    if not items:
        return "[]"
    parts = [_format_value(item) for item in items]
    flat = ", ".join(parts)
    if len(flat) <= 160 and len(items) <= 12:
        return flat
    return f"[{', '.join(parts[:6])}, …] ({len(items)})"


def _format_mapping(m: Mapping[str, Any]) -> str:
    kind = m.get("kind") if isinstance(m.get("kind"), str) else None

    # Dataset-backed refs — read as the method that built them.
    if kind == "dataset_label_ref":
        return f'dataset.labels("{m.get("name", "?")}")'
    if kind == "dataset_case_ref":
        return f'dataset.cases("{m.get("name", "?")}")'
    if kind == "artifact_label_ref":
        return f'artifact.label("{m.get("name", "?")}")'

    # Step refs.
    if kind == "step_ref":
        step = m.get("step", "?")
        output = m.get("output")
        if output and output != "artifact":
            return f'step("{step}").{output}'
        return f'step("{step}")'
    if kind == "step_feature_ref":
        return f'step("{m.get("step", "?")}").feature("{m.get("feature_name", "?")}")'
    if kind == "step_label_ref":
        return f'step("{m.get("step", "?")}").label("{m.get("label_name", "?")}")'

    # Token selection / pooling.
    if kind == "last":
        return "last"
    if kind == "first":
        return "first"
    if kind == "mean":
        return "mean"
    if kind == "full_sequence":
        return "full_sequence"
    if kind == "section":
        return f'section("{m.get("value", "?")}")'
    if kind == "slice":
        val = m.get("value")
        if isinstance(val, Mapping):
            start = val.get("start", "")
            stop = val.get("stop")
            return f'slice({start}..{"" if stop is None else stop})'
        return "slice"

    # Function-backed builders (PromptMetadataBuilder, TransformBuilder, …).
    if "import_path" in m and isinstance(m.get("import_path"), str):
        return f'builder("{m["import_path"]}")'

    # Engines — render as kind + model_id when present.
    if kind in {"toy", "vllm"} and "model_id" in m:
        return f'{kind}({m["model_id"]})'

    # Generic small map — render inline `{a=…, b=…}`.
    entries = [(k, v) for k, v in m.items() if not _is_trivial(v)]
    if kind and not any(k != "kind" for k, _ in entries):
        return kind
    if len(entries) <= 3:
        parts = [f"{k}={_format_value(v)}" if k != "kind" else str(v) for k, v in entries]
        return "{" + ", ".join(parts) + "}"
    preview_keys = [k for k, _ in entries[:4]]
    suffix = ", …" if len(entries) > 4 else ""
    return f"{{{', '.join(preview_keys)}{suffix}}} ({len(entries)} fields)"


def _resolve_deps(by_name: dict[str, StepSummary], names: list[str] | tuple[str, ...]) -> list[ResolvedDep]:
    out: list[ResolvedDep] = []
    for name in names:
        step = by_name.get(name)
        if step is None:
            out.append(ResolvedDep(step_name=name))
            continue
        out.append(
            ResolvedDep(
                step_name=step.step_name,
                runner=step.runner,
                artifact_id=step.artifact_id,
                artifact_kind=step.artifact_kind,
                status=step.status,
            )
        )
    return out


def _summarize_manifest(manifest: ArtifactManifest) -> ArtifactSummary:
    return ArtifactSummary(
        artifact_id=manifest.artifact_id,
        artifact_kind=manifest.artifact_kind,
        schema_version=manifest.schema_version,
        created_at=manifest.created_at,
        operation_spec_hash=manifest.operation_spec_hash,
        operation_semantic_hash=manifest.operation_semantic_hash,
        storage_refs=dict(manifest.storage_refs),
        metadata=dict(manifest.metadata),
        input_artifact_refs=list(manifest.input_artifact_refs),
        runner=dict(manifest.runner),
        engine=dict(manifest.engine),
        example_coverage=dict(manifest.example_coverage),
    )


def _has_storage_ref(refs: Mapping[str, Any], key: str) -> bool:
    value = refs.get(key)
    return value is not None and value != {}
