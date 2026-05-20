"""Normalize persisted run/step/workflow records into API response models.

The dashboard spec groups operation kinds into a small set of "families" for
coloring graph nodes: capture, derive, readout, representation, report. The
mapping here is built from the operation module layout under
`pipelines_v2.operations` so adding a new op kind only requires registering it
in the operation module tree, not editing the dashboard.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from pipelines_v2.dashboard.models import (
    DagEdge,
    DagNode,
    RunDetail,
    RunSummary,
    StepCounts,
    StepSummary,
)
from pipelines_v2.workflow.records import WorkflowRunRecord, WorkflowStepRecord
from pipelines_v2.workflow.specs import WorkflowSpec

# Operation kind -> family. Kept as a table so the mapping is visible in one
# place and easy to audit against `pipelines_v2.operations`.
_KIND_FAMILY: dict[str, str] = {
    "capture": "capture",
    "pair_delta": "derive",
    "label_map": "derive",
    "label_fields": "derive",
    "transform": "derive",
    "probe": "readout",
    "transfer_probe": "readout",
    "text_baseline": "readout",
    "residualized_probe": "readout",
    "direction": "representation",
    "basis": "representation",
    "geometry": "representation",
    "activation_patch": "representation",
    "coordinate_import": "mechinterp",
    "projection": "mechinterp",
    "projection_calibration": "mechinterp",
    "assistant_axis_precomputed_coordinate": "mechinterp",
    "assistant_axis_vector": "mechinterp",
    "assistant_axis_score": "mechinterp",
    "emotion_precomputed_vector_space": "mechinterp",
    "emotion_vector_space": "mechinterp",
    "emotion_score": "mechinterp",
    "emotion_direction": "mechinterp",
    "emotion_geometry": "mechinterp",
    "report": "report",
}


def _family_for_kind(kind: str | None) -> str | None:
    if kind is None:
        return None
    return _KIND_FAMILY.get(kind)


def _step_kind_from_payload(step_payload: Mapping[str, Any]) -> str | None:
    spec = step_payload.get("spec")
    if not isinstance(spec, Mapping):
        return None
    kind = spec.get("kind")
    return str(kind) if kind is not None else None


def _has_report_step(workflow_payload: Mapping[str, Any]) -> bool:
    steps = workflow_payload.get("steps", ()) if isinstance(workflow_payload, Mapping) else ()
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        if _step_kind_from_payload(step) == "report":
            return True
    return False


def tally_step_counts(steps: Iterable[WorkflowStepRecord]) -> StepCounts:
    counts = StepCounts(total=0)
    buckets = {"completed", "failed", "running", "reused", "pending"}
    for step in steps:
        counts.total += 1
        status = (step.status or "").lower()
        if status in buckets:
            setattr(counts, status, getattr(counts, status) + 1)
        else:
            counts.other += 1
    return counts


def build_run_summary(
    run: WorkflowRunRecord,
    steps: list[WorkflowStepRecord],
) -> RunSummary:
    return RunSummary(
        run_id=run.run_id,
        workflow_name=run.workflow_name,
        workflow_hash=run.workflow_hash,
        workflow_spec_hash=run.workflow_spec_hash,
        status=run.status,
        started_at=run.started_at,
        finished_at=run.finished_at,
        parent_run_id=run.parent_run_id,
        error=run.error,
        step_counts=tally_step_counts(steps),
        has_report=_has_report_step(run.workflow_payload),
    )


def build_run_summary_from_counts(
    run: WorkflowRunRecord,
    status_counts: Mapping[str, int],
) -> RunSummary:
    """Build a RunSummary from pre-aggregated `{status: count}` pairs.

    Used by the /api/runs fast path which pulls per-run histograms in a
    single SQL query instead of N × list_workflow_steps.
    """
    counts = StepCounts(total=0)
    buckets = {"completed", "failed", "running", "reused", "pending"}
    for status, n in status_counts.items():
        key = (status or "").lower()
        counts.total += int(n)
        if key in buckets:
            setattr(counts, key, getattr(counts, key) + int(n))
        else:
            counts.other += int(n)
    return RunSummary(
        run_id=run.run_id,
        workflow_name=run.workflow_name,
        workflow_hash=run.workflow_hash,
        workflow_spec_hash=run.workflow_spec_hash,
        status=run.status,
        started_at=run.started_at,
        finished_at=run.finished_at,
        parent_run_id=run.parent_run_id,
        error=run.error,
        step_counts=counts,
        has_report=_has_report_step(run.workflow_payload),
    )


def _index_workflow_steps(workflow_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    steps = workflow_payload.get("steps", ()) if isinstance(workflow_payload, Mapping) else ()
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        name = step.get("name")
        if name is None:
            continue
        out[str(name)] = dict(step)
    return out


def build_run_detail(
    run: WorkflowRunRecord,
    step_records: list[WorkflowStepRecord],
) -> RunDetail:
    workflow_payload = dict(run.workflow_payload) if run.workflow_payload else {}
    workflow_index = _index_workflow_steps(workflow_payload)

    # Resolved edges come from WorkflowSpec.from_dict — this rehydrates the
    # spec so StepRef/StepFeatureRef/StepLabelRef inference produces the same
    # edges the orchestrator executed against.
    resolved_deps: dict[str, tuple[str, ...]] = {}
    declared_deps: dict[str, tuple[str, ...]] = {}
    # Always seed declared deps from the raw payload — this is cheap, never
    # raises, and is the fallback when spec rehydration fails (e.g. unknown
    # operation kinds in older runs).
    for name, raw in workflow_index.items():
        declared_deps[name] = tuple(str(d) for d in raw.get("depends_on", ()) or ())
        resolved_deps[name] = declared_deps[name]
    try:
        spec = WorkflowSpec.from_dict(workflow_payload)
    except Exception:
        spec = None
    if spec is not None:
        for step in spec.steps:
            resolved_deps[step.name] = step.resolved_depends_on()
            declared_deps[step.name] = tuple(step.depends_on)

    # Steps come from the step records, ordered by (step_index, step_name).
    # If the workflow had no execution yet and records are empty, fall back to
    # the workflow payload order so the graph still renders.
    steps: list[StepSummary] = []
    if step_records:
        records_by_name: dict[str, WorkflowStepRecord] = {r.step_name: r for r in step_records}
        ordered_names = [r.step_name for r in step_records]
        # Include any workflow-declared steps with no record yet (pending).
        for name in workflow_index:
            if name not in records_by_name:
                ordered_names.append(name)
    else:
        records_by_name = {}
        ordered_names = list(workflow_index.keys())

    for idx, name in enumerate(ordered_names):
        record = records_by_name.get(name)
        wf_step = workflow_index.get(name, {})
        kind = _step_kind_from_payload(wf_step)
        runner = (
            record.runner
            if record is not None
            else str(wf_step.get("runner", ""))
        )
        steps.append(
            StepSummary(
                step_name=name,
                step_index=record.step_index if record is not None else idx,
                runner=runner,
                status=(record.status if record is not None else "pending"),
                spec_kind=kind,
                family=_family_for_kind(kind),
                artifact_id=record.artifact_id if record is not None else None,
                artifact_kind=record.artifact_kind if record is not None else None,
                started_at=record.started_at if record is not None else None,
                finished_at=record.finished_at if record is not None else None,
                runtime_app_id=record.runtime_app_id if record is not None else None,
                reused_from_run_id=record.reused_from_run_id if record is not None else None,
                reused_from_artifact_id=record.reused_from_artifact_id if record is not None else None,
                step_semantic_hash=record.step_semantic_hash if record is not None else None,
                step_spec_hash=record.step_spec_hash if record is not None else None,
                depends_on=list(declared_deps.get(name, ())),
                resolved_depends_on=list(resolved_deps.get(name, ())),
            )
        )

    nodes = [
        DagNode(
            id=s.step_name,
            step_name=s.step_name,
            runner=s.runner,
            spec_kind=s.spec_kind,
            family=s.family,
            status=s.status,
            artifact_id=s.artifact_id,
            artifact_kind=s.artifact_kind,
            reused=s.reused_from_run_id is not None,
            runtime_app_id=s.runtime_app_id,
        )
        for s in steps
    ]

    edges: list[DagEdge] = []
    node_ids = {n.id for n in nodes}
    for s in steps:
        declared_set = set(s.depends_on)
        for dep in s.resolved_depends_on:
            if dep not in node_ids:
                continue
            edges.append(
                DagEdge(
                    source=dep,
                    target=s.step_name,
                    kind="declared" if dep in declared_set else "resolved",
                )
            )

    run_summary = build_run_summary(run, step_records)
    return RunDetail(
        run=run_summary,
        workflow_payload=workflow_payload,
        nodes=nodes,
        edges=edges,
        steps=steps,
    )
