"""Workflow orchestration over named runners."""

from __future__ import annotations

import dataclasses
import inspect
import logging
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from pipelines_v2.core.types import SpecValidationError, stable_hash, utc_now_iso
from pipelines_v2.data.datasets import CaseSet, Dataset, LabelPredicate, LabelSet
from pipelines_v2.operations.derive import TransformSpec
from pipelines_v2.operations.execution.derive import run_transform
from pipelines_v2.operations.readouts import ProbeSpec, ResidualizedProbeSpec, TextBaselineSpec, TransferProbeSpec
from pipelines_v2.operations.representation import GeometrySpec
from pipelines_v2.operations.specs import CaptureSpec, TokenSelector
from pipelines_v2.runtime import Runner
from pipelines_v2.storage.artifacts import InlineOperationArtifact, artifact_from_manifest
from pipelines_v2.workflow.progress import WorkflowProgressEvent, WorkflowProgressSink
from pipelines_v2.workflow.records import WorkflowRunRecord, WorkflowStepContext, WorkflowStepRecord
from pipelines_v2.workflow.specs import (
    StepFeatureRef,
    StepLabelRef,
    StepRef,
    WorkflowPlan,
    WorkflowSpec,
    WorkflowStepPlan,
)

_LOG = logging.getLogger("pipelines_v2.workflow")


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Collected outputs from a completed workflow run."""

    run_id: str | None = None
    workflow_hash: str | None = None
    step_results: dict[str, Any] = field(default_factory=dict)

    def step(self, name: str) -> Any:
        """Return the result produced by one named workflow step."""
        return self.step_results[name]


@dataclass(slots=True)
class WorkflowOrchestrator:
    """Execute a workflow over named runners with dependency-aware fanout."""

    runners: Mapping[str, Runner]
    max_parallelism: int | None = None
    progress_sink: WorkflowProgressSink | None = None
    progress_heartbeat_seconds: float = 30.0

    def plan(self, workflow: WorkflowSpec) -> WorkflowPlan:
        """Preflight each workflow step against its assigned runner."""
        workflow_errors = _merge_workflow_errors(
            _workflow_section_metadata_errors(workflow),
            _workflow_row_alignment_errors(workflow),
        )
        step_plans: list[WorkflowStepPlan] = []
        for step in workflow.ordered_steps():
            try:
                runner = self.runners[step.runner]
            except KeyError as exc:
                raise SpecValidationError(
                    f"Workflow step {step.name!r} references unknown runner {step.runner!r}"
                ) from exc
            execution = runner.plan(step.spec)
            extra_errors = tuple(workflow_errors.get(step.name, ()))
            if extra_errors:
                execution = dataclasses.replace(
                    execution,
                    errors=tuple(execution.errors) + extra_errors,
                )
            step_plans.append(
                WorkflowStepPlan(
                    name=step.name,
                    runner=step.runner,
                    depends_on=step.resolved_depends_on(),
                    execution=execution,
                )
            )
        return WorkflowPlan(name=workflow.name, steps=tuple(step_plans))

    def run(
        self,
        workflow: WorkflowSpec,
        *,
        resume_run_id: str | None = None,
        reuse_completed: bool = False,
        reuse_from_run_id: str | None = None,
        force_rerun_steps: set[str] | frozenset[str] = frozenset(),
        parent_run_id: str | None = None,
    ) -> WorkflowResult:
        """Execute a workflow, resolving step refs as dependencies complete."""

        plan = self.plan(workflow)
        for step in plan.steps:
            step.execution.validate()

        ordered_steps = workflow.ordered_steps()
        step_by_name = {step.name: step for step in ordered_steps}
        step_index_by_name = {step.name: index for index, step in enumerate(ordered_steps)}
        dependencies = {step.name: set(step.resolved_depends_on()) for step in ordered_steps}
        forced_steps = {str(name) for name in force_rerun_steps}
        unknown_forced = sorted(name for name in forced_steps if name not in step_by_name)
        if unknown_forced:
            raise SpecValidationError(f"Unknown forced rerun steps: {unknown_forced}")
        workflow_hash = workflow.semantic_hash()
        workflow_spec_hash = workflow.spec_hash()
        run_id = resume_run_id or f"wr_{workflow_hash[:12]}_{uuid.uuid4().hex[:8]}"
        _LOG.info(
            "workflow started name=%s run_id=%s step_count=%d",
            workflow.name,
            run_id,
            len(ordered_steps),
        )
        self._emit_progress(
            WorkflowProgressEvent(
                run_id=run_id,
                workflow_name=workflow.name,
                status="running",
                stage="started",
                message=f"workflow started with {len(ordered_steps)} steps",
            )
        )
        catalog = self._workflow_catalog()
        existing_step_records: dict[str, WorkflowStepRecord] = {}
        reusable_step_records: dict[str, WorkflowStepRecord] = {}
        step_started_at: dict[str, str] = {}
        workflow_started_at = utc_now_iso()

        if resume_run_id is not None and (reuse_from_run_id is not None or forced_steps or parent_run_id is not None):
            raise SpecValidationError("resume_run_id cannot be combined with reuse_from_run_id, force_rerun_steps, or parent_run_id")

        if (resume_run_id is not None or reuse_completed) and catalog is None:
            raise SpecValidationError(
                "Workflow resume/reuse requires at least one shared non-null catalog across runners"
            )
        if (reuse_from_run_id is not None or forced_steps or parent_run_id is not None) and catalog is None:
            raise SpecValidationError(
                "Workflow rerun/reuse-from-run requires at least one shared non-null catalog across runners"
            )

        if catalog is not None:
            if resume_run_id is None:
                catalog.record_workflow_run(
                    WorkflowRunRecord(
                        run_id=run_id,
                        workflow_name=workflow.name,
                        workflow_hash=workflow_hash,
                        workflow_spec_hash=workflow_spec_hash,
                        workflow_payload=workflow.to_dict(),
                        status="running",
                        started_at=workflow_started_at,
                        parent_run_id=parent_run_id,
                    )
                )
            else:
                prior = catalog.load_workflow_run(resume_run_id)
                if prior is None:
                    raise SpecValidationError(f"Unknown workflow run id: {resume_run_id}")
                if prior.workflow_hash != workflow_hash:
                    raise SpecValidationError(
                        f"Workflow run {resume_run_id!r} has hash {prior.workflow_hash}, "
                        f"but current workflow hash is {workflow_hash}"
                    )
                workflow_started_at = prior.started_at
                catalog.record_workflow_run(
                    WorkflowRunRecord(
                        run_id=prior.run_id,
                        workflow_name=workflow.name,
                        workflow_hash=workflow_hash,
                        workflow_spec_hash=workflow_spec_hash,
                        workflow_payload=workflow.to_dict(),
                        status="running",
                        started_at=prior.started_at,
                        parent_run_id=prior.parent_run_id,
                        finished_at=None,
                        error=None,
                    )
                )
                existing_step_records = {
                    record.step_name: record for record in catalog.list_workflow_steps(run_id)
                }
            if reuse_from_run_id is not None:
                if catalog.load_workflow_run(reuse_from_run_id) is None:
                    raise SpecValidationError(f"Unknown workflow run id for reuse: {reuse_from_run_id}")
                reusable_step_records = {
                    record.step_name: record for record in catalog.list_workflow_steps(reuse_from_run_id)
                }

        results: dict[str, Any] = {}
        pending = set(step_by_name)
        running: dict[Future[Any], str] = {}
        max_workers = self.max_parallelism or max(1, len(ordered_steps))

        if catalog is not None and existing_step_records:
            for step_name, record in existing_step_records.items():
                if step_name in forced_steps:
                    continue
                if record.status not in {"completed", "reused", "running"}:
                    continue
                manifest = _load_manifest_for_workflow_step(catalog, record)
                if manifest is None:
                    continue
                runner = self.runners[step_by_name[step_name].runner]
                store = getattr(runner, "artifacts", None)
                if store is None:
                    continue
                if record.status == "running":
                    catalog.record_workflow_step(
                        WorkflowStepRecord(
                            run_id=run_id,
                            workflow_hash=workflow_hash,
                            workflow_step_key=record.workflow_step_key,
                            step_name=step_name,
                            step_index=step_index_by_name[step_name],
                            runner=step_by_name[step_name].runner,
                            status="completed",
                            step_semantic_hash=step_by_name[step_name].semantic_hash(),
                            step_spec_hash=step_by_name[step_name].spec_hash(),
                            input_artifact_refs=tuple(manifest.input_artifact_refs),
                            artifact_id=manifest.artifact_id,
                            artifact_kind=manifest.artifact_kind,
                            started_at=record.started_at or workflow_started_at,
                            finished_at=utc_now_iso(),
                            runtime_app_id=_manifest_runtime_app_id(manifest),
                        )
                    )
                results[step_name] = artifact_from_manifest(manifest, store=store)
                pending.discard(step_name)

        first_failure: Exception | None = None
        failed_steps: set[str] = set()

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while pending or running:
                progress_made = False

                if first_failure is None:
                    ready = [
                        step_by_name[name]
                        for name in sorted(pending)
                        if dependencies[name].issubset(results)
                    ]
                    for step in ready:
                        runner = self.runners[step.runner]
                        resolved_spec = _resolve_step_refs(step.spec, results)
                        step_context = WorkflowStepContext(
                            run_id=run_id,
                            workflow_name=workflow.name,
                            workflow_hash=workflow_hash,
                            workflow_spec_hash=workflow_spec_hash,
                            step_name=step.name,
                            step_index=step_index_by_name[step.name],
                            runner=step.runner,
                            step_semantic_hash=step.semantic_hash(),
                            step_spec_hash=step.spec_hash(),
                        )
                        input_artifact_refs = tuple(_input_artifact_ids_from_results(step, results))
                        if _should_inline_transform_step(step):
                            started_at = utc_now_iso()
                            _LOG.info("step starting name=%s runner=%s mode=inline_transform", step.name, step.runner)
                            self._emit_progress(
                                WorkflowProgressEvent(
                                    run_id=run_id,
                                    workflow_name=workflow.name,
                                    step_name=step.name,
                                    step_index=step_index_by_name[step.name],
                                    runner=step.runner,
                                    spec_kind=step.spec.kind,
                                    status="running",
                                    stage="inline_running",
                                    message="running inline transform",
                                )
                            )
                            try:
                                results[step.name] = _run_inline_transform_step(resolved_spec)
                            except Exception as exc:
                                _LOG.exception("step failed name=%s runner=%s mode=inline_transform", step.name, step.runner)
                                self._emit_progress(
                                    WorkflowProgressEvent(
                                        run_id=run_id,
                                        workflow_name=workflow.name,
                                        step_name=step.name,
                                        step_index=step_index_by_name[step.name],
                                        runner=step.runner,
                                        spec_kind=step.spec.kind,
                                        status="failed",
                                        stage="failed",
                                        message=str(exc),
                                    )
                                )
                                first_failure = exc
                                failed_steps.add(step.name)
                                pending.remove(step.name)
                                progress_made = True
                                if catalog is not None:
                                    catalog.record_workflow_step(
                                        WorkflowStepRecord(
                                            run_id=run_id,
                                            workflow_hash=workflow_hash,
                                            workflow_step_key=step_context.workflow_step_key,
                                            step_name=step.name,
                                            step_index=step_index_by_name[step.name],
                                            runner=step.runner,
                                            status="failed",
                                            step_semantic_hash=step_context.step_semantic_hash,
                                            step_spec_hash=step_context.step_spec_hash,
                                            input_artifact_refs=input_artifact_refs,
                                            started_at=started_at,
                                            finished_at=utc_now_iso(),
                                        )
                                    )
                                break
                            pending.remove(step.name)
                            progress_made = True
                            _LOG.info("step completed name=%s runner=%s kind=inline_transform", step.name, step.runner)
                            self._emit_progress(
                                WorkflowProgressEvent(
                                    run_id=run_id,
                                    workflow_name=workflow.name,
                                    step_name=step.name,
                                    step_index=step_index_by_name[step.name],
                                    runner=step.runner,
                                    spec_kind=step.spec.kind,
                                    status="completed",
                                    stage="completed",
                                    artifact_kind="inline_transform",
                                    message="inline transform completed",
                                )
                            )
                            if catalog is not None:
                                catalog.record_workflow_step(
                                    WorkflowStepRecord(
                                        run_id=run_id,
                                        workflow_hash=workflow_hash,
                                        workflow_step_key=step_context.workflow_step_key,
                                        step_name=step.name,
                                        step_index=step_index_by_name[step.name],
                                        runner=step.runner,
                                        status="completed",
                                        step_semantic_hash=step_context.step_semantic_hash,
                                        step_spec_hash=step_context.step_spec_hash,
                                        input_artifact_refs=input_artifact_refs,
                                        artifact_kind="inline_transform",
                                        started_at=started_at,
                                        finished_at=utc_now_iso(),
                                    )
                                )
                            continue
                        if catalog is not None:
                            resumed = existing_step_records.get(step.name)
                            if (
                                resumed is not None
                                and resumed.status in {"completed", "reused"}
                                and resumed.artifact_id
                                and step.name not in forced_steps
                            ):
                                pending.remove(step.name)
                                progress_made = True
                                continue
                            if step.name not in forced_steps:
                                reusable = reusable_step_records.get(step.name)
                                if (
                                    reusable is not None
                                    and reusable.status in {"completed", "reused"}
                                    and reusable.artifact_id
                                    and reusable.step_semantic_hash == step_context.step_semantic_hash
                                    and tuple(reusable.input_artifact_refs) == input_artifact_refs
                                ):
                                    manifest = _load_manifest_for_workflow_step(catalog, reusable)
                                    store = getattr(runner, "artifacts", None)
                                    if manifest is not None and store is not None:
                                        results[step.name] = artifact_from_manifest(manifest, store=store)
                                        now = utc_now_iso()
                                        _LOG.info(
                                            "step reused name=%s runner=%s source_run_id=%s artifact_id=%s",
                                            step.name,
                                            step.runner,
                                            reusable.run_id,
                                            reusable.artifact_id,
                                        )
                                        catalog.record_workflow_step(
                                            WorkflowStepRecord(
                                                run_id=run_id,
                                                workflow_hash=workflow_hash,
                                                workflow_step_key=step_context.workflow_step_key,
                                                step_name=step.name,
                                                step_index=step_index_by_name[step.name],
                                                runner=step.runner,
                                                status="reused",
                                                step_semantic_hash=step_context.step_semantic_hash,
                                                step_spec_hash=step_context.step_spec_hash,
                                                input_artifact_refs=input_artifact_refs,
                                                artifact_id=reusable.artifact_id,
                                                artifact_kind=reusable.artifact_kind,
                                                started_at=now,
                                                finished_at=now,
                                                runtime_app_id=reusable.runtime_app_id,
                                                reused_from_run_id=reusable.run_id,
                                                reused_from_artifact_id=reusable.artifact_id,
                                            )
                                        )
                                        pending.remove(step.name)
                                        progress_made = True
                                        self._emit_progress(
                                            WorkflowProgressEvent(
                                                run_id=run_id,
                                                workflow_name=workflow.name,
                                                step_name=step.name,
                                                step_index=step_index_by_name[step.name],
                                                runner=step.runner,
                                                spec_kind=step.spec.kind,
                                                status="reused",
                                                stage="reused",
                                                runtime_app_id=reusable.runtime_app_id,
                                                artifact_id=reusable.artifact_id,
                                                artifact_kind=reusable.artifact_kind,
                                                message=f"reused from run {reusable.run_id}",
                                            )
                                        )
                                        continue
                            if reuse_completed:
                                reusable = catalog.find_latest_reusable_step(
                                    step_name=step.name,
                                    step_semantic_hash=step_context.step_semantic_hash,
                                    input_artifact_refs=input_artifact_refs,
                                )
                                if reusable is not None and reusable.artifact_id:
                                    manifest = catalog.load_artifact(reusable.artifact_id)
                                    store = getattr(runner, "artifacts", None)
                                    if manifest is not None and store is not None:
                                        results[step.name] = artifact_from_manifest(manifest, store=store)
                                        _LOG.info(
                                            "step reused name=%s runner=%s source_run_id=%s artifact_id=%s",
                                            step.name,
                                            step.runner,
                                            reusable.run_id,
                                            reusable.artifact_id,
                                        )
                                        catalog.record_workflow_step(
                                            WorkflowStepRecord(
                                                run_id=run_id,
                                                workflow_hash=workflow_hash,
                                                workflow_step_key=step_context.workflow_step_key,
                                                step_name=step.name,
                                                step_index=step_index_by_name[step.name],
                                                runner=step.runner,
                                                status="reused",
                                                step_semantic_hash=step_context.step_semantic_hash,
                                                step_spec_hash=step_context.step_spec_hash,
                                                input_artifact_refs=input_artifact_refs,
                                                artifact_id=reusable.artifact_id,
                                                artifact_kind=reusable.artifact_kind,
                                                started_at=utc_now_iso(),
                                                finished_at=utc_now_iso(),
                                                runtime_app_id=reusable.runtime_app_id,
                                                reused_from_run_id=reusable.run_id,
                                                reused_from_artifact_id=reusable.artifact_id,
                                            )
                                        )
                                        pending.remove(step.name)
                                        progress_made = True
                                        self._emit_progress(
                                            WorkflowProgressEvent(
                                                run_id=run_id,
                                                workflow_name=workflow.name,
                                                step_name=step.name,
                                                step_index=step_index_by_name[step.name],
                                                runner=step.runner,
                                                spec_kind=step.spec.kind,
                                                status="reused",
                                                stage="reused",
                                                runtime_app_id=reusable.runtime_app_id,
                                                artifact_id=reusable.artifact_id,
                                                artifact_kind=reusable.artifact_kind,
                                                message=f"reused from run {reusable.run_id}",
                                            )
                                        )
                                        continue
                            started_at = utc_now_iso()
                            catalog.record_workflow_step(
                                WorkflowStepRecord(
                                    run_id=run_id,
                                    workflow_hash=workflow_hash,
                                    workflow_step_key=step_context.workflow_step_key,
                                    step_name=step.name,
                                    step_index=step_index_by_name[step.name],
                                    runner=step.runner,
                                    status="running",
                                    step_semantic_hash=step_context.step_semantic_hash,
                                    step_spec_hash=step_context.step_spec_hash,
                                    input_artifact_refs=input_artifact_refs,
                                    started_at=started_at,
                                    runtime_app_id=None,
                                )
                            )
                            step_started_at[step.name] = started_at
                        else:
                            step_started_at[step.name] = utc_now_iso()
                        _LOG.info("step starting name=%s runner=%s spec_kind=%s", step.name, step.runner, step.spec.kind)
                        self._emit_progress(
                            WorkflowProgressEvent(
                                run_id=run_id,
                                workflow_name=workflow.name,
                                step_name=step.name,
                                step_index=step_index_by_name[step.name],
                                runner=step.runner,
                                spec_kind=step.spec.kind,
                                status="running",
                                stage="dispatching",
                                message="submitting step to runner",
                            )
                        )
                        future = pool.submit(
                            _run_with_workflow_context,
                            runner,
                            resolved_spec,
                            step_context,
                            self._runner_progress_callback(step_context=step_context, spec_kind=step.spec.kind),
                        )
                        running[future] = step.name
                        pending.remove(step.name)
                        progress_made = True

                if progress_made and not running:
                    continue
                if not running:
                    if first_failure is not None:
                        break
                    unresolved = sorted(pending)
                    raise SpecValidationError(
                        f"Workflow could not make progress; unresolved steps remain: {unresolved}"
                    )

                done, _ = wait(
                    set(running),
                    timeout=self.progress_heartbeat_seconds if self.progress_sink is not None else None,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    for step_name in running.values():
                        step = step_by_name[step_name]
                        self._emit_progress(
                            WorkflowProgressEvent(
                                run_id=run_id,
                                workflow_name=workflow.name,
                                step_name=step_name,
                                step_index=step_index_by_name[step_name],
                                runner=step.runner,
                                spec_kind=step.spec.kind,
                                status="running",
                                stage="heartbeat",
                                message="step still running",
                            )
                        )
                    continue
                for future in done:
                    step_name = running.pop(future)
                    try:
                        result = future.result()
                        results[step_name] = result
                        if hasattr(result, "manifest"):
                            manifest = result.manifest()
                            _LOG.info(
                                "step completed name=%s runner=%s artifact_kind=%s artifact_id=%s runtime_app_id=%s",
                                step_name,
                                step_by_name[step_name].runner,
                                manifest.artifact_kind,
                                manifest.artifact_id,
                                _manifest_runtime_app_id(manifest),
                            )
                        else:
                            _LOG.info("step completed name=%s runner=%s", step_name, step_by_name[step_name].runner)
                        self._emit_progress(
                            WorkflowProgressEvent(
                                run_id=run_id,
                                workflow_name=workflow.name,
                                step_name=step_name,
                                step_index=step_index_by_name[step_name],
                                runner=step_by_name[step_name].runner,
                                spec_kind=step_by_name[step_name].spec.kind,
                                status="completed",
                                stage="completed",
                                runtime_app_id=(
                                    _manifest_runtime_app_id(manifest) if hasattr(result, "manifest") else None
                                ),
                                artifact_id=manifest.artifact_id if hasattr(result, "manifest") else None,
                                artifact_kind=manifest.artifact_kind if hasattr(result, "manifest") else None,
                                message="step completed",
                            )
                        )
                        if catalog is not None and hasattr(result, "manifest"):
                            manifest = result.manifest()
                            catalog.record_workflow_step(
                                WorkflowStepRecord(
                                    run_id=run_id,
                                    workflow_hash=workflow_hash,
                                    workflow_step_key=f"{workflow_hash}.{step_name}",
                                    step_name=step_name,
                                    step_index=step_index_by_name[step_name],
                                    runner=step_by_name[step_name].runner,
                                    status="completed",
                                    step_semantic_hash=step_by_name[step_name].semantic_hash(),
                                    step_spec_hash=step_by_name[step_name].spec_hash(),
                                    input_artifact_refs=tuple(manifest.input_artifact_refs),
                                    artifact_id=manifest.artifact_id,
                                    artifact_kind=manifest.artifact_kind,
                                    started_at=step_started_at.get(
                                        step_name,
                                        existing_step_records.get(step_name).started_at
                                        if step_name in existing_step_records
                                        else utc_now_iso(),
                                    ),
                                    finished_at=utc_now_iso(),
                                    runtime_app_id=_manifest_runtime_app_id(manifest),
                                )
                            )
                    except Exception as exc:
                        _LOG.exception("step failed name=%s runner=%s", step_name, step_by_name[step_name].runner)
                        self._emit_progress(
                            WorkflowProgressEvent(
                                run_id=run_id,
                                workflow_name=workflow.name,
                                step_name=step_name,
                                step_index=step_index_by_name[step_name],
                                runner=step_by_name[step_name].runner,
                                spec_kind=step_by_name[step_name].spec.kind,
                                status="failed",
                                stage="failed",
                                runtime_app_id=getattr(exc, "runtime_app_id", None),
                                message=str(exc),
                            )
                        )
                        failed_steps.add(step_name)
                        if catalog is not None:
                            catalog.record_workflow_step(
                                WorkflowStepRecord(
                                    run_id=run_id,
                                    workflow_hash=workflow_hash,
                                    workflow_step_key=f"{workflow_hash}.{step_name}",
                                    step_name=step_name,
                                    step_index=step_index_by_name[step_name],
                                    runner=step_by_name[step_name].runner,
                                    status="failed",
                                    step_semantic_hash=step_by_name[step_name].semantic_hash(),
                                    step_spec_hash=step_by_name[step_name].spec_hash(),
                                    input_artifact_refs=tuple(
                                        _input_artifact_ids_from_results(step_by_name[step_name], results)
                                    ),
                                    started_at=step_started_at.get(step_name, utc_now_iso()),
                                    finished_at=utc_now_iso(),
                                    runtime_app_id=getattr(exc, "runtime_app_id", None),
                                )
                            )
                        if first_failure is None:
                            first_failure = exc

        if first_failure is not None:
            blocked_steps = _blocked_pending_steps(pending, dependencies, failed_steps)
            if blocked_steps:
                _LOG.warning("workflow blocked run_id=%s blocked_steps=%s", run_id, sorted(blocked_steps))
            if catalog is not None:
                finished_at = utc_now_iso()
                for step_name in sorted(blocked_steps):
                    step = step_by_name[step_name]
                    self._emit_progress(
                        WorkflowProgressEvent(
                            run_id=run_id,
                            workflow_name=workflow.name,
                            step_name=step_name,
                            step_index=step_index_by_name[step_name],
                            runner=step.runner,
                            spec_kind=step.spec.kind,
                            status="blocked",
                            stage="blocked",
                            message="blocked by upstream failure",
                        )
                    )
                    catalog.record_workflow_step(
                        WorkflowStepRecord(
                            run_id=run_id,
                            workflow_hash=workflow_hash,
                            workflow_step_key=f"{workflow_hash}.{step_name}",
                            step_name=step_name,
                            step_index=step_index_by_name[step_name],
                            runner=step.runner,
                            status="blocked",
                            step_semantic_hash=step.semantic_hash(),
                            step_spec_hash=step.spec_hash(),
                            input_artifact_refs=tuple(_input_artifact_ids_from_results(step, results, strict=False)),
                            finished_at=finished_at,
                            runtime_app_id=None,
                        )
                    )
                catalog.record_workflow_run(
                    WorkflowRunRecord(
                        run_id=run_id,
                        workflow_name=workflow.name,
                        workflow_hash=workflow_hash,
                        workflow_spec_hash=workflow_spec_hash,
                        workflow_payload=workflow.to_dict(),
                        status="failed",
                        started_at=workflow_started_at,
                        parent_run_id=parent_run_id,
                        finished_at=finished_at,
                        error=str(first_failure),
                    )
                )
            _LOG.error("workflow failed name=%s run_id=%s error=%s", workflow.name, run_id, first_failure)
            self._emit_progress(
                WorkflowProgressEvent(
                    run_id=run_id,
                    workflow_name=workflow.name,
                    status="failed",
                    stage="failed",
                    message=str(first_failure),
                )
            )
            raise first_failure

        if catalog is not None:
            catalog.record_workflow_run(
                WorkflowRunRecord(
                    run_id=run_id,
                    workflow_name=workflow.name,
                    workflow_hash=workflow_hash,
                    workflow_spec_hash=workflow_spec_hash,
                    workflow_payload=workflow.to_dict(),
                    status="completed",
                    started_at=workflow_started_at,
                    parent_run_id=parent_run_id,
                    finished_at=utc_now_iso(),
                )
            )
        _LOG.info("workflow completed name=%s run_id=%s", workflow.name, run_id)
        self._emit_progress(
            WorkflowProgressEvent(
                run_id=run_id,
                workflow_name=workflow.name,
                status="completed",
                stage="completed",
                message="workflow completed",
            )
        )
        return WorkflowResult(run_id=run_id, workflow_hash=workflow_hash, step_results=results)

    def _workflow_catalog(self) -> Any | None:
        catalogs = [
            runner.catalog
            for runner in self.runners.values()
            if getattr(getattr(runner, "catalog", None), "kind", "none") != "none"
        ]
        if not catalogs:
            return None
        baseline = catalogs[0].identity()
        mismatched = [catalog.identity() for catalog in catalogs[1:] if catalog.identity() != baseline]
        if mismatched:
            raise SpecValidationError(
                "WorkflowOrchestrator requires one shared catalog identity across runners to persist workflow runs"
            )
        return catalogs[0]

    def _emit_progress(self, event: WorkflowProgressEvent) -> None:
        if self.progress_sink is None:
            return
        self.progress_sink.emit(event)

    def _runner_progress_callback(
        self,
        *,
        step_context: WorkflowStepContext,
        spec_kind: str,
    ) -> Callable[[Mapping[str, Any]], None] | None:
        if self.progress_sink is None:
            return None

        def _callback(payload: Mapping[str, Any]) -> None:
            self._emit_progress(
                WorkflowProgressEvent(
                    run_id=step_context.run_id,
                    workflow_name=step_context.workflow_name,
                    step_name=step_context.step_name,
                    step_index=step_context.step_index,
                    runner=step_context.runner,
                    spec_kind=spec_kind,
                    status=str(payload.get("status") or "running"),
                    stage=str(payload.get("stage") or "running"),
                    message=str(payload["message"]) if payload.get("message") is not None else None,
                    runtime_kind=str(payload["runtime_kind"]) if payload.get("runtime_kind") is not None else None,
                    runtime_app_id=(
                        str(payload["runtime_app_id"]) if payload.get("runtime_app_id") is not None else None
                    ),
                    artifact_id=str(payload["artifact_id"]) if payload.get("artifact_id") is not None else None,
                    artifact_kind=(
                        str(payload["artifact_kind"]) if payload.get("artifact_kind") is not None else None
                    ),
                    metrics=dict(payload.get("metrics", {})),
                )
            )

        return _callback


def _resolve_step_refs(value: Any, results: Mapping[str, Any]) -> Any:
    if isinstance(value, StepRef):
        try:
            return results[value.step]
        except KeyError as exc:
            raise SpecValidationError(f"Workflow step reference {value.step!r} is not available yet") from exc
    if isinstance(value, StepFeatureRef):
        try:
            artifact = results[value.step]
        except KeyError as exc:
            raise SpecValidationError(f"Workflow step reference {value.step!r} is not available yet") from exc
        feature = artifact.feature(value.feature_name)
        return feature.layer(value.layer_index) if value.layer_index is not None else feature
    if isinstance(value, StepLabelRef):
        try:
            artifact = results[value.step]
        except KeyError as exc:
            raise SpecValidationError(f"Workflow step reference {value.step!r} is not available yet") from exc
        return artifact.label(value.label_name)
    if dataclasses.is_dataclass(value):
        fields = {field.name: _resolve_step_refs(getattr(value, field.name), results) for field in dataclasses.fields(value)}
        return type(value)(**fields)
    if isinstance(value, tuple):
        return tuple(_resolve_step_refs(item, results) for item in value)
    if isinstance(value, list):
        return [_resolve_step_refs(item, results) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_step_refs(item, results) for key, item in value.items()}
    return value


def _workflow_section_metadata_errors(workflow: WorkflowSpec) -> dict[str, tuple[str, ...]]:
    step_by_name = {step.name: step for step in workflow.steps}
    errors: dict[str, tuple[str, ...]] = {}
    for step in workflow.steps:
        token_selector = getattr(step.spec, "tokens", None)
        feature_ref = getattr(step.spec, "feature", None)
        if not isinstance(token_selector, TokenSelector) or token_selector.kind != "section":
            continue
        if not isinstance(feature_ref, StepFeatureRef):
            continue
        source_step = step_by_name.get(feature_ref.step)
        if source_step is None or not isinstance(source_step.spec, CaptureSpec):
            continue
        if source_step.spec.provides_token_sections():
            continue
        errors[step.name] = (
            "Step uses TokenSelector.section(...), but upstream capture step "
            f"{source_step.name!r} does not define prompt_metadata_builder=... "
            "and does not carry explicit token_sections on every materialized example.",
        )
    return errors


def _workflow_row_alignment_errors(workflow: WorkflowSpec) -> dict[str, tuple[str, ...]]:
    step_by_name = {step.name: step for step in workflow.steps}
    errors: dict[str, tuple[str, ...]] = {}
    for step in workflow.steps:
        spec = step.spec
        if not isinstance(
            spec,
            (ProbeSpec, TransferProbeSpec, TextBaselineSpec, ResidualizedProbeSpec, GeometrySpec),
        ):
            continue
        if getattr(spec, "rows", None) is not None:
            continue
        primary_dataset = _primary_row_dataset(spec, step_by_name=step_by_name)
        if primary_dataset is None:
            continue
        primary_identity = _dataset_identity(primary_dataset)
        mismatched = []
        seen: set[str] = set()
        for dataset in _analysis_reference_datasets(spec, step_by_name=step_by_name):
            identity = _dataset_identity(dataset)
            if identity == primary_identity or identity in seen:
                continue
            seen.add(identity)
            mismatched.append(_dataset_display_name(dataset))
        if not mismatched:
            continue
        errors[step.name] = (
            "Step mixes feature/text rows from "
            f"{_dataset_display_name(primary_dataset)!r} with refs from {mismatched}. "
            "Add rows=... to declare the intended analysis row universe explicitly.",
        )
    return errors


def _merge_workflow_errors(*error_maps: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    merged: dict[str, tuple[str, ...]] = {}
    for error_map in error_maps:
        for step_name, messages in error_map.items():
            merged[step_name] = tuple(merged.get(step_name, ())) + tuple(messages)
    return merged


def _primary_row_dataset(spec: Any, *, step_by_name: Mapping[str, Any]) -> Dataset | None:
    if isinstance(spec, TextBaselineSpec):
        datasets = _datasets_in_value(spec.text, step_by_name=step_by_name)
        return datasets[0] if datasets else None
    feature = getattr(spec, "feature", None)
    datasets = _datasets_in_value(feature, step_by_name=step_by_name)
    return datasets[0] if datasets else None


def _analysis_reference_datasets(spec: Any, *, step_by_name: Mapping[str, Any]) -> list[Dataset]:
    if isinstance(spec, ProbeSpec):
        values = (spec.labels, spec.group_by, spec.split)
    elif isinstance(spec, TransferProbeSpec):
        values = (spec.labels, spec.group_by, spec.cohort_by, spec.split_by)
    elif isinstance(spec, TextBaselineSpec):
        values = (spec.labels, spec.group_by, spec.cohort_by, spec.split_by)
    elif isinstance(spec, ResidualizedProbeSpec):
        values = (spec.labels, spec.residualize_against, spec.group_by)
    elif isinstance(spec, GeometrySpec):
        values = (spec.label, spec.color_by, spec.subset)
    else:
        values = ()
    datasets: list[Dataset] = []
    for value in values:
        datasets.extend(_datasets_in_value(value, step_by_name=step_by_name))
    return datasets


def _datasets_in_value(value: Any, *, step_by_name: Mapping[str, Any]) -> list[Dataset]:
    if isinstance(value, Dataset):
        return [value]
    if isinstance(value, (LabelSet, CaseSet)):
        return [value.dataset]
    if isinstance(value, LabelPredicate):
        return _datasets_in_value(value.label_set, step_by_name=step_by_name)
    if isinstance(value, StepFeatureRef):
        source_step = step_by_name.get(value.step)
        if source_step is not None and isinstance(source_step.spec, CaptureSpec):
            return [source_step.spec.dataset]
        return []
    if isinstance(value, Mapping):
        datasets: list[Dataset] = []
        for item in value.values():
            datasets.extend(_datasets_in_value(item, step_by_name=step_by_name))
        return datasets
    if isinstance(value, tuple | list):
        datasets: list[Dataset] = []
        for item in value:
            datasets.extend(_datasets_in_value(item, step_by_name=step_by_name))
        return datasets
    return []


def _dataset_identity(dataset: Dataset) -> str:
    return stable_hash(dataset.semantic_dict())


def _dataset_display_name(dataset: Dataset) -> str:
    return str(dataset.name or dataset.id or _dataset_identity(dataset)[:8])


def _input_artifact_ids_from_results(step: Any, results: Mapping[str, Any], *, strict: bool = True) -> list[str]:
    from pipelines_v2.storage.artifacts import ArtifactLabelRef, FeatureLayerRef, FeatureRef

    try:
        resolved = _resolve_step_refs(step.spec, results)
    except SpecValidationError:
        if strict:
            raise
        resolved = step.spec
    artifact_ids: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, StepRef):
            artifact = results.get(value.step)
            if artifact is not None:
                visit(artifact)
            return
        elif isinstance(value, StepFeatureRef):
            artifact = results.get(value.step)
            if artifact is not None:
                artifact_ids.append(artifact.id)
            return
        elif isinstance(value, StepLabelRef):
            artifact = results.get(value.step)
            if artifact is not None:
                artifact_ids.append(artifact.id)
            return
        if isinstance(value, FeatureLayerRef):
            artifact_ids.append(value.feature.artifact.id)
        elif isinstance(value, FeatureRef):
            artifact_ids.append(value.artifact.id)
        elif isinstance(value, ArtifactLabelRef):
            artifact_ids.append(value.artifact.id)
        elif hasattr(value, "manifest"):
            try:
                artifact_ids.append(value.id)
            except Exception:
                pass
        elif hasattr(value, "__dataclass_fields__"):
            for field_name in value.__dataclass_fields__:
                visit(getattr(value, field_name))
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, tuple | list):
            for item in value:
                visit(item)

    visit(resolved)
    return sorted(set(artifact_ids))


def _run_with_workflow_context(
    runner: Any,
    spec: Any,
    step_context: WorkflowStepContext,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> Any:
    try:
        signature = inspect.signature(runner.run)
    except (TypeError, ValueError):
        return runner.run(spec)
    kwargs: dict[str, Any] = {}
    if "workflow_context" in signature.parameters:
        kwargs["workflow_context"] = step_context
    if "progress_callback" in signature.parameters:
        kwargs["progress_callback"] = progress_callback
    if kwargs:
        return runner.run(spec, **kwargs)
    return runner.run(spec)


def _should_inline_transform_step(step: Any) -> bool:
    return isinstance(getattr(step, "spec", None), TransformSpec) and bool(getattr(step.spec, "inline", False))


def _run_inline_transform_step(spec: TransformSpec) -> InlineOperationArtifact:
    result = run_transform(spec)
    return InlineOperationArtifact(
        payload=dict(result.payload),
        labels={str(name): dict(payload) for name, payload in result.labels.items()},
        metadata=dict(result.metadata),
        artifact_kind="inline_transform",
    )


def _load_manifest_for_workflow_step(catalog: Any, record: WorkflowStepRecord) -> Any | None:
    if record.artifact_id:
        manifest = catalog.load_artifact(record.artifact_id)
        if manifest is not None:
            return manifest
    finder = getattr(catalog, "find_artifact_for_workflow_step", None)
    if callable(finder):
        return finder(run_id=record.run_id, workflow_step_key=record.workflow_step_key)
    return None


def _manifest_runtime_app_id(manifest: Any) -> str | None:
    runner = getattr(manifest, "runner", {})
    if isinstance(runner, Mapping):
        app_id = runner.get("runtime_app_id")
        return str(app_id) if app_id is not None else None
    return None


def _blocked_pending_steps(
    pending: set[str],
    dependencies: Mapping[str, set[str]],
    failed_steps: set[str],
) -> set[str]:
    blocked: set[str] = set()
    changed = True
    while changed:
        changed = False
        for step_name in pending:
            if step_name in blocked:
                continue
            deps = dependencies[step_name]
            if deps & (failed_steps | blocked):
                blocked.add(step_name)
                changed = True
    return blocked
