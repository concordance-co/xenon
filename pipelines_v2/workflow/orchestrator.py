"""Workflow orchestration over named runners."""

from __future__ import annotations

import dataclasses
import inspect
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Mapping

from pipelines_v2.core.types import SpecValidationError, utc_now_iso
from pipelines_v2.operations.specs import CaptureSpec, TokenSelector
from pipelines_v2.runtime import Runner
from pipelines_v2.storage.artifacts import artifact_from_manifest
from pipelines_v2.workflow.specs import StepFeatureRef, StepLabelRef, StepRef, WorkflowPlan, WorkflowSpec, WorkflowStepPlan
from pipelines_v2.workflow.records import WorkflowRunRecord, WorkflowStepContext, WorkflowStepRecord


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

    def plan(self, workflow: WorkflowSpec) -> WorkflowPlan:
        """Preflight each workflow step against its assigned runner."""
        workflow_errors = _workflow_section_metadata_errors(workflow)
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
    ) -> WorkflowResult:
        """Execute a workflow, resolving step refs as dependencies complete."""
        plan = self.plan(workflow)
        for step in plan.steps:
            step.execution.validate()
        ordered_steps = workflow.ordered_steps()
        step_by_name = {step.name: step for step in ordered_steps}
        step_index_by_name = {step.name: index for index, step in enumerate(ordered_steps)}
        dependencies = {step.name: set(step.resolved_depends_on()) for step in ordered_steps}
        workflow_hash = workflow.semantic_hash()
        workflow_spec_hash = workflow.spec_hash()
        run_id = resume_run_id or f"wr_{workflow_hash[:12]}_{uuid.uuid4().hex[:8]}"
        catalog = self._workflow_catalog()
        existing_step_records: dict[str, WorkflowStepRecord] = {}
        step_started_at: dict[str, str] = {}

        if (resume_run_id is not None or reuse_completed) and catalog is None:
            raise SpecValidationError(
                "Workflow resume/reuse requires at least one shared non-null catalog across runners"
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
                        started_at=utc_now_iso(),
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
                catalog.record_workflow_run(
                    WorkflowRunRecord(
                        run_id=prior.run_id,
                        workflow_name=workflow.name,
                        workflow_hash=workflow_hash,
                        workflow_spec_hash=workflow_spec_hash,
                        workflow_payload=workflow.to_dict(),
                        status="running",
                        started_at=prior.started_at,
                        finished_at=None,
                        error=None,
                    )
                )
                existing_step_records = {
                    record.step_name: record
                    for record in catalog.list_workflow_steps(run_id)
                }

        results: dict[str, Any] = {}
        pending = set(step_by_name)
        running: dict[Future[Any], str] = {}
        max_workers = self.max_parallelism or max(1, len(ordered_steps))

        if catalog is not None and existing_step_records:
            for step_name, record in existing_step_records.items():
                if record.status not in {"completed", "reused"} or not record.artifact_id:
                    continue
                manifest = catalog.load_artifact(record.artifact_id)
                if manifest is None:
                    continue
                runner = self.runners[step_by_name[step_name].runner]
                store = getattr(runner, "artifacts", None)
                if store is None:
                    continue
                results[step_name] = artifact_from_manifest(manifest, store=store)
                pending.discard(step_name)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while pending or running:
                progress_made = False
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
                    if catalog is not None:
                        resumed = existing_step_records.get(step.name)
                        if resumed is not None and resumed.status in {"completed", "reused"} and resumed.artifact_id:
                            pending.remove(step.name)
                            progress_made = True
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
                                            reused_from_run_id=reusable.run_id,
                                            reused_from_artifact_id=reusable.artifact_id,
                                        )
                                    )
                                    pending.remove(step.name)
                                    progress_made = True
                                    continue
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
                                started_at=utc_now_iso(),
                            )
                        )
                        step_started_at[step.name] = utc_now_iso()
                    else:
                        step_started_at[step.name] = utc_now_iso()
                    future = pool.submit(_run_with_workflow_context, runner, resolved_spec, step_context)
                    running[future] = step.name
                    pending.remove(step.name)
                    progress_made = True

                if progress_made and not running:
                    continue
                if not running:
                    unresolved = sorted(pending)
                    raise SpecValidationError(
                        f"Workflow could not make progress; unresolved steps remain: {unresolved}"
                    )

                done, _ = wait(set(running), return_when=FIRST_COMPLETED)
                for future in done:
                    step_name = running.pop(future)
                    try:
                        result = future.result()
                        results[step_name] = result
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
                                )
                            )
                    except Exception as exc:
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
                                    input_artifact_refs=tuple(_input_artifact_ids_from_results(step_by_name[step_name], results)),
                                    started_at=step_started_at.get(step_name, utc_now_iso()),
                                    finished_at=utc_now_iso(),
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
                                    started_at=(
                                        catalog.load_workflow_run(run_id).started_at
                                        if catalog.load_workflow_run(run_id) is not None
                                        else utc_now_iso()
                                    ),
                                    finished_at=utc_now_iso(),
                                    error=str(exc),
                                )
                            )
                        for outstanding in running:
                            outstanding.cancel()
                        raise
        if catalog is not None:
            started = catalog.load_workflow_run(run_id)
            catalog.record_workflow_run(
                WorkflowRunRecord(
                    run_id=run_id,
                    workflow_name=workflow.name,
                    workflow_hash=workflow_hash,
                    workflow_spec_hash=workflow_spec_hash,
                    workflow_payload=workflow.to_dict(),
                    status="completed",
                    started_at=started.started_at if started is not None else utc_now_iso(),
                    finished_at=utc_now_iso(),
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


def _input_artifact_ids_from_results(step: Any, results: Mapping[str, Any]) -> list[str]:
    from pipelines_v2.storage.artifacts import ArtifactLabelRef, FeatureLayerRef, FeatureRef

    resolved = _resolve_step_refs(step.spec, results)
    artifact_ids: list[str] = []

    def visit(value: Any) -> None:
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


def _run_with_workflow_context(runner: Any, spec: Any, step_context: WorkflowStepContext) -> Any:
    try:
        signature = inspect.signature(runner.run)
    except (TypeError, ValueError):
        return runner.run(spec)
    if "workflow_context" in signature.parameters:
        return runner.run(spec, workflow_context=step_context)
    return runner.run(spec)
