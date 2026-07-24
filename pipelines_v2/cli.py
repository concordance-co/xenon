"""Command-line entrypoint for loading and executing pipelines_v2 workflows."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import inspect
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Mapping, Sequence

from pipelines_v2.core.config import WorkspaceConfig, load_workspace_config
from pipelines_v2.core.env import load_dotenv_if_present
from pipelines_v2.core.paths import pipelines_v2_catalog_root
from pipelines_v2.core.types import utc_now_iso
from pipelines_v2.storage.artifacts import InlineOperationArtifact, artifact_from_manifest
from pipelines_v2.storage.composite import iter_catalogs_depth_first
from pipelines_v2.storage.inference import artifact_store_from_manifest
from pipelines_v2.workflow.progress import (
    FileWorkflowProgressStore,
    WorkflowProgressEvent,
    WorkflowProgressSink,
)
from pipelines_v2.api import (
    CompositeCatalog,
    Dataset,
    DeploymentSpec,
    DeploymentTargetSpec,
    FileCatalog,
    LocalArtifactStore,
    LocalRunner,
    ModalResources,
    ModalRunner,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    NullCatalog,
    PostgresCatalog,
    PostgresSource,
    ReportSpec,
    RunnerSpec,
    StepRef,
    WorkflowOrchestrator,
    WorkflowResult,
    WorkflowSpec,
    WorkflowStep,
)
from pipelines_v2.runtime.deployments import controller_for_target
from pipelines_v2.workflow.records import WorkflowRunRecord

_LOG = logging.getLogger("pipelines_v2.cli")


def load_python_workflow_file(
    *,
    path: str | Path,
    dataset_fn_name: str = "build_dataset",
    workflow_fn_name: str = "build_workflow",
) -> tuple[Dataset, WorkflowSpec, dict[str, RunnerSpec] | None]:
    """Load dataset, workflow, and optional runner specs from a Python file."""
    module = _load_python_module(path)
    dataset_fn = _get_callable(module, dataset_fn_name)
    workflow_fn = _get_callable(module, workflow_fn_name)
    dataset = dataset_fn()
    if not isinstance(dataset, Dataset):
        raise TypeError(f"{dataset_fn_name}() must return Dataset, got {type(dataset).__name__}")
    workflow = _call_workflow_builder(workflow_fn, dataset)
    if not isinstance(workflow, WorkflowSpec):
        raise TypeError(f"{workflow_fn_name}(...) must return WorkflowSpec, got {type(workflow).__name__}")
    runner_specs = _load_runner_specs(module)
    return dataset, workflow, runner_specs


def _resolve_report_step(*args: Any, **kwargs: Any) -> Any:
    """Lazy wrapper so basic CLI commands do not import plotting dependencies."""
    from pipelines_v2.reporting.workflow import resolve_report_step

    return resolve_report_step(*args, **kwargs)


def load_python_deployment_file(
    *,
    path: str | Path,
    deployment_fn_name: str = "build_deployment",
    targets_fn_name: str = "build_deployment_targets",
) -> tuple[DeploymentSpec, dict[str, DeploymentTargetSpec]]:
    """Load a deployment spec and target specs from a Python file."""

    module = _load_python_module(path)
    deployment_fn = _get_callable(module, deployment_fn_name)
    deployment = deployment_fn()
    if not isinstance(deployment, DeploymentSpec):
        raise TypeError(f"{deployment_fn_name}() must return DeploymentSpec, got {type(deployment).__name__}")
    targets_fn = _get_callable(module, targets_fn_name)
    targets_payload = targets_fn()
    if not isinstance(targets_payload, dict):
        raise TypeError(f"{targets_fn_name}() must return dict[str, DeploymentTargetSpec]")
    targets: dict[str, DeploymentTargetSpec] = {}
    for name, target in targets_payload.items():
        if not isinstance(name, str):
            raise TypeError(f"{targets_fn_name}() keys must be strings")
        if not isinstance(target, DeploymentTargetSpec):
            raise TypeError(f"Deployment target {name!r} must be DeploymentTargetSpec, got {type(target).__name__}")
        targets[name] = target
    return deployment, targets


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv_if_present()
    parser = _build_parser()
    ns = parser.parse_args(list(argv) if argv is not None else None)
    setattr(ns, "_workspace_config", load_workspace_config(_config_start_path(ns)))
    if ns.command == "workflow":
        if ns.workflow_command == "plan":
            return _workflow_plan(ns)
        if ns.workflow_command == "run":
            return _workflow_run(ns)
        if ns.workflow_command == "report":
            return _workflow_report(ns)
        if ns.workflow_command == "resume":
            return _workflow_resume(ns)
        if ns.workflow_command == "runs":
            return _workflow_runs(ns)
        if ns.workflow_command == "show":
            return _workflow_show(ns)
        if ns.workflow_command == "cancel":
            return _workflow_cancel(ns)
        if ns.workflow_command == "rerun-step":
            return _workflow_rerun_step(ns, include_downstream=False)
        if ns.workflow_command == "rerun-from-step":
            return _workflow_rerun_step(ns, include_downstream=True)
    if ns.command == "deployment":
        if ns.deployment_command == "plan":
            return _deployment_plan(ns)
        if ns.deployment_command == "serve":
            return _deployment_serve(ns)
        if ns.deployment_command == "deploy":
            return _deployment_deploy(ns)
        if ns.deployment_command == "status":
            return _deployment_status(ns)
        if ns.deployment_command == "stop":
            return _deployment_stop(ns)
    parser.print_help()
    return 1


def _workflow_plan(ns: argparse.Namespace) -> int:
    _, workflow, runner_specs = load_python_workflow_file(
        path=ns.file,
        dataset_fn_name=ns.dataset_fn,
        workflow_fn_name=ns.workflow_fn,
    )
    orchestrator = WorkflowOrchestrator(runners=_build_runners(ns, runner_specs))
    plan = orchestrator.plan(workflow)
    payload = {
        "workflow": workflow.name,
        "steps": [
            {
                "name": step.name,
                "runner": step.runner,
                "description": step.description,
                "depends_on": list(step.depends_on),
                "spec_kind": step.execution.spec_kind,
                "artifact_kinds": list(step.execution.artifact_kinds),
                "errors": list(step.execution.errors),
                "warnings": list(step.execution.warnings),
                "missing_capabilities": sorted(cap.value for cap in step.execution.missing_capabilities),
            }
            for step in plan.steps
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _workflow_run(ns: argparse.Namespace) -> int:
    _configure_workflow_logging(getattr(ns, "logging", None))
    _, workflow, runner_specs = load_python_workflow_file(
        path=ns.file,
        dataset_fn_name=ns.dataset_fn,
        workflow_fn_name=ns.workflow_fn,
    )
    runners = _build_runners(ns, runner_specs)
    orchestrator = WorkflowOrchestrator(
        runners=runners,
        workflow_catalog=_registry_catalog(ns, runner_specs=runner_specs),
        progress_sink=_build_workflow_progress_sink(ns, runners=runners),
    )
    result = orchestrator.run(
        workflow,
        new_run_id=ns.run_id,
        resume_run_id=ns.resume_run_id,
        reuse_completed=bool(ns.reuse_completed),
    )
    print(json.dumps(_workflow_result_payload(workflow.name, result), indent=2, sort_keys=True))
    return 0


def _workflow_resume(ns: argparse.Namespace) -> int:
    _configure_workflow_logging(getattr(ns, "logging", None))
    _, workflow, runner_specs = load_python_workflow_file(
        path=ns.file,
        dataset_fn_name=ns.dataset_fn,
        workflow_fn_name=ns.workflow_fn,
    )
    workflow_catalog, run_id = _resolve_workflow_metadata_catalog(
        ns,
        runner_specs=runner_specs,
        run_id=ns.run_id,
        workflow=workflow,
        status="failed",
    )
    if run_id is None:
        raise RuntimeError("Could not resolve a workflow run id to resume")
    runners = _build_runners(ns, runner_specs)
    orchestrator = WorkflowOrchestrator(
        runners=runners,
        workflow_catalog=workflow_catalog,
        progress_sink=_build_workflow_progress_sink(ns, runners=runners),
    )
    result = orchestrator.run(workflow, resume_run_id=run_id)
    print(json.dumps(_workflow_result_payload(workflow.name, result), indent=2, sort_keys=True))
    return 0


def _workflow_report(ns: argparse.Namespace) -> int:
    catalog = _registry_catalog(ns)
    run = catalog.load_workflow_run(ns.run_id)
    if run is None:
        raise RuntimeError(f"Unknown workflow run id: {ns.run_id}")
    workflow = WorkflowSpec.from_dict(run.workflow_payload)
    report_step = resolve_report_step(workflow, step_name=ns.step, selector_label="--step")
    report_spec = _build_report_spec_from_run(
        run=run,
        report_step=report_step,
        workflow_catalog=catalog,
        local_cache_root=Path(ns.local_cache_root).expanduser().resolve() if ns.local_cache_root else None,
    )
    runner = LocalRunner(
        artifacts=_report_artifact_store_for_run(
            run=run,
            report_step=report_step,
            workflow_catalog=catalog,
            fallback_root=Path(ns.report_artifact_root),
            local_cache_root=Path(ns.local_cache_root).expanduser().resolve() if ns.local_cache_root else None,
        ),
        catalog=NullCatalog(),
    )
    artifact = runner.run(report_spec)
    payload = {
        "run_id": run.run_id,
        "step": report_step.name,
        "artifact_id": artifact.id,
        "artifact_kind": artifact.manifest().artifact_kind,
        "location": _artifact_location_hint(artifact.manifest()),
        "summary": artifact.summary(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _workflow_runs(ns: argparse.Namespace) -> int:
    runner_specs: dict[str, RunnerSpec] | None = None
    workflow_name = None
    if ns.file:
        _, workflow, runner_specs = load_python_workflow_file(
            path=ns.file,
            dataset_fn_name=ns.dataset_fn,
            workflow_fn_name=ns.workflow_fn,
        )
        workflow_name = workflow.name
    local_catalog = _registry_catalog(ns, runner_specs=runner_specs)
    records = local_catalog.list_workflow_runs(
        workflow_name=workflow_name,
        status=ns.status,
        limit=ns.limit,
    )
    payload = {
        "workflow": workflow_name,
        "runs": [record.to_dict() for record in records],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _workflow_show(ns: argparse.Namespace) -> int:
    local_catalog = _registry_catalog(ns)
    run = local_catalog.load_workflow_run(ns.run_id)
    if run is None:
        raise RuntimeError(f"Unknown workflow run id: {ns.run_id}")
    progress_store = FileWorkflowProgressStore(root=_workflow_progress_root(ns))
    payload = {
        "run": run.to_dict(),
        "steps": [record.to_dict() for record in local_catalog.list_workflow_steps(ns.run_id)],
        "progress": {
            "run": progress_store.load_run_snapshot(ns.run_id),
            "steps": progress_store.load_step_snapshots(ns.run_id),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _workflow_cancel(ns: argparse.Namespace) -> int:
    if not ns.yes:
        raise RuntimeError("workflow cancel requires --yes")
    catalog = _registry_catalog(ns)
    run = catalog.load_workflow_run(ns.run_id)
    if run is None:
        raise RuntimeError(f"Unknown workflow run id: {ns.run_id}")
    if run.status in {"completed", "failed", "cancelled", "canceled"}:
        print(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "status": run.status,
                    "canceled": run.status in {"cancelled", "canceled"},
                    "runtime_app_ids": [],
                    "stopped_runtime_app_ids": [],
                    "warnings": ["The workflow run is already terminal."],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    progress_root = _workflow_progress_root(ns)
    progress_store = FileWorkflowProgressStore(root=progress_root)
    progress_steps = progress_store.load_step_snapshots(ns.run_id)
    runtime_apps: dict[str, str] = {}
    for step in catalog.list_workflow_steps(ns.run_id):
        if step.runtime_app_id:
            runtime_apps[step.runtime_app_id] = step.status
    for snapshot in progress_steps.values():
        app_id = str(snapshot.get("runtime_app_id") or "").strip()
        if app_id:
            runtime_apps[app_id] = str(snapshot.get("status") or "")

    terminal_step_statuses = {
        "blocked",
        "cancelled",
        "canceled",
        "complete",
        "completed",
        "error",
        "failed",
        "skipped",
        "succeeded",
    }
    active_app_ids = sorted(
        app_id
        for app_id, status in runtime_apps.items()
        if str(status).strip().lower() not in terminal_step_statuses
    )
    stopped_app_ids: list[str] = []
    errors: list[str] = []
    for app_id in active_app_ids:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "modal", "app", "stop", app_id],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f"{app_id}: {error}")
            continue
        if completed.returncode == 0:
            stopped_app_ids.append(app_id)
            continue
        detail = completed.stderr.strip() or completed.stdout.strip()
        errors.append(f"{app_id}: {detail or f'exit {completed.returncode}'}")

    canceled = not errors
    if canceled:
        now = utc_now_iso()
        catalog.record_workflow_run(
            dataclasses.replace(
                run,
                status="cancelled",
                finished_at=now,
                error=None,
            )
        )
        progress_store.record_event(
            WorkflowProgressEvent(
                run_id=run.run_id,
                workflow_name=run.workflow_name,
                status="cancelled",
                stage="cancelled",
                created_at=now,
                message="Workflow cancellation confirmed.",
                metrics={
                    "runtime_app_ids": active_app_ids,
                    "stopped_runtime_app_ids": stopped_app_ids,
                },
            )
        )

    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "status": "cancelled" if canceled else "error",
                "canceled": canceled,
                "runtime_app_ids": active_app_ids,
                "stopped_runtime_app_ids": stopped_app_ids,
                "warnings": errors,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if canceled else 2


def _workflow_rerun_step(ns: argparse.Namespace, *, include_downstream: bool) -> int:
    _configure_workflow_logging(getattr(ns, "logging", None))
    _, workflow, runner_specs = load_python_workflow_file(
        path=ns.file,
        dataset_fn_name=ns.dataset_fn,
        workflow_fn_name=ns.workflow_fn,
    )
    workflow_catalog, source_run_id = _resolve_workflow_metadata_catalog(
        ns,
        runner_specs=runner_specs,
        run_id=ns.run_id,
        workflow=workflow,
        status="completed",
    )
    if source_run_id is None:
        raise RuntimeError("Could not resolve a completed workflow run id for rerun")
    _mirror_workflow_run_lineage(workflow_catalog, source_run_id)
    runners = _build_runners(ns, runner_specs)
    subworkflow = _workflow_slice_for_step(
        workflow,
        step_name=ns.step,
        include_downstream=include_downstream,
    )
    orchestrator = WorkflowOrchestrator(
        runners=runners,
        workflow_catalog=workflow_catalog,
        progress_sink=_build_workflow_progress_sink(ns, runners=runners),
    )
    result = orchestrator.run(
        subworkflow,
        reuse_from_run_id=source_run_id,
        force_rerun_steps=_forced_steps_for_rerun(
            workflow=subworkflow,
            target_step=ns.step,
            include_downstream=include_downstream,
        ),
        parent_run_id=source_run_id,
    )
    print(json.dumps(_workflow_result_payload(subworkflow.name, result), indent=2, sort_keys=True))
    return 0


def _deployment_plan(ns: argparse.Namespace) -> int:
    deployment, target_name, target = _load_selected_deployment(ns)
    controller = controller_for_target(target)
    plan = controller.plan(deployment, target, target_name=target_name)
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


def _deployment_serve(ns: argparse.Namespace) -> int:
    _configure_workflow_logging(getattr(ns, "logging", None))
    deployment, target_name, target = _load_selected_deployment(ns)
    controller = controller_for_target(target)
    handle = controller.serve(deployment, target, target_name=target_name)
    print(json.dumps(handle.to_dict(), indent=2, sort_keys=True))
    return 0


def _deployment_deploy(ns: argparse.Namespace) -> int:
    _configure_workflow_logging(getattr(ns, "logging", None))
    deployment, target_name, target = _load_selected_deployment(ns)
    controller = controller_for_target(target)
    handle = controller.deploy(deployment, target, target_name=target_name)
    print(json.dumps(handle.to_dict(), indent=2, sort_keys=True))
    return 0


def _deployment_status(ns: argparse.Namespace) -> int:
    deployment, target_name, target = _load_selected_deployment(ns)
    controller = controller_for_target(target)
    handle = controller.status(deployment, target, target_name=target_name)
    print(json.dumps(handle.to_dict(), indent=2, sort_keys=True))
    return 0


def _deployment_stop(ns: argparse.Namespace) -> int:
    if not bool(ns.yes):
        raise RuntimeError("deployment stop requires --yes")
    deployment, target_name, target = _load_selected_deployment(ns)
    controller = controller_for_target(target)
    handle = controller.stop(deployment, target, target_name=target_name)
    print(json.dumps(handle.to_dict(), indent=2, sort_keys=True))
    return 0


def _load_selected_deployment(ns: argparse.Namespace) -> tuple[DeploymentSpec, str, DeploymentTargetSpec]:
    deployment, targets = load_python_deployment_file(
        path=ns.file,
        deployment_fn_name=ns.deployment_fn,
        targets_fn_name=ns.targets_fn,
    )
    target_name = str(ns.target)
    try:
        target = targets[target_name]
    except KeyError as exc:
        available = ", ".join(sorted(targets)) or "<none>"
        raise RuntimeError(f"Deployment file does not define target {target_name!r}; available: {available}") from exc
    return deployment, target_name, target


def _mirror_workflow_run_lineage(catalog: Any, run_id: str, *, _seen: set[str] | None = None) -> None:
    """Mirror a source run and its ancestors through composite catalogs before reruns.

    Rerun commands can start from older runs that exist only in the local registry.
    When the active runner catalog also writes to Postgres, the child run's
    parent_run_id FK requires that source run to be present there too.
    """

    if getattr(catalog, "kind", None) != "composite":
        return
    seen = _seen if _seen is not None else set()
    if run_id in seen:
        return
    seen.add(run_id)
    record = catalog.load_workflow_run(run_id)
    if record is None:
        return
    if record.parent_run_id:
        _mirror_workflow_run_lineage(catalog, record.parent_run_id, _seen=seen)
    catalog.record_workflow_run(record)


def _build_runners(
    ns: argparse.Namespace,
    runner_specs: dict[str, RunnerSpec] | None,
) -> dict[str, object]:
    if runner_specs is not None:
        # Checked-in runner specs are the explicit workflow-level contract.
        # Workspace defaults fill wholly-unset catalog specs, but if the workflow
        # authors any catalog explicitly we leave the remaining runners local.
        runners = {name: spec.to_runner() for name, spec in runner_specs.items()}
        all_catalogs_unset = all(
            getattr(getattr(runner, "catalog", NullCatalog()), "kind", "none") == "none"
            for runner in runners.values()
        )
        if all_catalogs_unset:
            _apply_workspace_catalog_defaults(runners, ns)
    else:
        runners = _build_runners_from_args(ns)
        _apply_workspace_catalog_defaults(runners, ns)
    _apply_workspace_modal_defaults(runners, ns)
    _attach_workspace_root_to_modal_runners(runners, ns)
    return _attach_local_registry(runners, _local_registry_catalog(ns))


def _build_runners_from_args(ns: argparse.Namespace) -> dict[str, object]:
    secrets = tuple(_parse_secret_binding(value) for value in ns.secret)
    capture_volumes = tuple(_parse_volume_mount(value) for value in ns.capture_volume)
    capture_env = dict(_parse_env_assignment(value) for value in ns.capture_env)
    analysis_env = dict(_parse_env_assignment(value) for value in ns.analysis_env)
    catalog = _build_catalog(_configured_catalog_postgres_env(ns))
    artifact_store = ModalVolumeStore(
        name=ns.artifact_volume_name,
        root=ns.artifact_root,
        local_cache_root=Path(ns.local_cache_root) if ns.local_cache_root else None,
    )
    runners: dict[str, object] = {
        ns.capture_runner_name: ModalRunner(
            resources=ModalResources(
                gpu=ns.capture_gpu,
                timeout_seconds=ns.timeout_seconds,
                env=capture_env,
                secrets=secrets,
                volumes=capture_volumes,
            ),
            artifacts=artifact_store,
            catalog=catalog,
        ),
        ns.analysis_runner_name: ModalRunner(
            resources=ModalResources(
                cpu=ns.analysis_cpu,
                memory_mb=ns.analysis_memory_mb,
                timeout_seconds=ns.timeout_seconds,
                env=analysis_env,
                secrets=secrets,
            ),
            artifacts=artifact_store,
            catalog=catalog,
        ),
    }
    if ns.report_runner_name:
        runners[ns.report_runner_name] = LocalRunner(
            artifacts=LocalArtifactStore(Path(ns.report_artifact_root)),
        )
    return runners


def _build_catalog(env_var: str | None) -> object:
    if not env_var:
        return NullCatalog()
    return PostgresCatalog(source=PostgresSource.from_env(env_var))


def _local_registry_catalog(ns: argparse.Namespace) -> FileCatalog:
    configured = _configured_local_catalog_root(ns)
    root = configured if configured is not None else pipelines_v2_catalog_root()
    return FileCatalog(root=root)


def _build_workflow_progress_sink(
    ns: argparse.Namespace,
    *,
    runners: Mapping[str, object] | None = None,
) -> WorkflowProgressSink:
    store = FileWorkflowProgressStore(root=_workflow_progress_root(ns, runners=runners))
    numeric_level = _workflow_logging_level(getattr(ns, "logging", None))
    return WorkflowProgressSink(store=store, log_level=numeric_level)


def _workflow_progress_root(ns: argparse.Namespace, *, runners: Mapping[str, object] | None = None) -> Path:
    if runners is not None:
        for runner in runners.values():
            catalog = getattr(runner, "catalog", None)
            if catalog is None:
                continue
            for candidate in iter_catalogs_depth_first(catalog):
                if getattr(candidate, "kind", None) == "file" and hasattr(candidate, "root"):
                    return Path(candidate.root)
    return _local_registry_catalog(ns).root


def _registry_catalog(
    ns: argparse.Namespace,
    *,
    runner_specs: dict[str, RunnerSpec] | None = None,
) -> Any:
    if runner_specs is not None:
        runners = _build_runners(ns, runner_specs)
        catalogs = [
            runner.catalog
            for runner in runners.values()
            if getattr(getattr(runner, "catalog", None), "kind", "none") != "none"
        ]
        if not catalogs:
            return _local_registry_catalog(ns)
        baseline = catalogs[0].identity()
        mismatched = [catalog.identity() for catalog in catalogs[1:] if catalog.identity() != baseline]
        if mismatched:
            raise RuntimeError("Workflow runners do not share one catalog identity")
        return catalogs[0]
    local = _local_registry_catalog(ns)
    env_var = _configured_catalog_postgres_env(ns)
    if not env_var:
        return local
    return CompositeCatalog((local, PostgresCatalog(source=PostgresSource.from_env(env_var))))


def _workspace_config_for_ns(ns: argparse.Namespace) -> WorkspaceConfig:
    existing = getattr(ns, "_workspace_config", None)
    if isinstance(existing, WorkspaceConfig):
        return existing
    config = load_workspace_config(_config_start_path(ns))
    setattr(ns, "_workspace_config", config)
    return config


def _config_start_path(ns: argparse.Namespace) -> Path | None:
    path = getattr(ns, "file", None)
    if path:
        return Path(path)
    return None


def _configured_catalog_postgres_env(ns: argparse.Namespace) -> str | None:
    explicit = getattr(ns, "catalog_postgres_env", None)
    if explicit:
        return str(explicit)
    config = _workspace_config_for_ns(ns)
    configured = config.workflow_catalog_postgres_env()
    if configured and os.environ.get(configured):
        return configured
    return None


def _configured_local_catalog_root(ns: argparse.Namespace) -> Path | None:
    explicit = getattr(ns, "local_catalog_root", None)
    if explicit:
        return Path(str(explicit)).expanduser().resolve()
    return _workspace_config_for_ns(ns).workflow_local_catalog_root()


def _apply_workspace_catalog_defaults(runners: dict[str, object], ns: argparse.Namespace) -> None:
    env_var = _configured_catalog_postgres_env(ns)
    if not env_var:
        return
    shared_catalog = PostgresCatalog(source=PostgresSource.from_env(env_var))
    for runner in runners.values():
        current = getattr(runner, "catalog", NullCatalog())
        if getattr(current, "kind", "none") == "none":
            runner.catalog = shared_catalog


def _apply_workspace_modal_defaults(runners: dict[str, object], ns: argparse.Namespace) -> None:
    modal_defaults = _workspace_config_for_ns(ns).modal
    vllm_progress_env = {
        key: value
        for key in (
            "XENON_VLLM_SHARD_PROGRESS",
            "XENON_VLLM_CUSTOM_WORKER",
            "XENON_VLLM_PROGRESS_TRANSPORT",
            "XENON_VLLM_PROGRESS_WRITES",
        )
        if (value := str(os.environ.get(key) or "").strip())
    }
    if (
        modal_defaults.model_volume is None
        and not modal_defaults.use_vllm_torch_compile_cache
        and not vllm_progress_env
    ):
        return
    for runner in runners.values():
        if not isinstance(runner, ModalRunner):
            continue
        if runner.resources.gpu is None:
            continue
        runner.resources = _modal_resources_with_workspace_defaults(
            resources=runner.resources,
            modal_defaults=modal_defaults,
            vllm_progress_env=vllm_progress_env,
        )


def _attach_workspace_root_to_modal_runners(runners: dict[str, object], ns: argparse.Namespace) -> None:
    workspace_root = _workspace_config_for_ns(ns).workspace_root
    for runner in runners.values():
        if isinstance(runner, ModalRunner):
            runner.workspace_root = workspace_root


def _modal_resources_with_workspace_defaults(
    *,
    resources: ModalResources,
    modal_defaults: Any,
    vllm_progress_env: Mapping[str, str] | None = None,
) -> ModalResources:
    env = dict(resources.env)
    volumes = list(resources.volumes)
    changed = False

    for key, value in (vllm_progress_env or {}).items():
        if env.get(key) != value:
            env[key] = value
            changed = True

    model_volume = str(modal_defaults.model_volume or "").strip()
    model_volume_path = str(modal_defaults.model_volume_path or "").strip() or "/models"
    if model_volume:
        volumes, volume_changed = _ensure_modal_volume_mount(
            volumes,
            ModalVolumeMount(
                name=model_volume,
                mount_path=model_volume_path,
            ),
            required_path=model_volume_path,
        )
        changed = changed or volume_changed

    default_cache_root = modal_defaults.resolved_vllm_cache_root()
    default_cache_volume = modal_defaults.resolved_vllm_cache_volume()
    explicit_cache_root = env.get("VLLM_CACHE_ROOT")
    effective_cache_root = explicit_cache_root or default_cache_root

    if modal_defaults.use_vllm_torch_compile_cache and explicit_cache_root is None and default_cache_root:
        env["VLLM_CACHE_ROOT"] = default_cache_root
        changed = True

    if (
        modal_defaults.use_vllm_torch_compile_cache
        and effective_cache_root
        and default_cache_volume
        and (explicit_cache_root is None or explicit_cache_root == default_cache_root)
    ):
        cache_mount_path = _default_cache_mount_path(
            model_volume=model_volume,
            model_volume_path=model_volume_path,
            cache_volume=default_cache_volume,
            cache_root=effective_cache_root,
        )
        volumes, volume_changed = _ensure_modal_volume_mount(
            volumes,
            ModalVolumeMount(
                name=default_cache_volume,
                mount_path=cache_mount_path,
                create_if_missing=True,
                commit_on_success=True,
            ),
            required_path=effective_cache_root,
        )
        changed = changed or volume_changed

    if not changed:
        return resources
    return ModalResources(
        gpu=resources.gpu,
        cpu=resources.cpu,
        memory_mb=resources.memory_mb,
        timeout_seconds=resources.timeout_seconds,
        max_containers=resources.max_containers,
        shard_count=resources.shard_count,
        env=env,
        secrets=resources.secrets,
        volumes=tuple(volumes),
    )


def _default_cache_mount_path(
    *,
    model_volume: str,
    model_volume_path: str,
    cache_volume: str,
    cache_root: str,
) -> str:
    if model_volume and cache_volume == model_volume and _posix_path_covers(model_volume_path, cache_root):
        return model_volume_path
    return cache_root


def _ensure_modal_volume_mount(
    volumes: list[ModalVolumeMount],
    mount: ModalVolumeMount,
    *,
    required_path: str,
) -> tuple[list[ModalVolumeMount], bool]:
    for index, existing in enumerate(volumes):
        if existing.name == mount.name and _posix_path_covers(existing.mount_path, required_path):
            merged = ModalVolumeMount(
                name=existing.name,
                mount_path=existing.mount_path,
                create_if_missing=existing.create_if_missing or mount.create_if_missing,
                commit_on_success=existing.commit_on_success or mount.commit_on_success,
            )
            if merged != existing:
                volumes[index] = merged
                return volumes, True
            return volumes, False

    for existing in volumes:
        if _posix_path_covers(existing.mount_path, required_path):
            return volumes, False
        if _normalize_posix_path(existing.mount_path) == _normalize_posix_path(mount.mount_path):
            return volumes, False

    volumes.append(mount)
    return volumes, True


def _posix_path_covers(parent: str, child: str) -> bool:
    normalized_parent = PurePosixPath(_normalize_posix_path(parent))
    normalized_child = PurePosixPath(_normalize_posix_path(child))
    return normalized_parent == normalized_child or normalized_parent in normalized_child.parents


def _normalize_posix_path(path: str) -> str:
    pure = PurePosixPath(str(path).strip() or "/")
    text = str(pure)
    return text if text.startswith("/") else f"/{text}"


def _attach_local_registry(runners: dict[str, object], local_catalog: FileCatalog) -> dict[str, object]:
    shared_secondary: Any | None = None
    shared_secondary_identity: dict[str, Any] | None = None
    for runner in runners.values():
        current = getattr(runner, "catalog", NullCatalog())
        if getattr(current, "kind", "none") == "none":
            continue
        if hasattr(current, "identity") and current.identity() == local_catalog.identity():
            continue
        if shared_secondary is None:
            shared_secondary = current
            shared_secondary_identity = current.identity() if hasattr(current, "identity") else None
            continue
        current_identity = current.identity() if hasattr(current, "identity") else None
        if current_identity != shared_secondary_identity:
            shared_secondary = None
            shared_secondary_identity = None
            break

    for runner in runners.values():
        current = getattr(runner, "catalog", NullCatalog())
        if shared_secondary is not None:
            current_identity = current.identity() if hasattr(current, "identity") else None
            if getattr(current, "kind", "none") == "none" or current_identity == local_catalog.identity():
                runner.catalog = CompositeCatalog((local_catalog, shared_secondary))
                continue
            if current_identity == shared_secondary_identity:
                runner.catalog = CompositeCatalog((local_catalog, current))
                continue
        if getattr(current, "kind", "none") == "none":
            runner.catalog = local_catalog
            continue
        if hasattr(current, "identity") and current.identity() == local_catalog.identity():
            runner.catalog = local_catalog
            continue
        runner.catalog = CompositeCatalog((local_catalog, current))
    return runners


def _select_latest_run_id(
    catalog: Any,
    *,
    workflow: WorkflowSpec,
    status: str | None = None,
) -> str | None:
    records = catalog.list_workflow_runs(
        workflow_name=workflow.name,
        workflow_hash=workflow.semantic_hash(),
        status=status,
        limit=1,
    )
    if not records:
        return None
    return records[0].run_id


def _resolve_workflow_metadata_catalog(
    ns: argparse.Namespace,
    *,
    runner_specs: dict[str, RunnerSpec] | None,
    run_id: str | None,
    workflow: WorkflowSpec,
    status: str | None,
) -> tuple[Any, str | None]:
    primary = _registry_catalog(ns, runner_specs=runner_specs)
    fallback = _registry_catalog(ns)
    primary_identity = primary.identity() if hasattr(primary, "identity") else None
    fallback_identity = fallback.identity() if hasattr(fallback, "identity") else None

    if run_id is not None:
        if primary.load_workflow_run(run_id) is not None:
            return primary, run_id
        if fallback_identity != primary_identity and fallback.load_workflow_run(run_id) is not None:
            return fallback, run_id
        return primary, run_id

    resolved = _select_latest_run_id(primary, workflow=workflow, status=status)
    if resolved is not None:
        return primary, resolved
    if fallback_identity != primary_identity:
        fallback_resolved = _select_latest_run_id(fallback, workflow=workflow, status=status)
        if fallback_resolved is not None:
            return fallback, fallback_resolved
    return primary, None


def _workflow_slice_for_step(
    workflow: WorkflowSpec,
    *,
    step_name: str,
    include_downstream: bool,
) -> WorkflowSpec:
    step_by_name = {step.name: step for step in workflow.steps}
    try:
        target = step_by_name[step_name]
    except KeyError as exc:
        raise RuntimeError(f"Workflow does not contain step {step_name!r}") from exc

    ancestors: set[str] = set()
    queue: deque[str] = deque([step_name])
    while queue:
        current = queue.popleft()
        deps = step_by_name[current].resolved_depends_on()
        for dependency in deps:
            if dependency in ancestors:
                continue
            ancestors.add(dependency)
            queue.append(dependency)

    included = set(ancestors)
    included.add(step_name)
    if include_downstream:
        dependents: dict[str, list[str]] = defaultdict(list)
        for step in workflow.steps:
            for dependency in step.resolved_depends_on():
                dependents[dependency].append(step.name)
        queue = deque([step_name])
        while queue:
            current = queue.popleft()
            for dependent in dependents.get(current, ()):
                if dependent in included:
                    continue
                included.add(dependent)
                queue.append(dependent)

    steps = tuple(step for step in workflow.ordered_steps() if step.name in included)
    return WorkflowSpec(name=workflow.name, schema_version=workflow.schema_version, steps=steps)


def _build_report_spec_from_run(
    *,
    run: WorkflowRunRecord,
    report_step: WorkflowStep,
    workflow_catalog: Any,
    local_cache_root: Path | None,
) -> ReportSpec:
    workflow = WorkflowSpec.from_dict(run.workflow_payload)
    step_by_name = {step.name: step for step in workflow.steps}
    source_step_records = {record.step_name: record for record in workflow_catalog.list_workflow_steps(run.run_id)}
    resolved_inputs = tuple(
        _resolve_report_input_from_source_run(
            value,
            source_step_records=source_step_records,
            step_by_name=step_by_name,
            workflow_catalog=workflow_catalog,
            local_cache_root=local_cache_root,
        )
        for value in report_step.spec.inputs
    )
    return dataclasses.replace(report_step.spec, inputs=resolved_inputs)


def _resolve_report_input_from_source_run(
    value: Any,
    *,
    source_step_records: Mapping[str, Any],
    step_by_name: Mapping[str, WorkflowStep],
    workflow_catalog: Any,
    local_cache_root: Path | None,
) -> Any:
    if not isinstance(value, StepRef):
        return value
    step = step_by_name.get(value.step)
    if step is None:
        raise RuntimeError(f"Report input step {value.step!r} is not present in the persisted workflow")
    record = source_step_records.get(value.step)
    if record is None or not record.artifact_id:
        raise RuntimeError(f"Source run does not contain a completed artifact for report input step {value.step!r}")
    manifest = workflow_catalog.load_artifact(record.artifact_id)
    if manifest is None:
        raise RuntimeError(f"Could not load artifact manifest {record.artifact_id!r} for report input step {value.step!r}")
    store = artifact_store_from_manifest(manifest, local_cache_root=local_cache_root, purpose="report regeneration")
    return artifact_from_manifest(manifest, store=store)


def _report_artifact_store_for_run(
    *,
    run: WorkflowRunRecord,
    report_step: WorkflowStep,
    workflow_catalog: Any,
    fallback_root: Path,
    local_cache_root: Path | None,
) -> Any:
    source_step_records = {record.step_name: record for record in workflow_catalog.list_workflow_steps(run.run_id)}
    report_record = source_step_records.get(report_step.name)
    if report_record is not None and report_record.artifact_id:
        manifest = workflow_catalog.load_artifact(report_record.artifact_id)
        if manifest is not None:
            return artifact_store_from_manifest(manifest, local_cache_root=local_cache_root, purpose="report regeneration")
    return LocalArtifactStore(fallback_root)


def _forced_steps_for_rerun(
    *,
    workflow: WorkflowSpec,
    target_step: str,
    include_downstream: bool,
) -> set[str]:
    if not include_downstream:
        return {target_step}
    step_by_name = {step.name: step for step in workflow.steps}
    dependents: dict[str, list[str]] = defaultdict(list)
    for step in workflow.steps:
        for dependency in step.resolved_depends_on():
            dependents[dependency].append(step.name)
    forced = {target_step}
    queue: deque[str] = deque([target_step])
    while queue:
        current = queue.popleft()
        for dependent in dependents.get(current, ()):
            if dependent in forced or dependent not in step_by_name:
                continue
            forced.add(dependent)
            queue.append(dependent)
    return forced


def _workflow_result_payload(name: str | None, result: WorkflowResult) -> dict[str, Any]:
    steps: dict[str, Any] = {}
    for step_name, value in result.step_results.items():
        if isinstance(value, InlineOperationArtifact):
            steps[step_name] = {
                "artifact_id": None,
                "artifact_kind": value.artifact_kind,
                "summary": value.summary(),
            }
        elif hasattr(value, "manifest"):
            manifest = value.manifest()
            step_payload: dict[str, Any] = {
                "artifact_id": manifest.artifact_id,
                "artifact_kind": manifest.artifact_kind,
                "created_at": manifest.created_at,
                "runtime_app_id": (
                    manifest.runner.get("runtime_app_id")
                    if isinstance(manifest.runner, dict)
                    else None
                ),
                "location": _artifact_location_hint(manifest),
            }
            if hasattr(value, "summary"):
                try:
                    step_payload["summary"] = value.summary()
                except Exception:
                    pass
            steps[step_name] = step_payload
        else:
            steps[step_name] = value
    return {
        "workflow": name,
        "run_id": result.run_id,
        "workflow_hash": result.workflow_hash,
        "steps": steps,
    }


def _artifact_location_hint(manifest: Any) -> str | None:
    metadata = getattr(manifest, "metadata", {})
    if isinstance(metadata, dict):
        published = metadata.get("published_report")
        if isinstance(published, dict):
            report_path = published.get("report_path")
            if report_path is not None:
                return str(report_path)
    storage_refs = getattr(manifest, "storage_refs", {})
    if not isinstance(storage_refs, dict):
        return None
    for key in ("report", "result", "summary", "manifest"):
        ref = storage_refs.get(key)
        if isinstance(ref, dict) and ref.get("path") is not None:
            return str(ref["path"])
    return None


def _load_python_module(path: str | Path) -> ModuleType:
    resolved = Path(path)
    spec = importlib.util.spec_from_file_location(resolved.stem, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load workflow file: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _get_callable(module: ModuleType, name: str) -> Any:
    try:
        value = getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"Workflow file {module.__file__!r} is missing required callable {name!r}") from exc
    if not callable(value):
        raise TypeError(f"Workflow file attribute {name!r} must be callable")
    return value


def _load_runner_specs(module: ModuleType) -> dict[str, RunnerSpec] | None:
    value = getattr(module, "build_runner_specs", None)
    if value is None:
        return None
    if not callable(value):
        raise TypeError("Workflow file attribute 'build_runner_specs' must be callable")
    payload = value()
    if not isinstance(payload, dict):
        raise TypeError("build_runner_specs() must return dict[str, RunnerSpec]")
    runner_specs: dict[str, RunnerSpec] = {}
    for name, spec in payload.items():
        if not isinstance(name, str):
            raise TypeError("build_runner_specs() keys must be strings")
        if not hasattr(spec, "to_runner") or not hasattr(spec, "to_dict"):
            raise TypeError(f"Runner spec {name!r} must implement to_runner() and to_dict()")
        runner_specs[name] = spec
    return runner_specs


def _call_workflow_builder(builder: Any, dataset: Dataset) -> WorkflowSpec:
    signature = inspect.signature(builder)
    if "dataset" in signature.parameters:
        return builder(dataset=dataset)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = [parameter for parameter in positional if parameter.default is inspect._empty]
    if len(required) == 0:
        return builder()
    if len(required) == 1:
        return builder(dataset)
    raise TypeError(
        f"Workflow builder {getattr(builder, '__name__', builder)!r} must accept zero or one required positional args"
    )


def _parse_secret_binding(value: str) -> ModalSecret:
    try:
        name, env_vars_payload = value.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Secret binding must be NAME:ENV_VAR[,ENV_VAR2], got {value!r}") from exc
    env_vars = tuple(item.strip() for item in env_vars_payload.split(",") if item.strip())
    if not env_vars:
        raise ValueError(f"Secret binding must include at least one env var: {value!r}")
    return ModalSecret(name=name.strip(), env_vars=env_vars)


def _parse_volume_mount(value: str) -> ModalVolumeMount:
    try:
        name, mount_path = value.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Volume mount must be NAME:MOUNT_PATH, got {value!r}") from exc
    return ModalVolumeMount(name=name.strip(), mount_path=mount_path.strip())


def _parse_env_assignment(value: str) -> tuple[str, str]:
    try:
        name, env_value = value.split("=", 1)
    except ValueError as exc:
        raise ValueError(f"Env assignment must be NAME=VALUE, got {value!r}") from exc
    env_name = name.strip()
    if not env_name:
        raise ValueError(f"Env assignment must include a variable name: {value!r}")
    return env_name, env_value


def _configure_workflow_logging(level: str | None) -> None:
    logger = logging.getLogger("pipelines_v2")
    handler = next(
        (candidate for candidate in logger.handlers if getattr(candidate, "_pipelines_v2_cli_handler", False)),
        None,
    )
    if level is None:
        if handler is not None:
            logger.removeHandler(handler)
        if not logger.handlers:
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
        return
    level_name = str(level).strip().upper()
    numeric_level = getattr(logging, level_name, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported log level: {level!r}")
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        setattr(handler, "_pipelines_v2_cli_handler", True)
        logger.addHandler(handler)
    handler.setLevel(numeric_level)
    logger.setLevel(numeric_level)
    logger.propagate = False
    _LOG.debug("workflow CLI logging configured level=%s", level_name)


def _workflow_logging_level(level: str | None) -> int | None:
    if level is None:
        return None
    level_name = str(level).strip().upper()
    numeric_level = getattr(logging, level_name, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported log level: {level!r}")
    return numeric_level


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pipelines_v2.cli")
    subparsers = parser.add_subparsers(dest="command")

    workflow_parser = subparsers.add_parser("workflow", help="Plan or run a workflow from a Python file.")
    workflow_subparsers = workflow_parser.add_subparsers(dest="workflow_command")

    plan = workflow_subparsers.add_parser("plan", help="Plan a workflow from a Python file.")
    _add_workflow_file_args(plan, required=True)
    _add_workflow_runner_args(plan)

    run = workflow_subparsers.add_parser("run", help="Run a workflow from a Python file.")
    _add_workflow_file_args(run, required=True)
    _add_workflow_runner_args(run)
    _add_workflow_logging_arg(run)
    run.add_argument(
        "--run-id",
        default=None,
        help="Use this id for a new workflow run.",
    )
    run.add_argument(
        "--resume-run-id",
        default=None,
        help="Resume a previously recorded workflow run id.",
    )
    run.add_argument(
        "--reuse-completed",
        action="store_true",
        help="Reuse latest completed step artifacts whose semantic lineage matches.",
    )

    resume = workflow_subparsers.add_parser("resume", help="Resume the latest failed or a specific workflow run.")
    _add_workflow_file_args(resume, required=True)
    _add_workflow_runner_args(resume)
    _add_workflow_logging_arg(resume)
    resume.add_argument("--run-id", default=None, help="Explicit workflow run id to resume.")
    resume.add_argument(
        "--latest-failed",
        action="store_true",
        help="Resume the latest failed run for the current workflow file.",
    )

    report = workflow_subparsers.add_parser(
        "report",
        help="Regenerate a local report directly from an existing workflow run.",
    )
    report.add_argument("run_id", help="Workflow run id whose report step should be regenerated.")
    report.add_argument(
        "--step",
        default=None,
        help="Optional report step name. Required only when the run has multiple report steps.",
    )
    report.add_argument(
        "--report-artifact-root",
        default="tmp/pipelines_v2_cli_local_reports",
        help="Fallback local artifact root when the source run has no prior local report artifact to infer from.",
    )
    report.add_argument(
        "--local-cache-root",
        default=None,
        help="Optional local cache root for downloading remote artifact refs (for example, Modal volume results).",
    )
    report.add_argument(
        "--catalog-postgres-env",
        default=None,
        help="Optional env var name for a Postgres-backed catalog. Falls back to xenon.toml when omitted.",
    )
    report.add_argument(
        "--local-catalog-root",
        default=None,
        help="Optional local workflow state root; falls back to xenon.toml or ~/.xenon/pipelines_v2/catalog.",
    )

    runs = workflow_subparsers.add_parser("runs", help="List locally tracked workflow runs.")
    _add_workflow_file_args(runs, required=False)
    runs.add_argument("--status", default=None, help="Optional run status filter.")
    runs.add_argument("--limit", type=int, default=20, help="Maximum runs to return.")
    runs.add_argument(
        "--local-catalog-root",
        default=None,
        help="Optional local workflow state root; defaults under ~/.xenon/pipelines_v2/catalog.",
    )

    show = workflow_subparsers.add_parser("show", help="Show one locally tracked workflow run.")
    show.add_argument("--run-id", required=True, help="Workflow run id to inspect.")
    show.add_argument(
        "--local-catalog-root",
        default=None,
        help="Optional local workflow state root; defaults under ~/.xenon/pipelines_v2/catalog.",
    )

    cancel = workflow_subparsers.add_parser(
        "cancel",
        help="Stop active Modal apps and mark a tracked workflow run cancelled.",
    )
    cancel.add_argument("--run-id", required=True, help="Workflow run id to cancel.")
    cancel.add_argument(
        "--local-catalog-root",
        default=None,
        help="Optional local workflow state root; defaults under ~/.xenon/pipelines_v2/catalog.",
    )
    cancel.add_argument(
        "--catalog-postgres-env",
        default=None,
        help="Optional env var name for a Postgres-backed catalog.",
    )
    cancel.add_argument(
        "--yes",
        action="store_true",
        help="Required confirmation for stopping remote workflow apps.",
    )

    rerun_step = workflow_subparsers.add_parser("rerun-step", help="Rerun one workflow step using artifacts from a prior run.")
    _add_workflow_file_args(rerun_step, required=True)
    _add_workflow_runner_args(rerun_step)
    _add_workflow_logging_arg(rerun_step)
    rerun_step.add_argument("--run-id", default=None, help="Source workflow run id. Defaults to latest completed for the workflow file.")
    rerun_step.add_argument("--step", required=True, help="Step name to rerun.")

    rerun_from = workflow_subparsers.add_parser("rerun-from-step", help="Rerun one step and all downstream dependents.")
    _add_workflow_file_args(rerun_from, required=True)
    _add_workflow_runner_args(rerun_from)
    _add_workflow_logging_arg(rerun_from)
    rerun_from.add_argument("--run-id", default=None, help="Source workflow run id. Defaults to latest completed for the workflow file.")
    rerun_from.add_argument("--step", required=True, help="Step name to rerun from.")

    deployment_parser = subparsers.add_parser("deployment", help="Plan, serve, or deploy a long-lived service.")
    deployment_subparsers = deployment_parser.add_subparsers(dest="deployment_command")

    deployment_plan = deployment_subparsers.add_parser("plan", help="Plan a deployment from a Python file.")
    _add_deployment_file_args(deployment_plan)

    deployment_serve = deployment_subparsers.add_parser("serve", help="Serve a deployment on its target backend.")
    _add_deployment_file_args(deployment_serve)
    _add_workflow_logging_arg(deployment_serve)

    deployment_deploy = deployment_subparsers.add_parser("deploy", help="Deploy a long-lived service.")
    _add_deployment_file_args(deployment_deploy)
    _add_workflow_logging_arg(deployment_deploy)

    deployment_status = deployment_subparsers.add_parser("status", help="Inspect a long-lived deployment.")
    _add_deployment_file_args(deployment_status)

    deployment_stop = deployment_subparsers.add_parser("stop", help="Stop a long-lived deployment.")
    _add_deployment_file_args(deployment_stop)
    deployment_stop.add_argument("--yes", action="store_true", help="Required confirmation for stopping a deployment.")

    return parser


def _add_workflow_file_args(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--file",
        required=required,
        help="Python file exporting build_dataset() and build_workflow(...).",
    )
    parser.add_argument("--dataset-fn", default="build_dataset", help="Dataset builder function name.")
    parser.add_argument("--workflow-fn", default="build_workflow", help="Workflow builder function name.")


def _add_deployment_file_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--file",
        required=True,
        help="Python file exporting build_deployment() and build_deployment_targets().",
    )
    parser.add_argument("--target", required=True, help="Deployment target name from build_deployment_targets().")
    parser.add_argument("--deployment-fn", default="build_deployment", help="Deployment builder function name.")
    parser.add_argument("--targets-fn", default="build_deployment_targets", help="Deployment targets builder function name.")


def _add_workflow_logging_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--logging",
        default=None,
        metavar="LEVEL",
        help="Emit workflow progress logs to stderr at LEVEL (for example: DEBUG, INFO, WARNING, ERROR).",
    )


def _add_workflow_runner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--capture-runner-name", default="capture_gpu", help="Runner name used by capture steps.")
    parser.add_argument("--analysis-runner-name", default="analysis_cpu", help="Runner name used by analysis steps.")
    parser.add_argument(
        "--report-runner-name",
        default="report_local",
        help="Optional local report runner name to register for workflows that use it.",
    )
    parser.add_argument(
        "--artifact-volume-name",
        default="xenon-data",
        help="Modal volume name for shared artifacts.",
    )
    parser.add_argument(
        "--artifact-root",
        default="/data/artifacts/pipelines_v2_cli_workflow",
        help="Root path inside the artifact volume.",
    )
    parser.add_argument(
        "--local-cache-root",
        default=None,
        help="Optional local cache root for Modal volume downloads.",
    )
    parser.add_argument("--capture-gpu", default="L4", help="GPU resource for the capture runner.")
    parser.add_argument("--analysis-cpu", type=float, default=6, help="CPU resource for the analysis runner.")
    parser.add_argument(
        "--analysis-memory-mb",
        type=int,
        default=24 * 1024,
        help="Memory for the analysis runner in MiB.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=7200,
        help="Timeout applied to Modal runners.",
    )
    parser.add_argument(
        "--secret",
        action="append",
        default=[],
        help="Modal secret binding as NAME:ENV_VAR[,ENV_VAR2]. Repeat as needed.",
    )
    parser.add_argument(
        "--capture-volume",
        action="append",
        default=[],
        help="Extra capture volume mount as NAME:MOUNT_PATH. Repeat as needed.",
    )
    parser.add_argument(
        "--capture-env",
        action="append",
        default=[],
        help="Extra env var for the capture runner as NAME=VALUE. Repeat as needed.",
    )
    parser.add_argument(
        "--analysis-env",
        action="append",
        default=[],
        help="Extra env var for the analysis runner as NAME=VALUE. Repeat as needed.",
    )
    parser.add_argument(
        "--catalog-postgres-env",
        default=None,
        help="Optional env var name for a Postgres-backed catalog. Falls back to xenon.toml when omitted.",
    )
    parser.add_argument(
        "--report-artifact-root",
        default="tmp/pipelines_v2_cli_local_reports",
        help="Local artifact root for optional report_local steps.",
    )
    parser.add_argument(
        "--local-catalog-root",
        default=None,
        help="Optional local workflow state root; falls back to xenon.toml or ~/.xenon/pipelines_v2/catalog.",
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
