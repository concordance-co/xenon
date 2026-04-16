"""Command-line entrypoint for loading and executing pipelines_v2 workflows."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from pipelines_v2.core.config import WorkspaceConfig, load_workspace_config
from pipelines_v2.core.env import load_dotenv_if_present
from pipelines_v2.core.paths import pipelines_v2_catalog_root
from pipelines_v2.api import (
    CompositeCatalog,
    Dataset,
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
    RunnerSpec,
    WorkflowOrchestrator,
    WorkflowResult,
    WorkflowSpec,
    WorkflowStep,
)


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
        if ns.workflow_command == "resume":
            return _workflow_resume(ns)
        if ns.workflow_command == "runs":
            return _workflow_runs(ns)
        if ns.workflow_command == "show":
            return _workflow_show(ns)
        if ns.workflow_command == "rerun-step":
            return _workflow_rerun_step(ns, include_downstream=False)
        if ns.workflow_command == "rerun-from-step":
            return _workflow_rerun_step(ns, include_downstream=True)
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
    _, workflow, runner_specs = load_python_workflow_file(
        path=ns.file,
        dataset_fn_name=ns.dataset_fn,
        workflow_fn_name=ns.workflow_fn,
    )
    orchestrator = WorkflowOrchestrator(runners=_build_runners(ns, runner_specs))
    result = orchestrator.run(
        workflow,
        resume_run_id=ns.resume_run_id,
        reuse_completed=bool(ns.reuse_completed),
    )
    print(json.dumps(_workflow_result_payload(workflow.name, result), indent=2, sort_keys=True))
    return 0


def _workflow_resume(ns: argparse.Namespace) -> int:
    _, workflow, runner_specs = load_python_workflow_file(
        path=ns.file,
        dataset_fn_name=ns.dataset_fn,
        workflow_fn_name=ns.workflow_fn,
    )
    local_catalog = _registry_catalog(ns, runner_specs=runner_specs)
    run_id = ns.run_id or _select_latest_run_id(
        local_catalog,
        workflow=workflow,
        status="failed",
    )
    if run_id is None:
        raise RuntimeError("Could not resolve a workflow run id to resume")
    orchestrator = WorkflowOrchestrator(runners=_build_runners(ns, runner_specs))
    result = orchestrator.run(workflow, resume_run_id=run_id)
    print(json.dumps(_workflow_result_payload(workflow.name, result), indent=2, sort_keys=True))
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
    payload = {
        "run": run.to_dict(),
        "steps": [record.to_dict() for record in local_catalog.list_workflow_steps(ns.run_id)],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _workflow_rerun_step(ns: argparse.Namespace, *, include_downstream: bool) -> int:
    _, workflow, runner_specs = load_python_workflow_file(
        path=ns.file,
        dataset_fn_name=ns.dataset_fn,
        workflow_fn_name=ns.workflow_fn,
    )
    local_catalog = _registry_catalog(ns, runner_specs=runner_specs)
    source_run_id = ns.run_id or _select_latest_run_id(local_catalog, workflow=workflow, status="completed")
    if source_run_id is None:
        raise RuntimeError("Could not resolve a completed workflow run id for rerun")
    subworkflow = _workflow_slice_for_step(workflow, step_name=ns.step, include_downstream=include_downstream)
    orchestrator = WorkflowOrchestrator(runners=_build_runners(ns, runner_specs))
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


def _build_runners(
    ns: argparse.Namespace,
    runner_specs: dict[str, RunnerSpec] | None,
) -> dict[str, object]:
    if runner_specs is not None:
        # Checked-in runner specs are the explicit workflow-level contract.
        # Workspace defaults should only fill CLI-built runners, not override
        # workflow-authored local/shared catalog choices.
        runners = {name: spec.to_runner() for name, spec in runner_specs.items()}
    else:
        runners = _build_runners_from_args(ns)
        _apply_workspace_catalog_defaults(runners, ns)
    return _attach_local_registry(runners, _local_registry_catalog(ns))


def _build_runners_from_args(ns: argparse.Namespace) -> dict[str, object]:
    secrets = tuple(_parse_secret_binding(value) for value in ns.secret)
    capture_volumes = tuple(_parse_volume_mount(value) for value in ns.capture_volume)
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
    catalog: FileCatalog,
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
        if hasattr(value, "manifest"):
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
    resume.add_argument("--run-id", default=None, help="Explicit workflow run id to resume.")
    resume.add_argument(
        "--latest-failed",
        action="store_true",
        help="Resume the latest failed run for the current workflow file.",
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

    rerun_step = workflow_subparsers.add_parser("rerun-step", help="Rerun one workflow step using artifacts from a prior run.")
    _add_workflow_file_args(rerun_step, required=True)
    _add_workflow_runner_args(rerun_step)
    rerun_step.add_argument("--run-id", default=None, help="Source workflow run id. Defaults to latest completed for the workflow file.")
    rerun_step.add_argument("--step", required=True, help="Step name to rerun.")

    rerun_from = workflow_subparsers.add_parser("rerun-from-step", help="Rerun one step and all downstream dependents.")
    _add_workflow_file_args(rerun_from, required=True)
    _add_workflow_runner_args(rerun_from)
    rerun_from.add_argument("--run-id", default=None, help="Source workflow run id. Defaults to latest completed for the workflow file.")
    rerun_from.add_argument("--step", required=True, help="Step name to rerun from.")

    return parser


def _add_workflow_file_args(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--file",
        required=required,
        help="Python file exporting build_dataset() and build_workflow(...).",
    )
    parser.add_argument("--dataset-fn", default="build_dataset", help="Dataset builder function name.")
    parser.add_argument("--workflow-fn", default="build_workflow", help="Workflow builder function name.")


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
