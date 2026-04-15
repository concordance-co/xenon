"""Command-line entrypoint for loading and executing pipelines_v2 workflows."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from pipelines_v2.api import (
    Dataset,
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
    parser = _build_parser()
    ns = parser.parse_args(list(argv) if argv is not None else None)
    if ns.command == "workflow":
        if ns.workflow_command == "plan":
            return _workflow_plan(ns)
        if ns.workflow_command == "run":
            return _workflow_run(ns)
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


def _build_runners(
    ns: argparse.Namespace,
    runner_specs: dict[str, RunnerSpec] | None,
) -> dict[str, object]:
    if runner_specs is not None:
        return {name: spec.to_runner() for name, spec in runner_specs.items()}
    return _build_runners_from_args(ns)


def _build_runners_from_args(ns: argparse.Namespace) -> dict[str, object]:
    secrets = tuple(_parse_secret_binding(value) for value in ns.secret)
    capture_volumes = tuple(_parse_volume_mount(value) for value in ns.capture_volume)
    catalog = _build_catalog(ns.catalog_postgres_env)
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

    for name in ("plan", "run"):
        command = workflow_subparsers.add_parser(name, help=f"{name.title()} a workflow from a Python file.")
        command.add_argument("--file", required=True, help="Python file exporting build_dataset() and build_workflow(...).")
        command.add_argument("--dataset-fn", default="build_dataset", help="Dataset builder function name.")
        command.add_argument("--workflow-fn", default="build_workflow", help="Workflow builder function name.")
        command.add_argument("--capture-runner-name", default="capture_gpu", help="Runner name used by capture steps.")
        command.add_argument("--analysis-runner-name", default="analysis_cpu", help="Runner name used by analysis steps.")
        command.add_argument(
            "--report-runner-name",
            default="report_local",
            help="Optional local report runner name to register for workflows that use it.",
        )
        command.add_argument(
            "--artifact-volume-name",
            default="xenon-data",
            help="Modal volume name for shared artifacts.",
        )
        command.add_argument(
            "--artifact-root",
            default="/data/artifacts/pipelines_v2_cli_workflow",
            help="Root path inside the artifact volume.",
        )
        command.add_argument(
            "--local-cache-root",
            default=None,
            help="Optional local cache root for Modal volume downloads.",
        )
        command.add_argument("--capture-gpu", default="L4", help="GPU resource for the capture runner.")
        command.add_argument("--analysis-cpu", type=float, default=6, help="CPU resource for the analysis runner.")
        command.add_argument(
            "--analysis-memory-mb",
            type=int,
            default=24 * 1024,
            help="Memory for the analysis runner in MiB.",
        )
        command.add_argument(
            "--timeout-seconds",
            type=int,
            default=7200,
            help="Timeout applied to Modal runners.",
        )
        command.add_argument(
            "--secret",
            action="append",
            default=[],
            help="Modal secret binding as NAME:ENV_VAR[,ENV_VAR2]. Repeat as needed.",
        )
        command.add_argument(
            "--capture-volume",
            action="append",
            default=[],
            help="Extra capture volume mount as NAME:MOUNT_PATH. Repeat as needed.",
        )
        command.add_argument(
            "--catalog-postgres-env",
            default=None,
            help="Optional env var name for a Postgres-backed catalog.",
        )
        command.add_argument(
            "--report-artifact-root",
            default="tmp/pipelines_v2_cli_local_reports",
            help="Local artifact root for optional report_local steps.",
        )
        if name == "run":
            command.add_argument(
                "--resume-run-id",
                default=None,
                help="Resume a previously recorded workflow run id.",
            )
            command.add_argument(
                "--reuse-completed",
                action="store_true",
                help="Reuse latest completed step artifacts whose semantic lineage matches.",
            )

    return parser


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
