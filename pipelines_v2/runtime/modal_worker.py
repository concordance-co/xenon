"""Generic Modal transport for remote execution."""

from __future__ import annotations

import re
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from pipelines_v2.core.paths import find_workspace_root, resolve_workspace_path
from pipelines_v2.core.types import RuntimeSecret
from pipelines_v2.engine import PythonRuntimeSpec
from pipelines_v2.operations import operation_spec_from_dict
from pipelines_v2.runtime.env import merged_runtime_env
from pipelines_v2.storage.modal import modal_volume_mount_path


_REMOTE_WORKSPACE_ROOT = "/root/pipelines_v2_workspace"
_LOG = logging.getLogger("pipelines_v2.modal")


@dataclass(frozen=True, slots=True)
class MountedVolume:
    name: str
    mount_path: str
    create_if_missing: bool = False
    commit_on_success: bool = False


def run_on_modal(
    *,
    runner_config: dict[str, Any],
    store_config: dict[str, Any],
    spec_payload: dict[str, Any],
    workflow_context: dict[str, Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Submit serialized work to Modal and return the remote result."""

    import modal

    runtime_spec = _resolved_runtime_spec(
        spec_payload=spec_payload,
    )
    if not isinstance(runtime_spec, PythonRuntimeSpec):
        raise NotImplementedError(
            f"ModalRunner requires a PythonRuntimeSpec, got {type(runtime_spec).__name__}"
        )
    resources = runner_config.get("resources", {})
    _validate_secret_bindings(runtime_spec=runtime_spec, resources=resources)
    mounted_volumes = _mounted_volumes(store_config=store_config, resources=resources)
    app = modal.App(
        _modal_app_name(
            spec_payload=spec_payload,
            workflow_context=workflow_context,
        )
    )
    image = modal.Image.debian_slim(python_version=runtime_spec.python_version)
    if runtime_spec.pip_packages:
        image = image.pip_install(*runtime_spec.pip_packages)
    runtime_env = merged_runtime_env(runtime_spec.env, resources.get("env"))
    source_mounts, pythonpath_entries = _resolved_local_python_sources(runtime_spec.local_python_sources)
    if pythonpath_entries:
        existing_pythonpath = runtime_env.get("PYTHONPATH", "")
        combined = [entry for entry in pythonpath_entries if entry]
        if existing_pythonpath:
            combined.append(existing_pythonpath)
        runtime_env["PYTHONPATH"] = ":".join(combined)
    if runtime_env:
        image = image.env(runtime_env)
    for local_path, remote_path in source_mounts:
        image = image.add_local_dir(str(local_path), remote_path=remote_path)
    secrets = [modal.Secret.from_name(str(secret["name"])) for secret in resources.get("secrets", [])]
    function_kwargs: dict[str, Any] = {
        "image": image,
        "volumes": {
            volume.mount_path: modal.Volume.from_name(
                volume.name,
                create_if_missing=volume.create_if_missing,
            )
            for volume in mounted_volumes
        },
        "secrets": secrets,
        "timeout": int(resources.get("timeout_seconds") or 7200),
        "serialized": True,
    }
    if runtime_env:
        function_kwargs["env"] = runtime_env
    if resources.get("gpu") is not None:
        function_kwargs["gpu"] = resources.get("gpu")
    if resources.get("cpu") is not None:
        function_kwargs["cpu"] = resources.get("cpu")
    if resources.get("memory_mb") is not None:
        function_kwargs["memory"] = int(resources["memory_mb"])
    if resources.get("max_containers") is not None:
        function_kwargs["max_containers"] = int(resources["max_containers"])

    @app.function(**function_kwargs)
    def _remote_execute(
        remote_runner_config: dict[str, Any],
        remote_store_config: dict[str, Any],
        remote_spec_payload: dict[str, Any],
        remote_workflow_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        from pipelines_v2.runtime.remote_executor import execute_remote

        result = execute_remote(
            runner_config=remote_runner_config,
            store_config=remote_store_config,
            spec_payload=remote_spec_payload,
            workflow_context=remote_workflow_context,
        )
        warnings = _commit_mounted_volumes(mounted_volumes)
        if warnings:
            result.setdefault("metadata", {})["volume_commit_warnings"] = warnings
        return result

    @app.function(**function_kwargs)
    def _remote_merge_shards(
        remote_runner_config: dict[str, Any],
        remote_store_config: dict[str, Any],
        remote_spec_payload: dict[str, Any],
        remote_shard_manifests: list[dict[str, Any]],
        remote_workflow_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        from pipelines_v2.runtime.remote_executor import merge_remote_shards

        result = merge_remote_shards(
            runner_config=remote_runner_config,
            store_config=remote_store_config,
            spec_payload=remote_spec_payload,
            shard_manifests=remote_shard_manifests,
            workflow_context=remote_workflow_context,
        )
        warnings = _commit_mounted_volumes(mounted_volumes)
        if warnings:
            result.setdefault("metadata", {})["volume_commit_warnings"] = warnings
        return result

    _LOG.info(
        "modal spin-up starting kind=%s store=%s gpu=%s cpu=%s memory_mb=%s source_mounts=%d",
        runner_config.get("kind"),
        store_config.get("name"),
        resources.get("gpu"),
        resources.get("cpu"),
        resources.get("memory_mb"),
        len(source_mounts),
    )
    if progress_callback is not None:
        progress_callback(
            {
                "status": "running",
                "stage": "modal_launching",
                "runtime_kind": "modal",
                "message": "Starting Modal app launch",
                "metrics": {
                    "source_mount_count": len(source_mounts),
                },
            }
        )
    with app.run() as running_app:
        runtime_app_id = getattr(running_app, "app_id", None)
        if progress_callback is not None:
            progress_callback(
                {
                    "status": "running",
                    "stage": "modal_app_started",
                    "runtime_kind": "modal",
                    "runtime_app_id": runtime_app_id,
                    "message": "Modal app started",
                }
            )
        _LOG.info(
            "modal run submitted kind=%s runtime_app_id=%s",
            runner_config.get("kind"),
            runtime_app_id,
        )
        try:
            shard_count = _modal_shard_count(resources=resources, spec_payload=spec_payload)
            if shard_count > 1:
                if progress_callback is not None:
                    progress_callback(
                        {
                            "status": "running",
                            "stage": "remote_shards_submitted",
                            "runtime_kind": "modal",
                            "runtime_app_id": runtime_app_id,
                            "message": f"Submitted {shard_count} remote execution shards",
                            "metrics": {"shard_count": shard_count},
                        }
                    )
                shard_contexts = [
                    _workflow_context_with_shard(workflow_context, index=index, count=shard_count)
                    for index in range(shard_count)
                ]
                with ThreadPoolExecutor(max_workers=shard_count) as executor:
                    futures = [
                        executor.submit(
                            _remote_execute.remote,
                            runner_config,
                            store_config,
                            spec_payload,
                            shard_context,
                        )
                        for shard_context in shard_contexts
                    ]
                    shard_results = [future.result() for future in futures]
                if progress_callback is not None:
                    progress_callback(
                        {
                            "status": "running",
                            "stage": "remote_shards_finished",
                            "runtime_kind": "modal",
                            "runtime_app_id": runtime_app_id,
                            "message": f"Finished {shard_count} remote execution shards",
                            "metrics": {"shard_count": shard_count},
                        }
                    )
                result = _remote_merge_shards.remote(
                    runner_config,
                    store_config,
                    spec_payload,
                    shard_results,
                    workflow_context,
                )
            else:
                result = _remote_execute.remote(runner_config, store_config, spec_payload, workflow_context)
        except Exception as exc:
            if runtime_app_id is not None:
                try:
                    setattr(exc, "runtime_app_id", runtime_app_id)
                except Exception:
                    pass
            raise
        if isinstance(result, dict) and runtime_app_id is not None:
            runner_payload = dict(result.get("runner", {}))
            runner_payload["runtime_app_id"] = runtime_app_id
            result["runner"] = runner_payload
        if progress_callback is not None:
            progress_callback(
                {
                    "status": "running",
                    "stage": "remote_execution_finished",
                    "runtime_kind": "modal",
                    "runtime_app_id": runtime_app_id,
                    "message": "Modal app finished remote execution",
                }
            )
        return result


def _modal_shard_count(*, resources: dict[str, Any], spec_payload: dict[str, Any]) -> int:
    count = int(resources.get("shard_count") or 1)
    if count < 1:
        raise ValueError("ModalResources shard_count must be >= 1")
    if count == 1:
        return 1
    kind = str(spec_payload.get("kind") or "")
    if kind not in {"capture", "generation_run"}:
        raise ValueError(f"ModalResources shard_count is not supported for {kind!r} specs")
    return count


def _workflow_context_with_shard(
    workflow_context: dict[str, Any] | None,
    *,
    index: int,
    count: int,
) -> dict[str, Any]:
    context = dict(workflow_context or {})
    context["execution_shard"] = {"index": int(index), "count": int(count)}
    return context


def _mounted_volumes(*, store_config: dict[str, Any], resources: dict[str, Any]) -> tuple[MountedVolume, ...]:
    requested_volumes = [
        MountedVolume(
            name=str(store_config["name"]),
            mount_path=modal_volume_mount_path(str(store_config["root"])),
            create_if_missing=True,
            commit_on_success=True,
        )
    ]
    for payload in resources.get("volumes", []):
        requested_volumes.append(
            MountedVolume(
                name=str(payload["name"]),
                mount_path=str(payload["mount_path"]),
                create_if_missing=bool(payload.get("create_if_missing", False)),
                commit_on_success=bool(payload.get("commit_on_success", False)),
            )
        )
    by_mount_path: dict[str, MountedVolume] = {}
    for volume in requested_volumes:
        existing = by_mount_path.get(volume.mount_path)
        if existing is None:
            by_mount_path[volume.mount_path] = volume
            continue
        if existing.name != volume.name:
            raise ValueError(
                "Duplicate Modal volume mount paths with different volumes: "
                f"{volume.mount_path!r} maps to both {existing.name!r} and {volume.name!r}"
            )
        by_mount_path[volume.mount_path] = MountedVolume(
            name=existing.name,
            mount_path=existing.mount_path,
            create_if_missing=existing.create_if_missing or volume.create_if_missing,
            commit_on_success=existing.commit_on_success or volume.commit_on_success,
        )
    return tuple(by_mount_path[mount_path] for mount_path in sorted(by_mount_path))


def _commit_mounted_volumes(volumes: tuple[MountedVolume, ...]) -> list[str]:
    import modal

    warnings: list[str] = []
    for volume in volumes:
        if not volume.commit_on_success:
            continue
        try:
            modal.Volume.from_name(
                volume.name,
                create_if_missing=volume.create_if_missing,
            ).commit()
        except Exception as exc:
            warnings.append(f"{volume.name}:{volume.mount_path}: {exc}")
    return warnings


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9-]+", "-", value).strip("-").lower() or "default"


def _modal_app_name(
    *,
    spec_payload: Mapping[str, Any],
    workflow_context: Mapping[str, Any] | None,
) -> str:
    step_name = None
    if isinstance(workflow_context, Mapping):
        step_name = workflow_context.get("step_name") or workflow_context.get("workflow_step_key")
    if step_name:
        return f"xenon-{_slug(str(step_name))}"[:80].rstrip("-")
    suffix = _slug(str(spec_payload.get("kind") or "operation"))
    return f"xenon-{suffix}"[:80].rstrip("-") or "xenon-operation"


def _resolved_runtime_spec(
    *,
    spec_payload: dict[str, Any],
) -> PythonRuntimeSpec:
    spec = operation_spec_from_dict(spec_payload)
    runtime_spec = spec.runtime_spec()
    if runtime_spec is None:
        raise NotImplementedError(f"ModalRunner requires a runtime spec, got {spec.kind!r}")
    if not isinstance(runtime_spec, PythonRuntimeSpec):
        raise NotImplementedError(
            f"ModalRunner requires a PythonRuntimeSpec, got {type(runtime_spec).__name__}"
        )
    return PythonRuntimeSpec(
        python_version=runtime_spec.python_version,
        pip_packages=runtime_spec.pip_packages,
        env=dict(runtime_spec.env),
        secrets=_merge_runtime_secrets(runtime_spec.secrets, spec.runtime_secrets()),
        local_python_sources=runtime_spec.local_python_sources,
    )


def _merge_runtime_secrets(
    base: tuple[RuntimeSecret, ...],
    extra: tuple[RuntimeSecret, ...],
) -> tuple[RuntimeSecret, ...]:
    unique: dict[str, RuntimeSecret] = {secret.env_var: secret for secret in base}
    for secret in extra:
        unique.setdefault(secret.env_var, secret)
    return tuple(unique[key] for key in sorted(unique))


def _validate_secret_bindings(*, runtime_spec: PythonRuntimeSpec, resources: dict[str, Any]) -> None:
    required = {secret.env_var for secret in runtime_spec.secrets}
    if not required:
        return
    provided = {
        str(env_var)
        for secret in resources.get("secrets", [])
        for env_var in secret.get("env_vars", [])
    }
    missing = sorted(required - provided)
    if missing:
        raise RuntimeError(f"Modal runtime is missing secret bindings for env vars: {missing}")


def _resolved_local_python_sources(sources: tuple[str, ...]) -> tuple[tuple[tuple[Path, str], ...], tuple[str, ...]]:
    workspace_root = find_workspace_root()
    resolved_mounts: list[tuple[Path, str]] = []
    pythonpath_entries: list[str] = []
    for source in sources:
        normalized = str(source).strip()
        if not normalized:
            continue
        if normalized == ".":
            local_path = workspace_root
            remote_path = _REMOTE_WORKSPACE_ROOT
            pythonpath_entry = _REMOTE_WORKSPACE_ROOT
        else:
            local_path = resolve_workspace_path(normalized, workspace_root=workspace_root)
            relative = local_path.relative_to(workspace_root)
            remote_path = f"{_REMOTE_WORKSPACE_ROOT}/{relative.as_posix()}"
            pythonpath_entry = _REMOTE_WORKSPACE_ROOT
        mount = (local_path, remote_path)
        if mount not in resolved_mounts:
            resolved_mounts.append(mount)
        if pythonpath_entry not in pythonpath_entries:
            pythonpath_entries.append(pythonpath_entry)
    return tuple(resolved_mounts), tuple(pythonpath_entries)
