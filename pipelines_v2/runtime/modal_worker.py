"""Generic Modal transport for remote execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipelines_v2.core.paths import find_workspace_root, resolve_workspace_path
from pipelines_v2.core.types import RuntimeSecret
from pipelines_v2.engine import PythonRuntimeSpec
from pipelines_v2.operations import operation_spec_from_dict
from pipelines_v2.storage.modal import modal_volume_mount_path


_REMOTE_WORKSPACE_ROOT = "/root/pipelines_v2_workspace"


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
        f"pipelines-v2-{_slug(runner_config.get('kind', 'modal'))}-{_slug(str(store_config['name']))}"
    )
    image = modal.Image.debian_slim(python_version=runtime_spec.python_version)
    if runtime_spec.pip_packages:
        image = image.pip_install(*runtime_spec.pip_packages)
    runtime_env = dict(runtime_spec.env)
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
    if resources.get("gpu") is not None:
        function_kwargs["gpu"] = resources.get("gpu")
    if resources.get("cpu") is not None:
        function_kwargs["cpu"] = resources.get("cpu")
    if resources.get("memory_mb") is not None:
        function_kwargs["memory"] = int(resources["memory_mb"])

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

    with app.run() as running_app:
        runtime_app_id = getattr(running_app, "app_id", None)
        try:
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
        return result


def _mounted_volumes(*, store_config: dict[str, Any], resources: dict[str, Any]) -> tuple[MountedVolume, ...]:
    volumes = [
        MountedVolume(
            name=str(store_config["name"]),
            mount_path=modal_volume_mount_path(str(store_config["root"])),
            create_if_missing=True,
            commit_on_success=True,
        )
    ]
    for payload in resources.get("volumes", []):
        volumes.append(
            MountedVolume(
                name=str(payload["name"]),
                mount_path=str(payload["mount_path"]),
                create_if_missing=bool(payload.get("create_if_missing", False)),
                commit_on_success=bool(payload.get("commit_on_success", False)),
            )
        )
    mounts = [volume.mount_path for volume in volumes]
    duplicates = {mount for mount in mounts if mounts.count(mount) > 1}
    if duplicates:
        raise ValueError(f"Duplicate Modal volume mount paths: {sorted(duplicates)}")
    return tuple(volumes)


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
