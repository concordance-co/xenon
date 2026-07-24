"""Generic Modal transport for remote execution."""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from pipelines_v2.core.paths import find_workspace_root
from pipelines_v2.core.types import RuntimeSecret
from pipelines_v2.engine import PythonRuntimeSpec
from pipelines_v2.operations import operation_spec_from_dict
from pipelines_v2.runtime.env import merged_runtime_env
from pipelines_v2.storage.modal import modal_volume_mount_path


_REMOTE_WORKSPACE_ROOT = "/root/pipelines_v2_workspace"
_LOG = logging.getLogger("pipelines_v2.modal")
_PROGRESS_INTERVAL_SECONDS = 0.1
_PROGRESS_QUEUE_LIMIT = 256


@dataclass(frozen=True, slots=True)
class MountedVolume:
    name: str
    mount_path: str
    create_if_missing: bool = False
    commit_on_success: bool = False


def _execute_with_progress_stream(
    execute: Callable[[Callable[[Mapping[str, Any]], None]], Any],
) -> Iterable[dict[str, Any]]:
    """Run blocking remote work while yielding progress callback payloads."""

    messages: queue.Queue[tuple[str, Any]] = queue.Queue(
        maxsize=_PROGRESS_QUEUE_LIMIT,
    )
    outcome: dict[str, Any] = {}
    last_emitted_at: dict[tuple[str, str, str], float] = {}

    def _report(payload: Mapping[str, Any]) -> None:
        event = dict(payload)
        metrics = event.get("metrics") if isinstance(event.get("metrics"), Mapping) else {}
        key = (
            str(event.get("step_name") or ""),
            str(metrics.get("container_id") or ""),
            str(event.get("stage") or ""),
        )
        status = str(event.get("status") or "running").lower()
        now = time.monotonic()
        if status == "running":
            previous = last_emitted_at.get(key)
            if previous is not None and now - previous < _PROGRESS_INTERVAL_SECONDS:
                return
        last_emitted_at[key] = now
        messages.put(("progress", event))

    def _run() -> None:
        try:
            outcome["result"] = execute(_report)
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            messages.put(("done", None))

    worker = threading.Thread(target=_run, name="xenon-modal-progress", daemon=True)
    worker.start()
    while True:
        kind, payload = messages.get()
        if kind == "done":
            break
        yield {"kind": kind, "event": payload}
    worker.join()
    if "error" in outcome:
        raise outcome["error"]
    yield {"kind": "result", "result": outcome.get("result")}


def _remote_progress_stream(remote_function: Any, *args: Any) -> Iterable[Mapping[str, Any]]:
    """Call a Modal generator through the streaming SDK surface."""

    remote_gen = getattr(remote_function, "remote_gen", None)
    if not callable(remote_gen):
        raise RuntimeError(
            "Installed Modal SDK does not expose Function.remote_gen; "
            "upgrade Modal to use live Xenon progress streaming"
        )
    return remote_gen(*args)


def _consume_progress_stream(
    stream: Iterable[Mapping[str, Any]],
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
) -> Any:
    result: Any = None
    result_seen = False
    for envelope in stream:
        kind = str(envelope.get("kind") or "")
        if kind == "progress":
            event = envelope.get("event")
            if progress_callback is not None and isinstance(event, Mapping):
                try:
                    progress_callback(event)
                except Exception:
                    _LOG.warning("local Modal progress callback failed", exc_info=True)
            continue
        if kind == "result":
            result = envelope.get("result")
            result_seen = True
            continue
        raise RuntimeError(f"Modal progress stream returned unknown envelope kind: {kind!r}")
    if not result_seen:
        raise RuntimeError("Modal progress stream ended without a result")
    return result


def _serialized_progress_callback(
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
) -> Callable[[Mapping[str, Any]], None] | None:
    if progress_callback is None:
        return None
    lock = threading.Lock()

    def _callback(payload: Mapping[str, Any]) -> None:
        with lock:
            progress_callback(payload)

    return _callback


def run_on_modal(
    *,
    runner_config: dict[str, Any],
    store_config: dict[str, Any],
    spec_payload: dict[str, Any],
    workflow_context: dict[str, Any] | None = None,
    workspace_root: str | Path | None = None,
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
    modal_app_name = _modal_app_name(
        spec_payload=spec_payload,
        workflow_context=workflow_context,
    )
    app = modal.App(modal_app_name)
    image = modal.Image.debian_slim(python_version=runtime_spec.python_version)
    if runtime_spec.pip_packages:
        image = image.pip_install(*runtime_spec.pip_packages)
    runtime_env = merged_runtime_env(runtime_spec.env, resources.get("env"))
    source_mounts, pythonpath_entries = _resolved_local_python_sources(
        runtime_spec.local_python_sources,
        workspace_root=workspace_root,
    )
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

    stream_function_kwargs = {**function_kwargs, "is_generator": True}

    @app.function(**stream_function_kwargs)
    def _remote_execute_stream(
        remote_runner_config: dict[str, Any],
        remote_store_config: dict[str, Any],
        remote_spec_payload: dict[str, Any],
        remote_workflow_context: dict[str, Any] | None,
    ) -> Iterable[dict[str, Any]]:
        _configure_remote_logging()
        from pipelines_v2.runtime.remote_executor import execute_remote

        def _execute(
            report_progress: Callable[[Mapping[str, Any]], None],
        ) -> dict[str, Any]:
            result = execute_remote(
                runner_config=remote_runner_config,
                store_config=remote_store_config,
                spec_payload=remote_spec_payload,
                workflow_context=remote_workflow_context,
                progress_callback=report_progress,
            )
            warnings = _commit_mounted_volumes(mounted_volumes)
            if warnings:
                result.setdefault("metadata", {})["volume_commit_warnings"] = warnings
            return result

        yield from _execute_with_progress_stream(_execute)

    @app.function(**function_kwargs)
    def _remote_merge_shards(
        remote_runner_config: dict[str, Any],
        remote_store_config: dict[str, Any],
        remote_spec_payload: dict[str, Any],
        remote_shard_manifests: list[dict[str, Any]],
        remote_workflow_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        _configure_remote_logging()
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
                    "app_name": modal_app_name,
                    "source_mount_count": len(source_mounts),
                },
            }
        )
    with app.run() as running_app:
        runtime_app_id = getattr(running_app, "app_id", None)
        streamed_progress_callback = _serialized_progress_callback(progress_callback)
        if progress_callback is not None:
            progress_callback(
                {
                    "status": "running",
                    "stage": "modal_app_started",
                    "runtime_kind": "modal",
                    "runtime_app_id": runtime_app_id,
                    "message": "Modal app started",
                    "metrics": {"app_name": modal_app_name},
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
                    _workflow_context_with_runtime(
                        workflow_context,
                        runtime_app_id=runtime_app_id,
                        runtime_app_name=modal_app_name,
                        index=index,
                        count=shard_count,
                    )
                    for index in range(shard_count)
                ]
                with ThreadPoolExecutor(max_workers=shard_count) as executor:
                    futures = [
                        executor.submit(
                            _consume_progress_stream,
                            _remote_progress_stream(
                                _remote_execute_stream,
                                runner_config,
                                store_config,
                                spec_payload,
                                shard_context,
                            ),
                            streamed_progress_callback,
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
                    _workflow_context_with_runtime(
                        workflow_context,
                        runtime_app_id=runtime_app_id,
                        runtime_app_name=modal_app_name,
                    ),
                )
            else:
                result = _consume_progress_stream(
                    _remote_progress_stream(
                        _remote_execute_stream,
                        runner_config,
                        store_config,
                        spec_payload,
                        _workflow_context_with_runtime(
                            workflow_context,
                            runtime_app_id=runtime_app_id,
                            runtime_app_name=modal_app_name,
                        ),
                    ),
                    streamed_progress_callback,
                )
        except Exception as exc:
            if progress_callback is not None:
                progress_callback(
                    {
                        "status": "error",
                        "stage": "remote_execution_failed",
                        "runtime_kind": "modal",
                        "runtime_app_id": runtime_app_id,
                        "message": f"Modal remote execution failed: {exc}",
                        "metrics": {"app_name": modal_app_name},
                    }
                )
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


def run_many_on_modal(
    *,
    runner_config: dict[str, Any],
    store_config: dict[str, Any],
    spec_payloads: list[dict[str, Any]],
    workflow_contexts: list[dict[str, Any] | None] | None = None,
    workspace_root: str | Path | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Submit multiple serialized work items to one Modal app invocation."""

    import modal

    if not spec_payloads:
        return []
    contexts = list(workflow_contexts) if workflow_contexts is not None else [None] * len(spec_payloads)
    if len(contexts) != len(spec_payloads):
        raise ValueError(
            "run_many_on_modal expected one workflow context per spec payload: "
            f"got {len(contexts)} contexts for {len(spec_payloads)} specs"
        )
    resources = runner_config.get("resources", {})
    shard_counts = [_modal_shard_count(resources=resources, spec_payload=payload) for payload in spec_payloads]
    distinct_shard_counts = sorted(set(shard_counts))
    if len(distinct_shard_counts) != 1:
        raise ValueError(
            "Modal batched execution requires one shard_count across all specs: "
            f"got {distinct_shard_counts}"
        )
    shard_count = int(distinct_shard_counts[0])

    runtime_spec = _resolved_runtime_spec_many(spec_payloads=spec_payloads)
    if not isinstance(runtime_spec, PythonRuntimeSpec):
        raise NotImplementedError(
            f"ModalRunner requires a PythonRuntimeSpec, got {type(runtime_spec).__name__}"
        )
    _validate_secret_bindings(runtime_spec=runtime_spec, resources=resources)
    mounted_volumes = _mounted_volumes(store_config=store_config, resources=resources)
    modal_app_name = _modal_batch_app_name(workflow_contexts=contexts)
    app = modal.App(modal_app_name)
    image = modal.Image.debian_slim(python_version=runtime_spec.python_version)
    if runtime_spec.pip_packages:
        image = image.pip_install(*runtime_spec.pip_packages)
    runtime_env = merged_runtime_env(runtime_spec.env, resources.get("env"))
    source_mounts, pythonpath_entries = _resolved_local_python_sources(
        runtime_spec.local_python_sources,
        workspace_root=workspace_root,
    )
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

    stream_function_kwargs = {**function_kwargs, "is_generator": True}

    @app.function(**stream_function_kwargs)
    def _remote_execute_many_stream(
        remote_runner_config: dict[str, Any],
        remote_store_config: dict[str, Any],
        remote_spec_payloads: list[dict[str, Any]],
        remote_workflow_contexts: list[dict[str, Any] | None],
    ) -> Iterable[dict[str, Any]]:
        _configure_remote_logging()
        from pipelines_v2.runtime.remote_executor import execute_remote_many

        def _execute(
            report_progress: Callable[[Mapping[str, Any]], None],
        ) -> list[dict[str, Any]]:
            results = execute_remote_many(
                runner_config=remote_runner_config,
                store_config=remote_store_config,
                spec_payloads=remote_spec_payloads,
                workflow_contexts=remote_workflow_contexts,
                progress_callback=report_progress,
            )
            warnings = _commit_mounted_volumes(mounted_volumes)
            if warnings:
                for result in results:
                    result.setdefault("metadata", {})["volume_commit_warnings"] = warnings
            return results

        yield from _execute_with_progress_stream(_execute)

    @app.function(**function_kwargs)
    def _remote_merge_shards(
        remote_runner_config: dict[str, Any],
        remote_store_config: dict[str, Any],
        remote_spec_payload: dict[str, Any],
        remote_shard_manifests: list[dict[str, Any]],
        remote_workflow_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        _configure_remote_logging()
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
        "modal batch spin-up starting kind=%s store=%s gpu=%s cpu=%s memory_mb=%s specs=%d source_mounts=%d",
        runner_config.get("kind"),
        store_config.get("name"),
        resources.get("gpu"),
        resources.get("cpu"),
        resources.get("memory_mb"),
        len(spec_payloads),
        len(source_mounts),
    )
    if progress_callback is not None:
        progress_callback(
            {
                "status": "running",
                "stage": "modal_batch_launching",
                "runtime_kind": "modal",
                "message": "Starting Modal batch app launch",
                "metrics": {
                    "app_name": modal_app_name,
                    "source_mount_count": len(source_mounts),
                    "spec_count": len(spec_payloads),
                },
            }
        )
    with app.run() as running_app:
        runtime_app_id = getattr(running_app, "app_id", None)
        streamed_progress_callback = _serialized_progress_callback(progress_callback)
        if progress_callback is not None:
            progress_callback(
                {
                    "status": "running",
                    "stage": "modal_batch_app_started",
                    "runtime_kind": "modal",
                    "runtime_app_id": runtime_app_id,
                    "message": "Modal batch app started",
                    "metrics": {
                        "app_name": modal_app_name,
                        "spec_count": len(spec_payloads),
                    },
                }
            )
        _LOG.info(
            "modal batch submitted kind=%s runtime_app_id=%s specs=%d",
            runner_config.get("kind"),
            runtime_app_id,
            len(spec_payloads),
        )
        try:
            if shard_count > 1:
                if progress_callback is not None:
                    progress_callback(
                        {
                            "status": "running",
                            "stage": "remote_batch_shards_submitted",
                            "runtime_kind": "modal",
                            "runtime_app_id": runtime_app_id,
                            "message": f"Submitted {shard_count} remote batch execution shards",
                            "metrics": {
                                "shard_count": shard_count,
                                "spec_count": len(spec_payloads),
                            },
                        }
                    )
                shard_contexts = [
                    [
                        _workflow_context_with_runtime(
                            context,
                            runtime_app_id=runtime_app_id,
                            runtime_app_name=modal_app_name,
                            index=index,
                            count=shard_count,
                        )
                        for context in contexts
                    ]
                    for index in range(shard_count)
                ]
                with ThreadPoolExecutor(max_workers=shard_count) as executor:
                    futures = [
                        executor.submit(
                            _consume_progress_stream,
                            _remote_progress_stream(
                                _remote_execute_many_stream,
                                runner_config,
                                store_config,
                                spec_payloads,
                                contexts_for_shard,
                            ),
                            streamed_progress_callback,
                        )
                        for contexts_for_shard in shard_contexts
                    ]
                    shard_batches = [future.result() for future in futures]
                if progress_callback is not None:
                    progress_callback(
                        {
                            "status": "running",
                            "stage": "remote_batch_shards_finished",
                            "runtime_kind": "modal",
                            "runtime_app_id": runtime_app_id,
                            "message": f"Finished {shard_count} remote batch execution shards",
                            "metrics": {
                                "shard_count": shard_count,
                                "spec_count": len(spec_payloads),
                            },
                        }
                    )
                for shard_index, shard_results in enumerate(shard_batches):
                    if len(shard_results) != len(spec_payloads):
                        raise RuntimeError(
                            "Modal batch shard returned a different number of results than specs: "
                            f"shard {shard_index} got {len(shard_results)}, expected {len(spec_payloads)}"
                        )
                results = [
                    _remote_merge_shards.remote(
                        runner_config,
                        store_config,
                        spec_payload,
                        [shard_results[spec_index] for shard_results in shard_batches],
                        _workflow_context_with_runtime(
                            contexts[spec_index],
                            runtime_app_id=runtime_app_id,
                            runtime_app_name=modal_app_name,
                        ),
                    )
                    for spec_index, spec_payload in enumerate(spec_payloads)
                ]
            else:
                runtime_contexts = [
                    _workflow_context_with_runtime(
                        context,
                        runtime_app_id=runtime_app_id,
                        runtime_app_name=modal_app_name,
                    )
                    for context in contexts
                ]
                results = _consume_progress_stream(
                    _remote_progress_stream(
                        _remote_execute_many_stream,
                        runner_config,
                        store_config,
                        spec_payloads,
                        runtime_contexts,
                    ),
                    streamed_progress_callback,
                )
        except Exception as exc:
            if progress_callback is not None:
                progress_callback(
                    {
                        "status": "error",
                        "stage": "remote_batch_execution_failed",
                        "runtime_kind": "modal",
                        "runtime_app_id": runtime_app_id,
                        "message": f"Modal remote batch execution failed: {exc}",
                        "metrics": {
                            "app_name": modal_app_name,
                            "spec_count": len(spec_payloads),
                        },
                    }
                )
            if runtime_app_id is not None:
                try:
                    setattr(exc, "runtime_app_id", runtime_app_id)
                except Exception:
                    pass
            raise
        if runtime_app_id is not None:
            for result in results:
                if isinstance(result, dict):
                    runner_payload = dict(result.get("runner", {}))
                    runner_payload["runtime_app_id"] = runtime_app_id
                    result["runner"] = runner_payload
        if progress_callback is not None:
            progress_callback(
                {
                    "status": "running",
                    "stage": "remote_batch_execution_finished",
                    "runtime_kind": "modal",
                    "runtime_app_id": runtime_app_id,
                    "message": "Modal batch app finished remote execution",
                    "metrics": {"spec_count": len(spec_payloads)},
                }
            )
        return results


def _modal_shard_count(*, resources: dict[str, Any], spec_payload: dict[str, Any]) -> int:
    count = int(resources.get("shard_count") or 1)
    if count < 1:
        raise ValueError("ModalResources shard_count must be >= 1")
    if count == 1:
        return 1
    kind = str(spec_payload.get("kind") or "")
    if kind not in {"capture", "generation_run", "patched_generation"}:
        return 1
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


def _workflow_context_with_runtime(
    workflow_context: dict[str, Any] | None,
    *,
    runtime_app_id: str | None,
    runtime_app_name: str,
    index: int = 0,
    count: int = 1,
) -> dict[str, Any]:
    context = _workflow_context_with_shard(
        workflow_context,
        index=index,
        count=count,
    )
    context.update(
        {
            "runtime_kind": "modal",
            "runtime_app_id": runtime_app_id,
            "runtime_app_name": runtime_app_name,
            "container_index": int(index),
            "container_count": int(count),
            "container_id": f"container-{int(index) + 1}",
            "container_label": f"Container {int(index) + 1}",
        }
    )
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


def _modal_batch_app_name(
    *,
    workflow_contexts: list[Mapping[str, Any] | None],
) -> str:
    for context in workflow_contexts:
        if not isinstance(context, Mapping):
            continue
        workflow_name = context.get("workflow_name")
        if workflow_name:
            return f"xenon-batch-{_slug(str(workflow_name))}"[:80].rstrip("-")
    return "xenon-batch"


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


def _resolved_runtime_spec_many(
    *,
    spec_payloads: list[dict[str, Any]],
) -> PythonRuntimeSpec:
    runtime_specs = [_resolved_runtime_spec(spec_payload=payload) for payload in spec_payloads]
    if not runtime_specs:
        raise ValueError("Cannot resolve runtime spec for an empty Modal batch")
    python_versions = {spec.python_version for spec in runtime_specs}
    if len(python_versions) != 1:
        raise ValueError(f"Modal batch requires one Python version, got {sorted(python_versions)}")
    env: dict[str, str] = {}
    for runtime_spec in runtime_specs:
        for key, value in dict(runtime_spec.env).items():
            key = str(key)
            value = str(value)
            existing = env.get(key)
            if existing is not None and existing != value:
                raise ValueError(
                    "Modal batch requires compatible runtime env across specs; "
                    f"{key!r} has both {existing!r} and {value!r}"
                )
            env[key] = value
    return PythonRuntimeSpec(
        python_version=runtime_specs[0].python_version,
        pip_packages=_merge_string_sequences(*(spec.pip_packages for spec in runtime_specs)),
        env=env,
        secrets=_merge_runtime_secrets_many(*(spec.secrets for spec in runtime_specs)),
        local_python_sources=_merge_string_sequences(*(spec.local_python_sources for spec in runtime_specs)),
    )


def _merge_string_sequences(*items: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    for group in items:
        for item in group:
            value = str(item)
            if value not in merged:
                merged.append(value)
    return tuple(merged)


def _merge_runtime_secrets_many(*items: tuple[RuntimeSecret, ...]) -> tuple[RuntimeSecret, ...]:
    merged: tuple[RuntimeSecret, ...] = ()
    for group in items:
        merged = _merge_runtime_secrets(merged, group)
    return merged


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


def _configure_remote_logging() -> None:
    import os

    level_name = str(os.getenv("PIPELINES_V2_REMOTE_LOGGING", "INFO")).strip().upper()
    numeric_level = getattr(logging, level_name, None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
        level_name = "INFO"
    logger = logging.getLogger("pipelines_v2")
    handler = next(
        (candidate for candidate in logger.handlers if getattr(candidate, "_pipelines_v2_modal_handler", False)),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        setattr(handler, "_pipelines_v2_modal_handler", True)
        logger.addHandler(handler)
    handler.setLevel(numeric_level)
    logger.setLevel(numeric_level)
    logger.propagate = False
    logger.debug("remote Modal logging configured level=%s", level_name)


def _resolved_local_python_sources(
    sources: tuple[str, ...],
    *,
    workspace_root: str | Path | None = None,
) -> tuple[tuple[tuple[Path, str], ...], tuple[str, ...]]:
    primary_root = (
        Path(workspace_root).expanduser().resolve()
        if workspace_root is not None
        else find_workspace_root()
    )
    library_root = find_workspace_root(Path(__file__))
    source_roots = tuple(dict.fromkeys((primary_root, library_root)))
    resolved_mounts: list[tuple[Path, str]] = []
    pythonpath_entries: list[str] = []
    for source in sources:
        normalized = str(source).strip()
        if not normalized:
            continue
        if normalized == ".":
            local_path = primary_root
            remote_path = _REMOTE_WORKSPACE_ROOT
            pythonpath_entry = _REMOTE_WORKSPACE_ROOT
        else:
            local_path, relative = _resolve_local_python_source(
                normalized,
                source_roots=source_roots,
            )
            remote_path = f"{_REMOTE_WORKSPACE_ROOT}/{relative.as_posix()}"
            pythonpath_entry = _REMOTE_WORKSPACE_ROOT
        mount = (local_path, remote_path)
        conflicting = next(
            (
                existing
                for existing in resolved_mounts
                if existing[1] == remote_path and existing[0] != local_path
            ),
            None,
        )
        if conflicting is not None:
            raise ValueError(
                f"local_python_sources map multiple directories to {remote_path}: "
                f"{conflicting[0]} and {local_path}"
            )
        if mount not in resolved_mounts:
            resolved_mounts.append(mount)
        if pythonpath_entry not in pythonpath_entries:
            pythonpath_entries.append(pythonpath_entry)
    return tuple(resolved_mounts), tuple(pythonpath_entries)


def _resolve_local_python_source(
    source: str,
    *,
    source_roots: tuple[Path, ...],
) -> tuple[Path, Path]:
    source_path = Path(source).expanduser()
    attempted: list[Path] = []
    for root in source_roots:
        local_path = (
            source_path.resolve()
            if source_path.is_absolute()
            else (root / source_path).resolve()
        )
        try:
            relative = local_path.relative_to(root)
        except ValueError:
            continue
        attempted.append(local_path)
        if local_path.is_dir():
            return local_path, relative

    allowed = ", ".join(str(root) for root in source_roots)
    if not attempted:
        raise ValueError(
            f"local_python_source {source!r} must be inside an allowed workspace: {allowed}"
        )
    tried = ", ".join(str(path) for path in attempted)
    raise FileNotFoundError(
        f"local_python_source {source!r} does not identify a directory; tried: {tried}"
    )
