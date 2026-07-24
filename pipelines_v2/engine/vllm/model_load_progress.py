"""Structured progress for vLLM model loading."""

from __future__ import annotations

import itertools
import json
import os
import socket
import struct
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_PROGRESS_PATH_ENV = "XENON_VLLM_PROGRESS_PATH"
_PROGRESS_SOCKET_ENV = "XENON_VLLM_PROGRESS_SOCKET"
_PROGRESS_TRANSPORT_ENV = "XENON_VLLM_PROGRESS_TRANSPORT"
_SHARD_PROGRESS_ENV = "XENON_VLLM_SHARD_PROGRESS"
_CUSTOM_WORKER_ENV = "XENON_VLLM_CUSTOM_WORKER"
_PROGRESS_WRITES_ENV = "XENON_VLLM_PROGRESS_WRITES"
_CHECKPOINT_DESCRIPTION = "Loading safetensors checkpoint shards"
_PROGRESS_WORKER_CLS = (
    "pipelines_v2.engine.vllm.model_load_progress.XenonProgressGPUWorker"
)
_LOADER_IDS = itertools.count(1)
_SOCKET_EVENT = struct.Struct("!IIiiB")
_SOCKET_STATUS = {"running": 0, "error": 1}
_SOCKET_CLIENTS: dict[str, socket.socket] = {}
_SOCKET_CLIENTS_LOCK = threading.Lock()


def enable_model_load_progress(
    llm_kwargs: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> None:
    """Opt into Xenon's exact shard instrumentation when explicitly requested.

    Supplying ``worker_cls`` bypasses vLLM's automatic platform worker
    selection. Keep that invasive path disabled by default so observability
    cannot change normal model-loading behavior.
    """

    if (
        progress_callback is not None
        and _progress_instrumentation_enabled()
        and not llm_kwargs.get("worker_cls")
    ):
        llm_kwargs["worker_cls"] = _PROGRESS_WORKER_CLS


@contextmanager
def model_load_progress_monitor(
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> Iterator[None]:
    """Forward progress emitted by vLLM worker processes to Xenon."""

    transport = _progress_transport()
    if progress_callback is None or transport == "off":
        yield
        return

    if transport == "socket":
        with _model_load_progress_socket_monitor(progress_callback):
            yield
        return

    if not _progress_writes_enabled():
        yield
        return

    with tempfile.TemporaryDirectory(prefix="xenon_vllm_progress_") as tmpdir:
        progress_path = Path(tmpdir) / "model_load.jsonl"
        stop = threading.Event()
        previous_path = os.environ.get(_PROGRESS_PATH_ENV)
        os.environ[_PROGRESS_PATH_ENV] = str(progress_path)
        thread = threading.Thread(
            target=_monitor_progress_file,
            args=(progress_path, progress_callback, stop),
            daemon=True,
            name="xenon-vllm-progress",
        )
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=2)
            if previous_path is None:
                os.environ.pop(_PROGRESS_PATH_ENV, None)
            else:
                os.environ[_PROGRESS_PATH_ENV] = previous_path


@contextmanager
def _model_load_progress_socket_monitor(
    progress_callback: Callable[[dict[str, Any]], None],
) -> Iterator[None]:
    with tempfile.TemporaryDirectory(prefix="xenon_vllm_socket_") as tmpdir:
        socket_path = Path(tmpdir) / "progress.sock"
        receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        receiver.bind(str(socket_path))
        receiver.settimeout(0.05)
        stop = threading.Event()
        previous_socket = os.environ.get(_PROGRESS_SOCKET_ENV)
        os.environ[_PROGRESS_SOCKET_ENV] = str(socket_path)
        thread = threading.Thread(
            target=_monitor_progress_socket,
            args=(receiver, progress_callback, stop),
            daemon=True,
            name="xenon-vllm-progress-socket",
        )
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=2)
            receiver.close()
            if previous_socket is None:
                os.environ.pop(_PROGRESS_SOCKET_ENV, None)
            else:
                os.environ[_PROGRESS_SOCKET_ENV] = previous_socket


@contextmanager
def tracked_safetensors_loading() -> Iterator[None]:
    """Instrument vLLM's checkpoint iterator inside a GPU worker process."""

    socket_path = str(os.getenv(_PROGRESS_SOCKET_ENV, "") or "").strip()
    progress_path = str(os.getenv(_PROGRESS_PATH_ENV, "") or "").strip()
    progress_target = socket_path or progress_path
    if not progress_target:
        yield
        return
    if not socket_path and not _progress_writes_enabled():
        yield
        return

    from vllm.model_executor.model_loader import weight_utils

    original_tqdm = weight_utils.tqdm
    weight_utils.tqdm = _tracked_tqdm(original_tqdm, Path(progress_target))
    try:
        yield
    finally:
        weight_utils.tqdm = original_tqdm


def _tracked_tqdm(
    original_tqdm: Callable[..., Any],
    progress_path: Path,
) -> Callable[..., Any]:
    def tracked(iterable: Any = None, *args: Any, **kwargs: Any) -> Any:
        progress = original_tqdm(iterable, *args, **kwargs)
        description = str(kwargs.get("desc") or getattr(progress, "desc", "") or "")
        if not description.startswith(_CHECKPOINT_DESCRIPTION):
            return progress

        total = _positive_int(getattr(progress, "total", None))
        if total is None:
            try:
                total = _positive_int(len(iterable))
            except (TypeError, AttributeError):
                total = None
        loader_id = f"{os.getpid()}:{next(_LOADER_IDS)}"

        def iterate() -> Iterator[Any]:
            current = 0
            _write_worker_progress(
                progress_path,
                loader_id=loader_id,
                current=current,
                total=total,
                status="running",
            )
            try:
                for item in progress:
                    yield item
                    current += 1
                    _write_worker_progress(
                        progress_path,
                        loader_id=loader_id,
                        current=current,
                        total=total,
                        status="running",
                    )
            except BaseException:
                _write_worker_progress(
                    progress_path,
                    loader_id=loader_id,
                    current=current,
                    total=total,
                    status="error",
                )
                raise

        return iterate()

    return tracked


def _progress_writes_enabled() -> bool:
    configured = str(os.getenv(_PROGRESS_WRITES_ENV, "1") or "").strip().lower()
    return configured not in {"0", "false", "no", "off"}


def _progress_instrumentation_enabled() -> bool:
    values = (
        os.getenv(_SHARD_PROGRESS_ENV, ""),
        os.getenv(_CUSTOM_WORKER_ENV, ""),
    )
    return any(
        str(value or "").strip().lower() in {"1", "true", "yes", "on"}
        for value in values
    )


def _progress_transport() -> str:
    if not _progress_instrumentation_enabled():
        return "off"
    configured = str(os.getenv(_PROGRESS_TRANSPORT_ENV, "") or "").strip().lower()
    if configured in {"off", "socket", "jsonl"}:
        return configured
    return "socket"


def _write_worker_progress(
    path: Path,
    *,
    loader_id: str,
    current: int,
    total: int | None,
    status: str,
) -> None:
    socket_path = str(os.getenv(_PROGRESS_SOCKET_ENV, "") or "").strip()
    if socket_path and str(path) == socket_path:
        _send_socket_progress(
            socket_path,
            loader_id=loader_id,
            current=current,
            total=total,
            status=status,
        )
        return

    payload = {
        "loader_id": loader_id,
        "current": current,
        "status": status,
        **({"total": total} if total is not None else {}),
    }
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def _send_socket_progress(
    path: str,
    *,
    loader_id: str,
    current: int,
    total: int | None,
    status: str,
) -> None:
    try:
        pid_value, sequence_value = loader_id.split(":", 1)
        payload = _SOCKET_EVENT.pack(
            int(pid_value),
            int(sequence_value),
            max(0, int(current)),
            int(total) if total is not None else -1,
            _SOCKET_STATUS.get(status, 0),
        )
        client = _progress_socket_client(path)
        client.sendto(payload, path)
    except (BlockingIOError, OSError, OverflowError, ValueError):
        _discard_progress_socket_client(path)


def _progress_socket_client(path: str) -> socket.socket:
    with _SOCKET_CLIENTS_LOCK:
        client = _SOCKET_CLIENTS.get(path)
        if client is None:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            client.setblocking(False)
            _SOCKET_CLIENTS[path] = client
        return client


def _discard_progress_socket_client(path: str) -> None:
    with _SOCKET_CLIENTS_LOCK:
        client = _SOCKET_CLIENTS.pop(path, None)
    if client is not None:
        client.close()


def _monitor_progress_socket(
    receiver: socket.socket,
    callback: Callable[[dict[str, Any]], None],
    stop: threading.Event,
) -> None:
    loaders: dict[str, tuple[int, int | None, str]] = {}
    idle_after_stop = 0
    while not stop.is_set() or idle_after_stop < 2:
        try:
            payload = receiver.recv(4096)
        except socket.timeout:
            if stop.is_set():
                idle_after_stop += 1
            continue
        except OSError:
            return
        event = _socket_worker_progress_event(payload)
        if event is None:
            continue
        idle_after_stop = 0
        loaders[str(event["loader_id"])] = (
            int(event["current"]),
            _positive_int(event.get("total")),
            str(event["status"]),
        )
        callback(_aggregate_loader_progress(loaders))


def _socket_worker_progress_event(payload: bytes) -> dict[str, Any] | None:
    if len(payload) != _SOCKET_EVENT.size:
        return None
    pid, sequence, current, total, status = _SOCKET_EVENT.unpack(payload)
    return {
        "loader_id": f"{pid}:{sequence}",
        "current": max(0, current),
        "total": total if total > 0 else None,
        "status": "error" if status == _SOCKET_STATUS["error"] else "running",
    }


def _monitor_progress_file(
    path: Path,
    callback: Callable[[dict[str, Any]], None],
    stop: threading.Event,
) -> None:
    offset = 0
    loaders: dict[str, tuple[int, int | None, str]] = {}
    idle_after_stop = 0
    while not stop.is_set() or idle_after_stop < 2:
        emitted = False
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(offset)
                while line := handle.readline():
                    offset = handle.tell()
                    event = _worker_progress_event(line)
                    if event is None:
                        continue
                    loaders[str(event["loader_id"])] = (
                        int(event["current"]),
                        _positive_int(event.get("total")),
                        str(event["status"]),
                    )
                    callback(_aggregate_loader_progress(loaders))
                    emitted = True
        if stop.is_set():
            idle_after_stop = 0 if emitted else idle_after_stop + 1
        time.sleep(0.05)


def _worker_progress_event(line: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(line)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or not payload.get("loader_id"):
        return None
    try:
        current = max(0, int(payload.get("current") or 0))
    except (TypeError, ValueError):
        return None
    return {
        "loader_id": str(payload["loader_id"]),
        "current": current,
        "total": payload.get("total"),
        "status": str(payload.get("status") or "running"),
    }


def _aggregate_loader_progress(
    loaders: Mapping[str, tuple[int, int | None, str]],
) -> dict[str, Any]:
    current = sum(item[0] for item in loaders.values())
    totals = [item[1] for item in loaders.values()]
    total = sum(value for value in totals if value is not None)
    has_complete_totals = bool(totals) and all(value is not None for value in totals)
    has_error = any(item[2] == "error" for item in loaders.values())
    return {
        "stage": "model_loading",
        "status": "error" if has_error else "running",
        "message": "Loading checkpoint shards",
        "current": current,
        **({"total": total, "unit": "shards"} if has_complete_totals else {}),
    }


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


try:
    from vllm.v1.worker.gpu_worker import Worker as _GPUWorker
except ImportError:
    _GPUWorker = object  # type: ignore[assignment,misc]


class XenonProgressGPUWorker(_GPUWorker):  # type: ignore[misc,valid-type]
    """vLLM GPU worker that emits structured checkpoint-shard progress."""

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        with tracked_safetensors_loading():
            super().load_model(load_dummy_weights=load_dummy_weights)


__all__ = [
    "XenonProgressGPUWorker",
    "enable_model_load_progress",
    "model_load_progress_monitor",
    "tracked_safetensors_loading",
]
