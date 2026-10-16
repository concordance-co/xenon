"""Finite client waits; timeouts never retry a possibly submitted computation."""
from __future__ import annotations

import queue
import threading
import time
from typing import Any


def bounded_call(function, *, timeout_seconds: float, label: str, on_late_result=None):
    if timeout_seconds <= 0:
        raise TimeoutError(f"{label} exceeded its client wall deadline")
    outcome = queue.Queue(maxsize=1)
    state_lock = threading.Lock()
    expired = False
    def cleanup(value):
        if on_late_result is not None:
            try:
                on_late_result(value)
            except Exception:
                pass
    def run():
        try:
            result = (True, function())
        except BaseException as exc:
            result = (False, exc)
        with state_lock:
            late = expired
            if not late:
                outcome.put_nowait(result)
        if late and result[0]:
            cleanup(result[1])
    threading.Thread(target=run, name="xenon-client-deadline", daemon=True).start()
    try:
        okay, result = outcome.get(timeout=timeout_seconds)
    except queue.Empty:
        with state_lock:
            expired = True
            try:
                delivered = outcome.get_nowait()
            except queue.Empty:
                delivered = None
        if delivered is not None and delivered[0]:
            cleanup(delivered[1])
        raise TimeoutError(f"{label} exceeded its client wall deadline ({timeout_seconds:g}s)") from None
    if not okay:
        raise result
    return result


def cancel_owned_call(call_id, *, expected_input_id=None):
    """Only caller-owned IDs may request cancellation/recovery certification."""
    from pipelines_v2.storage.leases import cancel_modal_writer_call
    return bounded_call(lambda: cancel_modal_writer_call(call_id, expected_input_id=expected_input_id),
                        timeout_seconds=10, label="Modal cancellation certificate")


def _cancel_call(call):
    # First cancel even if certificate persistence/access fails. A missing
    # certificate remains fail-closed for subsequent storage recovery.
    result = bounded_call(lambda: call.cancel(terminate_containers=True), timeout_seconds=5, label="Modal cancellation")
    call_id = getattr(call, "object_id", None)
    if call_id:
        cancel_owned_call(call_id)
    return result


def bounded_remote_call(function, *args, timeout_seconds: float = 120, **kwargs):
    """Bound submit/get/cancel. Never blindly resubmit after an unknown outcome."""
    started = time.monotonic()
    call = bounded_call(lambda: function.spawn(*args, **kwargs), timeout_seconds=timeout_seconds, label="Modal submission", on_late_result=_cancel_call)
    try:
        remaining = timeout_seconds - (time.monotonic() - started)
        return bounded_call(lambda: call.get(timeout=max(0, remaining)), timeout_seconds=remaining, label="Modal result")
    except BaseException:
        try:
            _cancel_call(call)
        except Exception:
            pass
        raise


def bounded_stream(stream, *, timeout_seconds: float):
    """Includes queue time and gaps between progress events in one wall budget."""
    queue_ = queue.Queue(maxsize=256)
    stopped = threading.Event()
    def put(item):
        while not stopped.is_set():
            try:
                queue_.put(item, timeout=0.1)
                return
            except queue.Full:
                pass
    def read():
        try:
            for item in stream:
                if stopped.is_set():
                    break
                put(("item", item))
        except BaseException as exc:
            put(("error", exc))
        finally:
            put(("done", None))
    threading.Thread(target=read, name="xenon-modal-stream-deadline", daemon=True).start()
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Modal progress stream exceeded its client wall deadline")
            try:
                kind, value = queue_.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError("Modal progress stream exceeded its client wall deadline") from None
            if kind == "done":
                return
            if kind == "error":
                raise value
            yield value
    finally:
        stopped.set()
