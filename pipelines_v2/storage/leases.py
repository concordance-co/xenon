"""Atomic ownership only, not a run catalog. No elapsed-time lease stealing."""
from __future__ import annotations

from contextlib import contextmanager
from uuid import uuid4
import re
from pipelines_v2.storage.finalization import IntegrityError

LOCK_DICT_NAME = "xenon-artifact-writers-v1"


@contextmanager
def atomic_owner(table, key, owner, is_terminated):
    """Monotonic immutable ownership slots avoid orphanable recovery mutexes.

    Release is a separate immutable marker. An owner killed before release is
    skipped only after coordinator cancellation and fresh terminal proof. Slots
    retain fencing history; no compare-and-delete race or clock-based stealing.
    """
    slot = None
    for generation in range(1000):
        candidate = f"{key}:owner:{generation}"
        previous = table.get(candidate)
        if previous is None:
            if table.put(candidate, owner, skip_if_exists=True):
                slot = candidate
                break
            previous = table.get(candidate)
        if previous is None:
            raise IntegrityError("Artifact owner could not be read after a conflicting acquisition")
        if table.get(candidate + ":released") == previous or is_terminated(previous):
            continue
        raise IntegrityError(f"Artifact writer still active or termination unverified: {previous.get('call_id')}")
    if slot is None:
        raise IntegrityError("Artifact ownership history exhausted; use a new attempt artifact after verifying old owners stopped")
    try:
        yield
    finally:
        # Only this owner can release its slot. A killed owner leaves no mutex.
        if table.get(slot) == owner:
            table.put(slot + ":released", owner, skip_if_exists=True)


def _canonical_input_id(raw):
    """Modal 1.3.4 exposes an attempt suffix absent from its public call graph.

    Keep the raw ID for diagnostics. Recognize only the observed provider format;
    normalization establishes identity correspondence, never terminal status.
    """
    if not isinstance(raw, str):
        return None
    match = re.fullmatch(r"(in-[0-9A-HJKMNP-TV-Z]{26})(?::[0-9]+-[0-9]+)?", raw)
    return match.group(1) if match else None


def _terminal_call_inputs(call, call_id):
    """Return exact-call canonical inputs only when every visible node is terminal.

    This observation is insufficient on its own: Modal can expose an earlier
    attempt's FAILURE while retrying the same input. Cancellation must precede it.
    """
    from modal.call_graph import InputStatus
    terminal = {InputStatus.SUCCESS, InputStatus.FAILURE, InputStatus.INIT_FAILURE,
                InputStatus.TERMINATED, InputStatus.TIMEOUT}
    nodes = []

    def visit(items):
        for node in items:
            if node.function_call_id == call_id:
                nodes.append(node)
            visit(node.children)

    visit(call.get_call_graph())
    canonical = [_canonical_input_id(node.input_id) for node in nodes]
    if not nodes or any(value is None for value in canonical):
        return None
    if any(node.status not in terminal for node in nodes):
        return None
    return sorted(set(canonical))


def _cancellation_certificate(call_id, canonical_inputs):
    return {"kind": "cancel_ack_and_terminal_graph_v1", "call_id": call_id,
            "canonical_input_ids": canonical_inputs, "terminate_containers": True}


def cancel_modal_writer_call(call_id: str, *, expected_input_id: str | None = None) -> dict:
    """Coordinator-only cancellation proof for a call the coordinator owns.

    Return an immutable certificate after acknowledged forced cancellation and
    a matching terminal graph. Raise IntegrityError when proof is unavailable;
    cancellation may already have succeeded, so callers may retry within their
    deadline. Acquisition never calls this helper or cancels another writer.
    """
    import modal
    expected = _canonical_input_id(expected_input_id) if expected_input_id is not None else None
    if not isinstance(call_id, str) or not call_id or (expected_input_id is not None and expected is None):
        raise IntegrityError("Cancellation requires a valid owned call/input identity")
    try:
        call = modal.FunctionCall.from_id(call_id)
        call.cancel(terminate_containers=True)
        inputs = _terminal_call_inputs(call, call_id)
        if inputs is None or (expected is not None and expected not in inputs):
            raise IntegrityError("Owned call cancellation acknowledged but matching terminal graph is unavailable")
        certificate = _cancellation_certificate(call_id, inputs)
        table = modal.Dict.from_name(LOCK_DICT_NAME, create_if_missing=False)
        key = f"stopped-call:{call_id}"
        if not table.put(key, certificate, skip_if_exists=True) and table.get(key) != certificate:
            raise IntegrityError("Conflicting or unreadable owned-call cancellation certificate")
        return certificate
    except IntegrityError:
        raise
    except Exception as exc:
        raise IntegrityError(f"Owned-call cancellation proof unavailable for {call_id}: {type(exc).__name__}") from exc


def _modal_terminated(owner):
    import modal
    canonical = _canonical_input_id(owner.get("input_id"))
    call_id = owner.get("call_id")
    if not call_id or canonical is None:
        return False
    if owner.get("canonical_input_id", canonical) != canonical:
        return False
    try:
        table = modal.Dict.from_name(LOCK_DICT_NAME, create_if_missing=False)
        certificate = table.get(f"stopped-call:{call_id}")
        if not isinstance(certificate, dict):
            return False
        inputs = certificate.get("canonical_input_ids")
        if not isinstance(inputs, list) or not inputs or any(_canonical_input_id(value) != value for value in inputs):
            return False
        if canonical not in inputs or inputs != sorted(set(inputs)):
            return False
        if certificate != _cancellation_certificate(call_id, inputs) or certificate.get("terminate_containers") is not True:
            return False
        # Never infer termination from a stale FAILURE graph alone. A coordinator
        # first forcibly cancels this exact call, then reuse rechecks its graph.
        fresh_inputs = _terminal_call_inputs(modal.FunctionCall.from_id(call_id), call_id)
        return fresh_inputs == inputs
    except Exception:
        return False


@contextmanager
def modal_writer_lock(store, key):
    import modal
    call_id, input_id = modal.current_function_call_id(), modal.current_input_id()
    if not call_id or not input_id:
        raise IntegrityError("Modal payload writing requires a worker call/input ownership identity")
    # Provision/readiness is explicit; offline planning never creates cloud resources.
    table = modal.Dict.from_name(LOCK_DICT_NAME, create_if_missing=False)
    owner = {"attempt_id": uuid4().hex, "call_id": call_id, "input_id": input_id,
             "canonical_input_id": _canonical_input_id(input_id)}
    scoped_key = f"{store.name}:{store.root}:{key}"
    with atomic_owner(table, scoped_key, owner, _modal_terminated):
        modal.Volume.from_name(store.name).reload()
        yield
