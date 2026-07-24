from __future__ import annotations

import json
import threading

import pytest

from pipelines_v2.api import FileCatalog, SpecValidationError, WorkflowOrchestrator, WorkflowSpec
from pipelines_v2.runtime import remote_executor
from pipelines_v2.runtime.modal_worker import (
    _consume_progress_stream,
    _execute_with_progress_stream,
    _remote_progress_stream,
)
from pipelines_v2.runtime.remote_executor import _emit_remote_progress
from pipelines_v2.workflow.progress import FileWorkflowProgressStore, WorkflowProgressEvent, WorkflowProgressSink
from pipelines_v2.workflow.records import WorkflowStepContext


def test_orchestrator_accepts_stable_new_run_id(tmp_path) -> None:
    catalog = FileCatalog(tmp_path / "catalog")
    orchestrator = WorkflowOrchestrator(runners={}, workflow_catalog=catalog)
    workflow = WorkflowSpec(name="empty", steps=())

    result = orchestrator.run(workflow, new_run_id="wr_harness_stable")

    assert result.run_id == "wr_harness_stable"
    with pytest.raises(SpecValidationError, match="already exists"):
        orchestrator.run(workflow, new_run_id="wr_harness_stable")


def test_workflow_progress_events_are_versioned_and_preserve_runtime_identity(tmp_path) -> None:
    store = FileWorkflowProgressStore(tmp_path / "catalog")
    first = WorkflowProgressEvent(
        run_id="wr_progress",
        workflow_name="demo",
        step_name="capture",
        status="running",
        stage="modal_app_started",
        runtime_kind="modal",
        runtime_app_id="ap-test",
        metrics={"app_name": "xenon-capture"},
    )
    store.record_event(first)
    store.record_event(
        WorkflowProgressEvent(
            run_id="wr_progress",
            workflow_name="demo",
            step_name="capture",
            status="running",
            stage="heartbeat",
        )
    )

    snapshot = store.load_step_snapshots("wr_progress")["capture"]
    assert snapshot["schema_version"] == "xenon.progress.v1"
    assert snapshot["event_id"].startswith("xpe_")
    assert snapshot["runtime_app_id"] == "ap-test"
    assert snapshot["metrics"]["app_name"] == "xenon-capture"


def test_remote_progress_includes_modal_container_identity(monkeypatch) -> None:
    messages: list[str] = []
    callbacks: list[dict[str, object]] = []
    monkeypatch.setattr(
        remote_executor._PROGRESS_LOG,
        "info",
        lambda _format, payload: messages.append(str(payload)),
    )

    _emit_remote_progress(
        workflow_context={
            "run_id": "wr_progress",
            "workflow_name": "demo",
            "step_name": "capture",
            "runner": "capture_gpu",
            "runtime_kind": "modal",
            "runtime_app_id": "ap-test",
            "runtime_app_name": "xenon-capture",
            "execution_shard": {"index": 1, "count": 3},
        },
        status="running",
        stage="generation",
        spec_kind="capture",
        message="Generating",
        metrics={"current": 12, "total": 30, "unit": "prompts"},
        progress_callback=lambda event: callbacks.append(dict(event)),
    )

    payload = json.loads(messages[-1])
    assert payload["schema_version"] == "xenon.progress.v1"
    assert payload["runtime_app_id"] == "ap-test"
    assert payload["metrics"] == {
        "app_name": "xenon-capture",
        "container_count": 3,
        "container_id": "container-2",
        "container_index": 1,
        "container_label": "Container 2",
        "current": 12,
        "total": 30,
        "unit": "prompts",
    }
    assert callbacks == [payload]


def test_modal_progress_stream_forwards_events_before_result() -> None:
    forwarded: list[dict[str, object]] = []

    def execute(report) -> dict[str, object]:
        report({"stage": "model_loading", "status": "running"})
        report({"stage": "model_loading", "status": "complete"})
        return {"artifact_id": "capture_test"}

    result = _consume_progress_stream(
        _execute_with_progress_stream(execute),
        lambda event: forwarded.append(dict(event)),
    )

    assert result == {"artifact_id": "capture_test"}
    assert [event["status"] for event in forwarded] == ["running", "complete"]


def test_modal_progress_stream_yields_before_blocking_execution_finishes() -> None:
    release = threading.Event()

    def execute(report) -> dict[str, object]:
        report({"stage": "model_loading", "status": "running"})
        release.wait(timeout=2)
        return {"artifact_id": "capture_test"}

    stream = iter(_execute_with_progress_stream(execute))
    first = next(stream)
    assert first["kind"] == "progress"
    assert first["event"]["stage"] == "model_loading"
    release.set()
    assert list(stream)[-1] == {
        "kind": "result",
        "result": {"artifact_id": "capture_test"},
    }


def test_modal_progress_stream_uses_remote_gen() -> None:
    calls: list[str] = []

    class FakeRemoteFunction:
        def remote(self, *_args):
            raise AssertionError("generator functions must not use remote()")

        def remote_gen(self, *_args):
            calls.append("remote_gen")
            yield {"kind": "result", "result": {"ok": True}}

    result = _consume_progress_stream(
        _remote_progress_stream(FakeRemoteFunction(), "argument"),
        None,
    )

    assert result == {"ok": True}
    assert calls == ["remote_gen"]


def test_modal_progress_stream_coalesces_rapid_running_updates() -> None:
    forwarded: list[dict[str, object]] = []

    def execute(report) -> dict[str, object]:
        for current in range(100):
            report(
                {
                    "stage": "generation",
                    "status": "running",
                    "metrics": {
                        "container_id": "container-1",
                        "current": current,
                        "total": 100,
                    },
                }
            )
        report(
            {
                "stage": "generation",
                "status": "complete",
                "metrics": {
                    "container_id": "container-1",
                    "current": 100,
                    "total": 100,
                },
            }
        )
        return {"artifact_id": "capture_test"}

    _consume_progress_stream(
        _execute_with_progress_stream(execute),
        lambda event: forwarded.append(dict(event)),
    )

    assert len(forwarded) < 10
    assert forwarded[-1]["status"] == "complete"


def test_batched_remote_progress_routes_to_its_originating_step(tmp_path) -> None:
    store = FileWorkflowProgressStore(tmp_path / "catalog")
    orchestrator = WorkflowOrchestrator(
        runners={},
        workflow_catalog=FileCatalog(tmp_path / "catalog"),
        progress_sink=WorkflowProgressSink(store=store),
    )
    contexts = [
        WorkflowStepContext(
            run_id="wr_batch",
            workflow_name="demo",
            workflow_hash="wf_hash",
            workflow_spec_hash="wf_spec_hash",
            step_name=f"step_{index}",
            step_index=index,
            runner="remote",
            step_semantic_hash=f"semantic_{index}",
            step_spec_hash=f"spec_{index}",
        )
        for index in range(2)
    ]
    callback = orchestrator._batch_runner_progress_callback(
        step_contexts=contexts,
        spec_kinds=["capture", "generation_run"],
    )
    assert callback is not None

    callback(
        {
            "step_name": "step_1",
            "status": "running",
            "stage": "generation",
            "runtime_kind": "modal",
            "runtime_app_id": "ap-test",
            "metrics": {"current": 2, "total": 4},
        }
    )

    snapshots = store.load_step_snapshots("wr_batch")
    assert set(snapshots) == {"step_1"}
    assert snapshots["step_1"]["stage"] == "generation"
