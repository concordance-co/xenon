from __future__ import annotations

import json
import subprocess

from pipelines_v2.cli import main
from pipelines_v2.core.types import utc_now_iso
from pipelines_v2.storage.local import FileCatalog
from pipelines_v2.workflow.records import WorkflowRunRecord, WorkflowStepRecord


def test_workflow_cancel_stops_active_modal_apps_and_marks_run_cancelled(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    run_id = "wr_cancel_test"
    catalog = FileCatalog(root=tmp_path)
    catalog.record_workflow_run(
        WorkflowRunRecord(
            run_id=run_id,
            workflow_name="cancel_test",
            workflow_hash="workflow-hash",
            workflow_spec_hash="workflow-spec-hash",
            workflow_payload={"name": "cancel_test", "steps": []},
            status="running",
            started_at=utc_now_iso(),
        )
    )
    catalog.record_workflow_step(
        WorkflowStepRecord(
            run_id=run_id,
            workflow_hash="workflow-hash",
            workflow_step_key="capture",
            step_name="capture",
            step_index=0,
            runner="capture_gpu",
            status="running",
            step_semantic_hash="semantic-hash",
            step_spec_hash="spec-hash",
            runtime_app_id="ap-active",
        )
    )
    calls: list[list[str]] = []

    def stop_app(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("pipelines_v2.cli.subprocess.run", stop_app)

    exit_code = main(
        [
            "workflow",
            "cancel",
            "--run-id",
            run_id,
            "--local-catalog-root",
            str(tmp_path),
            "--yes",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    persisted = catalog.load_workflow_run(run_id)

    assert exit_code == 0
    assert payload["canceled"] is True
    assert payload["stopped_runtime_app_ids"] == ["ap-active"]
    assert len(calls) == 1
    assert calls[0][1:] == ["-m", "modal", "app", "stop", "ap-active"]
    assert persisted is not None
    assert persisted.status == "cancelled"
    assert persisted.finished_at is not None


def test_workflow_cancel_reports_modal_stop_failure(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    run_id = "wr_cancel_failure"
    catalog = FileCatalog(root=tmp_path)
    catalog.record_workflow_run(
        WorkflowRunRecord(
            run_id=run_id,
            workflow_name="cancel_failure",
            workflow_hash="workflow-hash",
            workflow_spec_hash="workflow-spec-hash",
            workflow_payload={"name": "cancel_failure", "steps": []},
            status="running",
            started_at=utc_now_iso(),
        )
    )
    catalog.record_workflow_step(
        WorkflowStepRecord(
            run_id=run_id,
            workflow_hash="workflow-hash",
            workflow_step_key="capture",
            step_name="capture",
            step_index=0,
            runner="capture_gpu",
            status="running",
            step_semantic_hash="semantic-hash",
            step_spec_hash="spec-hash",
            runtime_app_id="ap-active",
        )
    )
    monkeypatch.setattr(
        "pipelines_v2.cli.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="not stopped",
        ),
    )

    exit_code = main(
        [
            "workflow",
            "cancel",
            "--run-id",
            run_id,
            "--local-catalog-root",
            str(tmp_path),
            "--yes",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    persisted = catalog.load_workflow_run(run_id)

    assert exit_code == 2
    assert payload["canceled"] is False
    assert payload["stopped_runtime_app_ids"] == []
    assert payload["warnings"] == ["ap-active: not stopped"]
    assert persisted is not None
    assert persisted.status == "running"
