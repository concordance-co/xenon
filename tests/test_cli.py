from __future__ import annotations

from pathlib import Path

import pytest

from pipelines import cli


class _DummyConn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_cli_help_prints_available_surfaces(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "surface" in out
    assert "action" in out


def test_dataset_build_passthrough_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def _fake_main(argv: list[str] | None = None) -> int:
        seen.append(list(argv or []))
        return 7

    monkeypatch.setattr("pipelines.datasets.prepare.main", _fake_main)

    assert cli.main(["dataset", "build", "--", "--limit", "5"]) == 7
    assert seen == [["--limit", "5"]]


def test_capture_run_passthrough_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def _fake_main(argv: list[str] | None = None) -> int:
        seen.append(list(argv or []))
        return 3

    monkeypatch.setattr("pipelines.interp.local_capture.main", _fake_main)

    assert cli.main(["capture", "run", "--", "--model", "demo"]) == 3
    assert seen == [["--model", "demo"]]


def test_analysis_run_passthrough_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def _fake_main(argv: list[str] | None = None) -> None:
        seen.append(list(argv or []))
        return None

    monkeypatch.setattr("pipelines.interp.analysis.main", _fake_main)

    assert cli.main(["analysis", "run", "--", "--target", "decision_type"]) == 0
    assert seen == [["--target", "decision_type"]]


def test_spec_create_uses_workflow_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        '{"name":"Demo","dataset":{"source":{"mode":"table","table":"interp_examples_v0"},"label":{"mode":"direct","expression_sql":"decision_type"}}}'
    )

    monkeypatch.setattr("pipelines.cli._open_conn", lambda: _DummyConn())

    def _fake_upsert(conn: object, spec: dict[str, object]) -> dict[str, object]:
        assert isinstance(conn, _DummyConn)
        assert spec["name"] == "Demo"
        return {"id": "spec123", **spec}

    monkeypatch.setattr("pipelines.workflows.upsert_workflow_spec", _fake_upsert)

    assert cli.main(["spec", "create", "--file", str(spec_path)]) == 0
    out = capsys.readouterr().out
    assert "spec123" in out


def test_run_show_reads_registry(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("pipelines.cli._open_conn", lambda: _DummyConn())
    monkeypatch.setattr(
        "pipelines.workflows.get_workflow_run",
        lambda conn, run_id: {"id": run_id, "run_type": "dataset", "status": "succeeded"},
    )

    assert cli.main(["run", "show", "--id", "run123"]) == 0
    out = capsys.readouterr().out
    assert "run123" in out


def test_report_build_dispatches(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class _Conn(_DummyConn):
        def execute(self, query: str, params: list[object] | None = None):
            assert "FROM workflow_runs" in query
            assert params == ["spec123"]

            class _Rows:
                @staticmethod
                def fetchall():
                    return [{"id": "analysis123"}]

            return _Rows()

    monkeypatch.setattr("pipelines.cli._open_conn", lambda: _Conn())
    monkeypatch.setattr(
        "pipelines.workflows.get_workflow_run",
        lambda conn, run_id: {"id": run_id, "run_type": "analysis"},
    )
    monkeypatch.setattr(
        "pipelines.reporting.build_workflow_report",
        lambda conn, analysis_run_id: {"report_path": f"/tmp/{analysis_run_id}/report.typ"},
    )

    assert cli.main(["report", "build", "--spec", "spec123"]) == 0
    out = capsys.readouterr().out
    assert "analysis123" in out


def test_capture_run_uses_modal_by_default_for_workflow_specs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("pipelines.cli._open_conn", lambda: _DummyConn())
    monkeypatch.setattr(
        "pipelines.cli._load_spec_from_args",
        lambda spec_id, file_path: {"id": "spec123", "capture": {}, "dataset": {"source": {"mode": "table", "table": "interp_examples_v0"}, "label": {"mode": "direct", "expression_sql": "decision_type"}}},
    )
    monkeypatch.setattr(
        "pipelines.workflows.get_latest_publication_for_spec",
        lambda conn, spec_id: {"relation_name": "workflow_dataset_spec123_v1"},
    )
    monkeypatch.setattr(
        "pipelines.workflows.start_workflow_run",
        lambda conn, spec, run_type, source, resolved_config: {"id": "run123"},
    )
    finished: list[dict[str, object]] = []
    monkeypatch.setattr(
        "pipelines.workflows.finish_workflow_run",
        lambda conn, run_id, status, result=None, error_text=None: finished.append(
            {"run_id": run_id, "status": status, "result": result, "error_text": error_text}
        ),
    )
    calls: list[list[str]] = []
    monkeypatch.setattr("pipelines.cli._run_command", lambda cmd: calls.append(list(cmd)))

    out_dir = tmp_path / "activations"
    assert cli.main(["capture", "run", "--spec", "spec123", "--output-dir", str(out_dir), "--layers", "16,24"]) == 0

    assert calls[0][:7] == ["uv", "run", "--extra", "interp", "--extra", "modal", "modal"]
    assert "pipelines/interp/modal_vllm_orchestrator.py" in calls[0]
    assert "--source-relation" in calls[0]
    assert "workflow_dataset_spec123_v1" in calls[0]
    assert finished[-1]["status"] == "succeeded"
    assert finished[-1]["result"]["execution"] == "modal"
    assert finished[-1]["result"]["remote_activations_subdir"] == "workflows/spec123/run123"
    assert "run123" in capsys.readouterr().out


def test_analysis_run_uses_modal_by_default_for_workflow_specs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("pipelines.cli._open_conn", lambda: _DummyConn())
    monkeypatch.setattr(
        "pipelines.workflows.get_workflow_run",
        lambda conn, run_id: {
            "id": run_id,
            "spec_snapshot_json": {"id": "spec123", "analysis": {}, "dataset": {"probe_defaults": {}}},
            "result_json": {
                "publication": "workflow_dataset_spec123_v1",
                "remote_activations_subdir": "workflows/spec123/run123",
            },
        },
    )
    monkeypatch.setattr(
        "pipelines.workflows.start_workflow_run",
        lambda conn, spec, run_type, source, resolved_config: {"id": "analysis123"},
    )
    monkeypatch.setattr(
        "pipelines.workflows.export_publication_labels",
        lambda conn, relation_name, output_path: output_path,
    )
    finished: list[dict[str, object]] = []
    monkeypatch.setattr(
        "pipelines.workflows.finish_workflow_run",
        lambda conn, run_id, status, result=None, error_text=None: finished.append(
            {"run_id": run_id, "status": status, "result": result, "error_text": error_text}
        ),
    )
    calls: list[list[str]] = []

    def _fake_run_command(cmd: list[str]) -> None:
        calls.append(list(cmd))
        if cmd[:4] == ["modal", "volume", "get", "xenon-data"]:
            local_dir = Path(cmd[5]) / "analysis123"
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "results.json").write_text('{"probe":[{"layer":34,"accuracy_mean":0.576}]}')

    monkeypatch.setattr("pipelines.cli._run_command", _fake_run_command)

    out_dir = tmp_path / "analysis"
    assert cli.main(["analysis", "run", "--capture-run", "capture123", "--output-dir", str(out_dir)]) == 0

    assert calls[0][:7] == ["uv", "run", "--extra", "analysis", "--extra", "modal", "modal"]
    assert "pipelines/interp/modal_analysis.py" in calls[0]
    assert "--relation-name" in calls[0]
    assert "workflow_dataset_spec123_v1" in calls[0]
    assert calls[1][:4] == ["modal", "volume", "get", "xenon-data"]
    assert calls[1][4] == "analysis_results/workflows/spec123/analysis123/"
    assert calls[1][5] == str(out_dir)
    assert out_dir.is_dir()
    assert finished[-1]["status"] == "succeeded"
    assert finished[-1]["result"]["execution"] == "modal"
    assert finished[-1]["result"]["output_dir"] == str(out_dir / "analysis123")
    assert finished[-1]["result"]["requested_output_dir"] == str(out_dir)
    assert finished[-1]["result"]["results"] == {"probe": [{"layer": 34, "accuracy_mean": 0.576}]}
    assert "analysis123" in capsys.readouterr().out


def test_modal_volume_get_creates_directory_for_remote_directory_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr("pipelines.cli._run_command", lambda cmd: calls.append(list(cmd)))

    local_dir = tmp_path / "research" / "prompt_confusion" / "outputs" / "analysis"
    cli._modal_volume_get("xenon-data", "analysis_results/workflows/spec123/run123/", local_dir)

    assert local_dir.is_dir()
    assert calls == [
        [
            "modal",
            "volume",
            "get",
            "xenon-data",
            "analysis_results/workflows/spec123/run123/",
            str(local_dir),
            "--force",
        ]
    ]


def test_modal_volume_get_creates_parent_for_remote_file_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr("pipelines.cli._run_command", lambda cmd: calls.append(list(cmd)))

    local_file = tmp_path / "research" / "prompt_confusion" / "outputs" / "probe.parquet"
    cli._modal_volume_get("xenon-data", "analysis_results/workflows/spec123/run123/probe.parquet", local_file)

    assert local_file.parent.is_dir()
    assert not local_file.exists()
    assert calls == [
        [
            "modal",
            "volume",
            "get",
            "xenon-data",
            "analysis_results/workflows/spec123/run123/probe.parquet",
            str(local_file),
            "--force",
        ]
    ]
