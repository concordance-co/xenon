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

    monkeypatch.setattr("pipelines.interp.prepare.main", _fake_main)

    assert cli.main(["dataset", "build", "--", "--limit", "5"]) == 7
    assert seen == [["--limit", "5"]]


def test_capture_run_passthrough_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def _fake_main(argv: list[str] | None = None) -> int:
        seen.append(list(argv or []))
        return 3

    monkeypatch.setattr("pipelines.interp.capture.main", _fake_main)

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
