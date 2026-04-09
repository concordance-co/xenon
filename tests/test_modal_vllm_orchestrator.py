import ast
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from pipelines.interp.modal_vllm_orchestrator import (
    _limit_uncaptured_rows,
    _run_neon_transaction,
    _residual_path_has_full_sequence_shape,
)


def test_residual_path_has_full_sequence_shape_for_3d_tensor(tmp_path: Path) -> None:
    path = tmp_path / "residual_3d.safetensors"
    save_file({"residual_stream": np.zeros((4, 8, 16), dtype=np.float16)}, str(path))

    assert _residual_path_has_full_sequence_shape(path) is True


def test_residual_path_has_full_sequence_shape_rejects_2d_tensor(tmp_path: Path) -> None:
    path = tmp_path / "residual_2d.safetensors"
    save_file({"residual_stream": np.zeros((8, 16), dtype=np.float16)}, str(path))

    assert _residual_path_has_full_sequence_shape(path) is False


def test_limit_uncaptured_rows_filters_then_applies_limit() -> None:
    rows = [{"log_id": 1}, {"log_id": 2}, {"log_id": 3}, {"log_id": 4}]

    filtered = _limit_uncaptured_rows(
        rows,
        {"log:2": {"log_id": 2}, "log:4": {"log_id": 4}},
        limit=1,
    )

    assert filtered == [{"log_id": 1}]


def test_limit_uncaptured_rows_returns_all_uncaptured_when_unbounded() -> None:
    rows = [{"log_id": 1}, {"log_id": 2}, {"log_id": 3}]

    filtered = _limit_uncaptured_rows(rows, {"log:1": {"log_id": 1}}, limit=0)

    assert filtered == [{"log_id": 2}, {"log_id": 3}]


def test_run_neon_transaction_retries_on_operational_error(monkeypatch) -> None:
    class FakeOperationalError(Exception):
        pass

    class FakeConnection:
        def __init__(self, *, should_fail: bool):
            self.should_fail = should_fail
            self.committed = False
            self.closed = False

        def commit(self) -> None:
            self.committed = True

        def close(self) -> None:
            self.closed = True

    conns: list[FakeConnection] = []

    def fake_connect_neon():
        conn = FakeConnection(should_fail=not conns)
        conns.append(conn)
        return conn

    def fake_transaction_fn(conn: FakeConnection) -> str:
        if conn.should_fail:
            raise FakeOperationalError("SSL connection has been closed unexpectedly")
        return "ok"

    monkeypatch.setattr("pipelines.db.connect_neon", fake_connect_neon)
    monkeypatch.setattr("psycopg.OperationalError", FakeOperationalError)

    result = _run_neon_transaction(fake_transaction_fn)

    assert result == "ok"
    assert len(conns) == 2
    assert conns[0].closed is True
    assert conns[0].committed is False
    assert conns[1].closed is True
    assert conns[1].committed is True


def test_capture_batch_signature_matches_map_kwargs() -> None:
    source_path = Path(__file__).resolve().parents[1] / "pipelines" / "interp" / "modal_vllm_orchestrator.py"
    module = ast.parse(source_path.read_text())

    capture_batch_args: set[str] = set()
    map_kwargs: set[str] = set()

    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "VLLMCaptureWorker":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "capture_batch":
                    capture_batch_args = {
                        arg.arg for arg in item.args.args if arg.arg not in {"self", "rows"}
                    }
        if isinstance(node, ast.FunctionDef) and node.name == "run_vllm_capture":
            for item in ast.walk(node):
                if not isinstance(item, ast.Call):
                    continue
                func = item.func
                if not (
                    isinstance(func, ast.Attribute)
                    and func.attr == "map"
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "capture_batch"
                ):
                    continue
                kwargs_arg = next((kw for kw in item.keywords if kw.arg == "kwargs"), None)
                if kwargs_arg is None:
                    continue
                kwargs_value = kwargs_arg.value
                if isinstance(kwargs_value, ast.Call) and isinstance(kwargs_value.func, ast.Name) and kwargs_value.func.id == "dict":
                    map_kwargs = {kw.arg for kw in kwargs_value.keywords if kw.arg}
                break

    assert map_kwargs
    assert map_kwargs <= capture_batch_args


def test_run_vllm_capture_decouples_capture_and_generation_prompt_formatting() -> None:
    source_path = Path(__file__).resolve().parents[1] / "pipelines" / "interp" / "modal_vllm_orchestrator.py"
    module = ast.parse(source_path.read_text())

    worker_call = None
    generation_worker_call = None

    for node in module.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "run_vllm_capture":
            continue
        for item in ast.walk(node):
            if not isinstance(item, ast.Assign) or len(item.targets) != 1:
                continue
            target = item.targets[0]
            if not isinstance(target, ast.Name) or not isinstance(item.value, ast.Call):
                continue
            if target.id == "worker":
                worker_call = item.value
            elif target.id == "generation_worker":
                generation_worker_call = item.value

    assert worker_call is not None
    assert generation_worker_call is not None

    worker_kwargs = {kw.arg: kw.value for kw in worker_call.keywords if kw.arg}
    generation_worker_kwargs = {kw.arg: kw.value for kw in generation_worker_call.keywords if kw.arg}

    assert isinstance(worker_kwargs["add_generation_prompt"], ast.Constant)
    assert worker_kwargs["add_generation_prompt"].value is False
    assert isinstance(generation_worker_kwargs["add_generation_prompt"], ast.Name)
    assert generation_worker_kwargs["add_generation_prompt"].id == "add_generation_prompt"


def test_run_vllm_capture_scopes_capture_metadata_by_run_id() -> None:
    source_path = Path(__file__).resolve().parents[1] / "pipelines" / "interp" / "modal_vllm_orchestrator.py"
    source = source_path.read_text()

    assert "ALTER TABLE capture_metadata ADD COLUMN IF NOT EXISTS run_id" in source
    assert "CREATE UNIQUE INDEX IF NOT EXISTS capture_metadata_run_log_id_idx ON capture_metadata (run_id, log_id)" in source
    assert "ON CONFLICT (run_id, log_id) DO UPDATE SET" in source
