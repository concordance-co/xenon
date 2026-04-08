import ast
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from pipelines.interp.modal_vllm_orchestrator import (
    _limit_uncaptured_rows,
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
