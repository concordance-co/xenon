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

    filtered = _limit_uncaptured_rows(rows, {2, 4}, limit=1)

    assert filtered == [{"log_id": 1}]


def test_limit_uncaptured_rows_returns_all_uncaptured_when_unbounded() -> None:
    rows = [{"log_id": 1}, {"log_id": 2}, {"log_id": 3}]

    filtered = _limit_uncaptured_rows(rows, {1}, limit=0)

    assert filtered == [{"log_id": 2}, {"log_id": 3}]
