from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pipelines_v2.engine.vllm.activation_patch_math import (  # noqa: E402
    ADD_DIRECTION_OPERATOR,
    PROJECT_OUT_OPERATOR,
    RANDOM_CONTROL_OPERATOR,
    RESIDUAL_PATH_OPERATOR,
    SWAP_COMPONENTS_OPERATOR,
    SWAP_MEAN_OPERATOR,
    apply_residual_path_transport,
    apply_subspace_operator,
    operator_mode_id,
)
from pipelines_v2.engine.vllm.patching.apply import apply_layer_output_patching  # noqa: E402
from pipelines_v2.engine.vllm.patching.custom_ops import (  # noqa: E402
    register_torch_library_interchange_batch_op,
    register_torch_library_residual_path_batch_op,
    register_torch_library_subspace_batch_op,
)
from pipelines_v2.engine.vllm.patching.state import (  # noqa: E402
    _ensure_batch_runtime_state_buffers,
    _ensure_batch_tensor_stats_buffers,
    collect_patch_stats,
    harvest_batch_patch_stats,
    register_activation_patch_bank,
    register_activation_patch_subspace,
    set_batch_patch_specs,
)


def _activation_patch_op(name: str) -> Any:
    namespace = getattr(torch.ops, "xenon_activation_patch_v2", None)
    op = getattr(namespace, name, None) if namespace is not None else None
    if op is None:
        pytest.skip(f"torch.library custom op {name!r} is not available")
    return op


def _hidden() -> Any:
    return torch.tensor(
        [
            [2.0, -1.0, 0.5, 4.0],
            [3.0, 0.0, 1.0, 2.0],
            [-1.0, 2.0, 3.0, -2.0],
            [0.5, -3.0, 2.0, 1.0],
            [4.0, 1.0, -1.0, 0.0],
            [-2.0, 0.5, 0.0, 3.0],
        ],
        dtype=torch.float32,
    )


def _subspace_inputs() -> dict[str, Any]:
    return {
        "mean": torch.tensor([1.0, -1.0, 0.5, 2.0], dtype=torch.float32),
        "scale": torch.tensor([2.0, 0.5, 1.5, 1.0], dtype=torch.float32),
        "safe_scale": torch.tensor([2.0, 0.5, 1.5, 1.0], dtype=torch.float32),
        "selected_rows": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "direction_raw": torch.tensor([0.25, -0.5, 0.75, 1.25], dtype=torch.float32),
        "direction_std": torch.tensor([0.5, -1.0, 0.25, 0.0], dtype=torch.float32),
        "donor_mean": torch.tensor([4.0, -3.0, 2.0, 0.0], dtype=torch.float32),
        "random_rows": torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    }


@pytest.mark.unit
@pytest.mark.vllm
@pytest.mark.interp
def test_register_activation_patch_subspace_materializes_runtime_unit_basis() -> None:
    layer = SimpleNamespace(
        input_layernorm=SimpleNamespace(weight=torch.empty((4,), dtype=torch.float32))
    )
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[layer]),
        _v2_activation_patch_initialized=True,
        _v2_activation_patch_device=torch.device("cpu"),
        _v2_activation_patch_subspace={},
    )

    summary = register_activation_patch_subspace(
        model,
        {
            0: {
                "mean": {"kind": "xenon_runtime_zeros"},
                "scale": {"kind": "xenon_runtime_ones"},
                "components": {"kind": "xenon_runtime_unit_basis", "indices": [0, 2]},
            }
        },
    )

    registered = model._v2_activation_patch_subspace[0]
    assert summary == {"layers": [0], "components": {"0": 2}}
    torch.testing.assert_close(registered["mean"], torch.zeros((4,), dtype=torch.float32))
    torch.testing.assert_close(registered["scale"], torch.ones((4,), dtype=torch.float32))
    torch.testing.assert_close(
        registered["components"],
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )


@pytest.mark.unit
@pytest.mark.vllm
@pytest.mark.interp
def test_compiled_runtime_buffers_are_capacity_based_and_not_replaced() -> None:
    model = SimpleNamespace(_v2_activation_patch_batch_runtime_state={})
    first = _ensure_batch_runtime_state_buffers(
        model,
        layer_idx=0,
        max_tokens=8,
        max_rows=4,
        hidden_dim=5,
        device=torch.device("cpu"),
    )
    first_query_positions = first["query_positions"]
    first_selected_rows = first["selected_rows"]

    second = _ensure_batch_runtime_state_buffers(
        model,
        layer_idx=0,
        max_tokens=3,
        max_rows=2,
        hidden_dim=5,
        device=torch.device("cpu"),
    )

    assert second is first
    assert second["query_positions"] is first_query_positions
    assert second["selected_rows"] is first_selected_rows
    assert second["query_positions"].shape[1] == 8
    assert second["selected_rows"].shape[1] == 4


@pytest.mark.unit
@pytest.mark.vllm
@pytest.mark.interp
def test_compiled_runtime_buffers_fail_loudly_when_capacity_is_exceeded() -> None:
    model = SimpleNamespace(
        _v2_activation_patch_batch_runtime_state={},
        _v2_activation_patch_batch_tensor_stats={},
    )
    _ensure_batch_runtime_state_buffers(
        model,
        layer_idx=0,
        max_tokens=2,
        max_rows=2,
        hidden_dim=4,
        device=torch.device("cpu"),
    )
    _ensure_batch_tensor_stats_buffers(
        model,
        layer_idx=0,
        coeff_dim=2,
        device=torch.device("cpu"),
    )
    model._v2_activation_patch_force_custom_op_presence = True

    with pytest.raises(RuntimeError, match="XENON_ACTIVATION_PATCH_MAX_TOKENS"):
        _ensure_batch_runtime_state_buffers(
            model,
            layer_idx=0,
            max_tokens=3,
            max_rows=2,
            hidden_dim=4,
            device=torch.device("cpu"),
        )
    with pytest.raises(RuntimeError, match="component capacity"):
        _ensure_batch_tensor_stats_buffers(
            model,
            layer_idx=0,
            coeff_dim=3,
            device=torch.device("cpu"),
        )


@pytest.mark.unit
@pytest.mark.vllm
@pytest.mark.interp
@pytest.mark.parametrize(
    ("operator", "row_count", "match_projected_norm"),
    [
        (PROJECT_OUT_OPERATOR, 2, True),
        (ADD_DIRECTION_OPERATOR, 0, True),
        (SWAP_MEAN_OPERATOR, 0, True),
        (SWAP_COMPONENTS_OPERATOR, 2, True),
        (RANDOM_CONTROL_OPERATOR, 2, True),
        (RANDOM_CONTROL_OPERATOR, 2, False),
    ],
)
def test_subspace_batch_custom_op_matches_fallback_math_for_each_operator(
    operator: str,
    row_count: int,
    match_projected_norm: bool,
) -> None:
    register_torch_library_subspace_batch_op()
    subspace_batch = _activation_patch_op("subspace_batch")
    hidden = _hidden()
    inputs = _subspace_inputs()
    selected_rows = inputs["selected_rows"][:row_count]
    span = (1, 4)
    strength = 0.75

    slot_count = 2
    max_rows = 2
    hidden_dim = int(hidden.shape[-1])
    stats_scalars = torch.full((slot_count, 8), -999.0, dtype=torch.float32)
    stats_coeff_before = torch.full((slot_count, max_rows), -999.0, dtype=torch.float32)
    stats_coeff_after = torch.full((slot_count, max_rows), -999.0, dtype=torch.float32)
    stats_valid = torch.zeros((slot_count,), dtype=torch.int32)
    batch_selected_rows = torch.zeros((slot_count, max_rows, hidden_dim), dtype=torch.float32)
    batch_random_rows = torch.zeros((slot_count, max_rows, hidden_dim), dtype=torch.float32)
    if row_count:
        batch_selected_rows[0, :row_count] = selected_rows
        batch_random_rows[0, :row_count] = inputs["random_rows"][:row_count]

    patched = subspace_batch(
        hidden,
        inputs["mean"],
        inputs["scale"],
        inputs["safe_scale"],
        torch.tensor([operator_mode_id(operator), operator_mode_id(PROJECT_OUT_OPERATOR)], dtype=torch.int32),
        batch_selected_rows,
        torch.tensor([row_count, max_rows], dtype=torch.int32),
        torch.tensor([[span[0], span[1]], [0, hidden.shape[0]]], dtype=torch.int32),
        torch.tensor([[span[0], span[1], span[1] - 1], [0, 1, 2]], dtype=torch.int32),
        torch.tensor([span[1] - span[0], 0], dtype=torch.int32),
        torch.tensor([strength, 1.0], dtype=torch.float32),
        torch.tensor([1, 0], dtype=torch.int32),
        stats_valid,
        stats_scalars,
        stats_coeff_before,
        stats_coeff_after,
        torch.stack((inputs["direction_raw"], torch.zeros_like(inputs["direction_raw"]))),
        torch.stack((inputs["direction_std"], torch.zeros_like(inputs["direction_std"]))),
        torch.stack((inputs["donor_mean"], torch.zeros_like(inputs["donor_mean"]))),
        batch_random_rows,
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([int(match_projected_norm), 1], dtype=torch.int32),
    )

    expected, expected_stats = apply_subspace_operator(
        hidden,
        query_positions=list(range(span[0], span[1])),
        operator=operator,
        mean=inputs["mean"],
        scale=inputs["scale"],
        safe_scale=inputs["safe_scale"],
        selected_rows=selected_rows,
        strength=strength,
        direction_raw=inputs["direction_raw"],
        direction_std=inputs["direction_std"],
        donor_mean=inputs["donor_mean"],
        random_rows=inputs["random_rows"][:row_count],
        match_projected_norm=match_projected_norm,
    )

    torch.testing.assert_close(patched, expected, rtol=1e-5, atol=1e-5)
    assert stats_valid.tolist() == [1, 0]
    assert stats_scalars[1].tolist() == [-999.0] * 8
    assert stats_coeff_before[1].tolist() == [-999.0] * max_rows
    assert stats_coeff_after[1].tolist() == [-999.0] * max_rows

    scalar_names = (
        "delta_norm_raw",
        "delta_norm_std",
        "mean_norm_before",
        "mean_norm_after",
        "mean_std_norm_before",
        "mean_std_norm_after",
    )
    for index, name in enumerate(scalar_names):
        assert stats_scalars[0, index].item() == pytest.approx(expected_stats[name], rel=1e-5, abs=1e-5)
    if operator == ADD_DIRECTION_OPERATOR:
        assert stats_scalars[0, 6].item() == pytest.approx(expected_stats["direction_norm_raw"], rel=1e-5, abs=1e-5)
    else:
        assert stats_scalars[0, 6].item() == pytest.approx(expected_stats["selected_proj_norm_before"], rel=1e-5, abs=1e-5)
    assert stats_scalars[0, 7].item() == pytest.approx(strength)
    if row_count:
        torch.testing.assert_close(
            stats_coeff_before[0, :row_count],
            torch.tensor(expected_stats["selected_coeff_before"], dtype=torch.float32),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            stats_coeff_after[0, :row_count],
            torch.tensor(expected_stats["selected_coeff_after"], dtype=torch.float32),
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.unit
@pytest.mark.vllm
@pytest.mark.interp
def test_interchange_batch_custom_op_matches_direct_row_replacement_oracle() -> None:
    register_torch_library_interchange_batch_op()
    interchange_batch = _activation_patch_op("interchange_batch")
    hidden = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
        dtype=torch.float32,
    )
    donor_rows = torch.tensor(
        [
            [[10.0, 10.0, 10.0], [30.0, 30.0, 30.0]],
            [[99.0, 99.0, 99.0], [99.0, 99.0, 99.0]],
            [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    stats_valid = torch.zeros((3,), dtype=torch.int32)
    stats_scalars = torch.full((3, 2), -7.0, dtype=torch.float32)

    patched = interchange_batch(
        hidden,
        torch.tensor([[1, 3], [0, 0], [99, 0]], dtype=torch.int32),
        donor_rows,
        torch.tensor([2, 1, 1], dtype=torch.int32),
        torch.tensor([1, 0, 1], dtype=torch.int32),
        stats_valid,
        stats_scalars,
    )

    expected = hidden.clone()
    expected[[1, 3]] = donor_rows[0, :2]
    torch.testing.assert_close(patched, expected)
    assert stats_valid.tolist() == [1, 0, 0]
    assert stats_scalars[0, 0].item() == pytest.approx(torch.linalg.norm(donor_rows[0, :2] - hidden[[1, 3]]).item())
    assert stats_scalars[0, 1].item() == pytest.approx(2.0)
    assert stats_scalars[1].tolist() == [-7.0, -7.0]
    assert stats_scalars[2].tolist() == [-7.0, -7.0]


@pytest.mark.unit
@pytest.mark.vllm
@pytest.mark.interp
def test_residual_path_batch_custom_op_matches_replace_and_delta_oracles() -> None:
    register_torch_library_residual_path_batch_op()
    residual_path_batch = _activation_patch_op("residual_path_batch")
    hidden = torch.tensor(
        [[1.0, 0.0], [2.0, -1.0], [3.0, 1.0], [4.0, 2.0]],
        dtype=torch.float32,
    )
    replace_donor = torch.tensor([[5.0, 2.0], [8.0, -3.0]], dtype=torch.float32)
    delta_rows = torch.tensor([[1.5, -2.5], [0.0, 0.0]], dtype=torch.float32)
    payload_rows = torch.stack((0.5 * replace_donor, delta_rows, torch.zeros_like(delta_rows)))
    stats_valid = torch.zeros((3,), dtype=torch.int32)
    stats_scalars = torch.full((3, 3), -11.0, dtype=torch.float32)

    patched = residual_path_batch(
        hidden,
        torch.tensor([[0, 1], [3, 0], [99, 0]], dtype=torch.int32),
        payload_rows,
        torch.tensor([2, 1, 1], dtype=torch.int32),
        torch.tensor([1, 0, 1], dtype=torch.int32),
        torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([1, 1, 1], dtype=torch.int32),
        stats_valid,
        stats_scalars,
    )

    expected, replace_stats = apply_residual_path_transport(
        hidden,
        query_positions=[0, 1],
        donor_rows=replace_donor,
        target_source_rows=None,
        weight=0.5,
        strength=1.0,
        transport="replace",
    )
    expected, delta_stats = apply_residual_path_transport(
        expected,
        query_positions=[3],
        donor_rows=torch.tensor([[5.5, -0.5]], dtype=torch.float32),
        target_source_rows=torch.tensor([[4.0, 2.0]], dtype=torch.float32),
        weight=1.0,
        strength=1.0,
        transport="delta",
    )

    torch.testing.assert_close(patched, expected, rtol=1e-5, atol=1e-5)
    assert stats_valid.tolist() == [1, 1, 0]
    assert stats_scalars[0, 0].item() == pytest.approx(replace_stats["delta_norm_raw"])
    assert stats_scalars[0, 1].item() == pytest.approx(2.0)
    assert stats_scalars[0, 2].item() == pytest.approx(0.5)
    assert stats_scalars[1, 0].item() == pytest.approx(delta_stats["delta_norm_raw"])
    assert stats_scalars[1, 1].item() == pytest.approx(1.0)
    assert stats_scalars[2].tolist() == [-11.0, -11.0, -11.0]


class _TorchPatchCustomOp:
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("batch_token_spans") is not None:
            return torch.ops.xenon_activation_patch_v2.subspace_batch(
                args[0],
                args[1],
                args[2],
                args[3],
                kwargs["batch_mode_ids"],
                kwargs["batch_selected_rows"],
                kwargs["batch_row_counts"],
                kwargs["batch_token_spans"],
                kwargs["batch_query_positions"],
                kwargs["batch_token_counts"],
                kwargs["batch_strengths"],
                kwargs["batch_active"],
                kwargs["stats_valid"],
                kwargs["stats_scalars"],
                kwargs["stats_coeff_before"],
                kwargs["stats_coeff_after"],
                kwargs["batch_direction_raw"],
                kwargs["batch_direction_std"],
                kwargs["batch_donor_means"],
                kwargs["batch_random_rows"],
                kwargs["batch_rowwise"],
                kwargs["batch_match_projected_norm"],
            )
        raise AssertionError("unexpected non-subspace custom-op invocation")


@pytest.mark.unit
@pytest.mark.vllm
@pytest.mark.interp
def test_compiled_subspace_path_harvests_stats_matching_fallback_oracle() -> None:
    register_torch_library_subspace_batch_op()
    _activation_patch_op("subspace_batch")
    hidden = _hidden()
    inputs = _subspace_inputs()
    components = inputs["selected_rows"]
    model = SimpleNamespace(
        _v2_activation_patch_initialized=True,
        _v2_activation_patch_device=torch.device("cpu"),
        _v2_activation_patch_bank={},
        _v2_activation_patch_subspace={
            0: {
                "mean": inputs["mean"],
                "scale": inputs["scale"],
                "safe_scale": inputs["safe_scale"],
                "components": components,
                "named_components": {},
            }
        },
        _v2_activation_patch_directions={},
        _v2_activation_patch_centroids={},
        _v2_activation_patch_batch_runtime_state={},
        _v2_activation_patch_batch_tensor_stats={},
        _v2_activation_patch_stats_by_req={},
        _v2_activation_patch_force_custom_op_presence=True,
        _v2_activation_patch_compiled_operator_hint="subspace",
    )
    batch_specs = [
        {
            "req_id": "req-project-out",
            "patch_spec": {
                "operator": PROJECT_OUT_OPERATOR,
                "target_layers": [0],
                "query_positions": [1, 2, 3],
                "source_layer_map": {"0": 0},
                "component_indices_by_layer": {"0": [0, 1]},
                "strength": 0.5,
                "target_abs_positions": [10, 11, 12],
                "covered_abs_positions": [10, 11, 12],
            },
        }
    ]
    set_batch_patch_specs(model, batch_specs)

    patched = apply_layer_output_patching(
        owner_model=model,
        layer_idx=0,
        custom_op=_TorchPatchCustomOp(),
        output=hidden,
    )
    expected, expected_stats = apply_subspace_operator(
        hidden,
        query_positions=[1, 2, 3],
        operator=PROJECT_OUT_OPERATOR,
        mean=inputs["mean"],
        scale=inputs["scale"],
        safe_scale=inputs["safe_scale"],
        selected_rows=components,
        strength=0.5,
    )
    torch.testing.assert_close(patched, expected, rtol=1e-5, atol=1e-5)

    harvest_batch_patch_stats(model, batch_specs)
    stats = collect_patch_stats(model, req_id="req-project-out")

    assert set(stats) == {0}
    layer_stats = stats[0]
    assert layer_stats["operator"] == PROJECT_OUT_OPERATOR
    assert layer_stats["dispatch"] == "compiled_custom_op"
    assert layer_stats["token_count"] == 3
    assert layer_stats["covered_abs_spans"] == [[10, 13]]
    assert layer_stats["coverage_fraction"] == 1.0
    assert layer_stats["delta_norm_raw"] == pytest.approx(expected_stats["delta_norm_raw"], rel=1e-5, abs=1e-5)
    assert layer_stats["selected_coeff_before"] == pytest.approx(expected_stats["selected_coeff_before"], rel=1e-5, abs=1e-5)
    assert layer_stats["selected_coeff_after"] == pytest.approx(expected_stats["selected_coeff_after"], rel=1e-5, abs=1e-5)
    assert layer_stats["strength"] == pytest.approx(0.5)
    assert layer_stats["operator"] != RESIDUAL_PATH_OPERATOR


@pytest.mark.unit
@pytest.mark.vllm
@pytest.mark.interp
def test_subspace_compiled_hint_does_not_route_paired_batches_through_subspace_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden = _hidden()
    inputs = _subspace_inputs()
    model = SimpleNamespace(
        _v2_activation_patch_initialized=True,
        _v2_activation_patch_device=torch.device("cpu"),
        _v2_activation_patch_bank={},
        _v2_activation_patch_subspace={},
        _v2_activation_patch_directions={},
        _v2_activation_patch_centroids={},
        _v2_activation_patch_batch_runtime_state={},
        _v2_activation_patch_batch_tensor_stats={},
        _v2_activation_patch_stats_by_req={},
        _v2_activation_patch_force_custom_op_presence=True,
        _v2_activation_patch_compiled_operator_hint="subspace",
    )
    register_activation_patch_subspace(
        model,
        {
            0: {
                "mean": inputs["mean"],
                "scale": inputs["scale"],
                "safe_scale": inputs["safe_scale"],
                "components": inputs["selected_rows"],
                "named_components": {},
            }
        },
    )
    register_activation_patch_bank(
        model,
        {
            0: {
                "donor": {
                    "values": hidden.tolist(),
                    "token_count": int(hidden.shape[0]),
                }
            }
        },
    )
    batch_specs = [
        {
            "req_id": "req-interchange",
            "patch_spec": {
                "operator": "interchange",
                "target_layers": [0],
                "query_positions": [1, 2],
                "donor_example_key": "donor",
                "donor_positions": [0, 1],
            },
        }
    ]
    set_batch_patch_specs(model, batch_specs)

    operator_ids: list[int] = []

    def fake_run_custom_op(*args: Any, **kwargs: Any) -> Any:
        del args
        operator_ids.append(int(kwargs["operator_id"]))
        return hidden

    monkeypatch.setattr(
        "pipelines_v2.engine.vllm.patching.apply.run_custom_op",
        fake_run_custom_op,
    )

    patched = apply_layer_output_patching(
        owner_model=model,
        layer_idx=0,
        custom_op=object(),
        output=hidden,
    )

    assert patched is hidden
    assert operator_ids == [operator_mode_id("interchange")]


@pytest.mark.unit
@pytest.mark.vllm
@pytest.mark.interp
def test_compiled_subspace_path_supports_query_span_specs() -> None:
    register_torch_library_subspace_batch_op()
    _activation_patch_op("subspace_batch")
    hidden = _hidden()
    inputs = _subspace_inputs()
    components = inputs["selected_rows"][:1]
    model = SimpleNamespace(
        _v2_activation_patch_initialized=True,
        _v2_activation_patch_device=torch.device("cpu"),
        _v2_activation_patch_bank={},
        _v2_activation_patch_subspace={
            0: {
                "mean": inputs["mean"],
                "scale": inputs["scale"],
                "safe_scale": inputs["safe_scale"],
                "components": components,
                "named_components": {},
            }
        },
        _v2_activation_patch_directions={},
        _v2_activation_patch_centroids={},
        _v2_activation_patch_batch_runtime_state={},
        _v2_activation_patch_batch_tensor_stats={},
        _v2_activation_patch_stats_by_req={},
        _v2_activation_patch_force_custom_op_presence=True,
        _v2_activation_patch_compiled_operator_hint="subspace",
    )
    batch_specs = [
        {
            "req_id": "req-query-span",
            "patch_spec": {
                "operator": PROJECT_OUT_OPERATOR,
                "target_layers": [0],
                "query_span": [1, 4],
                "covered_abs_spans": [[1, 4]],
                "source_layer_map": {"0": 0},
                "component_indices_by_layer": {"0": [0]},
                "strength": 1.0,
                "target_policy": {"kind": "every_token"},
                "rowwise": True,
            },
        }
    ]
    set_batch_patch_specs(model, batch_specs)

    patched = apply_layer_output_patching(
        owner_model=model,
        layer_idx=0,
        custom_op=_TorchPatchCustomOp(),
        output=hidden,
    )
    expected, _ = apply_subspace_operator(
        hidden,
        query_positions=[1, 2, 3],
        operator=PROJECT_OUT_OPERATOR,
        mean=inputs["mean"],
        scale=inputs["scale"],
        safe_scale=inputs["safe_scale"],
        selected_rows=components,
        strength=1.0,
        rowwise=True,
    )
    torch.testing.assert_close(patched, expected, rtol=1e-5, atol=1e-5)

    harvest_batch_patch_stats(model, batch_specs)
    stats = collect_patch_stats(model, req_id="req-query-span")

    assert stats[0]["status"] == "ok"
    assert stats[0]["dispatch"] == "compiled_custom_op"
    assert stats[0]["token_count"] == 3
    assert stats[0]["query_span"] == [1, 4]
    assert stats[0]["covered_abs_spans"] == [[1, 4]]
    assert stats[0]["target_policy"] == {"kind": "every_token"}
    assert stats[0]["rowwise"] is True
