from __future__ import annotations

import numpy as np
import torch

from pipelines.interp.patching.market_patch import (
    PATCH_MODE_ADD_DIRECTION,
    PATCH_MODE_PROJECT_OUT,
    PATCH_MODE_RANDOM_CONTROL,
    PATCH_MODE_SWAP_COMPONENTS,
    PATCH_MODE_SWAP_MEAN,
    MarketPatchSpec,
    clear_patch_spec,
    collect_patch_stats,
    init_market_patching,
    register_patch_basis,
    restore_original_forwards,
    set_batch_patch_specs,
    set_patch_spec,
)


def _basis_payload(dim: int = 4) -> dict[int, dict[str, object]]:
    return {
        0: {
            "mean": np.zeros((dim,), dtype=np.float32),
            "scale": np.ones((dim,), dtype=np.float32),
            "components": np.eye(2, dim, dtype=np.float32),
            "named_components": {"leader_axis": 0, "dispersion_axis": 1},
        }
    }


class _FakeSubmodule:
    def __init__(self):
        self.weight = torch.zeros((1,), dtype=torch.float32)


class _FakeLayer(torch.nn.Module):
    def __init__(self, offset: float = 0.0):
        super().__init__()
        self.offset = offset
        self.input_layernorm = _FakeSubmodule()

    def forward(self, hidden_states):
        return hidden_states + self.offset


class _FakeModel:
    def __init__(self):
        self.model = type(
            "Inner",
            (torch.nn.Module,),
            {
                "__init__": lambda inner: torch.nn.Module.__init__(inner)
                or setattr(inner, "layers", torch.nn.ModuleList([_FakeLayer(0.0), _FakeLayer(3.0)]))
            },
        )()


def test_project_out_removes_selected_mean_components():
    model = _FakeModel()
    init_market_patching(model)
    register_patch_basis(model, _basis_payload())

    set_patch_spec(
        model,
        MarketPatchSpec(
            mode=PATCH_MODE_PROJECT_OUT,
            target_layers=(0,),
            token_span=(0, 2),
            component_indices_by_layer={0: (0, 1)},
        ),
    )
    hidden = torch.tensor(
        [
            [2.0, 4.0, 7.0, 9.0],
            [2.0, 4.0, 5.0, 11.0],
            [10.0, 20.0, 30.0, 40.0],
        ],
        dtype=torch.float32,
    )

    patched = model.model.layers[0].forward(hidden)
    market_mean = patched[:2].mean(dim=0)
    stats = collect_patch_stats(model)[0]

    assert torch.allclose(market_mean[:2], torch.zeros((2,), dtype=torch.float32), atol=1e-5)
    assert torch.allclose(market_mean[2:], torch.tensor([6.0, 10.0]), atol=1e-5)
    assert stats["selected_proj_norm_before"] > 0.0
    restore_original_forwards(model)


def test_project_out_strength_scales_delta():
    model = _FakeModel()
    init_market_patching(model)
    register_patch_basis(model, _basis_payload())

    set_patch_spec(
        model,
        MarketPatchSpec(
            mode=PATCH_MODE_PROJECT_OUT,
            target_layers=(0,),
            token_span=(0, 2),
            component_indices_by_layer={0: (0,)},
            strength=0.5,
        ),
    )
    hidden = torch.tensor(
        [
            [4.0, 1.0, 0.0, 0.0],
            [4.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    patched = model.model.layers[0].forward(hidden)
    market_mean = patched[:2].mean(dim=0)

    assert torch.allclose(market_mean, torch.tensor([2.0, 1.0, 0.0, 0.0]), atol=1e-5)
    restore_original_forwards(model)


def test_add_direction_moves_market_mean_in_named_direction():
    model = _FakeModel()
    init_market_patching(model)
    register_patch_basis(model, _basis_payload())

    set_patch_spec(
        model,
        MarketPatchSpec(
            mode=PATCH_MODE_ADD_DIRECTION,
            target_layers=(0,),
            token_span=(0, 2),
            strength=2.5,
            direction_weights_by_layer={0: np.asarray([1.0, 0.0], dtype=np.float32)},
        ),
    )
    hidden = torch.zeros((3, 4), dtype=torch.float32)

    patched = model.model.layers[0].forward(hidden)
    market_mean = patched[:2].mean(dim=0)

    assert torch.allclose(market_mean, torch.tensor([2.5, 0.0, 0.0, 0.0]), atol=1e-5)
    restore_original_forwards(model)


def test_swap_mean_sets_market_mean_to_donor_mean():
    model = _FakeModel()
    init_market_patching(model)
    register_patch_basis(model, _basis_payload())

    set_patch_spec(
        model,
        MarketPatchSpec(
            mode=PATCH_MODE_SWAP_MEAN,
            target_layers=(0,),
            token_span=(0, 2),
            donor_mean_by_layer={0: np.asarray([9.0, 8.0, 7.0, 6.0], dtype=np.float32)},
        ),
    )
    hidden = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 5.0, 5.0, 5.0],
        ],
        dtype=torch.float32,
    )

    patched = model.model.layers[0].forward(hidden)
    assert torch.allclose(
        patched[:2].mean(dim=0),
        torch.tensor([9.0, 8.0, 7.0, 6.0]),
        atol=1e-5,
    )
    restore_original_forwards(model)


def test_swap_mean_strength_scales_toward_donor_mean():
    model = _FakeModel()
    init_market_patching(model)
    register_patch_basis(model, _basis_payload())

    set_patch_spec(
        model,
        MarketPatchSpec(
            mode=PATCH_MODE_SWAP_MEAN,
            target_layers=(0,),
            token_span=(0, 2),
            strength=0.5,
            donor_mean_by_layer={0: np.asarray([9.0, 8.0, 7.0, 6.0], dtype=np.float32)},
        ),
    )
    hidden = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=torch.float32,
    )

    patched = model.model.layers[0].forward(hidden)
    assert torch.allclose(
        patched[:2].mean(dim=0),
        torch.tensor([5.0, 5.0, 5.0, 5.0]),
        atol=1e-5,
    )
    restore_original_forwards(model)


def test_swap_components_only_replaces_selected_subspace_coefficients():
    model = _FakeModel()
    init_market_patching(model)
    register_patch_basis(model, _basis_payload())

    set_patch_spec(
        model,
        MarketPatchSpec(
            mode=PATCH_MODE_SWAP_COMPONENTS,
            target_layers=(0,),
            token_span=(0, 2),
            component_indices_by_layer={0: (0,)},
            donor_mean_by_layer={0: np.asarray([10.0, 99.0, 0.0, 0.0], dtype=np.float32)},
        ),
    )
    hidden = torch.tensor(
        [
            [2.0, 4.0, 3.0, 5.0],
            [2.0, 4.0, 3.0, 5.0],
        ],
        dtype=torch.float32,
    )

    patched = model.model.layers[0].forward(hidden)
    market_mean = patched[:2].mean(dim=0)

    assert torch.allclose(market_mean, torch.tensor([10.0, 4.0, 3.0, 5.0]), atol=1e-5)
    restore_original_forwards(model)


def test_random_control_changes_hidden_state_but_stays_orthogonal_to_target_space():
    model = _FakeModel()
    init_market_patching(model)
    register_patch_basis(model, _basis_payload())

    set_patch_spec(
        model,
        MarketPatchSpec(
            mode=PATCH_MODE_RANDOM_CONTROL,
            target_layers=(0,),
            token_span=(0, 2),
            component_indices_by_layer={0: (0, 1)},
            random_seed=123,
        ),
    )
    hidden = torch.tensor(
        [
            [4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0],
        ],
        dtype=torch.float32,
    )

    patched = model.model.layers[0].forward(hidden)
    stats = collect_patch_stats(model)[0]

    assert not torch.allclose(patched, hidden)
    coeff_after = torch.tensor(stats["selected_coeff_after"], dtype=torch.float32)
    assert coeff_after.numel() == 2
    restore_original_forwards(model)


def test_only_targeted_layer_is_patched_and_clear_resets_state():
    model = _FakeModel()
    init_market_patching(model)
    register_patch_basis(model, _basis_payload())

    set_patch_spec(
        model,
        MarketPatchSpec(
            mode=PATCH_MODE_ADD_DIRECTION,
            target_layers=(0,),
            token_span=(0, 2),
            strength=1.0,
            direction_weights_by_layer={0: np.asarray([1.0, 0.0], dtype=np.float32)},
        ),
    )
    hidden = torch.zeros((2, 4), dtype=torch.float32)

    patched0 = model.model.layers[0].forward(hidden)
    patched1 = model.model.layers[1].forward(hidden)

    assert torch.allclose(patched0[:2].mean(dim=0), torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(patched1, hidden + 3.0)

    stats_before_clear = collect_patch_stats(model)
    assert 0 in stats_before_clear
    clear_patch_spec(model)
    assert collect_patch_stats(model) == stats_before_clear
    restored = model.model.layers[0].forward(hidden)
    assert torch.allclose(restored, hidden)
    restore_original_forwards(model)


def test_generation_followup_calls_do_not_overwrite_successful_prefill_stats():
    model = _FakeModel()
    init_market_patching(model)
    register_patch_basis(model, _basis_payload())

    set_patch_spec(
        model,
        MarketPatchSpec(
            mode=PATCH_MODE_PROJECT_OUT,
            target_layers=(0,),
            token_span=(0, 2),
            component_indices_by_layer={0: (0, 1)},
        ),
    )

    prefill_hidden = torch.tensor(
        [
            [2.0, 4.0, 7.0, 9.0],
            [2.0, 4.0, 5.0, 11.0],
            [10.0, 20.0, 30.0, 40.0],
        ],
        dtype=torch.float32,
    )
    decode_hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)

    _ = model.model.layers[0].forward(prefill_hidden)
    stats_after_prefill = collect_patch_stats(model)[0].copy()
    decode_out = model.model.layers[0].forward(decode_hidden)
    stats_after_decode = collect_patch_stats(model)[0]

    assert "status" not in stats_after_prefill
    assert stats_after_decode == stats_after_prefill
    assert torch.allclose(decode_out, decode_hidden)
    restore_original_forwards(model)


def test_batch_patch_specs_apply_per_request_and_stats_can_be_collected_by_base_id():
    model = _FakeModel()
    init_market_patching(model)
    register_patch_basis(model, _basis_payload())

    req1 = MarketPatchSpec(
        mode=PATCH_MODE_PROJECT_OUT,
        target_layers=(0,),
        token_span=(0, 2),
        component_indices_by_layer={0: (0, 1)},
    ).to_payload()
    req2 = MarketPatchSpec(
        mode=PATCH_MODE_ADD_DIRECTION,
        target_layers=(0,),
        token_span=(2, 4),
        component_indices_by_layer={0: (0,)},
        direction_weights_by_layer={0: np.asarray([1.0, 0.0], dtype=np.float32)},
        strength=1.0,
    ).to_payload()

    set_batch_patch_specs(
        model,
        [
            {
                "req_id": "base-abc",
                "patch_spec": req1,
                "target_span": [0, 2],
                "chunk_abs_span": [0, 2],
                "overlap_abs_span": [0, 2],
                "query_span": [0, 2],
                "prefill_chunk_len": 2,
            },
            {
                "req_id": "other-xyz",
                "patch_spec": req2,
                "target_span": [0, 2],
                "chunk_abs_span": [0, 2],
                "overlap_abs_span": [0, 2],
                "query_span": [2, 4],
                "prefill_chunk_len": 2,
            },
        ],
    )

    hidden = torch.tensor(
        [
            [2.0, 4.0, 7.0, 9.0],
            [2.0, 4.0, 5.0, 11.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    patched = model.model.layers[0].forward(hidden)

    assert torch.allclose(patched[:2].mean(dim=0)[:2], torch.zeros((2,), dtype=torch.float32), atol=1e-5)
    assert torch.allclose(patched[2:4].mean(dim=0), torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-5)

    stats_base = collect_patch_stats(model, "base")[0]
    stats_other = collect_patch_stats(model, "other-xyz")[0]
    assert stats_base["req_id"] == "base-abc"
    assert stats_base["query_span"] == [0, 2]
    assert stats_base["coverage_fraction"] == 1.0
    assert stats_other["req_id"] == "other-xyz"
    assert stats_other["query_span"] == [2, 4]
    assert stats_other["coverage_fraction"] == 1.0
    restore_original_forwards(model)


def test_batch_patch_stats_survive_followup_decode_steps():
    model = _FakeModel()
    init_market_patching(model)
    register_patch_basis(model, _basis_payload())

    patch_payload = MarketPatchSpec(
        mode=PATCH_MODE_PROJECT_OUT,
        target_layers=(0,),
        token_span=(0, 2),
        component_indices_by_layer={0: (0, 1)},
    ).to_payload()

    set_batch_patch_specs(
        model,
        [
            {
                "req_id": "base-abc",
                "patch_spec": patch_payload,
                "target_span": [0, 2],
                "chunk_abs_span": [0, 3],
                "overlap_abs_span": [0, 2],
                "query_span": [0, 3],
                "prefill_chunk_len": 3,
            }
        ],
    )
    prefill_hidden = torch.tensor(
        [
            [2.0, 4.0, 7.0, 9.0],
            [2.0, 4.0, 5.0, 11.0],
            [10.0, 20.0, 30.0, 40.0],
        ],
        dtype=torch.float32,
    )
    _ = model.model.layers[0].forward(prefill_hidden)
    stats_after_prefill = collect_patch_stats(model, "base")[0].copy()

    set_batch_patch_specs(model, [])
    decode_hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    _ = model.model.layers[0].forward(decode_hidden)
    stats_after_decode = collect_patch_stats(model, "base")[0]

    assert stats_after_prefill == stats_after_decode
    restore_original_forwards(model)


def test_batch_patch_stats_accumulate_chunk_coverage_across_multiple_steps():
    model = _FakeModel()
    init_market_patching(model)
    register_patch_basis(model, _basis_payload())

    patch_payload = MarketPatchSpec(
        mode=PATCH_MODE_PROJECT_OUT,
        target_layers=(0,),
        token_span=(0, 1),
        component_indices_by_layer={0: (0,)},
    ).to_payload()

    set_batch_patch_specs(
        model,
        [
            {
                "req_id": "base-abc",
                "patch_spec": patch_payload,
                "target_span": [0, 2],
                "chunk_abs_span": [0, 1],
                "overlap_abs_span": [0, 1],
                "query_span": [0, 1],
                "prefill_chunk_len": 1,
            }
        ],
    )
    _ = model.model.layers[0].forward(torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float32))

    set_batch_patch_specs(
        model,
        [
            {
                "req_id": "base-abc",
                "patch_spec": patch_payload,
                "target_span": [0, 2],
                "chunk_abs_span": [1, 2],
                "overlap_abs_span": [1, 2],
                "query_span": [0, 1],
                "prefill_chunk_len": 1,
            }
        ],
    )
    _ = model.model.layers[0].forward(torch.tensor([[3.0, 0.0, 0.0, 0.0]], dtype=torch.float32))

    stats = collect_patch_stats(model, "base")[0]
    assert stats["covered_abs_spans"] == [[0, 2]]
    assert stats["covered_abs_tokens"] == 2
    assert stats["target_abs_tokens"] == 2
    assert stats["coverage_fraction"] == 1.0
    restore_original_forwards(model)


def test_swap_components_batch_op_replaces_selected_coefficients():
    hidden = torch.tensor(
        [
            [2.0, 4.0, 3.0, 5.0],
            [2.0, 4.0, 3.0, 5.0],
        ],
        dtype=torch.float32,
    )
    mean = torch.zeros((4,), dtype=torch.float32)
    scale = torch.ones((4,), dtype=torch.float32)
    safe_scale = torch.ones((4,), dtype=torch.float32)
    batch_selected_rows = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    batch_donor_means = torch.tensor([[10.0, 99.0, 0.0, 0.0]], dtype=torch.float32)
    row_counts = torch.tensor([1], dtype=torch.int32)
    token_spans = torch.tensor([[0, 2]], dtype=torch.int32)
    strengths = torch.tensor([1.0], dtype=torch.float32)
    active = torch.tensor([1], dtype=torch.int32)
    valid = torch.zeros((1,), dtype=torch.int32)
    scalars = torch.zeros((1, 8), dtype=torch.float32)
    coeff_before = torch.zeros((1, 1), dtype=torch.float32)
    coeff_after = torch.zeros((1, 1), dtype=torch.float32)

    patched = torch.ops.xenon_market_patch.swap_components_batch(
        hidden,
        mean,
        scale,
        safe_scale,
        batch_selected_rows,
        batch_donor_means,
        row_counts,
        token_spans,
        strengths,
        active,
        valid,
        scalars,
        coeff_before,
        coeff_after,
    )

    assert torch.allclose(patched[:2].mean(dim=0), torch.tensor([10.0, 4.0, 3.0, 5.0]), atol=1e-5)
    assert int(valid[0].item()) == 1
    assert torch.allclose(coeff_before[0], torch.tensor([2.0]), atol=1e-5)
    assert torch.allclose(coeff_after[0], torch.tensor([10.0]), atol=1e-5)


def test_compiled_batch_runtime_state_supports_swap_components():
    model = _FakeModel()
    init_market_patching(model)
    register_patch_basis(model, _basis_payload())
    model._market_patch_force_custom_op_presence = True

    patch_payload = MarketPatchSpec(
        mode=PATCH_MODE_SWAP_COMPONENTS,
        target_layers=(0,),
        token_span=(0, 2),
        component_indices_by_layer={0: (0,)},
        donor_mean_by_layer={0: np.asarray([10.0, 99.0, 0.0, 0.0], dtype=np.float32)},
    ).to_payload()

    set_batch_patch_specs(
        model,
        [
            {
                "req_id": "base-abc",
                "patch_spec": patch_payload,
                "target_span": [0, 2],
                "chunk_abs_span": [0, 2],
                "overlap_abs_span": [0, 2],
                "query_span": [0, 2],
                "prefill_chunk_len": 2,
            }
        ],
    )

    runtime_state = model._market_patch_batch_runtime_state[0]
    assert model._market_patch_compiled_batch_modes[0] == PATCH_MODE_SWAP_COMPONENTS
    assert torch.allclose(runtime_state["donor_means"][0], torch.tensor([10.0, 99.0, 0.0, 0.0]))
