from __future__ import annotations

import numpy as np
import torch

from pipelines.interp.vllm_market_patch import (
    PATCH_MODE_ADD_DIRECTION,
    PATCH_MODE_PROJECT_OUT,
    PATCH_MODE_RANDOM_CONTROL,
    PATCH_MODE_SWAP_MEAN,
    MarketPatchSpec,
    clear_patch_spec,
    collect_patch_stats,
    init_market_patching,
    register_patch_basis,
    restore_original_forwards,
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


class _FakeLayer:
    def __init__(self, offset: float = 0.0):
        self.offset = offset
        self.input_layernorm = _FakeSubmodule()

    def forward(self, hidden_states):
        return hidden_states + self.offset


class _FakeModel:
    def __init__(self):
        self.model = type("Inner", (), {"layers": [_FakeLayer(0.0), _FakeLayer(3.0)]})()


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
