from __future__ import annotations

import json

import numpy as np

from pipelines.interp.patching.basis import load_activation_patch_basis


def test_load_activation_patch_basis_reads_named_components(tmp_path):
    basis_path = tmp_path / "basis.npz"
    results_path = tmp_path / "results.json"

    np.savez_compressed(
        basis_path,
        market_mean_layer_4__mean=np.zeros((6,), dtype=np.float32),
        market_mean_layer_4__scale=np.ones((6,), dtype=np.float32),
        market_mean_layer_4__components=np.eye(4, 6, dtype=np.float32),
        market_mean_layer_35__mean=np.zeros((6,), dtype=np.float32),
        market_mean_layer_35__scale=np.ones((6,), dtype=np.float32),
        market_mean_layer_35__components=np.eye(4, 6, dtype=np.float32),
    )
    results_path.write_text(
        json.dumps(
            {
                "targets": {
                    "leader_axis": {
                        "state_key": "market_mean",
                        "layer": 4,
                        "pc_index": 1,
                    },
                    "dispersion_axis": {
                        "state_key": "market_mean",
                        "layer": 35,
                        "pc_index": 1,
                    },
                }
            }
        )
    )

    basis = load_activation_patch_basis(
        basis_npz_path=basis_path,
        state_key="market_mean",
        results_json_path=results_path,
        layers=(4, 35),
        components_per_layer=3,
    )

    assert basis.state_key == "market_mean"
    assert sorted(basis.layers) == [4, 35]
    assert basis.layers[4].components.shape == (3, 6)
    assert basis.layers[35].components.shape == (3, 6)
    assert basis.layers[4].named_components == {"leader_axis": 0}
    assert basis.layers[35].named_components == {"dispersion_axis": 0}


def test_load_activation_patch_basis_raises_for_missing_layer_payload(tmp_path):
    basis_path = tmp_path / "basis.npz"
    np.savez_compressed(
        basis_path,
        market_mean_layer_4__mean=np.zeros((4,), dtype=np.float32),
        market_mean_layer_4__scale=np.ones((4,), dtype=np.float32),
        market_mean_layer_4__components=np.eye(2, 4, dtype=np.float32),
    )

    try:
        load_activation_patch_basis(
            basis_npz_path=basis_path,
            state_key="market_mean",
            layers=(4, 35),
            components_per_layer=2,
        )
    except KeyError as exc:
        assert "market_mean_layer_35" in str(exc)
    else:
        raise AssertionError("Expected missing layer payload to raise KeyError")
