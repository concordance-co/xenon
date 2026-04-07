from __future__ import annotations

from pathlib import Path

from pipelines.interp.patching.basis import ActivationPatchBasis, load_activation_patch_basis


DEFAULT_SYNTHETIC_MARKET_BASIS_NPZ = Path(
    "data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/pca_basis.npz"
)
DEFAULT_SYNTHETIC_MARKET_BASIS_RESULTS = Path(
    "data/analysis_results/synthetic_market_axis_decomposition/phase17_market_axis_decomposition_v1/results.json"
)


def load_phase17_activation_patch_basis(
    *,
    basis_npz_path: Path = DEFAULT_SYNTHETIC_MARKET_BASIS_NPZ,
    results_json_path: Path = DEFAULT_SYNTHETIC_MARKET_BASIS_RESULTS,
    state_key: str = "market_mean",
    layers: tuple[int, ...] = (4, 35),
    components_per_layer: int = 4,
) -> ActivationPatchBasis:
    return load_activation_patch_basis(
        basis_npz_path=basis_npz_path,
        results_json_path=results_json_path,
        state_key=state_key,
        layers=layers,
        components_per_layer=components_per_layer,
    )
