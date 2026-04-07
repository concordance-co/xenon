"""Modal CPU workflows for synthetic market structure pooling and analysis."""

from __future__ import annotations

import asyncio
import inspect

import modal

app = modal.App("xenon-synthetic-analysis")

synthetic_volume = modal.Volume.from_name("xenon-synthetic-data", create_if_missing=True)
model_volume = modal.Volume.from_name("xenon-models", create_if_missing=True)
neon_secret = modal.Secret.from_name("xenon-neon")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "transformers",
        "tokenizers",
        "safetensors",
        "scikit-learn",
        "numpy",
        "pyarrow",
        "jinja2",
        "psycopg[binary]",
    )
    .add_local_python_source("pipelines")
)


def _resolve_blocking_result(value):
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


@app.function(
    volumes={"/data": synthetic_volume, "/models": model_volume},
    image=image,
    timeout=12 * 3600,
    cpu=8,
    memory=32 * 1024,
    secrets=[neon_secret],
)
def run_synthetic_structure_pooling_modal(
    phase_name: str = "phase1",
    model_id: str = "Qwen/Qwen3-30B-A3B",
    limit: int = 0,
    skip_existing: bool = True,
    cohort_view: str = "synthetic_market_phase1_capture_v0",
    order_mode: str = "selection_rank_asc",
    num_workers: int = 8,
    num_shards: int = 1,
    shard_index: int = 0,
) -> dict:
    from pathlib import Path

    from pipelines.datasets.synthetic.structure import (
        SyntheticStructureConfig,
        run_synthetic_structure_pooling,
    )

    config = SyntheticStructureConfig(
        activations_dir=Path(f"/data/activations/{phase_name}"),
        output_dir=Path(f"/data/activations/synthetic_structure/{phase_name}"),
        model_id=f"/models/{model_id}",
        limit=limit if limit > 0 else None,
        skip_existing=skip_existing,
        cohort_view=cohort_view,
        order_mode=order_mode,
        num_workers=num_workers,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    result = run_synthetic_structure_pooling(config)
    synthetic_volume.commit()
    return result


@app.function(
    volumes={"/data": synthetic_volume, "/models": model_volume},
    image=image,
    timeout=12 * 3600,
    cpu=2,
    memory=8 * 1024,
    secrets=[neon_secret],
)
def run_synthetic_structure_pooling_parallel(
    phase_name: str = "phase1",
    model_id: str = "Qwen/Qwen3-30B-A3B",
    limit: int = 0,
    skip_existing: bool = False,
    cohort_view: str = "synthetic_market_phase1_capture_v0",
    order_mode: str = "selection_rank_asc",
    num_workers: int = 8,
    num_shards: int = 8,
) -> dict:
    from pathlib import Path

    from modal.functions import FunctionCall

    from pipelines.datasets.synthetic.structure import (
        clear_synthetic_structure_shards,
        merge_synthetic_structure_shards,
    )

    if num_shards <= 1:
        return run_synthetic_structure_pooling_modal.remote(
            phase_name=phase_name,
            model_id=model_id,
            limit=limit,
            skip_existing=skip_existing,
            cohort_view=cohort_view,
            order_mode=order_mode,
            num_workers=num_workers,
            num_shards=1,
            shard_index=0,
        )

    output_dir = Path(f"/data/activations/synthetic_structure/{phase_name}")
    if not skip_existing:
        cleared = clear_synthetic_structure_shards(output_dir, num_shards=num_shards, clear_canonical=True)
        print(
            "Cleared synthetic-structure shard checkpoints before fresh sharded run: "
            f"removed={cleared['removed']} missing={cleared['missing']}",
        )
        synthetic_volume.commit()

    shard_skip_existing = True
    calls = [
        run_synthetic_structure_pooling_modal.spawn(
            phase_name=phase_name,
            model_id=model_id,
            limit=limit,
            skip_existing=shard_skip_existing,
            cohort_view=cohort_view,
            order_mode=order_mode,
            num_workers=num_workers,
            num_shards=num_shards,
            shard_index=shard_index,
        )
        for shard_index in range(num_shards)
    ]
    shard_results = _resolve_blocking_result(FunctionCall.gather(*calls))
    synthetic_volume.reload()
    merge = merge_synthetic_structure_shards(output_dir, num_shards=num_shards)
    synthetic_volume.commit()
    return {
        "num_shards": num_shards,
        "shards": list(shard_results),
        "merge": merge,
    }


@app.function(
    volumes={"/data": synthetic_volume},
    image=image,
    timeout=12 * 3600,
    cpu=8,
    memory=32 * 1024,
    secrets=[neon_secret],
)
def run_synthetic_manifold_analysis_modal(
    phase_name: str = "phase1",
    context_variant: str = "market_only",
    family_allowlist: str = "",
    scalar_family_name: str = "scalar_sweep",
    analysis_tag: str = "",
    layers: str = "",
    num_workers: int = 8,
) -> dict:
    from pathlib import Path

    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_manifold_analysis import (
        SyntheticManifoldAnalysisConfig,
        run_synthetic_manifold_analysis,
    )

    parsed_layers = [int(token) for token in layers.split(",") if token.strip()] or None
    config = SyntheticManifoldAnalysisConfig(
        structure_dir=Path(f"/data/activations/synthetic_structure/{phase_name}"),
        output_dir=(
            Path(f"/data/analysis_results/synthetic_manifold/{phase_name}") / analysis_tag
            if analysis_tag
            else Path(f"/data/analysis_results/synthetic_manifold/{phase_name}")
        ),
        phase_name=phase_name,
        context_variant=context_variant,
        family_allowlist=tuple(token.strip() for token in family_allowlist.split(",") if token.strip()),
        scalar_family_name=scalar_family_name,
        layers=parsed_layers,
        num_workers=num_workers,
    )
    result = run_synthetic_manifold_analysis(config)
    synthetic_volume.commit()
    return result


@app.function(
    volumes={"/data": synthetic_volume},
    image=image,
    timeout=12 * 3600,
    cpu=8,
    memory=32 * 1024,
    secrets=[neon_secret],
)
def run_synthetic_policy_analysis_modal(
    phase_name: str = "policy_algebra_v1",
    analysis_tag: str = "",
    num_workers: int = 8,
) -> dict:
    from pathlib import Path

    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_policy_analysis import (
        SyntheticPolicyAnalysisConfig,
        run_synthetic_policy_analysis,
    )

    config = SyntheticPolicyAnalysisConfig(
        phase_name=phase_name,
        structure_dir=Path(f"/data/activations/synthetic_structure/{phase_name}"),
        output_dir=(
            Path(f"/data/analysis_results/synthetic_policy/{phase_name}") / analysis_tag
            if analysis_tag
            else Path(f"/data/analysis_results/synthetic_policy/{phase_name}")
        ),
        max_workers=num_workers,
    )
    result = run_synthetic_policy_analysis(config)
    synthetic_volume.commit()
    return result


@app.function(
    volumes={"/data": synthetic_volume},
    image=image,
    timeout=12 * 3600,
    cpu=8,
    memory=32 * 1024,
    secrets=[neon_secret],
)
def run_synthetic_market_representation_analysis_modal(
    phase_name: str = "phase4_market_representation_v1",
    analysis_tag: str = "",
    layers: str = "",
    num_workers: int = 8,
) -> dict:
    from pathlib import Path

    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_representation_analysis import (
        SyntheticMarketRepresentationConfig,
        run_synthetic_market_representation_analysis,
    )

    parsed_layers = [int(token) for token in layers.split(",") if token.strip()] or None
    config = SyntheticMarketRepresentationConfig(
        structure_dir=Path(f"/data/activations/synthetic_structure/{phase_name}"),
        output_dir=(
            Path(f"/data/analysis_results/synthetic_market_representation/{phase_name}") / analysis_tag
            if analysis_tag
            else Path(f"/data/analysis_results/synthetic_market_representation/{phase_name}")
        ),
        phase_name=phase_name,
        layers=parsed_layers,
        num_workers=num_workers,
    )
    result = run_synthetic_market_representation_analysis(config)
    synthetic_volume.commit()
    return result


@app.function(
    volumes={"/data": synthetic_volume},
    image=image,
    timeout=12 * 3600,
    cpu=8,
    memory=32 * 1024,
    secrets=[neon_secret],
)
def run_synthetic_market_transform_analysis_modal(
    phase_name: str = "phase11_set_geometry_risk_ladder_v1",
    analysis_tag: str = "",
    num_workers: int = 8,
) -> dict:
    from pathlib import Path

    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_transform_analysis import (
        SyntheticMarketTransformConfig,
        run_synthetic_market_transform_analysis,
    )

    config = SyntheticMarketTransformConfig(
        structure_dir=Path(f"/data/activations/synthetic_structure/{phase_name}"),
        phase11_results_path=Path(f"/data/analysis_results/synthetic_market_representation/{phase_name}/results.json"),
        output_dir=(
            Path(f"/data/analysis_results/synthetic_market_transform/{phase_name}") / analysis_tag
            if analysis_tag
            else Path(f"/data/analysis_results/synthetic_market_transform/{phase_name}")
        ),
        phase_name=phase_name,
        num_workers=num_workers,
    )
    result = run_synthetic_market_transform_analysis(config)
    synthetic_volume.commit()
    return result


@app.function(
    volumes={"/data": synthetic_volume},
    image=image,
    timeout=12 * 3600,
    cpu=8,
    memory=32 * 1024,
    secrets=[neon_secret],
)
def run_synthetic_market_discovery_analysis_modal(
    phase_name: str = "phase15_market_basis_discovery_v1",
    analysis_tag: str = "",
    layers: str = "",
    num_workers: int = 8,
    residualize_nuisance: bool = False,
) -> dict:
    from pathlib import Path

    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_discovery_analysis import (
        SyntheticMarketDiscoveryConfig,
        run_synthetic_market_discovery_analysis,
    )

    parsed_layers = [int(token) for token in layers.split(",") if token.strip()] or None
    config = SyntheticMarketDiscoveryConfig(
        structure_dir=Path(f"/data/activations/synthetic_structure/{phase_name}"),
        output_dir=(
            Path(f"/data/analysis_results/synthetic_market_discovery/{phase_name}") / analysis_tag
            if analysis_tag
            else Path(f"/data/analysis_results/synthetic_market_discovery/{phase_name}")
        ),
        phase_name=phase_name,
        layers=parsed_layers,
        num_workers=num_workers,
        residualize_nuisance=residualize_nuisance,
    )
    result = run_synthetic_market_discovery_analysis(config)
    synthetic_volume.commit()
    return result


@app.function(
    volumes={"/data": synthetic_volume},
    image=image,
    timeout=12 * 3600,
    cpu=8,
    memory=32 * 1024,
    secrets=[neon_secret],
)
def run_synthetic_market_context_order_analysis_modal(
    phase_name: str = "phase16_context_order_v1",
    analysis_tag: str = "",
    layers: str = "",
    num_workers: int = 8,
    cross_basis_overrides: str = "",
    cross_basis_layers: str = "40,42",
) -> dict:
    from pathlib import Path

    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_context_order_analysis import (
        SyntheticMarketContextOrderConfig,
        run_synthetic_market_context_order_analysis,
    )

    parsed_layers = [int(token) for token in layers.split(",") if token.strip()] or None
    parsed_cross_basis_overrides: dict[str, str] = {}
    for token in cross_basis_overrides.split(","):
        token = token.strip()
        if not token:
            continue
        src, dst = token.split(":", 1)
        parsed_cross_basis_overrides[src.strip()] = dst.strip()
    parsed_cross_basis_layers = tuple(int(token) for token in cross_basis_layers.split(",") if token.strip())
    config = SyntheticMarketContextOrderConfig(
        structure_dir=Path(f"/data/activations/synthetic_structure/{phase_name}"),
        output_dir=(
            Path(f"/data/analysis_results/synthetic_market_context_order/{phase_name}") / analysis_tag
            if analysis_tag
            else Path(f"/data/analysis_results/synthetic_market_context_order/{phase_name}")
        ),
        phase_name=phase_name,
        basis_npz_path=Path(
            "/data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/pca_basis.npz"
        ),
        basis_results_path=Path(
            "/data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/results.json"
        ),
        layers=parsed_layers,
        num_workers=num_workers,
        cross_basis_overrides=parsed_cross_basis_overrides or None,
        cross_basis_layers=parsed_cross_basis_layers,
    )
    result = run_synthetic_market_context_order_analysis(config)
    synthetic_volume.commit()
    return result


@app.function(
    volumes={"/data": synthetic_volume},
    image=image,
    timeout=12 * 3600,
    cpu=8,
    memory=32 * 1024,
    secrets=[neon_secret],
)
def run_synthetic_market_axis_decomposition_analysis_modal(
    phase_name: str = "phase15_market_basis_discovery_v1",
    analysis_tag: str = "",
    context_variant: str = "market_only",
    num_workers: int = 8,
) -> dict:
    from pathlib import Path

    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_axis_decomposition_analysis import (
        SyntheticMarketAxisDecompositionConfig,
        run_synthetic_market_axis_decomposition_analysis,
    )

    config = SyntheticMarketAxisDecompositionConfig(
        structure_dir=Path(f"/data/activations/synthetic_structure/{phase_name}"),
        output_dir=(
            Path(f"/data/analysis_results/synthetic_market_axis_decomposition/{phase_name}") / analysis_tag
            if analysis_tag
            else Path(f"/data/analysis_results/synthetic_market_axis_decomposition/{phase_name}")
        ),
        phase_name=phase_name,
        context_variant=context_variant,
        basis_npz_path=Path(
            "/data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/pca_basis.npz"
        ),
        basis_results_path=Path(
            "/data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/results.json"
        ),
        num_workers=num_workers,
    )
    result = run_synthetic_market_axis_decomposition_analysis(config)
    synthetic_volume.commit()
    return result


@app.local_entrypoint()
def main(
    mode: str = "synthetic-structure",
    phase_name: str = "phase1",
    model_id: str = "Qwen/Qwen3-30B-A3B",
    limit: int = 0,
    skip_existing: bool = True,
    cohort_view: str = "synthetic_market_phase1_capture_v0",
    order_mode: str = "selection_rank_asc",
    num_workers: int = 8,
    num_shards: int = 1,
    layers: str = "",
    context_variant: str = "market_only",
    family_allowlist: str = "",
    scalar_family_name: str = "scalar_sweep",
    analysis_tag: str = "",
    residualize_nuisance: bool = False,
    cross_basis_overrides: str = "",
    cross_basis_layers: str = "40,42",
):
    if mode == "synthetic-structure":
        if num_shards > 1:
            result = run_synthetic_structure_pooling_parallel.remote(
                phase_name=phase_name,
                model_id=model_id,
                limit=limit,
                skip_existing=skip_existing,
                cohort_view=cohort_view,
                order_mode=order_mode,
                num_workers=num_workers,
                num_shards=num_shards,
            )
        else:
            result = run_synthetic_structure_pooling_modal.remote(
                phase_name=phase_name,
                model_id=model_id,
                limit=limit,
                skip_existing=skip_existing,
                cohort_view=cohort_view,
                order_mode=order_mode,
                num_workers=num_workers,
                num_shards=1,
                shard_index=0,
            )
        print(result)
    elif mode == "synthetic-manifold":
        result = run_synthetic_manifold_analysis_modal.remote(
            phase_name=phase_name,
            context_variant=context_variant,
            family_allowlist=family_allowlist,
            scalar_family_name=scalar_family_name,
            analysis_tag=analysis_tag,
            layers=layers,
            num_workers=num_workers,
        )
        print(result)
    elif mode == "synthetic-policy":
        result = run_synthetic_policy_analysis_modal.remote(
            phase_name=phase_name,
            analysis_tag=analysis_tag,
            num_workers=num_workers,
        )
        print(result)
    elif mode == "synthetic-representation":
        result = run_synthetic_market_representation_analysis_modal.remote(
            phase_name=phase_name,
            analysis_tag=analysis_tag,
            layers=layers,
            num_workers=num_workers,
        )
        print(result)
    elif mode == "synthetic-transform":
        result = run_synthetic_market_transform_analysis_modal.remote(
            phase_name=phase_name,
            analysis_tag=analysis_tag,
            num_workers=num_workers,
        )
        print(result)
    elif mode == "synthetic-discovery":
        result = run_synthetic_market_discovery_analysis_modal.remote(
            phase_name=phase_name,
            analysis_tag=analysis_tag,
            layers=layers,
            num_workers=num_workers,
            residualize_nuisance=residualize_nuisance,
        )
        print(result)
    elif mode == "synthetic-context-order":
        result = run_synthetic_market_context_order_analysis_modal.remote(
            phase_name=phase_name,
            analysis_tag=analysis_tag,
            layers=layers,
            num_workers=num_workers,
            cross_basis_overrides=cross_basis_overrides,
            cross_basis_layers=cross_basis_layers,
        )
        print(result)
    elif mode == "synthetic-axis-decomposition":
        result = run_synthetic_market_axis_decomposition_analysis_modal.remote(
            phase_name=phase_name,
            analysis_tag=analysis_tag,
            context_variant=context_variant,
            num_workers=num_workers,
        )
        print(result)
    else:
        print(f"Unknown mode: {mode}")
