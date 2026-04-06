"""Canonical Modal analysis orchestrator.

This module runs the shared analysis engine against activation artifacts stored
on the Modal volume, writes analysis outputs back to the volume, and exports
workflow labels from Neon when needed. It is the execution-plane counterpart to
the local engine in ``pipelines.interp.analysis``.
"""

import asyncio
import inspect
from pathlib import Path

import modal

app = modal.App("xenon-analysis")

volume = modal.Volume.from_name("xenon-data", create_if_missing=True)
projects_volume = modal.Volume.from_name("xenon-research-data", create_if_missing=True)
model_volume = modal.Volume.from_name("xenon-models", create_if_missing=True)

neon_secret = modal.Secret.from_name("xenon-neon")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch", "transformers", "safetensors", "scikit-learn", "matplotlib",
        "numpy", "pyarrow", "psycopg[binary]",
    )
    .add_local_python_source("pipelines")
)


def _resolve_blocking_result(value):
    """Handle APIs that may return either a ready value or an awaitable."""
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=1800,
    cpu=4,
    secrets=[neon_secret],
)
def run_analysis(
    mode: str,
    target: str,
    data_source: str = "router",
    pooling: str = "last_token",
    layers_csv: str = "",
    n_folds: int = 5,
    seed: int = 42,
    limit: int = 0,
    relation_name: str = "",
    activations_subdir: str = "",
    output_subdir: str = "",
    labels_subdir: str = "",
) -> dict:
    """Run analysis on Modal with volume-mounted activations."""
    from pipelines.interp.analysis import AnalysisConfig, dispatch
    from pipelines.db import connect_neon, ensure_schema
    from pipelines.workflows import export_publication_labels

    parsed_layers: list[int] | None = None
    if layers_csv:
        parsed_layers = [int(x.strip()) for x in layers_csv.split(",")]

    activations_dir = Path("/data/activations")
    if activations_subdir:
        activations_dir = activations_dir / activations_subdir

    output_dir = Path("/data/analysis_results")
    if output_subdir:
        output_dir = output_dir / output_subdir

    labels_path: Path | None = None
    if relation_name:
        labels_path = Path("/data/workflow_labels")
        if labels_subdir:
            labels_path = labels_path / labels_subdir
        labels_path.mkdir(parents=True, exist_ok=True)
        labels_path = labels_path / f"{relation_name}.parquet"
        with connect_neon(autocommit=True) as conn:
            ensure_schema(conn)
            export_publication_labels(conn, relation_name=relation_name, output_path=labels_path)

    config = AnalysisConfig(
        activations_dir=activations_dir,
        labels_path=labels_path,
        output_dir=output_dir,
        mode=mode,
        target=target,
        data_source=data_source,
        pooling=pooling,
        n_folds=n_folds,
        layers=parsed_layers,
        limit=limit if limit > 0 else None,
        seed=seed,
    )

    results = dispatch(config)
    volume.commit()
    return results


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=7200,
    cpu=16,
    memory=32 * 1024,  # 32 GB — ~5GB per dataset cache + sklearn overhead
    secrets=[neon_secret],
)
def run_counterfactual_analysis(
    experiment_id: str = "default",
    n_bootstrap: int = 1000,
    seed: int = 42,
    layers_csv: str = "",
    questions: str = "a",
) -> dict:
    """Run counterfactual experiment analysis on Modal.

    questions: comma-separated list of questions to run (a, b, c, all).
    Uses 16 threads to parallelize probe training across layers.
    """
    from pathlib import Path

    from projects.DX_TERMINAL.phases.counterfactual.analysis import (
        CounterfactualAnalysisConfig,
        apply_decision_rules,
        run_experiment_a,
        run_question_b,
        run_question_c,
    )

    parsed_layers: list[int] | None = None
    if layers_csv:
        parsed_layers = [int(x.strip()) for x in layers_csv.split(",")]

    config = CounterfactualAnalysisConfig(
        activations_dir=Path("/data/activations/counterfactual"),
        experiment_id=experiment_id,
        output_dir=Path("/data/analysis_results/counterfactual"),
        n_bootstrap=n_bootstrap,
        seed=seed,
        layers=parsed_layers,
    )

    import json
    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    q_set = set(q.strip().lower() for q in questions.split(","))
    run_all = "all" in q_set

    results: dict = {}
    _b_cache = None  # Shared cache between Q-B and Q-C

    # Question A
    if run_all or "a" in q_set:
        print("=== Question A: Pre-market entanglement ===")
        results["question_a"] = run_experiment_a(config, max_workers=16)
        (results_dir / "question_a_results.json").write_text(
            json.dumps(results["question_a"], indent=2, default=str),
        )

    # Question B — preloads Dataset B cache, reused by Q-C
    if run_all or "b" in q_set:
        print("\n=== Question B: Post-market reinterpretation ===")
        results["question_b"] = run_question_b(config, max_workers=16)
        (results_dir / "question_b_results.json").write_text(
            json.dumps(results["question_b"], indent=2, default=str),
        )

    # Question C — reuses Q-B cache if both ran, otherwise loads its own
    if run_all or "c" in q_set:
        print("\n=== Question C: Decision-layer interaction ===")
        # If Q-B ran, preload cache for Q-C to share (Q-B already loaded these)
        # Q-C's preload_all_activations will be fast since files are in OS page cache
        results["question_c"] = run_question_c(config, max_workers=16)
        (results_dir / "question_c_results.json").write_text(
            json.dumps(results["question_c"], indent=2, default=str),
        )

    # Apply decision rules if we have at least Question A
    if "question_a" in results:
        decision = apply_decision_rules(
            results.get("question_a", {}),
            results.get("question_b"),
            results.get("question_c"),
        )
        results["decision"] = decision
        (results_dir / "decision.json").write_text(
            json.dumps(decision, indent=2, default=str),
        )
        print(f"\n=== Decision: {decision['decision']} ===")
        print(f"    {decision['reasoning']}")

    volume.commit()
    return results


@app.function(
    volumes={"/data": volume, "/models": model_volume},
    image=image,
    timeout=7200,
    cpu=8,
    memory=24 * 1024,
    secrets=[neon_secret],
)
def run_counterfactual_structure_analysis(
    experiment_id: str = "default",
    seed: int = 42,
    layers_csv: str = "",
    train_variant: str = "settings_all1",
    compare_variant: str = "settings_all5",
    row_key: str = "row_mean",
    variance_threshold: float = 0.9,
) -> dict:
    """Run pre/post counterfactual structure analysis on Modal."""
    from pathlib import Path

    from projects.DX_TERMINAL.phases.counterfactual.structure import (
        CounterfactualStructureConfig,
        run_counterfactual_structure,
    )

    parsed_layers: list[int] | None = None
    if layers_csv:
        parsed_layers = [int(x.strip()) for x in layers_csv.split(",")]

    config = CounterfactualStructureConfig(
        activations_dir=Path("/data/activations/counterfactual"),
        experiment_id=experiment_id,
        output_dir=Path("/data/analysis_results/counterfactual_structure"),
        train_variant=train_variant,
        compare_variant=compare_variant,
        row_key=row_key,
        variance_threshold=variance_threshold,
        layers=parsed_layers,
        seed=seed,
    )

    results = run_counterfactual_structure(config)
    volume.commit()
    return results


@app.function(
    volumes={"/data": volume, "/models": model_volume},
    image=image,
    timeout=7200,
    cpu=8,
    memory=24 * 1024,
    secrets=[neon_secret],
)
def run_decision_structure_pooling(
    model_id: str = "Qwen/Qwen3-30B-A3B",
    limit: int = 0,
    skip_existing: bool = True,
    cohort_view: str = "",
    order_mode: str = "log_id",
    num_workers: int = 8,
    num_shards: int = 1,
    shard_index: int = 0,
) -> dict:
    """Pool full-sequence real-decision captures into row/section structure states."""
    from pathlib import Path

    from projects.DX_TERMINAL.phases.decision_structure import (
        DecisionStructureConfig,
        run_decision_structure_pooling as _run_pooling,
    )

    config = DecisionStructureConfig(
        activations_dir=Path("/data/activations"),
        output_dir=Path("/data/activations/decision_structure"),
        model_id=f"/models/{model_id}",
        limit=limit if limit > 0 else None,
        skip_existing=skip_existing,
        num_workers=num_workers,
        num_shards=num_shards,
        shard_index=shard_index,
        cohort_view=cohort_view or None,
        order_mode=order_mode,
    )
    results = _run_pooling(config)
    volume.commit()
    return results


@app.function(
    volumes={"/data": volume, "/models": model_volume},
    image=image,
    timeout=12 * 3600,
    cpu=2,
    memory=8 * 1024,
    secrets=[neon_secret],
)
def run_decision_structure_pooling_parallel(
    model_id: str = "Qwen/Qwen3-30B-A3B",
    limit: int = 0,
    skip_existing: bool = False,
    cohort_view: str = "",
    order_mode: str = "log_id",
    num_workers: int = 8,
    num_shards: int = 10,
) -> dict:
    """Pool full-sequence decision captures across multiple Modal containers, then merge shard outputs."""
    from pathlib import Path

    from modal.functions import FunctionCall

    from projects.DX_TERMINAL.phases.decision_structure import (
        clear_decision_structure_shards,
        merge_decision_structure_shards,
    )

    if num_shards <= 1:
        return run_decision_structure_pooling.remote(
            model_id=model_id,
            limit=limit,
            skip_existing=skip_existing,
            cohort_view=cohort_view,
            order_mode=order_mode,
            num_workers=num_workers,
            num_shards=1,
            shard_index=0,
        )

    output_dir = Path("/data/activations/decision_structure")
    if not skip_existing:
        cleared = clear_decision_structure_shards(output_dir, num_shards=num_shards, clear_canonical=True)
        print(
            "Cleared decision-structure shard checkpoints before fresh sharded run: "
            f"removed={cleared['removed']} missing={cleared['missing']}",
        )
        volume.commit()

    shard_skip_existing = True

    calls = [
        run_decision_structure_pooling.spawn(
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
    volume.reload()
    merge = merge_decision_structure_shards(
        output_dir,
        num_shards=num_shards,
    )
    volume.commit()
    return {
        "num_shards": num_shards,
        "shards": list(shard_results),
        "merge": merge,
    }


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=7200,
    cpu=8,
    memory=24 * 1024,
)
def run_decision_structure_analysis_modal(
    row_key: str = "row_mean",
    layers_csv: str = "",
    seed: int = 42,
    test_fraction: float = 0.2,
    num_workers: int = 8,
) -> dict:
    """Analyze pooled real-decision structure activations on Modal."""
    from pathlib import Path

    from projects.DX_TERMINAL.phases.decision_structure.analysis import (
        DecisionStructureAnalysisConfig,
        run_decision_structure_analysis,
    )

    parsed_layers: list[int] | None = None
    if layers_csv:
        parsed_layers = [int(x.strip()) for x in layers_csv.split(",")]

    config = DecisionStructureAnalysisConfig(
        structure_dir=Path("/data/activations/decision_structure"),
        output_dir=Path("/data/analysis_results/decision_structure"),
        row_key=row_key,
        layers=parsed_layers,
        seed=seed,
        test_fraction=test_fraction,
        num_workers=num_workers,
    )
    results = run_decision_structure_analysis(config)
    volume.commit()
    return results


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=7200,
    cpu=8,
    memory=24 * 1024,
)
def run_decision_structure_sanity_modal(
    target: str = "is_buy_target",
    summary_bucket: str = "best_pre",
    seed: int = 42,
    test_fraction: float = 0.2,
) -> dict:
    """Run a probe-vs-raw-metric sanity check on intact Modal pooled residuals."""
    from pathlib import Path

    from projects.DX_TERMINAL.phases.decision_structure.sanity import (
        DecisionStructureSanityConfig,
        run_decision_structure_sanity,
    )

    config = DecisionStructureSanityConfig(
        structure_dir=Path("/data/activations/decision_structure"),
        results_path=Path("/data/analysis_results/decision_structure/decision_structure_results.json"),
        output_path=Path(f"/data/analysis_results/decision_structure/{target}_{summary_bucket}_metric_sanity.json"),
        target=target,
        summary_bucket=summary_bucket,
        seed=seed,
        test_fraction=test_fraction,
    )
    results = run_decision_structure_sanity(config)
    volume.commit()
    return results


@app.function(
    volumes={"/data": volume, "/projects": projects_volume},
    image=image,
    timeout=7200,
    cpu=16,
    memory=64 * 1024,
    secrets=[neon_secret],
)
def run_research_rerun_analysis_modal(
    experiment_id: str = "blocked_valence_settings_twist_kickoff_v1",
    seed: int = 42,
    test_fraction: float = 0.2,
    num_workers: int = 16,
) -> dict:
    """Analyze real-prompt rerun captures against real decision-structure probes."""
    from pathlib import Path

    from projects.DX_TERMINAL.phases.research_rerun.analysis import (
        ResearchRerunAnalysisConfig,
        run_research_rerun_analysis,
    )

    config = ResearchRerunAnalysisConfig(
        decision_structure_dir=Path("/data/activations/decision_structure"),
        decision_results_path=Path("/data/analysis_results/decision_structure"),
        research_activations_dir=Path("/projects/activations/research_rerun"),
        output_dir=Path("/projects/analysis_results/research_rerun"),
        experiment_id=experiment_id,
        seed=seed,
        test_fraction=test_fraction,
        num_workers=num_workers,
    )
    results = run_research_rerun_analysis(config)
    projects_volume.commit()
    return results


@app.function(
    volumes={"/data": volume, "/projects": projects_volume},
    image=image,
    timeout=7200,
    cpu=16,
    memory=64 * 1024,
    secrets=[neon_secret],
)
def run_research_risk_geometry_analysis_modal(
    experiment_id: str = "real_risk_geometry_bridge_v1",
    seed: int = 42,
    test_fraction: float = 0.2,
    num_workers: int = 16,
) -> dict:
    """Analyze real DX risk-ladder reruns with the set-geometry lens."""
    from pathlib import Path

    from projects.DX_TERMINAL.phases.research_rerun.geometry import (
        ResearchRiskGeometryConfig,
        run_research_risk_geometry_analysis,
    )

    config = ResearchRiskGeometryConfig(
        research_activations_dir=Path("/projects/activations/research_rerun"),
        output_dir=Path("/projects/analysis_results/research_risk_geometry"),
        experiment_id=experiment_id,
        seed=seed,
        test_fraction=test_fraction,
        num_workers=num_workers,
    )
    results = run_research_risk_geometry_analysis(config)
    projects_volume.commit()
    return results


@app.function(
    volumes={"/data": volume, "/projects": projects_volume},
    image=image,
    timeout=7200,
    cpu=16,
    memory=64 * 1024,
    secrets=[neon_secret],
)
def run_research_postmarket_geometry_analysis_modal(
    experiment_id: str = "real_postmarket_geometry_bridge_v1",
    seed: int = 42,
    test_fraction: float = 0.2,
    num_workers: int = 16,
) -> dict:
    """Analyze real DX post-market risk and affordance ladders."""
    from pathlib import Path

    from projects.DX_TERMINAL.phases.research_rerun.postmarket_geometry import (
        PostMarketGeometryConfig,
        run_postmarket_geometry_analysis,
    )

    config = PostMarketGeometryConfig(
        research_activations_dir=Path("/projects/activations/research_rerun"),
        output_dir=Path("/projects/analysis_results/research_postmarket_geometry"),
        experiment_id=experiment_id,
        seed=seed,
        test_fraction=test_fraction,
        num_workers=num_workers,
    )
    results = run_postmarket_geometry_analysis(config)
    projects_volume.commit()
    return results


@app.local_entrypoint()
def main(
    mode: str = "probe",
    target: str = "decision_type",
    data_source: str = "router",
    pooling: str = "last_token",
    layers: str = "",
    n_folds: int = 5,
    seed: int = 42,
    limit: int = 0,
    relation_name: str = "",
    activations_subdir: str = "",
    output_subdir: str = "",
    labels_subdir: str = "",
    # Counterfactual args
    experiment_id: str = "default",
    n_bootstrap: int = 1000,
    questions: str = "a",
    train_variant: str = "settings_all1",
    compare_variant: str = "settings_all5",
    row_key: str = "row_mean",
    variance_threshold: float = 0.9,
    model_id: str = "Qwen/Qwen3-30B-A3B",
    skip_existing: bool = True,
    test_fraction: float = 0.2,
    num_workers: int = 8,
    num_shards: int = 1,
    cohort_view: str = "",
    order_mode: str = "log_id",
    summary_bucket: str = "best_pre",
):
    if mode == "counterfactual":
        results = run_counterfactual_analysis.remote(
            experiment_id=experiment_id,
            n_bootstrap=n_bootstrap,
            seed=seed,
            layers_csv=layers,
            questions=questions,
        )
        print(f"\nCounterfactual analysis complete. Results keys: {list(results.keys())}")
    elif mode == "counterfactual-structure":
        results = run_counterfactual_structure_analysis.remote(
            experiment_id=experiment_id,
            seed=seed,
            layers_csv=layers,
            train_variant=train_variant,
            compare_variant=compare_variant,
            row_key=row_key,
            variance_threshold=variance_threshold,
        )
        print(f"\nCounterfactual structure analysis complete. Results keys: {list(results.keys())}")
    elif mode == "decision-structure":
        if num_shards > 1:
            results = run_decision_structure_pooling_parallel.remote(
                model_id=model_id,
                limit=limit,
                skip_existing=skip_existing,
                num_workers=num_workers,
                num_shards=num_shards,
                cohort_view=cohort_view,
                order_mode=order_mode,
            )
        else:
            results = run_decision_structure_pooling.remote(
                model_id=model_id,
                limit=limit,
                skip_existing=skip_existing,
                num_workers=num_workers,
                num_shards=1,
                shard_index=0,
                cohort_view=cohort_view,
                order_mode=order_mode,
            )
        print(f"\nDecision structure pooling complete. Results: {results}")
    elif mode == "decision-structure-analysis":
        results = run_decision_structure_analysis_modal.remote(
            row_key=row_key,
            layers_csv=layers,
            seed=seed,
            test_fraction=test_fraction,
            num_workers=num_workers,
        )
        print(f"\nDecision structure analysis complete. Results keys: {list(results.keys())}")
    elif mode == "decision-structure-sanity":
        results = run_decision_structure_sanity_modal.remote(
            target=target,
            summary_bucket=summary_bucket,
            seed=seed,
            test_fraction=test_fraction,
        )
        print(f"\nDecision structure sanity complete. Results keys: {list(results.keys())}")
    elif mode == "research-rerun-analysis":
        results = run_research_rerun_analysis_modal.remote(
            experiment_id=experiment_id,
            seed=seed,
            test_fraction=test_fraction,
            num_workers=num_workers,
        )
        print(f"\nResearch rerun analysis complete. Results keys: {list(results.keys())}")
    elif mode == "research-risk-geometry-analysis":
        results = run_research_risk_geometry_analysis_modal.remote(
            experiment_id=experiment_id,
            seed=seed,
            test_fraction=test_fraction,
            num_workers=num_workers,
        )
        print(f"\nResearch risk geometry analysis complete. Results keys: {list(results.keys())}")
    elif mode == "research-postmarket-geometry-analysis":
        results = run_research_postmarket_geometry_analysis_modal.remote(
            experiment_id=experiment_id,
            seed=seed,
            test_fraction=test_fraction,
            num_workers=num_workers,
        )
        print(f"\nResearch post-market geometry analysis complete. Results keys: {list(results.keys())}")
    else:
        results = run_analysis.remote(
            mode=mode,
            target=target,
            data_source=data_source,
            pooling=pooling,
            layers_csv=layers,
            n_folds=n_folds,
            seed=seed,
            limit=limit,
            relation_name=relation_name,
            activations_subdir=activations_subdir,
            output_subdir=output_subdir,
            labels_subdir=labels_subdir,
        )
        print(f"\nAnalysis complete. Results: {results}")

    print("\nTo download results:")
    print("  modal volume get xenon-data analysis_results/ ./data/analysis_results/ --force")
    print("  modal volume get xenon-research-data analysis_results/ ./data/analysis_results/ --force")
