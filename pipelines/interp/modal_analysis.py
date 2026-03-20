"""Modal wrapper for running analysis on the xenon-data volume.

Mounts the volume read-only, runs probes/expert analysis/PCA on cheap CPU instances.
Results are saved to /data/analysis_results/ on the volume.

Usage (via wrapper script):
    ./scripts/modal_capture.sh analyze --mode probe --target decision_type
    ./scripts/modal_capture.sh analyze --mode all --target decision_type

Or directly:
    uv run --extra analysis --extra modal modal run pipelines/interp/modal_analysis.py \
        --mode probe --target decision_type
"""

import modal

app = modal.App("xenon-analysis")

volume = modal.Volume.from_name("xenon-data", create_if_missing=True)
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
) -> dict:
    """Run analysis on Modal with volume-mounted activations."""
    from pathlib import Path

    from pipelines.interp.analysis import AnalysisConfig, dispatch

    parsed_layers: list[int] | None = None
    if layers_csv:
        parsed_layers = [int(x.strip()) for x in layers_csv.split(",")]

    config = AnalysisConfig(
        activations_dir=Path("/data/activations"),
        output_dir=Path("/data/analysis_results"),
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

    from pipelines.interp.counterfactual_analysis import (
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

    from pipelines.interp.counterfactual_structure import (
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
) -> dict:
    """Pool full-sequence real-decision captures into row/section structure states."""
    from pathlib import Path

    from pipelines.interp.decision_structure import (
        DecisionStructureConfig,
        run_decision_structure_pooling as _run_pooling,
    )

    config = DecisionStructureConfig(
        activations_dir=Path("/data/activations"),
        output_dir=Path("/data/activations/decision_structure"),
        model_id=f"/models/{model_id}",
        limit=limit if limit > 0 else None,
        skip_existing=skip_existing,
        cohort_view=cohort_view or None,
        order_mode=order_mode,
    )
    results = _run_pooling(config)
    volume.commit()
    return results


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
) -> dict:
    """Analyze pooled real-decision structure activations on Modal."""
    from pathlib import Path

    from pipelines.interp.decision_structure_analysis import (
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
    )
    results = run_decision_structure_analysis(config)
    volume.commit()
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
    cohort_view: str = "",
    order_mode: str = "log_id",
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
        results = run_decision_structure_pooling.remote(
            model_id=model_id,
            limit=limit,
            skip_existing=skip_existing,
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
        )
        print(f"\nDecision structure analysis complete. Results keys: {list(results.keys())}")
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
        )
        print(f"\nAnalysis complete. Results: {results}")

    print("\nTo download results:")
    print("  modal volume get xenon-data analysis_results/ ./data/analysis_results/ --force")
