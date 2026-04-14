"""ARCH2 expected usage: fit a basis on CPU, then patch on GPU."""

from pipelines_v2.api import (
    ActivationPatchSpec,
    BasisSpec,
    CaptureSpec,
    Dataset,
    InterventionSite,
    Metric,
    ModalResources,
    ModalRunner,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PostgresCatalog,
    PostgresSource,
    PromptMetadataBuilder,
    ResidualSite,
    TensorStorage,
    TokenSelector,
    VLLMEngine,
)


def build_prompt_metadata(rendered_prompt: str) -> dict[str, object]:
    market_marker = "MARKET\n"
    decision_marker = "\n\nDECISION\n"
    market_start = rendered_prompt.index(market_marker) + len(market_marker)
    market_end = rendered_prompt.index(decision_marker, market_start)
    return {
        "token_sections": {
            "MARKET": {"char_start": market_start, "char_end": market_end},
        }
    }


def main() -> None:
    db = PostgresSource.from_env("XENON_DATABASE_URL")
    artifact_store = ModalVolumeStore(name="xenon-data", root="/data/artifacts")
    catalog = PostgresCatalog(source=db)
    engine = VLLMEngine(model_id="/models/Qwen/Qwen3-30B-A3B", max_model_len=8192)

    capture_runner = ModalRunner(
        resources=ModalResources(
            gpu="A100-80GB",
            secrets=(
                ModalSecret.from_env_var("XENON_DATABASE_URL", secret_name="xenon-db"),
            ),
            volumes=(
                ModalVolumeMount(name="xenon-models", mount_path="/models"),
            ),
        ),
        artifacts=artifact_store,
        catalog=catalog,
    )

    analysis_runner = ModalRunner(
        resources=ModalResources(
            cpu=6,
            memory_mb=24 * 1024,
            secrets=(
                ModalSecret.from_env_var("XENON_DATABASE_URL", secret_name="xenon-db"),
            ),
        ),
        artifacts=artifact_store,
        catalog=catalog,
    )

    dataset = Dataset.from_postgres(
        source=db,
        table="conflict_examples_v2",
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        prompt_hash_column="prompt_hash",
        label_columns=["conflict_label"],
        case_key_column="matched_pair_id",
    )

    cap = capture_runner.run(
        CaptureSpec(
            engine=engine,
            dataset=dataset,
            prompt_metadata_builder=PromptMetadataBuilder.from_function(build_prompt_metadata),
            sites=[
                ResidualSite(
                    name="resid_post_full",
                    site="resid_post",
                    layers=[16, 20, 24],
                    tokens=TokenSelector.full_sequence(),
                    storage=TensorStorage(dtype="float16"),
                )
            ],
        )
    )

    basis = analysis_runner.run(
        BasisSpec(
            feature=cap.feature("resid_post_full"),
            method="pca",
            by=dataset.labels("conflict_label"),
            layers=[16, 20, 24],
            tokens=TokenSelector.section("MARKET"),
            components=8,
        )
    )

    patch = capture_runner.run(
        ActivationPatchSpec(
            engine=engine,
            dataset=dataset,
            basis=basis,
            site=InterventionSite.residual("resid_post"),
            layers=[16, 20, 24],
            tokens=TokenSelector.section("MARKET"),
            mode="project_out",
            components=["conflict_pc1"],
            strengths=[0.0, 0.5, 1.0, 2.0],
            controls=["random_direction", "neighbor_component"],
            metrics=[
                Metric.choice_flip_rate(),
                Metric.logprob_margin(labels=["aligned", "conflict"]),
                Metric.kl_divergence(),
            ],
        )
    )

    print(patch.summary())


if __name__ == "__main__":
    main()
