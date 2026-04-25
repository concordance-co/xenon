"""ARCH expected usage: capture once, then run SAE analysis on CPU workers."""

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    FeatureSelectivitySpec,
    ModalResources,
    ModalRunner,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PostgresCatalog,
    PostgresSource,
    ResidualSite,
    SAEEncodeSpec,
    SAESource,
    TensorStorage,
    TokenSelector,
    VLLMEngine,
)


def main() -> None:
    db = PostgresSource.from_env("XENON_NEON_DATABASE_URL")
    artifact_store = ModalVolumeStore(name="xenon-data", root="/data/artifacts")
    catalog = PostgresCatalog(source=db)
    base_model_id = "Qwen/Qwen3-30B-A3B"

    capture_runner = ModalRunner(
        resources=ModalResources(
            gpu="A100-80GB",
            secrets=(
                ModalSecret.from_env_var("XENON_NEON_DATABASE_URL", secret_name="xenon-db"),
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
                ModalSecret.from_env_var("XENON_NEON_DATABASE_URL", secret_name="xenon-db"),
            ),
        ),
        artifacts=artifact_store,
        catalog=catalog,
    )

    dataset = Dataset.from_postgres(
        source=db,
        table="market_behavior_examples_v1",
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        prompt_hash_column="prompt_hash",
        label_columns=["behavior_label", "asset_family"],
        case_key_column="market_case_id",
    )

    sample = dataset.select(limit=10_000)

    cap = capture_runner.run(
        CaptureSpec(
            engine=VLLMEngine(model_id="/models/Qwen/Qwen3-30B-A3B"),
            dataset=sample,
            sites=[
                ResidualSite(
                    name="resid_post_layers",
                    site="resid_post",
                    layers=[12, 16, 20, 24, 28, 32],
                    tokens=TokenSelector.last(),
                    storage=TensorStorage(dtype="float16"),
                )
            ],
        )
    )

    sae_features = analysis_runner.run(
        SAEEncodeSpec(
            feature=cap.feature("resid_post_layers").layer(24),
            sae=SAESource.release(
                model_id=base_model_id,
                release="qwen3_30b_resid_post",
                layer=24,
            ),
            top_k=64,
        )
    )

    selectivity = analysis_runner.run(
        FeatureSelectivitySpec(
            feature=sae_features,
            labels=sample.labels("behavior_label"),
            group_by=sample.cases("market_case_id"),
            tests=["mutual_information", "logistic_probe", "bootstrap_ci"],
        )
    )

    print(selectivity.table("top_features").head(20))


if __name__ == "__main__":
    main()
