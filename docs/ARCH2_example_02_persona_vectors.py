"""ARCH2 expected usage: fit persona vectors from a local capture artifact."""

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    DirectionEvalSpec,
    DirectionSpec,
    FileCatalog,
    LocalArtifactStore,
    LocalResources,
    LocalRunner,
    ResidualSite,
    TensorStorage,
    TokenSelector,
    TransformersEngine,
)


def main() -> None:
    capture_runner = LocalRunner(
        resources=LocalResources(device="mps"),
        artifacts=LocalArtifactStore(root="artifacts/persona_vectors"),
        catalog=FileCatalog(root="artifacts/persona_vectors/catalog"),
    )

    analysis_runner = LocalRunner(
        artifacts=LocalArtifactStore(root="artifacts/persona_vectors"),
        catalog=FileCatalog(root="artifacts/persona_vectors/catalog"),
    )

    dataset = Dataset.from_parquet(
        path="projects/persona/examples/persona_contrast.parquet",
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        prompt_hash_column="prompt_hash",
        label_columns=["persona_family", "risk_style"],
        case_key_column="contrast_case_id",
    )

    cap = capture_runner.run(
        CaptureSpec(
            engine=TransformersEngine(model_id="Qwen/Qwen3-8B"),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_post_full",
                    site="resid_post",
                    layers=[8, 16, 24, 32],
                    tokens=TokenSelector.full_sequence(),
                    storage=TensorStorage(dtype="float16"),
                )
            ],
        )
    )

    persona_vector = analysis_runner.run(
        DirectionSpec(
            feature=cap.feature("resid_post_full"),
            positive=dataset.labels("risk_style").equals("risk_seeking"),
            negative=dataset.labels("risk_style").equals("risk_averse"),
            group_by=dataset.cases("contrast_case_id"),
            layers=[16, 24],
            tokens=TokenSelector.last(),
        )
    )

    eval_artifact = analysis_runner.run(
        DirectionEvalSpec(
            direction=persona_vector,
            feature=cap.feature("resid_post_full"),
            labels=dataset.labels("risk_style"),
            group_by=dataset.cases("contrast_case_id"),
            metrics=["auc", "effect_size", "bootstrap_ci"],
        )
    )

    print(eval_artifact.summary())


if __name__ == "__main__":
    main()
