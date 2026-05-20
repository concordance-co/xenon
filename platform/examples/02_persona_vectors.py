"""Fit a contrast direction from a local capture artifact."""

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    DirectionSpec,
    Example,
    FileCatalog,
    LocalArtifactStore,
    LocalRunner,
    ResidualSite,
    SubspaceSpec,
    TensorStorage,
    TokenSelector,
    ToyEngine,
)


def main() -> None:
    capture_runner = LocalRunner(
        artifacts=LocalArtifactStore(root="artifacts/persona_vectors"),
        catalog=FileCatalog(root="artifacts/persona_vectors/catalog"),
    )

    analysis_runner = LocalRunner(
        artifacts=LocalArtifactStore(root="artifacts/persona_vectors"),
        catalog=FileCatalog(root="artifacts/persona_vectors/catalog"),
    )

    dataset = Dataset.from_examples(
        (
            Example(
                key="pair_1_risk_seeking",
                prompt="Choose the high-variance strategy with large upside.",
                labels={"risk_style": "risk_seeking"},
                case_key="pair_1",
            ),
            Example(
                key="pair_1_risk_averse",
                prompt="Choose the conservative strategy with stable downside.",
                labels={"risk_style": "risk_averse"},
                case_key="pair_1",
            ),
            Example(
                key="pair_2_risk_seeking",
                prompt="Prefer the experimental plan if it can unlock outsized gains.",
                labels={"risk_style": "risk_seeking"},
                case_key="pair_2",
            ),
            Example(
                key="pair_2_risk_averse",
                prompt="Prefer the proven plan if it keeps losses bounded.",
                labels={"risk_style": "risk_averse"},
                case_key="pair_2",
            ),
        ),
        name="persona_vector_example",
    )

    cap = capture_runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=8, num_layers=4, sequence_length=8),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_post_full",
                    site="resid_post",
                    layers=[1, 2],
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
            layers=[1, 2],
            tokens=TokenSelector.last(),
        )
    )

    subspace = analysis_runner.run(
        SubspaceSpec(
            feature=cap.feature("resid_post_full"),
            layers=[1, 2],
            components=2,
            tokens=TokenSelector.last(),
        )
    )

    print(persona_vector.summary())
    print(subspace.summary())


if __name__ == "__main__":
    main()
