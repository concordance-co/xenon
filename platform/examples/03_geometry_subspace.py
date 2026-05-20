"""Capture activations locally, then build geometry and subspace artifacts."""

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    Example,
    FileCatalog,
    GeometrySpec,
    LocalArtifactStore,
    LocalRunner,
    ResidualSite,
    SubspaceSpec,
    TensorStorage,
    TokenSelector,
    ToyEngine,
)


def main() -> None:
    runner = LocalRunner(
        artifacts=LocalArtifactStore(root="artifacts/geometry_subspace"),
        catalog=FileCatalog(root="artifacts/geometry_subspace/catalog"),
    )

    dataset = Dataset.from_examples(
        (
            Example(key="safe_1", prompt="Use the verified fallback path.", labels={"style": "safe"}, case_key="case_1"),
            Example(key="safe_2", prompt="Prefer a reversible migration.", labels={"style": "safe"}, case_key="case_2"),
            Example(key="bold_1", prompt="Ship the aggressive optimization.", labels={"style": "bold"}, case_key="case_1"),
            Example(key="bold_2", prompt="Try the risky new routing plan.", labels={"style": "bold"}, case_key="case_2"),
        ),
        name="geometry_subspace_example",
    )

    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=8, num_layers=4, sequence_length=8),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_post_last",
                    site="resid_post",
                    layers=[1, 2],
                    tokens=TokenSelector.last(),
                    storage=TensorStorage(dtype="float16"),
                )
            ],
        )
    )

    geometry = runner.run(
        GeometrySpec(
            feature=capture.feature("resid_post_last"),
            rows=dataset,
            method="pca",
            layers=[1, 2],
            label=dataset.labels("style"),
            color_by={"case": dataset.cases()},
            components=2,
            tokens=TokenSelector.last(),
        )
    )

    subspace = runner.run(
        SubspaceSpec(
            feature=capture.feature("resid_post_last"),
            layers=[1, 2],
            components=2,
            tokens=TokenSelector.last(),
            named_components_by_layer={
                1: {"style_axis": 0},
                2: {"style_axis": 0},
            },
        )
    )

    print(geometry.summary())
    print(subspace.summary())


if __name__ == "__main__":
    main()
