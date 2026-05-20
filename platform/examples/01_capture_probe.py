"""ARCH expected usage: capture activations on GPU and train probes on CPU."""

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    GenerationSpec,
    ModalResources,
    ModalRunner,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    MoERoutingSite,
    PostgresCatalog,
    PostgresSource,
    ProbeSpec,
    ResidualSite,
    RoutingRecord,
    TensorStorage,
    TokenSelector,
    VLLMEngine,
)


def main() -> None:
    db = PostgresSource.from_env("XENON_NEON_DATABASE_URL")
    artifact_store = ModalVolumeStore(name="xenon-data", root="/data/artifacts")
    catalog = PostgresCatalog(source=db)

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
        table="conflict_examples_v2",
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        prompt_hash_column="prompt_hash",
        label_columns=["conflict_label"],
        case_key_column="matched_pair_id",
    )

    cap = capture_runner.run(
        CaptureSpec(
            engine=VLLMEngine(
                model_id="/models/Qwen/Qwen3-30B-A3B",
                max_model_len=8192,
                enforce_eager=False,
                max_num_seqs=8,
                enable_prefix_caching=False,
            ),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_post_last",
                    site="resid_post",
                    layers=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
                    tokens=TokenSelector.last(),
                    storage=TensorStorage(dtype="float16"),
                ),
                MoERoutingSite(
                    name="moe_routing_last",
                    layers=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
                    tokens=TokenSelector.last(),
                    record=[
                        RoutingRecord.gate_logits(dtype="float16"),
                        RoutingRecord.routing_decisions(required=False),
                        RoutingRecord.topk_from_gate(k=8, include_weights=True),
                        RoutingRecord.expert_load(source="topk_from_gate"),
                    ],
                ),
            ],
            generation=GenerationSpec(
                enabled=True,
                max_tokens=512,
                temperature=0.0,
                capture_reasoning=False,
            ),
        )
    )

    probe = analysis_runner.run(
        ProbeSpec(
            feature=cap.feature("resid_post_last"),
            labels=dataset.labels("conflict_label"),
            group_by=dataset.cases("matched_pair_id"),
            folds=5,
            baselines=["majority", "shuffled_label"],
            metrics=["accuracy", "balanced_accuracy", "selectivity"],
        )
    )

    print(probe.summary())


if __name__ == "__main__":
    main()
