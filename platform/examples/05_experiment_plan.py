"""ARCH expected usage: orchestrate a heterogeneous workflow."""

from pipelines_v2.api import (
    BasisSpec,
    CaptureSpec,
    Dataset,
    LocalRunner,
    ModalResources,
    ModalRunner,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PostgresCatalog,
    PostgresSource,
    ProbeSpec,
    ReportSpec,
    ResidualSite,
    StepRef,
    VLLMEngine,
    WorkflowOrchestrator,
    WorkflowSpec,
    WorkflowStep,
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

    report_runner = LocalRunner()

    dataset = Dataset.from_postgres(
        source=db,
        table="conflict_examples_v2",
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        prompt_hash_column="prompt_hash",
        label_columns=["conflict_label", "risk_style"],
        case_key_column="matched_pair_id",
    )

    orchestrator = WorkflowOrchestrator(
        runners={
            "capture_gpu": capture_runner,
            "analysis_cpu": analysis_runner,
            "report_local": report_runner,
        }
    )

    artifacts = orchestrator.run(
        WorkflowSpec(
            name="conflict_mechanism_sweep_v1",
            steps=(
                WorkflowStep(
                    name="capture",
                    runner="capture_gpu",
                    spec=CaptureSpec(
                        engine=VLLMEngine(model_id="/models/Qwen/Qwen3-30B-A3B", max_model_len=8192),
                        dataset=dataset,
                        sites=[
                            ResidualSite(
                                name="resid_post_last",
                                site="resid_post",
                                layers=[16, 20, 24],
                            )
                        ],
                    ),
                ),
                WorkflowStep(
                    name="probe_conflict",
                    runner="analysis_cpu",
                    spec=ProbeSpec(
                        feature=StepRef("capture").feature("resid_post_last"),
                        labels=dataset.labels("conflict_label"),
                        group_by=dataset.cases("matched_pair_id"),
                        folds=5,
                    ),
                    depends_on=("capture",),
                ),
                WorkflowStep(
                    name="basis_conflict",
                    runner="analysis_cpu",
                    spec=BasisSpec(
                        feature=StepRef("capture").feature("resid_post_last"),
                        method="pca",
                        by=dataset.labels("conflict_label"),
                        layers=[16, 20, 24],
                        components=8,
                    ),
                    depends_on=("capture",),
                ),
                WorkflowStep(
                    name="report",
                    runner="report_local",
                    spec=ReportSpec(
                        inputs=[
                            StepRef("capture"),
                            StepRef("probe_conflict"),
                            StepRef("basis_conflict"),
                        ],
                        template="mechanism_summary",
                        output_dir="projects/DX_TERMINAL/prompt_confusion/reports",
                    ),
                    depends_on=("probe_conflict", "basis_conflict"),
                ),
            ),
        )
    )

    print(artifacts.step("report").uri)


if __name__ == "__main__":
    main()
