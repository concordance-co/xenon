from __future__ import annotations

import json
from pathlib import Path

from pipelines_v2.api import CaptureSpec, Dataset, GenerationSpec, ResidualSite, TensorStorage, TokenSelector
from pipelines_v2.storage.artifacts import OperationArtifact, artifact_from_manifest

from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base


GENERATION_ARTIFACT_ID = "generation_run_1_b7d85acbea54"
REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_name_only_repair_capture")


def _load_generation_artifact() -> OperationArtifact:
    runner_specs = base.build_runner_specs()
    store = runner_specs["analysis_cpu"].artifacts
    catalog = runner_specs["analysis_cpu"].catalog
    manifest = catalog.load_artifact(GENERATION_ARTIFACT_ID)
    if manifest is None:
        raise RuntimeError(f"Could not load generation artifact {GENERATION_ARTIFACT_ID!r}")
    artifact = artifact_from_manifest(manifest, store=store)
    if not isinstance(artifact, OperationArtifact):
        raise TypeError(f"Artifact {GENERATION_ARTIFACT_ID!r} is not an operation artifact")
    return artifact


def main() -> None:
    generation = _load_generation_artifact()
    repaired = base.build_theory_persistence_capture_dataset(generation=generation)
    payload = dict(repaired["payload"])
    dataset_payload = dict(payload["dataset"])
    summary = dict(payload.get("summary", {}))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_json_path = REPORT_DIR / "repaired_capture_dataset.json"
    summary_json_path = REPORT_DIR / "repaired_capture_summary.json"
    dataset_json_path.write_text(json.dumps(dataset_payload, indent=2), encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    dataset = Dataset.from_dict(dataset_payload)
    capture_spec = CaptureSpec(
        engine=base._engine(max_num_seqs=8),
        dataset=dataset,
        sites=[
            ResidualSite(
                name="generated_sequence_residual",
                site="resid_post",
                layers=list(base.CAPTURED_LAYERS),
                tokens=TokenSelector.section("generated"),
                storage=TensorStorage(dtype="float16", format="safetensors"),
            )
        ],
        generation=GenerationSpec(enabled=False),
    )
    runner = base.build_runner_specs()["capture_gpu"].to_runner()
    capture_artifact = runner.run(capture_spec)

    report = {
        "generation_artifact_id": GENERATION_ARTIFACT_ID,
        "capture_dataset_json": str(dataset_json_path),
        "capture_summary_json": str(summary_json_path),
        "capture_artifact_id": capture_artifact.id,
        "capture_artifact_kind": capture_artifact.manifest().artifact_kind,
        "kept_capture_example_count": summary.get("kept_capture_example_count"),
        "flagged_direct_copy_count": summary.get("flagged_direct_copy_count"),
        "prime_condition_counts": summary.get("prime_condition_counts"),
    }
    report_path = REPORT_DIR / "repair_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
