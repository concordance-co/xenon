from __future__ import annotations

from pathlib import Path

from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base


DATASET_PATH = Path("projects/MOREBENCH/phase_03/outputs/experiment_02_name_only_generation_dataset.jsonl")
REPORT_OUTPUT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_name_only_report")


def build_dataset():
    return base.build_dataset(
        dataset_path=DATASET_PATH,
        dataset_name="morebench_phase03_experiment02_name_only_generation_batch",
    )


def build_runner_specs():
    return base.build_runner_specs()


def build_workflow():
    return base.build_workflow(
        dataset=build_dataset(),
        workflow_name="morebench_phase03_experiment02_name_only_generation_persistence",
        report_output_dir=REPORT_OUTPUT_DIR,
        generation_description=(
            "Generate one response for every matched dilemma under five name-only theory primes "
            "plus one generic ethics control. This is the broad batch used for within-family and "
            "cross-family generation-time persistence comparisons."
        ),
    )
