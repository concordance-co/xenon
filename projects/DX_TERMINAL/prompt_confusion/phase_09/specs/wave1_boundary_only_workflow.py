from __future__ import annotations

"""Boundary-only workflow for validating Phase 09 Wave 1 generation behavior."""

from pipelines_v2.api import CaptureSpec, GenerationSpec, WorkflowSpec, WorkflowStep

from projects.DX_TERMINAL.prompt_confusion.phase_09.specs.wave1_workflow import (
    _boundary_generation_engine,
    build_boundary_dataset,
    build_runner_specs,
)


def build_dataset():
    return build_boundary_dataset()


def build_workflow(dataset=None) -> WorkflowSpec:
    dataset = dataset or build_boundary_dataset()
    return WorkflowSpec(
        name="prompt_confusion_phase_09_wave1_boundary_only",
        steps=(
            WorkflowStep(
                name="boundary_generation",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=_boundary_generation_engine(),
                    dataset=dataset,
                    sites=[],
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=256,
                        temperature=0.0,
                        capture_reasoning=False,
                    ),
                ),
            ),
        ),
    )

