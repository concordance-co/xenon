"""Execution for truthfulness / residual ITI-style specs."""

from __future__ import annotations

from pipelines_v2.mechinterp._directional import (
    annotate_direction_payload,
    direction_payload_to_subspace,
    select_direction_by_projection_gap,
)
from pipelines_v2.operations.execution.common import OperationExecutionResult
from pipelines_v2.operations.execution.projections import run_projection
from pipelines_v2.operations.execution.representation import run_direction
from pipelines_v2.operations.projections import ProjectionSpec
from pipelines_v2.operations.representation import DirectionSpec

from .specs import (
    TruthfulnessAblationSubspaceSpec,
    TruthfulnessDirectionSelectionSpec,
    TruthfulnessDirectionSpec,
    TruthfulnessScoreSpec,
)


def run_truthfulness_direction(spec: TruthfulnessDirectionSpec) -> OperationExecutionResult:
    """Compute truthful-minus-untruthful residual directions."""

    result = run_direction(
        DirectionSpec(
            feature=spec.feature,
            positive=spec.truthful_when,
            negative=spec.untruthful_when,
            group_by=spec.group_by,
            layers=spec.layers,
            tokens=spec.tokens,
            pooling=spec.pooling,
        )
    )
    return OperationExecutionResult(
        payload=annotate_direction_payload(
            result.payload,
            name=spec.name,
            method="truthfulness_direction_spec",
            metadata={
                "paper": "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model",
                "formula": "mean(truthful_answer_activations) - mean(untruthful_answer_activations)",
                "intervention_scope": "residual_direction_smoke; attention-head ITI is a follow-up surface",
                **dict(spec.metadata),
            },
            summary={"positive_label": "truthful", "negative_label": "untruthful"},
        ),
        example_coverage=result.example_coverage,
    )


def run_truthfulness_score(spec: TruthfulnessScoreSpec) -> OperationExecutionResult:
    """Score activation slices against a truthfulness direction."""

    result = run_projection(
        ProjectionSpec(
            feature=spec.feature,
            coordinates=(spec.direction,),
            slices=spec.slices,
            rows=spec.rows,
            layers=spec.layers,
            pooling=spec.pooling,
            metric=spec.metric,
            summaries=spec.summaries,
            emit_labels=spec.emit_labels,
        )
    )
    payload = dict(result.payload)
    payload["kind"] = "truthfulness_score_result"
    payload["truthfulness_direction"] = {
        "metric": spec.metric,
        "summaries": list(spec.summaries),
    }
    return OperationExecutionResult(
        payload=payload,
        labels=result.labels,
        metadata=result.metadata,
        example_coverage=result.example_coverage,
    )


def run_truthfulness_direction_selection(spec: TruthfulnessDirectionSelectionSpec) -> OperationExecutionResult:
    """Select the truthfulness direction layer by validation projection gap."""

    return select_direction_by_projection_gap(
        direction=spec.direction,
        scores=spec.scores,
        positive_when=spec.truthful_when,
        negative_when=spec.untruthful_when,
        layers=spec.layers,
        summary_metric=spec.summary_metric,
        name=spec.name,
        method="truthfulness_direction_selection_spec",
        metadata={
            "paper": "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model",
            **dict(spec.metadata),
        },
    )


def run_truthfulness_ablation_subspace(spec: TruthfulnessAblationSubspaceSpec) -> OperationExecutionResult:
    """Convert selected truthfulness direction to a ProjectOutPatch-compatible subspace."""

    return direction_payload_to_subspace(
        spec.direction,
        layers=spec.layers,
        name=spec.name,
        method="truthfulness_ablation_subspace_spec",
        metadata={
            "paper": "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model",
            **dict(spec.metadata),
        },
    )


__all__ = [
    "run_truthfulness_ablation_subspace",
    "run_truthfulness_direction",
    "run_truthfulness_direction_selection",
    "run_truthfulness_score",
]
