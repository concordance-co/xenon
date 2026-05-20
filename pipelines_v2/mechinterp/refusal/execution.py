"""Execution for refusal-direction mech-interp specs."""

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
    RefusalAblationSubspaceSpec,
    RefusalDirectionSelectionSpec,
    RefusalDirectionSpec,
    RefusalScoreSpec,
)


def run_refusal_direction(spec: RefusalDirectionSpec) -> OperationExecutionResult:
    """Compute harmful-minus-harmless direction vectors."""

    result = run_direction(
        DirectionSpec(
            feature=spec.feature,
            positive=spec.harmful_when,
            negative=spec.harmless_when,
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
            method="refusal_direction_spec",
            metadata={
                "paper": "Refusal in Language Models Is Mediated by a Single Direction",
                "formula": "mean(harmful_prompt_activations) - mean(harmless_prompt_activations)",
                **dict(spec.metadata),
            },
            summary={"positive_label": "harmful", "negative_label": "harmless"},
        ),
        example_coverage=result.example_coverage,
    )


def run_refusal_score(spec: RefusalScoreSpec) -> OperationExecutionResult:
    """Score activation slices against a refusal direction."""

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
    payload["kind"] = "refusal_score_result"
    payload["refusal_direction"] = {
        "metric": spec.metric,
        "summaries": list(spec.summaries),
    }
    return OperationExecutionResult(
        payload=payload,
        labels=result.labels,
        metadata=result.metadata,
        example_coverage=result.example_coverage,
    )


def run_refusal_direction_selection(spec: RefusalDirectionSelectionSpec) -> OperationExecutionResult:
    """Select the refusal direction layer by validation projection gap."""

    return select_direction_by_projection_gap(
        direction=spec.direction,
        scores=spec.scores,
        positive_when=spec.harmful_when,
        negative_when=spec.harmless_when,
        layers=spec.layers,
        summary_metric=spec.summary_metric,
        name=spec.name,
        method="refusal_direction_selection_spec",
        metadata={
            "paper": "Refusal in Language Models Is Mediated by a Single Direction",
            **dict(spec.metadata),
        },
    )


def run_refusal_ablation_subspace(spec: RefusalAblationSubspaceSpec) -> OperationExecutionResult:
    """Convert selected refusal direction to a ProjectOutPatch-compatible subspace."""

    return direction_payload_to_subspace(
        spec.direction,
        layers=spec.layers,
        name=spec.name,
        method="refusal_ablation_subspace_spec",
        metadata={
            "paper": "Refusal in Language Models Is Mediated by a Single Direction",
            **dict(spec.metadata),
        },
    )


__all__ = [
    "run_refusal_ablation_subspace",
    "run_refusal_direction",
    "run_refusal_direction_selection",
    "run_refusal_score",
]
