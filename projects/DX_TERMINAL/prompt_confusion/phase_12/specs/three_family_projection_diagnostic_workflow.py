from __future__ import annotations

"""Projection / threshold diagnostic for the three-family geometry result."""

from pathlib import Path
from typing import Any

import numpy as np

from pipelines_v2.api import (
    LocalArtifactStore,
    LocalRunnerSpec,
    NullCatalog,
    ReportSpec,
    StepRef,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)
from pipelines_v2.operations.execution.common import feature_matrices, ordered_values

from projects.DX_TERMINAL.prompt_confusion.phase_12.specs.three_family_geometry_workflow import (
    CAPTURED_LAYERS,
    build_dataset,
    build_runner_specs as _build_base_runner_specs,
    _default_residual_engine,
)
from pipelines_v2.api import CaptureSpec, ResidualSite, TensorStorage, TokenSelector, VLLMEngine


DEFAULT_REPORT_DIR = "projects/DX_TERMINAL/prompt_confusion/phase_12/reports/three_family_projection_diagnostic"
SELECTED_LAYERS = (28, 36, 40, 44)
FAMILY_ORDER = ("trade_size", "risk_preference", "diversification_preference")


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    positives = y_true == 1
    negatives = y_true == 0
    tpr = float((y_pred[positives] == 1).mean()) if positives.any() else 0.0
    tnr = float((y_pred[negatives] == 0).mean()) if negatives.any() else 0.0
    return 0.5 * (tpr + tnr)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    q10, q50, q90 = np.quantile(values, [0.1, 0.5, 0.9])
    return {"p10": float(q10), "p50": float(q50), "p90": float(q90)}


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]
    if positives.size == 0 or negatives.size == 0:
        return 0.5
    comparisons = (positives[:, None] > negatives[None, :]).mean()
    ties = (positives[:, None] == negatives[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def _family_projection_summary(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    aligned = scores[labels == 0]
    conflict = scores[labels == 1]
    payload = {
        "count": int(scores.size),
        "overall_mean": float(scores.mean()),
        "overall_std": float(scores.std()),
        "aligned_mean": float(aligned.mean()) if aligned.size else 0.0,
        "aligned_std": float(aligned.std()) if aligned.size else 0.0,
        "conflict_mean": float(conflict.mean()) if conflict.size else 0.0,
        "conflict_std": float(conflict.std()) if conflict.size else 0.0,
    }
    payload.update({f"overall_{key}": value for key, value in _quantiles(scores).items()})
    payload.update({f"aligned_{key}": value for key, value in _quantiles(aligned).items()})
    payload.update({f"conflict_{key}": value for key, value in _quantiles(conflict).items()})
    return payload


def _projection_diagnostic_builder(
    capture: Any,
    target_dimension: Any,
    conflict_present: Any,
) -> TransformResult:
    matrices, example_keys = feature_matrices(
        capture.feature("residual_prompt_eos"),
        layers=SELECTED_LAYERS,
    )
    family_values = np.asarray(ordered_values(target_dimension, example_keys, label="target_dimension"), dtype=object)
    conflict_values = np.asarray(
        [1 if bool(value) else 0 for value in ordered_values(conflict_present, example_keys, label="conflict_present")],
        dtype=np.int64,
    )

    layers_payload: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for layer in SELECTED_LAYERS:
        X = matrices[int(layer)]
        directions: dict[str, np.ndarray] = {}
        thresholds: dict[str, float] = {}
        training_stats: dict[str, Any] = {}

        for family in FAMILY_ORDER:
            family_mask = family_values == family
            pos_mask = family_mask & (conflict_values == 1)
            neg_mask = family_mask & (conflict_values == 0)
            pos = X[pos_mask]
            neg = X[neg_mask]
            vector = pos.mean(axis=0) - neg.mean(axis=0)
            norm = float(np.linalg.norm(vector))
            unit = vector / norm if norm > 0 else vector
            scores = X @ unit
            pos_mean = float(scores[pos_mask].mean())
            neg_mean = float(scores[neg_mask].mean())
            if pos_mean < neg_mean:
                unit = -unit
                scores = -scores
                pos_mean = float(scores[pos_mask].mean())
                neg_mean = float(scores[neg_mask].mean())
            threshold = 0.5 * (pos_mean + neg_mean)

            directions[family] = unit.astype(np.float32)
            thresholds[family] = float(threshold)
            training_stats[family] = {
                "vector_norm": norm,
                "threshold": float(threshold),
                "aligned_mean": neg_mean,
                "conflict_mean": pos_mean,
                "within_family_auroc": _auroc(conflict_values[family_mask], scores[family_mask]),
                "within_family_balanced_accuracy": float(
                    _balanced_accuracy(conflict_values[family_mask], (scores[family_mask] >= threshold).astype(np.int64))
                ),
            }

        pairwise_cosines: dict[str, float] = {}
        for left in FAMILY_ORDER:
            for right in FAMILY_ORDER:
                if left >= right:
                    continue
                cosine = float(np.dot(directions[left], directions[right]))
                pairwise_cosines[f"{left}__vs__{right}"] = cosine

        direction_payloads: dict[str, Any] = {}
        for direction_family in FAMILY_ORDER:
            unit = directions[direction_family]
            threshold = thresholds[direction_family]
            scores = X @ unit
            eval_payloads: dict[str, Any] = {}
            train_family_overall_mean = float(scores[family_values == direction_family].mean())

            for eval_family in FAMILY_ORDER:
                eval_mask = family_values == eval_family
                eval_scores = scores[eval_mask]
                eval_labels = conflict_values[eval_mask]
                preds = (eval_scores >= threshold).astype(np.int64)
                tn = int(((preds == 0) & (eval_labels == 0)).sum())
                tp = int(((preds == 1) & (eval_labels == 1)).sum())
                fn = int(((preds == 0) & (eval_labels == 1)).sum())
                fp = int(((preds == 1) & (eval_labels == 0)).sum())
                neg_mask = eval_labels == 0
                pos_mask = eval_labels == 1
                fpr = float((preds[neg_mask] == 1).mean()) if neg_mask.any() else 0.0
                fnr = float((preds[pos_mask] == 0).mean()) if pos_mask.any() else 0.0

                projection_summary = _family_projection_summary(eval_scores, eval_labels)
                eval_payloads[eval_family] = {
                    "accuracy": float((preds == eval_labels).mean()),
                    "balanced_accuracy": float(_balanced_accuracy(eval_labels, preds)),
                    "auroc": _auroc(eval_labels, eval_scores),
                    "threshold": float(threshold),
                    "threshold_minus_overall_mean": float(threshold - projection_summary["overall_mean"]),
                    "shift_vs_training_family_mean": float(projection_summary["overall_mean"] - train_family_overall_mean),
                    "false_positive_count": fp,
                    "false_negative_count": fn,
                    "true_positive_count": tp,
                    "true_negative_count": tn,
                    "false_positive_rate": fpr,
                    "false_negative_rate": fnr,
                    "projection_summary": projection_summary,
                }
                summary_rows.append(
                    {
                        "layer": int(layer),
                        "direction_family": direction_family,
                        "eval_family": eval_family,
                        "threshold": float(threshold),
                        "balanced_accuracy": float(_balanced_accuracy(eval_labels, preds)),
                        "auroc": _auroc(eval_labels, eval_scores),
                        "shift_vs_training_family_mean": float(projection_summary["overall_mean"] - train_family_overall_mean),
                        "false_positive_rate": fpr,
                        "false_negative_rate": fnr,
                    }
                )

            direction_payloads[direction_family] = {
                "training_family_stats": training_stats[direction_family],
                "eval_families": eval_payloads,
            }

        layers_payload[str(layer)] = {
            "direction_similarity": pairwise_cosines,
            "directions": direction_payloads,
        }

    summary = {
        "layers": list(SELECTED_LAYERS),
        "families": list(FAMILY_ORDER),
        "row_count": len(example_keys),
        "summary_rows": summary_rows,
    }
    return TransformResult(
        payload={
            "kind": "projection_diagnostic_result",
            "feature": "residual_prompt_eos",
            "layers": layers_payload,
            "summary": summary,
        },
        example_keys=example_keys,
    )


def build_workflow(
    dataset=None,
    *,
    residual_engine: VLLMEngine | None = None,
    report_output_dir: str = DEFAULT_REPORT_DIR,
) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    residual_engine = residual_engine or _default_residual_engine()

    return WorkflowSpec(
        name="prompt_confusion_three_family_projection_diagnostic",
        steps=(
            WorkflowStep(
                name="capture_prompt_eos_residual",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=residual_engine,
                    dataset=dataset,
                    sites=[
                        ResidualSite(
                            name="residual_prompt_eos",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.last(),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ],
                ),
            ),
            WorkflowStep(
                name="projection_diagnostic",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        _projection_diagnostic_builder,
                        local_python_sources=(".",),
                    ),
                    inputs={
                        "capture": StepRef("capture_prompt_eos_residual"),
                        "target_dimension": dataset.labels("target_dimension"),
                        "conflict_present": dataset.labels("conflict_present"),
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(StepRef("projection_diagnostic"),),
                    template="default",
                    output_dir=report_output_dir,
                ),
            ),
        ),
    )


def build_runner_specs_for_workflow() -> dict[str, object]:
    runners = _build_base_runner_specs()
    runners["report_local"] = LocalRunnerSpec(
        artifacts=LocalArtifactStore(Path("artifacts") / "prompt_confusion_three_family_projection_diagnostic"),
        catalog=NullCatalog(),
    )
    return runners


# Keep the CLI loader contract consistent with the other workflow files.
def build_runner_specs() -> dict[str, object]:
    return build_runner_specs_for_workflow()
