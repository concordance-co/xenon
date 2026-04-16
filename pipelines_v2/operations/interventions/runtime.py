"""Internal runtime helpers for activation patching."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.data.datasets import Dataset, Example
from pipelines_v2.operations.common.builders import TransformResult
from pipelines_v2.operations.interventions import ActivationPatchControl, ActivationPatchSpec


def load_residual_source_feature(spec: ActivationPatchSpec) -> dict[str, Any]:
    feature = spec.source_feature
    if feature is None or not hasattr(feature, "load"):
        raise SpecValidationError("ActivationPatchSpec source_feature must be a feature ref")
    payload = feature.load()
    if not isinstance(payload, Mapping):
        raise TypeError("Activation patch source feature payload must be a mapping")
    if str(payload.get("kind") or "") != "residual":
        raise SpecValidationError("ActivationPatchSpec currently requires a residual source feature")
    if str(payload.get("site") or "") != str(spec.write_site.site):
        raise SpecValidationError(
            "ActivationPatchSpec source_feature site does not match write_site: "
            f"{payload.get('site')!r} vs {spec.write_site.site!r}"
        )
    layers = payload.get("layers")
    if not isinstance(layers, Mapping):
        raise TypeError("Residual source feature payload must contain a 'layers' mapping")
    missing = [int(layer) for layer in spec.write_site.layers if str(int(layer)) not in layers]
    if missing:
        raise SpecValidationError(
            "ActivationPatchSpec source_feature is missing required layers: "
            f"{sorted(missing)}"
        )
    return dict(payload)


def resolve_patch_cases(spec: ActivationPatchSpec) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset: Dataset = spec.dataset
    examples_by_key = {example.key: example for example in dataset.examples}
    dataset_keys = set(examples_by_key)
    if not dataset_keys:
        raise SpecValidationError("ActivationPatchSpec dataset resolved to no examples")

    if not hasattr(spec.pair_by, "resolve_values"):
        raise SpecValidationError("ActivationPatchSpec pair_by must resolve example_key -> case_key values")
    case_values = spec.pair_by.resolve_values()
    case_by_key = {
        str(key): str(value)
        for key, value in case_values.items()
        if str(key) in dataset_keys and value is not None
    }
    grouped: dict[str, list[Example]] = defaultdict(list)
    for key, case_key in case_by_key.items():
        grouped[case_key].append(examples_by_key[key])

    target_keys = _predicate_keys(spec.target_when, dataset_keys, label="target_when")
    donor_keys = _predicate_keys(spec.donor_when, dataset_keys, label="donor_when")
    control_keys = {
        control.name: _predicate_keys(control.donor_when, dataset_keys, label=f"controls[{control.name}]")
        for control in spec.controls
    }

    resolved: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for case_key in sorted(grouped):
        case_examples = grouped[case_key]
        target = _single_example(case_examples, target_keys, field="target_when")
        donor = _single_example(case_examples, donor_keys, field="donor_when")
        if isinstance(target, str):
            skipped.append({"case_key": case_key, "status": "skipped", "skip_reason": target})
            continue
        if isinstance(donor, str):
            skipped.append({"case_key": case_key, "status": "skipped", "skip_reason": donor})
            continue
        controls: dict[str, Example] = {}
        bad_control_reason: str | None = None
        for control in spec.controls:
            picked = _single_example(case_examples, control_keys[control.name], field=f"controls[{control.name}]")
            if isinstance(picked, str):
                bad_control_reason = picked
                break
            controls[control.name] = picked
        if bad_control_reason is not None:
            skipped.append({"case_key": case_key, "status": "skipped", "skip_reason": bad_control_reason})
            continue
        resolved.append(
            {
                "case_key": case_key,
                "target": target,
                "donor": donor,
                "controls": controls,
            }
        )
    return resolved, skipped


def evaluate_patch_row(
    *,
    spec: ActivationPatchSpec,
    example: Example,
    baseline: Mapping[str, Any],
    patched: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if spec.row_evaluator is None:
        raise SpecValidationError("ActivationPatchSpec requires row_evaluator")
    raw = spec.row_evaluator.build(
        {
            "example": example.to_dict(),
            "baseline": dict(baseline),
            "patched": dict(patched),
            "controls": {str(name): dict(payload) for name, payload in controls.items()},
        }
    )
    if isinstance(raw, TransformResult):
        return dict(raw.payload)
    if isinstance(raw, Mapping):
        return dict(raw)
    raise TypeError(
        "ActivationPatchSpec row_evaluator must return TransformResult or a mapping, "
        f"got {type(raw).__name__}"
    )


def aggregate_patch_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable_rows = [row for row in rows if str(row.get("status") or "ok") == "ok"]
    metrics: dict[str, dict[str, Any]] = {}
    for row in usable_rows:
        evaluation = row.get("evaluation")
        if not isinstance(evaluation, Mapping):
            continue
        raw_metrics = evaluation.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            continue
        for name, value in raw_metrics.items():
            normalized = _metric_value(value)
            if normalized is None:
                continue
            entry = metrics.setdefault(
                str(name),
                {
                    "count": 0,
                    "sum": 0.0,
                    "kind": "boolean_rate" if isinstance(value, bool) else "mean",
                },
            )
            entry["count"] += 1
            entry["sum"] += normalized
            if isinstance(value, bool):
                entry["kind"] = "boolean_rate"
    for entry in metrics.values():
        count = max(1, int(entry["count"]))
        entry["value"] = float(entry["sum"] / count)
        del entry["sum"]
    return {
        "example_count": len(rows),
        "patched_count": len(usable_rows),
        "skipped_count": len(rows) - len(usable_rows),
        "control_names": sorted(
            {
                str(name)
                for row in rows
                for name in dict(row.get("controls") or {}).keys()
            }
        ),
        "metrics": metrics,
    }


def control_for_name(
    controls: tuple[ActivationPatchControl, ...],
    name: str,
) -> ActivationPatchControl | None:
    for control in controls:
        if control.name == name:
            return control
    return None


def _predicate_keys(predicate: Any, dataset_keys: set[str], *, label: str) -> set[str]:
    if predicate is None or not hasattr(predicate, "resolve_example_keys"):
        raise SpecValidationError(f"ActivationPatchSpec requires {label} to resolve example keys")
    return {str(key) for key in predicate.resolve_example_keys() if str(key) in dataset_keys}


def _single_example(case_examples: list[Example], allowed_keys: set[str], *, field: str) -> Example | str:
    matches = [example for example in case_examples if example.key in allowed_keys]
    if len(matches) != 1:
        return f"{field} must resolve exactly one example per case"
    return matches[0]


def _metric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    return None


__all__ = [
    "aggregate_patch_rows",
    "control_for_name",
    "evaluate_patch_row",
    "load_residual_source_feature",
    "resolve_patch_cases",
]
