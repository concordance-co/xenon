"""Example and case resolution for intervention workflows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.data.datasets import Dataset, Example
from pipelines_v2.operations.common.tokens import TokenSelector

from ._runtime_sources import required_source_layers_for_patch
from .generation import GenerationRunSpec, PatchedGenerationSpec
from .recipes import InterchangePatch, ResidualPathPatch


def resolve_generation_examples(spec: GenerationRunSpec) -> list[Example]:
    dataset: Dataset = spec.dataset
    examples = list(dataset.examples)
    if not examples:
        raise SpecValidationError("GenerationRunSpec dataset resolved to no examples")
    if spec.select_when is None:
        return examples
    if not hasattr(spec.select_when, "resolve_example_keys"):
        raise SpecValidationError("GenerationRunSpec select_when must support resolve_example_keys()")
    allowed = {str(key) for key in spec.select_when.resolve_example_keys()}
    selected = [example for example in examples if example.key in allowed]
    if not selected:
        raise SpecValidationError("GenerationRunSpec select_when did not match any dataset examples")
    return selected


def resolve_patched_generation_targets(spec: PatchedGenerationSpec) -> list[Example]:
    dataset: Dataset = spec.dataset
    examples = list(dataset.examples)
    if not examples:
        raise SpecValidationError("PatchedGenerationSpec dataset resolved to no examples")
    if spec.select_when is None:
        return examples
    if not hasattr(spec.select_when, "resolve_example_keys"):
        raise SpecValidationError("PatchedGenerationSpec select_when must support resolve_example_keys()")
    allowed = {str(key) for key in spec.select_when.resolve_example_keys()}
    selected = [example for example in examples if example.key in allowed]
    if not selected:
        raise SpecValidationError("PatchedGenerationSpec select_when did not match any dataset examples")
    return selected


def resolve_patched_generation_cases(
    spec: PatchedGenerationSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset: Dataset = spec.dataset
    examples_by_key = {example.key: example for example in dataset.examples}
    dataset_keys = set(examples_by_key)
    if not dataset_keys:
        raise SpecValidationError("PatchedGenerationSpec dataset resolved to no examples")

    if not hasattr(spec.pair_by, "resolve_values"):
        raise SpecValidationError("PatchedGenerationSpec pair_by must resolve example_key -> case_key values")
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
        resolved.append({"case_key": case_key, "target": target, "donor": donor})
    return resolved, skipped


def partition_cases_by_activation_bank(
    *,
    spec: PatchedGenerationSpec,
    activation_bank: Mapping[str, Any],
    resolved_cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    usable: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in resolved_cases:
        issue = activation_bank_case_issue(spec=spec, activation_bank=activation_bank, item=item)
        if issue is None:
            usable.append(item)
            continue
        skipped.append(
            {
                "case_key": str(item["case_key"]),
                "status": "skipped",
                "skip_reason": issue,
            }
        )
    return usable, skipped


def selector_requires_sections(selector: TokenSelector | None) -> bool:
    return isinstance(selector, TokenSelector) and selector.kind == "section"


def activation_bank_case_errors(
    *,
    spec: PatchedGenerationSpec,
    activation_bank: Mapping[str, Any],
    resolved_cases: list[dict[str, Any]],
) -> list[str]:
    issues: dict[str, list[str]] = defaultdict(list)
    for item in resolved_cases:
        issue = activation_bank_case_issue(spec=spec, activation_bank=activation_bank, item=item)
        if issue is None:
            continue
        issues[issue].append(str(item["case_key"]))
    errors: list[str] = []
    for issue, case_keys in sorted(issues.items()):
        sample = ", ".join(case_keys[:5])
        suffix = " ..." if len(case_keys) > 5 else ""
        errors.append(f"{issue}; cases={sample}{suffix}")
    return errors


def activation_bank_case_issue(
    *,
    spec: PatchedGenerationSpec,
    activation_bank: Mapping[str, Any],
    item: Mapping[str, Any],
) -> str | None:
    layers_payload = activation_bank.get("layers")
    if not isinstance(layers_payload, Mapping):
        return "activation_bank source is missing a 'layers' mapping"
    donor = item.get("donor")
    donor_key = str(getattr(donor, "key", "") or "")
    if not donor_key:
        return "PatchedGenerationSpec case is missing a donor example key"
    for source_layer in required_source_layers_for_patch(spec.patch):
        layer_payload = layers_payload.get(str(int(source_layer)))
        if not isinstance(layer_payload, Mapping) or donor_key not in layer_payload:
            return (
                f"{type(spec.patch).__name__} activation_bank source is missing donor activation rows "
                f"for source layer {int(source_layer)}"
            )
    selector = None
    if isinstance(spec.patch, InterchangePatch):
        selector = spec.patch.donor_tokens or spec.patch.target_tokens
    elif isinstance(spec.patch, ResidualPathPatch):
        selector = spec.patch.read_tokens or spec.patch.target_tokens
    if not selector_requires_sections(selector):
        return None
    section_name = str(selector.value)
    first_layer = str(int(required_source_layers_for_patch(spec.patch)[0]))
    donor_record = layers_payload[first_layer][donor_key]
    if not isinstance(donor_record, Mapping):
        return f"{type(spec.patch).__name__} activation_bank source is missing donor activation rows"
    token_sections = donor_record.get("token_sections")
    if not isinstance(token_sections, Mapping):
        selector_field = "patch.donor_tokens" if isinstance(spec.patch, InterchangePatch) else "patch.read_tokens"
        return (
            f"{type(spec.patch).__name__} uses TokenSelector.section(...) for {selector_field}, "
            "but donor token-section metadata is missing from activation_bank"
        )
    if section_name not in token_sections:
        selector_field = "patch.donor_tokens" if isinstance(spec.patch, InterchangePatch) else "patch.read_tokens"
        return (
            f"{type(spec.patch).__name__} uses TokenSelector.section(...) for {selector_field}, "
            f"but donor section {section_name!r} is missing from activation_bank token_sections"
        )
    return None


def _predicate_keys(predicate: Any, dataset_keys: set[str], *, label: str) -> set[str]:
    if predicate is None or not hasattr(predicate, "resolve_example_keys"):
        raise SpecValidationError(f"PatchedGenerationSpec requires {label} to resolve example keys")
    return {str(key) for key in predicate.resolve_example_keys() if str(key) in dataset_keys}


def _single_example(case_examples: list[Example], allowed_keys: set[str], *, field: str) -> Example | str:
    matches = [example for example in case_examples if example.key in allowed_keys]
    if len(matches) != 1:
        return f"{field} must resolve exactly one example per case"
    return matches[0]


__all__ = [
    "activation_bank_case_errors",
    "activation_bank_case_issue",
    "partition_cases_by_activation_bank",
    "resolve_generation_examples",
    "resolve_patched_generation_cases",
    "resolve_patched_generation_targets",
    "selector_requires_sections",
]
