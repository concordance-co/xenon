from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pipelines_v2.api import (
    Dataset,
    Example,
    GenerationRunSpec,
    GenerationSpec,
    StepRef,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)

from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base


PHASE_02_OUTPUTS = Path("projects/MOREBENCH/phase_02/outputs")
DIRECT_THEORY_PATH = PHASE_02_OUTPUTS / "theory_prompt_augmentation_examples.jsonl"
REPAIR_PATH = PHASE_02_OUTPUTS / "theory_prompt_repair_examples.jsonl"
CONTROL_PATH = PHASE_02_OUTPUTS / "theory_control_augmentation_examples.jsonl"
PUBLIC_CONFLICT_EXTENSION_PATH = (
    Path("projects/MOREBENCH/phase_03/outputs/experiment_02_public_conflict_extension.jsonl")
)
BENCHMARK_ANALYSIS_INSTRUCTION = (
    "Use the moral framework below to analyze the dilemma and provide a final recommendation."
)
BENCHMARK_RECOMMENDATION_INSTRUCTION = (
    "Be explicit about your final recommendation, even if the case is difficult or uncertain."
)
GENERIC_CUE_TEXT = (
    "Use a moral framework. Focus on the important considerations, compare the options carefully, "
    "and explain which option seems best supported overall."
)
THEORY_SHORT = {
    "Act Utilitarianism": "utilitarian",
    "Aristotelian Virtue Ethics": "virtue_ethics",
    "Gauthierian Contractarianism": "contractarianism",
    "Kantian Deontology": "deontology",
    "Scanlonian Contractualism": "contractualism",
}
THEORY_ORDER = tuple(THEORY_SHORT)
GENERIC_CONTROL_LABEL = "generic_ethics_control"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _normalize_guidance(cue_text: str) -> str:
    prefix = "Use a moral framework. "
    if cue_text.startswith(prefix):
        return cue_text[len(prefix) :].strip()
    if cue_text.startswith("Analyze the dilemma through ") and cue_text.endswith("."):
        return cue_text[len("Analyze the dilemma through ") : -1].strip()
    return cue_text.strip()


def _build_generation_prompt(*, cue_text: str, dilemma: str) -> str:
    guidance = _normalize_guidance(cue_text)
    return (
        f"{BENCHMARK_ANALYSIS_INSTRUCTION}\n\n"
        f"MORAL FRAMEWORK GUIDANCE:\n{guidance}\n\n"
        f"DILEMMA:\n{dilemma}\n\n"
        f"{BENCHMARK_RECOMMENDATION_INSTRUCTION}"
    )


def build_dataset() -> Dataset:
    direct_rows = _load_jsonl(DIRECT_THEORY_PATH)
    repair_rows = _load_jsonl(REPAIR_PATH)
    control_rows = _load_jsonl(CONTROL_PATH)
    public_extension_rows = _load_jsonl(PUBLIC_CONFLICT_EXTENSION_PATH)

    description_rows = [row for row in repair_rows if str(row.get("variant_family")) == "description_only"]
    cue_by_theory: dict[str, str] = {}
    for row in description_rows:
        theory = str(row["theory"])
        bank = str(row.get("description_bank") or "")
        if theory in cue_by_theory or bank != "a":
            continue
        cue_by_theory[theory] = str(row["cue_text"])
    del control_rows
    generic_cue = GENERIC_CUE_TEXT

    grouped_direct_rows: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    for row in direct_rows:
        group_id = str(row["group_id"])
        if group_id in seen_group_ids:
            continue
        seen_group_ids.add(group_id)
        grouped_direct_rows.append(row)

    examples: list[Example] = []
    for row in grouped_direct_rows:
        group_id = str(row["group_id"])
        base_example_id = group_id
        dilemma = str(row["base_dilemma"])
        shared_labels = {
            "group_id": base_example_id,
            "base_dilemma_id": base_example_id,
            "source_group_id": group_id,
            "source_case_id": base_example_id,
            "source_case_theory": "benchmark_theory_group",
            "source_case_theory_name": "",
            "source_family": str(row["source_family"]),
            "dilemma_type": str(row["dilemma_type"]),
            "context": str(row["context"]),
            "role_domain": str(row["role_domain"]),
            "split": "behavior_all",
            "capture_enabled": False,
            "capture_tier": "behavior_only",
            "source_prompt_family": "phase02_direct",
        }

        for theory_name in THEORY_ORDER:
            cue_text = cue_by_theory[theory_name]
            prime_condition = THEORY_SHORT[theory_name]
            examples.append(
                Example(
                    key=f"{base_example_id}__prime_{prime_condition}",
                    prompt=[
                        {"role": "system", "content": base.SYSTEM_PROMPT},
                        {"role": "user", "content": _build_generation_prompt(cue_text=cue_text, dilemma=dilemma)},
                    ],
                    labels={
                        **shared_labels,
                        "prime_family": "description_only",
                        "prime_condition": prime_condition,
                        "theory_name": theory_name,
                        "is_theory_prime": True,
                        "is_generic_control": False,
                        "description_bank": "a",
                        "alias_bank": "",
                        "cue_text": cue_text,
                    },
                    metadata={"cue_text": cue_text},
                    cases={"group_id": base_example_id, "source_group_id": group_id, "source_case_id": base_example_id},
                    case_key=base_example_id,
                )
            )

        examples.append(
            Example(
                key=f"{base_example_id}__prime_{GENERIC_CONTROL_LABEL}",
                prompt=[
                    {"role": "system", "content": base.SYSTEM_PROMPT},
                    {"role": "user", "content": _build_generation_prompt(cue_text=generic_cue, dilemma=dilemma)},
                ],
                labels={
                    **shared_labels,
                    "prime_family": GENERIC_CONTROL_LABEL,
                    "prime_condition": GENERIC_CONTROL_LABEL,
                    "theory_name": "",
                    "is_theory_prime": False,
                    "is_generic_control": True,
                    "description_bank": "",
                    "alias_bank": "",
                    "cue_text": generic_cue,
                },
                metadata={"cue_text": generic_cue},
                cases={"group_id": base_example_id, "source_group_id": group_id, "source_case_id": base_example_id},
                case_key=base_example_id,
            )
        )

    for row in public_extension_rows:
        extension_id = f"public_conflict_{int(row['selection_rank']):03d}"
        dilemma = str(row["DILEMMA"])
        source_case_theory = str(row.get("THEORY") or "neutral")
        shared_labels = {
            "group_id": extension_id,
            "base_dilemma_id": extension_id,
            "source_group_id": extension_id,
            "source_case_id": extension_id,
            "source_case_theory": source_case_theory,
            "source_case_theory_name": source_case_theory,
            "source_family": str(row["DILEMMA_SOURCE"]),
            "dilemma_type": str(row["DILEMMA_TYPE"]),
            "context": str(row["CONTEXT"]),
            "role_domain": str(row["ROLE_DOMAIN"]),
            "split": "behavior_all",
            "capture_enabled": False,
            "capture_tier": "behavior_only",
            "source_prompt_family": str(row["selection_protocol"]),
            "extension_split": str(row["extension_split"]),
            "selection_rank": int(row["selection_rank"]),
            "pool_index": int(row["pool_index"]),
        }

        for theory_name in THEORY_ORDER:
            cue_text = cue_by_theory[theory_name]
            prime_condition = THEORY_SHORT[theory_name]
            examples.append(
                Example(
                    key=f"{extension_id}__prime_{prime_condition}",
                    prompt=[
                        {"role": "system", "content": base.SYSTEM_PROMPT},
                        {"role": "user", "content": _build_generation_prompt(cue_text=cue_text, dilemma=dilemma)},
                    ],
                    labels={
                        **shared_labels,
                        "prime_family": "description_only",
                        "prime_condition": prime_condition,
                        "theory_name": theory_name,
                        "is_theory_prime": True,
                        "is_generic_control": False,
                        "description_bank": "a",
                        "alias_bank": "",
                        "cue_text": cue_text,
                    },
                    metadata={"cue_text": cue_text},
                    cases={
                        "group_id": extension_id,
                        "source_group_id": extension_id,
                        "source_case_id": extension_id,
                    },
                    case_key=extension_id,
                )
            )

        examples.append(
            Example(
                key=f"{extension_id}__prime_{GENERIC_CONTROL_LABEL}",
                prompt=[
                    {"role": "system", "content": base.SYSTEM_PROMPT},
                    {"role": "user", "content": _build_generation_prompt(cue_text=generic_cue, dilemma=dilemma)},
                ],
                labels={
                    **shared_labels,
                    "prime_family": GENERIC_CONTROL_LABEL,
                    "prime_condition": GENERIC_CONTROL_LABEL,
                    "theory_name": "",
                    "is_theory_prime": False,
                    "is_generic_control": True,
                    "description_bank": "",
                    "alias_bank": "",
                    "cue_text": generic_cue,
                },
                metadata={"cue_text": generic_cue},
                cases={
                    "group_id": extension_id,
                    "source_group_id": extension_id,
                    "source_case_id": extension_id,
                },
                case_key=extension_id,
            )
        )

    return Dataset.from_examples(
        examples,
        name="morebench_phase03_experiment02_behavior_broad_generation_batch",
    )


def summarize_behavior_broad_generation(*, generation: Any) -> TransformResult:
    payload = generation.result()
    rows = list(payload.get("rows", []))
    finish_reason_counts = Counter(str(row.get("finish_reason") or "") for row in rows)
    family_counts = Counter(
        str(_mapping(row.get("example", {})).get("labels", {}).get("source_case_theory") or "")
        for row in rows
    )
    prime_counts = Counter(
        str(_mapping(row.get("example", {})).get("labels", {}).get("prime_condition") or "")
        for row in rows
    )
    nonempty = sum(bool(str(row.get("generated_text") or "").strip()) for row in rows)
    return TransformResult(
        payload={
            "row_count": len(rows),
            "nonempty_count": nonempty,
            "nonempty_rate": round(nonempty / len(rows), 4) if rows else 0.0,
            "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
            "source_case_theory_counts": dict(sorted(family_counts.items())),
            "prime_condition_counts": dict(sorted(prime_counts.items())),
            "sample_keys": [str(row.get("example_key") or "") for row in rows[:10]],
        }
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_runner_specs():
    return base.build_runner_specs()


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    return WorkflowSpec(
        name="morebench_phase03_experiment02_behavior_broad_generation",
        steps=(
            WorkflowStep(
                name="generate_theory_primed_responses",
                runner="capture_gpu",
                description=(
                    "Broad behavior-only run over the 30 grouped theory dilemmas plus a 60-dilemma public "
                    "conflict extension, each crossed with the five description-only primes plus one generic control."
                ),
                spec=GenerationRunSpec(
                    engine=base._engine(),
                    dataset=dataset,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=base.GENERATION_MAX_TOKENS,
                        temperature=0.0,
                        top_p=1.0,
                    ),
                ),
            ),
            WorkflowStep(
                name="summarize_behavior_broad_generation",
                runner="analysis_cpu",
                description="Check generation health for the broad behavior-only batch.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_behavior_broad_generation,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_theory_primed_responses")},
                ),
            ),
        ),
    )
