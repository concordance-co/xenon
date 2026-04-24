from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from pipelines_v2.api import (
    Dataset,
    Example,
    FileCatalog,
    GenerationRunSpec,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    StepRef,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)

from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base


DESCRIPTION_PATH = Path("projects/MOREBENCH/phase_02/outputs/theory_prompt_variant_sweep_examples.jsonl")
DESCRIPTION_CONTROL_PATH = Path("projects/MOREBENCH/phase_02/outputs/theory_prompt_variant_sweep_controls.jsonl")
NAME_PATH = Path("projects/MOREBENCH/phase_02/outputs/theory_name_only_crossformat_examples.jsonl")
NAME_CONTROL_PATH = Path("projects/MOREBENCH/phase_02/outputs/theory_name_only_crossformat_controls.jsonl")
BENCHMARK_ANALYSIS_INSTRUCTION = (
    "Use the moral framework below to analyze the dilemma and provide a final recommendation."
)
BENCHMARK_RECOMMENDATION_INSTRUCTION = (
    "Be explicit about your final recommendation, even if the case is difficult or uncertain."
)

PILOT_GROUPS = (
    "theory_group_005",
    "theory_group_009",
    "theory_group_013",
    "theory_group_015",
    "theory_group_022",
)

REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_variant_pilot")


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _normalize_prompt(cue_text: str, dilemma: str) -> str:
    return (
        f"{BENCHMARK_ANALYSIS_INSTRUCTION}\n\n"
        f"MORAL FRAMEWORK GUIDANCE:\n{cue_text}\n\n"
        f"DILEMMA:\n{dilemma}\n\n"
        f"{BENCHMARK_RECOMMENDATION_INSTRUCTION}"
    )


def build_dataset() -> Dataset:
    group_set = set(PILOT_GROUPS)
    desc_rows = [
        row for row in _load_jsonl(DESCRIPTION_PATH) if str(row.get("group_id")) in group_set
    ]
    desc_controls = [
        row for row in _load_jsonl(DESCRIPTION_CONTROL_PATH) if str(row.get("group_id")) in group_set
    ]
    name_rows = [
        row for row in _load_jsonl(NAME_PATH) if str(row.get("group_id")) in group_set
    ]
    name_controls = [
        row for row in _load_jsonl(NAME_CONTROL_PATH) if str(row.get("group_id")) in group_set
    ]

    examples: list[Example] = []
    for row in [*desc_rows, *desc_controls, *name_rows, *name_controls]:
        prime_condition = str(row.get("prime_condition") or row.get("control_type") or "")
        is_theory_prime = bool(row.get("theory"))
        group_id = str(row["group_id"])
        example_id = str(row["example_id"])
        cue_text = str(row["cue_text"])
        full_prompt = str(row["prompt"])
        dilemma = full_prompt.split("\n\nDILEMMA:\n", 1)[1]
        if str(row.get("cue_mode")) == "name_only":
            prime_family = "name_only_crossformat"
            bank = str(row.get("name_bank") or "")
        else:
            prime_family = "description_variant_sweep"
            bank = str(row.get("variant_bank") or row.get("description_bank") or "")
        examples.append(
            Example(
                key=example_id,
                prompt=[
                    {"role": "system", "content": base.SYSTEM_PROMPT},
                    {"role": "user", "content": _normalize_prompt(cue_text=cue_text, dilemma=dilemma)},
                ],
                labels={
                    "group_id": group_id,
                    "prime_condition": prime_condition,
                    "prime_family": prime_family,
                    "variant_bank": bank,
                    "cue_mode": str(row.get("cue_mode") or ""),
                    "cue_text": cue_text,
                    "is_theory_prime": is_theory_prime,
                    "is_generic_control": not is_theory_prime,
                    "split": str(row.get("split") or "train"),
                    "source_family": str(row.get("source_family") or ""),
                    "dilemma_type": str(row.get("dilemma_type") or ""),
                    "context": str(row.get("context") or ""),
                    "role_domain": str(row.get("role_domain") or ""),
                },
                metadata={"cue_text": cue_text},
                cases={"group_id": group_id},
                case_key=group_id,
            )
        )
    return Dataset.from_examples(examples, name="morebench_phase03_experiment02_variant_pilot")


def build_runner_specs() -> dict[str, object]:
    catalog = FileCatalog(root=Path("artifacts") / "morebench_phase03_experiment02_variant_pilot_catalog")
    modal_store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/morebench_phase_03_experiment02_variant_pilot",
    )
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 4,
                volumes=(ModalVolumeMount(name=base.MODEL_VOLUME_NAME, mount_path=base.MODEL_VOLUME_PATH),),
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "analysis_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(Path("artifacts") / "morebench_phase03_experiment02_variant_pilot"),
            catalog=catalog,
        ),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _fit_auc(train_texts: list[str], train_labels: list[int], test_texts: list[str], test_labels: list[int]) -> float:
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, train_labels)
    probs = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(test_labels, probs))


def summarize_variant_pilot(*, generation: Any) -> TransformResult:
    payload = generation.result() if hasattr(generation, "result") else {}
    rows = payload.get("rows", []) if isinstance(payload, Mapping) else []

    finish_reasons: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    sample_rows: list[dict[str, Any]] = []
    theory_response_texts: dict[str, dict[str, list[str]]] = {}
    generic_response_texts: dict[str, dict[str, list[str]]] = {}

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        finish_reasons[str(row.get("finish_reason") or "")] += 1
        generated_text = str(row.get("generated_text") or row.get("text") or "")
        source_example = _mapping(row.get("example"))
        labels = dict(_mapping(source_example.get("labels")))
        prime_family = str(labels.get("prime_family") or "")
        variant_bank = str(labels.get("variant_bank") or "")
        prime_condition = str(labels.get("prime_condition") or "")
        family_counts[prime_family] += 1
        variant_counts[f"{prime_family}:{variant_bank}"] += 1

        if sample_rows.__len__() < 12:
            sample_rows.append(
                {
                    "example_key": str(row.get("example_key") or ""),
                    "prime_family": prime_family,
                    "variant_bank": variant_bank,
                    "prime_condition": prime_condition,
                    "preview": generated_text[:350],
                }
            )

        if prime_condition == "generic_ethics_control":
            generic_response_texts.setdefault(prime_family, {}).setdefault(variant_bank, []).append(generated_text)
        else:
            theory_response_texts.setdefault(prime_condition, {}).setdefault(f"{prime_family}:{variant_bank}", []).append(generated_text)

    response_text_diagnostic: dict[str, object] = {}
    description_family = "description_variant_sweep"
    name_family = "name_only_crossformat"
    description_banks = sorted(
        {key.split(":", 1)[1] for key in variant_counts if key.startswith(f"{description_family}:")}
    )
    name_banks = sorted(
        {key.split(":", 1)[1] for key in variant_counts if key.startswith(f"{name_family}:")}
    )

    for theory, bank_map in sorted(theory_response_texts.items()):
        # Within-description held-out-bank
        desc_holdouts: list[dict[str, object]] = []
        if description_banks:
            for held_out in description_banks:
                train_texts: list[str] = []
                train_labels: list[int] = []
                test_texts: list[str] = []
                test_labels: list[int] = []
                for bank in description_banks:
                    theory_key = f"{description_family}:{bank}"
                    generic_key = bank
                    theory_rows = bank_map.get(theory_key, [])
                    generic_rows = generic_response_texts.get(description_family, {}).get(generic_key, [])
                    if bank == held_out:
                        test_texts.extend(theory_rows)
                        test_labels.extend([1] * len(theory_rows))
                        test_texts.extend(generic_rows)
                        test_labels.extend([0] * len(generic_rows))
                    else:
                        train_texts.extend(theory_rows)
                        train_labels.extend([1] * len(theory_rows))
                        train_texts.extend(generic_rows)
                        train_labels.extend([0] * len(generic_rows))
                if train_texts and test_texts and len(set(train_labels)) == 2 and len(set(test_labels)) == 2:
                    auc = _fit_auc(train_texts, train_labels, test_texts, test_labels)
                    desc_holdouts.append({"held_out_bank": held_out, "auroc": round(auc, 4)})
        # Cross-format
        desc_to_name = None
        name_to_desc = None
        desc_train_texts: list[str] = []
        desc_train_labels: list[int] = []
        name_test_texts: list[str] = []
        name_test_labels: list[int] = []
        for bank in description_banks:
            desc_train_texts.extend(bank_map.get(f"{description_family}:{bank}", []))
            desc_train_labels.extend([1] * len(bank_map.get(f"{description_family}:{bank}", [])))
            desc_train_texts.extend(generic_response_texts.get(description_family, {}).get(bank, []))
            desc_train_labels.extend([0] * len(generic_response_texts.get(description_family, {}).get(bank, [])))
        for bank in name_banks:
            name_test_texts.extend(bank_map.get(f"{name_family}:{bank}", []))
            name_test_labels.extend([1] * len(bank_map.get(f"{name_family}:{bank}", [])))
            name_test_texts.extend(generic_response_texts.get(name_family, {}).get(bank, []))
            name_test_labels.extend([0] * len(generic_response_texts.get(name_family, {}).get(bank, [])))
        if desc_train_texts and name_test_texts and len(set(desc_train_labels)) == 2 and len(set(name_test_labels)) == 2:
            desc_to_name = round(_fit_auc(desc_train_texts, desc_train_labels, name_test_texts, name_test_labels), 4)

        name_train_texts: list[str] = []
        name_train_labels: list[int] = []
        desc_test_texts: list[str] = []
        desc_test_labels: list[int] = []
        for bank in name_banks:
            name_train_texts.extend(bank_map.get(f"{name_family}:{bank}", []))
            name_train_labels.extend([1] * len(bank_map.get(f"{name_family}:{bank}", [])))
            name_train_texts.extend(generic_response_texts.get(name_family, {}).get(bank, []))
            name_train_labels.extend([0] * len(generic_response_texts.get(name_family, {}).get(bank, [])))
        for bank in description_banks:
            desc_test_texts.extend(bank_map.get(f"{description_family}:{bank}", []))
            desc_test_labels.extend([1] * len(bank_map.get(f"{description_family}:{bank}", [])))
            desc_test_texts.extend(generic_response_texts.get(description_family, {}).get(bank, []))
            desc_test_labels.extend([0] * len(generic_response_texts.get(description_family, {}).get(bank, [])))
        if name_train_texts and desc_test_texts and len(set(name_train_labels)) == 2 and len(set(desc_test_labels)) == 2:
            name_to_desc = round(_fit_auc(name_train_texts, name_train_labels, desc_test_texts, desc_test_labels), 4)

        response_text_diagnostic[theory] = {
            "description_holdout_banks": desc_holdouts,
            "description_holdout_mean_auroc": round(
                sum(item["auroc"] for item in desc_holdouts) / len(desc_holdouts), 4
            ) if desc_holdouts else None,
            "description_to_name_auroc": desc_to_name,
            "name_to_description_auroc": name_to_desc,
        }

    return TransformResult(
        payload={
            "workflow": "morebench_phase03_experiment02_variant_pilot",
            "group_ids": list(PILOT_GROUPS),
            "row_count": len(rows),
            "finish_reason_counts": dict(sorted(finish_reasons.items())),
            "prime_family_counts": dict(sorted(family_counts.items())),
            "variant_counts": dict(sorted(variant_counts.items())),
            "response_text_diagnostic": response_text_diagnostic,
            "sample_rows": sample_rows,
        }
    )


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    return WorkflowSpec(
        name="morebench_phase03_experiment02_variant_pilot",
        steps=(
            WorkflowStep(
                name="generate_variant_pilot_responses",
                runner="capture_gpu",
                description="Small generation pilot across description-variant and name-only cross-format theory prompts.",
                spec=GenerationRunSpec(
                    engine=base._engine(max_num_seqs=12),
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
                name="summarize_variant_pilot",
                runner="analysis_local",
                description="Summarize generation counts and response-side char-TF-IDF lexical ceilings.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_variant_pilot,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_variant_pilot_responses")},
                ),
            ),
        ),
    )
