from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pipelines_v2.api import (
    ArtifactDatasetSource,
    CaptureSpec,
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
    ResidualSite,
    StepRef,
    TensorStorage,
    TokenSelector,
    TransformBuilder,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)

from projects.MOREBENCH.phase_03.specs import experiment_02_cross_language_pilot_workflow as pilot
from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base


SOURCE_PATH = Path("projects/MOREBENCH/phase_02/outputs/theory_prompt_variant_sweep_examples.jsonl")
TARGET_PRIMES = ("deontology", "virtue_ethics")

LOCAL_CATALOG_ROOT = Path("artifacts") / "morebench_phase03_experiment02_cross_language_prompt_probe_full_catalog"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "morebench_phase03_experiment02_cross_language_prompt_probe_full"
MODAL_ARTIFACT_ROOT = "/data/artifacts/morebench_phase_03_experiment02_cross_language_prompt_probe_full"

TRANSLATION_LANGS = ("es", "zh")

TRANSLATION_SYSTEM_PROMPT = (
    "You are a precise translator. Translate the user-provided English moral dilemma into the requested target "
    "language. Preserve names, factual details, stakes, and uncertainty. Output only the translated dilemma text. "
    "Do not explain your choices. Do not add bullets or headers. Do not leave English words unless they are proper "
    "names already present in the source."
)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _extract_dilemma_from_prompt(prompt: str) -> str:
    return prompt.split("\n\nDILEMMA:\n", 1)[1]


def _english_rows() -> dict[tuple[str, str], dict[str, object]]:
    rows = {}
    for row in _load_jsonl(SOURCE_PATH):
        group_id = str(row.get("group_id"))
        prime = str(row.get("prime_condition"))
        bank = str(row.get("variant_bank") or row.get("description_bank") or "")
        if prime in TARGET_PRIMES and bank == "analytic":
            rows[(group_id, prime)] = row
    return rows


def _unique_group_ids(english_rows: Mapping[tuple[str, str], dict[str, object]]) -> list[str]:
    return sorted({group_id for group_id, _ in english_rows.keys()})


def build_dataset() -> Dataset:
    english_rows = _english_rows()
    examples: list[Example] = []
    for group_id in _unique_group_ids(english_rows):
        source_row = english_rows[(group_id, "deontology")]
        dilemma_text = _extract_dilemma_from_prompt(str(source_row["prompt"]))
        for language_code in TRANSLATION_LANGS:
            language_name = pilot.LANGUAGE_CONFIG[language_code]["name"]
            examples.append(
                Example(
                    key=f"translate__{group_id}__{language_code}",
                    prompt=[
                        {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Target language: {language_name}\n\n"
                                f"English dilemma:\n{dilemma_text}"
                            ),
                        },
                    ],
                    labels={
                        "group_id": group_id,
                        "language_code": language_code,
                        "language_name": language_name,
                        "source_family": str(source_row.get("source_family") or ""),
                        "context": str(source_row.get("context") or ""),
                        "role_domain": str(source_row.get("role_domain") or ""),
                    },
                    metadata={"english_dilemma_text": dilemma_text},
                    cases={"group_id": group_id},
                    case_key=group_id,
                )
            )
    return Dataset.from_examples(
        examples,
        name="morebench_phase03_experiment02_cross_language_translation_full30",
    )


def build_runner_specs() -> dict[str, object]:
    catalog = FileCatalog(root=LOCAL_CATALOG_ROOT)
    modal_store = ModalVolumeStore(
        name="xenon-data",
        root=MODAL_ARTIFACT_ROOT,
    )
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 6,
                volumes=(ModalVolumeMount(name=base.MODEL_VOLUME_NAME, mount_path=base.MODEL_VOLUME_PATH),),
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "analysis_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
            catalog=catalog,
        ),
    }


def _artifact_prompt_dataset() -> Dataset:
    return Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef("build_translated_prompt_dataset"),
        result_key="dataset",
        name="morebench_phase03_experiment02_cross_language_prompt_full30",
    )


def _user_prompt(*, language_code: str, cue_text: str, dilemma_text: str) -> str:
    cfg = pilot.LANGUAGE_CONFIG[language_code]
    return (
        f"{cfg['analysis_instruction']}\n\n"
        f"{cfg['framework_header']}\n{cue_text}\n\n"
        f"{cfg['dilemma_header']}\n{dilemma_text}\n\n"
        f"{cfg['recommendation_instruction']}"
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_translated_prompt_dataset(*, translations: Any) -> dict[str, Any]:
    payload = translations.result() if hasattr(translations, "result") else {}
    if not isinstance(payload, Mapping):
        raise TypeError("translations must resolve to a mapping payload")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError("translations payload must contain rows")

    translated_dilemmas: dict[str, dict[str, str]] = {}
    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        example = _mapping(row.get("example"))
        labels = dict(_mapping(example.get("labels")))
        group_id = str(labels.get("group_id") or "")
        language_code = str(labels.get("language_code") or "")
        translated_text = str(row.get("generated_text") or row.get("text") or "").strip()
        if group_id and language_code and translated_text:
            translated_dilemmas.setdefault(group_id, {})[language_code] = translated_text

    english_rows = _english_rows()
    examples: list[Example] = []
    for group_id in _unique_group_ids(english_rows):
        for prime in TARGET_PRIMES:
            source_row = english_rows[(group_id, prime)]
            english_cue = str(source_row["cue_text"])
            english_dilemma = _extract_dilemma_from_prompt(str(source_row["prompt"]))
            for language_code in ("en", "es", "zh"):
                if language_code == "en":
                    cue_text = english_cue
                    dilemma_text = english_dilemma
                else:
                    cue_text = pilot.TRANSLATED_CUES[prime][language_code]
                    dilemma_text = translated_dilemmas[group_id][language_code]
                cfg = pilot.LANGUAGE_CONFIG[language_code]
                key = f"{group_id}__{prime}__lang_{language_code}"
                examples.append(
                    Example(
                        key=key,
                        prompt=[
                            {"role": "system", "content": cfg["system_prompt"]},
                            {"role": "user", "content": _user_prompt(language_code=language_code, cue_text=cue_text, dilemma_text=dilemma_text)},
                        ],
                        labels={
                            "group_id": group_id,
                            "prime_condition": prime,
                            "prime_family": "cross_language_full30",
                            "language_code": language_code,
                            "language_name": cfg["name"],
                            "cue_mode": "translated_analytic",
                            "cue_text": cue_text,
                            "source_family": str(source_row.get("source_family") or ""),
                            "dilemma_type": str(source_row.get("dilemma_type") or ""),
                            "context": str(source_row.get("context") or ""),
                            "role_domain": str(source_row.get("role_domain") or ""),
                        },
                        metadata={"cue_text": cue_text, "dilemma_text": dilemma_text},
                        cases={"group_id": group_id},
                        case_key=group_id,
                    )
                )

    dataset = Dataset.from_examples(
        examples,
        name="morebench_phase03_experiment02_cross_language_prompt_full30",
    )
    return {
        "payload": {
            "kind": "morebench_cross_language_prompt_full30_dataset",
            "dataset": dataset.to_dict(),
            "summary": {
                "translated_group_count": len(translated_dilemmas),
                "prompt_example_count": len(examples),
                "language_counts": {
                    "en": len([example for example in examples if example.labels["language_code"] == "en"]),
                    "es": len([example for example in examples if example.labels["language_code"] == "es"]),
                    "zh": len([example for example in examples if example.labels["language_code"] == "zh"]),
                },
            },
        },
        "example_keys": [example.key for example in examples],
    }


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    return WorkflowSpec(
        name="morebench_phase03_experiment02_cross_language_prompt_probe_full",
        steps=(
            WorkflowStep(
                name="translate_dilemmas",
                runner="capture_gpu",
                description="Translate all 30 analytic-bank dilemmas into Spanish and Simplified Chinese.",
                spec=GenerationRunSpec(
                    engine=base._engine(max_num_seqs=8),
                    dataset=dataset,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=2200,
                        temperature=0.0,
                        top_p=1.0,
                    ),
                ),
            ),
            WorkflowStep(
                name="build_translated_prompt_dataset",
                runner="analysis_local",
                description="Assemble the full 180-prompt cross-language prompt-final dataset from the translation pass.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_translated_prompt_dataset,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"translations": StepRef("translate_dilemmas")},
                ),
            ),
            WorkflowStep(
                name="capture_prompt_eos_residual",
                runner="capture_gpu",
                description="Capture prompt-final residuals on the full translated deontology-vs-virtue prompt set.",
                spec=CaptureSpec(
                    engine=base._engine(max_num_seqs=8),
                    dataset=_artifact_prompt_dataset(),
                    sites=[
                        ResidualSite(
                            name="prompt_eos_residual",
                            site="resid_post",
                            layers=list(base.CAPTURED_LAYERS),
                            tokens=TokenSelector.last(),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        )
                    ],
                    generation=GenerationSpec(enabled=False),
                ),
            ),
        ),
    )
