from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
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
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PostgresCatalog,
    PostgresSource,
    ProbeSpec,
    ReportSpec,
    ResidualSite,
    StepRef,
    TensorStorage,
    TextBaselineSpec,
    TokenPooling,
    TokenSelector,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)


PHASE_ROOT = Path("projects/MOREBENCH/phase_03")
DATASET_PATH = PHASE_ROOT / "outputs" / "experiment_02_generation_dataset.jsonl"
REPORT_OUTPUT_DIR = PHASE_ROOT / "reports" / "experiment_02_report"

DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
MODEL_ID = "Qwen/Qwen3-30B-A3B"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
MODAL_ARTIFACT_ROOT = "/data/artifacts/morebench_phase_03_experiment02"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "morebench_phase_03_experiment02"
LOCAL_CATALOG_ROOT = Path("artifacts") / "morebench_phase_03_experiment02_catalog"
SYSTEM_PROMPT = (
    "Analyze the dilemma carefully. "
    "You must give a final recommendation, even if the case is difficult or uncertain."
)

CAPTURED_LAYERS = (0, 4, 8, 16, 24, 32, 40, 44)
GENERATION_MAX_TOKENS = 2000
MAX_MODEL_LEN = 16384
MAX_NUM_SEQS = 16
GPU_SHARD_COUNT = 4
GENERIC_CONTROL_LABEL = "generic_ethics_control"
THEORY_NAMES = (
    "Act Utilitarianism",
    "Aristotelian Virtue Ethics",
    "Gauthierian Contractarianism",
    "Kantian Deontology",
    "Scanlonian Contractualism",
)
GENERIC_CUE_STOPWORDS = {
    "analyze",
    "dilemma",
    "moral",
    "framework",
    "focus",
    "compare",
    "options",
    "carefully",
    "important",
    "considerations",
    "supported",
    "overall",
    "option",
    "choice",
    "choices",
    "explain",
    "which",
    "seems",
    "best",
    "would",
    "should",
    "using",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def build_dataset(
    dataset_path: Path = DATASET_PATH,
    *,
    dataset_name: str = "morebench_phase03_experiment02_generation_batch",
) -> Dataset:
    records = _load_jsonl(dataset_path)
    examples: list[Example] = []
    for record in records:
        examples.append(
            Example(
                key=str(record["example_id"]),
                prompt=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(record["prompt"])},
                ],
                labels={
                    "group_id": str(record["group_id"]),
                    "base_dilemma_id": str(record["base_dilemma_id"]),
                    "split": str(record["split"]),
                    "capture_enabled": bool(record["capture_enabled"]),
                    "capture_tier": str(record["capture_tier"]),
                    "prime_family": str(record["prime_family"]),
                    "prime_condition": str(record["prime_condition"]),
                    "theory_name": str(record["theory_name"]),
                    "is_theory_prime": bool(record["is_theory_prime"]),
                    "is_generic_control": bool(record["is_generic_control"]),
                    "description_bank": str(record["description_bank"]),
                    "alias_bank": str(record.get("alias_bank", "")),
                    "cue_text": str(record["cue_text"]),
                    "source_family": str(record["source_family"]),
                    "dilemma_type": str(record["dilemma_type"]),
                    "context": str(record["context"]),
                    "role_domain": str(record["role_domain"]),
                },
                metadata={"cue_text": str(record["cue_text"])},
                cases={"group_id": str(record["group_id"]), "base_dilemma_id": str(record["base_dilemma_id"])},
                case_key=str(record["group_id"]),
            )
        )
    return Dataset.from_examples(examples, name=dataset_name)


def _engine(*, max_num_seqs: int = MAX_NUM_SEQS) -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        model_path_root=MODEL_VOLUME_PATH,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=False,
        enable_prefix_caching=True,
        max_num_seqs=max_num_seqs,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_runner_specs() -> dict[str, object]:
    modal_store = ModalVolumeStore(name="xenon-data", root=MODAL_ARTIFACT_ROOT)
    if os.getenv(DB_ENV_VAR):
        db = PostgresSource.from_env(DB_ENV_VAR)
        workflow_catalog = PostgresCatalog(source=db)
        modal_secrets = (ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon"),)
    else:
        workflow_catalog = FileCatalog(root=LOCAL_CATALOG_ROOT)
        modal_secrets = ()
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 6,
                shard_count=GPU_SHARD_COUNT,
                secrets=modal_secrets,
                volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
            ),
            artifacts=modal_store,
            catalog=workflow_catalog,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=8,
                memory_mb=24 * 1024,
                timeout_seconds=60 * 60 * 2,
                secrets=modal_secrets,
            ),
            artifacts=modal_store,
            catalog=workflow_catalog,
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
            catalog=workflow_catalog,
        ),
    }


def _artifact_capture_dataset() -> Dataset:
    return Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef("build_theory_persistence_capture_dataset"),
        result_key="dataset",
        provides_token_sections=True,
        name="morebench_phase03_experiment02_generation_capture",
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _content_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower())
        if len(token) >= 5 and token not in GENERIC_CUE_STOPWORDS
    ]


def _copy_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower()) if len(token) >= 3]


def _longest_common_token_run(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    best = 0
    right_positions: dict[str, list[int]] = defaultdict(list)
    for index, token in enumerate(right):
        right_positions[token].append(index)
    for left_index, token in enumerate(left):
        for right_index in right_positions.get(token, ()):
            run = 0
            while (
                left_index + run < len(left)
                and right_index + run < len(right)
                and left[left_index + run] == right[right_index + run]
            ):
                run += 1
            if run > best:
                best = run
    return best


def _cue_copy_metrics(*, cue_text: str, generated_text: str) -> dict[str, float | int]:
    cue_content_tokens = set(_content_tokens(cue_text))
    generated_content_tokens = set(_content_tokens(generated_text))
    overlap_count = len(cue_content_tokens & generated_content_tokens)
    overlap_fraction = (
        overlap_count / len(cue_content_tokens)
        if cue_content_tokens
        else 0.0
    )
    longest_run = _longest_common_token_run(_copy_tokens(cue_text), _copy_tokens(generated_text))
    return {
        "cue_content_token_count": len(cue_content_tokens),
        "cue_overlap_count": overlap_count,
        "cue_overlap_fraction": overlap_fraction,
        "cue_longest_run": longest_run,
    }


def _near_verbatim_cue_copy(*, cue_text: str, generated_text: str) -> tuple[bool, dict[str, float | int]]:
    metrics = _cue_copy_metrics(cue_text=cue_text, generated_text=generated_text)
    cue_content_token_count = int(metrics["cue_content_token_count"])
    flag = bool(
        int(metrics["cue_longest_run"]) >= 8
        or (cue_content_token_count >= 4 and float(metrics["cue_overlap_fraction"]) >= 0.9)
    )
    return flag, metrics


def _theory_name_copy_metrics(*, theory_name: str, generated_text: str) -> dict[str, int | bool]:
    normalized_theory_name = re.sub(r"\s+", " ", theory_name.strip().lower())
    normalized_generated = re.sub(r"\s+", " ", generated_text.strip().lower())
    if not normalized_theory_name or not normalized_generated:
        return {
            "theory_name_mention_count": 0,
            "repeated_theory_name_copy": False,
        }
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(normalized_theory_name)}(?![A-Za-z0-9])")
    mention_count = len(pattern.findall(normalized_generated))
    return {
        "theory_name_mention_count": mention_count,
        "repeated_theory_name_copy": mention_count > 1,
    }


def _render_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, (list, tuple)):
        parts: list[str] = []
        for message in prompt:
            if isinstance(message, Mapping):
                role = str(message.get("role") or "").strip()
                content = message.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        str(item.get("text") or "") if isinstance(item, Mapping) else str(item)
                        for item in content
                    )
                label = role.capitalize() if role else "Message"
                parts.append(f"{label}:\n{content}")
            else:
                parts.append(str(message))
        return "\n\n".join(part for part in parts if part.strip())
    return str(prompt)


def _last_non_whitespace_span(text: str, start: int, end: int) -> dict[str, int]:
    index = int(end) - 1
    while index >= int(start) and text[index].isspace():
        index -= 1
    if index < int(start):
        index = max(int(start), int(end) - 1)
    return {"char_start": index, "char_end": index + 1}


def _combined_prompt_and_sections(*, source_prompt: str, generated_text: str) -> tuple[str, dict[str, Any]]:
    separator = "\n\nAssistant response:\n"
    prompt_end = len(source_prompt)
    generated_start = prompt_end + len(separator)
    combined = f"{source_prompt}{separator}{generated_text}"
    generated_end = len(combined)
    return combined, {
        "prompt": {"char_start": 0, "char_end": prompt_end},
        "generated": {"char_start": generated_start, "char_end": generated_end},
        "full": {"char_start": 0, "char_end": generated_end},
        "generated_end": _last_non_whitespace_span(combined, generated_start, generated_end),
        "full_end": _last_non_whitespace_span(combined, 0, generated_end),
    }


def build_theory_persistence_capture_dataset(*, generation: Any) -> dict[str, Any]:
    if not hasattr(generation, "result"):
        raise TypeError("build_theory_persistence_capture_dataset expects a generation artifact")

    payload = generation.result()
    if not isinstance(payload, Mapping):
        raise TypeError("generation artifact result must be a mapping")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError("generation artifact result must contain a rows list")

    examples: list[Example] = []
    generated_text_labels: dict[str, str] = {}
    group_id_labels: dict[str, str] = {}
    split_labels: dict[str, str] = {}
    prime_family_labels: dict[str, str] = {}
    prime_condition_labels: dict[str, str] = {}
    alias_bank_labels: dict[str, str] = {}
    finish_reason_labels: dict[str, str] = {}
    token_count_labels: dict[str, int] = {}
    source_family_labels: dict[str, str] = {}
    response_length_labels: dict[str, int] = {}
    theory_name_copy_labels: dict[str, str] = {}
    theory_name_mention_count_labels: dict[str, int] = {}
    cue_overlap_copy_labels: dict[str, str] = {}
    direct_copy_labels: dict[str, str] = {}

    skipped_length: list[str] = []
    skipped_empty: list[str] = []
    flagged_copy: list[str] = []
    distinct_responses_by_group: dict[str, set[str]] = defaultdict(set)
    all_generated_rows = 0

    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        source_example = _mapping(row.get("example"))
        key = str(row.get("example_key") or source_example.get("key") or "").strip()
        if not key:
            continue
        all_generated_rows += 1

        prompt_labels = dict(_mapping(source_example.get("labels")))
        group_id = str(prompt_labels.get("group_id") or key)
        split = str(prompt_labels.get("split") or "train")
        prime_family = str(prompt_labels.get("prime_family") or "")
        prime_condition = str(prompt_labels.get("prime_condition") or "")
        alias_bank = str(prompt_labels.get("alias_bank") or "")
        is_theory_prime = bool(prompt_labels.get("is_theory_prime"))
        capture_enabled = bool(prompt_labels.get("capture_enabled"))
        source_family = str(prompt_labels.get("source_family") or "")
        cue_text = str(prompt_labels.get("cue_text") or "")
        theory_name = str(prompt_labels.get("theory_name") or "")

        finish_reason = str(row.get("finish_reason") or "")
        generated_text = str(row.get("generated_text") or row.get("text") or "")
        source_prompt = _render_prompt_text(source_example.get("prompt") or "")
        token_ids = row.get("generated_token_ids")

        normalized_generated = re.sub(r"\s+", " ", generated_text.strip().lower())
        if normalized_generated:
            distinct_responses_by_group[group_id].add(normalized_generated)

        if finish_reason == "length":
            skipped_length.append(key)
            continue
        if not generated_text.strip() or not source_prompt.strip():
            skipped_empty.append(key)
            continue
        if not capture_enabled:
            continue

        theory_name_metrics = (
            _theory_name_copy_metrics(theory_name=theory_name, generated_text=generated_text)
            if is_theory_prime and bool(theory_name)
            else {"theory_name_mention_count": 0, "repeated_theory_name_copy": False}
        )
        theory_name_copy = bool(theory_name_metrics["repeated_theory_name_copy"])
        cue_overlap = False
        cue_overlap_count = 0
        cue_overlap_fraction = 0.0
        cue_longest_run = 0
        if is_theory_prime:
            cue_overlap, cue_metrics = _near_verbatim_cue_copy(
                cue_text=cue_text,
                generated_text=generated_text,
            )
            cue_overlap_count = int(cue_metrics["cue_overlap_count"])
            cue_overlap_fraction = float(cue_metrics["cue_overlap_fraction"])
            cue_longest_run = int(cue_metrics["cue_longest_run"])
        direct_copy = theory_name_copy or cue_overlap

        if direct_copy:
            flagged_copy.append(key)
            continue

        combined_prompt, token_sections = _combined_prompt_and_sections(
            source_prompt=source_prompt,
            generated_text=generated_text,
        )
        labels = {
            **prompt_labels,
            "generated_text": generated_text,
            "generation_finish_reason": finish_reason,
            "generated_token_count": len(token_ids) if isinstance(token_ids, list) else 0,
            "response_char_length": len(generated_text),
            "theory_name_copy_flag": "yes" if theory_name_copy else "no",
            "theory_name_mention_count": int(theory_name_metrics["theory_name_mention_count"]),
            "cue_overlap_copy_flag": "yes" if cue_overlap else "no",
            "cue_overlap_count": cue_overlap_count,
            "cue_overlap_fraction": round(cue_overlap_fraction, 4),
            "cue_longest_run": cue_longest_run,
            "direct_theory_copy_flag": "yes" if direct_copy else "no",
        }
        metadata = {
            **_mapping(source_example.get("metadata")),
            "source_generation_artifact_id": getattr(generation, "id", ""),
            "token_sections": token_sections,
        }
        examples.append(
            Example(
                key=key,
                prompt=combined_prompt,
                labels=labels,
                metadata=metadata,
                cases={
                    "group_id": group_id,
                    "base_dilemma_id": str(prompt_labels.get("base_dilemma_id") or group_id),
                },
                case_key=group_id,
            )
        )
        generated_text_labels[key] = generated_text
        group_id_labels[key] = group_id
        split_labels[key] = split
        prime_family_labels[key] = prime_family
        prime_condition_labels[key] = prime_condition
        alias_bank_labels[key] = alias_bank
        finish_reason_labels[key] = finish_reason
        token_count_labels[key] = len(token_ids) if isinstance(token_ids, list) else 0
        source_family_labels[key] = source_family
        response_length_labels[key] = len(generated_text)
        theory_name_copy_labels[key] = "yes" if theory_name_copy else "no"
        theory_name_mention_count_labels[key] = int(theory_name_metrics["theory_name_mention_count"])
        cue_overlap_copy_labels[key] = "yes" if cue_overlap else "no"
        direct_copy_labels[key] = "yes" if direct_copy else "no"

    dataset = Dataset.from_examples(examples, name="morebench_phase03_experiment02_generation_capture")

    divergence_counts = {
        group_id: len(responses) for group_id, responses in sorted(distinct_responses_by_group.items())
    }
    divergence_ready = sum(1 for count in divergence_counts.values() if count >= 3)

    return {
        "payload": {
            "kind": "morebench_theory_generation_persistence_capture_dataset",
            "dataset": dataset.to_dict(),
            "summary": {
                "source_generation_artifact_id": getattr(generation, "id", ""),
                "source_row_count": all_generated_rows,
                "kept_capture_example_count": len(examples),
                "skipped_length_count": len(skipped_length),
                "skipped_empty_count": len(skipped_empty),
                "flagged_direct_copy_count": len(flagged_copy),
                "prime_condition_counts": dict(sorted(Counter(prime_condition_labels.values()).items())),
                "source_family_counts": dict(sorted(Counter(source_family_labels.values()).items())),
                "behavioral_divergence_precheck": {
                    "group_distinct_response_counts": divergence_counts,
                    "groups_with_at_least_three_distinct_responses": divergence_ready,
                    "interpretation": "Use this as a cheap divergence pre-check before upgrading any generation-time persistence claim.",
                },
            },
        },
        "labels": {
            "generated_text": generated_text_labels,
            "group_id": group_id_labels,
            "split": split_labels,
            "prime_family": prime_family_labels,
            "prime_condition": prime_condition_labels,
            "alias_bank": alias_bank_labels,
            "generation_finish_reason": finish_reason_labels,
            "generated_token_count": token_count_labels,
            "source_family": source_family_labels,
            "response_char_length": response_length_labels,
            "theory_name_copy_flag": theory_name_copy_labels,
            "theory_name_mention_count": theory_name_mention_count_labels,
            "cue_overlap_copy_flag": cue_overlap_copy_labels,
            "direct_theory_copy_flag": direct_copy_labels,
        },
        "metadata": {
            "source": "GenerationRunSpec result rows",
            "unit": "group_id x prime_condition",
            "status": "length-finished and direct-copy rows removed from the main capture dataset",
        },
        "example_keys": sorted(generated_text_labels),
    }


def summarize_experiment_02(
    generation: Any,
    capture_dataset: Any,
    capture_result: Any,
    text_baseline: Any,
    probe_result: Any,
) -> TransformResult:
    capture_payload = capture_dataset.result() if hasattr(capture_dataset, "result") else {}
    probe_payload = probe_result.result() if hasattr(probe_result, "result") else {}
    text_payload = text_baseline.result() if hasattr(text_baseline, "result") else {}
    capture_summary = dict(capture_payload.get("summary", {})) if isinstance(capture_payload, Mapping) else {}
    probe_layers = list(probe_payload.get("layers", [])) if isinstance(probe_payload, Mapping) else []
    best_probe = max(probe_layers, key=lambda item: float(item.get("balanced_accuracy", 0.0))) if probe_layers else {}
    text_summary = dict(text_payload.get("summary", {})) if isinstance(text_payload, Mapping) else {}
    text_split_results = (
        dict(_mapping(_mapping(text_payload.get("results")).get("split_results")))
        if isinstance(text_payload, Mapping)
        else {}
    )
    text_split = dict(_mapping(text_split_results.get("split")))
    baseline_ba = float(text_split.get("balanced_accuracy", 0.0)) if text_split else 0.0
    probe_delta_by_layer = [
        {
            **dict(_mapping(layer_payload)),
            "probe_minus_text_baseline_balanced_accuracy": round(
                float(_mapping(layer_payload).get("balanced_accuracy", 0.0)) - baseline_ba,
                4,
            ),
        }
        for layer_payload in probe_layers
    ]
    generation_payload = generation.result() if hasattr(generation, "result") else {}
    generation_rows = generation_payload.get("rows", []) if isinstance(generation_payload, Mapping) else []
    finish_reason_counts = Counter(str(row.get("finish_reason") or "") for row in generation_rows if isinstance(row, Mapping))
    return TransformResult(
        payload={
            "benchmark": "morebench",
            "phase": "03",
            "experiment": "experiment_02_theory_conditioned_generation_persistence",
            "generation_finish_reason_counts": dict(sorted(finish_reason_counts.items())),
            "capture_dataset_summary": capture_summary,
            "capture_feature_artifact": getattr(capture_result, "id", ""),
            "generated_text_baseline_summary": text_summary,
            "generated_text_baseline_split": text_split,
            "best_generation_probe": best_probe,
            "probe_minus_text_baseline_by_layer": probe_delta_by_layer,
        }
    )


def build_workflow(
    dataset: Dataset | None = None,
    *,
    dataset_path: Path = DATASET_PATH,
    dataset_name: str = "morebench_phase03_experiment02_generation_batch",
    workflow_name: str = "morebench_phase03_experiment02_theory_generation_persistence",
    report_output_dir: Path = REPORT_OUTPUT_DIR,
    generation_description: str | None = None,
) -> WorkflowSpec:
    dataset = dataset or build_dataset(dataset_path=dataset_path, dataset_name=dataset_name)
    generation_description = generation_description or (
        "Generate one response for every matched dilemma under five description-only theory primes "
        "plus one generic ethics control. This is the broad batch used for divergence checks, "
        "text baselines, and the full-sequence capture run."
    )

    return WorkflowSpec(
        name=workflow_name,
        steps=(
            WorkflowStep(
                name="generate_theory_primed_responses",
                runner="capture_gpu",
                description=generation_description,
                spec=GenerationRunSpec(
                    engine=_engine(),
                    dataset=dataset,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=GENERATION_MAX_TOKENS,
                        temperature=0.0,
                        top_p=1.0,
                    ),
                ),
            ),
            WorkflowStep(
                name="build_theory_persistence_capture_dataset",
                runner="analysis_cpu",
                description=(
                    "Filter the broad generation batch into the main captured dataset. The main dataset excludes "
                    "length-finished rows and theory-copying rows, records generated-text labels for lexical "
                    "baselines, and emits a cheap behavioral-divergence pre-check summary."
                ),
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_theory_persistence_capture_dataset,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_theory_primed_responses")},
                ),
            ),
            WorkflowStep(
                name="capture_generated_sequence_residual",
                runner="capture_gpu",
                description=(
                    "Replay the captured prompt+generation pairs and store the full generated-token residual "
                    "sequence only. Prompt tokens are excluded from the main persistence readout."
                ),
                spec=CaptureSpec(
                    engine=_engine(max_num_seqs=8),
                    dataset=_artifact_capture_dataset(),
                    sites=[
                        ResidualSite(
                            name="generated_sequence_residual",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.section("generated"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        )
                    ],
                    generation=GenerationSpec(enabled=False),
                ),
            ),
            WorkflowStep(
                name="text_baseline_generation_prime_condition",
                runner="analysis_cpu",
                description=(
                    "Lexical baseline on generated text for prime-condition classification. This is the main "
                    "surface-text hurdle the generation-time persistence readout must beat."
                ),
                spec=TextBaselineSpec(
                    text=StepRef("build_theory_persistence_capture_dataset").label("generated_text"),
                    rows=StepRef("build_theory_persistence_capture_dataset").label("group_id"),
                    labels=StepRef("build_theory_persistence_capture_dataset").label("prime_condition"),
                    group_by=StepRef("build_theory_persistence_capture_dataset").label("group_id"),
                    split_by={"split": StepRef("build_theory_persistence_capture_dataset").label("split")},
                    train_values=("train",),
                    test_values=("test",),
                    model="countvectorizer_logreg",
                    metrics=("accuracy", "balanced_accuracy"),
                ),
            ),
            WorkflowStep(
                name="probe_generation_prime_condition_residual",
                runner="analysis_cpu",
                description=(
                    "First-pass generation-time persistence readout. Mean-pools over the full generated section "
                    "to detect broad generation-time signal before any later endpoint localization."
                ),
                spec=ProbeSpec(
                    feature=StepRef("capture_generated_sequence_residual").feature("generated_sequence_residual"),
                    rows=StepRef("build_theory_persistence_capture_dataset").label("group_id"),
                    labels=StepRef("build_theory_persistence_capture_dataset").label("prime_condition"),
                    group_by=StepRef("build_theory_persistence_capture_dataset").label("group_id"),
                    split=StepRef("build_theory_persistence_capture_dataset").label("split"),
                    train_values=("train",),
                    test_values=("test",),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.mean(),
                    metrics=("accuracy", "balanced_accuracy", "selectivity"),
                    baselines=("majority", "shuffled_label"),
                ),
            ),
            WorkflowStep(
                name="summarize_experiment_02",
                runner="analysis_cpu",
                description=(
                    "Summarize finish reasons, the behavioral-divergence pre-check, copy filtering, the generated-text "
                    "baseline, and the first-pass generation-time persistence readout."
                ),
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_experiment_02,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={
                        "generation": StepRef("generate_theory_primed_responses"),
                        "capture_dataset": StepRef("build_theory_persistence_capture_dataset"),
                        "capture_result": StepRef("capture_generated_sequence_residual"),
                        "text_baseline": StepRef("text_baseline_generation_prime_condition"),
                        "probe_result": StepRef("probe_generation_prime_condition_residual"),
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                description=(
                    "Assemble the Experiment 2 report. The main interpretation question is whether generated-token "
                    "signal survives the generated-text baseline after copy-flag filtering and divergence checks."
                ),
                spec=ReportSpec(
                    inputs=(
                        StepRef("generate_theory_primed_responses"),
                        StepRef("build_theory_persistence_capture_dataset"),
                        StepRef("capture_generated_sequence_residual"),
                        StepRef("text_baseline_generation_prime_condition"),
                        StepRef("probe_generation_prime_condition_residual"),
                        StepRef("summarize_experiment_02"),
                    ),
                    template="default",
                    output_dir=report_output_dir,
                ),
            ),
        ),
    )
