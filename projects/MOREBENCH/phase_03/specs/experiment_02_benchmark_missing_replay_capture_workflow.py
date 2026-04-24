from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from pipelines_v2.api import (
    ArtifactDatasetSource,
    CaptureSpec,
    Dataset,
    Example,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PostgresCatalog,
    PostgresSource,
    ResidualSite,
    StepRef,
    TensorStorage,
    TokenSelector,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)
from pipelines_v2.storage.artifacts import OperationArtifact, artifact_from_manifest

from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
GENERATION_ARTIFACT_ENV_VAR = "MOREBENCH_BENCHMARK_GENERATION_ARTIFACT_ID"
DEFAULT_GENERATION_ARTIFACT_ID = "generation_run_1_3d4009fb21d8"
WORKFLOW_NAME = "morebench_phase03_experiment02_benchmark_missing_replay_capture"
MODAL_ARTIFACT_ROOT = "/data/artifacts/morebench_phase_03_experiment02_benchmark_missing_capture"
PRIME_COUNT = 10

TARGET_EXAMPLE_KEYS = frozenset(
    {
        "theory_group_005__contractualism__description_c",
        "theory_group_009__contractualism__description_c",
        "theory_group_011__utilitarian__description_c",
        "theory_group_011__contractualism__description_c",
        "theory_group_013__contractualism__description_a",
        "theory_group_015__virtue_ethics__description_b",
        "theory_group_015__contractualism__description_b",
        "theory_group_022__utilitarian__description_b",
        "theory_group_022__virtue_ethics__description_b",
        "theory_group_022__contractualism__description_b",
    }
)


def _catalog() -> PostgresCatalog:
    return PostgresCatalog(source=PostgresSource.from_env(DB_ENV_VAR))


def _store() -> ModalVolumeStore:
    return ModalVolumeStore(name="xenon-data", root=base.MODAL_ARTIFACT_ROOT)


def _load_operation_artifact(artifact_id: str) -> OperationArtifact:
    manifest = _catalog().load_artifact(artifact_id)
    if manifest is None:
        raise RuntimeError(f"Could not load artifact manifest {artifact_id!r}")
    artifact = artifact_from_manifest(manifest, store=_store())
    if not isinstance(artifact, OperationArtifact):
        raise TypeError(f"Artifact {artifact_id!r} is not an operation artifact")
    return artifact


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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


def build_replay_capture_dataset() -> dict[str, Any]:
    artifact_id = os.getenv(GENERATION_ARTIFACT_ENV_VAR, "").strip() or DEFAULT_GENERATION_ARTIFACT_ID
    generation = _load_operation_artifact(artifact_id)
    payload = generation.result()
    if not isinstance(payload, Mapping):
        raise TypeError("generation artifact result must be a mapping")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError("generation artifact result must contain a rows list")

    examples: list[Example] = []
    finish_reason_counts: Counter[str] = Counter()
    kept_by_group: Counter[str] = Counter()
    empty_or_missing_prompt: list[str] = []
    skipped_non_target_rows: list[str] = []

    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        source_example = _mapping(row.get("example"))
        key = str(row.get("example_key") or source_example.get("key") or "").strip()
        if not key:
            continue
        if key not in TARGET_EXAMPLE_KEYS:
            skipped_non_target_rows.append(key)
            continue
        labels = dict(_mapping(source_example.get("labels")))
        group_id = str(labels.get("group_id") or key)
        prompt_text = _render_prompt_text(source_example.get("prompt") or "")
        generated_text = str(row.get("generated_text") or row.get("text") or "")
        finish_reason = str(row.get("finish_reason") or "")
        finish_reason_counts[finish_reason] += 1

        if not prompt_text.strip() or not generated_text.strip():
            empty_or_missing_prompt.append(key)
            continue

        combined_prompt, token_sections = _combined_prompt_and_sections(
            source_prompt=prompt_text,
            generated_text=generated_text,
        )
        metadata = {
            **_mapping(source_example.get("metadata")),
            "token_sections": token_sections,
            "source_generation_artifact_id": artifact_id,
        }
        example_labels = {
            **labels,
            "generated_text": generated_text,
            "generation_finish_reason": finish_reason,
            "generated_token_count": len(row.get("generated_token_ids") or []),
            "response_char_length": len(generated_text),
            "capture_enabled": True,
            "capture_tier": "benchmark_missing_replay",
        }
        base_dilemma_id = str(labels.get("base_dilemma_id") or group_id)
        examples.append(
            Example(
                key=key,
                prompt=combined_prompt,
                labels=example_labels,
                metadata=metadata,
                cases={"group_id": group_id, "base_dilemma_id": base_dilemma_id},
                case_key=group_id,
            )
        )
        kept_by_group[group_id] += 1

    dataset = Dataset.from_examples(
        examples,
        name="morebench_phase03_experiment02_benchmark_missing_replay_capture_dataset",
    )
    return {
        "payload": {
            "kind": "morebench_benchmark_missing_replay_capture_dataset",
            "dataset": dataset.to_dict(),
            "summary": {
                "source_generation_artifact_id": artifact_id,
                "source_row_count": len(raw_rows),
                "capture_example_count": len(examples),
                "target_example_count": len(TARGET_EXAMPLE_KEYS),
                "target_example_keys": sorted(TARGET_EXAMPLE_KEYS),
                "skipped_non_target_row_count": len(skipped_non_target_rows),
                "empty_or_missing_prompt_count": len(empty_or_missing_prompt),
                "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
                "group_size_distribution": dict(sorted(Counter(kept_by_group.values()).items())),
            },
        }
    }


def build_dataset() -> Dataset:
    payload = build_replay_capture_dataset()["payload"]
    return Dataset.from_dict(payload["dataset"])


def summarize_replay_capture(*, capture_result: Any) -> TransformResult:
    summary = {
        "source_generation_artifact_id": os.getenv(GENERATION_ARTIFACT_ENV_VAR, "").strip() or DEFAULT_GENERATION_ARTIFACT_ID,
        "target_example_count": len(TARGET_EXAMPLE_KEYS),
        "target_example_keys": sorted(TARGET_EXAMPLE_KEYS),
        "capture_example_count": len(TARGET_EXAMPLE_KEYS),
    }
    return TransformResult(
        payload={
            "workflow": WORKFLOW_NAME,
            "capture_dataset_summary": summary,
            "capture_feature_artifact_id": getattr(capture_result, "id", ""),
            "captured_layers": list(base.CAPTURED_LAYERS),
            "token_section": "generated",
            "note": (
                "Replay capture over the exact benchmark conflict rows that were missing from the old filtered "
                "main Experiment 2 capture. No copy filtering applied."
            ),
        }
    )


def _artifact_capture_dataset() -> Dataset:
    return Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef("build_replay_capture_dataset"),
        result_key="dataset",
        provides_token_sections=True,
        name="morebench_phase03_experiment02_benchmark_missing_replay_capture_dataset",
    )


def build_runner_specs() -> dict[str, object]:
    secrets = [ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon")]
    catalog = PostgresCatalog(source=PostgresSource.from_env(DB_ENV_VAR))
    modal_store = ModalVolumeStore(name="xenon-data", root=MODAL_ARTIFACT_ROOT)
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 4,
                shard_count=1,
                secrets=tuple(secrets),
                volumes=(ModalVolumeMount(name=base.MODEL_VOLUME_NAME, mount_path=base.MODEL_VOLUME_PATH),),
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=4,
                memory_mb=12 * 1024,
                timeout_seconds=60 * 60,
                secrets=tuple(secrets),
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
    }


def build_workflow() -> WorkflowSpec:
    dataset = build_dataset()
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="capture_generated_sequence_residual",
                runner="capture_gpu",
                description=(
                    "Replay the benchmark conflict rows missing from the old filtered capture and capture the "
                    "generated-token residual sequence."
                ),
                spec=CaptureSpec(
                    engine=base._engine(max_num_seqs=8),
                    dataset=dataset,
                    sites=[
                        ResidualSite(
                            name="generated_sequence_residual",
                            site="resid_post",
                            layers=list(base.CAPTURED_LAYERS),
                            tokens=TokenSelector.section("generated"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        )
                    ],
                    generation=base.GenerationSpec(enabled=False),
                ),
            ),
            WorkflowStep(
                name="summarize_replay_capture",
                runner="analysis_cpu",
                description="Summarize the benchmark-missing replay capture and resulting activation artifact.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_replay_capture,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"capture_result": StepRef("capture_generated_sequence_residual")},
                ),
            ),
        ),
    )
