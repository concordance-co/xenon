from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
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
GENERATION_ARTIFACT_ENV_VAR = "MOREBENCH_BROAD_GENERATION_ARTIFACT_ID"
WORKFLOW_NAME = "morebench_phase03_experiment02_behavior_broad_replay_capture"
MODAL_ARTIFACT_ROOT = "/data/artifacts/morebench_phase_03_experiment02_behavior_broad_capture"
PUBLIC_EXTENSION_PREFIX = "public_conflict_"
TARGET_PUBLIC_CONFLICT_GROUP_IDS = frozenset(
    {
        "public_conflict_004",
        "public_conflict_010",
        "public_conflict_011",
        "public_conflict_014",
        "public_conflict_020",
        "public_conflict_023",
        "public_conflict_030",
        "public_conflict_033",
        "public_conflict_037",
        "public_conflict_039",
        "public_conflict_050",
        "public_conflict_051",
        "public_conflict_052",
        "public_conflict_059",
        "public_conflict_060",
    }
)
PRIME_COUNT = 6


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
    artifact_id = os.getenv(GENERATION_ARTIFACT_ENV_VAR, "").strip()
    if not artifact_id:
        raise RuntimeError(
            f"Set {GENERATION_ARTIFACT_ENV_VAR} to a completed broad-generation artifact id before running this workflow."
        )
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
    skipped_non_target_groups: list[str] = []

    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        source_example = _mapping(row.get("example"))
        key = str(row.get("example_key") or source_example.get("key") or "").strip()
        if not key:
            continue
        labels = dict(_mapping(source_example.get("labels")))
        group_id = str(labels.get("group_id") or key)
        if not group_id.startswith(PUBLIC_EXTENSION_PREFIX):
            skipped_non_target_groups.append(key)
            continue
        if group_id not in TARGET_PUBLIC_CONFLICT_GROUP_IDS:
            skipped_non_target_groups.append(key)
            continue
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
            "capture_tier": "behavior_conflict_replay",
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
        name="morebench_phase03_experiment02_behavior_broad_replay_capture_dataset",
    )
    return {
        "payload": {
            "kind": "morebench_broad_behavior_replay_capture_dataset",
            "dataset": dataset.to_dict(),
            "summary": {
                "source_generation_artifact_id": artifact_id,
                "source_row_count": len(raw_rows),
                "capture_example_count": len(examples),
                "target_group_count": len(TARGET_PUBLIC_CONFLICT_GROUP_IDS),
                "target_group_ids": sorted(TARGET_PUBLIC_CONFLICT_GROUP_IDS),
                "skipped_non_target_group_count": len(skipped_non_target_groups),
                "empty_or_missing_prompt_count": len(empty_or_missing_prompt),
                "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
                "group_size_distribution": dict(sorted(Counter(kept_by_group.values()).items())),
                "captured_group_prefix": PUBLIC_EXTENSION_PREFIX,
            },
        }
    }


def build_dataset() -> Dataset:
    payload = build_replay_capture_dataset()["payload"]
    return Dataset.from_dict(payload["dataset"])


def summarize_replay_capture(*, capture_result: Any) -> TransformResult:
    summary = {
        "source_generation_artifact_id": os.getenv(GENERATION_ARTIFACT_ENV_VAR, "").strip(),
        "target_group_count": len(TARGET_PUBLIC_CONFLICT_GROUP_IDS),
        "target_group_ids": sorted(TARGET_PUBLIC_CONFLICT_GROUP_IDS),
        "capture_example_count": len(TARGET_PUBLIC_CONFLICT_GROUP_IDS) * PRIME_COUNT,
    }
    return TransformResult(
        payload={
            "workflow": WORKFLOW_NAME,
            "capture_dataset_summary": summary,
            "capture_feature_artifact_id": getattr(capture_result, "id", ""),
            "captured_layers": list(base.CAPTURED_LAYERS),
            "token_section": "generated",
            "note": (
                "Replay capture over the manually judged public conflict groups only; existing benchmark theory "
                "groups are excluded because prior captures already cover them. No copy filtering applied."
            ),
        }
    )


def _artifact_capture_dataset() -> Dataset:
    return Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef("build_replay_capture_dataset"),
        result_key="dataset",
        provides_token_sections=True,
        name="morebench_phase03_experiment02_behavior_broad_replay_capture_dataset",
    )


def build_runner_specs() -> dict[str, object]:
    secrets = [ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon")]
    catalog = PostgresCatalog(source=PostgresSource.from_env(DB_ENV_VAR))
    modal_store = ModalVolumeStore(name="xenon-data", root=MODAL_ARTIFACT_ROOT)
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 6,
                shard_count=base.GPU_SHARD_COUNT,
                secrets=tuple(secrets),
                volumes=(ModalVolumeMount(name=base.MODEL_VOLUME_NAME, mount_path=base.MODEL_VOLUME_PATH),),
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=8,
                memory_mb=24 * 1024,
                timeout_seconds=60 * 60 * 2,
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
                    "Replay the judged public conflict rows and capture the generated-token residual sequence."
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
                description="Summarize the replay-capture dataset and resulting activation artifact.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_replay_capture,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={
                        "capture_result": StepRef("capture_generated_sequence_residual"),
                    },
                ),
            ),
        ),
    )
