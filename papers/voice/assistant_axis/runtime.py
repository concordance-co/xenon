"""Workflow helper surface for the Assistant Axis paper package."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines_v2.api import (
    Dataset,
    Example,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    TransformResult,
    VLLMEngine,
)

from papers.voice.assistant_axis.paper import (
    PAPER_ACTIVATION_CONFIG,
    PAPER_GENERATION_CONFIG,
    PAPER_JUDGE_CONFIG,
    SUPPORTED_RELEASED_MODELS,
    expand_source_row,
)
from papers.voice.storage import (
    ARTIFACT_VOLUME_NAME,
    MODEL_VOLUME_PATH,
    YORA_MODEL_VOLUME_NAME,
    local_vector_root,
    modal_vector_root,
)


MODEL_VOLUME_NAME = YORA_MODEL_VOLUME_NAME
DEFAULT_MODEL_KEY = "llama_3_3_70b"
PAPER_ENV_PATH = Path(__file__).with_name(".env")


def load_paper_env(path: str | Path = PAPER_ENV_PATH) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_paper_env()


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_list(name: str, default: Sequence[str]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return tuple(default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def model_key_from_env() -> str:
    key = os.getenv("ASSISTANT_AXIS_MODEL_KEY", DEFAULT_MODEL_KEY).strip()
    if key not in SUPPORTED_RELEASED_MODELS:
        known = ", ".join(sorted(SUPPORTED_RELEASED_MODELS))
        raise ValueError(f"Unknown ASSISTANT_AXIS_MODEL_KEY={key!r}; expected one of: {known}")
    return key


def model_config(model_key: str | None = None) -> dict[str, Any]:
    return dict(SUPPORTED_RELEASED_MODELS[model_key or model_key_from_env()])


def model_id(model_key: str | None = None) -> str:
    return str(model_config(model_key)["model_id"])


def target_layer(model_key: str | None = None) -> int:
    return int(model_config(model_key)["target_layer"])


def model_short_name(model_key: str | None = None) -> str:
    key = model_key or model_key_from_env()
    if key == "gemma_2_27b":
        return "Gemma"
    if key == "qwen_3_32b":
        return "Qwen"
    if key == "llama_3_3_70b":
        return "Llama"
    return "{model_name}"


def vllm_engine(
    *,
    model_key: str | None = None,
    max_model_len: int | None = None,
    max_num_seqs: int = 1,
    patched: bool = False,
    add_generation_prompt: bool = True,
) -> VLLMEngine:
    return VLLMEngine(
        model_id=model_id(model_key),
        model_path_root=MODEL_VOLUME_PATH,
        max_model_len=int(max_model_len or PAPER_GENERATION_CONFIG.max_model_len),
        tensor_parallel_size=4 if patched else 2,
        gpu_memory_utilization=0.88,
        enforce_eager=bool(patched),
        max_num_seqs=int(max_num_seqs),
        enable_prefix_caching=not patched,
        add_generation_prompt=bool(add_generation_prompt),
    )


def runner_specs(
    *,
    workflow_name: str,
    local_artifact_root: str | Path | None = None,
    include_patch_env: bool = False,
    include_judge_env: bool = False,
) -> dict[str, object]:
    hf_secret = ModalSecret.from_env_var("HF_TOKEN", secret_name="huggingface")
    analysis_secrets = [hf_secret]
    if include_judge_env:
        judge_api_key_env = os.getenv("ASSISTANT_AXIS_JUDGE_API_KEY_ENV", "OPENAI_API_KEY")
        judge_secret_name = os.getenv("ASSISTANT_AXIS_JUDGE_SECRET_NAME", judge_api_key_env)
        analysis_secrets.append(ModalSecret.from_env_var(judge_api_key_env, secret_name=judge_secret_name))
    model_mount = ModalVolumeMount(
        name=MODEL_VOLUME_NAME,
        mount_path=MODEL_VOLUME_PATH,
        create_if_missing=True,
        commit_on_success=True,
    )
    shared_env = {
        "HF_HOME": f"{MODEL_VOLUME_PATH}/hf_home",
        "TRANSFORMERS_CACHE": f"{MODEL_VOLUME_PATH}/hf_home/transformers",
    }
    if include_patch_env:
        shared_env["XENON_ACTIVATION_PATCH_MAX_TOKENS"] = "1"
    modal_store = ModalVolumeStore(
        name=ARTIFACT_VOLUME_NAME,
        root=modal_vector_root("assistant-axis", workflow_name),
    )
    local_root = Path(local_artifact_root or local_vector_root("assistant-axis", workflow_name))
    return {
        "generation_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H100:4" if include_patch_env else "H100:2",
                cpu=8,
                memory_mb=96 * 1024,
                timeout_seconds=60 * 60 * 3,
                max_containers=1,
                env=shared_env,
                secrets=(hf_secret,),
                volumes=(model_mount,),
            ),
            artifacts=modal_store,
        ),
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H100:2",
                cpu=8,
                memory_mb=96 * 1024,
                timeout_seconds=60 * 60 * 3,
                max_containers=1,
                env=shared_env,
                secrets=(hf_secret,),
                volumes=(model_mount,),
            ),
            artifacts=modal_store,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=4,
                memory_mb=16 * 1024,
                timeout_seconds=60 * 45,
                env=shared_env,
                secrets=tuple(analysis_secrets),
                volumes=(model_mount,),
            ),
            artifacts=modal_store,
        ),
        "report_local": LocalRunnerSpec(artifacts=LocalArtifactStore(local_root)),
    }


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL line {line_number} must be an object")
            records.append(dict(payload))
    return records


def trace_dataset_from_records(records: Sequence[Mapping[str, Any]], *, name: str) -> Dataset:
    examples: list[Example] = []
    for index, row in enumerate(records):
        text = str(row.get("text") or row.get("trace") or row.get("prompt") or "")
        if not text.strip():
            raise ValueError(f"Trace row {index} is missing non-empty text/trace/prompt")
        span = _span_from_row(row, text=text)
        labels = dict(row.get("labels") or {})
        for key in ("axis_kind", "role", "adherence_score", "model_id", "split", "surface", "source"):
            if key in row and key not in labels:
                labels[key] = row[key]
        examples.append(
            Example(
                key=str(row.get("example_id") or row.get("key") or f"trace_{index:06d}"),
                prompt=text,
                labels=labels,
                metadata={
                    **(dict(row.get("metadata") or {})),
                    "token_sections": {"assistant_response": span},
                    "section_records": [
                        {
                            "name": "assistant_response",
                            "unit": "turn",
                            "role": "assistant",
                            "index": int(row.get("turn_index", 0)),
                            "char_start": int(span["char_start"]),
                            "char_end": int(span["char_end"]),
                        }
                    ],
                },
            )
        )
    return Dataset.from_examples(examples, name=name)


def byod_dataset_from_env(*, workflow_name: str) -> Dataset:
    path = os.getenv("ASSISTANT_AXIS_BYOD_JSONL")
    if path:
        return trace_dataset_from_records(load_jsonl_records(path), name=f"{workflow_name}_byod")
    return trace_dataset_from_records(
        [
            {
                "example_id": "default_0",
                "text": "Human: Explain how to triage a failed deploy.\n\nAssistant: Check the failing step, verify the migration state, and choose rollback or fix-forward based on observed impact.",
                "axis_kind": "default",
                "role": "default",
                "adherence_score": 0,
            },
            {
                "example_id": "default_1",
                "text": "Human: Explain how to triage a failed deploy.\n\nAssistant: Separate symptoms from causes, inspect logs, and avoid rerunning destructive steps until the database state is known.",
                "axis_kind": "default",
                "role": "default",
                "adherence_score": 0,
            },
            {
                "example_id": "role_terse_0",
                "text": "Human: Explain how to triage a failed deploy.\n\nAssistant: Logs first. Confirm the migration version. Freeze writes. Roll back only after checking whether partial writes occurred.",
                "axis_kind": "role",
                "role": "terse_operator",
                "adherence_score": 3,
            },
            {
                "example_id": "role_teacher_0",
                "text": "Human: Explain how to triage a failed deploy.\n\nAssistant: Think of it as narrowing a fault tree: identify the exact step, preserve evidence, then choose the least risky recovery path.",
                "axis_kind": "role",
                "role": "teacher",
                "adherence_score": 3,
            },
            {
                "example_id": "probe_0",
                "text": "Human: The deploy failed and people are escalating. What now?\n\nAssistant: Pause new changes, identify whether the database is partially migrated, and communicate one recovery owner plus the next checkpoint.",
                "axis_kind": "probe",
                "role": "probe",
                "adherence_score": 0,
            },
        ],
        name=f"{workflow_name}_byod_fixture",
    )


def byot_dataset_from_env(*, workflow_name: str) -> Dataset:
    path = os.getenv("ASSISTANT_AXIS_BYOT_JSONL")
    if path:
        return trace_dataset_from_records(load_jsonl_records(path), name=f"{workflow_name}_byot")
    trace = os.getenv(
        "ASSISTANT_AXIS_BYOT_TRACE",
        "Human: The migration failed and the incident channel is getting noisy. What should I do?\n\n"
        "Assistant: First, stop additional deploy attempts so the state stays inspectable. Then identify the exact migration revision, check whether any writes completed, and name one person to coordinate updates while the technical owner chooses rollback or fix-forward.",
    )
    return trace_dataset_from_records(
        [{"example_id": "trace_0", "text": trace, "surface": "byot_default"}],
        name=f"{workflow_name}_byot_fixture",
    )


def byop_dataset_from_env(*, workflow_name: str) -> Dataset:
    prompt = os.getenv(
        "ASSISTANT_AXIS_BYOP_PROMPT",
        "The production deploy failed during a database migration and the team is tense. Write a short assistant response that helps me decide what to do next.",
    )
    return Dataset.from_examples(
        [
            Example(
                key="prompt_0",
                prompt=[{"role": "user", "content": prompt}],
                labels={"surface": "byop", "trait": os.getenv("ASSISTANT_AXIS_STEERING_TRAIT", "calm")},
            )
        ],
        name=f"{workflow_name}_byop_prompt",
    )


def build_paper_generation_prompt_dataset(
    *,
    source_dataset: Any,
    model_key: str,
    role_limit: int,
    question_limit: int,
    instruction_limit: int,
) -> TransformResult:
    dataset = _dataset_from_value(source_dataset).resolve()
    default_rows: list[Mapping[str, Any]] = []
    role_rows: list[Mapping[str, Any]] = []
    for example in dataset.examples:
        labels = dict(example.labels)
        row = {"key": example.key, "labels": labels, "metadata": dict(example.metadata)}
        if labels.get("is_default") or labels.get("source_type") == "default":
            default_rows.append(row)
        else:
            role_rows.append(row)
    selected_sources = [*default_rows[:1], *role_rows[: int(role_limit)]]
    expanded = []
    short_name = model_short_name(model_key)
    for source in selected_sources:
        for row in expand_source_row(source, model_short_name=short_name):
            instruction_index = int(row.example_id.rsplit("_i", 1)[1].split("_q", 1)[0])
            question_index = int(row.example_id.rsplit("_q", 1)[1])
            if instruction_index >= int(instruction_limit) or question_index >= int(question_limit):
                continue
            expanded.append(row)
    prompt_dataset = Dataset.from_examples(
        [
            Example(
                key=row.example_id,
                prompt=row.prompt,
                labels={
                    "axis_kind": row.axis_kind,
                    "role": row.role,
                    "source_name": row.source_name,
                },
                metadata={
                    "instruction": row.instruction,
                    "question": row.question,
                    "paper_source": "assistant_axis",
                    "paper_adherence_status": "unjudged",
                },
            )
            for row in expanded
        ],
        name=f"assistant_axis_paper_prompts_{model_key}",
    )
    return TransformResult(
        payload={
            "kind": "assistant_axis_paper_generation_prompt_dataset",
            "dataset": prompt_dataset.to_dict(),
            "summary": {
                "model_key": model_key,
                "source_count": len(selected_sources),
                "prompt_count": len(prompt_dataset.examples),
                "role_limit": int(role_limit),
                "question_limit": int(question_limit),
                "instruction_limit": int(instruction_limit),
                "paper_generation": asdict(PAPER_GENERATION_CONFIG),
                "paper_activation": asdict(PAPER_ACTIVATION_CONFIG),
                "paper_judge": asdict(PAPER_JUDGE_CONFIG),
            },
        },
        example_keys=prompt_dataset.example_keys(),
    )


def generation_result_to_axis_capture_dataset(
    *,
    generation: Any,
    fallback_axis_kind: str = "probe",
    fallback_role: str = "probe",
    surface: str = "generated",
) -> TransformResult:
    payload = _artifact_result(generation)
    rows = payload.get("rows") if isinstance(payload, Mapping) else []
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows if isinstance(rows, list) else []):
        if not isinstance(row, Mapping):
            continue
        example = row.get("example") if isinstance(row.get("example"), Mapping) else {}
        labels = dict(example.get("labels") or {})
        prompt_text = _prompt_to_text(example.get("prompt"))
        generated_text = str(row.get("generated_text") or "")
        trace = f"{prompt_text}\n\nAssistant: {generated_text}".strip()
        records.append(
            {
                "example_id": f"{surface}_{example.get('key') or index}",
                "text": trace,
                "assistant_response": _assistant_span(trace),
                "axis_kind": labels.get("axis_kind", fallback_axis_kind),
                "role": labels.get("role", fallback_role),
                "adherence_score": labels.get("adherence_score", PAPER_JUDGE_CONFIG.fully_role_playing_score),
                "surface": surface,
                "metadata": {
                    "source_example_key": example.get("key"),
                    "finish_reason": row.get("finish_reason"),
                },
            }
        )
    dataset = trace_dataset_from_records(records, name=f"assistant_axis_{surface}_capture")
    return TransformResult(
        payload={
            "kind": "assistant_axis_generated_capture_dataset",
            "dataset": dataset.to_dict(),
            "summary": {"surface": surface, "example_count": len(dataset.examples)},
        },
        example_keys=dataset.example_keys(),
    )


def summarize_byop_generations(*, baseline: Any, steered: Any, score: Any | None = None) -> TransformResult:
    baseline_payload = _artifact_result(baseline)
    steered_payload = _artifact_result(steered)
    score_payload = _artifact_result(score) if score is not None else {}
    return TransformResult(
        payload={
            "kind": "assistant_axis_byop_summary",
            "baseline_text": _first_generation_text(baseline_payload),
            "steered_text": _first_generation_text(steered_payload),
            "score_summary": score_payload.get("summary") if isinstance(score_payload, Mapping) else None,
        },
        example_keys=["prompt_0"],
    )


def coordinate_to_unit_direction(*, coordinate: Any, name: str) -> TransformResult:
    payload = _artifact_result(coordinate)
    layers = payload.get("layers")
    if not isinstance(layers, Mapping) or not layers:
        raise ValueError("coordinate payload must contain non-empty layers")
    direction_layers: dict[str, dict[str, Any]] = {}
    for layer, raw_layer_payload in layers.items():
        if not isinstance(raw_layer_payload, Mapping):
            continue
        vector = raw_layer_payload.get("vector")
        if vector is None:
            raise ValueError(f"coordinate layer {layer!r} is missing vector")
        direction_layers[str(layer)] = {
            **dict(raw_layer_payload),
            "vector": list(vector),
            "raw_vector": list(vector),
            "norm": 1.0,
        }
    return TransformResult(
        payload={
            "kind": "direction_result",
            "feature": payload.get("feature"),
            "name": str(name),
            "layers": direction_layers,
            "metadata": {
                **(dict(payload.get("metadata")) if isinstance(payload.get("metadata"), Mapping) else {}),
                "source": "released_assistant_axis_coordinate_unit_direction",
            },
            "summary": {"layer_count": len(direction_layers), "unit_raw_vector": True},
        },
        example_keys=[],
    )


def _dataset_from_value(value: Any) -> Dataset:
    if isinstance(value, Dataset):
        return value
    if isinstance(value, Mapping):
        return Dataset.from_dict(value)
    if hasattr(value, "result"):
        result = value.result()
        if isinstance(result, Mapping) and isinstance(result.get("dataset"), Mapping):
            return Dataset.from_dict(result["dataset"])
    raise TypeError(f"Cannot coerce {type(value).__name__} to Dataset")


def _artifact_result(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "result"):
        result = value.result()
        return result if isinstance(result, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def _span_from_row(row: Mapping[str, Any], *, text: str) -> dict[str, int]:
    raw = row.get("assistant_response")
    if isinstance(raw, Mapping):
        return {"char_start": int(raw["char_start"]), "char_end": int(raw["char_end"])}
    return _assistant_span(text)


def _assistant_span(text: str) -> dict[str, int]:
    marker = "Assistant:"
    start = text.rfind(marker)
    if start < 0:
        return {"char_start": 0, "char_end": len(text)}
    start += len(marker)
    while start < len(text) and text[start].isspace():
        start += 1
    return {"char_start": start, "char_end": len(text)}


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for item in prompt:
            if isinstance(item, Mapping):
                role = str(item.get("role") or "user").title()
                content = str(item.get("content") or "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    return str(prompt)


def _first_generation_text(payload: Mapping[str, Any]) -> str:
    rows = payload.get("rows", [])
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, Mapping) and isinstance(row.get("generated_text"), str):
            return str(row["generated_text"]).strip()
    return ""
