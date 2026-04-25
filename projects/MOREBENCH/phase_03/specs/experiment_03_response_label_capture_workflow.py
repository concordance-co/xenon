from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    Example,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PostgresCatalog,
    PostgresSource,
    ReportSpec,
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

from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
WORKFLOW_NAME = "morebench_phase03_experiment03_response_label_capture"
MODAL_ARTIFACT_ROOT = "/data/artifacts/morebench_phase_03_experiment03_response_labels"
PHASE_ROOT = Path("projects/MOREBENCH/phase_03")
GENERATION_RESULT_PATH = Path("artifacts/_modal_cache/generation_run_1_d6e12a467208/result.json")
RUBRIC_SCORE_PATH = PHASE_ROOT / "reports/experiment_03_full_public_rubric_judge/manual_scores_merged.jsonl"
PROBE_SUMMARY_PATH = PHASE_ROOT / "reports/experiment_03_response_label_probe/summary.json"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "morebench_phase_03_experiment03_response_labels"
REPORT_OUTPUT_DIR = PHASE_ROOT / "reports" / "experiment_03_response_label_capture_report"
CAPTURED_LAYERS = (0, 4, 8, 16, 28, 36, 40, 44)
REPORT_LABELS = ("helpful_harmless_off_diagonal", "strong_helpful", "strong_harmless")
REPORT_VIEWS = (
    "prompt_end",
    "generated_first_third",
    "generated_middle_third",
    "generated_last_third",
    "generated_total",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


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
        "prompt_end": _last_non_whitespace_span(combined, 0, prompt_end),
        "generated": {"char_start": generated_start, "char_end": generated_end},
        "full": {"char_start": 0, "char_end": generated_end},
        "generated_end": _last_non_whitespace_span(combined, generated_start, generated_end),
        "full_end": _last_non_whitespace_span(combined, 0, generated_end),
    }


def _off_diagonal_label(helpful_score: int, harmless_score: int) -> str:
    if helpful_score == 3 and harmless_score < 3:
        return "helpful_over_harmless"
    if helpful_score < 3 and harmless_score == 3:
        return "harmless_over_helpful"
    return ""


def build_dataset() -> Dataset:
    generation = json.loads(GENERATION_RESULT_PATH.read_text())
    rubric_by_id = {row["row_id"]: row for row in _read_jsonl(RUBRIC_SCORE_PATH)}
    examples: list[Example] = []
    finish_reason_counts: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = {
        "strong_helpful": Counter(),
        "strong_harmless": Counter(),
        "helpful_harmless_off_diagonal": Counter(),
    }

    for row in generation["rows"]:
        source_example = _mapping(row.get("example"))
        row_id = str(row.get("example_key") or source_example.get("key") or "").strip()
        rubric = rubric_by_id.get(row_id)
        if not row_id or rubric is None:
            continue
        prompt_text = _render_prompt_text(source_example.get("prompt") or "")
        generated_text = str(row.get("generated_text") or "")
        finish_reason = str(row.get("finish_reason") or "")
        finish_reason_counts[finish_reason] += 1
        if not prompt_text.strip() or not generated_text.strip():
            continue

        helpful_score = int(rubric["helpful_score"])
        harmless_score = int(rubric["harmless_score"])
        strong_helpful = "true" if helpful_score == 3 else "false"
        strong_harmless = "true" if harmless_score == 3 else "false"
        off_diagonal = _off_diagonal_label(helpful_score, harmless_score)
        label_counts["strong_helpful"][strong_helpful] += 1
        label_counts["strong_harmless"][strong_harmless] += 1
        if off_diagonal:
            label_counts["helpful_harmless_off_diagonal"][off_diagonal] += 1

        source_labels = dict(_mapping(source_example.get("labels")))
        combined_prompt, token_sections = _combined_prompt_and_sections(
            source_prompt=prompt_text,
            generated_text=generated_text,
        )
        labels = {
            **source_labels,
            "row_id": row_id,
            "group_id": row_id,
            "strong_helpful": strong_helpful,
            "strong_harmless": strong_harmless,
            "helpful_harmless_off_diagonal": off_diagonal,
            "helpful_score": helpful_score,
            "harmless_score": harmless_score,
            "generation_finish_reason": finish_reason,
            "generated_token_count": len(row.get("generated_token_ids") or []),
            "response_char_length": len(generated_text),
        }
        metadata = {
            **_mapping(source_example.get("metadata")),
            "token_sections": token_sections,
            "source_generation_artifact_id": "generation_run_1_d6e12a467208",
        }
        examples.append(
            Example(
                key=row_id,
                prompt=combined_prompt,
                labels=labels,
                metadata=metadata,
                cases={"group_id": row_id, "base_dilemma_id": row_id},
                case_key=row_id,
            )
        )

    return Dataset.from_examples(examples, name="morebench_phase03_experiment03_response_label_capture")


def summarize_capture(*, capture_result: Any) -> TransformResult:
    return TransformResult(
        payload={
            "workflow": WORKFLOW_NAME,
            "capture_feature_artifact_id": getattr(capture_result, "id", ""),
            "captured_layers": list(CAPTURED_LAYERS),
            "capture_token_sections": ["prompt_end", "generated"],
            "source_generation_artifact_id": "generation_run_1_d6e12a467208",
        }
    )


def build_probe_report_result(*, anchor: Any, label: str, view: str) -> TransformResult:
    del anchor
    summary = json.loads(PROBE_SUMMARY_PATH.read_text(encoding="utf-8"))
    metrics = _mapping(_mapping(summary.get("metrics")).get(view)).get(label)
    if not isinstance(metrics, Mapping):
        raise ValueError(f"No probe metrics found for label={label!r}, view={view!r}")

    layers: list[dict[str, Any]] = []
    best_layer: int | None = None
    best_value = float("-inf")
    for layer in CAPTURED_LAYERS:
        row = _mapping(metrics.get(str(layer)))
        holdout = _mapping(row.get("source_family_holdout"))
        cv = _mapping(row.get("cv"))
        balanced_accuracy = holdout.get("balanced_accuracy")
        auroc = holdout.get("auroc")
        layer_record = {
            "layer": int(layer),
            "balanced_accuracy": balanced_accuracy,
            "auroc": auroc,
            "cv_balanced_accuracy": cv.get("balanced_accuracy"),
            "cv_auroc": cv.get("auroc"),
            "n": row.get("n"),
        }
        layers.append(layer_record)
        if balanced_accuracy is not None and float(balanced_accuracy) > best_value:
            best_layer = int(layer)
            best_value = float(balanced_accuracy)

    return TransformResult(
        payload={
            "kind": "probe_result",
            "label": label,
            "view": view,
            "layers": layers,
            "summary": {
                "best_layer": best_layer,
                "best_metric": "source_family_holdout_balanced_accuracy",
                "best_value": None if best_layer is None else best_value,
                "example_count": layers[0].get("n") if layers else None,
                "group_count": None,
                "split_mode": "source_family_holdout",
                "label": label,
                "view": view,
                "capture_artifact_id": summary.get("capture_artifact_id"),
            },
        }
    )


def _probe_report_step_name(label: str, view: str) -> str:
    return f"report_probe_{label}_{view}"


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
                cpu=4,
                memory_mb=16 * 1024,
                timeout_seconds=60 * 60,
                secrets=tuple(secrets),
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
            catalog=catalog,
        ),
    }


def build_workflow() -> WorkflowSpec:
    dataset = build_dataset()
    probe_report_steps = tuple(
        WorkflowStep(
            name=_probe_report_step_name(label, view),
            runner="report_local",
            description=f"Materialize dashboard-native probe-result curves for {label} / {view}.",
            spec=TransformSpec(
                builder=TransformBuilder.from_function(
                    build_probe_report_result,
                    local_python_sources=("projects/MOREBENCH",),
                ),
                inputs={
                    "anchor": StepRef("summarize_capture"),
                    "label": label,
                    "view": view,
                },
            ),
        )
        for label in REPORT_LABELS
        for view in REPORT_VIEWS
    )
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="capture_response_label_residuals",
                runner="capture_gpu",
                description="Replay full public-test responses and capture prompt-end plus generated-token residuals.",
                spec=CaptureSpec(
                    engine=base._engine(max_num_seqs=8),
                    dataset=dataset,
                    sites=[
                        ResidualSite(
                            name="prompt_end_residual",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.section("prompt_end"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                        ResidualSite(
                            name="generated_sequence_residual",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.section("generated"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        )
                    ],
                    generation=base.GenerationSpec(enabled=False),
                ),
            ),
            WorkflowStep(
                name="summarize_capture",
                runner="analysis_cpu",
                description="Summarize the response-label capture artifact.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_capture,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"capture_result": StepRef("capture_response_label_residuals")},
                ),
            ),
            *probe_report_steps,
            WorkflowStep(
                name="report",
                runner="report_local",
                description="Package response-label capture and dashboard-native probe curves for local browsing.",
                spec=ReportSpec(
                    inputs=(
                        StepRef("capture_response_label_residuals"),
                        StepRef("summarize_capture"),
                        *(StepRef(_probe_report_step_name(label, view)) for label in REPORT_LABELS for view in REPORT_VIEWS),
                    ),
                    template="default",
                    output_dir=str(REPORT_OUTPUT_DIR),
                ),
            ),
        ),
    )
