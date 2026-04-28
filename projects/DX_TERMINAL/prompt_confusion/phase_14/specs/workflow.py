from __future__ import annotations

"""Phase 14 mid-prompt synthetic geometry workflow."""

import os
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    DirectionSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PostgresSource,
    ProbeSpec,
    PromptMetadataBuilder,
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
from projects.DX_TERMINAL.prompt_confusion.catalogs import build_prompt_confusion_catalog


MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
DEFAULT_CORPUS_TABLE = os.environ.get(
    "PHASE14_THREE_FAMILY_TABLE",
    "prompt_confusion_three_family_settled_v1",
)
DEFAULT_REPORT_DIR = "projects/DX_TERMINAL/prompt_confusion/phase_14/reports/mid_prompt_geometry"
CAPTURED_LAYERS = (12, 16, 20, 24, 28, 32, 36, 40, 44)
FAMILIES = ("trade_size", "risk_preference", "diversification_preference")
SITES = ("strategies_end", "settings_end", "portfolio_end", "market_end", "prompt_eos")

_ROLE_MARKERS = {
    "system": "<|im_start|>system\n",
    "user": "<|im_start|>user\n",
}
_IM_END = "<|im_end|>"
_SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "strategies": re.compile(r"^\s*STRATEGY\s*$", re.IGNORECASE | re.MULTILINE),
    "settings": re.compile(r"^\s*ACTIVE SETTINGS\s*$", re.IGNORECASE | re.MULTILINE),
    "portfolio": re.compile(r"^\s*PORTFOLIO\s*$", re.IGNORECASE | re.MULTILINE),
    "market": re.compile(r"^\s*MARKET\s*$", re.IGNORECASE | re.MULTILINE),
}


def _dataset_limit_from_env() -> int | None:
    raw = os.environ.get("PHASE14_SYNTHETIC_GEOMETRY_LIMIT")
    if raw is None or not raw.strip():
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError("PHASE14_SYNTHETIC_GEOMETRY_LIMIT must be positive")
    return value


def _validate_table_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe table name: {value}")
    return value


def build_dataset_sql(table_name: str = DEFAULT_CORPUS_TABLE) -> str:
    table_name = _validate_table_name(table_name)
    family_values = ", ".join("'" + family + "'" for family in FAMILIES)
    direction_columns = ",\n        ".join(
        f"""CASE
            WHEN target_dimension = '{family}' AND conflict_present THEN 'positive'
            WHEN target_dimension = '{family}' AND NOT conflict_present THEN 'negative'
            ELSE 'exclude'
        END AS {family}_direction_class"""
        for family in FAMILIES
    )
    return f"""
    SELECT
        src.*,
        CASE
            WHEN strategy_lexical_split = 'train' AND settings_lexical_split = 'train' THEN 'strict_train'
            WHEN strategy_lexical_split = 'test' AND settings_lexical_split = 'test' THEN 'strict_test'
            ELSE 'mixed'
        END AS phase14_strict_combined_split,
        TRUE AS phase14_probe_row,
        {direction_columns}
    FROM {table_name} src
    WHERE prompt_messages_json IS NOT NULL
      AND target_dimension IN ({family_values})
      AND COALESCE(main_benchmark_row, TRUE)
      AND NOT COALESCE(edge_conflict, FALSE)
    ORDER BY source_phase, target_dimension, example_id
    """


def build_dataset(*, limit: int | None = None) -> Dataset:
    dataset = Dataset.from_postgres(
        source=PostgresSource.from_env(DB_ENV_VAR),
        sql=build_dataset_sql(),
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        label_columns=[
            "source_phase",
            "family_name",
            "user_text",
            "target_dimension",
            "strategy_direction",
            "setting_value",
            "setting_implied_direction",
            "conflict_present",
            "edge_conflict",
            "conflict_band",
            "main_benchmark_row",
            "phase14_probe_row",
            "lexical_split",
            "strategy_lexical_split",
            "settings_lexical_split",
            "phase14_strict_combined_split",
            "context_variant_id",
            "trade_size_direction_class",
            "risk_preference_direction_class",
            "diversification_preference_direction_class",
        ],
        case_columns=["matched_group_id", "matched_pair_id", "target_dimension"],
        case_key_column="matched_group_id",
        name="prompt_confusion_phase_14_mid_prompt_synthetic_geometry",
    )
    final_limit = _dataset_limit_from_env() if limit is None else limit
    return dataset.select(limit=final_limit) if final_limit is not None else dataset


def _default_residual_engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=8192,
        enforce_eager=False,
        max_num_seqs=16,
    )


def _trim_end(text: str, *, start: int, end: int) -> int:
    while end > start and text[end - 1].isspace():
        end -= 1
    return max(start, end)


def _last_char_span(text: str, *, start: int, end: int) -> tuple[int, int]:
    end = _trim_end(text, start=start, end=end)
    if end <= start:
        return start, min(start + 1, len(text))
    return end - 1, end


def _role_content_span(rendered_prompt: str, role: str) -> tuple[int, int] | None:
    marker = _ROLE_MARKERS[role]
    marker_start = rendered_prompt.find(marker)
    if marker_start < 0:
        return None
    start = marker_start + len(marker)
    end = rendered_prompt.find(_IM_END, start)
    if end < 0:
        return None
    return start, _trim_end(rendered_prompt, start=start, end=end)


def _prompt_eos_span(rendered_prompt: str) -> tuple[int, int] | None:
    last = rendered_prompt.rfind(_IM_END)
    if last < 0:
        return None
    return last, last + len(_IM_END)


def _find_user_sections(rendered_prompt: str, user_span: tuple[int, int]) -> dict[str, tuple[int, int]]:
    user_start, user_end = user_span
    user_text = rendered_prompt[user_start:user_end]
    headings: list[tuple[str, int, int]] = []
    for name, pattern in _SECTION_PATTERNS.items():
        match = pattern.search(user_text)
        if match:
            headings.append((name, user_start + match.start(), user_start + match.end()))
    headings.sort(key=lambda item: item[1])

    sections: dict[str, tuple[int, int]] = {}
    for index, (name, heading_start, heading_end) in enumerate(headings):
        next_start = headings[index + 1][1] if index + 1 < len(headings) else user_end
        sections[name] = (heading_start, _trim_end(rendered_prompt, start=heading_end, end=next_start))
    return sections


def build_mid_prompt_metadata(rendered_prompt: str) -> dict[str, Any]:
    token_sections: dict[str, dict[str, int]] = {}
    missing_sections: list[str] = []
    user_span = _role_content_span(rendered_prompt, "user")
    if user_span is not None:
        sections = _find_user_sections(rendered_prompt, user_span)
        for name in ("strategies", "settings", "portfolio", "market"):
            span = sections.get(name)
            if span is None:
                missing_sections.append(name)
                span = user_span
            token_sections[name] = {"char_start": span[0], "char_end": span[1]}
            start, end = _last_char_span(rendered_prompt, start=span[0], end=span[1])
            token_sections[f"{name}_end"] = {"char_start": start, "char_end": end}
    else:
        missing_sections.extend(["user", "strategy", "settings", "portfolio", "market"])

    prompt_eos = _prompt_eos_span(rendered_prompt)
    if prompt_eos is not None:
        token_sections["prompt_eos"] = {"char_start": prompt_eos[0], "char_end": prompt_eos[1]}
    else:
        missing_sections.append("prompt_eos")

    return {
        "token_sections": token_sections,
        "phase14_section_parse": {"missing_sections": sorted(set(missing_sections))},
    }


def build_runner_specs() -> dict[str, object]:
    neon_secret = ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon")
    artifact_store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/prompt_confusion_phase_14_mid_prompt_geometry",
    )
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H100",
                secrets=(neon_secret,),
                volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
            ),
            artifacts=artifact_store,
            catalog=build_prompt_confusion_catalog(__file__),
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(cpu=6, memory_mb=24 * 1024, secrets=(neon_secret,)),
            artifacts=artifact_store,
            catalog=build_prompt_confusion_catalog(__file__),
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(Path("artifacts") / "prompt_confusion_phase_14_mid_prompt_geometry"),
            catalog=build_prompt_confusion_catalog(__file__),
        ),
    }


def _residual_site(site: str) -> ResidualSite:
    return ResidualSite(
        name=f"residual_{site}",
        site="resid_post",
        layers=list(CAPTURED_LAYERS),
        tokens=TokenSelector.section(site),
        storage=TensorStorage(dtype="float16", format="safetensors"),
    )


def _text_baseline_step(*, family: str, dataset: Dataset) -> WorkflowStep:
    return WorkflowStep(
        name=f"text_gate__{family}",
        runner="analysis_cpu",
        spec=TextBaselineSpec(
            text=dataset.labels("user_text"),
            rows=dataset.labels("target_dimension").equals(family),
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_group_id"),
            split_by={"phase14_strict_combined_split": dataset.labels("phase14_strict_combined_split")},
            train_values=("strict_train",),
            test_values=("strict_test",),
            model="countvectorizer_logreg",
            metrics=("accuracy", "balanced_accuracy", "auroc"),
        ),
    )


def _probe_step(*, site: str, family: str, dataset: Dataset) -> WorkflowStep:
    return WorkflowStep(
        name=f"probe__{site}__{family}",
        runner="analysis_cpu",
        spec=ProbeSpec(
            feature=StepRef("capture_mid_prompt_residuals").feature(f"residual_{site}"),
            rows=dataset.labels("target_dimension").equals(family),
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_group_id"),
            split=dataset.labels("phase14_strict_combined_split"),
            train_values=("strict_train",),
            test_values=("strict_test",),
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.mean(),
            metrics=("accuracy", "balanced_accuracy", "auroc", "selectivity"),
            baselines=("majority", "shuffled_label"),
        ),
    )


def _direction_step(*, site: str, family: str, dataset: Dataset) -> WorkflowStep:
    return WorkflowStep(
        name=f"direction__{site}__{family}",
        runner="analysis_cpu",
        spec=DirectionSpec(
            feature=StepRef("capture_mid_prompt_residuals").feature(f"residual_{site}"),
            positive=dataset.labels(f"{family}_direction_class").equals("positive"),
            negative=dataset.labels(f"{family}_direction_class").equals("negative"),
            layers=list(CAPTURED_LAYERS),
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.mean(),
        ),
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def _artifact_result(artifact: Any) -> Mapping[str, Any]:
    payload = artifact.result() if hasattr(artifact, "result") else artifact
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected direction artifact result mapping, got {type(payload).__name__}")
    return payload


def _direction_vector(payload: Mapping[str, Any], *, layer: int) -> np.ndarray:
    raw = payload["layers"][str(int(layer))]["vector"]
    return _unit(np.asarray(raw, dtype=np.float32))


def compare_mid_prompt_direction_geometry(
    direction_index: Mapping[str, Any],
    **direction_artifacts: Any,
) -> TransformResult:
    directions: dict[tuple[str, str, int], np.ndarray] = {}
    for step_name, descriptor in direction_index.items():
        payload = _artifact_result(direction_artifacts[str(step_name)])
        site = str(descriptor["site"])
        family = str(descriptor["family"])
        for raw_layer in payload.get("layers", {}):
            layer = int(raw_layer)
            directions[(site, family, layer)] = _direction_vector(payload, layer=layer)

    rows: list[dict[str, Any]] = []
    for layer in CAPTURED_LAYERS:
        for family in FAMILIES:
            eos = directions.get(("prompt_eos", family, int(layer)))
            for site in SITES:
                if site == "prompt_eos":
                    continue
                rows.append(
                    {
                        "comparison": "same_family_vs_prompt_eos",
                        "layer": int(layer),
                        "site": site,
                        "family": family,
                        "other_family": family,
                        "cosine": _cosine(directions.get((site, family, int(layer))), eos),
                    }
                )

        for site in SITES:
            for left_index, left in enumerate(FAMILIES):
                for right in FAMILIES[left_index + 1 :]:
                    rows.append(
                        {
                            "comparison": "family_pair_same_site",
                            "layer": int(layer),
                            "site": site,
                            "family": left,
                            "other_family": right,
                            "cosine": _cosine(
                                directions.get((site, left, int(layer))),
                                directions.get((site, right, int(layer))),
                            ),
                        }
                    )

            site_vectors = [directions.get((site, family, int(layer))) for family in FAMILIES]
            eos_vectors = [directions.get(("prompt_eos", family, int(layer))) for family in FAMILIES]
            if all(vector is not None for vector in site_vectors) and all(vector is not None for vector in eos_vectors):
                site_shared = _unit(np.stack([vector for vector in site_vectors if vector is not None], axis=0).mean(axis=0))
                eos_shared = _unit(np.stack([vector for vector in eos_vectors if vector is not None], axis=0).mean(axis=0))
                rows.append(
                    {
                        "comparison": "shared_mean_vs_prompt_eos",
                        "layer": int(layer),
                        "site": site,
                        "family": "shared_mean",
                        "other_family": "shared_mean",
                        "cosine": _cosine(site_shared, eos_shared),
                    }
                )

    def best(comparison: str, *, site: str | None = None) -> dict[str, Any] | None:
        candidates = [
            row
            for row in rows
            if row["comparison"] == comparison
            and row["cosine"] is not None
            and (site is None or row["site"] == site)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda row: float(row["cosine"]))

    return TransformResult(
        payload={
            "kind": "phase14_mid_prompt_direction_geometry",
            "layers": list(CAPTURED_LAYERS),
            "sites": list(SITES),
            "families": list(FAMILIES),
            "rows": rows,
            "summary": {
                "best_same_family_vs_prompt_eos": best("same_family_vs_prompt_eos"),
                "best_settings_same_family_vs_prompt_eos": best("same_family_vs_prompt_eos", site="settings_end"),
                "best_settings_family_pair": best("family_pair_same_site", site="settings_end"),
                "settings_shared_mean_vs_prompt_eos": [
                    row
                    for row in rows
                    if row["comparison"] == "shared_mean_vs_prompt_eos"
                    and row["site"] == "settings_end"
                ],
            },
        },
    )


def build_workflow(
    dataset: Dataset | None = None,
    *,
    residual_engine: VLLMEngine | None = None,
    report_output_dir: str = DEFAULT_REPORT_DIR,
) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    residual_engine = residual_engine or _default_residual_engine()
    prompt_metadata = PromptMetadataBuilder.from_function(
        build_mid_prompt_metadata,
        local_python_sources=("projects",),
    )
    geometry_builder = TransformBuilder.from_function(
        compare_mid_prompt_direction_geometry,
        local_python_sources=("projects",),
    )

    steps: list[WorkflowStep] = [
        *[_text_baseline_step(family=family, dataset=dataset) for family in FAMILIES],
        WorkflowStep(
            name="capture_mid_prompt_residuals",
            runner="capture_gpu",
            spec=CaptureSpec(
                engine=residual_engine,
                dataset=dataset,
                sites=[_residual_site(site) for site in SITES],
                prompt_metadata_builder=prompt_metadata,
            ),
        ),
    ]

    direction_index: dict[str, dict[str, str]] = {}
    for site in SITES:
        for family in FAMILIES:
            steps.append(_probe_step(site=site, family=family, dataset=dataset))
            direction_step = _direction_step(site=site, family=family, dataset=dataset)
            steps.append(direction_step)
            direction_index[direction_step.name] = {"site": site, "family": family}

    geometry_inputs: dict[str, Any] = {"direction_index": direction_index}
    geometry_inputs.update({name: StepRef(name) for name in direction_index})
    steps.append(
        WorkflowStep(
            name="mid_prompt_direction_geometry",
            runner="analysis_cpu",
            spec=TransformSpec(
                builder=geometry_builder,
                inputs=geometry_inputs,
            ),
        )
    )

    report_inputs = [StepRef(step.name) for step in steps if step.name != "capture_mid_prompt_residuals"]
    steps.append(
        WorkflowStep(
            name="report",
            runner="report_local",
            spec=ReportSpec(
                inputs=tuple(report_inputs),
                template="default",
                output_dir=report_output_dir,
            ),
        )
    )

    return WorkflowSpec(name="prompt_confusion_phase14_mid_prompt_geometry", steps=tuple(steps))
