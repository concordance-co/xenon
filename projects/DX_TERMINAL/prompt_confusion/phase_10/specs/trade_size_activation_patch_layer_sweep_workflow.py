from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pipelines_v2.api import (
    CentroidSpec,
    CaptureSpec,
    Dataset,
    GenerationRunSpec,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    PatchComparisonSpec,
    PatchedGenerationSpec,
    RandomControlPatch,
    ResidualInterventionSite,
    ResidualSite,
    StepRef,
    SubspaceSpec,
    SwapMeanPatch,
    TensorStorage,
    TokenPooling,
    TokenSelector,
    TransformBuilder,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)
from projects.DX_TERMINAL.prompt_confusion.catalogs import build_prompt_confusion_catalog


WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
ARTIFACTS_VOLUME = "xenon-data"
PHASE_09_DATASET_PATH = Path(
    os.environ.get(
        "PHASE_09_DATASET_PATH",
        str(
            WORKSPACE_ROOT
            / "projects/DX_TERMINAL/prompt_confusion/phase_09/outputs/phase_09_dataset/phase_09_dataset.jsonl"
        ),
    )
)
MODAL_ARTIFACT_ROOT = "/data/artifacts/prompt_confusion_trade_size_activation_patch_layer_sweep"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "prompt_confusion_trade_size_activation_patch_layer_sweep"
WRITE_LAYERS = (28, 32, 36, 40)


def _load_trade_size_rows() -> list[dict[str, Any]]:
    if not PHASE_09_DATASET_PATH.exists():
        raise FileNotFoundError(f"Phase 09 dataset not found: {PHASE_09_DATASET_PATH}")
    rows: list[dict[str, Any]] = []
    with PHASE_09_DATASET_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if row.get("target_dimension") != "trade_size":
                continue
            if not bool(row.get("main_benchmark_row", True)):
                continue
            row["conflict_label_name"] = "conflict" if bool(row.get("conflict_present")) else "aligned"
            row["scope_conflict_rows"] = bool(row.get("conflict_present"))
            rows.append(row)
    if not rows:
        raise ValueError("No main-benchmark trade_size rows found in Phase 09 dataset")
    return rows


def build_dataset() -> Dataset:
    return Dataset.from_records(
        _load_trade_size_rows(),
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        label_columns=[
            "user_text",
            "target_dimension",
            "strategy_direction",
            "setting_value",
            "setting_implied_direction",
            "conflict_present",
            "conflict_label_name",
            "scope_conflict_rows",
            "lexical_split",
            "strategy_lexical_split",
            "settings_lexical_split",
            "matched_group_id",
            "matched_pair_id",
        ],
        case_columns=["matched_group_id", "matched_pair_id"],
        case_key_column="matched_group_id",
        name="prompt_confusion_trade_size_activation_patch_layer_sweep",
    )


def build_runner_specs() -> dict[str, object]:
    artifact_store = ModalVolumeStore(name=ARTIFACTS_VOLUME, root=MODAL_ARTIFACT_ROOT)
    return {
        "gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="A100-80GB",
                timeout_seconds=60 * 60 * 4,
                volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
            ),
            artifacts=artifact_store,
            catalog=build_prompt_confusion_catalog(__file__),
        ),
        "cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=8,
                memory_mb=24 * 1024,
                timeout_seconds=60 * 60,
            ),
            artifacts=artifact_store,
            catalog=build_prompt_confusion_catalog(__file__),
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
            catalog=build_prompt_confusion_catalog(__file__),
        ),
    }


def build_engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=8192,
        enforce_eager=False,
        max_num_seqs=16,
        max_num_batched_tokens=32768,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        async_scheduling=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        payload = json.loads(raw[start : end + 1])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _example_value(example: dict[str, Any], key: str) -> Any:
    if key in example:
        return example.get(key)
    labels = example.get("labels")
    if isinstance(labels, dict):
        return labels.get(key)
    return None


def evaluate_trade_size_patch_row(
    *,
    example: dict[str, Any],
    baseline: dict[str, Any],
    variants: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    strategy_size = str(_example_value(example, "strategy_direction") or "")
    setting_size = str(_example_value(example, "setting_implied_direction") or "")
    baseline_payload = _parse_json_payload(str(baseline.get("generated_text") or ""))
    baseline_size = str((baseline_payload or {}).get("size") or "")
    results: dict[str, Any] = {}
    summary: dict[str, Any] = {
        "example_key": str(example.get("key") or example.get("example_key") or ""),
        "strategy_size": strategy_size,
        "setting_size": setting_size,
        "baseline_text": str(baseline.get("generated_text") or ""),
        "baseline_valid": baseline_payload is not None,
        "baseline_size": baseline_size,
        "baseline_follows_setting": baseline_size == setting_size,
        "baseline_follows_strategy": baseline_size == strategy_size,
    }
    for variant_name, row in dict(variants or {}).items():
        payload = _parse_json_payload(str(row.get("generated_text") or ""))
        patched_size = str((payload or {}).get("size") or "")
        results[variant_name] = {
            "valid_json": payload is not None,
            "size_changed": patched_size != baseline_size,
            "patched_follows_setting": patched_size == setting_size,
            "patched_follows_strategy": patched_size == strategy_size,
            "intended_erasure_flip": baseline_size == setting_size and patched_size == strategy_size,
            "reverse_flip": baseline_size == strategy_size and patched_size == setting_size,
            "malformed": payload is None,
        }
        summary[f"{variant_name}_text"] = str(row.get("generated_text") or "")
        summary[f"{variant_name}_size"] = patched_size
    return {"metrics": results, "evaluation": summary}


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    engine = build_engine()
    conflict_rows = dataset.labels("scope_conflict_rows").equals(True)
    local_sources = (str(Path(__file__).parent),)
    row_evaluator = TransformBuilder.from_function(
        evaluate_trade_size_patch_row,
        local_python_sources=local_sources,
    )

    steps: list[WorkflowStep] = [
        WorkflowStep(
            name="capture_prompt_eos_residual",
            runner="gpu",
            spec=CaptureSpec(
                engine=engine,
                dataset=dataset,
                sites=(
                    ResidualSite(
                        name="residual_prompt_eos",
                        site="resid_post",
                        layers=WRITE_LAYERS,
                        tokens=TokenSelector.last(),
                        storage=TensorStorage(dtype="float16", format="safetensors"),
                    ),
                ),
                generation=GenerationSpec(enabled=False),
            ),
        ),
    ]

    centroid_and_subspace_steps: list[str] = []
    for layer in WRITE_LAYERS:
        aligned_centroid_step = f"trade_size_centroids_aligned_patch_l{layer}"
        same_label_centroid_step = f"trade_size_centroids_same_label_control_l{layer}"
        subspace_step = f"trade_size_subspace_l{layer}"
        centroid_and_subspace_steps.extend(
            [aligned_centroid_step, same_label_centroid_step, subspace_step]
        )
        steps.extend(
            [
                WorkflowStep(
                    name=aligned_centroid_step,
                    runner="cpu",
                    depends_on=("capture_prompt_eos_residual",),
                    spec=CentroidSpec(
                        feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
                        by=dataset.labels("conflict_label_name"),
                        rows=None,
                        layers=(layer,),
                        tokens=TokenSelector.full_sequence(),
                        pooling=TokenPooling.last(),
                    ),
                ),
                WorkflowStep(
                    name=same_label_centroid_step,
                    runner="cpu",
                    depends_on=("capture_prompt_eos_residual",),
                    spec=CentroidSpec(
                        feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
                        by=dataset.labels("conflict_label_name"),
                        rows=None,
                        layers=(layer,),
                        tokens=TokenSelector.full_sequence(),
                        pooling=TokenPooling.last(),
                    ),
                ),
                WorkflowStep(
                    name=subspace_step,
                    runner="cpu",
                    depends_on=("capture_prompt_eos_residual",),
                    spec=SubspaceSpec(
                        feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
                        layers=(layer,),
                        components=8,
                        tokens=TokenSelector.full_sequence(),
                        pooling=TokenPooling.last(),
                    ),
                ),
            ]
        )

    steps.append(
        WorkflowStep(
            name="baseline_conflict_rows",
            runner="gpu",
            depends_on=tuple(centroid_and_subspace_steps),
            spec=GenerationRunSpec(
                engine=engine,
                dataset=dataset,
                select_when=conflict_rows,
                generation=GenerationSpec(
                    enabled=True,
                    max_tokens=128,
                    temperature=0.0,
                    capture_reasoning=False,
                ),
            ),
        )
    )

    compare_variant_refs: dict[str, StepRef] = {}
    compare_dependencies: list[str] = []
    for layer in WRITE_LAYERS:
        aligned_centroid_step = f"trade_size_centroids_aligned_patch_l{layer}"
        same_label_centroid_step = f"trade_size_centroids_same_label_control_l{layer}"
        subspace_step = f"trade_size_subspace_l{layer}"
        swap_step = f"swap_to_aligned_centroid_l{layer}"
        same_label_step = f"swap_to_conflict_centroid_control_l{layer}"
        random_step = f"random_control_patch_l{layer}"
        compare_dependencies.extend([swap_step, same_label_step, random_step])
        compare_variant_refs[f"swap_to_aligned_l{layer}"] = StepRef(swap_step)
        compare_variant_refs[f"same_label_control_l{layer}"] = StepRef(same_label_step)
        compare_variant_refs[f"random_control_l{layer}"] = StepRef(random_step)
        steps.extend(
            [
                WorkflowStep(
                    name=swap_step,
                    runner="gpu",
                    depends_on=("baseline_conflict_rows", aligned_centroid_step),
                    spec=PatchedGenerationSpec(
                        engine=engine,
                        dataset=dataset,
                        patch=SwapMeanPatch(
                            centroids=StepRef(aligned_centroid_step),
                            centroid_name="aligned",
                            write_site=ResidualInterventionSite(site="resid_post", layers=(layer,)),
                            target_tokens=TokenSelector.last(),
                            strength=1.0,
                        ),
                        select_when=conflict_rows,
                        generation=GenerationSpec(
                            enabled=True,
                            max_tokens=128,
                            temperature=0.0,
                            capture_reasoning=False,
                        ),
                    ),
                ),
                WorkflowStep(
                    name=same_label_step,
                    runner="gpu",
                    depends_on=("baseline_conflict_rows", same_label_centroid_step),
                    spec=PatchedGenerationSpec(
                        engine=engine,
                        dataset=dataset,
                        patch=SwapMeanPatch(
                            centroids=StepRef(same_label_centroid_step),
                            centroid_name="conflict",
                            write_site=ResidualInterventionSite(site="resid_post", layers=(layer,)),
                            target_tokens=TokenSelector.last(),
                            strength=1.0,
                        ),
                        select_when=conflict_rows,
                        generation=GenerationSpec(
                            enabled=True,
                            max_tokens=128,
                            temperature=0.0,
                            capture_reasoning=False,
                        ),
                    ),
                ),
                WorkflowStep(
                    name=random_step,
                    runner="gpu",
                    depends_on=("baseline_conflict_rows", subspace_step),
                    spec=PatchedGenerationSpec(
                        engine=engine,
                        dataset=dataset,
                        patch=RandomControlPatch(
                            subspace=StepRef(subspace_step),
                            write_site=ResidualInterventionSite(site="resid_post", layers=(layer,)),
                            target_tokens=TokenSelector.last(),
                            strength=1.0,
                            random_seed=11,
                            match_projected_norm=True,
                        ),
                        select_when=conflict_rows,
                        generation=GenerationSpec(
                            enabled=True,
                            max_tokens=128,
                            temperature=0.0,
                            capture_reasoning=False,
                        ),
                    ),
                ),
            ]
        )

    steps.append(
        WorkflowStep(
            name="compare_patch_runs",
            runner="cpu",
            depends_on=tuple(compare_dependencies),
            spec=PatchComparisonSpec(
                baseline=StepRef("baseline_conflict_rows"),
                variants=compare_variant_refs,
                row_evaluator=row_evaluator,
            ),
        )
    )

    return WorkflowSpec(
        name="prompt_confusion_trade_size_activation_patch_layer_sweep",
        steps=tuple(steps),
    )


workflow = build_workflow()
runner_specs = build_runner_specs()
