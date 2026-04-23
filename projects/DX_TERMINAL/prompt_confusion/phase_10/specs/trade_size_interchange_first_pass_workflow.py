from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from pipelines_v2.api import (
    ActivationBankSpec,
    CaptureSpec,
    Dataset,
    GenerationRunSpec,
    GenerationSpec,
    InterchangePatch,
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
SCRIPT_ROOT = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

from trade_size_patch_eval import evaluate_trade_size_patch_row

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
MODAL_ARTIFACT_ROOT = "/data/artifacts/prompt_confusion_trade_size_interchange_first_pass"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "prompt_confusion_trade_size_interchange_first_pass"
WRITE_LAYERS = (28, 36)


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
        name="prompt_confusion_trade_size_interchange_first_pass",
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
        "compare_local": LocalRunnerSpec(
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


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    engine = build_engine()
    conflict_rows = dataset.labels("conflict_label_name").equals("conflict")
    aligned_rows = dataset.labels("conflict_label_name").equals("aligned")
    local_sources = (
        str(Path(__file__).parent.parent / "scripts"),
    )
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

    prep_deps: list[str] = []
    for layer in WRITE_LAYERS:
        bank_step = f"trade_size_activation_bank_l{layer}"
        subspace_step = f"trade_size_subspace_l{layer}"
        prep_deps.extend([bank_step, subspace_step])
        steps.extend(
            [
                WorkflowStep(
                    name=bank_step,
                    runner="cpu",
                    depends_on=("capture_prompt_eos_residual",),
                    spec=ActivationBankSpec(
                        feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
                        layers=(layer,),
                        rows=aligned_rows,
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
            depends_on=tuple(prep_deps),
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

    compare_variants: dict[str, StepRef] = {}
    compare_deps: list[str] = []
    for layer in WRITE_LAYERS:
        interchange_step = f"interchange_from_aligned_l{layer}"
        random_step = f"random_control_patch_l{layer}"
        compare_deps.extend([interchange_step, random_step])
        compare_variants[f"interchange_l{layer}"] = StepRef(interchange_step)
        compare_variants[f"random_control_l{layer}"] = StepRef(random_step)
        steps.extend(
            [
                WorkflowStep(
                    name=interchange_step,
                    runner="gpu",
                    depends_on=("baseline_conflict_rows", f"trade_size_activation_bank_l{layer}"),
                    spec=PatchedGenerationSpec(
                        engine=engine,
                        dataset=dataset,
                        patch=InterchangePatch(
                            activation_bank=StepRef(f"trade_size_activation_bank_l{layer}"),
                            write_site=ResidualInterventionSite(site="resid_post", layers=(layer,)),
                            target_tokens=TokenSelector.last(),
                            donor_tokens=TokenSelector.last(),
                        ),
                        pair_by=dataset.cases("matched_group_id"),
                        target_when=conflict_rows,
                        donor_when=aligned_rows,
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
                    depends_on=("baseline_conflict_rows", f"trade_size_subspace_l{layer}"),
                    spec=PatchedGenerationSpec(
                        engine=engine,
                        dataset=dataset,
                        patch=RandomControlPatch(
                            subspace=StepRef(f"trade_size_subspace_l{layer}"),
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
            runner="compare_local",
            depends_on=tuple(compare_deps),
            spec=PatchComparisonSpec(
                baseline=StepRef("baseline_conflict_rows"),
                variants=compare_variants,
                row_evaluator=row_evaluator,
            ),
        )
    )

    return WorkflowSpec(
        name="prompt_confusion_trade_size_interchange_first_pass",
        steps=tuple(steps),
    )


workflow = build_workflow()
runner_specs = build_runner_specs()
