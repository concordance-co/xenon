"""Marshall's confound battery for Phase 09.

Reuses the Phase 09 dataset (`conflict_probe_examples_v4`-era relational
rebuild) and runs a set of confound + sanity tests that either were not
in Trent's first pipelines_v2 run, or that answer a question he left
open.

What this adds on top of `phase_09/specs/workflow.py`:

1. Fresh capture with GenerationSpec enabled so the behavioral audit
   script has outputs to parse (Trent's capture had generation
   disabled). Residual config is otherwise identical.
2. Strict both-axes lexical holdout on `conflict_present`
   (`strategy_lexical_split == 'train'` AND
   `settings_lexical_split == 'train'` vs the same with `'test'` on
   both). Trent's main probe used his `lexical_split` column, which
   is the XOR of the two per-axis splits and therefore sees all
   individual variants during training.
3. `setting_value`-alone probe -- checks whether the residual-stream
   signal that decodes `conflict_present` is partially leaking
   through a direct `setting_value` encoding.
4. `strategy_direction` cross-transfer probe on `trade_size` --
   train on rows where strategy says `small`, test on rows where it
   says `large`, and vice versa. Distinguishes real relational
   conflict from a "small-word plus large-word both present" lexical
   shortcut.
5. Text baseline on the strict both-axes holdout for completeness.

The behavioral audit lives separately in `behavioral_audit.py` and
consumes the capture's `generations.json` once the workflow has run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    ProbeSpec,
    ReportSpec,
    ResidualSite,
    StepRef,
    TensorStorage,
    TextBaselineSpec,
    TokenPooling,
    TokenSelector,
    TransferProbeSpec,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)
from projects.DX_TERMINAL.prompt_confusion.catalogs import build_prompt_confusion_catalog


PHASE_ROOT = Path("projects/DX_TERMINAL/prompt_confusion/phase_09")
DEFAULT_DATASET_PATH = Path(
    os.environ.get(
        "PHASE_09_DATASET_PATH",
        str(PHASE_ROOT / "outputs" / "phase_09_dataset" / "phase_09_dataset.jsonl"),
    )
)
BATTERY_REPORT_DIR = str(PHASE_ROOT / "reports" / "marshalls_battery")
MODAL_ARTIFACT_ROOT = "/data/artifacts/prompt_confusion_phase_09_marshalls_battery"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "prompt_confusion_phase_09_marshalls_battery"

MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
ARTIFACTS_VOLUME = "xenon-data"

CAPTURED_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)


def _load_dataset_records() -> list[dict[str, object]]:
    if not DEFAULT_DATASET_PATH.exists():
        raise FileNotFoundError(f"Phase 09 dataset not found: {DEFAULT_DATASET_PATH}")
    rows: list[dict[str, object]] = []
    with DEFAULT_DATASET_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            # Derived strict both-axes split: strict_train iff BOTH axes are
            # train, strict_test iff BOTH are test, else mixed (excluded at
            # probe time via train/test value selection).
            s = row.get("strategy_lexical_split")
            t = row.get("settings_lexical_split")
            if s == "train" and t == "train":
                row["strict_combined_split"] = "strict_train"
            elif s == "test" and t == "test":
                row["strict_combined_split"] = "strict_test"
            else:
                row["strict_combined_split"] = "mixed"
            # Derived per-dimension main-benchmark flags. main_benchmark_row
            # is False for 96 behaviorally-muddy boundary rows (all
            # trading_activity). Probes should filter those out of scope.
            main = bool(row.get("main_benchmark_row"))
            dim = row.get("target_dimension")
            row["scope_main_only"] = main
            row["scope_main_trade_size"] = main and dim == "trade_size"
            row["scope_main_trading_activity"] = main and dim == "trading_activity"
            rows.append(row)
    if not rows:
        raise ValueError(f"Phase 09 dataset is empty: {DEFAULT_DATASET_PATH}")
    return rows


def build_dataset() -> Dataset:
    return Dataset.from_records(
        _load_dataset_records(),
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        label_columns=[
            "user_text",
            "target_dimension",
            "strategy_direction",
            "setting_value",
            "setting_implied_direction",
            "conflict_present",
            "edge_conflict",
            "conflict_band",
            "main_benchmark_row",
            "lexical_split",
            "strategy_lexical_split",
            "settings_lexical_split",
            "strict_combined_split",
            "scope_main_only",
            "scope_main_trade_size",
            "scope_main_trading_activity",
        ],
        case_columns=["matched_group_id", "matched_pair_id"],
        case_key_column="matched_group_id",
        name="prompt_confusion_phase_09_marshalls_battery",
    )


def _engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=8192,
        enforce_eager=False,
        enable_prefix_caching=True,
        max_num_seqs=16,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_runner_specs() -> dict[str, object]:
    artifact_store = ModalVolumeStore(name=ARTIFACTS_VOLUME, root=MODAL_ARTIFACT_ROOT)
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="A100-80GB",
                timeout_seconds=60 * 60 * 2,
                volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
            ),
            artifacts=artifact_store,
            catalog=build_prompt_confusion_catalog(__file__),
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(cpu=6, memory_mb=24 * 1024),
            artifacts=artifact_store,
            catalog=build_prompt_confusion_catalog(__file__),
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
            catalog=build_prompt_confusion_catalog(__file__),
        ),
    }


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    # Scope filters. main_benchmark_row=False marks 96 behaviorally-muddy
    # boundary rows (all trading_activity) that should be out of scope for
    # the probe. Trade_size has no muddy rows; its main-scope is the same
    # as its full 384-row set. Trading_activity main-scope is 384 rows
    # (from a full 480), dropping the boundary cells.
    main_only = dataset.labels("scope_main_only").equals(True)
    main_trade_size = dataset.labels("scope_main_trade_size").equals(True)
    main_trading_activity = dataset.labels("scope_main_trading_activity").equals(True)

    # ---- Text baselines (cheap controls) -----------------------------------

    text_baseline_strict_combined = WorkflowStep(
        name="text_baseline_conflict_strict_combined",
        runner="analysis_cpu",
        spec=TextBaselineSpec(
            text=dataset.labels("user_text"),
            rows=main_only,
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_group_id"),
            split_by={"combined": dataset.labels("strict_combined_split")},
            train_values=("strict_train",),
            test_values=("strict_test",),
            model="countvectorizer_logreg",
            metrics=("balanced_accuracy", "auroc"),
        ),
    )

    # ---- Capture with generations ------------------------------------------

    capture = WorkflowStep(
        name="capture_residual_with_generation",
        runner="capture_gpu",
        spec=CaptureSpec(
            engine=_engine(),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="residual_prompt_eos",
                    site="resid_post",
                    layers=list(CAPTURED_LAYERS),
                    tokens=TokenSelector.last(),
                    storage=TensorStorage(dtype="float16", format="safetensors"),
                )
            ],
            generation=GenerationSpec(
                enabled=True,
                max_tokens=256,
                temperature=0.0,
                capture_reasoning=False,
            ),
        ),
    )

    # ---- Probes ------------------------------------------------------------

    probe_conflict_strict_combined = WorkflowStep(
        name="probe_conflict_strict_combined_holdout",
        runner="analysis_cpu",
        spec=ProbeSpec(
            feature=StepRef("capture_residual_with_generation").feature("residual_prompt_eos"),
            rows=main_only,
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_group_id"),
            split=dataset.labels("strict_combined_split"),
            train_values=("strict_train",),
            test_values=("strict_test",),
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.last(),
            metrics=("balanced_accuracy", "auroc", "selectivity"),
            baselines=("majority", "shuffled_label"),
        ),
    )

    probe_conflict_strict_combined_trade_size = WorkflowStep(
        name="probe_conflict_strict_combined_trade_size",
        runner="analysis_cpu",
        spec=ProbeSpec(
            feature=StepRef("capture_residual_with_generation").feature("residual_prompt_eos"),
            rows=main_trade_size,
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_group_id"),
            split=dataset.labels("strict_combined_split"),
            train_values=("strict_train",),
            test_values=("strict_test",),
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.last(),
            metrics=("balanced_accuracy", "auroc", "selectivity"),
            baselines=("majority", "shuffled_label"),
        ),
    )

    probe_conflict_strict_combined_trading_activity = WorkflowStep(
        name="probe_conflict_strict_combined_trading_activity",
        runner="analysis_cpu",
        spec=ProbeSpec(
            feature=StepRef("capture_residual_with_generation").feature("residual_prompt_eos"),
            rows=main_trading_activity,
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_group_id"),
            split=dataset.labels("strict_combined_split"),
            train_values=("strict_train",),
            test_values=("strict_test",),
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.last(),
            metrics=("balanced_accuracy", "auroc", "selectivity"),
            baselines=("majority", "shuffled_label"),
        ),
    )

    text_baseline_strict_combined_trade_size = WorkflowStep(
        name="text_baseline_conflict_strict_combined_trade_size",
        runner="analysis_cpu",
        spec=TextBaselineSpec(
            text=dataset.labels("user_text"),
            rows=main_trade_size,
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_group_id"),
            split_by={"combined": dataset.labels("strict_combined_split")},
            train_values=("strict_train",),
            test_values=("strict_test",),
            model="countvectorizer_logreg",
            metrics=("balanced_accuracy", "auroc"),
        ),
    )

    text_baseline_strict_combined_trading_activity = WorkflowStep(
        name="text_baseline_conflict_strict_combined_trading_activity",
        runner="analysis_cpu",
        spec=TextBaselineSpec(
            text=dataset.labels("user_text"),
            rows=main_trading_activity,
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_group_id"),
            split_by={"combined": dataset.labels("strict_combined_split")},
            train_values=("strict_train",),
            test_values=("strict_test",),
            model="countvectorizer_logreg",
            metrics=("balanced_accuracy", "auroc"),
        ),
    )

    probe_setting_value_grouped_cv = WorkflowStep(
        name="probe_setting_value_grouped_cv",
        runner="analysis_cpu",
        spec=ProbeSpec(
            feature=StepRef("capture_residual_with_generation").feature("residual_prompt_eos"),
            rows=main_only,
            labels=dataset.labels("setting_value"),
            group_by=dataset.cases("matched_group_id"),
            folds=5,
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.last(),
            metrics=("balanced_accuracy", "auroc", "selectivity"),
            baselines=("majority", "shuffled_label"),
        ),
    )

    # Direction-transfer probe: on trade_size, train on one
    # strategy_direction cohort and test on the other. If the probe
    # decodes real relational conflict, it transfers. If it's shortcutting
    # on "small-word plus large-word both present," it collapses.
    probe_direction_transfer_trade_size = WorkflowStep(
        name="probe_direction_transfer_trade_size",
        runner="analysis_cpu",
        spec=TransferProbeSpec(
            feature=StepRef("capture_residual_with_generation").feature("residual_prompt_eos"),
            rows=main_trade_size,
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_group_id"),
            cohort_by=dataset.labels("strategy_direction"),
            cohort_values=("small", "large"),
            metrics=("balanced_accuracy", "auroc"),
            compare_within_baseline=True,
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.last(),
        ),
    )

    # Grouped-CV conflict probe with explicit selectivity (belt-and-suspenders
    # on Trent's main probe; makes sure shuffled-label baseline is computed
    # with matched-pair grouping).
    probe_conflict_grouped_cv = WorkflowStep(
        name="probe_conflict_grouped_cv_selectivity",
        runner="analysis_cpu",
        spec=ProbeSpec(
            feature=StepRef("capture_residual_with_generation").feature("residual_prompt_eos"),
            rows=main_only,
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_group_id"),
            folds=5,
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.last(),
            metrics=("balanced_accuracy", "auroc", "selectivity"),
            baselines=("majority", "shuffled_label"),
        ),
    )

    # ---- Report ------------------------------------------------------------

    report = WorkflowStep(
        name="report",
        runner="report_local",
        spec=ReportSpec(
            template="marshalls_battery",
            output_dir=BATTERY_REPORT_DIR,
            inputs=(
                StepRef("text_baseline_conflict_strict_combined"),
                StepRef("probe_conflict_strict_combined_holdout"),
                StepRef("probe_setting_value_grouped_cv"),
                StepRef("probe_direction_transfer_trade_size"),
                StepRef("probe_conflict_grouped_cv_selectivity"),
            ),
        ),
    )

    return WorkflowSpec(
        name="phase_09_marshalls_battery",
        steps=(
            text_baseline_strict_combined,
            text_baseline_strict_combined_trade_size,
            text_baseline_strict_combined_trading_activity,
            capture,
            probe_conflict_strict_combined,
            probe_conflict_strict_combined_trade_size,
            probe_conflict_strict_combined_trading_activity,
            probe_setting_value_grouped_cv,
            probe_direction_transfer_trade_size,
            probe_conflict_grouped_cv,
            report,
        ),
    )
