from __future__ import annotations

import argparse
import builtins
import importlib
import importlib.util
import sys
import types
from dataclasses import replace
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from pipelines_v2.core.types import to_primitive, utc_now_iso
from pipelines_v2.api import (
    ActivationPatchSpec,
    ActivationBankSpec,
    AddDirectionPatch,
    ArtifactDatasetSource,
    ArtifactManifest,
    ArtifactLabelRef,
    CentroidSpec,
    ExplicitPathEdge,
    ExplicitPathMaskSpec,
    InterchangePatch,
    ProjectOutPatch,
    RandomControlPatch,
    ResidualPathPatch,
    CapabilityError,
    CaptureArtifact,
    CaptureSpec,
    Dataset,
    DirectionSpec,
    EngineCapability,
    EngineCaptureResult,
    GenerationRunSpec,
    Example,
    FileCatalog,
    GeometrySpec,
    GenerationSpec,
    HuggingFaceSource,
    InMemorySource,
    LabelMapSpec,
    LocalResources,
    LocalArtifactStore,
    LocalRunner,
    MoERoutingSite,
    ModalResources,
    ModalRunner,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    CompositeCatalog,
    OperationArtifact,
    PatchComparisonSpec,
    PairDeltaSpec,
    PatchedGenerationSpec,
    PromptMetadataBuilder,
    ResidualSite,
    ResidualInterventionSite,
    ResidualizedProbeSpec,
    RoutingRecord,
    StepLabelRef,
    TextBaselineSpec,
    TransferProbeSpec,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    SwapComponentsPatch,
    SwapMeanPatch,
    TokenPooling,
    TokenSelector,
    ToyEngine,
    TransferPolicy,
    TransferPolicyError,
    VLLMEngine,
    PostgresSource,
    ProbeSpec,
    PostgresCatalog,
    SpecValidationError,
    StepRef,
    WorkflowOrchestrator,
    WorkflowResult,
    WorkflowSpec,
    WorkflowStep,
    ReportSpec,
    SubspaceSpec,
)
from pipelines_v2.cli import (
    _build_report_spec_from_run,
    _build_runners,
    _mirror_workflow_run_lineage,
    _registry_catalog,
    _resolve_report_step,
    _resolve_workflow_metadata_catalog,
    _workflow_result_payload,
    load_python_workflow_file,
    main as pipelines_v2_cli_main,
)
from pipelines_v2.engine.vllm.capture import _capture_prompt_batch, _fill_router_features, _split_router_capture_batch
from pipelines_v2.engine.vllm.intervention_build import build_llm_kwargs, paired_request_payload
from pipelines_v2.operations.execution.interventions import run_patch_comparison
from pipelines_v2.runtime import ExecutionPlan
from pipelines_v2.runtime.specs import runner_spec_from_dict
from pipelines_v2.runtime.remote_executor import _artifact_id_for, execute_remote, execute_remote_many, merge_remote_shards
from pipelines_v2.runtime.modal_worker import (
    _modal_shard_count,
    _mounted_volumes,
    _resolved_local_python_sources,
    _resolved_runtime_spec,
    run_many_on_modal,
    run_on_modal,
)
from pipelines_v2.storage.artifacts import InlineOperationArtifact
from pipelines_v2.storage.composite import preferred_workflow_metadata_catalog
from pipelines_v2.workflow.progress import FileWorkflowProgressStore, WorkflowProgressSink
from pipelines_v2.testing import (
    ArtifactStoreContractSuite,
    CatalogContractSuite,
    EngineContractSuite,
    RunnerContractSuite,
    assert_artifact_manifest_valid,
    make_toy_capture_spec,
    make_toy_dataset,
)
from pipelines_v2.workflow.records import WorkflowRunRecord, WorkflowStepContext, WorkflowStepRecord
def _test_prompt_section_metadata(rendered_prompt: str) -> dict[str, object]:
    strategy_marker = "STRATEGY\n"
    settings_marker = "\n\nSETTINGS\n"
    strategy_start = rendered_prompt.index(strategy_marker) + len(strategy_marker)
    strategy_end = rendered_prompt.index(settings_marker, strategy_start)
    settings_start = strategy_end + len(settings_marker)
    return {
        "token_sections": {
            "STRATEGY": {"char_start": strategy_start, "char_end": strategy_end},
            "SETTINGS": {"char_start": settings_start, "char_end": len(rendered_prompt)},
        }
    }


def _inline_transform_seed(*, value: int) -> dict[str, Any]:
    return {"payload": {"kind": "transform_result", "value": int(value)}}


def _inline_transform_consume(*, seed: Any) -> dict[str, Any]:
    payload = seed.result() if hasattr(seed, "result") else dict(seed)
    return {"payload": {"kind": "transform_result", "value": int(payload["value"]) + 1}}


def _test_behavior_transform(
    *,
    generations: Any,
    workflow_expected_action: Any,
    workflow_expected_asset: Any,
    workflow_expected_size: Any,
    strategy_expected_action: Any,
    strategy_expected_asset: Any,
    strategy_expected_size: Any,
    setting_expected_action: Any,
    setting_expected_asset: Any,
    setting_expected_size: Any,
) -> TransformResult:
    workflow_action = workflow_expected_action.resolve_values()
    workflow_asset = workflow_expected_asset.resolve_values()
    workflow_size = workflow_expected_size.resolve_values()
    strategy_action = strategy_expected_action.resolve_values()
    strategy_asset = strategy_expected_asset.resolve_values()
    strategy_size = strategy_expected_size.resolve_values()
    setting_action = setting_expected_action.resolve_values()
    setting_asset = setting_expected_asset.resolve_values()
    setting_size = setting_expected_size.resolve_values()

    generated: dict[str, dict[str, str]] = {}
    for item in generations.generations():
        payload = json.loads(str(item["text"]))
        generated[str(item["example_key"])] = {
            "action": str(payload.get("action") or "").lower(),
            "asset": str(payload.get("asset") or "").upper(),
            "size": str(payload.get("size") or "").lower(),
        }

    def matches(example_key: str, expected_action: Any, expected_asset: Any, expected_size: Any) -> bool:
        value = generated[example_key]
        return (
            value["action"] == str(expected_action or "").lower()
            and value["asset"] == str(expected_asset or "").upper()
            and value["size"] == str(expected_size or "").lower()
        )

    labels = {
        "generated_action": {key: value["action"] for key, value in generated.items()},
        "matches_workflow_expected": {
            key: matches(key, workflow_action.get(key), workflow_asset.get(key), workflow_size.get(key))
            for key in generated
        },
        "source_following_side": {
            key: (
                "both"
                if matches(key, strategy_action.get(key), strategy_asset.get(key), strategy_size.get(key))
                and matches(key, setting_action.get(key), setting_asset.get(key), setting_size.get(key))
                else "strategy"
                if matches(key, strategy_action.get(key), strategy_asset.get(key), strategy_size.get(key))
                else "setting"
                if matches(key, setting_action.get(key), setting_asset.get(key), setting_size.get(key))
                else "neither"
            )
            for key in generated
        },
    }
    return TransformResult(
        payload={"kind": "test_behavior_transform"},
        labels=labels,
        example_keys=sorted(generated),
    )


def _patch_comparison_row_evaluator(
    *,
    example: Any,
    baseline: Any,
    variants: Any,
) -> dict[str, Any]:
    patched = dict(variants or {}).get("main", {})
    return {
        "metrics": {
            "flipped": str(baseline.get("generated_text") or "") != str(patched.get("generated_text") or ""),
        },
        "evaluation": {
            "example_key": str(example.get("key") or ""),
            "baseline_text": str(baseline.get("generated_text") or ""),
            "patched_text": str(patched.get("generated_text") or ""),
        },
    }


def _make_phase5_like_dataset() -> Dataset:
    return Dataset.from_examples(
        (
            Example(
                key="size_conflict_train",
                prompt="SYSTEM\nChoose one.\n\nSTRATEGY\nBuy ALPHA.\n\nSETTINGS\nUse the largest size.\n",
                labels={
                    "user_text": "Buy ALPHA with the largest size",
                    "strategy_family": "trade_size_force_large",
                    "family_group": "size",
                    "conflict_present": True,
                    "strategy_lexical_split": "train",
                    "setting_lexical_split": "train",
                },
                cases={"matched_pair_id": "pair_size_train"},
                case_key="pair_size_train",
            ),
            Example(
                key="size_aligned_train",
                prompt="SYSTEM\nChoose one.\n\nSTRATEGY\nBuy ALPHA.\n\nSETTINGS\nUse the standard size.\n",
                labels={
                    "user_text": "Buy ALPHA with the standard size",
                    "strategy_family": "trade_size_force_small",
                    "family_group": "size",
                    "conflict_present": False,
                    "strategy_lexical_split": "train",
                    "setting_lexical_split": "train",
                },
                cases={"matched_pair_id": "pair_size_train"},
                case_key="pair_size_train",
            ),
            Example(
                key="size_conflict_test",
                prompt="SYSTEM\nChoose one.\n\nSTRATEGY\nAcquire ALPHA.\n\nSETTINGS\nPush size to the maximum.\n",
                labels={
                    "user_text": "Acquire ALPHA and push size to the maximum",
                    "strategy_family": "trade_size_force_large",
                    "family_group": "size",
                    "conflict_present": True,
                    "strategy_lexical_split": "test",
                    "setting_lexical_split": "test",
                },
                cases={"matched_pair_id": "pair_size_test"},
                case_key="pair_size_test",
            ),
            Example(
                key="size_aligned_test",
                prompt="SYSTEM\nChoose one.\n\nSTRATEGY\nAcquire ALPHA.\n\nSETTINGS\nKeep size modest.\n",
                labels={
                    "user_text": "Acquire ALPHA and keep size modest",
                    "strategy_family": "trade_size_force_small",
                    "family_group": "size",
                    "conflict_present": False,
                    "strategy_lexical_split": "test",
                    "setting_lexical_split": "test",
                },
                cases={"matched_pair_id": "pair_size_test"},
                case_key="pair_size_test",
            ),
            Example(
                key="activity_conflict_train",
                prompt="SYSTEM\nChoose one.\n\nSTRATEGY\nExecute a trade now.\n\nSETTINGS\nStay in observation mode.\n",
                labels={
                    "user_text": "Execute a trade now while staying in observation mode",
                    "strategy_family": "activity_force_trade",
                    "family_group": "activity",
                    "conflict_present": True,
                    "strategy_lexical_split": "train",
                    "setting_lexical_split": "train",
                },
                cases={"matched_pair_id": "pair_activity_train"},
                case_key="pair_activity_train",
            ),
            Example(
                key="activity_aligned_train",
                prompt="SYSTEM\nChoose one.\n\nSTRATEGY\nObserve only.\n\nSETTINGS\nStay in observation mode.\n",
                labels={
                    "user_text": "Observe only while staying in observation mode",
                    "strategy_family": "activity_force_observe",
                    "family_group": "activity",
                    "conflict_present": False,
                    "strategy_lexical_split": "train",
                    "setting_lexical_split": "train",
                },
                cases={"matched_pair_id": "pair_activity_train"},
                case_key="pair_activity_train",
            ),
            Example(
                key="activity_conflict_test",
                prompt="SYSTEM\nChoose one.\n\nSTRATEGY\nPlace a trade immediately.\n\nSETTINGS\nRemain in monitor-only mode.\n",
                labels={
                    "user_text": "Place a trade immediately while remaining in monitor-only mode",
                    "strategy_family": "activity_force_trade",
                    "family_group": "activity",
                    "conflict_present": True,
                    "strategy_lexical_split": "test",
                    "setting_lexical_split": "test",
                },
                cases={"matched_pair_id": "pair_activity_test"},
                case_key="pair_activity_test",
            ),
            Example(
                key="activity_aligned_test",
                prompt="SYSTEM\nChoose one.\n\nSTRATEGY\nMonitor only.\n\nSETTINGS\nRemain in monitor-only mode.\n",
                labels={
                    "user_text": "Monitor only while remaining in monitor-only mode",
                    "strategy_family": "activity_force_observe",
                    "family_group": "activity",
                    "conflict_present": False,
                    "strategy_lexical_split": "test",
                    "setting_lexical_split": "test",
                },
                cases={"matched_pair_id": "pair_activity_test"},
                case_key="pair_activity_test",
            ),
        ),
        name="phase5_like_dataset",
    )


def _write_cli_local_workflow_file(tmp_path: Path) -> Path:
    workflow_file = tmp_path / "cli_local_workflow.py"
    workflow_file.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "from pipelines_v2.api import (",
                "    CaptureSpec,",
                "    Dataset,",
                "    Example,",
                "    LocalArtifactStore,",
                "    LocalRunnerSpec,",
                "    ProbeSpec,",
                "    ReportSpec,",
                "    ResidualSite,",
                "    StepRef,",
                "    ToyEngine,",
                "    WorkflowSpec,",
                "    WorkflowStep,",
                ")",
                "",
                f"ARTIFACT_ROOT = Path({tmp_path.joinpath('artifacts').as_posix()!r})",
                f"REPORT_ARTIFACT_ROOT = Path({tmp_path.joinpath('report_artifacts').as_posix()!r})",
                f"REPORT_OUTPUT_ROOT = {str(tmp_path.joinpath('published_reports'))!r}",
                "",
                "def build_dataset():",
                "    return Dataset.from_examples(",
                "        [",
                "            Example(key='a', prompt='alpha', labels={'class': 'pos'}, case_key='c1'),",
                "            Example(key='b', prompt='beta', labels={'class': 'neg'}, case_key='c2'),",
                "            Example(key='c', prompt='gamma', labels={'class': 'pos'}, case_key='c3'),",
                "            Example(key='d', prompt='delta', labels={'class': 'neg'}, case_key='c4'),",
                "        ],",
                "        name='cli_local_workflow_dataset',",
                "    )",
                "",
                "def build_runner_specs():",
                "    return {",
                "        'capture_gpu': LocalRunnerSpec(artifacts=LocalArtifactStore(ARTIFACT_ROOT)),",
                "        'analysis_cpu': LocalRunnerSpec(artifacts=LocalArtifactStore(ARTIFACT_ROOT)),",
                "        'report_local': LocalRunnerSpec(artifacts=LocalArtifactStore(REPORT_ARTIFACT_ROOT)),",
                "    }",
                "",
                "def build_workflow(dataset=None):",
                "    dataset = dataset or build_dataset()",
                "    return WorkflowSpec(",
                "        name='cli_local_workflow',",
                "        steps=(",
                "            WorkflowStep(",
                "                name='capture',",
                "                runner='capture_gpu',",
                "                spec=CaptureSpec(",
                "                    engine=ToyEngine(hidden_size=4, num_layers=2),",
                "                    dataset=dataset,",
                "                    sites=[ResidualSite(name='resid_last', site='resid_post', layers=[0, 1])],",
                "                ),",
                "            ),",
                "            WorkflowStep(",
                "                name='probe',",
                "                runner='analysis_cpu',",
                "                spec=ProbeSpec(",
                "                    feature=StepRef('capture').feature('resid_last'),",
                "                    labels=dataset.labels('class'),",
                "                    folds=2,",
                "                    baselines=['majority'],",
                "                ),",
                "            ),",
                "            WorkflowStep(",
                "                name='report',",
                "                runner='report_local',",
                "                spec=ReportSpec(",
                "                    template='cli_local_workflow',",
                "                    output_dir=REPORT_OUTPUT_ROOT,",
                "                    inputs=[StepRef('capture'), StepRef('probe')],",
                "                ),",
                "            ),",
                "        ),",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return workflow_file


def _write_cli_mixed_catalog_workflow_file(tmp_path: Path) -> Path:
    workflow_file = tmp_path / "cli_mixed_catalog_workflow.py"
    workflow_file.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "from pipelines_v2.api import (",
                "    CaptureSpec,",
                "    Dataset,",
                "    Example,",
                "    FileCatalog,",
                "    LocalArtifactStore,",
                "    LocalRunnerSpec,",
                "    ProbeSpec,",
                "    ReportSpec,",
                "    ResidualSite,",
                "    StepRef,",
                "    ToyEngine,",
                "    WorkflowSpec,",
                "    WorkflowStep,",
                ")",
                "",
                f"ARTIFACT_ROOT = Path({tmp_path.joinpath('artifacts').as_posix()!r})",
                f"REPORT_ARTIFACT_ROOT = Path({tmp_path.joinpath('report_artifacts').as_posix()!r})",
                f"REPORT_OUTPUT_ROOT = {str(tmp_path.joinpath('published_reports'))!r}",
                f"SHARED_CATALOG_ROOT = Path({tmp_path.joinpath('shared_catalog').as_posix()!r})",
                "",
                "def build_dataset():",
                "    return Dataset.from_examples(",
                "        [",
                "            Example(key='a', prompt='alpha', labels={'class': 'pos'}, case_key='c1'),",
                "            Example(key='b', prompt='beta', labels={'class': 'neg'}, case_key='c2'),",
                "            Example(key='c', prompt='gamma', labels={'class': 'pos'}, case_key='c3'),",
                "            Example(key='d', prompt='delta', labels={'class': 'neg'}, case_key='c4'),",
                "        ],",
                "        name='cli_mixed_catalog_workflow_dataset',",
                "    )",
                "",
                "def build_runner_specs():",
                "    shared_catalog = FileCatalog(SHARED_CATALOG_ROOT)",
                "    return {",
                "        'capture_gpu': LocalRunnerSpec(artifacts=LocalArtifactStore(ARTIFACT_ROOT), catalog=shared_catalog),",
                "        'analysis_cpu': LocalRunnerSpec(artifacts=LocalArtifactStore(ARTIFACT_ROOT), catalog=shared_catalog),",
                "        'report_local': LocalRunnerSpec(artifacts=LocalArtifactStore(REPORT_ARTIFACT_ROOT)),",
                "    }",
                "",
                "def build_workflow(dataset=None):",
                "    dataset = dataset or build_dataset()",
                "    return WorkflowSpec(",
                "        name='cli_mixed_catalog_workflow',",
                "        steps=(",
                "            WorkflowStep(",
                "                name='capture',",
                "                runner='capture_gpu',",
                "                spec=CaptureSpec(",
                "                    engine=ToyEngine(hidden_size=4, num_layers=2),",
                "                    dataset=dataset,",
                "                    sites=[ResidualSite(name='resid_last', site='resid_post', layers=[0, 1])],",
                "                ),",
                "            ),",
                "            WorkflowStep(",
                "                name='probe',",
                "                runner='analysis_cpu',",
                "                spec=ProbeSpec(",
                "                    feature=StepRef('capture').feature('resid_last'),",
                "                    labels=dataset.labels('class'),",
                "                    folds=2,",
                "                    baselines=['majority'],",
                "                ),",
                "            ),",
                "            WorkflowStep(",
                "                name='report',",
                "                runner='report_local',",
                "                spec=ReportSpec(",
                "                    template='cli_mixed_catalog_workflow',",
                "                    output_dir=REPORT_OUTPUT_ROOT,",
                "                    inputs=[StepRef('capture'), StepRef('probe')],",
                "                ),",
                "            ),",
                "        ),",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return workflow_file


class _FailOnceRunner:
    def __init__(self, inner: Any, *, fail_step: str | None = None, delay_seconds: float = 0.0) -> None:
        self.inner = inner
        self.fail_step = fail_step
        self.delay_seconds = delay_seconds
        self.failed = False
        self.calls: list[str] = []
        self.catalog = inner.catalog
        self.artifacts = inner.artifacts

    def plan(self, spec: Any) -> Any:
        return self.inner.plan(spec)

    def run(self, spec: Any, *, workflow_context: Any | None = None) -> Any:
        step_name = workflow_context.step_name if workflow_context is not None else "<unknown>"
        self.calls.append(step_name)
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        if self.fail_step == step_name and not self.failed:
            self.failed = True
            raise RuntimeError(f"intentional failure for {step_name}")
        return self.inner.run(spec, workflow_context=workflow_context)


def test_dataset_labels_cases_and_selection_stay_aligned() -> None:
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="one two", labels={"kind": "x"}, case_key="case_1"),
            Example(key="b", prompt="three four", labels={"kind": "y"}, case_key="case_1"),
            Example(key="c", prompt="five six", labels={"kind": "x"}, case_key="case_2"),
        ],
        name="alignment",
    )

    small = dataset.select(keys=["a", "c"])

    assert small.example_keys() == ["a", "c"]
    assert small.labels("kind").for_examples(small.example_keys()) == ["x", "x"]
    assert small.cases("case_key").for_examples(small.example_keys()) == ["case_1", "case_2"]
    assert set(small.coverage()["prompt_hashes"]) == {"a", "c"}


def test_in_memory_source_fetches_dataset() -> None:
    source = InMemorySource.from_records(
        [
            {
                "example_id": "a",
                "prompt": "hello",
                "class": "positive",
                "case_id": "case_1",
            }
        ]
    )

    dataset = Dataset.from_source(
        source=source,
        prompt_column="prompt",
        example_key_column="example_id",
        label_columns=["class"],
        case_key_column="case_id",
    )

    assert dataset.labels("class").values == {"a": "positive"}
    assert dataset.cases("case_id").values == {"a": "case_1"}


def test_dataset_loads_jsonl_file(tmp_path: Path) -> None:
    dataset_path = tmp_path / "examples.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"example_id": "a", "prompt": "hello", "class": "positive", "case_id": "case_1"}),
                json.dumps({"example_id": "b", "prompt": "world", "class": "negative", "case_id": "case_2"}),
            ]
        ),
        encoding="utf-8",
    )

    dataset = Dataset.from_file(
        dataset_path,
        prompt_column="prompt",
        example_key_column="example_id",
        label_columns=["class"],
        case_key_column="case_id",
    )

    assert dataset.example_keys() == ["a", "b"]
    assert dataset.labels("class").values == {"a": "positive", "b": "negative"}
    assert dataset.cases("case_id").values == {"a": "case_1", "b": "case_2"}


def test_huggingface_dataset_stays_deferred_until_runtime() -> None:
    source = HuggingFaceSource(
        path="morebench/morebench",
        name="morebench_public",
        revision="abc123",
        token_env_var="HF_TOKEN",
    )

    dataset = Dataset.from_huggingface(
        source=source,
        split="test",
        prompt_column="DILEMMA",
        example_key_column="TASK_ID",
        label_columns=["THEORY"],
        case_key_column="TASK_ID",
        name="morebench_public",
    )

    payload = dataset.to_dict()

    assert dataset.is_deferred is True
    assert payload["source"] == {
        "kind": "huggingface",
        "path": "morebench/morebench",
        "name": "morebench_public",
        "revision": "abc123",
        "token_env_var": "HF_TOKEN",
    }
    assert "token" not in payload["source"]
    assert dataset.fetch["split"] == "test"
    assert [secret.env_var for secret in dataset.runtime_secrets()] == ["HF_TOKEN"]
    assert dataset.runtime_pip_packages() == ("datasets",)

    baseline = TextBaselineSpec(
        text=dataset.labels("DILEMMA"),
        labels=dataset.labels("THEORY"),
    )
    assert "datasets" in baseline.runtime_spec().pip_packages


def test_dataset_maps_official_hf_dataset_objects() -> None:
    class FakeHFDataset:
        def to_list(self) -> list[dict[str, object]]:
            return [
                {"prompt": "first", "class": "positive"},
                {"prompt": "second", "class": "negative"},
            ]

    dataset = Dataset.from_hf_dataset(
        FakeHFDataset(),
        prompt_column="prompt",
        example_key_column="example_id",
        label_columns=["class"],
        index_column="example_id",
        index_prefix="hf",
    )

    assert dataset.example_keys() == ["hf_000000", "hf_000001"]
    assert dataset.labels("class").values == {
        "hf_000000": "positive",
        "hf_000001": "negative",
    }


def test_dataset_maps_hf_hash_columns_for_grouping() -> None:
    class FakeHFDataset:
        def to_list(self) -> list[dict[str, object]]:
            return [
                {"example_id": "a", "prompt": "same", "class": "x"},
                {"example_id": "b", "prompt": "same", "class": "y"},
                {"example_id": "c", "prompt": "different", "class": "x"},
            ]

    dataset = Dataset.from_hf_dataset(
        FakeHFDataset(),
        prompt_column="prompt",
        example_key_column="example_id",
        label_columns=["class"],
        case_columns=["base_prompt_id"],
        hash_columns={"base_prompt_id": "prompt"},
    )

    groups = dataset.cases("base_prompt_id").values

    assert groups["a"] == groups["b"]
    assert groups["a"] != groups["c"]


def test_dataset_maps_hf_nested_records() -> None:
    class FakeHFDataset:
        def to_list(self) -> list[dict[str, object]]:
            return [
                {
                    "DILEMMA": "Should the agent disclose the risk?",
                    "RUBRIC": [
                        {
                            "id": "c1",
                            "title": "Identify affected stakeholders",
                            "weight": 2,
                            "annotations": {"rubric_dimension": "Identifying"},
                        },
                        {
                            "id": "c2",
                            "title": "Avoid causing unnecessary harm",
                            "weight": -3,
                            "annotations": {"rubric_dimension": "Harmless Outcome"},
                        },
                    ],
                }
            ]

    dataset = Dataset.from_hf_dataset(
        FakeHFDataset(),
        prompt_column="criterion_text",
        example_key_column="criterion_id",
        label_columns=["DILEMMA", "criterion_text", "rubric_dimension", "criterion_weight"],
        case_columns=["base_dilemma_id"],
        index_column="criterion_id",
        index_prefix="rubric",
        hash_columns={"base_dilemma_id": "DILEMMA"},
        nested_record_column="RUBRIC",
        nested_record_index_column="criterion_index",
        nested_record_field_paths={
            "criterion_text": ("title", "criterion"),
            "rubric_dimension": "annotations.rubric_dimension",
            "criterion_weight": "weight",
        },
    )

    assert dataset.example_keys() == ["rubric_000000", "rubric_000001"]
    assert dataset.labels("criterion_text").values["rubric_000000"] == "Identify affected stakeholders"
    assert dataset.labels("rubric_dimension").values["rubric_000001"] == "Harmless Outcome"
    assert dataset.labels("criterion_weight").values["rubric_000001"] == -3
    assert dataset.cases("base_dilemma_id").values["rubric_000000"] == dataset.cases("base_dilemma_id").values["rubric_000001"]


def test_transform_spec_runtime_packages_include_deferred_input_packages() -> None:
    dataset = Dataset.from_huggingface(
        source=HuggingFaceSource(path="org/dataset", name="config"),
        split="test",
        prompt_column="prompt",
        example_key_column="example_id",
        label_columns=["class"],
    )

    spec = TransformSpec(
        builder=TransformBuilder.from_function(_inline_transform_seed),
        inputs={"dataset": dataset},
    )

    assert "datasets" in spec.runtime_spec().pip_packages


def test_transform_spec_from_dict_rehydrates_dataset_inputs() -> None:
    materialized = Dataset.from_examples(
        [
            Example(
                key="a",
                prompt="hello",
                labels={"class": "positive"},
            )
        ]
    )
    deferred = Dataset.from_huggingface(
        source=HuggingFaceSource(path="org/dataset", name="config"),
        split="test",
        prompt_column="prompt",
        example_key_column="example_id",
        label_columns=["class"],
    )

    spec = TransformSpec(
        builder=TransformBuilder.from_function(_inline_transform_seed),
        inputs={"materialized": materialized, "deferred": deferred},
    )

    roundtripped = TransformSpec.from_dict(spec.to_dict())

    assert isinstance(roundtripped.inputs["materialized"], Dataset)
    assert roundtripped.inputs["materialized"].example_keys() == ["a"]
    assert isinstance(roundtripped.inputs["deferred"], Dataset)
    assert roundtripped.inputs["deferred"].is_deferred
    assert "datasets" in roundtripped.runtime_spec().pip_packages


def test_dataset_prompt_template_formats_record_columns() -> None:
    dataset = Dataset.from_records(
        [{"example_id": "a", "DILEMMA": "Choose carefully.", "label": "x"}],
        prompt_column="DILEMMA",
        prompt_template="Instruction\n{DILEMMA}",
        example_key_column="example_id",
        label_columns=["label"],
    )

    assert dataset.examples[0].prompt == "Instruction\nChoose carefully."


def test_dataset_prompt_template_formats_chat_messages() -> None:
    dataset = Dataset.from_records(
        [{"example_id": "a", "DILEMMA": "Choose carefully.", "label": "x"}],
        prompt_column="DILEMMA",
        prompt_template=(
            {
                "role": "user",
                "content": "Instruction\n{DILEMMA}",
            },
        ),
        example_key_column="example_id",
        label_columns=["label"],
    )

    assert dataset.examples[0].prompt == [
        {
            "role": "user",
            "content": "Instruction\nChoose carefully.",
        }
    ]


def test_artifact_dataset_source_materializes_dataset_from_operation_result(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    artifact_id = "dataset_artifact"
    store.make_artifact_dir(artifact_id)
    dataset = Dataset.from_examples(
        [
            Example(
                key="ex1",
                prompt="prompt\n\nanswer",
                labels={"label": "yes"},
                metadata={
                    "token_sections": {
                        "prompt": {"char_start": 0, "char_end": 6},
                        "generated": {"char_start": 8, "char_end": 14},
                    }
                },
                cases={"case_key": "case1"},
                case_key="case1",
            )
        ],
        name="artifact_backed_dataset",
    )
    result_ref = store.write_json(
        artifact_id,
        "result.json",
        {"kind": "dataset_transform_result", "dataset": dataset.to_dict()},
    )
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_kind="transform",
        schema_version=1,
        operation_spec_hash="spec_hash",
        operation_semantic_hash="semantic_hash",
        created_at=utc_now_iso(),
        engine={},
        runner={},
        input_artifact_refs=(),
        example_coverage=dataset.coverage(),
        storage_refs={"result": result_ref},
    )
    artifact = OperationArtifact(_manifest=manifest, store=store)

    loaded = ArtifactDatasetSource().fetch_dataset(artifact=artifact)
    loaded_from_dict = ArtifactDatasetSource().fetch_dataset(artifact=to_primitive(artifact))
    loaded_checked = ArtifactDatasetSource().fetch_dataset(
        artifact=artifact,
        provides_token_sections=True,
    )

    assert loaded.example_keys() == ["ex1"]
    assert loaded.examples[0].metadata["token_sections"]["generated"]["char_end"] == 14
    assert loaded_from_dict.example_keys() == ["ex1"]
    assert loaded_from_dict.examples[0].labels["label"] == "yes"
    assert loaded_checked.examples[0].metadata["token_sections"]["prompt"]["char_start"] == 0

    missing_sections = Dataset.from_examples(
        [
            Example(
                key="ex2",
                prompt="plain",
            )
        ],
        name="artifact_backed_dataset_without_sections",
    )
    missing_ref = store.write_json(
        "dataset_without_sections",
        "result.json",
        {"kind": "dataset_transform_result", "dataset": missing_sections.to_dict()},
    )
    missing_manifest = ArtifactManifest(
        artifact_id="dataset_without_sections",
        artifact_kind="transform",
        schema_version=1,
        operation_spec_hash="spec_hash",
        operation_semantic_hash="semantic_hash",
        created_at=utc_now_iso(),
        engine={},
        runner={},
        input_artifact_refs=(),
        example_coverage=missing_sections.coverage(),
        storage_refs={"result": missing_ref},
    )
    missing_artifact = OperationArtifact(_manifest=missing_manifest, store=store)

    with pytest.raises(ValueError, match="provides_token_sections=True"):
        ArtifactDatasetSource().fetch_dataset(
            artifact=missing_artifact,
            provides_token_sections=True,
        )


def test_huggingface_source_materializes_records(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class FakeHFDataset:
        def to_list(self) -> list[dict[str, object]]:
            return [
                {
                    "example_id": "a",
                    "prompt": "hello",
                    "class": "positive",
                    "case_id": "case_1",
                }
            ]

    fake_datasets = types.ModuleType("datasets")

    def fake_load_dataset(path: str, *args: object, **kwargs: object) -> FakeHFDataset:
        calls.append((path, args, kwargs))
        return FakeHFDataset()

    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setenv("HF_TOKEN", "secret-token")

    dataset = Dataset.from_source(
        source=HuggingFaceSource(
            path="org/dataset",
            name="config",
            revision="main",
            token_env_var="HF_TOKEN",
        ),
        defer=False,
        split="test",
        prompt_column="prompt",
        example_key_column="example_id",
        label_columns=["class"],
        case_key_column="case_id",
    )

    assert dataset.labels("class").values == {"a": "positive"}
    assert dataset.cases("case_id").values == {"a": "case_1"}
    assert calls == [
        (
            "org/dataset",
            ("config",),
            {
                "split": "test",
                "revision": "main",
                "token": "secret-token",
            },
        )
    ]


def test_huggingface_source_materializes_nested_records(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHFDataset:
        def to_list(self) -> list[dict[str, object]]:
            return [
                {
                    "DILEMMA": "Should the agent disclose the risk?",
                    "RUBRIC": [
                        {
                            "title": "State the relevant risk clearly",
                            "weight": 1,
                            "annotations": {"rubric_dimension": "Clear Process"},
                        }
                    ],
                }
            ]

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda path, *args, **kwargs: FakeHFDataset()
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    dataset = Dataset.from_source(
        source=HuggingFaceSource(path="org/dataset", name="config"),
        defer=False,
        split="test",
        prompt_column="criterion_text",
        example_key_column="criterion_id",
        label_columns=["criterion_text", "rubric_dimension", "criterion_weight"],
        index_column="criterion_id",
        index_prefix="rubric",
        nested_record_column="RUBRIC",
        nested_record_field_paths={
            "criterion_text": "title",
            "rubric_dimension": "annotations.rubric_dimension",
            "criterion_weight": "weight",
        },
    )

    assert dataset.example_keys() == ["rubric_000000"]
    assert dataset.labels("rubric_dimension").values == {"rubric_000000": "Clear Process"}


def test_dataset_supports_multiple_named_case_refs() -> None:
    dataset = Dataset.from_records(
        [
            {
                "example_id": "a",
                "prompt": "hello",
                "class": "positive",
                "matched_pair_id": "pair_1",
                "setting_lexical_family_id": "setting_a",
                "context_variant_id": "ctx_0",
            },
            {
                "example_id": "b",
                "prompt": "world",
                "class": "negative",
                "matched_pair_id": "pair_1",
                "setting_lexical_family_id": "setting_b",
                "context_variant_id": "ctx_1",
            },
        ],
        prompt_column="prompt",
        example_key_column="example_id",
        label_columns=["class"],
        case_columns=["matched_pair_id", "setting_lexical_family_id", "context_variant_id"],
        case_key_column="matched_pair_id",
    )

    assert dataset.cases("matched_pair_id").values == {"a": "pair_1", "b": "pair_1"}
    assert dataset.cases("setting_lexical_family_id").values == {"a": "setting_a", "b": "setting_b"}
    assert dataset.cases("context_variant_id").values == {"a": "ctx_0", "b": "ctx_1"}


def test_postgres_dataset_stays_deferred_until_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    source = PostgresSource.from_env("XENON_DATABASE_URL")

    def fail_fetch(self: PostgresSource, **kwargs: object) -> Dataset:
        raise AssertionError(f"should not fetch locally: {kwargs}")

    monkeypatch.setattr(PostgresSource, "fetch_dataset", fail_fetch)

    dataset = Dataset.from_postgres(
        source=source,
        table="public.capture_examples",
        prompt_column="prompt",
        example_key_column="example_id",
        label_columns=["class"],
        case_key_column="case_id",
        name="capture_examples",
    )

    assert dataset.is_deferred is True
    assert dataset.examples == ()
    assert dataset.to_dict()["source"]["kind"] == "postgres"
    assert dataset.to_dict()["source"]["url_env_var"] == "XENON_DATABASE_URL"
    assert "url" not in dataset.to_dict()["source"]
    assert "examples" not in dataset.to_dict()


def test_deferred_dataset_label_and_case_refs_round_trip() -> None:
    dataset = Dataset.from_postgres(
        source=PostgresSource.from_env("XENON_DATABASE_URL"),
        table="public.capture_examples",
        prompt_column="prompt",
        example_key_column="example_id",
        label_columns=["class"],
        case_key_column="case_id",
    )

    probe = ProbeSpec(
        feature="placeholder_feature",
        labels=dataset.labels("class"),
        group_by=dataset.cases("case_id"),
    )

    restored = ProbeSpec.from_dict(probe.to_dict())

    assert restored.labels.name == "class"
    assert restored.labels.dataset.is_deferred is True
    assert restored.group_by.name == "case_id"


def test_readout_specs_persist_predictions_round_trip() -> None:
    dataset = _make_phase5_like_dataset()

    probe = ProbeSpec(
        feature="placeholder_feature",
        labels=dataset.labels("conflict_present"),
        group_by=dataset.cases("matched_pair_id"),
        persist_predictions=True,
    )
    transfer = TransferProbeSpec(
        feature="placeholder_feature",
        labels=dataset.labels("conflict_present"),
        cohort_by=dataset.labels("family_group"),
        persist_predictions=True,
    )
    text = TextBaselineSpec(
        text=dataset.labels("user_text"),
        labels=dataset.labels("strategy_family"),
        persist_predictions=True,
    )

    restored_probe = ProbeSpec.from_dict(probe.to_dict())
    restored_transfer = TransferProbeSpec.from_dict(transfer.to_dict())
    restored_text = TextBaselineSpec.from_dict(text.to_dict())

    assert restored_probe.persist_predictions is True
    assert restored_transfer.persist_predictions is True
    assert restored_text.persist_predictions is True


def test_deferred_postgres_dataset_rejects_raw_url_serialization() -> None:
    source = PostgresSource(url="postgresql://example/xenon")

    with pytest.raises(ValueError, match="environment variable reference"):
        Dataset.from_postgres(
            source=source,
            table="public.capture_examples",
            prompt_column="prompt",
            example_key_column="example_id",
        )


def test_postgres_query_dataset_stays_deferred_until_runtime() -> None:
    dataset = Dataset.from_postgres(
        source=PostgresSource.from_env("XENON_DATABASE_URL"),
        sql="""
            SELECT example_id, prompt, class
            FROM public.capture_examples
            WHERE active = true
        """,
        prompt_column="prompt",
        example_key_column="example_id",
        label_columns=["class"],
        name="query_examples",
    )

    restored = Dataset.from_dict(dataset.to_dict())

    assert dataset.is_deferred is True
    assert dataset.fetch["sql"].strip().startswith("SELECT example_id")
    assert dataset.fetch.get("table") is None
    assert restored.fetch["sql"] == dataset.fetch["sql"]
    assert restored.name == "query_examples"


def test_postgres_source_query_mode_wraps_sql_and_applies_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[tuple[str, list[object] | None]] = []

    class FakeResult:
        def fetchall(self) -> list[dict[str, object]]:
            return [{"example_id": "ex_a", "prompt": "hello", "class": "positive"}]

    class FakeConnection:
        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def execute(self, sql: str, params: list[object] | None = None) -> FakeResult:
            executed.append((" ".join(sql.split()), params))
            return FakeResult()

    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.connect = lambda url, row_factory=None: FakeConnection()
    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)

    dataset = PostgresSource(url="postgresql://example/xenon").fetch_dataset(
        sql="""
            SELECT example_id, prompt, class
            FROM public.capture_examples
            WHERE active = true
        """,
        prompt_column="prompt",
        example_key_column="example_id",
        label_columns=["class"],
        limit=5,
    )

    assert dataset.name == "postgres_query"
    assert dataset.labels("class").values == {"ex_a": "positive"}
    assert executed == [
        (
            'SELECT "example_id", "prompt", "class" FROM (SELECT example_id, prompt, class FROM public.capture_examples WHERE active = true) AS src LIMIT %s',
            [5],
        )
    ]


def test_postgres_source_query_mode_pushes_prompt_hash_shard(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[tuple[str, list[object] | None]] = []

    class FakeResult:
        def fetchall(self) -> list[dict[str, object]]:
            return [
                {
                    "example_id": "ex_a",
                    "prompt": "hello",
                    "prompt_hash": "00000000000000000000000000000000",
                    "class": "positive",
                }
            ]

    class FakeConnection:
        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def execute(self, sql: str, params: list[object] | None = None) -> FakeResult:
            executed.append((" ".join(sql.split()), params))
            return FakeResult()

    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.connect = lambda url, row_factory=None: FakeConnection()
    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)

    dataset = PostgresSource(url="postgresql://example/xenon").fetch_dataset(
        sql="""
            SELECT example_id, prompt, prompt_hash, class
            FROM public.capture_examples
            WHERE active = true
        """,
        prompt_column="prompt",
        example_key_column="example_id",
        prompt_hash_column="prompt_hash",
        label_columns=["class"],
        execution_shard={"index": 2, "count": 4},
        limit=5,
    )

    assert dataset.example_keys() == ["ex_a"]
    assert executed
    sql, params = executed[0]
    assert 'FROM (SELECT example_id, prompt, prompt_hash, class FROM public.capture_examples WHERE active = true) AS src WHERE MOD(' in sql
    assert 'src."prompt_hash"' in sql
    assert params == [4, 2, 5]


def test_postgres_source_iter_dataset_batches_streams_with_shard_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[tuple[str, list[object] | None]] = []
    records = [
        {
            "example_id": f"ex_{idx}",
            "prompt": f"prompt {idx}",
            "prompt_hash": f"{idx:032x}",
            "class": "positive",
        }
        for idx in range(3)
    ]

    class FakeResult:
        def __init__(self, page: list[dict[str, object]]) -> None:
            self._page = page

        def fetchall(self) -> list[dict[str, object]]:
            return self._page

    class FakeConnection:
        def __init__(self, row_factory: object = None) -> None:
            assert row_factory is not None

        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def execute(self, sql: str, params: list[object] | None = None) -> FakeResult:
            executed.append((" ".join(sql.split()), params))
            assert params is not None
            batch_size = int(params[-2])
            offset = int(params[-1])
            return FakeResult(records[offset : offset + batch_size])

    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.connect = lambda url, row_factory=None: FakeConnection(row_factory=row_factory)
    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)

    batches = list(
        PostgresSource(url="postgresql://example/xenon").iter_dataset_batches(
            batch_size=2,
            sql="SELECT example_id, prompt, prompt_hash, class FROM public.capture_examples",
            prompt_column="prompt",
            example_key_column="example_id",
            prompt_hash_column="prompt_hash",
            label_columns=["class"],
            execution_shard={"index": 1, "count": 4},
        )
    )

    assert [batch.example_keys() for batch in batches] == [["ex_0", "ex_1"], ["ex_2"]]
    sql, params = executed[0]
    assert sql.startswith("SELECT * FROM (SELECT")
    assert " AS src WHERE MOD(" in sql
    assert 'src."prompt_hash"' in sql
    assert (
        'AS pipelines_v2_dataset_page ORDER BY pipelines_v2_dataset_page."prompt_hash", '
        'pipelines_v2_dataset_page."example_id" LIMIT %s OFFSET %s'
    ) in sql
    assert params == [4, 1, 2, 0]
    assert executed[1][1] == [4, 1, 2, 2]


def test_capture_spec_exposes_runtime_secret_requirements() -> None:
    dataset = Dataset.from_postgres(
        source=PostgresSource.from_env("XENON_DATABASE_URL"),
        table="public.capture_examples",
        prompt_column="prompt",
        example_key_column="example_id",
    )
    spec = CaptureSpec(
        engine=ToyEngine(),
        dataset=dataset,
        sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0])],
    )

    assert [secret.env_var for secret in spec.runtime_secrets()] == ["XENON_DATABASE_URL"]


def test_local_runner_capture_writes_manifest_features_and_generations(tmp_path: Path) -> None:
    runner = LocalRunner(
        artifacts=LocalArtifactStore(tmp_path / "artifacts"),
        catalog=FileCatalog(tmp_path / "catalog"),
    )
    spec = CaptureSpec(
        engine=ToyEngine(hidden_size=3, num_layers=2),
        dataset=make_toy_dataset(),
        sites=[
            ResidualSite(
                name="resid_last",
                site="resid_post",
                layers=[0, 1],
                tokens=TokenSelector.last(),
            ),
            MoERoutingSite(
                name="router_last",
                layers=[1],
                tokens=TokenSelector.last(),
                record=[
                    RoutingRecord.gate_logits(),
                    RoutingRecord.routing_decisions(required=False),
                    RoutingRecord.topk_from_gate(k=2),
                    RoutingRecord.expert_load(source="topk_from_gate"),
                ],
            ),
        ],
        generation=GenerationSpec(enabled=True, max_tokens=4),
    )

    artifact = runner.run(spec)

    assert_artifact_manifest_valid(artifact.manifest())
    assert artifact.localize().exists()
    assert (tmp_path / "catalog" / f"{artifact.id}.json").exists()
    resid_ref = artifact.manifest().storage_refs["features"]["resid_last"]
    router_ref = artifact.manifest().storage_refs["features"]["router_last"]
    assert resid_ref["format"] == "residual_safetensors_v2"
    assert resid_ref["tensor_path"].endswith("features/feature_tensors.safetensors")
    assert resid_ref["metadata_path"].endswith("features/resid_last.metadata.json")
    assert router_ref["format"] == "moe_routing_safetensors_v1"
    assert router_ref["tensor_path"] == resid_ref["tensor_path"]
    assert router_ref["metadata_path"].endswith("features/router_last.metadata.json")
    tensors = load_file(artifact.localize() / "features" / "feature_tensors.safetensors")
    assert "feature_0_layer_0_values_0" in tensors
    assert "feature_0_layer_1_values_0" in tensors

    resid = artifact.feature("resid_last").load()
    assert resid["kind"] == "residual"
    assert resid["storage"]["format"] == "safetensors"
    assert set(resid["layers"]) == {"0", "1"}
    assert set(resid["layers"]["0"]) == {"ex_a", "ex_b"}
    assert len(resid["layers"]["0"]["ex_a"]["values"][0]) == 3

    routing = artifact.feature("router_last").load()
    token_record = next(iter(routing["layers"]["1"]["ex_a"]["records"].values()))
    assert "gate_logits" in token_record
    assert token_record["gate_logits"].dtype == np.float16
    assert token_record["routing_decisions"]["source"] == "observed"
    assert token_record["topk_from_gate"]["source"] == "derived_from_gate_logits"
    assert len(token_record["topk_from_gate"]["expert_ids"]) == 2
    assert len(token_record["expert_load"]["counts"]) == 2

    generations = artifact.generations()
    assert [item["example_key"] for item in generations] == ["ex_a", "ex_b"]

    manifest_file = artifact.localize() / "manifest.json"
    with manifest_file.open("r", encoding="utf-8") as f:
        manifest_payload = json.load(f)
    assert manifest_payload["artifact_kind"] == "capture"
    assert manifest_payload["example_coverage"]["example_count"] == 2


def test_capture_can_select_prompt_and_generated_sections_in_one_pass(tmp_path: Path) -> None:
    runner = LocalRunner(
        artifacts=LocalArtifactStore(tmp_path / "artifacts"),
        catalog=FileCatalog(tmp_path / "catalog"),
    )
    spec = CaptureSpec(
        engine=ToyEngine(hidden_size=3, num_layers=1, sequence_length=8),
        dataset=make_toy_dataset(),
        sites=[
            ResidualSite(
                name="prompt_tokens",
                site="resid_post",
                layers=[0],
                tokens=TokenSelector.section("prompt"),
            ),
            ResidualSite(
                name="generated_tokens",
                site="resid_post",
                layers=[0],
                tokens=TokenSelector.section("generated"),
            ),
        ],
        generation=GenerationSpec(
            enabled=True,
            max_tokens=2,
            capture_generated_tokens=True,
        ),
    )

    artifact = runner.run(spec)
    prompt_feature = artifact.feature("prompt_tokens").load()
    generated_feature = artifact.feature("generated_tokens").load()

    assert prompt_feature["layers"]["0"]["ex_a"]["tokens"] == list(range(8))
    assert generated_feature["layers"]["0"]["ex_a"]["tokens"] == [8, 9]
    assert generated_feature["layers"]["0"]["ex_a"]["token_sections"]["generated"] == [0, 1]


def test_local_runner_bundles_tensor_features_into_one_safetensors_file(tmp_path: Path) -> None:
    runner = LocalRunner(
        artifacts=LocalArtifactStore(tmp_path / "artifacts"),
    )
    spec = CaptureSpec(
        engine=ToyEngine(hidden_size=3, num_layers=2),
        dataset=make_toy_dataset(),
        sites=[
            ResidualSite(name="resid_last", site="resid_post", layers=[0], tokens=TokenSelector.last()),
            MoERoutingSite(
                name="router_last",
                layers=[1],
                tokens=TokenSelector.last(),
                record=[RoutingRecord.gate_logits(dtype="float16")],
            ),
            ResidualSite(
                name="resid_full",
                site="resid_post",
                layers=[1],
                tokens=TokenSelector.full_sequence(),
            ),
        ],
    )

    artifact = runner.run(spec)
    resid_last_ref = artifact.manifest().storage_refs["features"]["resid_last"]
    router_ref = artifact.manifest().storage_refs["features"]["router_last"]
    resid_full_ref = artifact.manifest().storage_refs["features"]["resid_full"]

    assert resid_last_ref["tensor_path"] == resid_full_ref["tensor_path"]
    assert router_ref["tensor_path"] == resid_last_ref["tensor_path"]
    shared_tensor = Path(resid_last_ref["tensor_path"])
    assert shared_tensor.name == "feature_tensors.safetensors"
    assert shared_tensor.exists()
    bundle = load_file(str(shared_tensor))
    assert len(bundle) == 4
    assert "feature_0_layer_0_values_0" in bundle
    assert "feature_2_layer_1_values_0" in bundle


def test_local_runner_applies_runtime_env_and_runner_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pipelines_v2.engine.base import PythonRuntimeSpec

    class EnvEngine:
        def identity(self) -> dict[str, Any]:
            return {"kind": "env_engine"}

        def semantic_identity(self) -> dict[str, Any]:
            return {"kind": "env_engine"}

        def capabilities(self) -> set[EngineCapability]:
            return {EngineCapability.RESIDUAL_CAPTURE}

        def runtime_spec(self) -> PythonRuntimeSpec:
            return PythonRuntimeSpec(
                env={
                    "XENON_TEST_SPEC_ONLY": "spec_only",
                    "XENON_TEST_SHARED": "spec_value",
                }
            )

        def planning_errors(self, spec: Any) -> tuple[str, ...]:
            del spec
            return ()

        def capture(self, spec: CaptureSpec) -> EngineCaptureResult:
            del spec
            return EngineCaptureResult(
                features={},
                metadata={
                    "spec_only": os.environ.get("XENON_TEST_SPEC_ONLY"),
                    "shared": os.environ.get("XENON_TEST_SHARED"),
                    "runner_only": os.environ.get("XENON_TEST_RUNNER_ONLY"),
                },
            )

    monkeypatch.delenv("XENON_TEST_SPEC_ONLY", raising=False)
    monkeypatch.delenv("XENON_TEST_RUNNER_ONLY", raising=False)
    monkeypatch.setenv("XENON_TEST_SHARED", "outside")

    runner = LocalRunner(
        resources=LocalResources(
            env={
                "XENON_TEST_SHARED": "runner_value",
                "XENON_TEST_RUNNER_ONLY": "runner_only",
            }
        ),
        artifacts=LocalArtifactStore(tmp_path / "artifacts"),
    )
    artifact = runner.run(
        CaptureSpec(
            engine=EnvEngine(),
            dataset=make_toy_dataset(),
            sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0])],
        )
    )

    assert artifact.manifest().metadata == {
        "spec_only": "spec_only",
        "shared": "runner_value",
        "runner_only": "runner_only",
    }
    assert os.environ["XENON_TEST_SHARED"] == "outside"
    assert "XENON_TEST_SPEC_ONLY" not in os.environ
    assert "XENON_TEST_RUNNER_ONLY" not in os.environ


def test_fill_router_features_error_includes_actual_and_discovered_layers() -> None:
    feature_payloads = {
        "router_last": {
            "kind": "moe_routing",
            "routing_policy": {"source": "vllm_gate_logits", "observed_routing_decisions": True},
            "layers": {"0": {}, "4": {}},
        }
    }
    site = MoERoutingSite(name="router_last", layers=[0, 4], tokens=TokenSelector.last())
    example = Example(key="ex_a", prompt="alpha")

    with pytest.raises(RuntimeError, match="captured router layers=\\[4\\].*discovered MoE layers=\\[4, 8\\]"):
        _fill_router_features(
            feature_payloads=feature_payloads,
            routing_sites=[site],
            router_data={4: {"logits": np.zeros((1, 8), dtype=np.float32)}},
            example=example,
            token_count=1,
            token_sections={},
            discovered_router_layers=[4, 8],
        )


def test_fill_router_features_error_includes_captured_length_for_token_mismatch() -> None:
    feature_payloads = {
        "router_last": {
            "kind": "moe_routing",
            "routing_policy": {"source": "vllm_gate_logits", "observed_routing_decisions": True},
            "layers": {"0": {}},
        }
    }
    site = MoERoutingSite(name="router_last", layers=[0], tokens=TokenSelector.last())

    with pytest.raises(RuntimeError, match="captured router logits only have length 4"):
        _fill_router_features(
            feature_payloads=feature_payloads,
            routing_sites=[site],
            router_data={0: {"logits": np.zeros((4, 8), dtype=np.float32)}},
            example=Example(key="ex_a", prompt="alpha"),
            token_count=10,
            token_sections={},
            discovered_router_layers=[0],
        )


def test_split_router_capture_batch_preserves_example_order() -> None:
    examples = [
        Example(key="ex_a", prompt="alpha"),
        Example(key="ex_b", prompt="beta"),
    ]
    raw_router = {
        4: {
            "logits": np.asarray(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0],
                ],
                dtype=np.float32,
            ),
            "topk_ids": np.asarray(
                [
                    [2, 1],
                    [2, 1],
                    [1, 0],
                    [1, 0],
                    [0, 2],
                ],
                dtype=np.int64,
            ),
            "topk_weights": np.asarray(
                [
                    [0.7, 0.3],
                    [0.8, 0.2],
                    [0.6, 0.4],
                    [0.55, 0.45],
                    [0.9, 0.1],
                ],
                dtype=np.float32,
            ),
        }
    }

    split = _split_router_capture_batch(
        raw_router=raw_router,
        examples=examples,
        prompt_lengths=[2, 3],
    )

    assert split["ex_a"][4]["logits"].shape == (2, 3)
    assert split["ex_b"][4]["logits"].shape == (3, 3)
    assert np.allclose(split["ex_a"][4]["logits"][0], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.array_equal(split["ex_b"][4]["topk_ids"][-1], np.asarray([0, 2], dtype=np.int64))


def test_split_router_capture_batch_can_ignore_decode_suffix_rows() -> None:
    examples = [
        Example(key="ex_a", prompt="alpha"),
        Example(key="ex_b", prompt="beta"),
    ]
    raw_router = {
        4: {
            "logits": np.asarray(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0],
                    [99.0, 99.0, 99.0],
                    [88.0, 88.0, 88.0],
                ],
                dtype=np.float32,
            ),
            "topk_ids": np.asarray(
                [
                    [2, 1],
                    [2, 1],
                    [1, 0],
                    [1, 0],
                    [0, 2],
                    [0, 1],
                    [1, 2],
                ],
                dtype=np.int64,
            ),
            "topk_weights": np.asarray(
                [
                    [0.7, 0.3],
                    [0.8, 0.2],
                    [0.6, 0.4],
                    [0.55, 0.45],
                    [0.9, 0.1],
                    [1.0, 0.0],
                    [1.0, 0.0],
                ],
                dtype=np.float32,
            ),
        }
    }

    split = _split_router_capture_batch(
        raw_router=raw_router,
        examples=examples,
        prompt_lengths=[2, 3],
        allow_trailing_rows=True,
    )

    assert split["ex_a"][4]["logits"].shape == (2, 3)
    assert split["ex_b"][4]["logits"].shape == (3, 3)
    assert np.allclose(split["ex_b"][4]["logits"][-1], np.asarray([13.0, 14.0, 15.0], dtype=np.float32))


def test_capture_prompt_batch_uses_one_generate_call_when_generation_enabled() -> None:
    vllm_module = types.ModuleType("vllm")

    class _SamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            self.max_tokens = kwargs.get("max_tokens")
            self.temperature = kwargs.get("temperature")
            self.extra_args = kwargs.get("extra_args")

    vllm_module.SamplingParams = _SamplingParams
    sys.modules["vllm"] = vllm_module

    class _FakeTokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool) -> Any:
            assert add_special_tokens is False
            assert return_offsets_mapping is True
            token_ids = [ord(char) % 17 for char in text]
            offsets = [(idx, idx + 1) for idx in range(len(text))]
            return types.SimpleNamespace(input_ids=token_ids, offset_mapping=offsets)

    class _FakeCompletion:
        def __init__(self, text: str, token_ids: list[int], *, reasoning_content: str | None = None) -> None:
            self.text = text
            self.token_ids = token_ids
            self.finish_reason = "length"
            self.reasoning_content = reasoning_content

    class _FakeRequestOutput:
        def __init__(
            self,
            request_id: str,
            text: str,
            token_ids: list[int],
            *,
            reasoning_content: str | None = None,
        ) -> None:
            self.request_id = request_id
            self.outputs = [
                _FakeCompletion(
                    text=text,
                    token_ids=token_ids,
                    reasoning_content=reasoning_content,
                )
            ]

    class _FakeLLM:
        def __init__(self) -> None:
            self.generate_calls: list[dict[str, Any]] = []

        def generate(self, *, prompts: list[dict[str, Any]], sampling_params: Any) -> list[Any]:
            self.generate_calls.append({"prompts": prompts, "sampling_params": sampling_params})
            return [
                _FakeRequestOutput("req-0", "answer-a", [101, 102]),
                _FakeRequestOutput("req-1", "answer-b", [201]),
            ]

    llm = _FakeLLM()
    tokenizer = _FakeTokenizer()
    examples = [
        Example(key="ex_a", prompt="ab"),
        Example(key="ex_b", prompt="cde"),
    ]

    records = _capture_prompt_batch(
        llm=llm,
        tokenizer=tokenizer,
        examples=examples,
        add_generation_prompt=False,
        require_sections=False,
        prompt_metadata_builder=None,
        wants_residual=False,
        wants_routing=False,
        wants_generation=True,
        generation_max_tokens=12,
        generation_temperature=0.3,
        capture_reasoning=False,
    )

    assert len(llm.generate_calls) == 1
    sampling_params = llm.generate_calls[0]["sampling_params"]
    assert sampling_params.max_tokens == 12
    assert sampling_params.temperature == 0.3
    assert [record["generation_result"]["text"] for record in records] == ["answer-a", "answer-b"]
    assert [record["generation_result"]["request_id"] for record in records] == ["req-0", "req-1"]
    performance = records[0]["_batch_performance"]
    assert performance["request_count"] == 2
    assert performance["prompt_tokens"] == 5
    assert performance["generated_tokens"] == 3
    assert performance["generation_seconds"] >= 0.0


def test_generation_spec_preserves_uncapped_max_tokens() -> None:
    spec = GenerationSpec.from_dict(
        {
            "enabled": True,
            "max_tokens": None,
            "temperature": 0.0,
            "top_p": 1.0,
        }
    )

    assert spec.max_tokens is None
    capture = replace(make_toy_capture_spec(), generation=spec)
    payload = capture.to_dict()
    assert payload["generation"]["max_tokens"] is None
    assert CaptureSpec.from_dict(payload).generation.max_tokens is None


def test_capture_prompt_batch_passes_uncapped_max_tokens_to_vllm() -> None:
    vllm_module = types.ModuleType("vllm")

    class _SamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            self.max_tokens = kwargs.get("max_tokens")
            self.temperature = kwargs.get("temperature")

    vllm_module.SamplingParams = _SamplingParams
    sys.modules["vllm"] = vllm_module

    class _FakeTokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool) -> Any:
            return types.SimpleNamespace(
                input_ids=[1, 2],
                offset_mapping=[(0, 1), (1, 2)],
            )

    class _FakeCompletion:
        text = "answer"
        token_ids = [101]
        finish_reason = "stop"

    class _FakeRequestOutput:
        request_id = "req-0"
        outputs = [_FakeCompletion()]

    class _FakeLLM:
        def __init__(self) -> None:
            self.sampling_params: Any | None = None

        def generate(self, *, prompts: list[dict[str, Any]], sampling_params: Any) -> list[Any]:
            self.sampling_params = sampling_params
            return [_FakeRequestOutput()]

    llm = _FakeLLM()
    _capture_prompt_batch(
        llm=llm,
        tokenizer=_FakeTokenizer(),
        examples=[Example(key="ex_a", prompt="ab")],
        add_generation_prompt=False,
        require_sections=False,
        prompt_metadata_builder=None,
        wants_residual=False,
        wants_routing=False,
        wants_generation=True,
        generation_max_tokens=None,
        generation_temperature=0.0,
        capture_reasoning=False,
    )

    assert llm.sampling_params is not None
    assert llm.sampling_params.max_tokens is None


def test_capture_prompt_batch_generated_section_uses_saved_hidden_rows(tmp_path: Path) -> None:
    vllm_module = types.ModuleType("vllm")

    class _SamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            self.max_tokens = kwargs.get("max_tokens")
            self.temperature = kwargs.get("temperature")
            self.extra_args = kwargs.get("extra_args")

    vllm_module.SamplingParams = _SamplingParams
    sys.modules["vllm"] = vllm_module

    hidden_path = tmp_path / "req-0.safetensors"
    save_file(
        {"hidden_states": np.zeros((3, 1, 2), dtype=np.float32)},
        str(hidden_path),
    )

    class _FakeTokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool) -> Any:
            assert add_special_tokens is False
            assert return_offsets_mapping is True
            token_ids = [ord(char) % 17 for char in text]
            offsets = [(idx, idx + 1) for idx in range(len(text))]
            return types.SimpleNamespace(input_ids=token_ids, offset_mapping=offsets)

    class _FakeCompletion:
        text = "answer-a"
        token_ids = [101, 102]
        finish_reason = "length"

    class _FakeRequestOutput:
        request_id = "req-0"
        outputs = [_FakeCompletion()]
        kv_transfer_params = {"hidden_states_path": str(hidden_path)}

    class _FakeLLM:
        def __init__(self) -> None:
            self.sampling_params: Any | None = None

        def generate(self, *, prompts: list[dict[str, Any]], sampling_params: Any) -> list[Any]:
            self.sampling_params = sampling_params
            return [_FakeRequestOutput()]

    llm = _FakeLLM()
    records = _capture_prompt_batch(
        llm=llm,
        tokenizer=_FakeTokenizer(),
        examples=[Example(key="ex_a", prompt="ab")],
        add_generation_prompt=False,
        require_sections=False,
        prompt_metadata_builder=None,
        wants_residual=True,
        wants_routing=False,
        wants_generation=True,
        generation_max_tokens=12,
        generation_temperature=0.3,
        capture_reasoning=False,
        capture_generated_tokens=True,
    )

    assert records[0]["generated_token_count"] == 2
    assert records[0]["captured_generated_token_count"] == 1
    assert records[0]["residual_token_count"] == 3
    assert records[0]["residual_token_sections"]["prompt"] == [0, 1]
    assert records[0]["residual_token_sections"]["generated"] == [2]
    assert records[0]["residual"].shape == (1, 3, 2)
    assert not hidden_path.exists()
    assert llm.sampling_params.extra_args == {
        "kv_transfer_params": {"include_output_tokens": True}
    }


def test_generation_result_from_output_keeps_full_generated_token_stream() -> None:
    from pipelines_v2.engine.vllm.capture import _generation_result_from_output

    class _FakeReasoningParser:
        start_token = "<think>"
        end_token = "</think>"
        start_token_id = 101
        end_token_id = 102

        def extract_reasoning(self, model_output: str, request: Any) -> tuple[str | None, str | None]:
            body = model_output.replace(self.start_token, "", 1)
            reasoning, _, content = body.partition(self.end_token)
            return reasoning.strip(), content.strip()

    class _FakeCompletion:
        def __init__(self) -> None:
            self.text = "<think>\ncompare both options\n</think>\nSELL"
            self.token_ids = [101, 201, 202, 102, 301]
            self.finish_reason = "stop"

    class _FakeRequestOutput:
        def __init__(self) -> None:
            self.request_id = "req-0"
            self.outputs = [_FakeCompletion()]

    result = _generation_result_from_output(
        _FakeRequestOutput(),
        capture_reasoning=False,
        reasoning_parser=_FakeReasoningParser(),
    )

    assert result["text"] == "SELL"
    assert result["generated_token_ids"] == [101, 201, 202, 102, 301]
    assert "reasoning_text" not in result


def test_generation_result_from_output_includes_reasoning_fields_when_requested() -> None:
    from pipelines_v2.engine.vllm.capture import _generation_result_from_output

    class _FakeReasoningParser:
        start_token = "<think>"
        end_token = "</think>"
        start_token_id = 101
        end_token_id = 102

        def extract_reasoning(self, model_output: str, request: Any) -> tuple[str | None, str | None]:
            body = model_output.replace(self.start_token, "", 1)
            reasoning, _, content = body.partition(self.end_token)
            return reasoning.strip(), content.strip()

    class _FakeCompletion:
        def __init__(self) -> None:
            self.text = "<think>\ncompare both options\n</think>\nSELL"
            self.token_ids = [101, 201, 202, 102, 301]
            self.finish_reason = "stop"

    class _FakeRequestOutput:
        def __init__(self) -> None:
            self.request_id = "req-0"
            self.outputs = [_FakeCompletion()]

    result = _generation_result_from_output(
        _FakeRequestOutput(),
        capture_reasoning=True,
        reasoning_parser=_FakeReasoningParser(),
    )

    assert result["text"] == "SELL"
    assert result["generated_token_ids"] == [101, 201, 202, 102, 301]
    assert result["reasoning_text"] == "compare both options"


def test_generation_result_supports_vllm_parser_engine_reasoning_markers() -> None:
    from pipelines_v2.engine.vllm.capture import _generation_result_from_output

    class _FakeReasoningParser:
        reasoning_start_str = "<think>"
        reasoning_end_str = "</think>"

        def extract_reasoning(self, model_output: str, request: Any) -> tuple[str | None, str | None]:
            body = model_output.replace(self.reasoning_start_str, "", 1)
            reasoning, _, content = body.partition(self.reasoning_end_str)
            return reasoning.strip(), content.strip()

    class _FakeCompletion:
        text = "<think>compare both options</think>SELL"
        token_ids = [101, 201, 102, 301]
        finish_reason = "stop"

    class _FakeRequestOutput:
        request_id = "req-0"
        outputs = [_FakeCompletion()]

    result = _generation_result_from_output(
        _FakeRequestOutput(),
        capture_reasoning=True,
        reasoning_parser=_FakeReasoningParser(),
    )

    assert result["text"] == "SELL"
    assert result["reasoning_text"] == "compare both options"


def test_prompt_token_ids_use_chat_template_tokenization_and_rebase_sections() -> None:
    from pipelines_v2.engine.vllm.capture import _prompt_token_ids

    class _FakeTokenizer:
        def apply_chat_template(self, prompt: Any, **kwargs: Any) -> Any:
            assert isinstance(prompt, list)
            if kwargs.get("tokenize") is True:
                return [99, 1, 2, 88]
            return "AB"

        def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool) -> Any:
            assert text == "AB"
            assert add_special_tokens is False
            assert return_offsets_mapping is True
            return types.SimpleNamespace(
                input_ids=[1, 2],
                offset_mapping=[(0, 1), (1, 2)],
            )

    example = Example(
        key="ex-a",
        prompt=[{"role": "user", "content": "irrelevant"}],
        metadata={"token_sections": {"MARKET": {"char_start": 0, "char_end": 2}}},
    )

    result = _prompt_token_ids(
        tokenizer=_FakeTokenizer(),
        example=example,
        add_generation_prompt=True,
        require_sections=True,
        prompt_metadata_builder=None,
        tool_choice="required",
    )

    assert result["token_ids"] == [99, 1, 2, 88]
    assert result["token_sections"] == {"MARKET": [1, 2]}


def test_runner_plan_reports_missing_capabilities() -> None:
    runner = LocalRunner()
    spec = CaptureSpec(
        engine=ToyEngine(enabled_capabilities=frozenset({EngineCapability.GENERATION})),
        dataset=make_toy_dataset(),
        sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0])],
    )

    plan = runner.plan(spec)

    assert plan.missing_capabilities == {EngineCapability.RESIDUAL_CAPTURE}
    with pytest.raises(CapabilityError):
        plan.validate()


def test_vllm_engine_plan_allows_router_capture_with_batch_gt_1(tmp_path: Path) -> None:
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    spec = CaptureSpec(
        engine=VLLMEngine(
            model_id="/models/Qwen/Qwen3-30B-A3B",
            max_num_seqs=2,
            enable_prefix_caching=False,
        ),
        dataset=make_toy_dataset(),
        sites=[
            MoERoutingSite(
                name="router_last",
                layers=[0],
                tokens=TokenSelector.last(),
                record=[RoutingRecord.gate_logits()],
            )
        ],
    )

    plan = runner.plan(spec)

    assert not any("MoE routing capture" in error for error in plan.errors)
    plan.validate()


def test_vllm_engine_plan_allows_router_capture_without_eager(tmp_path: Path) -> None:
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    spec = CaptureSpec(
        engine=VLLMEngine(
            model_id="/models/Qwen/Qwen3-30B-A3B",
            enforce_eager=False,
            max_num_seqs=1,
            enable_prefix_caching=False,
        ),
        dataset=make_toy_dataset(),
        sites=[
            MoERoutingSite(
                name="router_last",
                layers=[0],
                tokens=TokenSelector.last(),
                record=[RoutingRecord.gate_logits()],
            )
        ],
    )

    plan = runner.plan(spec)

    assert not any("enforce_eager=True" in error for error in plan.errors)
    plan.validate()


def test_vllm_engine_plan_rejects_router_capture_with_prefix_caching(tmp_path: Path) -> None:
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    spec = CaptureSpec(
        engine=VLLMEngine(
            model_id="/models/Qwen/Qwen3-30B-A3B",
            enforce_eager=True,
            max_num_seqs=1,
            enable_prefix_caching=True,
        ),
        dataset=make_toy_dataset(),
        sites=[
            MoERoutingSite(
                name="router_last",
                layers=[0],
                tokens=TokenSelector.last(),
                record=[RoutingRecord.gate_logits()],
            )
        ],
    )

    plan = runner.plan(spec)

    assert any("enable_prefix_caching=False" in error for error in plan.errors)
    with pytest.raises(SpecValidationError, match="enable_prefix_caching=False"):
        plan.validate()


def test_local_runner_resolves_deferred_dataset_in_runtime(tmp_path: Path) -> None:
    dataset = Dataset.from_source(
        source=InMemorySource.from_records(
            [
                {
                    "example_id": "a",
                    "prompt": "hello",
                    "class": "positive",
                    "case_id": "case_1",
                }
            ]
        ),
        defer=True,
        prompt_column="prompt",
        example_key_column="example_id",
        label_columns=["class"],
        case_key_column="case_id",
        name="runtime_bound",
    )
    runner = LocalRunner(
        artifacts=LocalArtifactStore(tmp_path / "artifacts"),
    )
    spec = CaptureSpec(
        engine=ToyEngine(hidden_size=3, num_layers=2),
        dataset=dataset,
        sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0], tokens=TokenSelector.last())],
    )

    artifact = runner.run(spec)
    residual = artifact.feature("resid_last").load()

    assert artifact.manifest().example_coverage["example_count"] == 1
    assert sorted(residual["layers"]["0"].keys()) == ["a"]


def test_spec_hash_is_stable_for_same_capture_spec() -> None:
    spec_a = make_toy_capture_spec()
    spec_b = make_toy_capture_spec()

    assert spec_a.spec_hash() == spec_b.spec_hash()


def test_contract_helpers_smoke(tmp_path: Path) -> None:
    EngineContractSuite(ToyEngine).run(
        required_capabilities=[
            EngineCapability.GENERATION,
            EngineCapability.RESIDUAL_CAPTURE,
        ]
    )
    RunnerContractSuite(LocalRunner).run_capture_smoke(tmp_path)
    ArtifactStoreContractSuite(LocalArtifactStore).run_json_roundtrip(tmp_path)
    CatalogContractSuite(FileCatalog).run_record_smoke(tmp_path)


def test_capture_spec_round_trips_from_dict() -> None:
    spec = make_toy_capture_spec()

    restored = CaptureSpec.from_dict(spec.to_dict())

    assert restored.to_dict() == spec.to_dict()


def test_vllm_engine_from_dict_preserves_unknown_runtime_options() -> None:
    engine = VLLMEngine.from_dict(
        {
            "kind": "vllm",
            "model_id": "Qwen/Qwen3-30B-A3B",
            "distributed_executor_backend": "mp",
            "future_backend_option": "kept",
            "extra": {"existing": "value"},
        }
    )

    assert engine.distributed_executor_backend == "mp"
    assert engine.extra == {"existing": "value", "future_backend_option": "kept"}


def test_workflow_spec_rehydrates_vllm_engine_with_unknown_runtime_options() -> None:
    capture_payload = make_toy_capture_spec().to_dict()
    capture_payload["engine"] = {
        "kind": "vllm",
        "model_id": "Qwen/Qwen3-30B-A3B",
        "distributed_executor_backend": "mp",
        "future_backend_option": "kept",
    }
    workflow_payload = WorkflowSpec(
        name="vllm_capture",
        steps=(
            WorkflowStep(
                name="capture",
                runner="gpu",
                spec=CaptureSpec.from_dict(capture_payload),
            ),
        ),
    ).to_dict()

    restored = WorkflowSpec.from_dict(workflow_payload)
    engine = restored.steps[0].spec.engine

    assert isinstance(engine, VLLMEngine)
    assert engine.distributed_executor_backend == "mp"
    assert engine.extra["future_backend_option"] == "kept"


def test_vllm_engine_capture_calls_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = VLLMEngine(model_id="Qwen/Qwen2.5-0.5B-Instruct")
    spec = make_toy_capture_spec()
    observed: dict[str, object] = {}

    def fake_run_vllm_capture(*, engine: VLLMEngine, spec: CaptureSpec) -> EngineCaptureResult:
        observed["engine"] = engine
        observed["spec"] = spec
        return EngineCaptureResult(features={}, generations=[], metadata={"backend": "vllm"})

    monkeypatch.setattr("pipelines_v2.engine.vllm.capture.run_vllm_capture", fake_run_vllm_capture)

    result = engine.capture(spec)

    assert result.metadata["backend"] == "vllm"
    assert observed == {"engine": engine, "spec": spec}


def test_remote_executor_uses_engine_and_store_registries(tmp_path: Path) -> None:
    spec = make_toy_capture_spec()

    manifest = execute_remote(
        runner_config={"kind": "modal", "resources": {"gpu": "L4"}},
        store_config={"kind": "local", "root": str(tmp_path / "artifacts")},
        spec_payload=spec.to_dict(),
    )

    assert manifest["artifact_kind"] == "capture"
    manifest_path = Path(manifest["storage_refs"]["manifest"]["path"])
    assert manifest_path.exists()
    with manifest_path.open("r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["engine"]["kind"] == "toy"
    assert saved["storage_refs"]["features"]["resid_last"]["store"] == "local"
    assert saved["storage_refs"]["features"]["resid_last"]["format"] == "residual_safetensors_v2"
    assert saved["storage_refs"]["features"]["resid_last"]["tensor_path"].endswith("features/feature_tensors.safetensors")


def test_remote_executor_resolves_deferred_dataset_in_runtime(tmp_path: Path) -> None:
    dataset = Dataset.from_source(
        source=InMemorySource.from_records(
            [
                {
                    "example_id": "a",
                    "prompt": "hello",
                    "class": "positive",
                    "case_id": "case_1",
                }
            ]
        ),
        defer=True,
        prompt_column="prompt",
        example_key_column="example_id",
        label_columns=["class"],
        case_key_column="case_id",
        name="remote_runtime_bound",
    )
    spec = CaptureSpec(
        engine=ToyEngine(),
        dataset=dataset,
        sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0], tokens=TokenSelector.last())],
    )

    manifest = execute_remote(
        runner_config={"kind": "modal", "resources": {"gpu": "L4"}},
        store_config={"kind": "local", "root": str(tmp_path / "artifacts")},
        spec_payload=spec.to_dict(),
    )

    assert manifest["example_coverage"]["example_count"] == 1
    assert manifest["example_coverage"]["dataset_name"] == "remote_runtime_bound"


def test_deferred_label_resolution_uses_label_only_source_fetch() -> None:
    dataset = Dataset.from_source(
        source=InMemorySource.from_records(
            [
                {"example_id": "a", "emotion": "happy"},
                {"example_id": "b", "emotion": "sad"},
            ]
        ),
        defer=True,
        prompt_column="missing_prompt",
        example_key_column="example_id",
        label_columns=["emotion"],
        name="label_only",
    )

    assert dataset.labels("emotion").resolve_values() == {"a": "happy", "b": "sad"}


def test_remote_executor_pushes_shard_into_deferred_postgres_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fetch_calls: list[dict[str, Any]] = []

    def fake_fetch(self: PostgresSource, **kwargs: Any) -> Dataset:
        fetch_calls.append(dict(kwargs))
        return Dataset.from_records(
            [
                {
                    "example_id": "ex_a",
                    "prompt": "hello",
                    "prompt_hash": "00000000000000000000000000000000",
                    "class": "positive",
                }
            ],
            prompt_column="prompt",
            example_key_column="example_id",
            prompt_hash_column="prompt_hash",
            label_columns=["class"],
            name=str(kwargs.get("name") or "postgres_shard"),
        )

    monkeypatch.setenv("XENON_DATABASE_URL", "postgresql://example/xenon")
    monkeypatch.setattr(PostgresSource, "fetch_dataset", fake_fetch)
    dataset = Dataset.from_postgres(
        source=PostgresSource.from_env("XENON_DATABASE_URL"),
        sql="SELECT example_id, prompt, prompt_hash, class FROM public.capture_examples",
        prompt_column="prompt",
        example_key_column="example_id",
        prompt_hash_column="prompt_hash",
        label_columns=["class"],
        name="remote_postgres_shard",
    )
    spec = CaptureSpec(
        engine=ToyEngine(),
        dataset=dataset,
        sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0], tokens=TokenSelector.last())],
    )

    manifest = execute_remote(
        runner_config={"kind": "modal", "resources": {"gpu": "L4", "shard_count": 2}},
        store_config={"kind": "local", "root": str(tmp_path / "artifacts")},
        spec_payload=spec.to_dict(),
        workflow_context={
            "run_id": "wr_postgres_shard",
            "workflow_step_key": "wf.capture",
            "execution_shard": {"index": 0, "count": 2},
        },
    )

    assert fetch_calls[0]["execution_shard"] == {"index": 0, "count": 2}
    assert manifest["example_coverage"]["example_keys"] == ["ex_a"]


def test_remote_executor_streams_deferred_postgres_capture_batches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    iter_calls: list[dict[str, Any]] = []

    def fail_fetch(self: PostgresSource, **kwargs: Any) -> Dataset:
        raise AssertionError("streaming capture should not fetch the full Postgres dataset")

    def fake_iter(self: PostgresSource, *, batch_size: int, **kwargs: Any) -> Any:
        iter_calls.append({"batch_size": batch_size, **dict(kwargs)})
        del self
        common = {
            "prompt_column": "prompt",
            "example_key_column": "example_id",
            "prompt_hash_column": "prompt_hash",
            "label_columns": ["class"],
        }
        yield Dataset.from_records(
            [
                {
                    "example_id": "ex_a",
                    "prompt": "hello",
                    "prompt_hash": "00000000000000000000000000000000",
                    "class": "positive",
                },
                {
                    "example_id": "ex_b",
                    "prompt": "world",
                    "prompt_hash": "00000000000000020000000000000000",
                    "class": "positive",
                },
            ],
            **common,
        )
        yield Dataset.from_records(
            [
                {
                    "example_id": "ex_c",
                    "prompt": "again",
                    "prompt_hash": "00000000000000040000000000000000",
                    "class": "positive",
                }
            ],
            **common,
        )

    monkeypatch.setenv("XENON_DATABASE_URL", "postgresql://example/xenon")
    monkeypatch.setattr(PostgresSource, "fetch_dataset", fail_fetch)
    monkeypatch.setattr(PostgresSource, "iter_dataset_batches", fake_iter)
    dataset = Dataset.from_postgres(
        source=PostgresSource.from_env("XENON_DATABASE_URL"),
        sql="SELECT example_id, prompt, prompt_hash, class FROM public.capture_examples",
        prompt_column="prompt",
        example_key_column="example_id",
        prompt_hash_column="prompt_hash",
        label_columns=["class"],
        name="remote_postgres_stream",
    )
    spec = CaptureSpec(
        engine=ToyEngine(),
        dataset=dataset,
        sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0], tokens=TokenSelector.last())],
    )

    (manifest,) = execute_remote_many(
        runner_config={"kind": "modal", "resources": {"gpu": "L4", "shard_count": 2}},
        store_config={"kind": "local", "root": str(tmp_path / "artifacts")},
        spec_payloads=[spec.to_dict()],
        workflow_contexts=[
            {
                "run_id": "wr_postgres_stream",
                "workflow_step_key": "wf.capture",
                "execution_shard": {"index": 0, "count": 2},
            }
        ],
    )

    assert iter_calls[0]["execution_shard"] == {"index": 0, "count": 2}
    assert manifest["metadata"]["streamed_deferred_dataset"] is True
    assert manifest["metadata"]["streamed_dataset_batches"] == 2
    assert manifest["example_coverage"]["example_count"] == 3
    assert manifest["example_coverage"]["example_keys"] == ["ex_a", "ex_b", "ex_c"]


def test_remote_executor_shards_generation_by_prompt_hash_and_merges(tmp_path: Path) -> None:
    dataset = Dataset.from_examples(
        [
            Example(key=f"ex_{idx}", prompt=f"prompt {idx}", labels={"class": "x"})
            for idx in range(6)
        ],
        name="generation_shard_dataset",
    )
    spec = GenerationRunSpec(
        engine=ToyEngine(),
        dataset=dataset,
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )
    store_config = {"kind": "local", "root": str(tmp_path / "artifacts")}
    runner_config = {"kind": "modal", "resources": {"gpu": "L4", "shard_count": 3}}
    shard_manifests = [
        execute_remote(
            runner_config=runner_config,
            store_config=store_config,
            spec_payload=spec.to_dict(),
            workflow_context={
                "run_id": "wr_sharded",
                "workflow_step_key": "wf.generate",
                "execution_shard": {"index": index, "count": 3},
            },
        )
        for index in range(3)
    ]

    merged = merge_remote_shards(
        runner_config=runner_config,
        store_config=store_config,
        spec_payload=spec.to_dict(),
        shard_manifests=shard_manifests,
        workflow_context={"run_id": "wr_sharded", "workflow_step_key": "wf.generate"},
    )

    assert merged["artifact_kind"] == "generation_run"
    payload_path = Path(merged["storage_refs"]["result"]["path"])
    with payload_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert [row["example_key"] for row in payload["rows"]] == [f"ex_{idx}" for idx in range(6)]
    assert payload["summary"]["sharded"] is True
    assert payload["summary"]["shard_count"] == 3


def test_remote_executor_reuses_completed_shard_artifact_on_resume(tmp_path: Path) -> None:
    dataset = Dataset.from_examples(
        [Example(key="ex_a", prompt="prompt a")],
        name="resume_shard_dataset",
    )
    spec = GenerationRunSpec(
        engine=ToyEngine(),
        dataset=dataset,
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )
    context = {
        "run_id": "wr_resume",
        "workflow_step_key": "wf.generate",
        "execution_shard": {"index": 0, "count": 2},
    }
    first = execute_remote(
        runner_config={"kind": "modal", "resources": {"gpu": "L4", "shard_count": 2}},
        store_config={"kind": "local", "root": str(tmp_path / "artifacts")},
        spec_payload=spec.to_dict(),
        workflow_context=context,
    )
    result_path = Path(first["storage_refs"]["result"]["path"])
    with result_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["rows"] = [{"example_key": "sentinel", "example": {"prompt_hash": "sentinel"}}]
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    second = execute_remote(
        runner_config={"kind": "modal", "resources": {"gpu": "L4", "shard_count": 2}},
        store_config={"kind": "local", "root": str(tmp_path / "artifacts")},
        spec_payload=spec.to_dict(),
        workflow_context=context,
    )

    assert second["artifact_id"] == first["artifact_id"]
    with result_path.open("r", encoding="utf-8") as f:
        assert json.load(f)["rows"][0]["example_key"] == "sentinel"

    branched = execute_remote(
        runner_config={"kind": "modal", "resources": {"gpu": "L4", "shard_count": 2}},
        store_config={"kind": "local", "root": str(tmp_path / "artifacts")},
        spec_payload=spec.to_dict(),
        workflow_context={**context, "run_id": "wr_rerun"},
    )

    assert branched["artifact_id"] != first["artifact_id"]


def test_remote_executor_reuses_generation_artifact_when_only_vllm_batch_size_changes(tmp_path: Path) -> None:
    dataset = Dataset.from_examples(
        [Example(key="ex_a", prompt="prompt a")],
        name="resume_vllm_batch_dataset",
    )
    base_engine = VLLMEngine(
        model_id="Qwen/Qwen3-30B-A3B",
        model_path_root="/models",
        max_model_len=55_000,
        max_num_seqs=4,
        enable_thinking=False,
    )
    larger_batch_engine = replace(base_engine, max_num_seqs=24)
    base_spec = GenerationRunSpec(
        engine=base_engine,
        dataset=dataset,
        generation=GenerationSpec(enabled=True, max_tokens=None),
    )
    larger_batch_spec = replace(base_spec, engine=larger_batch_engine)
    context = {
        "run_id": "wr_resume_vllm",
        "workflow_step_key": "wf.generate",
        "execution_shard": {"index": 0, "count": 2},
    }

    assert base_spec.spec_hash() != larger_batch_spec.spec_hash()
    assert base_spec.semantic_hash() == larger_batch_spec.semantic_hash()
    artifact_id = _artifact_id_for(spec=base_spec, workflow_context=context)
    assert artifact_id == _artifact_id_for(spec=larger_batch_spec, workflow_context=context)

    store = LocalArtifactStore(root=tmp_path / "artifacts")
    store.ensure_artifact_dir(artifact_id)
    result_ref = store.write_json(
        artifact_id,
        "result.json",
        {
            "kind": "generation_run_result",
            "summary": {"example_count": 1},
            "rows": [{"example_key": "sentinel", "example": {"prompt_hash": "sentinel"}}],
        },
    )
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_kind=base_spec.kind,
        schema_version=1,
        operation_spec_hash=base_spec.spec_hash(),
        operation_semantic_hash=base_spec.semantic_hash(),
        created_at=utc_now_iso(),
        engine=base_engine.identity(),
        runner={"kind": "modal", "resources": {"gpu": "H200", "shard_count": 2}},
        input_artifact_refs=(),
        example_coverage={"count": 1},
        storage_refs={"result": result_ref},
        metadata={"execution_shard": {"index": 0, "count": 2}},
        workflow_context=context,
    )
    store.write_json(artifact_id, "manifest.json", manifest.to_dict())

    reused = execute_remote(
        runner_config={"kind": "modal", "resources": {"gpu": "H200", "shard_count": 2}},
        store_config=store.identity(),
        spec_payload=larger_batch_spec.to_dict(),
        workflow_context=context,
    )

    assert reused["artifact_id"] == artifact_id
    assert reused["operation_spec_hash"] == base_spec.spec_hash()
    assert reused["operation_semantic_hash"] == larger_batch_spec.semantic_hash()


def test_remote_executor_resumes_generation_from_partial_result_rows(tmp_path: Path) -> None:
    examples = [
        Example(key="ex_a", prompt="prompt a"),
        Example(key="ex_b", prompt="prompt b"),
        Example(key="ex_c", prompt="prompt c"),
    ]
    dataset = Dataset.from_examples(examples, name="partial_generation_dataset")
    spec = GenerationRunSpec(
        engine=ToyEngine(),
        dataset=dataset,
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )
    context = {
        "run_id": "wr_partial_resume",
        "workflow_step_key": "wf.generate",
        "execution_shard": {"index": 0, "count": 2},
    }
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    artifact_id = _artifact_id_for(spec=spec, workflow_context=context)
    store.ensure_artifact_dir(artifact_id)
    store.write_json(
        artifact_id,
        "result.json",
        {
            "kind": "generation_run_result",
            "summary": {"example_count": 1, "partial": True},
            "rows": [
                {
                    "example_key": "ex_b",
                    "example": examples[1].to_dict(),
                    "generated_text": "already done",
                    "generated_token_ids": [1, 2],
                    "finish_reason": "stop",
                    "request_id": "partial",
                }
            ],
        },
    )

    manifest = execute_remote(
        runner_config={"kind": "modal", "resources": {"gpu": "L4", "shard_count": 2}},
        store_config=store.identity(),
        spec_payload=spec.to_dict(),
        workflow_context=context,
    )

    result_path = Path(manifest["storage_refs"]["result"]["path"])
    with result_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows_by_key = {str(row["example_key"]): row for row in payload["rows"]}

    assert manifest["artifact_id"] == artifact_id
    assert payload["summary"]["partial"] is False
    assert set(rows_by_key) == {"ex_a", "ex_b", "ex_c"}
    assert rows_by_key["ex_b"]["generated_text"] == "already done"
    assert rows_by_key["ex_a"]["generated_text"] == "toy_generation:ex_a"
    assert rows_by_key["ex_c"]["generated_text"] == "toy_generation:ex_c"


def test_vllm_generation_reports_rows_after_each_capture_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    import pipelines_v2.engine.vllm.generate as vllm_generate_module

    examples = [
        Example(key="ex_a", prompt="prompt a"),
        Example(key="ex_b", prompt="prompt b"),
    ]
    dataset = Dataset.from_examples(examples, name="vllm_generation_callback_dataset")
    spec = GenerationRunSpec(
        engine=VLLMEngine(model_id="Qwen/Qwen3-30B-A3B"),
        dataset=dataset,
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )

    def fake_run_vllm_capture(*, engine: Any, spec: CaptureSpec, batch_callback: Any = None) -> EngineCaptureResult:
        del engine
        generations: list[dict[str, Any]] = []
        for example in spec.dataset.examples:
            generation = {
                "example_key": example.key,
                "text": f"answer {example.key}",
                "generated_token_ids": [1],
                "finish_reason": "stop",
                "request_id": f"req-{example.key}",
            }
            generations.append(generation)
            if batch_callback is not None:
                batch_callback(
                    [example],
                    [generation],
                    [{"example_key": example.key, "generated_token_count": 1}],
                )
        return EngineCaptureResult(features={}, generations=generations, metadata={"backend": "fake_vllm"})

    monkeypatch.setattr(vllm_generate_module, "run_vllm_capture", fake_run_vllm_capture)
    checkpoints: list[list[dict[str, Any]]] = []

    result = vllm_generate_module.run_vllm_generation(
        engine=spec.engine,
        spec=spec,
        batch_callback=lambda rows, metadata: checkpoints.append(list(rows)),
    )

    assert [[row["example_key"] for row in rows] for rows in checkpoints] == [["ex_a"], ["ex_b"]]
    assert [row["generated_text"] for row in result.rows] == ["answer ex_a", "answer ex_b"]


def test_modal_volume_store_local_roundtrip(tmp_path: Path) -> None:
    store = ModalVolumeStore(name="xenon-data", root=str(tmp_path / "mounted"))
    artifact_id = "capture_123"

    store.make_artifact_dir(artifact_id)
    ref = store.write_json(artifact_id, "features/resid.json", {"ok": True})

    assert store.read_json_ref(ref) == {"ok": True}
    assert store.localize(artifact_id) == tmp_path / "mounted" / artifact_id


def test_modal_volume_store_preserves_mount_path_in_refs() -> None:
    store = ModalVolumeStore(name="xenon-data", root="/data/artifacts")

    path = store._resolve_inside_artifact("capture_123", "features/resid.json")

    assert str(path) == "/data/artifacts/capture_123/features/resid.json"


def test_modal_volume_store_retries_transient_volume_get_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines_v2.storage.modal as modal_store_module

    store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts",
        local_cache_root=tmp_path / "modal_cache",
    )
    results = iter(
        [
            subprocess.CompletedProcess(
                args=(),
                returncode=1,
                stdout="",
                stderr="StatusCode.DEADLINE_EXCEEDED: Deadline Exceeded",
            ),
            subprocess.CompletedProcess(args=(), returncode=0, stdout="", stderr=""),
        ]
    )
    sleeps: list[float] = []
    monkeypatch.setattr(modal_store_module.subprocess, "run", lambda *args, **kwargs: next(results))
    monkeypatch.setattr(modal_store_module.time, "sleep", sleeps.append)

    destination = tmp_path / "result.json"
    store._run_modal_volume_get("artifacts/capture_123/result.json", destination)

    assert sleeps == [1.0]


def test_modal_volume_store_does_not_retry_permanent_volume_get_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines_v2.storage.modal as modal_store_module

    store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts",
        local_cache_root=tmp_path / "modal_cache",
    )
    calls = 0

    def fail(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            args=(),
            returncode=1,
            stdout="",
            stderr="StatusCode.PERMISSION_DENIED",
        )

    monkeypatch.setattr(modal_store_module.subprocess, "run", fail)

    with pytest.raises(RuntimeError, match=r"after 1 attempt"):
        store._run_modal_volume_get(
            "artifacts/capture_123/result.json",
            tmp_path / "result.json",
        )

    assert calls == 1


def test_modal_volume_store_localize_downloads_into_parent_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts",
        local_cache_root=tmp_path / "modal_cache",
    )
    observed: dict[str, Path | str] = {}

    def fake_get(self: ModalVolumeStore, remote_path: str, destination: Path) -> None:
        observed["remote_path"] = remote_path
        observed["destination"] = destination
        (destination / "capture_123").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ModalVolumeStore, "_run_modal_volume_get", fake_get)

    localized = store.localize("capture_123")

    assert localized == tmp_path / "modal_cache" / "capture_123"
    assert observed == {
        "remote_path": "artifacts/capture_123",
        "destination": tmp_path / "modal_cache",
    }


def test_modal_runner_uses_generic_engine_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run_on_modal(
        *,
        runner_config: dict[str, object],
        store_config: dict[str, object],
        spec_payload: dict[str, object],
    ) -> dict[str, object]:
        observed["runner_config"] = runner_config
        observed["store_config"] = store_config
        observed["spec_payload"] = spec_payload
        return {
            "artifact_id": "capture_test",
            "artifact_kind": "capture",
            "schema_version": 1,
            "operation_spec_hash": "abc123",
            "created_at": "2026-04-13T00:00:00+00:00",
            "engine": spec_payload["engine"],
            "runner": runner_config,
            "input_artifact_refs": [],
            "example_coverage": make_toy_dataset().coverage(),
            "storage_refs": {"features": {}},
            "metadata": {},
        }

    monkeypatch.setattr("pipelines_v2.runtime.modal.run_on_modal", fake_run_on_modal)
    runner = ModalRunner(
        resources=ModalResources(gpu="L4"),
        artifacts=ModalVolumeStore(name="xenon-data", root=str(tmp_path / "mounted")),
    )

    artifact = runner.run(make_toy_capture_spec())

    assert artifact.id == "capture_test"
    assert observed["spec_payload"]["engine"] == ToyEngine().identity()
    assert observed["store_config"] == {
        "kind": "modal_volume",
        "name": "xenon-data",
        "root": str(tmp_path / "mounted"),
        "transfer_policy": {
            "allow_large_transfer": False,
            "max_download_bytes": 64 * 1024 * 1024,
        },
    }


def test_modal_runner_forwards_workspace_root_without_serializing_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_run_on_modal(
        *,
        runner_config: dict[str, object],
        store_config: dict[str, object],
        spec_payload: dict[str, object],
        workspace_root: Path | None = None,
    ) -> dict[str, object]:
        del store_config
        observed["workspace_root"] = workspace_root
        return {
            "artifact_id": "capture_workspace_root",
            "artifact_kind": "capture",
            "schema_version": 1,
            "operation_spec_hash": "abc123",
            "operation_semantic_hash": "abc123",
            "created_at": "2026-07-23T00:00:00+00:00",
            "engine": spec_payload["engine"],
            "runner": runner_config,
            "input_artifact_refs": [],
            "example_coverage": make_toy_dataset().coverage(),
            "storage_refs": {"features": {}},
            "metadata": {},
        }

    monkeypatch.setattr("pipelines_v2.runtime.modal.run_on_modal", fake_run_on_modal)
    workspace = tmp_path / "selected-project"
    workspace.mkdir()
    runner = ModalRunner(
        resources=ModalResources(gpu="L4"),
        artifacts=ModalVolumeStore(name="xenon-data", root=str(tmp_path / "mounted")),
        workspace_root=workspace,
    )

    artifact = runner.run(make_toy_capture_spec())

    assert artifact.id == "capture_workspace_root"
    assert observed["workspace_root"] == workspace
    assert "workspace_root" not in runner.identity()


def test_modal_runner_batch_forwards_workspace_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_run_many_on_modal(
        *,
        runner_config: dict[str, object],
        store_config: dict[str, object],
        spec_payloads: list[dict[str, object]],
        workflow_contexts: list[dict[str, object] | None],
        workspace_root: Path | None = None,
        progress_callback: Any | None = None,
    ) -> list[dict[str, object]]:
        del store_config, workflow_contexts, progress_callback
        observed["workspace_root"] = workspace_root
        return [
            {
                "artifact_id": "capture_batch_workspace_root",
                "artifact_kind": "capture",
                "schema_version": 1,
                "operation_spec_hash": "abc123",
                "operation_semantic_hash": "abc123",
                "created_at": "2026-07-23T00:00:00+00:00",
                "engine": spec_payloads[0]["engine"],
                "runner": runner_config,
                "input_artifact_refs": [],
                "example_coverage": make_toy_dataset().coverage(),
                "storage_refs": {"features": {}},
                "metadata": {},
            }
        ]

    monkeypatch.setattr(
        "pipelines_v2.runtime.modal.run_many_on_modal",
        fake_run_many_on_modal,
    )
    workspace = tmp_path / "selected-project"
    workspace.mkdir()
    runner = ModalRunner(
        resources=ModalResources(gpu="L4"),
        artifacts=ModalVolumeStore(name="xenon-data", root=str(tmp_path / "mounted")),
        workspace_root=workspace,
    )

    artifacts = runner.run_many([make_toy_capture_spec()])

    assert [artifact.id for artifact in artifacts] == ["capture_batch_workspace_root"]
    assert observed["workspace_root"] == workspace


def test_modal_runner_serializes_additional_volume_mounts(tmp_path: Path) -> None:
    runner = ModalRunner(
        resources=ModalResources(
            gpu="L4",
            volumes=(
                ModalVolumeMount(name="xenon-models", mount_path="/models"),
                ModalVolumeMount(
                    name="xenon-scratch",
                    mount_path="/scratch",
                    create_if_missing=True,
                    commit_on_success=True,
                ),
            ),
        ),
        artifacts=ModalVolumeStore(name="xenon-data", root=str(tmp_path / "mounted")),
    )

    identity = runner.identity()

    assert identity["resources"]["volumes"] == [
        {
            "name": "xenon-models",
            "mount_path": "/models",
            "create_if_missing": False,
            "commit_on_success": False,
        },
        {
            "name": "xenon-scratch",
            "mount_path": "/scratch",
            "create_if_missing": True,
            "commit_on_success": True,
        },
    ]


def test_modal_runner_serializes_secret_bindings(tmp_path: Path) -> None:
    runner = ModalRunner(
        resources=ModalResources(
            gpu="L4",
            secrets=(
                ModalSecret.from_env_var("XENON_DATABASE_URL", secret_name="xenon-db"),
            ),
        ),
        artifacts=ModalVolumeStore(name="xenon-data", root=str(tmp_path / "mounted")),
    )

    identity = runner.identity()

    assert identity["resources"]["secrets"] == [
        {
            "name": "xenon-db",
            "env_vars": ["XENON_DATABASE_URL"],
        }
    ]


def test_modal_runner_serializes_runtime_env(tmp_path: Path) -> None:
    runner = ModalRunner(
        resources=ModalResources(
            gpu="L4",
            env={"VLLM_CACHE_ROOT": "/cache/vllm"},
        ),
        artifacts=ModalVolumeStore(name="xenon-data", root=str(tmp_path / "mounted")),
    )

    identity = runner.identity()

    assert identity["resources"]["env"] == {"VLLM_CACHE_ROOT": "/cache/vllm"}


def test_modal_runner_serializes_cpu_analysis_resources(tmp_path: Path) -> None:
    runner = ModalRunner(
        resources=ModalResources(
            cpu=6,
            memory_mb=24 * 1024,
            timeout_seconds=1800,
            max_containers=1,
            shard_count=3,
        ),
        artifacts=ModalVolumeStore(name="xenon-data", root=str(tmp_path / "mounted")),
    )

    identity = runner.identity()

    assert identity["resources"]["gpu"] is None
    assert identity["resources"]["cpu"] == 6
    assert identity["resources"]["memory_mb"] == 24 * 1024
    assert identity["resources"]["timeout_seconds"] == 1800
    assert identity["resources"]["max_containers"] == 1
    assert identity["resources"]["shard_count"] == 3
    assert ModalResources.from_dict(identity["resources"]).max_containers == 1
    assert ModalResources.from_dict(identity["resources"]).shard_count == 3


def test_modal_runner_rejects_missing_runtime_secret_bindings(tmp_path: Path) -> None:
    dataset = Dataset.from_postgres(
        source=PostgresSource.from_env("XENON_DATABASE_URL"),
        table="public.capture_examples",
        prompt_column="prompt",
        example_key_column="example_id",
    )
    runner = ModalRunner(
        resources=ModalResources(gpu="L4"),
        artifacts=ModalVolumeStore(name="xenon-data", root=str(tmp_path / "mounted")),
    )

    with pytest.raises(SpecValidationError, match="XENON_DATABASE_URL"):
        runner.run(
            CaptureSpec(
                engine=ToyEngine(),
                dataset=dataset,
                sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0])],
            )
        )


def test_mounted_volumes_include_artifact_store_and_extra_volumes() -> None:
    mounted = _mounted_volumes(
        store_config={"kind": "modal_volume", "name": "xenon-data", "root": "/data/artifacts"},
        resources={
            "gpu": "L4",
            "volumes": [
                {"name": "xenon-models", "mount_path": "/models"},
            ],
        },
    )

    assert [(volume.name, volume.mount_path, volume.commit_on_success) for volume in mounted] == [
        ("xenon-data", "/data", True),
        ("xenon-models", "/models", False),
    ]


def test_mounted_volumes_merge_duplicate_mount_for_same_volume() -> None:
    mounted = _mounted_volumes(
        store_config={"kind": "modal_volume", "name": "xenon-data", "root": "/data/artifacts"},
        resources={
            "gpu": "L4",
            "volumes": [
                {"name": "xenon-data", "mount_path": "/data", "commit_on_success": False},
            ],
        },
    )

    assert [(volume.name, volume.mount_path, volume.create_if_missing, volume.commit_on_success) for volume in mounted] == [
        ("xenon-data", "/data", True, True),
    ]


def test_mounted_volumes_reject_duplicate_mount_paths_for_different_volumes() -> None:
    with pytest.raises(ValueError, match="different volumes"):
        _mounted_volumes(
            store_config={"kind": "modal_volume", "name": "xenon-data", "root": "/data/artifacts"},
            resources={
                "gpu": "L4",
                "volumes": [
                    {"name": "xenon-models", "mount_path": "/data"},
                ],
            },
        )


def test_modal_worker_resolves_local_sources_against_explicit_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "selected-project"
    workflows = workspace / "workflows"
    workflows.mkdir(parents=True)
    fallback = tmp_path / "xenon-checkout"
    fallback.mkdir()
    (fallback / "pyproject.toml").touch()
    monkeypatch.chdir(fallback)

    mounts, pythonpath = _resolved_local_python_sources(
        ("workflows",),
        workspace_root=workspace,
    )

    assert mounts == ((workflows, "/root/pipelines_v2_workspace/workflows"),)
    assert pythonpath == ("/root/pipelines_v2_workspace",)


def test_modal_worker_resolves_project_and_library_sources_from_distinct_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "selected-project"
    workflows = workspace / "workflows"
    workflows.mkdir(parents=True)
    library = tmp_path / "xenon-checkout"
    pipelines = library / "pipelines_v2"
    pipelines.mkdir(parents=True)
    monkeypatch.setattr(
        "pipelines_v2.runtime.modal_worker.find_workspace_root",
        lambda start=None: library,
    )

    mounts, pythonpath = _resolved_local_python_sources(
        ("pipelines_v2", "workflows"),
        workspace_root=workspace,
    )

    assert mounts == (
        (pipelines, "/root/pipelines_v2_workspace/pipelines_v2"),
        (workflows, "/root/pipelines_v2_workspace/workflows"),
    )
    assert pythonpath == ("/root/pipelines_v2_workspace",)


def test_modal_worker_reports_all_roots_for_missing_local_source(tmp_path: Path) -> None:
    workspace = tmp_path / "selected-project"
    workspace.mkdir()

    with pytest.raises(FileNotFoundError, match="tried:"):
        _resolved_local_python_sources(("missing_package",), workspace_root=workspace)


def test_modal_worker_rejects_local_source_outside_explicit_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "selected-project"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError):
        _resolved_local_python_sources((str(outside),), workspace_root=workspace)


def test_modal_worker_merges_spec_runtime_secrets_into_runtime_spec() -> None:
    dataset = Dataset.from_postgres(
        source=PostgresSource.from_env("XENON_DATABASE_URL"),
        table="public.capture_examples",
        prompt_column="prompt",
        example_key_column="example_id",
    )
    spec = CaptureSpec(
        engine=ToyEngine(),
        dataset=dataset,
        sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0])],
    )

    runtime_spec = _resolved_runtime_spec(
        spec_payload=spec.to_dict(),
    )

    assert [secret.env_var for secret in runtime_spec.secrets] == ["XENON_DATABASE_URL"]


def test_modal_worker_threads_runtime_env_to_function_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    progress_events: list[dict[str, object]] = []

    class FakeImage:
        def pip_install(self, *packages: str) -> "FakeImage":
            captured["pip_packages"] = packages
            return self

        def env(self, env_payload: dict[str, str]) -> "FakeImage":
            captured["image_env"] = dict(env_payload)
            return self

        def add_local_dir(self, local_path: str, *, remote_path: str) -> "FakeImage":
            mounts = list(captured.get("image_mounts", []))
            mounts.append((local_path, remote_path))
            captured["image_mounts"] = mounts
            return self

    class FakeImageFactory:
        @staticmethod
        def debian_slim(*, python_version: str) -> FakeImage:
            captured["python_version"] = python_version
            return FakeImage()

    class FakeVolume:
        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> object:
            return {"name": name, "create_if_missing": create_if_missing}

    class FakeSecret:
        @staticmethod
        def from_name(name: str) -> object:
            return {"name": name}

    class FakeFunction:
        def __init__(self, fn: object) -> None:
            self._fn = fn

        def remote(self, *args: object) -> object:
            return self._fn(*args)

        def remote_gen(self, *args: object):
            yield from self._fn(*args)

    class FakeAppRun:
        app_id = "ap-test-env"

        def __enter__(self) -> "FakeAppRun":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class FakeApp:
        def __init__(self, name: str) -> None:
            captured["app_name"] = name

        def function(self, **kwargs: object):
            captured["function_kwargs"] = dict(kwargs)

            def decorator(fn):
                return FakeFunction(fn)

            return decorator

        def run(self) -> FakeAppRun:
            return FakeAppRun()

    fake_modal = types.SimpleNamespace(
        App=FakeApp,
        Image=FakeImageFactory,
        Volume=FakeVolume,
        Secret=FakeSecret,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(
        "pipelines_v2.runtime.remote_executor.execute_remote",
        lambda **kwargs: {
            "artifact_id": "capture_test",
            "artifact_kind": "capture",
            "schema_version": 1,
            "operation_spec_hash": "abc123",
            "operation_semantic_hash": "abc123",
            "created_at": "2026-04-17T00:00:00+00:00",
            "engine": kwargs["spec_payload"]["engine"],
            "runner": {},
            "input_artifact_refs": [],
            "example_coverage": make_toy_dataset().coverage(),
            "storage_refs": {"features": {}},
            "metadata": {},
        },
    )
    monkeypatch.setenv("XENON_ACTIVATION_PATCH_DEBUG", "project_out_gate")

    spec = replace(
        make_toy_capture_spec(),
        engine=VLLMEngine(
            model_id="fake/model",
            enforce_eager=False,
            enable_prefix_caching=False,
        ),
    )
    result = run_on_modal(
        runner_config={
            "kind": "modal",
            "resources": {"max_containers": 1},
        },
        store_config={
            "kind": "modal_volume",
            "name": "xenon-data",
            "root": str(tmp_path / "artifacts"),
        },
        spec_payload=spec.to_dict(),
        workflow_context={"step_name": "capture_prompt_generated_residual"},
        progress_callback=lambda payload: progress_events.append(dict(payload)),
    )

    function_kwargs = dict(captured["function_kwargs"])
    assert captured["app_name"] == "xenon-capture-prompt-generated-residual"
    assert function_kwargs["max_containers"] == 1
    assert function_kwargs["env"]["XENON_ACTIVATION_PATCH_DEBUG"] == "project_out_gate"
    assert function_kwargs["env"]["VLLM_COMPILE_CACHE_SAVE_FORMAT"] == "binary"
    assert function_kwargs["env"]["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert function_kwargs["env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert captured["image_env"]["XENON_ACTIVATION_PATCH_DEBUG"] == "project_out_gate"
    assert captured["image_env"]["VLLM_COMPILE_CACHE_SAVE_FORMAT"] == "binary"
    assert captured["image_env"]["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert captured["image_env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert result["runner"]["runtime_app_id"] == "ap-test-env"
    assert [event["stage"] for event in progress_events] == [
        "modal_launching",
        "modal_app_started",
        "remote_execution_finished",
    ]
    assert progress_events[0]["metrics"]["source_mount_count"] >= 0


def test_modal_worker_shards_model_bound_specs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    shard_contexts: list[dict[str, object]] = []
    merged_inputs: list[dict[str, object]] = []
    progress_events: list[dict[str, object]] = []

    class FakeImage:
        def pip_install(self, *packages: str) -> "FakeImage":
            del packages
            return self

        def env(self, env_payload: dict[str, str]) -> "FakeImage":
            del env_payload
            return self

        def add_local_dir(self, local_path: str, *, remote_path: str) -> "FakeImage":
            del local_path, remote_path
            return self

    class FakeImageFactory:
        @staticmethod
        def debian_slim(*, python_version: str) -> FakeImage:
            del python_version
            return FakeImage()

    class FakeVolume:
        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> object:
            return {"name": name, "create_if_missing": create_if_missing}

    class FakeSecret:
        @staticmethod
        def from_name(name: str) -> object:
            return {"name": name}

    class FakeFunction:
        def __init__(self, fn: object) -> None:
            self._fn = fn

        def remote(self, *args: object) -> object:
            return self._fn(*args)

        def remote_gen(self, *args: object):
            yield from self._fn(*args)

    class FakeAppRun:
        app_id = "ap-test-shards"

        def __enter__(self) -> "FakeAppRun":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class FakeApp:
        def __init__(self, name: str) -> None:
            captured["app_name"] = name

        def function(self, **kwargs: object):
            captured["function_kwargs"] = dict(kwargs)

            def decorator(fn):
                return FakeFunction(fn)

            return decorator

        def run(self) -> FakeAppRun:
            return FakeAppRun()

    def fake_execute_remote(**kwargs: object) -> dict[str, object]:
        context = dict(kwargs["workflow_context"] or {})
        shard_contexts.append(context)
        shard = dict(context["execution_shard"])
        return {
            "artifact_id": f"capture_shard_{shard['index']}",
            "artifact_kind": "capture",
            "schema_version": 1,
            "operation_spec_hash": "abc123",
            "operation_semantic_hash": "abc123",
            "created_at": "2026-04-17T00:00:00+00:00",
            "engine": kwargs["spec_payload"]["engine"],
            "runner": {},
            "input_artifact_refs": [],
            "example_coverage": {"count": 1},
            "storage_refs": {"features": {}},
            "metadata": {"execution_shard": shard},
        }

    def fake_merge_remote_shards(**kwargs: object) -> dict[str, object]:
        merged_inputs.extend(dict(item) for item in kwargs["shard_manifests"])
        return {
            "artifact_id": "capture_merged",
            "artifact_kind": "capture",
            "schema_version": 1,
            "operation_spec_hash": "abc123",
            "operation_semantic_hash": "abc123",
            "created_at": "2026-04-17T00:00:00+00:00",
            "engine": kwargs["spec_payload"]["engine"],
            "runner": {},
            "input_artifact_refs": [],
            "example_coverage": make_toy_dataset().coverage(),
            "storage_refs": {"features": {}},
            "metadata": {"sharded": True},
        }

    fake_modal = types.SimpleNamespace(
        App=FakeApp,
        Image=FakeImageFactory,
        Volume=FakeVolume,
        Secret=FakeSecret,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr("pipelines_v2.runtime.remote_executor.execute_remote", fake_execute_remote)
    monkeypatch.setattr("pipelines_v2.runtime.remote_executor.merge_remote_shards", fake_merge_remote_shards)

    result = run_on_modal(
        runner_config={
            "kind": "modal",
            "resources": {"max_containers": 3, "shard_count": 3},
        },
        store_config={
            "kind": "modal_volume",
            "name": "xenon-data",
            "root": str(tmp_path / "artifacts"),
        },
        spec_payload=make_toy_capture_spec().to_dict(),
        workflow_context={
            "run_id": "wr_test",
            "workflow_step_key": "wf.capture_prompt_generated_residual",
            "step_name": "capture_prompt_generated_residual",
        },
        progress_callback=lambda payload: progress_events.append(dict(payload)),
    )

    assert result["artifact_id"] == "capture_merged"
    assert result["runner"]["runtime_app_id"] == "ap-test-shards"
    assert captured["app_name"] == "xenon-capture-prompt-generated-residual"
    assert sorted(dict(context["execution_shard"])["index"] for context in shard_contexts) == [0, 1, 2]
    assert len(merged_inputs) == 3
    assert "remote_shards_submitted" in [event["stage"] for event in progress_events]
    assert "remote_shards_finished" in [event["stage"] for event in progress_events]


def test_modal_worker_ignores_shard_count_for_unshardable_specs() -> None:
    assert (
        _modal_shard_count(
            resources={"shard_count": 3},
            spec_payload={"kind": "subspace"},
        )
        == 1
    )
    assert (
        _modal_shard_count(
            resources={"shard_count": 3},
            spec_payload=make_toy_capture_spec().to_dict(),
        )
        == 3
    )


def test_modal_worker_shards_batched_model_bound_specs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    execute_many_calls: list[list[dict[str, object]]] = []
    merge_calls: list[dict[str, object]] = []
    progress_events: list[dict[str, object]] = []

    class FakeImage:
        def pip_install(self, *packages: str) -> "FakeImage":
            del packages
            return self

        def env(self, env_payload: dict[str, str]) -> "FakeImage":
            del env_payload
            return self

        def add_local_dir(self, local_path: str, *, remote_path: str) -> "FakeImage":
            del local_path, remote_path
            return self

    class FakeImageFactory:
        @staticmethod
        def debian_slim(*, python_version: str) -> FakeImage:
            del python_version
            return FakeImage()

    class FakeVolume:
        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> object:
            return {"name": name, "create_if_missing": create_if_missing}

    class FakeSecret:
        @staticmethod
        def from_name(name: str) -> object:
            return {"name": name}

    class FakeFunction:
        def __init__(self, fn: object) -> None:
            self._fn = fn

        def remote(self, *args: object) -> object:
            return self._fn(*args)

        def remote_gen(self, *args: object):
            yield from self._fn(*args)

    class FakeAppRun:
        app_id = "ap-test-batch-shards"

        def __enter__(self) -> "FakeAppRun":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class FakeApp:
        def __init__(self, name: str) -> None:
            captured["app_name"] = name

        def function(self, **kwargs: object):
            captured.setdefault("function_kwargs", []).append(dict(kwargs))

            def decorator(fn):
                return FakeFunction(fn)

            return decorator

        def run(self) -> FakeAppRun:
            return FakeAppRun()

    def fake_execute_remote_many(**kwargs: object) -> list[dict[str, object]]:
        contexts = [dict(context or {}) for context in kwargs["workflow_contexts"]]
        execute_many_calls.append(contexts)
        shard = dict(contexts[0]["execution_shard"])
        return [
            {
                "artifact_id": f"{payload['kind']}_shard_{shard['index']}",
                "artifact_kind": payload["kind"],
                "schema_version": 1,
                "operation_spec_hash": "abc123",
                "operation_semantic_hash": "abc123",
                "created_at": "2026-04-17T00:00:00+00:00",
                "engine": payload.get("engine", {}),
                "runner": {},
                "input_artifact_refs": [],
                "example_coverage": {"count": 1},
                "storage_refs": {"result": {"store": "fake", "path": "unused.json"}},
                "metadata": {"execution_shard": shard},
            }
            for payload in kwargs["spec_payloads"]
        ]

    def fake_merge_remote_shards(**kwargs: object) -> dict[str, object]:
        payload = dict(kwargs["spec_payload"])
        shard_manifests = [dict(item) for item in kwargs["shard_manifests"]]
        merge_calls.append(
            {
                "kind": payload["kind"],
                "shard_ids": [item["artifact_id"] for item in shard_manifests],
            }
        )
        return {
            "artifact_id": f"{payload['kind']}_merged",
            "artifact_kind": payload["kind"],
            "schema_version": 1,
            "operation_spec_hash": "abc123",
            "operation_semantic_hash": "abc123",
            "created_at": "2026-04-17T00:00:00+00:00",
            "engine": payload.get("engine", {}),
            "runner": {},
            "input_artifact_refs": [],
            "example_coverage": make_toy_dataset().coverage(),
            "storage_refs": {"result": {"store": "fake", "path": "unused.json"}},
            "metadata": {"sharded": True},
        }

    fake_modal = types.SimpleNamespace(
        App=FakeApp,
        Image=FakeImageFactory,
        Volume=FakeVolume,
        Secret=FakeSecret,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr("pipelines_v2.runtime.remote_executor.execute_remote_many", fake_execute_remote_many)
    monkeypatch.setattr("pipelines_v2.runtime.remote_executor.merge_remote_shards", fake_merge_remote_shards)

    dataset = make_toy_dataset()
    subspace = InlineOperationArtifact(
        payload={
            "kind": "subspace_result",
            "layers": {
                "0": {
                    "mean": [0.0, 0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0, 1.0],
                    "safe_scale": [1.0, 1.0, 1.0, 1.0],
                    "components": [[1.0, 0.0, 0.0, 0.0]],
                    "component_count": 1,
                    "named_components": {},
                }
            },
        },
        artifact_kind="subspace",
    )
    engine = ToyEngine()
    specs = [
        make_toy_capture_spec(dataset),
        GenerationRunSpec(
            engine=engine,
            dataset=dataset,
            generation=GenerationSpec(enabled=True, max_tokens=1),
        ),
        PatchedGenerationSpec(
            engine=engine,
            dataset=dataset,
            patch=ProjectOutPatch(
                subspace=subspace,
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.last(),
                component_indices_by_layer={0: (0,)},
            ),
            select_when=dataset.labels("class").equals("positive"),
            generation=GenerationSpec(enabled=True, max_tokens=1),
        ),
    ]

    results = run_many_on_modal(
        runner_config={
            "kind": "modal",
            "resources": {"max_containers": 2, "shard_count": 2},
        },
        store_config={
            "kind": "modal_volume",
            "name": "xenon-data",
            "root": str(tmp_path / "artifacts"),
        },
        spec_payloads=[spec.to_dict() for spec in specs],
        workflow_contexts=[
            {"run_id": "wr_test", "workflow_step_key": f"wf.step_{index}", "step_name": f"step_{index}"}
            for index in range(len(specs))
        ],
        progress_callback=lambda payload: progress_events.append(dict(payload)),
    )

    assert [result["artifact_id"] for result in results] == [
        "capture_merged",
        "generation_run_merged",
        "patched_generation_merged",
    ]
    assert [result["runner"]["runtime_app_id"] for result in results] == [
        "ap-test-batch-shards",
        "ap-test-batch-shards",
        "ap-test-batch-shards",
    ]
    assert len(execute_many_calls) == 2
    assert sorted(
        [
            sorted({dict(context["execution_shard"])["index"] for context in contexts})
            for contexts in execute_many_calls
        ]
    ) == [[0], [1]]
    assert [call["kind"] for call in merge_calls] == ["capture", "generation_run", "patched_generation"]
    assert all(len(call["shard_ids"]) == 2 for call in merge_calls)
    assert "remote_batch_shards_submitted" in [event["stage"] for event in progress_events]
    assert "remote_batch_shards_finished" in [event["stage"] for event in progress_events]


def test_modal_worker_runner_env_overrides_spec_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeImage:
        def pip_install(self, *packages: str) -> "FakeImage":
            del packages
            return self

        def env(self, env_payload: dict[str, str]) -> "FakeImage":
            captured["image_env"] = dict(env_payload)
            return self

        def add_local_dir(self, local_path: str, *, remote_path: str) -> "FakeImage":
            del local_path, remote_path
            return self

    class FakeImageFactory:
        @staticmethod
        def debian_slim(*, python_version: str) -> FakeImage:
            del python_version
            return FakeImage()

    class FakeVolume:
        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> object:
            return {"name": name, "create_if_missing": create_if_missing}

    class FakeSecret:
        @staticmethod
        def from_name(name: str) -> object:
            return {"name": name}

    class FakeFunction:
        def __init__(self, fn: object) -> None:
            self._fn = fn

        def remote(self, *args: object) -> object:
            return self._fn(*args)

        def remote_gen(self, *args: object):
            yield from self._fn(*args)

    class FakeAppRun:
        app_id = "ap-test-env-override"

        def __enter__(self) -> "FakeAppRun":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class FakeApp:
        def __init__(self, name: str) -> None:
            del name

        def function(self, **kwargs: object):
            captured["function_kwargs"] = dict(kwargs)

            def decorator(fn):
                return FakeFunction(fn)

            return decorator

        def run(self) -> FakeAppRun:
            return FakeAppRun()

    fake_modal = types.SimpleNamespace(
        App=FakeApp,
        Image=FakeImageFactory,
        Volume=FakeVolume,
        Secret=FakeSecret,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(
        "pipelines_v2.runtime.remote_executor.execute_remote",
        lambda **kwargs: {
            "artifact_id": "capture_test",
            "artifact_kind": "capture",
            "schema_version": 1,
            "operation_spec_hash": "abc123",
            "operation_semantic_hash": "abc123",
            "created_at": "2026-04-17T00:00:00+00:00",
            "engine": kwargs["spec_payload"]["engine"],
            "runner": {},
            "input_artifact_refs": [],
            "example_coverage": make_toy_dataset().coverage(),
            "storage_refs": {"features": {}},
            "metadata": {},
        },
    )
    monkeypatch.setenv("XENON_ACTIVATION_PATCH_DEBUG", "spec_debug")

    spec = replace(
        make_toy_capture_spec(),
        engine=VLLMEngine(
            model_id="fake/model",
            enforce_eager=False,
            enable_prefix_caching=False,
        ),
    )
    run_on_modal(
        runner_config={
            "kind": "modal",
            "resources": {
                "env": {
                    "XENON_ACTIVATION_PATCH_DEBUG": "runner_debug",
                    "VLLM_CACHE_ROOT": "/cache/vllm",
                }
            },
        },
        store_config={
            "kind": "modal_volume",
            "name": "xenon-data",
            "root": str(tmp_path / "artifacts"),
        },
        spec_payload=spec.to_dict(),
        workflow_context=None,
    )

    function_kwargs = dict(captured["function_kwargs"])
    assert function_kwargs["env"]["XENON_ACTIVATION_PATCH_DEBUG"] == "runner_debug"
    assert function_kwargs["env"]["VLLM_CACHE_ROOT"] == "/cache/vllm"
    assert captured["image_env"]["XENON_ACTIVATION_PATCH_DEBUG"] == "runner_debug"
    assert captured["image_env"]["VLLM_CACHE_ROOT"] == "/cache/vllm"


def test_modal_runner_raises_clear_error_on_cancelled_remote_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("pipelines_v2.runtime.modal.run_on_modal", lambda **_: None)
    runner = ModalRunner(
        resources=ModalResources(gpu="L4"),
        artifacts=ModalVolumeStore(name="xenon-data", root=str(tmp_path / "mounted")),
    )

    with pytest.raises(RuntimeError, match="manifest payload"):
        runner.run(make_toy_capture_spec())


def test_workflow_orchestrator_records_runtime_app_id_for_completed_remote_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_on_modal(
        *,
        runner_config: dict[str, object],
        store_config: dict[str, object],
        spec_payload: dict[str, object],
        workflow_context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "artifact_id": "capture_test",
            "artifact_kind": "capture",
            "schema_version": 1,
            "operation_spec_hash": "abc123",
            "operation_semantic_hash": "abc123",
            "created_at": "2026-04-14T00:00:00+00:00",
            "engine": spec_payload["engine"],
            "runner": {
                **runner_config,
                "runtime_app_id": "ap-runtime-test",
            },
            "input_artifact_refs": [],
            "example_coverage": make_toy_dataset().coverage(),
            "storage_refs": {"features": {}},
            "metadata": {},
            "workflow_context": dict(workflow_context or {}),
        }

    monkeypatch.setattr("pipelines_v2.runtime.modal.run_on_modal", fake_run_on_modal)
    catalog = FileCatalog(tmp_path / "catalog")
    runner = ModalRunner(
        resources=ModalResources(gpu="L4"),
        artifacts=ModalVolumeStore(name="xenon-data", root=str(tmp_path / "mounted")),
        catalog=catalog,
    )
    orchestrator = WorkflowOrchestrator(runners={"capture_gpu": runner})
    workflow = WorkflowSpec(
        name="modal_runtime_id_success",
        steps=(
            WorkflowStep(
                name="capture",
                runner="capture_gpu",
                spec=make_toy_capture_spec(),
            ),
        ),
    )

    result = orchestrator.run(workflow)

    step_records = {record.step_name: record for record in catalog.list_workflow_steps(result.run_id or "")}
    assert step_records["capture"].status == "completed"
    assert step_records["capture"].runtime_app_id == "ap-runtime-test"
    assert result.step("capture").manifest().runner["runtime_app_id"] == "ap-runtime-test"


def test_workflow_orchestrator_persists_progress_snapshot_for_remote_step_submission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    progress_store = FileWorkflowProgressStore(tmp_path / "catalog")

    def fake_run_on_modal(
        *,
        runner_config: dict[str, object],
        store_config: dict[str, object],
        spec_payload: dict[str, object],
        workflow_context: dict[str, object] | None = None,
        progress_callback: Any | None = None,
    ) -> dict[str, object]:
        if progress_callback is not None:
            progress_callback(
                {
                    "status": "running",
                    "stage": "modal_app_started",
                    "runtime_kind": "modal",
                    "runtime_app_id": "ap-progress-test",
                    "message": "Modal app started",
                }
            )
        return {
            "artifact_id": "capture_test_progress",
            "artifact_kind": "capture",
            "schema_version": 1,
            "operation_spec_hash": "abc123",
            "operation_semantic_hash": "abc123",
            "created_at": "2026-04-17T00:00:00+00:00",
            "engine": spec_payload["engine"],
            "runner": {
                **runner_config,
                "runtime_app_id": "ap-progress-test",
            },
            "input_artifact_refs": [],
            "example_coverage": make_toy_dataset().coverage(),
            "storage_refs": {"features": {}},
            "metadata": {},
            "workflow_context": dict(workflow_context or {}),
        }

    monkeypatch.setattr("pipelines_v2.runtime.modal.run_on_modal", fake_run_on_modal)
    catalog = FileCatalog(tmp_path / "catalog")
    runner = ModalRunner(
        resources=ModalResources(gpu="L4"),
        artifacts=ModalVolumeStore(name="xenon-data", root=str(tmp_path / "mounted")),
        catalog=catalog,
    )
    orchestrator = WorkflowOrchestrator(
        runners={"capture_gpu": runner},
        progress_sink=WorkflowProgressSink(store=progress_store),
    )
    workflow = WorkflowSpec(
        name="modal_runtime_progress",
        steps=(
            WorkflowStep(
                name="capture",
                runner="capture_gpu",
                spec=make_toy_capture_spec(),
            ),
        ),
    )

    result = orchestrator.run(workflow)

    snapshot = progress_store.load_step_snapshots(result.run_id or "")["capture"]
    assert snapshot["status"] == "completed"
    assert snapshot["runtime_app_id"] == "ap-progress-test"


def test_workflow_orchestrator_records_runtime_app_id_for_failed_remote_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_on_modal(**_: object) -> dict[str, object]:
        exc = RuntimeError("remote failure")
        setattr(exc, "runtime_app_id", "ap-runtime-fail")
        raise exc

    monkeypatch.setattr("pipelines_v2.runtime.modal.run_on_modal", fake_run_on_modal)
    catalog = FileCatalog(tmp_path / "catalog")
    runner = ModalRunner(
        resources=ModalResources(gpu="L4"),
        artifacts=ModalVolumeStore(name="xenon-data", root=str(tmp_path / "mounted")),
        catalog=catalog,
    )
    orchestrator = WorkflowOrchestrator(runners={"capture_gpu": runner})
    workflow = WorkflowSpec(
        name="modal_runtime_id_failure",
        steps=(
            WorkflowStep(
                name="capture",
                runner="capture_gpu",
                spec=make_toy_capture_spec(),
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="remote failure"):
        orchestrator.run(workflow)

    run_files = sorted((tmp_path / "catalog" / "workflow_runs").glob("*.json"))
    assert len(run_files) == 1
    run_id = run_files[0].stem
    step_records = {record.step_name: record for record in catalog.list_workflow_steps(run_id)}
    assert step_records["capture"].status == "failed"
    assert step_records["capture"].runtime_app_id == "ap-runtime-fail"


def test_modal_volume_store_blocks_large_remote_read_without_allow_large_transfer(
    tmp_path: Path,
) -> None:
    store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts",
        local_cache_root=tmp_path / "modal_cache",
        transfer_policy=TransferPolicy(max_download_bytes=32),
    )

    with pytest.raises(TransferPolicyError, match="allow_large_transfer=True"):
        store.read_json_ref(
            {
                "store": "modal_volume",
                "name": "xenon-data",
                "path": "/data/artifacts/capture_123/features/resid.json",
                "bytes": 4096,
            }
        )


def test_modal_volume_store_allows_large_remote_read_when_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts",
        local_cache_root=tmp_path / "modal_cache",
        transfer_policy=TransferPolicy(allow_large_transfer=True, max_download_bytes=32),
    )

    def fake_get(self: ModalVolumeStore, remote_path: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as f:
            json.dump({"ok": True}, f)

    monkeypatch.setattr(ModalVolumeStore, "_run_modal_volume_get", fake_get)

    payload = store.read_json_ref(
        {
            "store": "modal_volume",
            "name": "xenon-data",
            "path": "/data/artifacts/capture_123/features/resid.json",
            "bytes": 4096,
        }
    )

    assert payload == {"ok": True}


def test_modal_volume_store_reads_cached_safetensors_without_large_transfer_override(tmp_path: Path) -> None:
    store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts",
        local_cache_root=tmp_path / "modal_cache",
        transfer_policy=TransferPolicy(max_download_bytes=32),
    )
    cached_tensor = tmp_path / "modal_cache" / "_refs" / "artifacts" / "capture_123" / "features" / "feature_tensors.safetensors"
    cached_tensor.parent.mkdir(parents=True, exist_ok=True)
    save_file({"tensor_0": np.zeros((2, 3), dtype=np.float32)}, str(cached_tensor))

    payload = store.read_safetensors_ref(
        {
            "store": "modal_volume",
            "name": "xenon-data",
            "path": "/data/artifacts/capture_123/features/feature_tensors.safetensors",
            "format": "safetensors",
            "bytes": 4096,
        }
    )

    assert payload["tensor_0"].shape == (2, 3)


def test_modal_volume_store_reads_localized_artifact_safetensors_without_redownloading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts",
        local_cache_root=tmp_path / "modal_cache",
        transfer_policy=TransferPolicy(max_download_bytes=32),
    )
    cached_tensor = tmp_path / "modal_cache" / "capture_123" / "features" / "feature_tensors.safetensors"
    cached_tensor.parent.mkdir(parents=True, exist_ok=True)
    save_file({"tensor_0": np.zeros((1, 2), dtype=np.float16)}, str(cached_tensor))

    def fail_get(self: ModalVolumeStore, remote_path: str, destination: Path) -> None:
        raise AssertionError(f"unexpected download: {remote_path} -> {destination}")

    monkeypatch.setattr(ModalVolumeStore, "_run_modal_volume_get", fail_get)

    payload = store.read_safetensors_ref(
        {
            "store": "modal_volume",
            "name": "xenon-data",
            "path": "/data/artifacts/capture_123/features/feature_tensors.safetensors",
            "format": "safetensors",
            "bytes": 4096,
        }
    )

    assert payload["tensor_0"].shape == (1, 2)


def test_capture_artifact_localize_blocks_large_remote_transfer(tmp_path: Path) -> None:
    store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts",
        local_cache_root=tmp_path / "modal_cache",
        transfer_policy=TransferPolicy(max_download_bytes=128),
    )
    manifest = {
        "artifact_id": "capture_123",
        "artifact_kind": "capture",
        "schema_version": 1,
        "operation_spec_hash": "abc",
        "created_at": "2026-04-13T00:00:00+00:00",
        "engine": {"kind": "vllm"},
        "runner": {"kind": "modal"},
        "input_artifact_refs": [],
        "example_coverage": {"example_count": 2},
        "storage_refs": {
            "features": {
                "resid_last": {
                    "store": "modal_volume",
                    "name": "xenon-data",
                    "path": "/data/artifacts/capture_123/features/resid_last.json",
                    "bytes": 80,
                }
            },
            "generations": {
                "store": "modal_volume",
                "name": "xenon-data",
                "path": "/data/artifacts/capture_123/generations.json",
                "bytes": 80,
            },
            "manifest": {
                "store": "modal_volume",
                "name": "xenon-data",
                "path": "/data/artifacts/capture_123/manifest.json",
                "bytes": 80,
            },
        },
        "metadata": {},
    }
    capture_artifact = CaptureArtifact(
        _manifest=ArtifactManifest.from_dict(manifest),
        store=store,
    )

    with pytest.raises(TransferPolicyError, match="capture artifact"):
        capture_artifact.localize()


def test_capture_artifact_localize_uses_existing_cache_without_large_transfer_override(tmp_path: Path) -> None:
    store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts",
        local_cache_root=tmp_path / "modal_cache",
        transfer_policy=TransferPolicy(max_download_bytes=128),
    )
    cached_artifact = tmp_path / "modal_cache" / "capture_123"
    cached_artifact.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_id": "capture_123",
        "artifact_kind": "capture",
        "schema_version": 1,
        "operation_spec_hash": "abc",
        "created_at": "2026-04-13T00:00:00+00:00",
        "engine": {"kind": "vllm"},
        "runner": {"kind": "modal"},
        "input_artifact_refs": [],
        "example_coverage": {"example_count": 2},
        "storage_refs": {
            "features": {
                "resid_last": {
                    "store": "modal_volume",
                    "name": "xenon-data",
                    "path": "/data/artifacts/capture_123/features/resid_last.metadata.json",
                    "format": "residual_safetensors_v1",
                    "metadata_path": "/data/artifacts/capture_123/features/resid_last.metadata.json",
                    "tensor_path": "/data/artifacts/capture_123/features/feature_tensors.safetensors",
                    "metadata_bytes": 80,
                    "tensor_bytes": 4096,
                    "bytes": 4176,
                }
            },
            "manifest": {
                "store": "modal_volume",
                "name": "xenon-data",
                "path": "/data/artifacts/capture_123/manifest.json",
                "format": "json",
                "bytes": 80,
            },
        },
        "metadata": {},
    }
    capture_artifact = CaptureArtifact(
        _manifest=ArtifactManifest.from_dict(manifest),
        store=store,
    )

    assert capture_artifact.localize() == cached_artifact


def test_workflow_spec_round_trips_from_dict() -> None:
    workflow = WorkflowSpec(
        name="capture_then_probe",
        steps=(
            WorkflowStep(
                name="capture",
                runner="gpu",
                spec=make_toy_capture_spec(),
                description="Capture residual activations for the probe dataset.",
            ),
            WorkflowStep(
                name="probe",
                runner="cpu",
                spec=ProbeSpec(feature="resid_last", labels="class"),
                depends_on=("capture",),
            ),
        ),
    )

    restored = WorkflowSpec.from_dict(workflow.to_dict())

    assert restored.to_dict() == workflow.to_dict()
    assert restored.steps[0].description == "Capture residual activations for the probe dataset."


def test_workflow_step_description_is_not_semantic_identity() -> None:
    base = WorkflowStep(
        name="capture",
        runner="gpu",
        spec=make_toy_capture_spec(),
        description="Old operator note.",
    )
    edited = WorkflowStep(
        name="capture",
        runner="gpu",
        spec=make_toy_capture_spec(),
        description="New operator note.",
    )

    assert base.semantic_hash() == edited.semantic_hash()
    assert base.spec_hash() != edited.spec_hash()


def test_workflow_orchestrator_uses_named_runners_and_dependency_order() -> None:
    observed: list[tuple[str, str]] = []

    class RecordingRunner:
        def __init__(self, name: str) -> None:
            self.name = name

        def plan(self, spec: object) -> ExecutionPlan:
            return ExecutionPlan(
                spec_kind=getattr(spec, "kind", "unknown"),
                required_capabilities=frozenset(),
                engine_capabilities=frozenset(),
                artifact_kinds=(),
                checks=(),
            )

        def run(self, spec: object) -> dict[str, str]:
            observed.append((self.name, getattr(spec, "kind", "unknown")))
            return {"runner": self.name, "kind": getattr(spec, "kind", "unknown")}

    orchestrator = WorkflowOrchestrator(
        runners={
            "gpu": RecordingRunner("gpu"),
            "cpu": RecordingRunner("cpu"),
        }
    )
    workflow = WorkflowSpec(
        name="capture_then_probe",
        steps=(
            WorkflowStep(
                name="capture",
                runner="gpu",
                spec=make_toy_capture_spec(),
                description="Capture toy residual features.",
            ),
            WorkflowStep(
                name="probe",
                runner="cpu",
                spec=ProbeSpec(feature="resid_last", labels="class"),
                depends_on=("capture",),
            ),
        ),
    )

    plan = orchestrator.plan(workflow)
    result = orchestrator.run(workflow)

    assert [step.name for step in plan.steps] == ["capture", "probe"]
    assert [step.description for step in plan.steps] == ["Capture toy residual features.", None]
    assert observed == [("gpu", "capture"), ("cpu", "probe")]
    assert result.step("probe") == {"runner": "cpu", "kind": "probe"}


def test_local_runner_executes_probe_from_capture_feature(tmp_path: Path) -> None:
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="alpha", labels={"class": "pos"}, case_key="c1"),
            Example(key="b", prompt="beta", labels={"class": "pos"}, case_key="c2"),
            Example(key="c", prompt="gamma", labels={"class": "neg"}, case_key="c3"),
            Example(key="d", prompt="delta", labels={"class": "neg"}, case_key="c4"),
        ],
        name="probe_dataset",
    )
    capture_runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    analysis_runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))

    cap = capture_runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=4, num_layers=2),
            dataset=dataset,
            sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0, 1])],
        )
    )

    probe = analysis_runner.run(
        ProbeSpec(
            feature=cap.feature("resid_last"),
            labels=dataset.labels("class"),
            folds=2,
            baselines=["majority", "shuffled_label"],
            metrics=["accuracy", "balanced_accuracy", "selectivity"],
        )
    )

    summary = probe.summary()

    assert probe.manifest().artifact_kind == "probe"
    assert summary["example_count"] == 4
    assert summary["best_layer"] in {0, 1}


def test_probe_selectivity_uses_shuffled_control_without_exposing_baseline_by_default(tmp_path: Path) -> None:
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="alpha", labels={"class": "pos"}),
            Example(key="b", prompt="beta", labels={"class": "pos"}),
            Example(key="c", prompt="gamma", labels={"class": "pos"}),
            Example(key="d", prompt="delta", labels={"class": "pos"}),
            Example(key="e", prompt="epsilon", labels={"class": "neg"}),
            Example(key="f", prompt="zeta", labels={"class": "neg"}),
            Example(key="g", prompt="eta", labels={"class": "neg"}),
            Example(key="h", prompt="theta", labels={"class": "neg"}),
        ],
        name="probe_selectivity_dataset",
    )
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=4, num_layers=2),
            dataset=dataset,
            sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0, 1])],
        )
    )

    probe = runner.run(
        ProbeSpec(
            feature=capture.feature("resid_last"),
            labels=dataset.labels("class"),
            folds=2,
            metrics=["accuracy", "selectivity"],
        )
    )

    payload = probe.result()

    assert any(layer["selectivity"] != layer["accuracy"] for layer in payload["layers"])
    assert all("baseline_shuffled" not in layer for layer in payload["layers"])


def test_remote_executor_executes_probe_spec_with_serialized_refs(tmp_path: Path) -> None:
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="alpha", labels={"class": "pos"}, case_key="c1"),
            Example(key="b", prompt="beta", labels={"class": "pos"}, case_key="c2"),
            Example(key="c", prompt="gamma", labels={"class": "neg"}, case_key="c3"),
            Example(key="d", prompt="delta", labels={"class": "neg"}, case_key="c4"),
        ],
        name="probe_dataset",
    )
    capture_runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    cap = capture_runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=4, num_layers=2),
            dataset=dataset,
            sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0, 1])],
        )
    )
    probe_spec = ProbeSpec(
        feature=cap.feature("resid_last"),
        labels=dataset.labels("class"),
        folds=2,
        baselines=["majority", "shuffled_label"],
    )

    manifest = execute_remote(
        runner_config={"kind": "modal", "resources": {"cpu": 4}},
        store_config={"kind": "local", "root": str(tmp_path / "artifacts")},
        spec_payload=probe_spec.to_dict(),
    )

    result_path = Path(manifest["storage_refs"]["result"]["path"])
    with result_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert manifest["artifact_kind"] == "probe"
    assert payload["summary"]["example_count"] == 4
    assert payload["summary"]["best_layer"] in {0, 1}


def test_workflow_orchestrator_resolves_step_feature_refs_with_real_runners(tmp_path: Path) -> None:
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="alpha", labels={"class": "pos"}, case_key="c1"),
            Example(key="b", prompt="beta", labels={"class": "pos"}, case_key="c2"),
            Example(key="c", prompt="gamma", labels={"class": "neg"}, case_key="c3"),
            Example(key="d", prompt="delta", labels={"class": "neg"}, case_key="c4"),
        ],
        name="workflow_probe_dataset",
    )
    shared_store = LocalArtifactStore(tmp_path / "artifacts")
    orchestrator = WorkflowOrchestrator(
        runners={
            "capture": LocalRunner(artifacts=shared_store),
            "analysis": LocalRunner(artifacts=shared_store),
        }
    )

    result = orchestrator.run(
        WorkflowSpec(
            name="capture_then_probe",
            steps=(
                WorkflowStep(
                    name="capture",
                    runner="capture",
                    spec=CaptureSpec(
                        engine=ToyEngine(hidden_size=4, num_layers=2),
                        dataset=dataset,
                        sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0, 1])],
                    ),
                ),
                WorkflowStep(
                    name="probe",
                    runner="analysis",
                    spec=ProbeSpec(
                        feature=StepRef("capture").feature("resid_last"),
                        labels=dataset.labels("class"),
                        folds=2,
                        baselines=["majority"],
                    ),
                    depends_on=("capture",),
                ),
            ),
        )
    )

    assert result.step("probe").manifest().artifact_kind == "probe"
    assert result.step("probe").summary()["example_count"] == 4


def test_workflow_orchestrator_infers_dependencies_and_fans_out_parallel() -> None:
    class SleepingRunner:
        def __init__(self, name: str, delay: float) -> None:
            self.name = name
            self.delay = delay

        def plan(self, spec: object) -> ExecutionPlan:
            return ExecutionPlan(
                spec_kind=getattr(spec, "kind", "unknown"),
                required_capabilities=frozenset(),
                engine_capabilities=frozenset(),
                artifact_kinds=(),
            )

        def run(self, spec: object) -> object:
            time.sleep(self.delay)
            return {"runner": self.name, "spec_kind": getattr(spec, "kind", "unknown")}

    orchestrator = WorkflowOrchestrator(
        runners={
            "capture": SleepingRunner("capture", 0.10),
            "analysis": SleepingRunner("analysis", 0.25),
        }
    )
    workflow = WorkflowSpec(
        name="fanout",
        steps=(
            WorkflowStep(
                name="capture",
                runner="capture",
                spec=ReportSpec(template="capture"),
            ),
            WorkflowStep(
                name="probe_a",
                runner="analysis",
                spec=ReportSpec(template="probe_a", inputs=[StepRef("capture")]),
            ),
            WorkflowStep(
                name="probe_b",
                runner="analysis",
                spec=ReportSpec(template="probe_b", inputs=[StepRef("capture")]),
            ),
        ),
    )

    start = time.perf_counter()
    result = orchestrator.run(workflow)
    elapsed = time.perf_counter() - start

    assert result.step("probe_a")["runner"] == "analysis"
    assert result.step("probe_b")["runner"] == "analysis"
    assert elapsed < 0.55


def test_capture_spec_semantic_hash_ignores_vllm_batching_runtime_tuning() -> None:
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="alpha"),
        ]
    )
    spec_a = CaptureSpec(
        engine=VLLMEngine(
            model_id="/models/Qwen/Qwen3-30B-A3B",
            enforce_eager=True,
            max_num_seqs=1,
            gpu_memory_utilization=0.8,
        ),
        dataset=dataset,
        sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0])],
    )
    spec_b = CaptureSpec(
        engine=VLLMEngine(
            model_id="/models/Qwen/Qwen3-30B-A3B",
            enforce_eager=False,
            max_num_seqs=16,
            gpu_memory_utilization=0.95,
        ),
        dataset=dataset,
        sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0])],
    )

    assert spec_a.spec_hash() != spec_b.spec_hash()
    assert spec_a.semantic_hash() == spec_b.semantic_hash()


def test_workflow_orchestrator_records_workflow_lineage_and_can_resume(tmp_path: Path) -> None:
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="alpha", labels={"class": "pos", "split": "train"}, case_key="c1"),
            Example(key="b", prompt="beta", labels={"class": "neg", "split": "train"}, case_key="c2"),
            Example(key="c", prompt="gamma", labels={"class": "pos", "split": "test"}, case_key="c3"),
            Example(key="d", prompt="delta", labels={"class": "neg", "split": "test"}, case_key="c4"),
        ],
        name="resume_dataset",
    )
    catalog = FileCatalog(tmp_path / "catalog")
    shared_store = LocalArtifactStore(tmp_path / "artifacts")
    capture_runner = _FailOnceRunner(LocalRunner(artifacts=shared_store, catalog=catalog))
    analysis_runner = _FailOnceRunner(LocalRunner(artifacts=shared_store, catalog=catalog))
    report_runner = _FailOnceRunner(LocalRunner(artifacts=shared_store, catalog=catalog), fail_step="report")
    orchestrator = WorkflowOrchestrator(
        runners={
            "capture": capture_runner,
            "analysis": analysis_runner,
            "report": report_runner,
        }
    )
    workflow = WorkflowSpec(
        name="resume_workflow",
        steps=(
            WorkflowStep(
                name="capture",
                runner="capture",
                spec=CaptureSpec(
                    engine=ToyEngine(hidden_size=4, num_layers=2),
                    dataset=dataset,
                    sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0, 1])],
                ),
            ),
            WorkflowStep(
                name="probe",
                runner="analysis",
                spec=ProbeSpec(
                    feature=StepRef("capture").feature("resid_last"),
                    labels=dataset.labels("class"),
                    split=dataset.labels("split"),
                    folds=2,
                    baselines=["majority"],
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report",
                spec=ReportSpec(
                    template="resume_test",
                    output_dir=str(tmp_path / "reports"),
                    inputs=[StepRef("capture"), StepRef("probe")],
                ),
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="intentional failure"):
        orchestrator.run(workflow)

    run_files = sorted((tmp_path / "catalog" / "workflow_runs").glob("*.json"))
    assert len(run_files) == 1
    run_id = run_files[0].stem
    first_run = catalog.load_workflow_run(run_id)
    assert first_run is not None
    assert first_run.status == "failed"

    step_records = {record.step_name: record for record in catalog.list_workflow_steps(run_id)}
    assert step_records["capture"].status == "completed"
    assert step_records["probe"].status == "completed"
    assert step_records["report"].status == "failed"

    capture_manifest = catalog.load_artifact(step_records["capture"].artifact_id or "")
    assert capture_manifest is not None
    assert capture_manifest.workflow_context["workflow_step_key"] == f"{workflow.semantic_hash()}.capture"
    assert capture_manifest.workflow_context["run_id"] == run_id

    resumed = orchestrator.run(workflow, resume_run_id=run_id)
    assert resumed.run_id == run_id
    assert resumed.step("report").manifest().artifact_kind == "report"
    assert capture_runner.calls.count("capture") == 1
    assert analysis_runner.calls.count("probe") == 1
    assert report_runner.calls.count("report") == 2

    resumed_run = catalog.load_workflow_run(run_id)
    assert resumed_run is not None
    assert resumed_run.status == "completed"


def test_workflow_orchestrator_finishes_inflight_siblings_and_resume_skips_completed_capture(tmp_path: Path) -> None:
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="alpha", labels={"class": "pos", "split": "train"}, case_key="c1"),
            Example(key="b", prompt="beta", labels={"class": "neg", "split": "train"}, case_key="c2"),
            Example(key="c", prompt="gamma", labels={"class": "pos", "split": "test"}, case_key="c3"),
            Example(key="d", prompt="delta", labels={"class": "neg", "split": "test"}, case_key="c4"),
        ],
        name="parallel_resume_dataset",
    )
    catalog = FileCatalog(tmp_path / "catalog")
    shared_store = LocalArtifactStore(tmp_path / "artifacts")
    residual_runner = _FailOnceRunner(
        LocalRunner(artifacts=shared_store, catalog=catalog),
        delay_seconds=0.2,
    )
    router_runner = _FailOnceRunner(
        LocalRunner(artifacts=shared_store, catalog=catalog),
        fail_step="capture_router",
    )
    analysis_runner = _FailOnceRunner(LocalRunner(artifacts=shared_store, catalog=catalog))
    orchestrator = WorkflowOrchestrator(
        runners={
            "capture_residual": residual_runner,
            "capture_router": router_runner,
            "analysis": analysis_runner,
        },
        max_parallelism=2,
    )
    workflow = WorkflowSpec(
        name="parallel_branch_resume",
        steps=(
            WorkflowStep(
                name="capture_residual",
                runner="capture_residual",
                spec=CaptureSpec(
                    engine=ToyEngine(hidden_size=4, num_layers=2),
                    dataset=dataset,
                    sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0, 1])],
                ),
            ),
                WorkflowStep(
                    name="capture_router",
                    runner="capture_router",
                    spec=CaptureSpec(
                        engine=ToyEngine(hidden_size=4, num_layers=2),
                        dataset=dataset,
                        sites=[MoERoutingSite(name="router_last", layers=[0])],
                    ),
                ),
            WorkflowStep(
                name="probe",
                runner="analysis",
                spec=ProbeSpec(
                    feature=StepRef("capture_residual").feature("resid_last"),
                    labels=dataset.labels("class"),
                    split=dataset.labels("split"),
                    folds=2,
                    baselines=["majority"],
                ),
            ),
            WorkflowStep(
                name="probe_router",
                runner="analysis",
                spec=ProbeSpec(
                    feature=StepRef("capture_router").feature("router_last"),
                    labels=dataset.labels("class"),
                    split=dataset.labels("split"),
                    folds=2,
                    baselines=["majority"],
                ),
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="intentional failure for capture_router"):
        orchestrator.run(workflow)

    run_files = sorted((tmp_path / "catalog" / "workflow_runs").glob("*.json"))
    assert len(run_files) == 1
    run_id = run_files[0].stem
    step_records = {record.step_name: record for record in catalog.list_workflow_steps(run_id)}
    assert step_records["capture_residual"].status == "completed"
    assert step_records["capture_router"].status == "failed"
    assert "probe" not in step_records
    assert step_records["probe_router"].status == "blocked"
    assert residual_runner.calls.count("capture_residual") == 1
    assert router_runner.calls.count("capture_router") == 1
    assert analysis_runner.calls.count("probe") == 0
    assert analysis_runner.calls.count("probe_router") == 0

    resumed = orchestrator.run(workflow, resume_run_id=run_id)

    assert resumed.run_id == run_id
    assert resumed.step("probe").manifest().artifact_kind == "probe"
    assert residual_runner.calls.count("capture_residual") == 1
    assert router_runner.calls.count("capture_router") == 2
    assert analysis_runner.calls.count("probe") == 1
    assert analysis_runner.calls.count("probe_router") == 1

    resumed_steps = {record.step_name: record for record in catalog.list_workflow_steps(run_id)}
    assert resumed_steps["capture_residual"].status == "completed"
    assert resumed_steps["capture_router"].status == "completed"
    assert resumed_steps["probe"].status == "completed"
    assert resumed_steps["probe_router"].status == "completed"


def test_workflow_orchestrator_resume_recovers_persisted_artifact_from_running_step_record(tmp_path: Path) -> None:
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="alpha", labels={"class": "pos", "split": "train"}, case_key="c1"),
            Example(key="b", prompt="beta", labels={"class": "neg", "split": "train"}, case_key="c2"),
            Example(key="c", prompt="gamma", labels={"class": "pos", "split": "test"}, case_key="c3"),
            Example(key="d", prompt="delta", labels={"class": "neg", "split": "test"}, case_key="c4"),
        ],
        name="recover_running_step_dataset",
    )
    catalog = FileCatalog(tmp_path / "catalog")
    shared_store = LocalArtifactStore(tmp_path / "artifacts")
    capture_inner = LocalRunner(artifacts=shared_store, catalog=catalog)
    capture_runner = _FailOnceRunner(LocalRunner(artifacts=shared_store, catalog=catalog))
    analysis_runner = _FailOnceRunner(LocalRunner(artifacts=shared_store, catalog=catalog))
    orchestrator = WorkflowOrchestrator(
        runners={
            "capture": capture_runner,
            "analysis": analysis_runner,
        }
    )
    capture_spec = CaptureSpec(
        engine=ToyEngine(hidden_size=4, num_layers=2),
        dataset=dataset,
        sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0, 1])],
    )
    workflow = WorkflowSpec(
        name="recover_running_step",
        steps=(
            WorkflowStep(
                name="capture",
                runner="capture",
                spec=capture_spec,
            ),
            WorkflowStep(
                name="probe",
                runner="analysis",
                spec=ProbeSpec(
                    feature=StepRef("capture").feature("resid_last"),
                    labels=dataset.labels("class"),
                    split=dataset.labels("split"),
                    folds=2,
                    baselines=["majority"],
                ),
            ),
        ),
    )

    run_id = f"wr_{workflow.semantic_hash()[:12]}_recover"
    started_at = utc_now_iso()
    catalog.record_workflow_run(
        WorkflowRunRecord(
            run_id=run_id,
            workflow_name=workflow.name,
            workflow_hash=workflow.semantic_hash(),
            workflow_spec_hash=workflow.spec_hash(),
            workflow_payload=workflow.to_dict(),
            status="running",
            started_at=started_at,
        )
    )
    capture_step = workflow.ordered_steps()[0]
    capture_context = WorkflowStepContext(
        run_id=run_id,
        workflow_name=workflow.name,
        workflow_hash=workflow.semantic_hash(),
        workflow_spec_hash=workflow.spec_hash(),
        step_name=capture_step.name,
        step_index=0,
        runner=capture_step.runner,
        step_semantic_hash=capture_step.semantic_hash(),
        step_spec_hash=capture_step.spec_hash(),
    )
    persisted_capture = capture_inner.run(capture_spec, workflow_context=capture_context)
    catalog.record_workflow_step(
        WorkflowStepRecord(
            run_id=run_id,
            workflow_hash=workflow.semantic_hash(),
            workflow_step_key=capture_context.workflow_step_key,
            step_name="capture",
            step_index=0,
            runner="capture",
            status="running",
            step_semantic_hash=capture_step.semantic_hash(),
            step_spec_hash=capture_step.spec_hash(),
            started_at=started_at,
        )
    )

    resumed = orchestrator.run(workflow, resume_run_id=run_id)

    assert resumed.run_id == run_id
    assert resumed.step("capture").id == persisted_capture.id
    assert capture_runner.calls.count("capture") == 0
    assert analysis_runner.calls.count("probe") == 1

    step_records = {record.step_name: record for record in catalog.list_workflow_steps(run_id)}
    assert step_records["capture"].status == "completed"
    assert step_records["capture"].artifact_id == persisted_capture.id
    assert step_records["probe"].status == "completed"


def test_workflow_orchestrator_reuses_matching_completed_steps_across_runs(tmp_path: Path) -> None:
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="alpha", labels={"class": "pos", "split": "train"}, case_key="c1"),
            Example(key="b", prompt="beta", labels={"class": "neg", "split": "train"}, case_key="c2"),
            Example(key="c", prompt="gamma", labels={"class": "pos", "split": "test"}, case_key="c3"),
            Example(key="d", prompt="delta", labels={"class": "neg", "split": "test"}, case_key="c4"),
        ]
    )
    catalog = FileCatalog(tmp_path / "catalog")
    shared_store = LocalArtifactStore(tmp_path / "artifacts")
    capture_runner = _FailOnceRunner(LocalRunner(artifacts=shared_store, catalog=catalog))
    analysis_runner = _FailOnceRunner(LocalRunner(artifacts=shared_store, catalog=catalog))
    report_runner = _FailOnceRunner(LocalRunner(artifacts=shared_store, catalog=catalog))
    orchestrator = WorkflowOrchestrator(
        runners={
            "capture": capture_runner,
            "analysis": analysis_runner,
            "report": report_runner,
        }
    )

    def build_workflow(template: str) -> WorkflowSpec:
        return WorkflowSpec(
            name=f"reuse_{template}",
            steps=(
                WorkflowStep(
                    name="capture",
                    runner="capture",
                    spec=CaptureSpec(
                        engine=ToyEngine(hidden_size=4, num_layers=2),
                        dataset=dataset,
                        sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0, 1])],
                    ),
                ),
                WorkflowStep(
                    name="probe",
                    runner="analysis",
                    spec=ProbeSpec(
                        feature=StepRef("capture").feature("resid_last"),
                        labels=dataset.labels("class"),
                        split=dataset.labels("split"),
                        folds=2,
                        baselines=["majority"],
                    ),
                ),
                WorkflowStep(
                    name="report",
                    runner="report",
                    spec=ReportSpec(
                        template=template,
                        output_dir=str(tmp_path / "reports" / template),
                        inputs=[StepRef("capture"), StepRef("probe")],
                    ),
                ),
            ),
        )

    first = orchestrator.run(build_workflow("v1"))
    first_capture_id = first.step("capture").id
    first_probe_id = first.step("probe").id

    second = orchestrator.run(build_workflow("v2"), reuse_completed=True)
    assert second.step("capture").id == first_capture_id
    assert second.step("probe").id == first_probe_id
    assert second.step("report").id != first.step("report").id

    second_run = catalog.load_workflow_run(second.run_id or "")
    assert second_run is not None
    second_steps = {record.step_name: record for record in catalog.list_workflow_steps(second.run_id or "")}
    assert second_steps["capture"].status == "reused"
    assert second_steps["probe"].status == "reused"
    assert second_steps["report"].status == "completed"


class _CatalogProbe:
    def __init__(self, *, kind: str, name: str) -> None:
        self.kind = kind
        self.name = name

    def identity(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name}


class _LineageCheckingCatalog(_CatalogProbe):
    def __init__(self, *, kind: str = "postgres", name: str = "remote") -> None:
        super().__init__(kind=kind, name=name)
        self.artifacts: dict[str, ArtifactManifest] = {}
        self.runs: dict[str, WorkflowRunRecord] = {}
        self.steps: dict[tuple[str, str], WorkflowStepRecord] = {}

    def record_artifact(self, manifest: ArtifactManifest) -> None:
        context = dict(manifest.workflow_context)
        run_id = context.get("run_id")
        step_name = context.get("step_name")
        if run_id is not None and step_name is not None and (str(run_id), str(step_name)) not in self.steps:
            raise AssertionError(f"missing workflow step for artifact lineage: {run_id}:{step_name}")
        self.artifacts[manifest.artifact_id] = manifest

    def load_artifact(self, artifact_id: str) -> ArtifactManifest | None:
        return self.artifacts.get(artifact_id)

    def find_artifact_for_workflow_step(
        self,
        *,
        run_id: str,
        workflow_step_key: str,
    ) -> ArtifactManifest | None:
        for artifact in self.artifacts.values():
            context = dict(artifact.workflow_context)
            if context.get("run_id") == run_id and context.get("workflow_step_key") == workflow_step_key:
                return artifact
        return None

    def record_workflow_run(self, record: WorkflowRunRecord) -> None:
        if record.parent_run_id is not None and record.parent_run_id not in self.runs:
            raise AssertionError(f"missing parent workflow run: {record.parent_run_id}")
        self.runs[record.run_id] = record

    def load_workflow_run(self, run_id: str) -> WorkflowRunRecord | None:
        return self.runs.get(run_id)

    def list_workflow_runs(
        self,
        *,
        workflow_name: str | None = None,
        workflow_hash: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[WorkflowRunRecord]:
        records = list(self.runs.values())
        if workflow_name is not None:
            records = [record for record in records if record.workflow_name == workflow_name]
        if workflow_hash is not None:
            records = [record for record in records if record.workflow_hash == workflow_hash]
        if status is not None:
            records = [record for record in records if record.status == status]
        records.sort(key=lambda item: (item.started_at, item.run_id), reverse=True)
        return records[:limit] if limit is not None else records

    def record_workflow_step(self, record: WorkflowStepRecord) -> None:
        self.steps[(record.run_id, record.step_name)] = record

    def list_workflow_steps(self, run_id: str) -> list[WorkflowStepRecord]:
        records = [record for (record_run_id, _), record in self.steps.items() if record_run_id == run_id]
        records.sort(key=lambda item: (item.step_index, item.step_name))
        return records

    def find_latest_reusable_step(
        self,
        *,
        step_name: str,
        step_semantic_hash: str,
        input_artifact_refs: tuple[str, ...],
    ) -> WorkflowStepRecord | None:
        matches = [
            record
            for record in self.steps.values()
            if record.step_name == step_name
            and record.step_semantic_hash == step_semantic_hash
            and tuple(record.input_artifact_refs) == input_artifact_refs
            and record.status in {"completed", "reused"}
        ]
        matches.sort(key=lambda item: (item.finished_at or "", item.run_id), reverse=True)
        return matches[0] if matches else None


def test_preferred_workflow_metadata_catalog_prefers_file_catalog(tmp_path: Path) -> None:
    local = FileCatalog(tmp_path / "catalog")
    remote = _CatalogProbe(kind="postgres", name="remote")
    nested = CompositeCatalog((remote,))
    composite = CompositeCatalog((nested, local))

    preferred = preferred_workflow_metadata_catalog(composite)

    assert preferred.kind == "file"
    assert preferred.identity() == local.identity()


def test_workflow_orchestrator_prefers_local_file_catalog_for_metadata(tmp_path: Path) -> None:
    local = FileCatalog(tmp_path / "catalog")
    remote = _CatalogProbe(kind="postgres", name="remote")
    orchestrator = WorkflowOrchestrator(
        runners={
            "capture": LocalRunner(
                artifacts=LocalArtifactStore(tmp_path / "artifacts"),
                catalog=CompositeCatalog((local, remote)),
            ),
            "analysis": LocalRunner(
                artifacts=LocalArtifactStore(tmp_path / "artifacts"),
                catalog=CompositeCatalog((local, remote)),
            ),
        }
    )

    catalog = orchestrator._workflow_catalog()

    assert catalog is not None
    assert catalog.kind == "composite"
    assert catalog.identity() == CompositeCatalog((local, remote)).identity()


def test_registry_catalog_keeps_shared_runner_catalog_from_runner_specs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local = FileCatalog(tmp_path / "catalog")
    remote = _CatalogProbe(kind="postgres", name="remote")

    class _Runner:
        def __init__(self, catalog: Any) -> None:
            self.catalog = catalog

    monkeypatch.setattr(
        "pipelines_v2.cli._build_runners",
        lambda ns, runner_specs: {
            "capture_gpu": _Runner(CompositeCatalog((local, remote))),
            "analysis_cpu": _Runner(CompositeCatalog((local, remote))),
        },
    )

    catalog = _registry_catalog(types.SimpleNamespace(), runner_specs={})

    assert catalog.kind == "composite"
    assert catalog.identity() == CompositeCatalog((local, remote)).identity()


def test_runner_spec_registry_mirrors_workflow_lineage_before_remote_artifact_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local = FileCatalog(tmp_path / "catalog")
    remote = _LineageCheckingCatalog()
    runner = LocalRunner(
        artifacts=LocalArtifactStore(tmp_path / "artifacts"),
        catalog=CompositeCatalog((local, remote)),
    )
    monkeypatch.setattr("pipelines_v2.cli._build_runners", lambda ns, runner_specs: {"analysis": runner})

    catalog = _registry_catalog(types.SimpleNamespace(), runner_specs={})
    workflow = WorkflowSpec(
        name="composite_lineage",
        steps=(
            WorkflowStep(
                name="seed",
                runner="analysis",
                spec=TransformSpec(builder=TransformBuilder.from_function(_inline_transform_seed), inputs={"value": 7}),
            ),
        ),
    )

    result = WorkflowOrchestrator(runners={"analysis": runner}, workflow_catalog=catalog).run(workflow)
    artifact = result.step("seed")

    assert remote.load_workflow_run(result.run_id or "") is not None
    assert remote.list_workflow_steps(result.run_id or "")[0].status == "completed"
    assert remote.load_artifact(artifact.id) is not None


def test_mirror_workflow_run_lineage_backfills_local_only_parent_for_composite_catalog(tmp_path: Path) -> None:
    local = FileCatalog(tmp_path / "catalog")
    remote = _LineageCheckingCatalog()
    workflow = WorkflowSpec(name="lineage_backfill", steps=())
    parent = WorkflowRunRecord(
        run_id="wr_parent",
        workflow_name=workflow.name,
        workflow_hash=workflow.semantic_hash(),
        workflow_spec_hash=workflow.spec_hash(),
        workflow_payload=workflow.to_dict(),
        status="failed",
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
        error="old local-only failure",
    )
    local.record_workflow_run(parent)
    catalog = CompositeCatalog((local, remote))

    _mirror_workflow_run_lineage(catalog, parent.run_id)
    catalog.record_workflow_run(
        WorkflowRunRecord(
            run_id="wr_child",
            workflow_name=workflow.name,
            workflow_hash=workflow.semantic_hash(),
            workflow_spec_hash=workflow.spec_hash(),
            workflow_payload=workflow.to_dict(),
            status="running",
            started_at=utc_now_iso(),
            parent_run_id=parent.run_id,
        )
    )

    assert remote.load_workflow_run(parent.run_id) is not None
    assert remote.load_workflow_run("wr_child") is not None


def test_resolve_workflow_metadata_catalog_falls_back_to_workspace_registry_for_explicit_run_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    primary = FileCatalog(tmp_path / "runner_catalog")
    fallback = FileCatalog(tmp_path / "workspace_catalog")
    workflow = WorkflowSpec(name="fallback_lookup", steps=())
    record = WorkflowRunRecord(
        run_id="wr_lookup_only_in_workspace",
        workflow_name=workflow.name,
        workflow_hash=workflow.semantic_hash(),
        workflow_spec_hash=workflow.spec_hash(),
        workflow_payload=workflow.to_dict(),
        status="completed",
        started_at=utc_now_iso(),
    )
    fallback.record_workflow_run(record)

    def fake_registry(ns: Any, *, runner_specs: Any = None) -> Any:
        return primary if runner_specs is not None else fallback

    monkeypatch.setattr("pipelines_v2.cli._registry_catalog", fake_registry)

    catalog, run_id = _resolve_workflow_metadata_catalog(
        types.SimpleNamespace(),
        runner_specs={},
        run_id=record.run_id,
        workflow=workflow,
        status="completed",
    )

    assert run_id == record.run_id
    assert catalog.identity() == fallback.identity()


def test_workflow_orchestrator_can_reuse_steps_from_explicit_workflow_catalog(tmp_path: Path) -> None:
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="alpha", labels={"class": "pos"}),
            Example(key="b", prompt="beta", labels={"class": "neg"}),
        ]
    )
    shared_catalog = FileCatalog(tmp_path / "shared_catalog")
    local_catalog = FileCatalog(tmp_path / "local_catalog")
    shared_store = LocalArtifactStore(tmp_path / "artifacts")
    source_runner = _FailOnceRunner(LocalRunner(artifacts=shared_store, catalog=shared_catalog))
    source_workflow = WorkflowSpec(
        name="explicit_workflow_catalog_reuse",
        steps=(
            WorkflowStep(
                name="capture",
                runner="capture",
                spec=CaptureSpec(
                    engine=ToyEngine(hidden_size=4, num_layers=2),
                    dataset=dataset,
                    sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0])],
                ),
            ),
        ),
    )

    first = WorkflowOrchestrator(runners={"capture": source_runner}).run(source_workflow)

    reuse_runner = _FailOnceRunner(LocalRunner(artifacts=shared_store, catalog=local_catalog))
    reused = WorkflowOrchestrator(
        runners={"capture": reuse_runner},
        workflow_catalog=shared_catalog,
    ).run(source_workflow, reuse_from_run_id=first.run_id)

    assert reused.step("capture").id == first.step("capture").id
    assert reuse_runner.calls.count("capture") == 0


def test_build_report_spec_from_run_uses_source_artifacts_without_rerunning_ancestors(tmp_path: Path) -> None:
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="alpha", labels={"class": "pos"}),
            Example(key="b", prompt="beta", labels={"class": "neg"}),
        ]
    )
    catalog = FileCatalog(tmp_path / "catalog")
    store = LocalArtifactStore(tmp_path / "artifacts")
    capture_runner = _FailOnceRunner(LocalRunner(artifacts=store, catalog=catalog))
    report_runner = _FailOnceRunner(LocalRunner(artifacts=store, catalog=catalog))
    runners = {"capture": capture_runner, "report": report_runner}
    workflow = WorkflowSpec(
        name="report_regeneration",
        steps=(
            WorkflowStep(
                name="capture",
                runner="capture",
                spec=CaptureSpec(
                    engine=ToyEngine(hidden_size=4, num_layers=2),
                    dataset=dataset,
                    sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0])],
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report",
                spec=ReportSpec(
                    template="summary",
                    output_dir=str(tmp_path / "reports"),
                    inputs=(StepRef("capture"),),
                ),
            ),
        ),
    )

    first = WorkflowOrchestrator(runners=runners, workflow_catalog=catalog).run(workflow)

    report_step = _resolve_report_step(workflow, step_name="report")
    regenerated = _build_report_spec_from_run(
        run=catalog.load_workflow_run(first.run_id or "") or WorkflowRunRecord(
            run_id="",
            workflow_name=workflow.name,
            workflow_hash=workflow.semantic_hash(),
            workflow_spec_hash=workflow.spec_hash(),
            workflow_payload=workflow.to_dict(),
            status="completed",
            started_at=utc_now_iso(),
        ),
        report_step=report_step,
        workflow_catalog=catalog,
        local_cache_root=None,
    )

    assert isinstance(regenerated, ReportSpec)
    assert not any(isinstance(value, StepRef) for value in regenerated.inputs)

    second = WorkflowOrchestrator(runners=runners, workflow_catalog=catalog).run(
        WorkflowSpec(
            name=workflow.name,
            steps=(WorkflowStep(name="report", runner="report", spec=regenerated),),
        ),
    )

    assert second.step("report").manifest().artifact_kind == "report"
    assert capture_runner.calls.count("capture") == 1
    assert report_runner.calls.count("report") == 2


def test_composite_catalog_prefers_local_workflow_step_reads_without_touching_remote(tmp_path: Path) -> None:
    local = FileCatalog(tmp_path / "catalog")
    run_id = "wr_test_local_preferred"
    local.record_workflow_run(
        WorkflowRunRecord(
            run_id=run_id,
            workflow_name="workflow",
            workflow_hash="hash",
            workflow_spec_hash="spec",
            workflow_payload={"kind": "workflow"},
            status="completed",
            started_at=utc_now_iso(),
            finished_at=utc_now_iso(),
        )
    )
    local.record_workflow_step(
        WorkflowStepRecord(
            run_id=run_id,
            workflow_hash="hash",
            workflow_step_key="hash.step",
            step_name="step",
            step_index=0,
            runner="capture",
            status="completed",
            step_semantic_hash="sem",
            step_spec_hash="spec",
            artifact_id="artifact_1",
            artifact_kind="patched_generation",
            input_artifact_refs=(),
            started_at=utc_now_iso(),
            finished_at=utc_now_iso(),
        )
    )

    class _FailingCatalog(_CatalogProbe):
        def list_workflow_steps(self, run_id: str) -> list[WorkflowStepRecord]:
            raise AssertionError("remote list_workflow_steps should not be called when local has records")

        def find_latest_reusable_step(
            self,
            *,
            step_name: str,
            step_semantic_hash: str,
            input_artifact_refs: tuple[str, ...],
        ) -> WorkflowStepRecord | None:
            raise AssertionError("remote find_latest_reusable_step should not be called when local has records")

    composite = CompositeCatalog((local, _FailingCatalog(kind="postgres", name="remote")))

    records = composite.list_workflow_steps(run_id)

    assert len(records) == 1
    assert records[0].step_name == "step"

    reusable = composite.find_latest_reusable_step(
        step_name="step",
        step_semantic_hash="sem",
        input_artifact_refs=(),
    )

    assert reusable is not None
    assert reusable.artifact_id == "artifact_1"


def test_postgres_catalog_records_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    import pipelines_v2.storage.postgres as postgres_storage

    executed: list[tuple[str, tuple[object, ...] | None]] = []
    committed = {"value": False}
    monkeypatch.setenv("XENON_DATABASE_URL", "postgresql://example/xenon")
    postgres_storage._SCHEMA_ENSURED_URLS.clear()

    class FakeCursor:
        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
            executed.append((" ".join(sql.split()), params))

    class FakeConnection:
        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            committed["value"] = True

    fake_psycopg = types.SimpleNamespace(connect=lambda url, **kwargs: FakeConnection())
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    manifest = ArtifactManifest(
        artifact_id="probe_123",
        artifact_kind="probe",
        schema_version=1,
        operation_spec_hash="abc123",
        operation_semantic_hash="abc123",
        created_at="2026-04-13T00:00:00+00:00",
        engine={},
        runner={"kind": "local"},
        input_artifact_refs=("capture_123",),
        example_coverage={"example_count": 4},
        storage_refs={"result": {"store": "local", "path": "/tmp/result.json"}},
        metadata={},
        workflow_context={
            "run_id": "run_123",
            "step_name": "probe",
            "workflow_step_key": "workflow_hash.probe",
        },
    )

    PostgresCatalog(source=PostgresSource.from_env("XENON_DATABASE_URL")).record_artifact(manifest)

    assert committed["value"] is True
    assert any("CREATE TABLE IF NOT EXISTS pipelines_v2_artifacts" in sql for sql, _ in executed)
    assert any("CREATE TABLE IF NOT EXISTS pipelines_v2_workflow_step_inputs" in sql for sql, _ in executed)
    assert any("pipelines_v2_workflow_steps_run_id_fkey" in sql for sql, _ in executed)
    insert = next(params for sql, params in executed if "INSERT INTO pipelines_v2_artifacts" in sql)
    assert insert is not None
    assert insert[0] == "probe_123"
    assert insert[6] == "run_123:probe"
    assert insert[7] == "run_123"
    assert insert[8] == "probe"
    assert insert[9] == "workflow_hash.probe"


def test_postgres_catalog_records_step_input_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    import pipelines_v2.storage.postgres as postgres_storage

    executed: list[tuple[str, tuple[object, ...] | None]] = []
    committed = {"value": False}
    monkeypatch.setenv("XENON_DATABASE_URL", "postgresql://example/xenon_step_inputs")
    postgres_storage._SCHEMA_ENSURED_URLS.clear()

    class FakeCursor:
        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
            executed.append((" ".join(sql.split()), params))

    class FakeConnection:
        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            committed["value"] = True

    fake_psycopg = types.SimpleNamespace(connect=lambda url, **kwargs: FakeConnection())
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    record = WorkflowStepRecord(
        run_id="run_123",
        workflow_hash="workflow_hash",
        workflow_step_key="workflow_hash.probe",
        step_name="probe",
        step_index=1,
        runner="analysis",
        status="completed",
        step_semantic_hash="sem_hash",
        step_spec_hash="spec_hash",
        input_artifact_refs=("capture_a", "labels_b"),
        artifact_id="probe_123",
        artifact_kind="probe",
        started_at="2026-04-13T00:00:00+00:00",
        finished_at="2026-04-13T00:01:00+00:00",
    )

    PostgresCatalog(source=PostgresSource.from_env("XENON_DATABASE_URL")).record_workflow_step(record)

    assert committed["value"] is True
    delete = next(
        params for sql, params in executed if "DELETE FROM pipelines_v2_workflow_step_inputs" in sql
    )
    assert delete == ("run_123:probe",)
    inserts = [
        params for sql, params in executed if "INSERT INTO pipelines_v2_workflow_step_inputs" in sql
    ]
    assert inserts == [
        ("run_123:probe", 0, "capture_a"),
        ("run_123:probe", 1, "labels_b"),
    ]


def test_postgres_catalog_finds_artifact_for_workflow_step_via_relational_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pipelines_v2.storage.postgres as postgres_storage

    executed: list[tuple[str, tuple[object, ...] | None]] = []
    monkeypatch.setenv("XENON_DATABASE_URL", "postgresql://example/xenon_find_artifact")
    postgres_storage._SCHEMA_ENSURED_URLS.clear()

    class FakeCursor:
        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
            executed.append((" ".join(sql.split()), params))

        def fetchone(self) -> None:
            return None

    class FakeConnection:
        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def cursor(self) -> FakeCursor:
            return FakeCursor()

    fake_psycopg = types.SimpleNamespace(connect=lambda url, **kwargs: FakeConnection())
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    result = PostgresCatalog(source=PostgresSource.from_env("XENON_DATABASE_URL")).find_artifact_for_workflow_step(
        run_id="run_123",
        workflow_step_key="workflow_hash.probe",
    )

    assert result is None
    select_sql, params = next(
        (sql, params)
        for sql, params in executed
        if "SELECT a.manifest FROM pipelines_v2_artifacts a" in sql
    )
    assert "SELECT workflow_step_id" in select_sql
    assert "a.produced_by_step_id = step.workflow_step_id" in select_sql
    assert params == (
        "run_123",
        "workflow_hash.probe",
        "run_123",
        "workflow_hash.probe",
        "run_123",
        "workflow_hash.probe",
    )


def test_postgres_catalog_ensures_schema_once_outside_query_transactions(monkeypatch: pytest.MonkeyPatch) -> None:
    import pipelines_v2.storage.postgres as postgres_storage

    monkeypatch.setenv("XENON_DATABASE_URL", "postgresql://example/xenon_schema_once")
    postgres_storage._SCHEMA_ENSURED_URLS.clear()

    ensure_calls: list[bool] = []
    connect_calls: list[bool] = []
    query_executes: list[str] = []

    class FakeCursor:
        def __init__(self, *, autocommit: bool) -> None:
            self.autocommit = autocommit

        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
            query_executes.append(" ".join(sql.split()))

        def fetchall(self) -> list[tuple[object, ...]]:
            return []

    class FakeConnection:
        def __init__(self, *, autocommit: bool) -> None:
            self.autocommit = autocommit

        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def cursor(self) -> FakeCursor:
            return FakeCursor(autocommit=self.autocommit)

    def fake_connect(url: str, **kwargs: object) -> FakeConnection:
        autocommit = bool(kwargs.get("autocommit", False))
        connect_calls.append(autocommit)
        return FakeConnection(autocommit=autocommit)

    def fake_ensure_schema(self: PostgresCatalog, cur: object) -> None:
        ensure_calls.append(getattr(cur, "autocommit"))

    monkeypatch.setitem(sys.modules, "psycopg", types.SimpleNamespace(connect=fake_connect))
    monkeypatch.setattr(PostgresCatalog, "_ensure_schema", fake_ensure_schema)

    catalog = PostgresCatalog(source=PostgresSource.from_env("XENON_DATABASE_URL"))
    first = catalog.list_workflow_runs()
    second = catalog.list_workflow_runs()

    assert first == []
    assert second == []
    assert ensure_calls == [True]
    assert connect_calls == [True, False, False]
    assert len([sql for sql in query_executes if "FROM pipelines_v2_workflow_runs" in sql]) == 2


def test_local_runner_pair_delta_produces_feature_and_label_artifact(tmp_path: Path) -> None:
    dataset = Dataset.from_examples(
        [
            Example(
                key="p1_aligned",
                prompt="alpha",
                labels={"pair_member": "aligned", "conflict_dimension": "size", "lexical_split": "train"},
                cases={"matched_pair_id": "p1"},
                case_key="p1",
            ),
            Example(
                key="p1_conflict",
                prompt="beta",
                labels={"pair_member": "strong_conflict", "conflict_dimension": "size", "lexical_split": "train"},
                cases={"matched_pair_id": "p1"},
                case_key="p1",
            ),
            Example(
                key="p2_aligned",
                prompt="gamma",
                labels={"pair_member": "aligned", "conflict_dimension": "activity", "lexical_split": "train"},
                cases={"matched_pair_id": "p2"},
                case_key="p2",
            ),
            Example(
                key="p2_conflict",
                prompt="delta",
                labels={"pair_member": "strong_conflict", "conflict_dimension": "activity", "lexical_split": "train"},
                cases={"matched_pair_id": "p2"},
                case_key="p2",
            ),
            Example(
                key="p3_aligned",
                prompt="epsilon",
                labels={"pair_member": "aligned", "conflict_dimension": "size", "lexical_split": "test"},
                cases={"matched_pair_id": "p3"},
                case_key="p3",
            ),
            Example(
                key="p3_conflict",
                prompt="zeta",
                labels={"pair_member": "strong_conflict", "conflict_dimension": "size", "lexical_split": "test"},
                cases={"matched_pair_id": "p3"},
                case_key="p3",
            ),
            Example(
                key="p4_aligned",
                prompt="eta",
                labels={"pair_member": "aligned", "conflict_dimension": "activity", "lexical_split": "test"},
                cases={"matched_pair_id": "p4"},
                case_key="p4",
            ),
            Example(
                key="p4_conflict",
                prompt="theta",
                labels={"pair_member": "strong_conflict", "conflict_dimension": "activity", "lexical_split": "test"},
                cases={"matched_pair_id": "p4"},
                case_key="p4",
            ),
        ],
        name="pair_delta_dataset",
    )
    capture_runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    capture = capture_runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=3, num_layers=2),
            dataset=dataset,
            sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0, 1], tokens=TokenSelector.last())],
        )
    )
    analysis_runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "analysis"))

    pair_delta = analysis_runner.run(
        PairDeltaSpec(
            feature=capture.feature("resid_last"),
            case=dataset.cases("matched_pair_id"),
            positive=dataset.labels("pair_member").equals("strong_conflict"),
            negative=dataset.labels("pair_member").equals("aligned"),
            labels={
                "conflict_dimension": dataset.labels("conflict_dimension"),
                "lexical_split": dataset.labels("lexical_split"),
            },
        )
    )

    delta_feature = pair_delta.feature("delta").load()
    assert delta_feature["kind"] == "residual"
    assert sorted(delta_feature["layers"]["0"]) == ["p1", "p2", "p3", "p4"]
    assert pair_delta.label("conflict_dimension").resolve_values() == {
        "p1": "size",
        "p2": "activity",
        "p3": "size",
        "p4": "activity",
    }

    probe = analysis_runner.run(
        ProbeSpec(
            feature=pair_delta.feature("delta"),
            labels=pair_delta.label("conflict_dimension"),
            split=pair_delta.label("lexical_split"),
            metrics=["accuracy", "balanced_accuracy"],
            baselines=["majority"],
        )
    )
    assert probe.summary()["split_mode"] == "fixed"


def test_feature_matrix_pooling_respects_selected_token_segment(tmp_path: Path) -> None:
    from pipelines_v2.operations.execute import _feature_matrices

    dataset = Dataset.from_examples(
        [
            Example(
                key="ex_a",
                prompt="prompt a",
                metadata={"token_sections": {"STRATEGY": [1, 2, 3], "SETTINGS": [4, 5]}},
            ),
            Example(
                key="ex_b",
                prompt="prompt b",
                metadata={"token_sections": {"STRATEGY": [1, 2, 3], "SETTINGS": [4, 5]}},
            ),
        ]
    )
    capture_runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    artifact = capture_runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=3, num_layers=1, sequence_length=6),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_full",
                    site="resid_post",
                    layers=[0],
                    tokens=TokenSelector.full_sequence(),
                )
            ],
        )
    )

    feature = artifact.feature("resid_full")
    payload = feature.load()
    ex_a_values = np.asarray(payload["layers"]["0"]["ex_a"]["values"], dtype=np.float32)

    mean_matrices, example_keys = _feature_matrices(
        feature,
        token_selector=TokenSelector.section("STRATEGY"),
        token_pooling=TokenPooling.mean(),
    )
    last_matrices, _ = _feature_matrices(
        feature,
        token_selector=TokenSelector.section("STRATEGY"),
        token_pooling=TokenPooling.last(),
    )

    assert example_keys == ["ex_a", "ex_b"]
    assert np.allclose(mean_matrices[0][0], ex_a_values[[1, 2, 3]].mean(axis=0))
    assert np.allclose(last_matrices[0][0], ex_a_values[3])


def test_residual_capture_can_pool_selected_tokens_before_persisting(tmp_path: Path) -> None:
    from pipelines_v2.operations.execute import _feature_matrices

    dataset = Dataset.from_examples(
        [
            Example(
                key="ex_a",
                prompt="prompt a",
                metadata={"token_sections": {"SPAN": [1, 2, 3]}},
            )
        ]
    )
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    engine = ToyEngine(hidden_size=3, num_layers=1, sequence_length=6)
    pooled_spec = CaptureSpec(
        engine=engine,
        dataset=dataset,
        sites=[
            ResidualSite(
                name="resid_span_mean",
                site="resid_post",
                layers=[0],
                tokens=TokenSelector.section("SPAN"),
                pooling=TokenPooling.mean(),
            )
        ],
    )
    restored_spec = CaptureSpec.from_dict(pooled_spec.to_dict())
    assert restored_spec.to_dict() == pooled_spec.to_dict()

    full = runner.run(
        CaptureSpec(
            engine=engine,
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_full",
                    site="resid_post",
                    layers=[0],
                    tokens=TokenSelector.full_sequence(),
                )
            ],
        )
    )
    pooled = runner.run(pooled_spec)

    full_values = np.asarray(full.feature("resid_full").load()["layers"]["0"]["ex_a"]["values"], dtype=np.float32)
    pooled_feature = pooled.feature("resid_span_mean")
    pooled_payload = pooled_feature.load()
    pooled_record = pooled_payload["layers"]["0"]["ex_a"]
    pooled_values = np.asarray(pooled_record["values"], dtype=np.float32)
    matrices, example_keys = _feature_matrices(
        pooled_feature,
        token_selector=TokenSelector.full_sequence(),
        token_pooling=TokenPooling.mean(),
    )

    assert example_keys == ["ex_a"]
    assert pooled_payload["pooling"] == {"kind": "mean"}
    assert pooled_record["tokens"] == [0]
    assert pooled_record["pooled"] is True
    assert pooled_record["pooling"] == {"kind": "mean"}
    assert pooled_record["pooled_token_count"] == 3
    assert pooled_record["token_sections"]["SPAN"] == [0]
    assert pooled_values.shape == (1, 3)
    assert np.allclose(pooled_values[0], full_values[[1, 2, 3]].mean(axis=0), atol=5e-4)
    assert np.allclose(matrices[0][0], pooled_values[0])


def test_capture_spec_rejects_section_selector_without_explicit_metadata_source() -> None:
    dataset = Dataset.from_examples(
        [
            Example(
                key="a",
                prompt="SYSTEM\nChoose.\n\nSTRATEGY\nBuy.\n\nSETTINGS\nLarge.\n",
            )
        ]
    )

    with pytest.raises(SpecValidationError, match="prompt_metadata_builder"):
        CaptureSpec(
            engine=ToyEngine(hidden_size=4, num_layers=2),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_strategy",
                    site="resid_post",
                    layers=[0],
                    tokens=TokenSelector.section("STRATEGY"),
                )
            ],
        )


def test_workflow_plan_reports_missing_section_metadata_builder_for_probe(tmp_path: Path) -> None:
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="SYSTEM\nChoose.\n\nSTRATEGY\nBuy.\n\nSETTINGS\nLarge.\n", labels={"class": "positive", "split": "train"}),
            Example(key="b", prompt="SYSTEM\nChoose.\n\nSTRATEGY\nHold.\n\nSETTINGS\nNone.\n", labels={"class": "negative", "split": "test"}),
        ]
    )
    workflow = WorkflowSpec(
        name="missing_section_metadata",
        steps=(
            WorkflowStep(
                name="capture",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=ToyEngine(hidden_size=4, num_layers=2),
                    dataset=dataset,
                    sites=[
                        ResidualSite(
                            name="resid_full",
                            site="resid_post",
                            layers=[0],
                            tokens=TokenSelector.full_sequence(),
                        )
                    ],
                ),
            ),
            WorkflowStep(
                name="probe",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("capture").feature("resid_full"),
                    labels=dataset.labels("class"),
                    split=dataset.labels("split"),
                    tokens=TokenSelector.section("STRATEGY"),
                    pooling=TokenPooling.mean(),
                    folds=2,
                ),
            ),
        ),
    )
    orchestrator = WorkflowOrchestrator(
        runners={
            "capture_gpu": LocalRunner(artifacts=LocalArtifactStore(tmp_path / "workflow_section_capture")),
            "analysis_cpu": LocalRunner(artifacts=LocalArtifactStore(tmp_path / "workflow_section_analysis")),
        }
    )

    plan = orchestrator.plan(workflow)
    probe_plan = next(step for step in plan.steps if step.name == "probe")

    assert probe_plan.execution.errors
    assert "prompt_metadata_builder" in probe_plan.execution.errors[0]
    with pytest.raises(SpecValidationError, match="prompt_metadata_builder"):
        orchestrator.run(workflow)


def test_label_predicate_resolves_deferred_dataset_once(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch_count = {"dataset": 0, "labels": 0}

    def fake_fetch(self: PostgresSource, **kwargs: Any) -> Dataset:
        del self, kwargs
        fetch_count["dataset"] += 1
        raise AssertionError("label predicate should not fetch full deferred dataset")

    def fake_fetch_label_values(self: PostgresSource, **kwargs: Any) -> Mapping[str, Any]:
        del self
        fetch_count["labels"] += 1
        assert kwargs["label_name"] == "class"
        return {"a": "positive", "b": "negative", "c": "positive"}

    monkeypatch.setattr(PostgresSource, "fetch_dataset", fake_fetch)
    monkeypatch.setattr(PostgresSource, "fetch_label_values", fake_fetch_label_values)

    dataset = Dataset.from_postgres(
        source=PostgresSource.from_env("XENON_DATABASE_URL"),
        table="public.capture_examples",
        prompt_column="prompt",
        example_key_column="example_id",
        label_columns=["class"],
    )

    predicate = dataset.labels("class").equals("positive")

    assert predicate.resolve_example_keys() == ["a", "c"]
    assert fetch_count == {"dataset": 0, "labels": 1}


def test_local_runner_label_map_spec_emits_derived_labels(tmp_path: Path) -> None:
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="alpha", labels={"strategy_family": "trade_size_force_large"}),
            Example(key="b", prompt="beta", labels={"strategy_family": "activity_force_trade"}),
            Example(key="c", prompt="gamma", labels={"strategy_family": "diversification_force_concentrate"}),
        ]
    )
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))

    artifact = runner.run(
        LabelMapSpec(
            source=dataset.labels("strategy_family"),
            output_name="conflict_dimension",
            mapping={
                "trade_size_force_large": "size",
                "activity_force_trade": "action",
                "diversification_force_concentrate": "asset",
            },
        )
    )

    assert artifact.label("conflict_dimension").resolve_values() == {
        "a": "size",
        "b": "action",
        "c": "asset",
    }


def test_transform_spec_emits_project_specific_behavior_labels(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    artifact_id = "capture_generation_labels"
    store.make_artifact_dir(artifact_id)
    generations_ref = store.write_json(
        artifact_id,
        "generations.json",
        [
            {"example_key": "a", "text": '{"action":"buy","asset":"ALPHA","size":"large"}'},
            {"example_key": "b", "text": '{"action":"observe","asset":"NONE","size":"none"}'},
        ],
    )
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_kind="capture",
        schema_version=1,
        operation_spec_hash="abc",
        operation_semantic_hash="abc",
        created_at="2026-04-13T00:00:00+00:00",
        engine={"kind": "toy"},
        runner={"kind": "local"},
        input_artifact_refs=(),
        example_coverage={"example_count": 2},
        storage_refs={"generations": generations_ref},
        metadata={},
        workflow_context={},
    )
    capture_artifact = CaptureArtifact(_manifest=manifest, store=store)
    dataset = Dataset.from_examples(
        [
            Example(
                key="a",
                prompt="alpha",
                labels={
                    "workflow_expected_action": "buy",
                    "workflow_expected_asset": "ALPHA",
                    "workflow_expected_size": "large",
                    "strategy_expected_action": "buy",
                    "strategy_expected_asset": "ALPHA",
                    "strategy_expected_size": "large",
                    "setting_expected_action": "observe",
                    "setting_expected_asset": "NONE",
                    "setting_expected_size": "none",
                },
            ),
            Example(
                key="b",
                prompt="beta",
                labels={
                    "workflow_expected_action": "observe",
                    "workflow_expected_asset": "NONE",
                    "workflow_expected_size": "none",
                    "strategy_expected_action": "buy",
                    "strategy_expected_asset": "ALPHA",
                    "strategy_expected_size": "medium",
                    "setting_expected_action": "observe",
                    "setting_expected_asset": "NONE",
                    "setting_expected_size": "none",
                },
            ),
        ]
    )
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "analysis"))

    behavior_labels = runner.run(
        TransformSpec(
            builder=TransformBuilder.from_function(_test_behavior_transform, local_python_sources=("tests",)),
            inputs={
                "generations": capture_artifact,
                "workflow_expected_action": dataset.labels("workflow_expected_action"),
                "workflow_expected_asset": dataset.labels("workflow_expected_asset"),
                "workflow_expected_size": dataset.labels("workflow_expected_size"),
                "strategy_expected_action": dataset.labels("strategy_expected_action"),
                "strategy_expected_asset": dataset.labels("strategy_expected_asset"),
                "strategy_expected_size": dataset.labels("strategy_expected_size"),
                "setting_expected_action": dataset.labels("setting_expected_action"),
                "setting_expected_asset": dataset.labels("setting_expected_asset"),
                "setting_expected_size": dataset.labels("setting_expected_size"),
            },
        )
    )

    assert behavior_labels.label("generated_action").resolve_values() == {"a": "buy", "b": "observe"}
    assert behavior_labels.label("matches_workflow_expected").resolve_values() == {"a": True, "b": True}
    assert behavior_labels.label("source_following_side").resolve_values() == {"a": "strategy", "b": "setting"}


def test_prompt_metadata_builder_derives_explicit_section_spans_for_vllm_capture() -> None:
    from pipelines_v2.engine.prompt_metadata import resolve_prompt_metadata, token_sections_from_metadata

    rendered = (
        "SYSTEM\nChoose exactly one action.\n\n"
        "STRATEGY\nGo all in on the best setup.\n\n"
        "SETTINGS\nTrade size: 5/5. Use the largest size.\n"
    )
    offsets = [(index, index + 1) for index in range(len(rendered))]
    builder = PromptMetadataBuilder.from_function(_test_prompt_section_metadata, local_python_sources=("tests",))

    metadata = resolve_prompt_metadata(
        metadata={},
        rendered_prompt=rendered,
        builder=builder,
    )
    sections = token_sections_from_metadata(
        metadata=metadata,
        offsets=offsets,
        require_sections=True,
        allow_char_spans=True,
    )

    assert set(sections) == {"SETTINGS", "STRATEGY"}
    assert sections["STRATEGY"]
    assert sections["SETTINGS"]


def test_capture_rebases_token_sections_after_token_slicing(tmp_path: Path) -> None:
    from pipelines_v2.operations.execute import _feature_matrices

    dataset = Dataset.from_examples(
        [
            Example(
                key="ex_a",
                prompt="prompt a",
                metadata={"token_sections": {"STRATEGY": [1, 2, 3], "SETTINGS": [4, 5]}},
            ),
            Example(
                key="ex_b",
                prompt="prompt b",
                metadata={"token_sections": {"STRATEGY": [1, 2, 3], "SETTINGS": [4, 5]}},
            ),
        ]
    )
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    artifact = runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=3, num_layers=1, sequence_length=6),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_strategy",
                    site="resid_post",
                    layers=[0],
                    tokens=TokenSelector.section("STRATEGY"),
                )
            ],
        )
    )

    feature = artifact.feature("resid_strategy")
    payload = feature.load()
    ex_a = payload["layers"]["0"]["ex_a"]

    assert ex_a["tokens"] == [1, 2, 3]
    assert ex_a["token_sections"] == {"STRATEGY": [0, 1, 2]}

    matrices, example_keys = _feature_matrices(
        feature,
        token_selector=TokenSelector.section("STRATEGY"),
        token_pooling=TokenPooling.mean(),
    )

    assert example_keys == ["ex_a", "ex_b"]
    assert np.allclose(matrices[0][0], np.asarray(ex_a["values"], dtype=np.float32).mean(axis=0))

    with pytest.raises(SpecValidationError, match="section 'SETTINGS'"):
        _feature_matrices(
            feature,
            token_selector=TokenSelector.section("SETTINGS"),
            token_pooling=TokenPooling.mean(),
        )


def test_function_builder_from_function_is_not_tied_to_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    working_dir = tmp_path / "detached"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)

    builder = PromptMetadataBuilder.from_function(
        _test_prompt_section_metadata,
        local_python_sources=("tests",),
    )
    rendered = (
        "SYSTEM\nChoose exactly one action.\n\n"
        "STRATEGY\nBuy ALPHA immediately.\n\n"
        "SETTINGS\nUse the largest size.\n"
    )

    metadata = builder.build(rendered)

    assert builder.import_path == "tests.test_pipelines_v2_basics:_test_prompt_section_metadata"
    assert builder.local_python_sources == ("tests",)
    assert "token_sections" in metadata


def test_local_report_runner_materializes_output_dir(tmp_path: Path) -> None:
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))

    report = runner.run(
        ReportSpec(
            template="phase_04_summary",
            output_dir=str(tmp_path / "reports"),
            inputs=[],
        )
    )

    published = report.manifest().metadata["published_report"]
    assert Path(published["output_dir"]).exists()
    assert Path(published["report_path"]).exists()
    assert Path(published["summary_path"]).exists()
    assert Path(published["asset_manifest_path"]).exists()
    assert Path(published["assets_dir"]).exists()
    assert Path(published["tables_dir"]).exists()
    assert report.manifest().storage_refs["report"]["path"].endswith("report.md")


def test_local_report_runner_downloads_only_direct_operation_inputs(tmp_path: Path) -> None:
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="alpha", labels={"class": "pos"}, case_key="c1"),
            Example(key="b", prompt="beta", labels={"class": "neg"}, case_key="c2"),
            Example(key="c", prompt="gamma", labels={"class": "pos"}, case_key="c3"),
            Example(key="d", prompt="delta", labels={"class": "neg"}, case_key="c4"),
        ],
        name="report_download_dataset",
    )
    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=4, num_layers=2),
            dataset=dataset,
            sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0, 1])],
        ),
        workflow_context=WorkflowStepContext(
            run_id="wr_test",
            workflow_name="report_download",
            workflow_hash="workflow_hash",
            workflow_spec_hash="workflow_spec_hash",
            step_name="capture",
            step_index=0,
            runner="capture",
            step_semantic_hash="capture_semantic_hash",
            step_spec_hash="capture_spec_hash",
        ),
    )
    probe = runner.run(
        ProbeSpec(
            feature=capture.feature("resid_last"),
            labels=dataset.labels("class"),
            folds=2,
            baselines=["majority"],
        ),
        workflow_context=WorkflowStepContext(
            run_id="wr_test",
            workflow_name="report_download",
            workflow_hash="workflow_hash",
            workflow_spec_hash="workflow_spec_hash",
            step_name="probe",
            step_index=1,
            runner="analysis",
            step_semantic_hash="probe_semantic_hash",
            step_spec_hash="probe_spec_hash",
        ),
    )
    report = runner.run(
        ReportSpec(
            template="download_test",
            output_dir=str(tmp_path / "reports"),
            inputs=[capture, probe],
        ),
        workflow_context=WorkflowStepContext(
            run_id="wr_test",
            workflow_name="report_download",
            workflow_hash="workflow_hash",
            workflow_spec_hash="workflow_spec_hash",
            step_name="report",
            step_index=2,
            runner="report",
            step_semantic_hash="report_semantic_hash",
            step_spec_hash="report_spec_hash",
        ),
    )

    published = report.manifest().metadata["published_report"]
    results_dir = Path(published["results_dir"])
    report_json_path = Path(published["report_json_path"])
    manifest_path = Path(published["asset_manifest_path"])
    tables_dir = Path(published["tables_dir"])
    assets_dir = Path(published["assets_dir"])
    probe_results_path = results_dir / "probe_results.json"
    capture_results_path = results_dir / "capture_results.json"
    probe_table_path = tables_dir / "probe.json"
    probe_chart_path = assets_dir / "probe" / "balanced_accuracy_by_layer.png"

    assert results_dir.exists()
    assert probe_results_path.exists()
    assert not capture_results_path.exists()
    assert manifest_path.exists()
    assert probe_table_path.exists()
    assert probe_chart_path.exists()
    assert published["downloaded_results"] == [
        {
            "name": "probe",
            "artifact_id": probe.id,
            "artifact_kind": "probe",
            "path": str(probe_results_path),
            "source": {
                "store": probe.manifest().storage_refs["result"]["store"],
                "path": probe.manifest().storage_refs["result"]["path"],
                "format": probe.manifest().storage_refs["result"]["format"],
                "bytes": probe.manifest().storage_refs["result"]["bytes"],
            },
        }
    ]

    with probe_results_path.open("r", encoding="utf-8") as f:
        downloaded_probe_payload = json.load(f)
    assert downloaded_probe_payload == probe.result()

    with report_json_path.open("r", encoding="utf-8") as f:
        report_payload = json.load(f)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_payload = json.load(f)
    capture_input = report_payload["inputs"][0]
    probe_input = report_payload["inputs"][1]
    assert "downloaded_result_path" not in capture_input
    assert probe_input["downloaded_result_path"] == str(probe_results_path)
    assert capture_input.get("assets") in (None, [])
    assert probe_input["table_path"] == "tables/probe.json"
    assert probe_input["assets"]
    assert probe_input["headline_metrics"]["best_layer"] is not None
    assert report_payload["summary"]["figures"]
    assert report_payload["summary"]["tables"]["probe"]["path"] == "tables/probe.json"
    assert report_payload["summary"]["step_summaries"]["probe"]["kind"] == "probe_result"
    assert manifest_payload["figures"]
    assert manifest_payload["tables"]["probe"]["path"] == "tables/probe.json"
    assert manifest_payload["unsupported_inputs"] == []


def test_report_spec_includes_richer_artifact_input_details(tmp_path: Path) -> None:
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="alpha", labels={"class": "pos"}, case_key="c1"),
            Example(key="b", prompt="beta", labels={"class": "neg"}, case_key="c2"),
            Example(key="c", prompt="gamma", labels={"class": "pos"}, case_key="c3"),
            Example(key="d", prompt="delta", labels={"class": "neg"}, case_key="c4"),
        ],
        name="report_detail_dataset",
    )
    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=4, num_layers=2),
            dataset=dataset,
            sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0, 1])],
        )
    )
    probe = runner.run(
        ProbeSpec(
            feature=capture.feature("resid_last"),
            labels=dataset.labels("class"),
            folds=2,
            baselines=["majority"],
        )
    )
    report = runner.run(
        ReportSpec(
            template="detail_test",
            inputs=[capture, probe],
        )
    )

    payload = report.result()
    capture_input = payload["inputs"][0]
    probe_input = payload["inputs"][1]

    assert capture_input["name"] == capture.id
    assert capture_input["artifact_kind"] == "capture"
    assert capture_input["feature_names"] == ["resid_last"]
    assert capture_input["primary_output"]["name"] == "manifest"
    assert capture_input["storage"]["features"]["count"] == 1
    assert capture_input["example_coverage"]["dataset_name"] == "report_detail_dataset"

    assert probe_input["name"] == probe.id
    assert probe_input["artifact_kind"] == "probe"
    assert probe_input["summary"]["example_count"] == 4
    assert probe_input["primary_output"]["name"] == "result"
    assert probe_input["storage"]["result"]["path"].endswith("result.json")


def test_report_spec_includes_modal_volume_mappings_for_inputs() -> None:
    from pipelines_v2.operations.execution.common import summarize_report_input
    from pipelines_v2.storage.modal import ModalVolumeStore

    manifest = ArtifactManifest(
        artifact_id="probe_test_modal",
        artifact_kind="probe",
        schema_version=1,
        operation_spec_hash="spec_hash",
        operation_semantic_hash="semantic_hash",
        created_at="2026-04-15T00:00:00+00:00",
        engine={},
        runner={
            "kind": "modal",
            "runtime_app_id": "ap-test",
            "resources": {
                "volumes": [
                    {"name": "xenon-models", "mount_path": "/models"},
                ]
            },
        },
        input_artifact_refs=(),
        example_coverage={"materialized": True, "example_count": 4, "example_keys": ["a", "b", "c", "d"]},
        storage_refs={
            "result": {
                "store": "modal_volume",
                "name": "xenon-data",
                "path": "/data/artifacts/test/probe_test_modal/result.json",
                "format": "json",
                "bytes": 42,
            }
        },
        metadata={},
        workflow_context={"step_name": "family_identity_residual", "run_id": "wr_test"},
    )
    artifact = OperationArtifact(
        _manifest=manifest,
        store=ModalVolumeStore(name="xenon-data", root="/data/artifacts/test"),
    )

    payload = summarize_report_input(artifact)

    assert payload["runtime"]["volume_mappings"] == [
        {"name": "xenon-data", "mount_path": "/data", "role": "artifact_store"},
        {"name": "xenon-models", "mount_path": "/models", "role": "runner_resource"},
    ]


def test_workflow_result_payload_does_not_localize_non_report_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="alpha", labels={"class": "pos"}, case_key="c1"),
            Example(key="b", prompt="beta", labels={"class": "neg"}, case_key="c2"),
            Example(key="c", prompt="gamma", labels={"class": "pos"}, case_key="c3"),
            Example(key="d", prompt="delta", labels={"class": "neg"}, case_key="c4"),
        ],
        name="cli_payload_dataset",
    )
    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=4, num_layers=2),
            dataset=dataset,
            sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0, 1])],
        )
    )
    probe = runner.run(
        ProbeSpec(
            feature=capture.feature("resid_last"),
            labels=dataset.labels("class"),
            folds=2,
            baselines=["majority"],
        )
    )

    def _fail_localize(self: Any) -> Any:
        raise AssertionError("cli should not localize non-report artifacts while rendering workflow results")

    monkeypatch.setattr(type(probe), "localize", _fail_localize)

    payload = _workflow_result_payload(
        "cli_payload_test",
        WorkflowResult(
            run_id="wr_test",
            workflow_hash="wh_test",
            step_results={"probe": probe},
        ),
    )

    assert payload["steps"]["probe"]["artifact_id"] == probe.id
    assert payload["steps"]["probe"]["location"].endswith("result.json")


def test_phase5_style_specs_execute_over_residual_router_and_text(tmp_path: Path) -> None:
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    dataset = _make_phase5_like_dataset()
    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=6, num_layers=2, num_experts=8, sequence_length=1),
            dataset=dataset,
            sites=[
                ResidualSite(name="residual_prompt_eos", site="resid_post", layers=[0, 1], tokens=TokenSelector.last()),
                MoERoutingSite(
                    name="router_prompt_eos",
                    layers=[0, 1],
                    tokens=TokenSelector.last(),
                    record=[RoutingRecord.gate_logits(dtype="float16")],
                ),
            ],
        )
    )

    residual_transfer = runner.run(
        TransferProbeSpec(
            feature=capture.feature("residual_prompt_eos"),
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_pair_id"),
            cohort_by=dataset.labels("family_group"),
            cohort_values=("size", "activity"),
            metrics=("balanced_accuracy", "auroc"),
        )
    )
    router_transfer = runner.run(
        TransferProbeSpec(
            feature=capture.feature("router_prompt_eos"),
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_pair_id"),
            cohort_by=dataset.labels("family_group"),
            cohort_values=("size", "activity"),
            metrics=("balanced_accuracy", "auroc"),
        )
    )
    text_baseline = runner.run(
        TextBaselineSpec(
            text=dataset.labels("user_text"),
            labels=dataset.labels("strategy_family"),
            group_by=dataset.cases("matched_pair_id"),
            model="countvectorizer_logreg",
            metrics=("balanced_accuracy", "auroc"),
        )
    )
    residualized = runner.run(
        ResidualizedProbeSpec(
            feature=capture.feature("residual_prompt_eos"),
            labels=dataset.labels("conflict_present"),
            residualize_against=dataset.labels("strategy_family"),
            group_by=dataset.cases("matched_pair_id"),
            metrics=("balanced_accuracy", "auroc"),
        )
    )
    geometry_pca = runner.run(
        GeometrySpec(
            feature=capture.feature("residual_prompt_eos"),
            method="pca",
            layers=[0],
            color_by={"family": dataset.labels("strategy_family")},
            subset=dataset.labels("conflict_present").equals(True),
            normalize="rms_per_row",
            components=2,
        )
    )
    geometry_lda = runner.run(
        GeometrySpec(
            feature=capture.feature("residual_prompt_eos"),
            method="lda",
            layers=[1],
            label=dataset.labels("strategy_family"),
            normalize="rms_per_row",
            components=2,
        )
    )

    residual_transfer_payload = residual_transfer.result()
    router_transfer_payload = router_transfer.result()
    text_baseline_payload = text_baseline.result()
    residualized_payload = residualized.result()
    geometry_pca_payload = geometry_pca.result()
    geometry_lda_payload = geometry_lda.result()

    assert residual_transfer_payload["kind"] == "transfer_probe_result"
    assert residual_transfer_payload["layers"][0]["cross_cohort_transfer"]
    assert "test_predictions" not in residual_transfer_payload["layers"][0]["cross_cohort_transfer"]["size_to_activity"]
    assert router_transfer_payload["kind"] == "transfer_probe_result"
    assert "size_to_activity" in router_transfer_payload["layers"][0]["cross_cohort_transfer"]
    assert text_baseline_payload["kind"] == "text_baseline_result"
    assert text_baseline_payload["mode"] == "grouped_cv"
    assert "test_predictions" not in text_baseline_payload["results"]["grouped_cv"]
    assert residualized_payload["kind"] == "residualized_probe_result"
    assert residualized_payload["layers"][0]["family_subspace_rank"] >= 1
    assert residualized_payload["summary"]["best_residualized_balanced_accuracy"] is not None
    assert residualized_payload["summary"]["max_nuisance_accuracy_on_null_training_fit"] is not None
    assert residualized_payload["summary"]["residualization_diagnostic"] in {
        "nuisance_reduced",
        "nuisance_still_decodable",
    }
    assert geometry_pca_payload["kind"] == "geometry_result"
    assert geometry_pca_payload["layers"][0]["example_count"] == 4
    assert geometry_lda_payload["kind"] == "geometry_result"
    assert geometry_lda_payload["layers"][0]["label_name"] == "strategy_family"


def test_probe_grouped_cv_persists_prediction_rows_when_requested(tmp_path: Path) -> None:
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    dataset = _make_phase5_like_dataset()
    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=6, num_layers=1, sequence_length=1),
            dataset=dataset,
            sites=[ResidualSite(name="residual_prompt_eos", site="resid_post", layers=[0], tokens=TokenSelector.last())],
        )
    )

    probe = runner.run(
        ProbeSpec(
            feature=capture.feature("residual_prompt_eos"),
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_pair_id"),
            metrics=("balanced_accuracy", "auroc"),
            persist_predictions=True,
        )
    )

    payload = probe.result()
    layer_payload = payload["layers"][0]
    prediction_rows = layer_payload["test_predictions"]

    assert layer_payload["test_prediction_count"] == len(dataset.example_keys())
    assert len(prediction_rows) == len(dataset.example_keys())
    assert {row["example_key"] for row in prediction_rows} == set(dataset.example_keys())

    first = prediction_rows[0]
    assert first["evaluation_kind"] == "probe"
    assert first["layer"] == 0
    assert first["split_mode"] in {"group_kfold", "stratified_group_kfold", "stratified_kfold"}
    assert isinstance(first["fold_index"], int)
    assert isinstance(first["correct"], bool)
    assert first["binary_outcome"] in {"true_positive", "false_positive", "false_negative", "true_negative"}
    assert "positive_class_probability" in first


def test_transfer_probe_cross_cohort_persists_prediction_rows_when_requested(tmp_path: Path) -> None:
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    dataset = _make_phase5_like_dataset()
    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=6, num_layers=1, sequence_length=1),
            dataset=dataset,
            sites=[ResidualSite(name="residual_prompt_eos", site="resid_post", layers=[0], tokens=TokenSelector.last())],
        )
    )

    transfer = runner.run(
        TransferProbeSpec(
            feature=capture.feature("residual_prompt_eos"),
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_pair_id"),
            cohort_by=dataset.labels("family_group"),
            cohort_values=("size", "activity"),
            metrics=("balanced_accuracy", "auroc"),
            persist_predictions=True,
        )
    )

    payload = transfer.result()
    layer_payload = payload["layers"][0]
    cross_payload = layer_payload["cross_cohort_transfer"]["size_to_activity"]
    within_payload = layer_payload["within_cohort_baseline"]["activity"]

    assert cross_payload["test_prediction_count"] == 4
    assert len(cross_payload["test_predictions"]) == 4
    assert within_payload["test_prediction_count"] == 4

    first = cross_payload["test_predictions"][0]
    assert first["evaluation_kind"] == "cross_cohort_transfer"
    assert first["layer"] == 0
    assert first["train_cohort"] == "size"
    assert first["test_cohort"] == "activity"
    assert first["split_mode"] == "cross_transfer"
    assert "positive_class_probability" in first


def test_text_baseline_grouped_cv_persists_prediction_rows_when_requested(tmp_path: Path) -> None:
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    dataset = _make_phase5_like_dataset()

    baseline = runner.run(
        TextBaselineSpec(
            text=dataset.labels("user_text"),
            labels=dataset.labels("strategy_family"),
            group_by=dataset.cases("matched_pair_id"),
            model="countvectorizer_logreg",
            metrics=("balanced_accuracy", "auroc"),
            persist_predictions=True,
        )
    )

    payload = baseline.result()
    grouped = payload["results"]["grouped_cv"]
    prediction_rows = grouped["test_predictions"]

    assert grouped["test_prediction_count"] == len(dataset.example_keys())
    assert len(prediction_rows) == len(dataset.example_keys())
    assert {row["example_key"] for row in prediction_rows} == set(dataset.example_keys())

    first = prediction_rows[0]
    assert first["evaluation_kind"] == "grouped_cv"
    assert first["model"] == "countvectorizer_logreg"
    assert first["split_mode"] in {"group_kfold", "stratified_group_kfold", "stratified_kfold"}
    assert isinstance(first["fold_index"], int)
    assert isinstance(first["correct"], bool)
    assert "class_probabilities" in first


def test_probe_rows_align_subset_dataset_to_feature_rows(tmp_path: Path) -> None:
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    dataset = _make_phase5_like_dataset()
    subset = dataset.select(
        keys=[
            "size_conflict_train",
            "size_aligned_train",
            "activity_conflict_test",
            "activity_aligned_test",
        ]
    )
    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=6, num_layers=2, sequence_length=1),
            dataset=dataset,
            sites=[ResidualSite(name="residual_prompt_eos", site="resid_post", layers=[0, 1], tokens=TokenSelector.last())],
        )
    )

    probe = runner.run(
        ProbeSpec(
            feature=capture.feature("residual_prompt_eos"),
            rows=subset,
            labels=subset.labels("conflict_present"),
            split=subset.labels("strategy_lexical_split"),
            train_values=("train",),
            test_values=("test",),
            metrics=("balanced_accuracy",),
        )
    )

    payload = probe.result()
    assert payload["kind"] == "probe_result"
    assert payload["summary"]["example_count"] == 4
    assert probe.manifest().example_coverage["example_keys"] == sorted(subset.example_keys())


def test_probe_rows_must_be_subset_of_feature_rows(tmp_path: Path) -> None:
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    dataset = _make_phase5_like_dataset()
    bad_rows = Dataset.from_examples(
        [
            Example(
                key="missing_key",
                prompt="SYSTEM\nChoose one.\n",
                labels={"conflict_present": True, "strategy_lexical_split": "test"},
                case_key="missing_pair",
            )
        ],
        name="bad_rows",
    )
    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=6, num_layers=2, sequence_length=1),
            dataset=dataset,
            sites=[ResidualSite(name="residual_prompt_eos", site="resid_post", layers=[0, 1], tokens=TokenSelector.last())],
        )
    )

    with pytest.raises(SpecValidationError, match="rows requested 1 example keys not present"):
        runner.run(
            ProbeSpec(
                feature=capture.feature("residual_prompt_eos"),
                rows=bad_rows,
                labels=bad_rows.labels("conflict_present"),
                split=bad_rows.labels("strategy_lexical_split"),
                train_values=("train",),
                test_values=("test",),
                metrics=("balanced_accuracy",),
            )
        )


def test_workflow_orchestrator_plan_errors_on_mixed_analysis_datasets_without_rows(tmp_path: Path) -> None:
    dataset = _make_phase5_like_dataset()
    arbitration_dataset = dataset.select(
        keys=[
            "size_conflict_train",
            "size_aligned_train",
            "activity_conflict_train",
            "activity_aligned_train",
        ]
    )
    shared_store = LocalArtifactStore(tmp_path / "artifacts")
    orchestrator = WorkflowOrchestrator(
        runners={
            "capture": LocalRunner(artifacts=shared_store),
            "analysis": LocalRunner(artifacts=shared_store),
        }
    )
    workflow = WorkflowSpec(
        name="mixed_rows_without_rows",
        steps=(
            WorkflowStep(
                name="capture",
                runner="capture",
                spec=CaptureSpec(
                    engine=ToyEngine(hidden_size=4, num_layers=2),
                    dataset=dataset,
                    sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0, 1])],
                ),
            ),
            WorkflowStep(
                name="probe",
                runner="analysis",
                spec=ProbeSpec(
                    feature=StepRef("capture").feature("resid_last"),
                    labels=arbitration_dataset.labels("conflict_present"),
                    split=arbitration_dataset.labels("strategy_lexical_split"),
                    train_values=("train",),
                    test_values=("test",),
                    metrics=("balanced_accuracy",),
                ),
            ),
        ),
    )

    plan = orchestrator.plan(workflow)

    assert len(plan.steps) == 2
    assert any("Add rows=..." in message for message in plan.steps[1].execution.errors)
    with pytest.raises(SpecValidationError, match="Add rows=..."):
        plan.steps[1].execution.validate()


def test_pipelines_v2_modal_smoke_file_builders() -> None:
    module_path = Path("scripts/pipelines_v2_orchestrator_smoke.py")
    spec = importlib.util.spec_from_file_location("pipelines_v2_orchestrator_smoke", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    dataset = module.build_dataset()
    runner_specs = module.build_runner_specs()
    workflow = module.build_workflow(dataset)

    assert dataset.name == "pipelines_v2_modal_capture_probe_smoke"
    assert dataset.example_keys() == [
        "ex_pos_train",
        "ex_neg_train",
        "ex_pos_test",
        "ex_neg_test",
    ]
    assert [step.name for step in workflow.steps] == ["capture", "probe"]
    assert workflow.steps[0].runner == "capture_gpu"
    assert workflow.steps[1].runner == "analysis_cpu"
    assert workflow.steps[0].spec.engine.identity()["kind"] == "vllm"
    assert workflow.steps[0].spec.prompt_metadata_builder is not None
    local_sources = workflow.steps[0].spec.runtime_spec().local_python_sources
    assert "." not in local_sources
    assert "pipelines_v2" in local_sources
    assert "scripts" in local_sources
    assert workflow.steps[0].spec.prompt_metadata_builder.import_path == (
        "scripts.pipelines_v2_orchestrator_smoke:build_prompt_metadata"
    )
    assert workflow.steps[1].spec.tokens.kind == "section"
    assert workflow.steps[1].spec.pooling.kind == "mean"
    assert sorted(runner_specs) == ["analysis_cpu", "capture_gpu"]
    assert runner_specs["capture_gpu"].resources.gpu == "L4"
    assert runner_specs["analysis_cpu"].resources.cpu == 4


def test_pipelines_v2_activation_patch_smoke_file_builders() -> None:
    module_path = Path("scripts/pipelines_v2_activation_patch_smoke.py")
    spec = importlib.util.spec_from_file_location("pipelines_v2_activation_patch_smoke", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    dataset = module.build_dataset()
    runner_specs = module.build_runner_specs()
    workflow = module.build_workflow(dataset)

    assert dataset.name == "pipelines_v2_activation_patch_smoke"
    assert dataset.example_keys() == [
        "pair1_target_buy",
        "pair1_donor_sell",
        "pair2_target_sell",
        "pair2_donor_buy",
    ]
    assert sorted(runner_specs) == ["analysis_cpu", "capture_gpu"]
    assert [step.name for step in workflow.steps] == [
        "capture_prompt_residual",
        "learn_strategy_subspace",
        "baseline_targets",
        "lesion_generated_tokens",
        "validate_compiled_patch_stats",
        "compare_patch_runs",
    ]
    assert workflow.steps[0].runner == "capture_gpu"
    assert workflow.steps[1].runner == "analysis_cpu"
    assert workflow.steps[2].runner == "capture_gpu"
    assert workflow.steps[3].runner == "capture_gpu"
    assert workflow.steps[4].runner == "analysis_cpu"
    assert workflow.steps[5].runner == "analysis_cpu"
    patch = workflow.steps[3].spec.patch
    assert patch.application.kind == "every_token"
    assert patch.application.include_prompt is False
    assert patch.application.include_decode is True
    assert patch.target_tokens.kind == "section"
    assert workflow.steps[4].spec.builder.import_path == (
        "scripts.pipelines_v2_activation_patch_smoke:validate_compiled_patch_smoke"
    )


def test_pipelines_v2_router_layer_probe_uses_compile_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = Path("scripts/pipelines_v2_router_layer_probe.py")
    spec = importlib.util.spec_from_file_location("pipelines_v2_router_layer_probe", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    dataset = module.build_dataset()
    runner_specs = module.build_runner_specs()
    workflow = module.build_workflow(dataset)
    runners = _build_runners(
        argparse.Namespace(
            file=str(module_path),
            catalog_postgres_env=None,
            local_catalog_root=None,
        ),
        runner_specs,
    )

    assert dataset.name == "router_layer_probe"
    assert [step.name for step in workflow.steps] == ["capture_router_probe"]
    assert sorted(runner_specs) == ["capture_gpu"]
    assert runner_specs["capture_gpu"].resources.env == {}
    assert runner_specs["capture_gpu"].resources.volumes == (
        ModalVolumeMount(
            name="xenon-models",
            mount_path="/models",
        ),
    )
    assert runners["capture_gpu"].resources.env == {"VLLM_CACHE_ROOT": "/models"}
    assert runners["capture_gpu"].resources.volumes == (
        ModalVolumeMount(
            name="xenon-models",
            mount_path="/models",
            create_if_missing=True,
            commit_on_success=True,
        ),
    )
    assert workflow.steps[0].spec.engine.identity()["kind"] == "vllm"
    assert workflow.steps[0].spec.engine.model_id == "/models/Qwen/Qwen3-30B-A3B"
    assert workflow.steps[0].spec.generation.enabled is True
    assert workflow.steps[0].spec.generation.max_tokens == 256
    assert workflow.steps[0].spec.engine.enforce_eager is False


def test_workflow_orchestrator_inlines_transform_steps_into_downstream_inputs(tmp_path: Path) -> None:
    store = LocalArtifactStore(root=tmp_path / "artifacts")
    runner = LocalRunner(resources=LocalResources(), artifacts=store)
    orchestrator = WorkflowOrchestrator({"local": runner})

    workflow = WorkflowSpec(
        name="inline_transform_smoke",
        steps=(
            WorkflowStep(
                name="seed",
                runner="local",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(_inline_transform_seed),
                    inputs={"value": 41},
                    inline=True,
                ),
            ),
            WorkflowStep(
                name="consumer",
                runner="local",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(_inline_transform_consume),
                    inputs={"seed": StepRef("seed")},
                ),
            ),
        ),
    )

    result = orchestrator.run(workflow)

    assert result.step("seed").result()["value"] == 41
    assert result.step("consumer").result()["value"] == 42


def test_pipelines_v2_cli_workflow_plan_loads_python_file(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = pipelines_v2_cli_main(
        [
            "workflow",
            "plan",
            "--file",
            "scripts/pipelines_v2_orchestrator_smoke.py",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["workflow"] == "pipelines_v2_modal_capture_probe_smoke"
    assert [step["name"] for step in payload["steps"]] == ["capture", "probe"]
    assert all(step["errors"] == [] for step in payload["steps"])


def test_load_python_workflow_file_passes_dataset_into_optional_dataset_builder(tmp_path: Path) -> None:
    workflow_file = tmp_path / "workflow_optional_dataset.py"
    workflow_file.write_text(
        "\n".join(
            [
                "from pipelines_v2.api import CaptureSpec, Dataset, Example, ResidualSite, ToyEngine, WorkflowSpec, WorkflowStep",
                "",
                "def build_dataset():",
                "    return Dataset.from_examples([Example(key='limited', prompt='alpha')], name='limited_dataset')",
                "",
                "def build_workflow(dataset=None):",
                "    if dataset is None:",
                "        dataset = Dataset.from_examples([Example(key='fallback', prompt='beta')], name='fallback_dataset')",
                "    return WorkflowSpec(",
                "        name='optional_dataset_builder',",
                "        steps=(",
                "            WorkflowStep(",
                "                name='capture',",
                "                runner='capture_gpu',",
                "                spec=CaptureSpec(",
                "                    engine=ToyEngine(hidden_size=4, num_layers=2),",
                "                    dataset=dataset,",
                "                    sites=[ResidualSite(name='resid_last', site='resid_post', layers=[0])],",
                "                ),",
                "            ),",
                "        ),",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )

    dataset, workflow, _ = load_python_workflow_file(path=workflow_file)

    assert dataset.name == "limited_dataset"
    assert workflow.steps[0].spec.dataset.name == "limited_dataset"


def test_pipelines_v2_cli_tracks_runs_in_local_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    workflow_file = _write_cli_local_workflow_file(tmp_path)
    monkeypatch.setenv("XENON_HOME", str(tmp_path / ".xenon"))
    monkeypatch.setenv("XENON_NEON_DATABASE_URL", "")

    exit_code = pipelines_v2_cli_main(
        [
            "workflow",
            "run",
            "--file",
            str(workflow_file),
        ]
    )
    run_payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert run_payload["workflow"] == "cli_local_workflow"
    run_id = run_payload["run_id"]

    exit_code = pipelines_v2_cli_main(
        [
            "workflow",
            "runs",
            "--file",
            str(workflow_file),
        ]
    )
    runs_payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert [record["run_id"] for record in runs_payload["runs"]] == [run_id]
    assert runs_payload["runs"][0]["status"] == "completed"

    exit_code = pipelines_v2_cli_main(
        [
            "workflow",
            "show",
            "--run-id",
            run_id,
        ]
    )
    show_payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert show_payload["run"]["run_id"] == run_id
    assert [step["step_name"] for step in show_payload["steps"]] == ["capture", "probe", "report"]
    assert show_payload["progress"]["run"]["status"] == "completed"
    assert show_payload["progress"]["steps"]["report"]["status"] == "completed"


def test_pipelines_v2_cli_workflow_run_logging_emits_progress_to_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workflow_file = _write_cli_local_workflow_file(tmp_path)
    monkeypatch.setenv("XENON_HOME", str(tmp_path / ".xenon"))
    monkeypatch.setenv("XENON_NEON_DATABASE_URL", "")

    exit_code = pipelines_v2_cli_main(
        [
            "workflow",
            "run",
            "--file",
            str(workflow_file),
            "--logging",
            "INFO",
        ]
    )
    captured = capsys.readouterr()
    run_payload = json.loads(captured.out)

    assert exit_code == 0
    assert run_payload["workflow"] == "cli_local_workflow"
    assert "workflow started name=cli_local_workflow" in captured.err
    assert "workflow progress run=" in captured.err
    assert "step completed name=report" in captured.err
    assert "workflow completed name=cli_local_workflow" in captured.err


def test_workflow_result_payload_serializes_inline_transform_results() -> None:
    payload = _workflow_result_payload(
        "inline_workflow",
        WorkflowResult(
            run_id="wr_inline",
            workflow_hash="hash_inline",
            step_results={
                "seed": InlineOperationArtifact(payload={"kind": "transform_result", "summary": {"value": 3}})
            },
        ),
    )

    assert payload["steps"]["seed"]["artifact_id"] is None
    assert payload["steps"]["seed"]["artifact_kind"] == "inline_transform"
    assert payload["steps"]["seed"]["summary"] == {"value": 3}


def test_execute_artifact_operation_transform_does_not_import_readout_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pipelines_v2.operations.execution as execution_module

    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "sklearn" or name.startswith("sklearn."):
            raise AssertionError(f"unexpected readout dependency import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    execution_module = importlib.reload(execution_module)

    result = execution_module.execute_artifact_operation(
        TransformSpec(builder=TransformBuilder.from_function(_inline_transform_seed), inputs={"value": 7})
    )

    assert result.payload["kind"] == "transform_result"
    assert result.payload["value"] == 7


def test_pipelines_v2_cli_rerun_step_and_from_step_use_prior_run_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workflow_file = _write_cli_local_workflow_file(tmp_path)
    monkeypatch.setenv("XENON_HOME", str(tmp_path / ".xenon"))
    monkeypatch.setenv("XENON_NEON_DATABASE_URL", "")

    exit_code = pipelines_v2_cli_main(
        [
            "workflow",
            "run",
            "--file",
            str(workflow_file),
        ]
    )
    initial_payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    initial_run_id = initial_payload["run_id"]
    initial_capture_id = initial_payload["steps"]["capture"]["artifact_id"]
    initial_probe_id = initial_payload["steps"]["probe"]["artifact_id"]
    initial_report_id = initial_payload["steps"]["report"]["artifact_id"]

    exit_code = pipelines_v2_cli_main(
        [
            "workflow",
            "rerun-step",
            "--file",
            str(workflow_file),
            "--run-id",
            initial_run_id,
            "--step",
            "report",
        ]
    )
    rerun_step_payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    rerun_step_run_id = rerun_step_payload["run_id"]
    assert rerun_step_payload["steps"]["capture"]["artifact_id"] == initial_capture_id
    assert rerun_step_payload["steps"]["probe"]["artifact_id"] == initial_probe_id
    assert rerun_step_payload["steps"]["report"]["artifact_id"] != initial_report_id

    exit_code = pipelines_v2_cli_main(
        [
            "workflow",
            "rerun-from-step",
            "--file",
            str(workflow_file),
            "--run-id",
            initial_run_id,
            "--step",
            "probe",
        ]
    )
    rerun_from_payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    rerun_from_run_id = rerun_from_payload["run_id"]
    assert rerun_from_payload["steps"]["capture"]["artifact_id"] == initial_capture_id
    assert rerun_from_payload["steps"]["probe"]["artifact_id"] != initial_probe_id
    assert rerun_from_payload["steps"]["report"]["artifact_id"] != initial_report_id

    catalog = FileCatalog(Path(tmp_path / ".xenon" / "pipelines_v2" / "catalog"))
    rerun_step_run = catalog.load_workflow_run(rerun_step_run_id)
    rerun_from_run = catalog.load_workflow_run(rerun_from_run_id)
    assert rerun_step_run is not None
    assert rerun_step_run.parent_run_id == initial_run_id
    assert rerun_from_run is not None
    assert rerun_from_run.parent_run_id == initial_run_id

    rerun_step_records = {record.step_name: record for record in catalog.list_workflow_steps(rerun_step_run_id)}
    assert rerun_step_records["capture"].status == "reused"
    assert rerun_step_records["probe"].status == "reused"
    assert rerun_step_records["report"].status == "completed"

    rerun_from_records = {record.step_name: record for record in catalog.list_workflow_steps(rerun_from_run_id)}
    assert rerun_from_records["capture"].status == "reused"
    assert rerun_from_records["probe"].status == "completed"
    assert rerun_from_records["report"].status == "completed"


def test_pipelines_v2_cli_rerun_step_supports_mixed_shared_and_local_catalog_runners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workflow_file = _write_cli_mixed_catalog_workflow_file(tmp_path)
    monkeypatch.setenv("XENON_HOME", str(tmp_path / ".xenon"))

    exit_code = pipelines_v2_cli_main(
        [
            "workflow",
            "run",
            "--file",
            str(workflow_file),
        ]
    )
    initial_payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    initial_run_id = initial_payload["run_id"]
    initial_capture_id = initial_payload["steps"]["capture"]["artifact_id"]
    initial_probe_id = initial_payload["steps"]["probe"]["artifact_id"]
    initial_report_id = initial_payload["steps"]["report"]["artifact_id"]

    exit_code = pipelines_v2_cli_main(
        [
            "workflow",
            "rerun-step",
            "--file",
            str(workflow_file),
            "--run-id",
            initial_run_id,
            "--step",
            "report",
        ]
    )
    rerun_payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert rerun_payload["steps"]["capture"]["artifact_id"] == initial_capture_id
    assert rerun_payload["steps"]["probe"]["artifact_id"] == initial_probe_id
    assert rerun_payload["steps"]["report"]["artifact_id"] != initial_report_id


def test_pipelines_v2_cli_loads_dotenv_before_loading_workflow_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workflow_file = tmp_path / "dotenv_workflow.py"
    workflow_file.write_text(
        "\n".join(
            [
                "import os",
                "",
                "from pipelines_v2.api import Dataset, Example, WorkflowSpec",
                "",
                "def build_dataset():",
                "    required = os.environ['TEST_CLI_DOTENV_VALUE']",
                "    return Dataset.from_examples(",
                "        [Example(key='a', prompt='alpha', labels={'class': required}, case_key='c1')],",
                "        name='dotenv_cli_workflow_dataset',",
                "    )",
                "",
                "def build_workflow(dataset=None):",
                "    return WorkflowSpec(name='dotenv_cli_workflow', steps=())",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("TEST_CLI_DOTENV_VALUE=loaded_from_env_file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TEST_CLI_DOTENV_VALUE", raising=False)

    exit_code = pipelines_v2_cli_main(
        [
            "workflow",
            "plan",
            "--file",
            str(workflow_file),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["workflow"] == "dotenv_cli_workflow"
    assert os.environ["TEST_CLI_DOTENV_VALUE"] == "loaded_from_env_file"


def test_runner_spec_round_trips_from_dict() -> None:
    payload = {
        "kind": "modal",
        "resources": {
            "gpu": "L4",
            "cpu": None,
            "memory_mb": None,
            "timeout_seconds": 3600,
            "secrets": [{"name": "xenon-db", "env_vars": ["XENON_DATABASE_URL"]}],
            "volumes": [{"name": "xenon-models", "mount_path": "/models"}],
        },
        "artifacts": {
            "kind": "modal_volume",
            "name": "xenon-data",
            "root": "/data/artifacts/pipelines_v2_smoke",
            "transfer_policy": {
                "allow_large_transfer": False,
                "max_download_bytes": 64 * 1024 * 1024,
            },
        },
        "catalog": {"kind": "none"},
    }

    spec = runner_spec_from_dict(payload)
    restored = spec.to_dict()

    assert restored["kind"] == "modal"
    assert restored["resources"]["gpu"] == "L4"
    assert restored["resources"]["secrets"] == [{"name": "xenon-db", "env_vars": ["XENON_DATABASE_URL"]}]
    assert restored["artifacts"]["name"] == "xenon-data"
    assert spec.to_runner().identity()["kind"] == "modal"


def test_patched_generation_spec_requires_target_token_sections_for_section_selectors() -> None:
    dataset = make_toy_dataset()

    with pytest.raises(SpecValidationError, match="target-side token-section metadata source"):
        PatchedGenerationSpec(
            engine=ToyEngine(),
            dataset=dataset,
            patch=InterchangePatch(
                activation_bank="placeholder_bank",
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.section("BODY"),
            ),
            pair_by=dataset.cases("case_key"),
            target_when=dataset.labels("class").equals("positive"),
            donor_when=dataset.labels("class").equals("negative"),
            generation=GenerationSpec(enabled=True, max_tokens=2),
        )


def test_patch_and_generation_specs_round_trip() -> None:
    dataset = make_toy_dataset()
    patch = InterchangePatch(
        activation_bank="placeholder_bank",
        write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
        target_tokens=TokenSelector.full_sequence(),
    )
    generation = GenerationRunSpec(
        engine=ToyEngine(),
        dataset=dataset,
        select_when=dataset.labels("class").equals("positive"),
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )
    patched = PatchedGenerationSpec(
        engine=ToyEngine(),
        dataset=dataset,
        patch=patch,
        pair_by=dataset.cases("case_key"),
        target_when=dataset.labels("class").equals("positive"),
        donor_when=dataset.labels("class").equals("negative"),
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )
    comparison = PatchComparisonSpec(
        baseline="baseline_ref",
        variants={"main": "patched_ref"},
        row_evaluator=TransformBuilder.from_function(_patch_comparison_row_evaluator),
    )

    restored_patch = ActivationPatchSpec.from_dict(patch.to_dict())
    restored_generation = GenerationRunSpec.from_dict(generation.to_dict())
    restored_patched = PatchedGenerationSpec.from_dict(patched.to_dict())
    restored_comparison = PatchComparisonSpec.from_dict(comparison.to_dict())

    assert restored_patch.write_site.site == "resid_post"
    assert restored_patch.write_site.layers == (0,)
    assert restored_patch.target_tokens.kind == "full_sequence"
    assert restored_generation.generation.enabled is True
    assert restored_generation.select_when.op == "equals"
    assert restored_patched.patch.write_site.layers == (0,)
    assert restored_patched.generation.max_tokens == 2
    assert sorted(restored_comparison.variants) == ["main"]


def test_activation_patch_request_helper_rebases_query_positions_and_drops_absolute_positions() -> None:
    from types import SimpleNamespace

    from pipelines_v2.engine.vllm.activation_patch_request_worker import ActivationPatchRequestHelper

    helper = ActivationPatchRequestHelper()
    helper.process_new_reqs(
        [
            SimpleNamespace(
                req_id="req-1",
                sampling_params=SimpleNamespace(
                    extra_args={
                        "activation_patch_spec": {
                            "target_layers": [24],
                            "target_positions": [16, 17],
                            "donor_example_key": "donor-1",
                            "donor_positions": [9, 10],
                            "case_key": "pair-1",
                            "control_name": "",
                        }
                    }
                ),
            )
        ]
    )

    helper.build_step_specs(
        input_batch=SimpleNamespace(
            req_ids=["req-1"],
            num_computed_tokens_cpu=[12],
            num_prompt_tokens=[26],
        ),
        num_scheduled_tokens=[14],
    )

    assert len(helper.current_step_specs) == 1
    payload = helper.current_step_specs[0]["patch_spec"]
    assert payload["query_positions"] == [4, 5]
    assert payload["donor_positions"] == [9, 10]
    assert "target_positions" not in payload


def test_activation_patch_request_helper_matches_suffixed_request_ids() -> None:
    from types import SimpleNamespace

    from pipelines_v2.engine.vllm.activation_patch_request_worker import ActivationPatchRequestHelper

    helper = ActivationPatchRequestHelper()
    helper.process_new_reqs(
        [
            SimpleNamespace(
                req_id="req-1",
                sampling_params=SimpleNamespace(
                    extra_args={
                        "activation_patch_spec": {
                            "target_layers": [24],
                            "target_positions": [16],
                            "donor_example_key": "donor-1",
                            "donor_positions": [9],
                            "case_key": "pair-1",
                            "control_name": "",
                        }
                    }
                ),
            )
        ]
    )

    helper.build_step_specs(
        input_batch=SimpleNamespace(
            req_ids=["req-1-abc"],
            num_computed_tokens_cpu=[12],
            num_prompt_tokens=[26],
        ),
        num_scheduled_tokens=[14],
    )

    assert len(helper.current_step_specs) == 1
    payload = helper.current_step_specs[0]["patch_spec"]
    assert payload["query_positions"] == [4]
    assert payload["donor_positions"] == [9]


def test_activation_patch_request_helper_rebases_subspace_query_positions_without_donor_positions() -> None:
    from types import SimpleNamespace

    from pipelines_v2.engine.vllm.activation_patch_request_worker import ActivationPatchRequestHelper

    helper = ActivationPatchRequestHelper()
    helper.process_new_reqs(
        [
            SimpleNamespace(
                req_id="req-1",
                sampling_params=SimpleNamespace(
                    extra_args={
                        "activation_patch_spec": {
                            "operator": "project_out",
                            "target_layers": [24],
                            "target_positions": [16, 17],
                            "source_layer_map": {"24": 24},
                            "component_indices_by_layer": {"24": [0, 1]},
                            "strength": 1.0,
                        }
                    }
                ),
            )
        ]
    )

    helper.build_step_specs(
        input_batch=SimpleNamespace(
            req_ids=["req-1"],
            num_computed_tokens_cpu=[12],
            num_prompt_tokens=[26],
        ),
        num_scheduled_tokens=[14],
    )

    assert len(helper.current_step_specs) == 1
    payload = helper.current_step_specs[0]["patch_spec"]
    assert payload["query_positions"] == [4, 5]


def test_activation_patch_request_helper_rebases_residual_path_target_read_positions_per_chunk() -> None:
    from types import SimpleNamespace

    from pipelines_v2.engine.vllm.activation_patch_request_worker import ActivationPatchRequestHelper

    helper = ActivationPatchRequestHelper()
    helper.process_new_reqs(
        [
            SimpleNamespace(
                req_id="req-1",
                sampling_params=SimpleNamespace(
                    extra_args={
                        "activation_patch_spec": {
                            "operator": "residual_path",
                            "target_layers": [24],
                            "target_positions": [11, 13, 15],
                            "donor_example_key": "donor-1",
                            "donor_positions": [21, 23, 25],
                            "target_read_positions": [31, 33, 35],
                            "transport": "delta",
                            "path_edges": [{"source_layer": 4, "write_layer": 24, "weight": 1.0}],
                            "case_key": "pair-1",
                        }
                    }
                ),
            )
        ]
    )

    helper.build_step_specs(
        input_batch=SimpleNamespace(
            req_ids=["req-1"],
            num_computed_tokens_cpu=[12],
            num_prompt_tokens=[20],
        ),
        num_scheduled_tokens=[2],
    )

    assert len(helper.current_step_specs) == 1
    payload = helper.current_step_specs[0]["patch_spec"]
    assert payload["query_positions"] == [1]
    assert payload["donor_positions"] == [23]
    assert payload["target_read_positions"] == [33]
    assert payload["covered_abs_positions"] == [13]


def test_patch_application_roundtrips_and_plan_blocks_unimplemented_modes() -> None:
    from pipelines_v2.api import (
        AddDirectionPatch,
        Dataset,
        Example,
        GenerationSpec,
        PatchApplication,
        PatchedGenerationSpec,
        ResidualInterventionSite,
        VLLMEngine,
    )
    from pipelines_v2.operations.interventions.runtime import patched_generation_plan_errors

    patch = AddDirectionPatch(
        direction="direction-artifact",
        write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
        application=PatchApplication.every_token(),
    )
    restored = AddDirectionPatch.from_dict(patch.to_dict())
    assert restored.application.kind == "every_token"
    assert restored.application.include_prompt is True
    assert restored.application.include_decode is True

    dataset = Dataset.from_examples((Example(key="ex", prompt="prompt"),))
    spec = PatchedGenerationSpec(
        engine=VLLMEngine(model_id="dummy", enable_prefix_caching=False),
        dataset=dataset,
        patch=AddDirectionPatch(
            direction="direction-artifact",
            write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
            application=PatchApplication.trigger_word("unsafe"),
        ),
        generation=GenerationSpec(enabled=True, max_tokens=1),
    )
    assert patched_generation_plan_errors(spec) == [
        "PatchApplication.trigger_word is not implemented for PatchedGenerationSpec yet"
    ]
    spec = PatchedGenerationSpec(
        engine=VLLMEngine(model_id="dummy", enable_prefix_caching=False),
        dataset=dataset,
        patch=AddDirectionPatch(
            direction="direction-artifact",
            write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
            application=PatchApplication.probe_activated(probe="probe-artifact", threshold=0.5),
        ),
        generation=GenerationSpec(enabled=True, max_tokens=1),
    )
    assert patched_generation_plan_errors(spec) == [
        "PatchApplication.probe_activated is not implemented for PatchedGenerationSpec yet"
    ]


def test_patch_application_every_token_blocks_paired_operators_in_plan() -> None:
    from pipelines_v2.api import (
        Dataset,
        Example,
        GenerationSpec,
        InterchangePatch,
        PatchApplication,
        PatchedGenerationSpec,
        ResidualInterventionSite,
        VLLMEngine,
    )
    from pipelines_v2.operations.interventions.runtime import patched_generation_plan_errors

    dataset = Dataset.from_examples(
        (
            Example(key="target", prompt="target", labels={"role": "target"}, case_key="case"),
            Example(key="donor", prompt="donor", labels={"role": "donor"}, case_key="case"),
        )
    )
    spec = PatchedGenerationSpec(
        engine=VLLMEngine(model_id="dummy", enable_prefix_caching=False),
        dataset=dataset,
        patch=InterchangePatch(
            activation_bank="activation-bank",
            write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
            application=PatchApplication.every_token(),
        ),
        pair_by=dataset.cases("case_key"),
        target_when=dataset.labels("role").equals("target"),
        donor_when=dataset.labels("role").equals("donor"),
        generation=GenerationSpec(enabled=True, max_tokens=1),
    )

    errors = patched_generation_plan_errors(spec)
    assert len(errors) == 1
    assert "only supported for unpaired patch operators" in errors[0]


def test_activation_patch_request_helper_every_token_covers_prefill_and_decode() -> None:
    from types import SimpleNamespace

    from pipelines_v2.engine.vllm.activation_patch_request_worker import ActivationPatchRequestHelper

    helper = ActivationPatchRequestHelper()
    helper.process_new_reqs(
        [
            SimpleNamespace(
                req_id="req-1",
                sampling_params=SimpleNamespace(
                    extra_args={
                        "activation_patch_spec": {
                            "operator": "project_out",
                            "target_layers": [24],
                            "target_policy": {
                                "kind": "every_token",
                                "include_prompt": True,
                                "include_decode": True,
                                "config": {},
                            },
                            "source_layer_map": {"24": 24},
                            "component_indices_by_layer": {"24": [0]},
                        }
                    }
                ),
            )
        ]
    )

    helper.build_step_specs(
        input_batch=SimpleNamespace(
            req_ids=["req-1"],
            num_computed_tokens_cpu=[12],
            num_prompt_tokens=[20],
        ),
        num_scheduled_tokens=[5],
    )
    assert len(helper.current_step_specs) == 1
    payload = helper.current_step_specs[0]["patch_spec"]
    assert payload["query_span"] == [0, 5]
    assert payload["covered_abs_spans"] == [[12, 17]]
    assert payload["phase_counts"] == {"prompt": 5, "decode": 0}
    assert payload["rowwise"] is True
    assert "target_positions" not in payload

    helper.build_step_specs(
        input_batch=SimpleNamespace(
            req_ids=["req-1"],
            num_computed_tokens_cpu=[20],
            num_prompt_tokens=[20],
        ),
        num_scheduled_tokens=[1],
    )
    assert len(helper.current_step_specs) == 1
    payload = helper.current_step_specs[0]["patch_spec"]
    assert payload["query_span"] == [0, 1]
    assert payload["covered_abs_spans"] == [[20, 21]]
    assert payload["phase_counts"] == {"prompt": 0, "decode": 1}


def test_activation_patch_request_helper_every_token_rebases_mixed_batch_offsets() -> None:
    from types import SimpleNamespace

    from pipelines_v2.engine.vllm.activation_patch_request_worker import ActivationPatchRequestHelper

    helper = ActivationPatchRequestHelper()
    helper.process_new_reqs(
        [
            SimpleNamespace(
                req_id="req-2",
                sampling_params=SimpleNamespace(
                    extra_args={
                        "activation_patch_spec": {
                            "operator": "add_direction",
                            "target_layers": [7],
                            "target_policy": {
                                "kind": "every_token",
                                "include_prompt": True,
                                "include_decode": True,
                                "config": {},
                            },
                            "source_layer_map": {"7": 7},
                        }
                    }
                ),
            )
        ]
    )

    helper.build_step_specs(
        input_batch=SimpleNamespace(
            req_ids=["unpatched", "req-2"],
            num_computed_tokens_cpu=[0, 20],
            num_prompt_tokens=[10, 20],
        ),
        num_scheduled_tokens=[3, 1],
    )

    assert len(helper.current_step_specs) == 1
    assert helper.current_step_specs[0]["patch_spec"]["query_span"] == [3, 4]
    assert helper.current_step_specs[0]["query_span"] == [3, 4]


def test_static_activation_patch_request_helper_still_skips_decode_steps() -> None:
    from types import SimpleNamespace

    from pipelines_v2.engine.vllm.activation_patch_request_worker import ActivationPatchRequestHelper

    helper = ActivationPatchRequestHelper()
    helper.process_new_reqs(
        [
            SimpleNamespace(
                req_id="req-1",
                sampling_params=SimpleNamespace(
                    extra_args={
                        "activation_patch_spec": {
                            "target_layers": [24],
                            "target_positions": [16],
                            "donor_example_key": "donor-1",
                            "donor_positions": [9],
                            "case_key": "pair-1",
                        }
                    }
                ),
            )
        ]
    )

    helper.build_step_specs(
        input_batch=SimpleNamespace(
            req_ids=["req-1"],
            num_computed_tokens_cpu=[26],
            num_prompt_tokens=[26],
        ),
        num_scheduled_tokens=[1],
    )

    assert helper.current_step_specs == []


def test_every_token_project_out_is_rowwise_not_span_mean() -> None:
    from types import SimpleNamespace

    import torch

    from pipelines_v2.engine.vllm.patching.apply import patch_hidden_states_for_layer

    model = SimpleNamespace(
        _v2_activation_patch_subspace={
            0: {
                "mean": torch.zeros((2,), dtype=torch.float32),
                "scale": torch.ones((2,), dtype=torch.float32),
                "safe_scale": torch.ones((2,), dtype=torch.float32),
                "components": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                "named_components": {},
            }
        }
    )
    hidden = torch.tensor([[1.0, 1.0], [2.0, 0.0]], dtype=torch.float32)
    patched, stats = patch_hidden_states_for_layer(
        hidden,
        owner_model=model,
        layer_idx=0,
        batch_spec={
            "req_id": "req-1",
            "patch_spec": {
                "operator": "project_out",
                "target_layers": [0],
                "query_span": [0, 2],
                "rowwise": True,
                "source_layer_map": {"0": 0},
                "component_indices_by_layer": {"0": [0]},
                "strength": 1.0,
                "target_policy": {
                    "kind": "every_token",
                    "include_prompt": True,
                    "include_decode": True,
                    "config": {},
                },
            },
        },
    )

    assert torch.allclose(patched, torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
    assert stats is not None
    assert stats["rowwise"] is True
    assert stats["token_count"] == 2


def test_collect_patch_stats_matches_short_request_ids() -> None:
    from types import SimpleNamespace

    from pipelines_v2.engine.vllm.activation_patch_core import collect_patch_stats

    model = SimpleNamespace(
        _v2_activation_patch_stats_by_req={
            "2-b0ccd374": {
                24: {
                    "layer": 24,
                    "status": "ok",
                    "token_count": 1,
                }
            }
        }
    )

    stats = collect_patch_stats(model, req_id="2")

    assert stats == {
        24: {
            "layer": 24,
            "status": "ok",
            "token_count": 1,
        }
    }


def test_record_patch_stats_merges_coverage_spans() -> None:
    from types import SimpleNamespace

    from pipelines_v2.engine.vllm.patching.state import _record_patch_stats

    model = SimpleNamespace(_v2_activation_patch_stats_by_req={})

    _record_patch_stats(
        model,
        req_id="req-1",
        layer_idx=4,
        stats={
            "layer": 4,
            "status": "ok",
            "covered_abs_spans": [[10, 14]],
            "covered_abs_tokens": 4,
            "target_abs_tokens": 8,
            "coverage_fraction": 0.5,
        },
    )
    _record_patch_stats(
        model,
        req_id="req-1",
        layer_idx=4,
        stats={
            "layer": 4,
            "status": "ok",
            "covered_abs_spans": [[14, 18]],
            "covered_abs_tokens": 4,
            "target_abs_tokens": 8,
            "coverage_fraction": 0.5,
        },
    )

    assert model._v2_activation_patch_stats_by_req["req-1"][4]["covered_abs_spans"] == [[10, 18]]
    assert model._v2_activation_patch_stats_by_req["req-1"][4]["covered_abs_tokens"] == 8
    assert model._v2_activation_patch_stats_by_req["req-1"][4]["coverage_fraction"] == 1.0


def test_record_patch_stats_merges_residual_path_chunk_scalars_without_last_chunk_overwrite() -> None:
    from types import SimpleNamespace

    from pipelines_v2.engine.vllm.patching.state import _record_patch_stats

    model = SimpleNamespace(_v2_activation_patch_stats_by_req={})

    _record_patch_stats(
        model,
        req_id="req-1",
        layer_idx=4,
        stats={
            "layer": 4,
            "status": "ok",
            "operator": "residual_path",
            "token_count": 2,
            "query_positions": [0, 1],
            "donor_positions": [10, 11],
            "target_read_positions": [20, 21],
            "covered_abs_spans": [[10, 12]],
            "covered_abs_tokens": 2,
            "target_abs_tokens": 4,
            "coverage_fraction": 0.5,
            "delta_norm_raw": 3.0,
            "replace_alpha": 0.0,
        },
    )
    _record_patch_stats(
        model,
        req_id="req-1",
        layer_idx=4,
        stats={
            "layer": 4,
            "status": "ok",
            "operator": "residual_path",
            "token_count": 2,
            "query_positions": [0, 1],
            "donor_positions": [12, 13],
            "target_read_positions": [22, 23],
            "covered_abs_spans": [[20, 22]],
            "covered_abs_tokens": 2,
            "target_abs_tokens": 4,
            "coverage_fraction": 0.5,
            "delta_norm_raw": 4.0,
            "replace_alpha": 0.0,
        },
    )

    merged = model._v2_activation_patch_stats_by_req["req-1"][4]
    assert merged["chunk_count"] == 2
    assert len(merged["chunk_stats"]) == 2
    assert merged["token_count"] == 4
    assert merged["delta_norm_raw"] == pytest.approx(5.0)
    assert merged["covered_abs_spans"] == [[10, 12], [20, 22]]
    assert merged["covered_abs_tokens"] == 4
    assert merged["coverage_fraction"] == 1.0
    assert "query_positions" not in merged
    assert "donor_positions" not in merged
    assert "target_read_positions" not in merged


def test_record_patch_stats_keeps_chunk_stats_for_multichunk_subspace_rows() -> None:
    from types import SimpleNamespace

    from pipelines_v2.engine.vllm.patching.state import _record_patch_stats

    model = SimpleNamespace(_v2_activation_patch_stats_by_req={})

    _record_patch_stats(
        model,
        req_id="req-1",
        layer_idx=4,
        stats={
            "layer": 4,
            "status": "ok",
            "operator": "project_out",
            "strength": 1.0,
            "token_count": 2,
            "query_positions": [0, 1],
            "covered_abs_spans": [[10, 12]],
            "covered_abs_tokens": 2,
            "target_abs_tokens": 4,
            "coverage_fraction": 0.5,
            "delta_norm_raw": 1.0,
            "selected_coeff_before": [0.5],
            "selected_coeff_after": [0.0],
            "selected_component_count": 1,
        },
    )
    _record_patch_stats(
        model,
        req_id="req-1",
        layer_idx=4,
        stats={
            "layer": 4,
            "status": "ok",
            "operator": "project_out",
            "strength": 1.0,
            "token_count": 2,
            "query_positions": [0, 1],
            "covered_abs_spans": [[12, 14]],
            "covered_abs_tokens": 2,
            "target_abs_tokens": 4,
            "coverage_fraction": 0.5,
            "delta_norm_raw": 2.0,
            "selected_coeff_before": [0.25],
            "selected_coeff_after": [0.0],
            "selected_component_count": 1,
        },
    )

    merged = model._v2_activation_patch_stats_by_req["req-1"][4]
    assert merged["chunk_count"] == 2
    assert len(merged["chunk_stats"]) == 2
    assert merged["token_count"] == 4
    assert merged["covered_abs_spans"] == [[10, 14]]
    assert merged["covered_abs_tokens"] == 4
    assert merged["coverage_fraction"] == 1.0
    assert "delta_norm_raw" not in merged
    assert "selected_coeff_before" not in merged
    assert "selected_coeff_after" not in merged


def test_harvest_batch_patch_stats_residual_path_includes_runtime_coverage() -> None:
    from types import SimpleNamespace

    import torch

    from pipelines_v2.engine.vllm.activation_patch_core import harvest_batch_patch_stats

    model = SimpleNamespace(
        _v2_activation_patch_stats_by_req={},
        _v2_activation_patch_batch_tensor_stats={
            4: {
                "valid": torch.tensor([1], dtype=torch.int32),
                "scalars": torch.tensor([[1.25, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                "coeff_before": torch.zeros((1, 1), dtype=torch.float32),
                "coeff_after": torch.zeros((1, 1), dtype=torch.float32),
            }
        },
    )
    batch_specs = [
        {
            "req_id": "req-1",
            "patch_spec": {
                "operator": "residual_path",
                "target_layers": [4],
                "query_positions": [0, 1],
                "target_abs_positions": [10, 11],
                "covered_abs_positions": [10, 11],
                "example_key": "target-1",
                "donor_example_key": "donor-1",
                "case_key": "case-1",
                "transport": "delta",
                "path_edges": [{"source_layer": 4, "write_layer": 4, "weight": 1.0}],
                "source_layer_map": {"4": 4},
            },
        }
    ]

    harvest_batch_patch_stats(model, batch_specs)

    stats = model._v2_activation_patch_stats_by_req["req-1"][4]
    assert stats["status"] == "ok"
    assert stats["operator"] == "residual_path"
    assert stats["delta_norm_raw"] == pytest.approx(1.25)
    assert stats["covered_abs_spans"] == [[10, 12]]
    assert stats["covered_abs_tokens"] == 2
    assert stats["target_abs_tokens"] == 2
    assert stats["coverage_fraction"] == pytest.approx(1.0)
    assert stats["path_edges"] == [{"source_layer": 4, "write_layer": 4, "weight": 1.0}]


def test_residual_path_batch_custom_op_records_stats() -> None:
    import torch

    from pipelines_v2.engine.vllm.patching.custom_ops import register_torch_library_residual_path_batch_op

    register_torch_library_residual_path_batch_op()

    hidden = torch.zeros((6, 4), dtype=torch.float32)
    batch_query_positions = torch.tensor([[1, 2]], dtype=torch.int32)
    batch_payload_rows = torch.full((1, 2, 4), 0.5, dtype=torch.float32)
    batch_token_counts = torch.tensor([2], dtype=torch.int32)
    batch_transport_modes = torch.tensor([0], dtype=torch.int32)
    batch_replace_alphas = torch.tensor([0.0], dtype=torch.float32)
    batch_active = torch.tensor([1], dtype=torch.int32)
    stats_valid = torch.zeros((1,), dtype=torch.int32)
    stats_scalars = torch.zeros((1, 8), dtype=torch.float32)

    patched = torch.ops.xenon_activation_patch_v2.residual_path_batch(
        hidden,
        batch_query_positions,
        batch_payload_rows,
        batch_token_counts,
        batch_transport_modes,
        batch_replace_alphas,
        batch_active,
        stats_valid,
        stats_scalars,
    )

    assert torch.allclose(patched[1:3], torch.full((2, 4), 0.5, dtype=torch.float32))
    assert int(stats_valid[0].item()) == 1
    assert float(stats_scalars[0, 0].item()) > 0.0
    assert float(stats_scalars[0, 1].item()) == pytest.approx(2.0)


def test_set_batch_patch_specs_uses_dedicated_residual_path_runtime_fields() -> None:
    from types import SimpleNamespace

    import torch

    from pipelines_v2.engine.vllm.patching.state import set_batch_patch_specs

    model = SimpleNamespace(
        _v2_activation_patch_bank={
            4: {
                "donor-1": {"values": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)},
                "target-1": {"values": torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=torch.float32)},
            }
        },
        _v2_activation_patch_subspace={},
        _v2_activation_patch_batch_runtime_state={},
        _v2_activation_patch_batch_tensor_stats={},
    )

    set_batch_patch_specs(
        model,
        [
            {
                "req_id": "req-1",
                "patch_spec": {
                    "operator": "residual_path",
                    "target_layers": [4],
                    "query_positions": [0, 1],
                    "donor_example_key": "donor-1",
                    "donor_positions": [0, 1],
                    "target_read_positions": [0, 1],
                    "example_key": "target-1",
                    "transport": "replace",
                    "path_edges": [{"source_layer": 4, "write_layer": 4, "weight": 0.5}],
                    "strength": 2.0,
                },
            }
        ],
    )

    layer_state = model._v2_activation_patch_batch_runtime_state[4]
    assert int(layer_state["residual_path_transport_modes"][0].item()) == 1
    assert float(layer_state["residual_path_replace_alphas"][0].item()) == pytest.approx(1.0)
    assert int(layer_state["row_counts"][0].item()) == 0
    assert float(layer_state["strengths"][0].item()) == pytest.approx(0.0)


def test_run_custom_op_passes_residual_path_state_as_named_kwargs() -> None:
    import torch

    from pipelines_v2.engine.vllm.activation_patch_math import RESIDUAL_PATH_MODE_ID
    from pipelines_v2.engine.vllm.patching.custom_ops import run_custom_op

    seen: dict[str, object] = {}

    def fake_custom_op(*args: object, **kwargs: object) -> torch.Tensor:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return args[0]  # type: ignore[index]

    hidden = torch.zeros((2, 3), dtype=torch.float32)
    run_custom_op(
        hidden=hidden,
        custom_op=fake_custom_op,
        operator_id=RESIDUAL_PATH_MODE_ID,
        mean=torch.zeros((3,), dtype=torch.float32),
        scale=torch.ones((3,), dtype=torch.float32),
        safe_scale=torch.ones((3,), dtype=torch.float32),
        query_positions=torch.zeros((1, 1), dtype=torch.int32),
        token_counts=torch.zeros((1,), dtype=torch.int32),
        donor_rows=torch.zeros((1, 1, 3), dtype=torch.float32),
        selected_rows=torch.zeros((1, 1, 3), dtype=torch.float32),
        row_counts=torch.zeros((1,), dtype=torch.int32),
        strengths=torch.zeros((1,), dtype=torch.float32),
        active=torch.zeros((1,), dtype=torch.int32),
        stats_valid=torch.zeros((1,), dtype=torch.int32),
        stats_scalars=torch.zeros((1, 8), dtype=torch.float32),
        stats_coeff_before=torch.zeros((1, 1), dtype=torch.float32),
        stats_coeff_after=torch.zeros((1, 1), dtype=torch.float32),
        residual_path_transport_modes=torch.tensor([1], dtype=torch.int32),
        residual_path_replace_alphas=torch.tensor([0.25], dtype=torch.float32),
    )

    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert "batch_residual_path_transport_modes" in kwargs
    assert "batch_residual_path_replace_alphas" in kwargs


def test_activation_patch_model_init_hook_restores_without_mutating_layer_class() -> None:
    from types import SimpleNamespace

    import torch

    from pipelines_v2.engine.vllm.patching.hooks import (
        install_activation_patch_model_init_hook,
        restore_activation_patch_model_init_hook,
    )

    class DummyLayer(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + 1

    class DummyModel:
        def __init__(self) -> None:
            self.model = SimpleNamespace(layers=[DummyLayer()])

    original_model_init = DummyModel.__init__
    original_layer_forward = DummyLayer.forward

    assert install_activation_patch_model_init_hook(DummyModel) is True
    patched_model = DummyModel()
    patched_layer = patched_model.model.layers[0]

    assert getattr(patched_layer, "_v2_activation_patch_instance_hooked", False) is True
    assert callable(getattr(patched_layer, "_v2_activation_patch_original_forward", None))
    assert DummyLayer.forward is original_layer_forward

    assert restore_activation_patch_model_init_hook(DummyModel) is True
    assert DummyModel.__init__ is original_model_init

    restored_model = DummyModel()
    restored_layer = restored_model.model.layers[0]
    assert not getattr(restored_layer, "_v2_activation_patch_instance_hooked", False)


def test_project_out_batch_custom_op_preserves_valid_stats_across_invalid_followup() -> None:
    import torch

    from pipelines_v2.engine.vllm.activation_patch_core import (
        _register_torch_library_project_out_batch_op,
    )

    _register_torch_library_project_out_batch_op()

    hidden = torch.randn((12, 4), dtype=torch.float32)
    mean = torch.zeros((4,), dtype=torch.float32)
    scale = torch.ones((4,), dtype=torch.float32)
    safe_scale = torch.ones((4,), dtype=torch.float32)
    batch_selected_rows = torch.eye(4, dtype=torch.float32)[:2].unsqueeze(0)
    batch_row_counts = torch.tensor([2], dtype=torch.int32)
    batch_strengths = torch.tensor([1.0], dtype=torch.float32)
    batch_active = torch.tensor([1], dtype=torch.int32)
    stats_valid = torch.zeros((1,), dtype=torch.int32)
    stats_scalars = torch.zeros((1, 8), dtype=torch.float32)
    stats_coeff_before = torch.zeros((1, 2), dtype=torch.float32)
    stats_coeff_after = torch.zeros((1, 2), dtype=torch.float32)

    torch.ops.xenon_activation_patch_v2.project_out_batch(
        hidden,
        mean,
        scale,
        safe_scale,
        batch_selected_rows,
        batch_row_counts,
        torch.tensor([[2, 6]], dtype=torch.int32),
        batch_strengths,
        batch_active,
        stats_valid,
        stats_scalars,
        stats_coeff_before,
        stats_coeff_after,
    )

    valid_after_prefill = stats_valid.clone()
    scalars_after_prefill = stats_scalars.clone()
    coeff_before_after_prefill = stats_coeff_before.clone()
    coeff_after_after_prefill = stats_coeff_after.clone()

    torch.ops.xenon_activation_patch_v2.project_out_batch(
        hidden,
        mean,
        scale,
        safe_scale,
        batch_selected_rows,
        batch_row_counts,
        torch.tensor([[40, 44]], dtype=torch.int32),
        batch_strengths,
        batch_active,
        stats_valid,
        stats_scalars,
        stats_coeff_before,
        stats_coeff_after,
    )

    assert valid_after_prefill.tolist() == [1]
    assert stats_valid.tolist() == [1]
    assert torch.allclose(stats_scalars, scalars_after_prefill)
    assert torch.allclose(stats_coeff_before, coeff_before_after_prefill)
    assert torch.allclose(stats_coeff_after, coeff_after_after_prefill)


def test_token_selector_section_deduplicates_positions() -> None:
    positions = TokenSelector.section("BODY").resolve(
        6,
        token_sections={"BODY": [1, 1, 3, 3, 5]},
    )

    assert positions == [1, 3, 5]


def test_patched_generation_plan_rejects_missing_donor_rows_from_source_feature(tmp_path: Path) -> None:
    dataset = make_toy_dataset()
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"), catalog=FileCatalog(tmp_path / "catalog"))

    class FakeFeature:
        def result(self) -> dict[str, Any]:
            return {
                "kind": "activation_bank_result",
                "site": "resid_post",
                "layers": {
                    "0": {
                        "ex_a": {"values": [[0.0, 0.0, 0.0, 0.0]], "token_sections": {"BODY": [0]}},
                    }
                },
            }

    spec = PatchedGenerationSpec(
        engine=ToyEngine(sequence_length=8),
        dataset=dataset,
        patch=InterchangePatch(
            activation_bank=FakeFeature(),
            write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
            target_tokens=TokenSelector.full_sequence(),
        ),
        pair_by=dataset.cases("case_key"),
        target_when=dataset.labels("class").equals("positive"),
        donor_when=dataset.labels("class").equals("negative"),
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )

    plan = runner.plan(spec)

    assert any("missing donor activation rows" in error for error in plan.errors)


def test_patched_generation_plan_rejects_missing_donor_section_metadata(tmp_path: Path) -> None:
    dataset = make_toy_dataset()
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"), catalog=FileCatalog(tmp_path / "catalog"))

    class FakeFeature:
        def result(self) -> dict[str, Any]:
            return {
                "kind": "activation_bank_result",
                "site": "resid_post",
                "layers": {
                    "0": {
                        "ex_a": {"values": [[0.0, 0.0, 0.0, 0.0]], "token_sections": {"BODY": [0]}},
                        "ex_b": {"values": [[1.0, 1.0, 1.0, 1.0]]},
                    }
                },
            }

    spec = PatchedGenerationSpec(
        engine=ToyEngine(sequence_length=8),
        dataset=dataset,
        patch=InterchangePatch(
            activation_bank=FakeFeature(),
            write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
            target_tokens=TokenSelector.full_sequence(),
            donor_tokens=TokenSelector.section("BODY"),
        ),
        pair_by=dataset.cases("case_key"),
        target_when=dataset.labels("class").equals("positive"),
        donor_when=dataset.labels("class").equals("negative"),
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )

    plan = runner.plan(spec)

    assert any("patch.donor_tokens" in error for error in plan.errors)


def test_patched_generation_plan_skips_local_download_for_remote_activation_bank(tmp_path: Path) -> None:
    dataset = make_toy_dataset()
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"), catalog=FileCatalog(tmp_path / "catalog"))

    activation_bank = OperationArtifact(
        _manifest=ArtifactManifest(
            artifact_id="activation_bank_remote",
            artifact_kind="activation_bank",
            schema_version=1,
            operation_spec_hash="spec",
            operation_semantic_hash="semantic",
            created_at="2026-04-17T00:00:00+00:00",
            engine={},
            runner={"kind": "modal"},
            input_artifact_refs=(),
            example_coverage={"example_count": 2},
            storage_refs={
                "result": {
                    "store": "modal_volume",
                    "name": "xenon-data",
                    "path": "/data/artifacts/activation_bank_remote/result.json",
                    "format": "json",
                    "bytes": 3_000_000_000,
                }
            },
            metadata={},
            workflow_context={},
        ),
        store=ModalVolumeStore(name="xenon-data", root="/data/artifacts"),
    )

    class FakePathMask:
        def result(self) -> dict[str, Any]:
            return {
                "kind": "explicit_path_mask_result",
                "edges": [{"source_layer": 0, "write_layer": 0, "weight": 1.0}],
            }

    spec = PatchedGenerationSpec(
        engine=ToyEngine(sequence_length=8),
        dataset=dataset,
        patch=ResidualPathPatch(
            activation_bank=activation_bank,
            path_mask=FakePathMask(),
            write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
            target_tokens=TokenSelector.full_sequence(),
            read_tokens=TokenSelector.full_sequence(),
            transport="delta",
        ),
        pair_by=dataset.cases("case_key"),
        target_when=dataset.labels("class").equals("positive"),
        donor_when=dataset.labels("class").equals("negative"),
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )

    plan = runner.plan(spec)

    assert plan.errors == ()


def test_toy_engine_intervention_skips_missing_donor_rows_without_crashing() -> None:
    dataset = make_toy_dataset()

    class FakeFeature:
        def result(self) -> dict[str, Any]:
            return {
                "kind": "activation_bank_result",
                "site": "resid_post",
                "layers": {
                    "0": {
                        "ex_a": {"values": [[0.0, 0.0, 0.0, 0.0]], "token_sections": {"BODY": [0]}},
                    }
                },
            }

    spec = PatchedGenerationSpec(
        engine=ToyEngine(sequence_length=8),
        dataset=dataset,
        patch=InterchangePatch(
            activation_bank=FakeFeature(),
            write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
            target_tokens=TokenSelector.full_sequence(),
        ),
        pair_by=dataset.cases("case_key"),
        target_when=dataset.labels("class").equals("positive"),
        donor_when=dataset.labels("class").equals("negative"),
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )

    result = ToyEngine(sequence_length=8).intervene(spec)

    assert result.summary["patched_count"] == 0
    assert result.summary["skipped_count"] == 1
    assert result.rows[0]["status"] == "skipped"
    assert "missing donor activation rows" in result.rows[0]["skip_reason"]


def test_patch_comparison_rejects_variant_row_set_mismatch() -> None:
    class FakeArtifact:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self._rows = rows

        def result(self) -> dict[str, Any]:
            return {"rows": list(self._rows)}

    class FakeBuilder:
        local_python_sources: tuple[str, ...] = ()

        def build(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {"metrics": {"seen": True}}

    with pytest.raises(SpecValidationError, match="row sets must match exactly"):
        run_patch_comparison(
            PatchComparisonSpec(
                baseline=FakeArtifact([{"example_key": "a", "example": {"key": "a"}}]),
                variants={"main": FakeArtifact([{"example_key": "a"}, {"example_key": "b"}])},
                row_evaluator=FakeBuilder(),
            )
        )


def test_local_runner_executes_split_activation_patch_flow_with_toy_engine(tmp_path: Path) -> None:
    dataset = make_toy_dataset()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    catalog = FileCatalog(tmp_path / "catalog")
    runner = LocalRunner(artifacts=artifacts, catalog=catalog)

    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(sequence_length=8),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_full",
                    site="resid_post",
                    layers=[0],
                    tokens=TokenSelector.full_sequence(),
                )
            ],
        )
    )

    baseline = runner.run(
        GenerationRunSpec(
            engine=ToyEngine(sequence_length=8),
            dataset=dataset,
            select_when=dataset.labels("class").equals("positive"),
            generation=GenerationSpec(enabled=True, max_tokens=2),
        )
    )
    activation_bank = runner.run(
        ActivationBankSpec(
            feature=capture.feature("resid_full"),
            layers=[0],
        )
    )

    patch = runner.run(
        PatchedGenerationSpec(
            engine=ToyEngine(sequence_length=8),
            dataset=dataset,
            patch=InterchangePatch(
                activation_bank=activation_bank,
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.full_sequence(),
            ),
            pair_by=dataset.cases("case_key"),
            target_when=dataset.labels("class").equals("positive"),
            donor_when=dataset.labels("class").equals("negative"),
            generation=GenerationSpec(enabled=True, max_tokens=2),
        )
    )

    comparison = runner.run(
        PatchComparisonSpec(
            baseline=baseline,
            variants={"main": patch},
            row_evaluator=TransformBuilder.from_function(_patch_comparison_row_evaluator),
        )
    )

    baseline_payload = baseline.result()
    patch_payload = patch.result()
    comparison_payload = comparison.result()

    assert baseline_payload["kind"] == "generation_run_result"
    assert baseline_payload["summary"]["example_count"] == 1
    assert baseline_payload["rows"][0]["generated_text"] == "toy_generation:ex_a"

    assert patch_payload["kind"] == "patched_generation_result"
    assert patch_payload["summary"]["patched_count"] == 1
    assert len(patch_payload["rows"]) == 1
    assert patch_payload["rows"][0]["status"] == "ok"
    assert patch_payload["rows"][0]["generated_text"] == "toy_generation:negative"
    assert patch_payload["rows"][0]["patch_stats"]["0"]["token_count"] == 8
    assert patch.manifest().example_coverage["example_keys"] == ["ex_a"]
    assert patch.manifest().example_coverage["example_count"] == 1
    assert comparison_payload["kind"] == "patch_comparison_result"
    assert comparison_payload["summary"]["compared_count"] == 1
    assert comparison_payload["summary"]["metrics"]["flipped"]["kind"] == "boolean_rate"
    assert comparison_payload["summary"]["metrics"]["flipped"]["value"] == pytest.approx(1.0)


def test_local_runner_executes_subspace_spec_and_project_out_patch_with_toy_engine(tmp_path: Path) -> None:
    dataset = make_toy_dataset()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    catalog = FileCatalog(tmp_path / "catalog")
    runner = LocalRunner(artifacts=artifacts, catalog=catalog)

    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(sequence_length=8),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_full",
                    site="resid_post",
                    layers=[0],
                    tokens=TokenSelector.full_sequence(),
                )
            ],
        )
    )
    subspace = runner.run(
        SubspaceSpec(
            feature=capture.feature("resid_full"),
            layers=[0],
            components=2,
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.mean(),
        )
    )
    patched = runner.run(
        PatchedGenerationSpec(
            engine=ToyEngine(sequence_length=8),
            dataset=dataset,
            patch=ProjectOutPatch(
                subspace=subspace,
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.full_sequence(),
                component_indices_by_layer={0: (0, 1)},
            ),
            select_when=dataset.labels("class").equals("positive"),
            generation=GenerationSpec(enabled=True, max_tokens=2),
        )
    )

    subspace_payload = subspace.result()
    patch_payload = patched.result()

    assert subspace_payload["kind"] == "subspace_result"
    assert subspace_payload["summary"]["layer_count"] == 1
    assert subspace_payload["layers"]["0"]["component_count"] == 2
    assert patch_payload["kind"] == "patched_generation_result"
    assert patch_payload["summary"]["patched_count"] == 1
    assert patch_payload["rows"][0]["generated_text"] == "toy_generation:project_out:ex_a"
    assert patch_payload["rows"][0]["patch_stats"]["0"]["operator"] == "project_out"
    assert patch_payload["rows"][0]["patch_stats"]["0"]["source_layer"] == 0
    assert patched.manifest().example_coverage["example_keys"] == ["ex_a"]
    assert patched.manifest().example_coverage["example_count"] == 1


def test_local_runner_executes_random_control_patch_with_toy_engine(tmp_path: Path) -> None:
    dataset = make_toy_dataset()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    catalog = FileCatalog(tmp_path / "catalog")
    runner = LocalRunner(artifacts=artifacts, catalog=catalog)

    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(sequence_length=8),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_full",
                    site="resid_post",
                    layers=[0],
                    tokens=TokenSelector.full_sequence(),
                )
            ],
        )
    )
    subspace = runner.run(
        SubspaceSpec(
            feature=capture.feature("resid_full"),
            layers=[0],
            components=2,
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.mean(),
        )
    )
    patched = runner.run(
        PatchedGenerationSpec(
            engine=ToyEngine(sequence_length=8),
            dataset=dataset,
            patch=RandomControlPatch(
                subspace=subspace,
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.full_sequence(),
                component_indices_by_layer={0: (0, 1)},
                random_seed=7,
            ),
            select_when=dataset.labels("class").equals("positive"),
            generation=GenerationSpec(enabled=True, max_tokens=2),
        )
    )

    patch_payload = patched.result()

    assert patch_payload["summary"]["patched_count"] == 1
    assert patch_payload["rows"][0]["generated_text"] == "toy_generation:random_control:ex_a"
    assert patch_payload["rows"][0]["patch_stats"]["0"]["operator"] == "random_control"
    assert patch_payload["rows"][0]["patch_stats"]["0"]["selected_component_count"] == 2


def test_local_runner_executes_add_direction_patch_with_toy_engine(tmp_path: Path) -> None:
    dataset = make_toy_dataset()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    catalog = FileCatalog(tmp_path / "catalog")
    runner = LocalRunner(artifacts=artifacts, catalog=catalog)

    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(sequence_length=8),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_full",
                    site="resid_post",
                    layers=[0],
                    tokens=TokenSelector.full_sequence(),
                )
            ],
        )
    )
    subspace = runner.run(
        SubspaceSpec(
            feature=capture.feature("resid_full"),
            layers=[0],
            components=2,
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.mean(),
        )
    )
    direction = runner.run(
        DirectionSpec(
            feature=capture.feature("resid_full"),
            layers=[0],
            positive=dataset.labels("class").equals("positive"),
            negative=dataset.labels("class").equals("negative"),
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.mean(),
            subspace=subspace,
        )
    )
    patched = runner.run(
        PatchedGenerationSpec(
            engine=ToyEngine(sequence_length=8),
            dataset=dataset,
            patch=AddDirectionPatch(
                direction=direction,
                subspace=subspace,
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.full_sequence(),
                component_indices_by_layer={0: (0, 1)},
            ),
            select_when=dataset.labels("class").equals("positive"),
            generation=GenerationSpec(enabled=True, max_tokens=2),
        )
    )

    patch_payload = patched.result()

    assert patch_payload["summary"]["patched_count"] == 1
    assert patch_payload["rows"][0]["generated_text"] == "toy_generation:add_direction:ex_a"
    assert patch_payload["rows"][0]["patch_stats"]["0"]["operator"] == "add_direction"
    assert patch_payload["rows"][0]["patch_stats"]["0"]["source_layer"] == 0


def test_local_runner_executes_swap_mean_patch_with_toy_engine(tmp_path: Path) -> None:
    dataset = make_toy_dataset()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    catalog = FileCatalog(tmp_path / "catalog")
    runner = LocalRunner(artifacts=artifacts, catalog=catalog)

    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(sequence_length=8),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_full",
                    site="resid_post",
                    layers=[0],
                    tokens=TokenSelector.full_sequence(),
                )
            ],
        )
    )
    centroids = runner.run(
        CentroidSpec(
            feature=capture.feature("resid_full"),
            by=dataset.labels("class"),
            layers=[0],
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.mean(),
        )
    )
    patched = runner.run(
        PatchedGenerationSpec(
            engine=ToyEngine(sequence_length=8),
            dataset=dataset,
            patch=SwapMeanPatch(
                centroids=centroids,
                centroid_name="negative",
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.full_sequence(),
            ),
            select_when=dataset.labels("class").equals("positive"),
            generation=GenerationSpec(enabled=True, max_tokens=2),
        )
    )

    patch_payload = patched.result()

    assert patch_payload["summary"]["patched_count"] == 1
    assert patch_payload["rows"][0]["generated_text"] == "toy_generation:swap_mean:ex_a"
    assert patch_payload["rows"][0]["patch_stats"]["0"]["operator"] == "swap_mean"


def test_local_runner_executes_swap_components_patch_with_toy_engine(tmp_path: Path) -> None:
    dataset = make_toy_dataset()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    catalog = FileCatalog(tmp_path / "catalog")
    runner = LocalRunner(artifacts=artifacts, catalog=catalog)

    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(sequence_length=8),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_full",
                    site="resid_post",
                    layers=[0],
                    tokens=TokenSelector.full_sequence(),
                )
            ],
        )
    )
    subspace = runner.run(
        SubspaceSpec(
            feature=capture.feature("resid_full"),
            layers=[0],
            components=2,
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.mean(),
        )
    )
    centroids = runner.run(
        CentroidSpec(
            feature=capture.feature("resid_full"),
            by=dataset.labels("class"),
            layers=[0],
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.mean(),
            subspace=subspace,
        )
    )
    patched = runner.run(
        PatchedGenerationSpec(
            engine=ToyEngine(sequence_length=8),
            dataset=dataset,
            patch=SwapComponentsPatch(
                subspace=subspace,
                centroids=centroids,
                centroid_name="negative",
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.full_sequence(),
                component_indices_by_layer={0: (0, 1)},
            ),
            select_when=dataset.labels("class").equals("positive"),
            generation=GenerationSpec(enabled=True, max_tokens=2),
        )
    )

    patch_payload = patched.result()

    assert patch_payload["summary"]["patched_count"] == 1
    assert patch_payload["rows"][0]["generated_text"] == "toy_generation:swap_components:ex_a"
    assert patch_payload["rows"][0]["patch_stats"]["0"]["operator"] == "swap_components"
    assert patch_payload["rows"][0]["patch_stats"]["0"]["selected_component_count"] == 2


def test_local_runner_executes_residual_path_patch_with_toy_engine(tmp_path: Path) -> None:
    dataset = make_toy_dataset()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    catalog = FileCatalog(tmp_path / "catalog")
    runner = LocalRunner(artifacts=artifacts, catalog=catalog)

    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(sequence_length=8),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_full",
                    site="resid_post",
                    layers=[0],
                    tokens=TokenSelector.full_sequence(),
                )
            ],
        )
    )
    activation_bank = runner.run(
        ActivationBankSpec(
            feature=capture.feature("resid_full"),
            layers=[0],
        )
    )
    path_mask = runner.run(
        ExplicitPathMaskSpec(
            edges=(ExplicitPathEdge(source_layer=0, write_layer=0, weight=1.0),),
        )
    )
    patched = runner.run(
        PatchedGenerationSpec(
            engine=ToyEngine(sequence_length=8),
            dataset=dataset,
            patch=ResidualPathPatch(
                activation_bank=activation_bank,
                path_mask=path_mask,
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.full_sequence(),
                read_tokens=TokenSelector.full_sequence(),
                transport="delta",
            ),
            pair_by=dataset.cases("case_key"),
            target_when=dataset.labels("class").equals("positive"),
            donor_when=dataset.labels("class").equals("negative"),
            generation=GenerationSpec(enabled=True, max_tokens=2),
        )
    )

    patch_payload = patched.result()

    assert patch_payload["summary"]["patched_count"] == 1
    assert patch_payload["rows"][0]["generated_text"] == "toy_generation:negative"
    assert patch_payload["rows"][0]["patch_stats"]["0"]["operator"] == "residual_path"
    assert patch_payload["rows"][0]["patch_stats"]["0"]["transport"] == "delta"


def test_paired_request_payload_residual_path_reads_target_positions_from_activation_bank() -> None:
    dataset = make_toy_dataset()
    target = dataset.examples[0]
    donor = dataset.examples[1]
    spec = PatchedGenerationSpec(
        engine=ToyEngine(sequence_length=8),
        dataset=dataset,
        patch=ResidualPathPatch(
            activation_bank=StepRef("build_activation_bank"),
            path_mask=StepRef("build_path_mask"),
            write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
            target_tokens=TokenSelector.full_sequence(),
            read_tokens=TokenSelector.section("BODY"),
            transport="delta",
        ),
        pair_by=dataset.cases("case_key"),
        target_when=dataset.labels("class").equals("positive"),
        donor_when=dataset.labels("class").equals("negative"),
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )
    activation_bank = {
        "layers": {
            "0": {
                "ex_a": {
                    "values": [[0.0, 0.0], [0.0, 0.0]],
                    "token_sections": {"BODY": [1]},
                },
                "ex_b": {
                    "values": [[0.0, 0.0], [0.0, 0.0]],
                    "token_sections": {"BODY": [1]},
                },
            }
        }
    }
    path_mask = {"edges": [{"source_layer": 0, "write_layer": 0, "weight": 1.0}]}
    tokenized = {
        "token_ids": [101, 102, 103, 104],
        "token_sections": {"BODY": [0, 1, 2, 3]},
    }

    payload = paired_request_payload(
        spec=spec,
        activation_bank=activation_bank,
        path_mask_payload=path_mask,
        target=target,
        donor=donor,
        case_key="case_1",
        tokenized=tokenized,
        target_positions=[0],
    )

    assert isinstance(payload, dict)
    assert payload["donor_positions"] == [1]
    assert payload["target_read_positions"] == [1]


def test_vllm_engine_allows_noneager_project_out_subspace_patch() -> None:
    dataset = make_toy_dataset()

    class FakeSubspaceArtifact:
        def result(self) -> dict[str, Any]:
            return {
                "kind": "subspace_result",
                "layers": {
                    "0": {
                        "mean": [0.0, 0.0, 0.0, 0.0],
                        "scale": [1.0, 1.0, 1.0, 1.0],
                        "components": [[1.0, 0.0, 0.0, 0.0]],
                        "named_components": {},
                    }
                },
            }

    spec = PatchedGenerationSpec(
        engine=VLLMEngine(
            model_id="Qwen/Qwen3-0.6B",
            enforce_eager=False,
            enable_prefix_caching=False,
        ),
        dataset=dataset,
        patch=ProjectOutPatch(
            subspace=FakeSubspaceArtifact(),
            write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
            target_tokens=TokenSelector.full_sequence(),
        ),
        select_when=dataset.labels("class").equals("positive"),
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )

    errors = spec.engine.planning_errors(spec)  # type: ignore[union-attr]

    assert not any("enforce_eager=True" in error for error in errors)


def test_register_activation_patch_subspace_preserves_placeholder_layers() -> None:
    import torch

    from pipelines_v2.engine.vllm.activation_patch_core import (
        init_activation_patching,
        register_activation_patch_subspace,
    )

    class DummyNorm(torch.nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((dim,), dtype=torch.float32))

    class DummyLayer(torch.nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.input_layernorm = DummyNorm(dim)

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return args[1] if len(args) > 1 else args[0]

    class DummyInner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([DummyLayer(4), DummyLayer(4)])

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = DummyInner()

    model = DummyModel()
    init_activation_patching(model)

    subspace_state = model._v2_activation_patch_subspace
    subspace_state_id = id(subspace_state)
    assert sorted(int(layer) for layer in subspace_state) == [0, 1]

    register_activation_patch_subspace(
        model,
        {
            1: {
                "mean": [0.0, 0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0, 1.0],
                "components": [[1.0, 0.0, 0.0, 0.0]],
                "named_components": {"c0": 0},
            }
        },
    )

    updated = model._v2_activation_patch_subspace
    assert id(updated) == subspace_state_id
    assert sorted(int(layer) for layer in updated) == [0, 1]
    assert int(updated[0]["components"].shape[0]) == 0
    assert int(updated[1]["components"].shape[0]) == 1
    assert updated[1]["named_components"] == {"c0": 0}


def test_vllm_engine_allows_noneager_random_control_subspace_patch() -> None:
    dataset = make_toy_dataset()

    class FakeSubspaceArtifact:
        def result(self) -> dict[str, Any]:
            return {
                "kind": "subspace_result",
                "layers": {
                    "0": {
                        "mean": [0.0, 0.0, 0.0, 0.0],
                        "scale": [1.0, 1.0, 1.0, 1.0],
                        "components": [[1.0, 0.0, 0.0, 0.0]],
                        "named_components": {},
                    }
                },
            }

    spec = PatchedGenerationSpec(
        engine=VLLMEngine(
            model_id="Qwen/Qwen3-0.6B",
            enforce_eager=False,
            enable_prefix_caching=False,
        ),
        dataset=dataset,
        patch=RandomControlPatch(
            subspace=FakeSubspaceArtifact(),
            write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
            target_tokens=TokenSelector.full_sequence(),
        ),
        select_when=dataset.labels("class").equals("positive"),
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )

    errors = spec.engine.planning_errors(spec)  # type: ignore[union-attr]

    assert not any("enforce_eager=True" in error for error in errors)


def test_vllm_engine_allows_noneager_add_direction_subspace_patch() -> None:
    dataset = make_toy_dataset()

    class FakeDirectionArtifact:
        def result(self) -> dict[str, Any]:
            return {
                "kind": "direction_result",
                "layers": {
                    "0": {
                        "raw_vector": [1.0, 0.0, 0.0, 0.0],
                    }
                },
            }

    spec = PatchedGenerationSpec(
        engine=VLLMEngine(
            model_id="Qwen/Qwen3-0.6B",
            enforce_eager=False,
            enable_prefix_caching=False,
        ),
        dataset=dataset,
        patch=AddDirectionPatch(
            direction=FakeDirectionArtifact(),
            write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
            target_tokens=TokenSelector.full_sequence(),
        ),
        select_when=dataset.labels("class").equals("positive"),
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )

    errors = spec.engine.planning_errors(spec)  # type: ignore[union-attr]

    assert not any("enforce_eager=True" in error for error in errors)
    assert spec.runtime_spec().env["XENON_ACTIVATION_PATCH_COMPILED_OPERATOR"] == "subspace"  # type: ignore[union-attr]


def test_vllm_engine_runtime_spec_sets_vllm_runtime_compatibility_env() -> None:
    engine = VLLMEngine(model_id="Qwen/Qwen3-0.6B")

    runtime_spec = engine.runtime_spec()

    assert runtime_spec.env["VLLM_COMPILE_CACHE_SAVE_FORMAT"] == "binary"
    assert runtime_spec.env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_vllm_engine_resolves_canonical_model_id_under_model_path_root() -> None:
    engine = VLLMEngine(
        model_id="Qwen/Qwen3-30B-A3B",
        model_path_root="/models",
    )

    assert engine.identity()["model_id"] == "Qwen/Qwen3-30B-A3B"
    assert engine.identity()["model_path_root"] == "/models"
    assert engine.semantic_identity()["model_id"] == "Qwen/Qwen3-30B-A3B"
    assert "model_path_root" not in engine.semantic_identity()
    assert engine.resolved_model_path() == "/models/Qwen/Qwen3-30B-A3B"
    assert engine.canonical_model_name() == "Qwen/Qwen3-30B-A3B"


def test_build_llm_kwargs_uses_local_model_path_and_canonical_served_name() -> None:
    llm_kwargs, _ = build_llm_kwargs(
        VLLMEngine(
            model_id="Qwen/Qwen3-30B-A3B",
            model_path_root="/models",
            enforce_eager=True,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            distributed_executor_backend="mp",
        )
    )

    assert llm_kwargs["model"] == "/models/Qwen/Qwen3-30B-A3B"
    assert llm_kwargs["served_model_name"] == "Qwen/Qwen3-30B-A3B"
    assert llm_kwargs["tensor_parallel_size"] == 2
    assert llm_kwargs["pipeline_parallel_size"] == 1
    assert llm_kwargs["distributed_executor_backend"] == "mp"


def test_build_llm_kwargs_adds_additional_config_for_compiled_patch_worker() -> None:
    llm_kwargs, _ = build_llm_kwargs(
        VLLMEngine(
            model_id="Qwen/Qwen3-0.6B",
            enforce_eager=False,
            enable_prefix_caching=False,
        ),
        compiled_operator_hint="subspace",
    )

    assert llm_kwargs["worker_cls"] == (
        "pipelines_v2.engine.vllm.activation_patch_request_worker.ActivationPatchGPUWorker"
    )
    assert llm_kwargs["compilation_config"] == {
        "custom_ops": ["none", "+activation_patch_hidden_states"],
        "cudagraph_mode": "PIECEWISE",
    }
    assert llm_kwargs["additional_config"] == {
        "xenon_activation_patch_worker_cls": (
            "pipelines_v2.engine.vllm.activation_patch_request_worker.ActivationPatchGPUWorker"
        ),
        "xenon_activation_patch_compiled_operator": "subspace",
    }


def test_vllm_engine_allows_noneager_swap_mean_subspace_patch() -> None:
    dataset = make_toy_dataset()

    class FakeCentroidArtifact:
        def result(self) -> dict[str, Any]:
            return {
                "kind": "centroid_result",
                "layers": {
                    "0": {
                        "centroids": {
                            "positive": [1.0, 0.0, 0.0, 0.0],
                        }
                    }
                },
            }

    spec = PatchedGenerationSpec(
        engine=VLLMEngine(
            model_id="Qwen/Qwen3-0.6B",
            enforce_eager=False,
            enable_prefix_caching=False,
        ),
        dataset=dataset,
        patch=SwapMeanPatch(
            centroids=FakeCentroidArtifact(),
            centroid_name="positive",
            write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
            target_tokens=TokenSelector.full_sequence(),
        ),
        select_when=dataset.labels("class").equals("positive"),
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )

    errors = spec.engine.planning_errors(spec)  # type: ignore[union-attr]

    assert not any("enforce_eager=True" in error for error in errors)
    assert spec.runtime_spec().env["XENON_ACTIVATION_PATCH_COMPILED_OPERATOR"] == "subspace"  # type: ignore[union-attr]


def test_vllm_engine_allows_noneager_swap_components_subspace_patch() -> None:
    dataset = make_toy_dataset()

    class FakeSubspaceArtifact:
        def result(self) -> dict[str, Any]:
            return {
                "kind": "subspace_result",
                "layers": {
                    "0": {
                        "mean": [0.0, 0.0, 0.0, 0.0],
                        "scale": [1.0, 1.0, 1.0, 1.0],
                        "components": [[1.0, 0.0, 0.0, 0.0]],
                        "named_components": {},
                    }
                },
            }

    class FakeCentroidArtifact:
        def result(self) -> dict[str, Any]:
            return {
                "kind": "centroid_result",
                "layers": {
                    "0": {
                        "centroids": {
                            "positive": [1.0, 0.0, 0.0, 0.0],
                        }
                    }
                },
            }

    spec = PatchedGenerationSpec(
        engine=VLLMEngine(
            model_id="Qwen/Qwen3-0.6B",
            enforce_eager=False,
            enable_prefix_caching=False,
        ),
        dataset=dataset,
        patch=SwapComponentsPatch(
            subspace=FakeSubspaceArtifact(),
            centroids=FakeCentroidArtifact(),
            centroid_name="positive",
            write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
            target_tokens=TokenSelector.full_sequence(),
        ),
        select_when=dataset.labels("class").equals("positive"),
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )

    errors = spec.engine.planning_errors(spec)  # type: ignore[union-attr]

    assert not any("enforce_eager=True" in error for error in errors)
    assert spec.runtime_spec().env["XENON_ACTIVATION_PATCH_COMPILED_OPERATOR"] == "subspace"  # type: ignore[union-attr]
