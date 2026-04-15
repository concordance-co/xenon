from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from pipelines_v2.core.types import utc_now_iso
from pipelines_v2.api import (
    ArtifactManifest,
    ArtifactLabelRef,
    CapabilityError,
    CaptureArtifact,
    CaptureSpec,
    Dataset,
    DirectionSpec,
    EngineCapability,
    EngineCaptureResult,
    Example,
    FileCatalog,
    GeometrySpec,
    GenerationSpec,
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
    OperationArtifact,
    PairDeltaSpec,
    PromptMetadataBuilder,
    ResidualSite,
    ResidualizedProbeSpec,
    RoutingRecord,
    StepLabelRef,
    TextBaselineSpec,
    TransferProbeSpec,
    TransformBuilder,
    TransformResult,
    TransformSpec,
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
)
from pipelines_v2.cli import _workflow_result_payload, load_python_workflow_file, main as pipelines_v2_cli_main
from pipelines_v2.engine.vllm.capture import _fill_router_features
from pipelines_v2.engine.vllm.moe_hooks import _make_patched_forward
from pipelines_v2.runtime import ExecutionPlan
from pipelines_v2.runtime.specs import runner_spec_from_dict
from pipelines_v2.runtime.remote_executor import execute_remote
from pipelines_v2.runtime.modal_worker import _mounted_volumes, _resolved_runtime_spec
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
    assert resid_ref["format"] == "residual_safetensors_v1"
    assert resid_ref["tensor_path"].endswith("features/feature_tensors.safetensors")
    assert resid_ref["metadata_path"].endswith("features/resid_last.metadata.json")
    assert router_ref["format"] == "moe_routing_safetensors_v1"
    assert router_ref["tensor_path"] == resid_ref["tensor_path"]
    assert router_ref["metadata_path"].endswith("features/router_last.metadata.json")

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
    assert len(bundle) == 6


def test_fill_router_features_error_includes_actual_and_discovered_layers() -> None:
    feature_payloads = {
        "router_last": {
            "kind": "moe_routing",
            "routing_policy": {"source": "vllm_gate_logits", "observed_routing_decisions": False},
            "layers": {"0": {}, "4": {}},
        }
    }
    site = MoERoutingSite(name="router_last", layers=[0, 4], tokens=TokenSelector.last())
    example = Example(key="ex_a", prompt="alpha")

    with pytest.raises(RuntimeError, match="captured router layers=\\[4\\].*discovered MoE layers=\\[4, 8\\]"):
        _fill_router_features(
            feature_payloads=feature_payloads,
            routing_sites=[site],
            router_data={4: np.zeros((1, 8), dtype=np.float32)},
            example=example,
            token_count=1,
            token_sections={},
            discovered_router_layers=[4, 8],
        )


def test_fill_router_features_error_includes_captured_length_for_token_mismatch() -> None:
    feature_payloads = {
        "router_last": {
            "kind": "moe_routing",
            "routing_policy": {"source": "vllm_gate_logits", "observed_routing_decisions": False},
            "layers": {"0": {}},
        }
    }
    site = MoERoutingSite(name="router_last", layers=[0], tokens=TokenSelector.last())

    with pytest.raises(RuntimeError, match="captured router logits only have length 4"):
        _fill_router_features(
            feature_payloads=feature_payloads,
            routing_sites=[site],
            router_data={0: np.zeros((4, 8), dtype=np.float32)},
            example=Example(key="ex_a", prompt="alpha"),
            token_count=10,
            token_sections={},
            discovered_router_layers=[0],
        )


def test_vllm_router_hook_appends_logits_across_chunked_forwards() -> None:
    torch = pytest.importorskip("torch")

    class _FakeGate:
        def __init__(self) -> None:
            self.weight = torch.zeros((3, 2), dtype=torch.float32)

        def __call__(self, hidden_states: Any) -> tuple[Any, Any]:
            sums = hidden_states.sum(dim=1, keepdim=True)
            router_logits = torch.cat((sums, sums + 1, sums + 2), dim=1)
            return router_logits, None

    class _FakeExperts:
        def __call__(self, *, hidden_states: Any, router_logits: Any) -> tuple[Any, Any]:
            return None, hidden_states

    class _FakeBlock:
        def __init__(self) -> None:
            self.gate = _FakeGate()
            self.experts = _FakeExperts()
            self.is_sequence_parallel = False
            self.tp_size = 1
            self._router_logits_buffer = torch.zeros((8, 3), dtype=torch.float32)
            self._router_num_captured = 0
            self._router_capture_enabled = True

    block = _FakeBlock()
    patched = _make_patched_forward(block)

    patched(torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float32))
    patched(torch.tensor([[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]], dtype=torch.float32))

    captured = block._router_logits_buffer[: block._router_num_captured].cpu().numpy()
    expected = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
        ],
        dtype=np.float32,
    )

    assert block._router_num_captured == 5
    assert np.allclose(captured, expected)


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


def test_vllm_engine_plan_rejects_router_capture_with_batch_gt_1(tmp_path: Path) -> None:
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    spec = CaptureSpec(
        engine=VLLMEngine(model_id="/models/Qwen/Qwen3-30B-A3B", max_num_seqs=2),
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

    assert any("MoE routing capture" in error for error in plan.errors)
    with pytest.raises(SpecValidationError, match="MoE routing capture"):
        plan.validate()


def test_vllm_engine_plan_rejects_router_capture_without_eager(tmp_path: Path) -> None:
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    spec = CaptureSpec(
        engine=VLLMEngine(
            model_id="/models/Qwen/Qwen3-30B-A3B",
            enforce_eager=False,
            max_num_seqs=1,
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

    assert any("enforce_eager=True" in error for error in plan.errors)
    with pytest.raises(SpecValidationError, match="enforce_eager=True"):
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
    assert saved["storage_refs"]["features"]["resid_last"]["format"] == "residual_safetensors_v1"
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


def test_modal_runner_serializes_cpu_analysis_resources(tmp_path: Path) -> None:
    runner = ModalRunner(
        resources=ModalResources(
            cpu=6,
            memory_mb=24 * 1024,
            timeout_seconds=1800,
        ),
        artifacts=ModalVolumeStore(name="xenon-data", root=str(tmp_path / "mounted")),
    )

    identity = runner.identity()

    assert identity["resources"]["gpu"] is None
    assert identity["resources"]["cpu"] == 6
    assert identity["resources"]["memory_mb"] == 24 * 1024
    assert identity["resources"]["timeout_seconds"] == 1800


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


def test_mounted_volumes_reject_duplicate_mount_paths() -> None:
    with pytest.raises(ValueError, match="Duplicate Modal volume mount paths"):
        _mounted_volumes(
            store_config={"kind": "modal_volume", "name": "xenon-data", "root": "/data/artifacts"},
            resources={
                "gpu": "L4",
                "volumes": [
                    {"name": "xenon-models", "mount_path": "/data"},
                ],
            },
        )


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
            WorkflowStep(name="capture", runner="gpu", spec=make_toy_capture_spec()),
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
            WorkflowStep(name="capture", runner="gpu", spec=make_toy_capture_spec()),
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


def test_postgres_catalog_records_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: list[tuple[str, tuple[object, ...] | None]] = []
    committed = {"value": False}
    monkeypatch.setenv("XENON_DATABASE_URL", "postgresql://example/xenon")

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

    fake_psycopg = types.SimpleNamespace(connect=lambda url: FakeConnection())
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
        workflow_context={},
    )

    PostgresCatalog(source=PostgresSource.from_env("XENON_DATABASE_URL")).record_artifact(manifest)

    assert committed["value"] is True
    assert any("CREATE TABLE IF NOT EXISTS pipelines_v2_artifacts" in sql for sql, _ in executed)
    insert = next(params for sql, params in executed if "INSERT INTO pipelines_v2_artifacts" in sql)
    assert insert is not None
    assert insert[0] == "probe_123"


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
    fetch_count = {"value": 0}

    def fake_fetch(self: PostgresSource, **kwargs: Any) -> Dataset:
        fetch_count["value"] += 1
        return Dataset.from_examples(
            [
                Example(key="a", prompt="alpha", labels={"class": "positive"}),
                Example(key="b", prompt="beta", labels={"class": "negative"}),
                Example(key="c", prompt="gamma", labels={"class": "positive"}),
            ]
        )

    monkeypatch.setattr(PostgresSource, "fetch_dataset", fake_fetch)

    dataset = Dataset.from_postgres(
        source=PostgresSource.from_env("XENON_DATABASE_URL"),
        table="public.capture_examples",
        prompt_column="prompt",
        example_key_column="example_id",
        label_columns=["class"],
    )

    predicate = dataset.labels("class").equals("positive")

    assert predicate.resolve_example_keys() == ["a", "c"]
    assert fetch_count["value"] == 1


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

    assert builder.import_path == "test_pipelines_v2_basics:_test_prompt_section_metadata"
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
    assert router_transfer_payload["kind"] == "transfer_probe_result"
    assert "size_to_activity" in router_transfer_payload["layers"][0]["cross_cohort_transfer"]
    assert text_baseline_payload["kind"] == "text_baseline_result"
    assert text_baseline_payload["mode"] == "grouped_cv"
    assert residualized_payload["kind"] == "residualized_probe_result"
    assert residualized_payload["layers"][0]["family_subspace_rank"] >= 1
    assert geometry_pca_payload["kind"] == "geometry_result"
    assert geometry_pca_payload["layers"][0]["example_count"] == 4
    assert geometry_lda_payload["kind"] == "geometry_result"
    assert geometry_lda_payload["layers"][0]["label_name"] == "strategy_family"


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


def test_phase_05_workflow_json_roundtrips_to_real_library_specs() -> None:
    module_path = Path("projects/DX_TERMINAL/prompt_confusion/phase_05/specs/workflow.py")
    spec = importlib.util.spec_from_file_location("phase_05_workflow", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    loaded = module.load_workflow_json()
    workflow = WorkflowSpec.from_dict(loaded)

    assert isinstance(workflow.steps[5].spec, TransferProbeSpec)
    assert isinstance(workflow.steps[9].spec, TextBaselineSpec)
    assert isinstance(workflow.steps[13].spec, ResidualizedProbeSpec)
    assert isinstance(workflow.steps[16].spec, GeometrySpec)
    assert isinstance(workflow.steps[7].spec.rows, Dataset)
    assert workflow.steps[7].spec.rows.name == "prompt_confusion_phase_05_arbitration"
    assert isinstance(workflow.steps[8].spec.rows, Dataset)
    assert workflow.steps[8].spec.rows.name == "prompt_confusion_phase_05_arbitration"


def test_phase_04_arch2_target_builders_and_json_loader() -> None:
    module_path = Path("projects/DX_TERMINAL/prompt_confusion/phase_04/specs/arch2_target.py")
    spec = importlib.util.spec_from_file_location("phase_04_arch2_target", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    workflow = module.build_workflow()
    loaded = module.load_phase_04_target_json()

    assert isinstance(workflow, WorkflowSpec)
    assert [step.name for step in workflow.steps] == [
        "capture_prompt_state_residual",
        "capture_prompt_state_router",
        "probe_conflict_present",
        "report",
    ]
    assert "capture_gpu" in module.build_runner_specs()
    assert "prompt_state" in loaded["workflows"]
    assert isinstance(loaded["workflows"]["prompt_state"], WorkflowSpec)
    assert isinstance(loaded["orchestrator"], WorkflowOrchestrator)
    assert "conflict_probe_examples_v3" in loaded["dataset"].fetch["sql"]
    assert loaded["dataset"].fetch["case_key_column"] == "matched_pair_id"
    assert loaded["dataset"].fetch.get("metadata_columns", ()) == ()
    prompt_state = loaded["workflows"]["prompt_state"]
    capture_router = next(step.spec for step in prompt_state.steps if step.name == "capture_prompt_state_router")
    probe_conflict = next(step.spec for step in prompt_state.steps if step.name == "probe_conflict_present")
    assert capture_router.engine.enforce_eager is True
    assert capture_router.engine.enable_prefix_caching is False
    assert probe_conflict.tokens.kind == "full_sequence"
    assert probe_conflict.pooling.kind == "last"


def test_phase_04_arch2_target_workflows_plan_cleanly() -> None:
    module_path = Path("projects/DX_TERMINAL/prompt_confusion/phase_04/specs/arch2_target.py")
    spec = importlib.util.spec_from_file_location("phase_04_arch2_target", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    loaded = module.load_phase_04_target_json()
    orchestrator = loaded["orchestrator"]

    for workflow in loaded["workflows"].values():
        plan = orchestrator.plan(workflow)
        assert plan.steps
        assert all(not step.execution.errors for step in plan.steps)


def test_phase_05_pipelines_v2_workflow_builders_and_json_loader() -> None:
    module_path = Path("projects/DX_TERMINAL/prompt_confusion/phase_05/specs/workflow.py")
    spec = importlib.util.spec_from_file_location("phase_05_workflow", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    base_dataset = module.build_dataset()
    arbitration_dataset = module.build_phase_05_arbitration_dataset()
    workflow = module.build_workflow(base_dataset, arbitration_dataset=arbitration_dataset)
    loaded = module.load_workflow_json()

    assert isinstance(workflow, WorkflowSpec)
    assert [step.name for step in workflow.steps] == [
        "derive_family_group",
        "capture_prompt_eos_residual",
        "capture_prompt_eos_router",
        "family_identity_residual",
        "family_identity_router",
        "detection_transfer_residual",
        "detection_transfer_router",
        "arbitration_transfer_residual",
        "arbitration_transfer_router",
        "lexical_family_identity",
        "lexical_cross_family_detection_transfer",
        "lexical_holdout_detection_residual",
        "lexical_holdout_detection_text",
        "family_residualized_conflict_residual",
        "family_residualized_conflict_router",
        "detection_transfer_regularization_sweep",
        "family_geometry_pca_full",
        "family_geometry_pca_conflict_only",
        "family_geometry_lda_conflict_only",
        "report",
    ]
    assert "capture_gpu" in module.build_runner_specs()
    assert base_dataset.fetch["table"] == "workflow_dataset_conflict_probe_v3_v1"
    assert arbitration_dataset.fetch["table"] == "workflow_dataset_conflict_probe_v3_conflict_readout_side_v1"
    assert isinstance(loaded, dict)
    assert loaded["kind"] == "workflow"
    assert [step["name"] for step in loaded["steps"]] == [step.name for step in workflow.steps]
    assert loaded["steps"][1]["spec"]["engine"]["kind"] == "vllm"
    assert loaded["steps"][2]["spec"]["engine"]["kind"] == "vllm"
    assert loaded["steps"][1]["spec"]["engine"]["enforce_eager"] is False
    assert loaded["steps"][2]["spec"]["engine"]["enforce_eager"] is True
    assert loaded["steps"][1]["spec"]["engine"]["enable_prefix_caching"] is True
    assert loaded["steps"][2]["spec"]["engine"]["enable_prefix_caching"] is False
    assert loaded["steps"][5]["spec"]["kind"] == "transfer_probe"
    assert loaded["steps"][7]["spec"]["rows"]["name"] == "prompt_confusion_phase_05_arbitration"
    assert loaded["steps"][8]["spec"]["rows"]["name"] == "prompt_confusion_phase_05_arbitration"
    assert loaded["steps"][9]["spec"]["kind"] == "text_baseline"
    assert loaded["steps"][13]["spec"]["kind"] == "residualized_probe"
    assert loaded["steps"][16]["spec"]["kind"] == "geometry"


def test_phase_05_pipelines_v2_workflow_records_missing_pieces_explicitly() -> None:
    module_path = Path("projects/DX_TERMINAL/prompt_confusion/phase_05/specs/workflow.py")
    spec = importlib.util.spec_from_file_location("phase_05_workflow", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    payload = module.build_phase_05_target_payload()
    missing = payload["missing_pieces"]

    assert payload["mode"] == "library_backed"
    assert any("report outputs are still thin" in item.lower() for item in missing)
    assert not any("transfer probe spec" in item.lower() for item in missing)
    assert not any("text-baseline" in item.lower() for item in missing)
    assert not any("residualization" in item.lower() for item in missing)
    assert not any("geometry" in item.lower() and "still" not in item.lower() for item in missing)


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
    assert "." in workflow.steps[0].spec.runtime_spec().local_python_sources
    assert workflow.steps[0].spec.prompt_metadata_builder.import_path == (
        "scripts.pipelines_v2_orchestrator_smoke:build_prompt_metadata"
    )
    assert workflow.steps[1].spec.tokens.kind == "section"
    assert workflow.steps[1].spec.pooling.kind == "mean"
    assert sorted(runner_specs) == ["analysis_cpu", "capture_gpu"]
    assert runner_specs["capture_gpu"].resources.gpu == "L4"
    assert runner_specs["analysis_cpu"].resources.cpu == 4


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


def test_pipelines_v2_cli_rerun_step_and_from_step_use_prior_run_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workflow_file = _write_cli_local_workflow_file(tmp_path)
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
