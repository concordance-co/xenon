from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

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
    PairDeltaSpec,
    PromptMetadataBuilder,
    ResidualSite,
    RoutingRecord,
    StepLabelRef,
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
    WorkflowSpec,
    WorkflowStep,
    ReportSpec,
)
from pipelines_v2.cli import main as pipelines_v2_cli_main
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


class _FailOnceRunner:
    def __init__(self, inner: Any, *, fail_step: str | None = None) -> None:
        self.inner = inner
        self.fail_step = fail_step
        self.failed = False
        self.calls: list[str] = []
        self.catalog = inner.catalog
        self.artifacts = inner.artifacts

    def plan(self, spec: Any) -> Any:
        return self.inner.plan(spec)

    def run(self, spec: Any, *, workflow_context: Any | None = None) -> Any:
        step_name = workflow_context.step_name if workflow_context is not None else "<unknown>"
        self.calls.append(step_name)
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
    assert report.manifest().storage_refs["report"]["path"].endswith("report.md")


def test_phase_04_arch2_target_builders_and_json_loader() -> None:
    module_path = Path("projects/DX_TERMINAL/prompt_confusion/phase_04/specs/arch2_target.py")
    spec = importlib.util.spec_from_file_location("phase_04_arch2_target", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
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
    probe_conflict = next(step.spec for step in prompt_state.steps if step.name == "probe_conflict_present")
    assert probe_conflict.tokens.kind == "full_sequence"
    assert probe_conflict.pooling.kind == "last"


def test_phase_04_arch2_target_workflows_plan_cleanly() -> None:
    module_path = Path("projects/DX_TERMINAL/prompt_confusion/phase_04/specs/arch2_target.py")
    spec = importlib.util.spec_from_file_location("phase_04_arch2_target", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    loaded = module.load_phase_04_target_json()
    orchestrator = loaded["orchestrator"]

    for workflow in loaded["workflows"].values():
        plan = orchestrator.plan(workflow)
        assert plan.steps
        assert all(not step.execution.errors for step in plan.steps)


def test_pipelines_v2_modal_smoke_file_builders() -> None:
    module_path = Path("scripts/pipelines_v2_orchestrator_smoke.py")
    spec = importlib.util.spec_from_file_location("pipelines_v2_orchestrator_smoke", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
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
