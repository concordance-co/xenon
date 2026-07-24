from __future__ import annotations

from pathlib import Path

import pytest

from pipelines_v2.engine import ToyEngine
from pipelines_v2.engine.base import EngineCaptureResult, EngineGenerationResult, EngineInterventionResult
from pipelines_v2.engine.vllm.engine import VLLMEngine
from pipelines_v2.operations.capture import CaptureSpec, GenerationSpec
from pipelines_v2.operations.derive import LabelMapSpec
from pipelines_v2.operations.interventions import (
    GenerationRunSpec,
    InterchangePatch,
    PatchedGenerationSpec,
    ProjectOutPatch,
    ResidualInterventionSite,
)
from pipelines_v2.operations.specs import TokenSelector
from pipelines_v2.runtime.modal import ModalResources, ModalRunner
from pipelines_v2.runtime.modal_worker import _resolved_runtime_spec_many
from pipelines_v2.runtime.remote_executor import execute_remote, execute_remote_many, merge_remote_shards
from pipelines_v2.runtime import LocalRunner
from pipelines_v2.storage import LocalArtifactStore
from pipelines_v2.storage.artifacts import InlineOperationArtifact
from pipelines_v2.storage.modal import ModalVolumeStore
from pipelines_v2.testing import EngineRunnerContractSuite, RunnerContractSuite, make_toy_dataset


@pytest.mark.contract
@pytest.mark.integration_local
def test_local_runner_satisfies_capture_contract(tmp_path: Path) -> None:
    RunnerContractSuite(LocalRunner).run_capture_smoke(tmp_path)


@pytest.mark.contract
@pytest.mark.integration_local
def test_local_runner_satisfies_capture_and_label_operation_contract(tmp_path: Path) -> None:
    RunnerContractSuite(LocalRunner).run_capture_and_label_operation_smoke(tmp_path)


@pytest.mark.contract
@pytest.mark.integration_local
def test_engine_runner_batched_contract_prefers_run_many(tmp_path: Path) -> None:
    batch_calls: list[list[str]] = []

    class RecordingBatchRunner:
        def __init__(self, wrapped: LocalRunner) -> None:
            self.wrapped = wrapped

        def run(self, spec: object, **kwargs: object) -> object:
            return self.wrapped.run(spec, **kwargs)

        def run_many(self, specs: object, **kwargs: object) -> list[object]:
            del kwargs
            specs_list = list(specs)  # type: ignore[arg-type]
            batch_calls.append([str(getattr(spec, "kind", "")) for spec in specs_list])
            return [self.wrapped.run(spec) for spec in specs_list]

    def runner_factory(*, artifacts: object, catalog: object) -> RecordingBatchRunner:
        return RecordingBatchRunner(LocalRunner(artifacts=artifacts, catalog=catalog))  # type: ignore[arg-type]

    suite = EngineRunnerContractSuite(
        runner_factory=runner_factory,
        engine_factory=ToyEngine,
    )
    suite.run_batched_capture_generation_and_project_out_contract(tmp_path / "basic")
    suite.run_unpaired_patch_operator_contracts(tmp_path / "unpaired")
    suite.run_paired_patch_operator_contracts(tmp_path / "paired")

    assert batch_calls == [
        ["capture", "generation_run", "patched_generation"],
        [
            "patched_generation",
            "patched_generation",
            "patched_generation",
            "patched_generation",
            "patched_generation",
        ],
        ["patched_generation", "patched_generation"],
    ]


@pytest.mark.contract
@pytest.mark.integration_local
def test_engine_runner_requested_contracts_batch_all_patch_families_together(tmp_path: Path) -> None:
    batch_calls: list[list[str]] = []

    class RecordingBatchRunner:
        def __init__(self, wrapped: LocalRunner) -> None:
            self.wrapped = wrapped

        def run(self, spec: object, **kwargs: object) -> object:
            return self.wrapped.run(spec, **kwargs)

        def run_many(self, specs: object, **kwargs: object) -> list[object]:
            del kwargs
            specs_list = list(specs)  # type: ignore[arg-type]
            batch_calls.append([str(getattr(spec, "kind", "")) for spec in specs_list])
            return [self.wrapped.run(spec) for spec in specs_list]

    def runner_factory(*, artifacts: object, catalog: object) -> RecordingBatchRunner:
        return RecordingBatchRunner(LocalRunner(artifacts=artifacts, catalog=catalog))  # type: ignore[arg-type]

    coverage = EngineRunnerContractSuite(
        runner_factory=runner_factory,
        engine_factory=ToyEngine,
    ).run_requested_model_bound_contracts(
        tmp_path,
        basic=True,
        unpaired=True,
        paired=True,
    )

    assert coverage == {"basic", "unpaired", "paired"}
    assert batch_calls == [
        ["capture", "generation_run", "patched_generation"],
        [
            "patched_generation",
            "patched_generation",
            "patched_generation",
            "patched_generation",
            "patched_generation",
            "patched_generation",
            "patched_generation",
        ],
    ]


@pytest.mark.contract
@pytest.mark.integration_local
def test_remote_executor_many_runs_specs_in_one_process(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    dataset = make_toy_dataset()
    specs = [
        LabelMapSpec(
            source=dataset.labels("class"),
            output_name="positive_flag",
            mapping={"positive": 1, "negative": 0},
        ),
        LabelMapSpec(
            source=dataset.labels("class"),
            output_name="negative_flag",
            mapping={"positive": 0, "negative": 1},
        ),
    ]

    manifests = execute_remote_many(
        runner_config={"kind": "contract"},
        store_config=store.identity(),
        spec_payloads=[spec.to_dict() for spec in specs],
        workflow_contexts=[
            {"run_id": "wr_batch", "step_name": "positive", "workflow_step_key": "wf.positive"},
            {"run_id": "wr_batch", "step_name": "negative", "workflow_step_key": "wf.negative"},
        ],
    )

    assert [manifest["artifact_kind"] for manifest in manifests] == ["label_map", "label_map"]
    assert [manifest["workflow_context"]["step_name"] for manifest in manifests] == ["positive", "negative"]
    assert [store.read_json_ref(manifest["storage_refs"]["result"])["output_name"] for manifest in manifests] == [
        "positive_flag",
        "negative_flag",
    ]


@pytest.mark.unit
def test_remote_executor_sharded_generation_handles_empty_selected_shards(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    dataset = make_toy_dataset()
    spec = GenerationRunSpec(
        engine=ToyEngine(),
        dataset=dataset,
        select_when=dataset.labels("class").equals("positive"),
        generation=GenerationSpec(enabled=True, max_tokens=1),
    )
    contexts = [
        {
            "run_id": "wr_sharded_generation",
            "workflow_step_key": "wf.generation",
            "step_name": "generation",
            "execution_shard": {"index": index, "count": 8},
        }
        for index in range(8)
    ]

    shard_manifests = [
        execute_remote(
            runner_config={"kind": "contract"},
            store_config=store.identity(),
            spec_payload=spec.to_dict(),
            workflow_context=context,
        )
        for context in contexts
    ]
    merged = merge_remote_shards(
        runner_config={"kind": "contract"},
        store_config=store.identity(),
        spec_payload=spec.to_dict(),
        shard_manifests=shard_manifests,
        workflow_context={
            "run_id": "wr_sharded_generation",
            "workflow_step_key": "wf.generation",
            "step_name": "generation",
        },
    )

    payload = store.read_json_ref(merged["storage_refs"]["result"])
    assert payload["summary"]["sharded"] is True
    assert payload["summary"]["shard_count"] == 8
    assert [row["example_key"] for row in payload["rows"]] == ["ex_a"]


@pytest.mark.unit
def test_remote_executor_sharded_patched_generation_merges_rows(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
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
    spec = PatchedGenerationSpec(
        engine=ToyEngine(),
        dataset=dataset,
        patch=ProjectOutPatch(
            subspace=subspace,
            write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
            target_tokens=TokenSelector.last(),
            component_indices_by_layer={0: (0,)},
        ),
        select_when=dataset.labels("class").equals("positive"),
        generation=GenerationSpec(enabled=True, max_tokens=1),
    )
    contexts = [
        {
            "run_id": "wr_sharded_patch",
            "workflow_step_key": "wf.patch",
            "step_name": "patch",
            "execution_shard": {"index": index, "count": 8},
        }
        for index in range(8)
    ]

    shard_manifests = [
        execute_remote(
            runner_config={"kind": "contract"},
            store_config=store.identity(),
            spec_payload=spec.to_dict(),
            workflow_context=context,
        )
        for context in contexts
    ]
    merged = merge_remote_shards(
        runner_config={"kind": "contract"},
        store_config=store.identity(),
        spec_payload=spec.to_dict(),
        shard_manifests=shard_manifests,
        workflow_context={
            "run_id": "wr_sharded_patch",
            "workflow_step_key": "wf.patch",
            "step_name": "patch",
        },
    )

    payload = store.read_json_ref(merged["storage_refs"]["result"])
    assert payload["summary"]["sharded"] is True
    assert payload["summary"]["shard_count"] == 8
    assert payload["summary"]["patched_count"] == 1
    assert [row["example_key"] for row in payload["rows"]] == ["ex_a"]


@pytest.mark.unit
def test_remote_executor_many_reuses_one_vllm_session_across_model_bound_specs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    dataset = make_toy_dataset()
    engine = VLLMEngine(model_id="fake-model", enable_prefix_caching=False, enforce_eager=False)
    build_specs: list[list[str]] = []
    runtime_calls: list[str] = []

    class FakeRuntime:
        def __init__(self, spec_kinds: list[str]) -> None:
            self.spec_kinds = spec_kinds

        def capture(self, spec: CaptureSpec) -> EngineCaptureResult:
            runtime_calls.append(f"capture:{len(self.spec_kinds)}")
            rows = [
                {
                    "example_key": example.key,
                    "text": f"capture {example.key}",
                    "generated_token_ids": [1],
                    "finish_reason": "stop",
                    "request_id": f"capture-{example.key}",
                }
                for example in spec.dataset.examples
            ]
            return EngineCaptureResult(features={}, generations=rows, metadata={"backend": "fake_vllm_session"})

        def generate(self, spec: GenerationRunSpec, *, batch_callback: object | None = None) -> EngineGenerationResult:
            runtime_calls.append(f"generate:{len(self.spec_kinds)}")
            rows = [
                {
                    "example_key": example.key,
                    "example": example.to_dict(),
                    "generated_text": f"generation {example.key}",
                    "generated_token_ids": [2],
                    "finish_reason": "stop",
                    "request_id": f"generation-{example.key}",
                }
                for example in spec.dataset.examples
            ]
            if callable(batch_callback):
                batch_callback(rows, {"backend": "fake_vllm_session", "checkpoint": True})
            return EngineGenerationResult(rows=rows, metadata={"backend": "fake_vllm_session"})

        def intervene(self, spec: PatchedGenerationSpec) -> EngineInterventionResult:
            runtime_calls.append(f"patch:{len(self.spec_kinds)}")
            rows = [
                {
                    "example_key": example.key,
                    "example": example.to_dict(),
                    "status": "ok",
                    "skip_reason": "",
                    "generated_text": f"patched {example.key}",
                    "generated_token_ids": [3],
                    "finish_reason": "stop",
                    "request_id": f"patch-{example.key}",
                    "patch_stats": {},
                }
                for example in spec.dataset.examples
            ]
            return EngineInterventionResult(
                summary={"example_count": len(rows), "patched_count": len(rows), "skipped_count": 0},
                rows=rows,
                metadata={"backend": "fake_vllm_session"},
            )

    def fake_build_vllm_session_runtime(
        *,
        engine: VLLMEngine,
        specs: object,
        progress_callback: object | None = None,
    ) -> FakeRuntime:
        del engine, progress_callback
        kinds = [getattr(spec, "kind", "unknown") for spec in specs]
        build_specs.append(kinds)
        return FakeRuntime(kinds)

    monkeypatch.setattr(
        "pipelines_v2.engine.vllm.session.build_vllm_session_runtime",
        fake_build_vllm_session_runtime,
    )

    capture = CaptureSpec(
        engine=engine,
        dataset=dataset,
        generation=GenerationSpec(enabled=True, max_tokens=1),
    )
    generation = GenerationRunSpec(
        engine=engine,
        dataset=dataset,
        generation=GenerationSpec(enabled=True, max_tokens=1),
    )
    patch = PatchedGenerationSpec(
        engine=engine,
        dataset=dataset,
        patch=ProjectOutPatch(
            write_site=ResidualInterventionSite(site="resid_post", layers=(4,)),
            target_tokens=TokenSelector.last(),
            subspace={"layers": {"4": {"components": []}}},
        ),
        generation=GenerationSpec(enabled=True, max_tokens=1),
    )
    paired_patch = PatchedGenerationSpec(
        engine=engine,
        dataset=dataset,
        patch=InterchangePatch(
            activation_bank={"layers": {}},
            write_site=ResidualInterventionSite(site="resid_post", layers=(4,)),
            target_tokens=TokenSelector.last(),
            donor_tokens=TokenSelector.last(),
        ),
        pair_by=dataset.cases("case_key"),
        target_when=dataset.labels("class").equals("positive"),
        donor_when=dataset.labels("class").equals("negative"),
        generation=GenerationSpec(enabled=True, max_tokens=1),
    )

    manifests = execute_remote_many(
        runner_config={"kind": "contract"},
        store_config=store.identity(),
        spec_payloads=[capture.to_dict(), generation.to_dict(), patch.to_dict(), paired_patch.to_dict()],
        workflow_contexts=[
            {"run_id": "wr_vllm_batch", "step_name": "capture", "workflow_step_key": "wf.capture"},
            {"run_id": "wr_vllm_batch", "step_name": "generation", "workflow_step_key": "wf.generation"},
            {"run_id": "wr_vllm_batch", "step_name": "patch", "workflow_step_key": "wf.patch"},
            {"run_id": "wr_vllm_batch", "step_name": "paired_patch", "workflow_step_key": "wf.paired_patch"},
        ],
    )

    assert [manifest["artifact_kind"] for manifest in manifests] == [
        "capture",
        "generation_run",
        "patched_generation",
        "patched_generation",
    ]
    assert build_specs == [["capture", "generation_run", "patched_generation", "patched_generation"]]
    assert runtime_calls == ["capture:4", "generate:4", "patch:4", "patch:4"]


@pytest.mark.unit
def test_modal_vllm_workflow_batch_key_and_runtime_env_support_capture_generation_patch(tmp_path: Path) -> None:
    dataset = make_toy_dataset()
    engine = VLLMEngine(model_id="fake-model", enable_prefix_caching=False, enforce_eager=False)
    capture = CaptureSpec(
        engine=engine,
        dataset=dataset,
        generation=GenerationSpec(enabled=True, max_tokens=1),
    )
    generation = GenerationRunSpec(
        engine=engine,
        dataset=dataset,
        generation=GenerationSpec(enabled=True, max_tokens=1),
    )
    patch = PatchedGenerationSpec(
        engine=engine,
        dataset=dataset,
        patch=ProjectOutPatch(
            write_site=ResidualInterventionSite(site="resid_post", layers=(4,)),
            target_tokens=TokenSelector.last(),
            subspace={"layers": {"4": {"components": []}}},
        ),
        generation=GenerationSpec(enabled=True, max_tokens=1),
    )
    runner = ModalRunner(
        resources=ModalResources(gpu="L4", enable_workflow_batching=True),
        artifacts=ModalVolumeStore(name="xenon-test", root=str(tmp_path / "artifacts")),
    )

    assert runner.workflow_batch_key(capture) == runner.workflow_batch_key(generation)
    assert runner.workflow_batch_key(capture) == runner.workflow_batch_key(patch)

    runtime_spec = _resolved_runtime_spec_many(
        spec_payloads=[capture.to_dict(), generation.to_dict(), patch.to_dict()]
    )
    assert runtime_spec.env["VLLM_COMPILE_CACHE_SAVE_FORMAT"] == "binary"
    assert runtime_spec.env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert runtime_spec.env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert runtime_spec.env["XENON_ACTIVATION_PATCH_COMPILED_OPERATOR"] == "subspace"

    sharded_runner = ModalRunner(
        resources=ModalResources(gpu="L4", shard_count=2, enable_workflow_batching=True),
        artifacts=ModalVolumeStore(name="xenon-test", root=str(tmp_path / "artifacts")),
    )
    assert sharded_runner.workflow_batch_key(capture) == sharded_runner.workflow_batch_key(generation)
    assert sharded_runner.workflow_batch_key(capture) == sharded_runner.workflow_batch_key(patch)
