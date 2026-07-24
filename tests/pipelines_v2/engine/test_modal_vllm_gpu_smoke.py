from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pipelines_v2.operations.execution.common import coerce_transform_result


_RUN_ENV = "XENON_RUN_MODAL_VLLM_GPU_SMOKE"
_CONTRACT_RUN_ENV = "XENON_RUN_MODAL_VLLM_ENGINE_CONTRACTS"
_PATCH_OPERATOR_CONTRACT_RUN_ENV = "XENON_RUN_MODAL_VLLM_PATCH_OPERATOR_CONTRACTS"
_PAIRED_PATCH_CONTRACT_RUN_ENV = "XENON_RUN_MODAL_VLLM_PAIRED_PATCH_CONTRACTS"
_CAPTURE_CONTRACT_RUN_ENV = "XENON_RUN_MODAL_VLLM_CAPTURE_CONTRACTS"
_ROUTING_V2_CONTRACT_RUN_ENV = "XENON_RUN_MODAL_VLLM_ROUTING_V2_CONTRACT"
_OUTPUT_CONTRACT_RUN_ENV = "XENON_RUN_MODAL_VLLM_OUTPUT_CONTRACTS"
_TIMEOUT_ENV = "XENON_MODAL_VLLM_GPU_SMOKE_TIMEOUT_SECONDS"
_CONTRACT_SHARD_COUNT_ENV = "XENON_MODAL_VLLM_ENGINE_CONTRACT_SHARD_COUNT"
_CONTRACT_MAX_CONTAINERS_ENV = "XENON_MODAL_VLLM_ENGINE_CONTRACT_MAX_CONTAINERS"
_MODAL_CONTRACT_CACHE: dict[tuple[bool, bool, bool], set[str] | BaseException] = {}


def test_compiled_project_out_smoke_validator_returns_transform_payload() -> None:
    from scripts.pipelines_v2_activation_patch_smoke import validate_compiled_patch_smoke

    raw = validate_compiled_patch_smoke(
        patched={
            "metadata": {"model_runner": "v2"},
            "rows": [
                {
                    "status": "ok",
                    "example_key": "example",
                    "patch_stats": {
                        "24": {
                            "status": "ok",
                            "operator": "project_out",
                            "dispatch": "compiled_custom_op",
                            "token_count": 1,
                            "phase_counts": {"prompt": 0, "decode": 1},
                        }
                    },
                }
            ]
        }
    )

    result = coerce_transform_result(raw)
    assert result.payload["summary"]["patched_count"] == 1
    assert result.payload["summary"]["missing_runtime_stats_count"] == 0
    assert result.payload["summary"]["model_runner"] == "v2"


@pytest.mark.modal
@pytest.mark.vllm
@pytest.mark.network
@pytest.mark.slow
def test_modal_vllm_compiled_project_out_smoke(tmp_path: Path) -> None:
    if os.getenv(_RUN_ENV) != "1":
        pytest.skip(f"set {_RUN_ENV}=1 to run the real Modal/vLLM GPU smoke")
    pytest.importorskip("modal", reason="Modal SDK is required for the real GPU smoke")

    repo_root = Path(__file__).resolve().parents[3]
    workflow_file = repo_root / "scripts" / "pipelines_v2_activation_patch_smoke.py"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["XENON_HOME"] = str(tmp_path / ".xenon")
    timeout_seconds = int(os.getenv(_TIMEOUT_ENV, "7200"))

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipelines_v2.cli",
            "workflow",
            "run",
            "--file",
            str(workflow_file),
            "--logging",
            "INFO",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )

    assert completed.returncode == 0, (
        "Modal/vLLM GPU smoke failed\n"
        f"stdout:\n{completed.stdout[-4000:]}\n"
        f"stderr:\n{completed.stderr[-8000:]}"
    )
    payload = json.loads(completed.stdout)
    validation = payload["steps"]["validate_compiled_patch_stats"]["summary"]

    assert validation["patched_count"] > 0
    assert validation["checked_patch_stat_layers"] > 0
    assert validation["missing_runtime_stats_count"] == 0
    assert validation["operator_counts"] == {"project_out": validation["checked_patch_stat_layers"]}
    assert validation["dispatch_counts"] == {
        "compiled_custom_op": validation["checked_patch_stat_layers"]
    }
    assert validation["status_counts"] == {"ok": validation["checked_patch_stat_layers"]}


@pytest.mark.contract
@pytest.mark.modal
@pytest.mark.vllm
@pytest.mark.network
@pytest.mark.slow
def test_modal_vllm_satisfies_model_bound_runner_contract(tmp_path_factory: pytest.TempPathFactory) -> None:
    if os.getenv(_CONTRACT_RUN_ENV) != "1":
        pytest.skip(f"set {_CONTRACT_RUN_ENV}=1 to run the real Modal/vLLM engine contracts")
    pytest.importorskip("modal", reason="Modal SDK is required for the real GPU contract")

    coverage = _run_requested_modal_contracts(tmp_path_factory)
    assert "basic" in coverage


@pytest.mark.contract
@pytest.mark.modal
@pytest.mark.vllm
@pytest.mark.network
@pytest.mark.slow
def test_modal_vllm_satisfies_unpaired_patch_operator_contracts(tmp_path_factory: pytest.TempPathFactory) -> None:
    if os.getenv(_PATCH_OPERATOR_CONTRACT_RUN_ENV) != "1":
        pytest.skip(
            f"set {_PATCH_OPERATOR_CONTRACT_RUN_ENV}=1 to run real Modal/vLLM unpaired patch contracts"
        )
    pytest.importorskip("modal", reason="Modal SDK is required for the real GPU contract")

    coverage = _run_requested_modal_contracts(tmp_path_factory)
    assert "unpaired" in coverage


@pytest.mark.contract
@pytest.mark.modal
@pytest.mark.vllm
@pytest.mark.network
@pytest.mark.slow
def test_modal_vllm_satisfies_paired_patch_operator_contracts(tmp_path_factory: pytest.TempPathFactory) -> None:
    if os.getenv(_PAIRED_PATCH_CONTRACT_RUN_ENV) != "1":
        pytest.skip(
            f"set {_PAIRED_PATCH_CONTRACT_RUN_ENV}=1 to run real Modal/vLLM paired patch contracts"
        )
    pytest.importorskip("modal", reason="Modal SDK is required for the real GPU contract")

    coverage = _run_requested_modal_contracts(tmp_path_factory)
    assert "paired" in coverage


@pytest.mark.contract
@pytest.mark.modal
@pytest.mark.vllm
@pytest.mark.network
@pytest.mark.slow
def test_modal_vllm_generated_residual_and_moe_routing_contract(tmp_path: Path) -> None:
    if os.getenv(_CAPTURE_CONTRACT_RUN_ENV) != "1":
        pytest.skip(
            f"set {_CAPTURE_CONTRACT_RUN_ENV}=1 to run the generated-residual/MoE capture contract"
        )
    pytest.importorskip("modal", reason="Modal SDK is required for the real GPU contract")

    from pipelines_v2.api import (
        CaptureSpec,
        Dataset,
        Example,
        GenerationSpec,
        MoERoutingSite,
        ModalResources,
        ModalRunner,
        ModalVolumeMount,
        ModalVolumeStore,
        ResidualSite,
        RoutingRecord,
        TokenSelector,
        VLLMEngine,
    )

    model_id = os.getenv(
        "XENON_MODAL_VLLM_ENGINE_CONTRACT_MODEL_ID",
        "/models/Qwen/Qwen3-30B-A3B",
    )
    model_volume = os.getenv("XENON_MODAL_MODEL_VOLUME", "xenon-models")
    model_mount = os.getenv("XENON_MODAL_MODEL_VOLUME_PATH", "/models")
    dataset = Dataset.from_examples(
        [
            Example(
                key="capture_contract_buy",
                prompt="State three short reasons a cautious investor might buy a diversified fund.",
            ),
            Example(
                key="capture_contract_hold",
                prompt="State three short reasons a cautious investor might hold cash.",
            ),
        ],
        name="vllm_generated_residual_moe_contract",
    )
    engine = VLLMEngine(
        model_id=model_id,
        max_model_len=512,
        enforce_eager=False,
        max_num_seqs=2,
        max_num_batched_tokens=1024,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    runner = ModalRunner(
        resources=ModalResources(
            gpu=os.getenv("XENON_MODAL_VLLM_ENGINE_CONTRACT_GPU", "A100-80GB"),
            timeout_seconds=int(os.getenv(_TIMEOUT_ENV, "7200")),
            max_containers=1,
            volumes=(ModalVolumeMount(name=model_volume, mount_path=model_mount),),
        ),
        artifacts=ModalVolumeStore(
            name="xenon-data",
            root="/data/artifacts/pipelines_v2_vllm_capture_contract",
            local_cache_root=tmp_path / "modal_cache",
        ),
    )
    artifact = runner.run(
        CaptureSpec(
            engine=engine,
            dataset=dataset,
            sites=(
                ResidualSite(
                    name="generated_residual",
                    site="resid_post",
                    layers=(0, 24),
                    tokens=TokenSelector.full_sequence(),
                ),
                MoERoutingSite(
                    name="moe_routing",
                    layers=(0, 24),
                    tokens=TokenSelector.full_sequence(),
                    record=(
                        RoutingRecord.gate_logits(dtype="float16"),
                        RoutingRecord.routing_decisions(required=True),
                    ),
                ),
            ),
            generation=GenerationSpec(
                enabled=True,
                max_tokens=16,
                temperature=0.0,
                capture_generated_tokens=True,
            ),
        )
    )

    metadata = artifact.manifest().metadata
    assert metadata["model_runner"] == "v1"
    assert metadata["router_enabled"] is True
    assert {0, 24}.issubset(set(metadata["discovered_router_layers"]))
    example_metadata = list(metadata["example_metadata"])
    assert len(example_metadata) == 2
    assert all(int(row["generated_token_count"]) > 0 for row in example_metadata)
    assert all(int(row["captured_generated_token_count"]) > 0 for row in example_metadata)
    performance = dict(metadata.get("performance") or {})
    if performance:
        assert int(performance["request_count"]) == 2
        assert int(performance["generated_tokens"]) > 0
        assert float(performance["generation_seconds"]) > 0.0


@pytest.mark.contract
@pytest.mark.modal
@pytest.mark.vllm
@pytest.mark.network
@pytest.mark.slow
def test_modal_vllm_moe_routing_only_model_runner_v2_contract(
    tmp_path: Path,
) -> None:
    if os.getenv(_ROUTING_V2_CONTRACT_RUN_ENV) != "1":
        pytest.skip(
            f"set {_ROUTING_V2_CONTRACT_RUN_ENV}=1 to run the Model Runner V2 "
            "routing-only contract"
        )
    pytest.importorskip("modal", reason="Modal SDK is required for the real GPU contract")

    from pipelines_v2.api import (
        CaptureSpec,
        Dataset,
        Example,
        GenerationSpec,
        MoERoutingSite,
        ModalResources,
        ModalRunner,
        ModalVolumeMount,
        ModalVolumeStore,
        RoutingRecord,
        TokenSelector,
        VLLMEngine,
    )

    runner = ModalRunner(
        resources=ModalResources(
            gpu=os.getenv("XENON_MODAL_VLLM_ENGINE_CONTRACT_GPU", "A100-80GB"),
            timeout_seconds=int(os.getenv(_TIMEOUT_ENV, "7200")),
            max_containers=1,
            volumes=(
                ModalVolumeMount(
                    name=os.getenv("XENON_MODAL_MODEL_VOLUME", "xenon-models"),
                    mount_path=os.getenv("XENON_MODAL_MODEL_VOLUME_PATH", "/models"),
                ),
            ),
        ),
        artifacts=ModalVolumeStore(
            name="xenon-data",
            root="/data/artifacts/pipelines_v2_vllm_routing_v2_contract",
            local_cache_root=tmp_path / "modal_cache",
        ),
    )
    artifact = runner.run(
        CaptureSpec(
            engine=VLLMEngine(
                model_id=os.getenv(
                    "XENON_MODAL_VLLM_ENGINE_CONTRACT_MODEL_ID",
                    "/models/Qwen/Qwen3-30B-A3B",
                ),
                max_model_len=256,
                enforce_eager=False,
                max_num_seqs=2,
                max_num_batched_tokens=512,
                enable_prefix_caching=False,
                enable_chunked_prefill=True,
                enable_thinking=False,
            ),
            dataset=Dataset.from_examples(
                [
                    Example(
                        key="routing_v2_a",
                        prompt="Name one advantage of diversification.",
                    ),
                    Example(
                        key="routing_v2_b",
                        prompt="Name one advantage of holding cash.",
                    ),
                ],
                name="vllm_routing_v2_contract",
            ),
            sites=(
                MoERoutingSite(
                    name="moe_routing",
                    layers=(0, 24),
                    tokens=TokenSelector.full_sequence(),
                    record=(
                        RoutingRecord.gate_logits(dtype="float16"),
                        RoutingRecord.routing_decisions(required=True),
                    ),
                ),
            ),
            generation=GenerationSpec(
                enabled=True,
                max_tokens=8,
                temperature=0.0,
            ),
        )
    )

    metadata = artifact.manifest().metadata
    assert metadata["model_runner"] == "v2"
    assert metadata["router_enabled"] is True
    assert {0, 24}.issubset(set(metadata["discovered_router_layers"]))
    assert all(
        int(row["generated_token_count"]) > 0
        for row in metadata["example_metadata"]
    )


@pytest.mark.contract
@pytest.mark.modal
@pytest.mark.vllm
@pytest.mark.network
@pytest.mark.slow
def test_modal_vllm_structured_output_contract(tmp_path: Path) -> None:
    if os.getenv(_OUTPUT_CONTRACT_RUN_ENV) != "1":
        pytest.skip(f"set {_OUTPUT_CONTRACT_RUN_ENV}=1 to run the structured-output contract")
    pytest.importorskip("modal", reason="Modal SDK is required for the real GPU contract")

    from pipelines_v2.api import (
        Dataset,
        Example,
        GenerationRunSpec,
        GenerationSpec,
        ModalResources,
        ModalRunner,
        ModalVolumeMount,
        ModalVolumeStore,
        VLLMEngine,
    )

    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["hold"]},
            "confidence": {"type": "integer", "enum": [73]},
        },
        "required": ["action", "confidence"],
        "additionalProperties": False,
    }
    runner = ModalRunner(
        resources=ModalResources(
            gpu=os.getenv("XENON_MODAL_VLLM_ENGINE_CONTRACT_GPU", "A100-80GB"),
            timeout_seconds=int(os.getenv(_TIMEOUT_ENV, "7200")),
            max_containers=1,
            volumes=(
                ModalVolumeMount(
                    name=os.getenv("XENON_MODAL_MODEL_VOLUME", "xenon-models"),
                    mount_path=os.getenv("XENON_MODAL_MODEL_VOLUME_PATH", "/models"),
                ),
            ),
        ),
        artifacts=ModalVolumeStore(
            name="xenon-data",
            root="/data/artifacts/pipelines_v2_vllm_structured_output_contract",
            local_cache_root=tmp_path / "modal_cache",
        ),
    )
    artifact = runner.run(
        GenerationRunSpec(
            engine=VLLMEngine(
                model_id=os.getenv(
                    "XENON_MODAL_VLLM_ENGINE_CONTRACT_MODEL_ID",
                    "/models/Qwen/Qwen3-30B-A3B",
                ),
                max_model_len=512,
                enforce_eager=False,
                max_num_seqs=2,
                max_num_batched_tokens=1024,
                enable_prefix_caching=False,
                enable_chunked_prefill=True,
                add_generation_prompt=False,
                enable_thinking=False,
            ),
            dataset=Dataset.from_examples(
                [
                    Example(
                        key="structured_output",
                        prompt="Return the requested investment decision as a JSON object.",
                    ),
                ],
                name="vllm_structured_output_contract",
            ),
            generation=GenerationSpec(
                enabled=True,
                max_tokens=32,
                temperature=0.0,
                structured_output=schema,
            ),
        )
    )

    rows = list(artifact.result()["rows"])
    assert artifact.manifest().metadata["model_runner"] == "v2"
    assert len(rows) == 1
    parsed = json.loads(str(rows[0]["generated_text"]))
    assert parsed == {"action": "hold", "confidence": 73}


def _run_requested_modal_contracts(tmp_path_factory: pytest.TempPathFactory) -> set[str]:
    requested = (
        os.getenv(_CONTRACT_RUN_ENV) == "1",
        os.getenv(_PATCH_OPERATOR_CONTRACT_RUN_ENV) == "1",
        os.getenv(_PAIRED_PATCH_CONTRACT_RUN_ENV) == "1",
    )
    cached = _MODAL_CONTRACT_CACHE.get(requested)
    if isinstance(cached, BaseException):
        raise cached
    if cached is not None:
        return set(cached)
    tmp_path = tmp_path_factory.mktemp("modal_vllm_engine_contracts")
    try:
        coverage = _modal_contract_suite(tmp_path).run_requested_model_bound_contracts(
            tmp_path,
            basic=requested[0],
            unpaired=requested[1],
            paired=requested[2],
        )
    except BaseException as exc:
        _MODAL_CONTRACT_CACHE[requested] = exc
        raise
    _MODAL_CONTRACT_CACHE[requested] = set(coverage)
    return set(coverage)


def _modal_contract_suite(tmp_path: Path) -> object:
    from pipelines_v2.api import ModalResources, ModalRunner, ModalVolumeMount, ModalVolumeStore, VLLMEngine
    from pipelines_v2.testing import EngineRunnerContractSuite

    model_id = os.getenv("XENON_MODAL_VLLM_ENGINE_CONTRACT_MODEL_ID", "/models/Qwen/Qwen3-30B-A3B")
    gpu = os.getenv("XENON_MODAL_VLLM_ENGINE_CONTRACT_GPU", "A100-80GB")
    artifact_root = os.getenv(
        "XENON_MODAL_VLLM_ENGINE_CONTRACT_ARTIFACT_ROOT",
        "/data/artifacts/pipelines_v2_engine_contracts",
    )
    model_volume = os.getenv("XENON_MODAL_MODEL_VOLUME", "xenon-models")
    model_mount = os.getenv("XENON_MODAL_MODEL_VOLUME_PATH", "/models")

    def engine_factory() -> VLLMEngine:
        return VLLMEngine(
            model_id=model_id,
            max_model_len=1024,
            enforce_eager=False,
            max_num_seqs=2,
            max_num_batched_tokens=4096,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )

    def runner_factory(*, artifacts: object, catalog: object) -> ModalRunner:
        del artifacts
        shard_count = int(os.getenv(_CONTRACT_SHARD_COUNT_ENV, "2"))
        max_containers = int(os.getenv(_CONTRACT_MAX_CONTAINERS_ENV, str(max(1, shard_count))))
        return ModalRunner(
            resources=ModalResources(
                gpu=gpu,
                timeout_seconds=int(os.getenv(_TIMEOUT_ENV, "7200")),
                max_containers=max_containers,
                shard_count=shard_count,
                enable_workflow_batching=True,
                volumes=(ModalVolumeMount(name=model_volume, mount_path=model_mount),),
            ),
            artifacts=ModalVolumeStore(
                name="xenon-data",
                root=artifact_root,
                local_cache_root=tmp_path / "modal_cache",
            ),
            catalog=catalog,
        )

    return EngineRunnerContractSuite(
        runner_factory=runner_factory,
        engine_factory=engine_factory,
        expected_patch_dispatch="compiled_custom_op",
    )
