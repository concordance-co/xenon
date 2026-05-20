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
_TIMEOUT_ENV = "XENON_MODAL_VLLM_GPU_SMOKE_TIMEOUT_SECONDS"
_CONTRACT_SHARD_COUNT_ENV = "XENON_MODAL_VLLM_ENGINE_CONTRACT_SHARD_COUNT"
_CONTRACT_MAX_CONTAINERS_ENV = "XENON_MODAL_VLLM_ENGINE_CONTRACT_MAX_CONTAINERS"
_MODAL_CONTRACT_CACHE: dict[tuple[bool, bool, bool], set[str] | BaseException] = {}


def test_compiled_project_out_smoke_validator_returns_transform_payload() -> None:
    from scripts.pipelines_v2_activation_patch_smoke import validate_compiled_patch_smoke

    raw = validate_compiled_patch_smoke(
        patched={
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
                        }
                    },
                }
            ]
        }
    )

    result = coerce_transform_result(raw)
    assert result.payload["summary"]["patched_count"] == 1
    assert result.payload["summary"]["missing_runtime_stats_count"] == 0


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
