from __future__ import annotations

from pathlib import Path

import pytest

from pipelines_v2.engine import ToyEngine
from pipelines_v2.runtime import LocalRunner
from pipelines_v2.testing import EngineRunnerContractSuite


@pytest.mark.contract
@pytest.mark.integration_local
def test_toy_engine_satisfies_capture_contract(tmp_path: Path) -> None:
    EngineRunnerContractSuite(
        runner_factory=LocalRunner,
        engine_factory=lambda: ToyEngine(sequence_length=8),
    ).run_capture_contract(tmp_path)


@pytest.mark.contract
@pytest.mark.integration_local
def test_toy_engine_satisfies_generation_contract(tmp_path: Path) -> None:
    EngineRunnerContractSuite(
        runner_factory=LocalRunner,
        engine_factory=lambda: ToyEngine(sequence_length=8),
    ).run_generation_contract(tmp_path)


@pytest.mark.contract
@pytest.mark.integration_local
def test_toy_engine_satisfies_project_out_contract(tmp_path: Path) -> None:
    EngineRunnerContractSuite(
        runner_factory=LocalRunner,
        engine_factory=lambda: ToyEngine(sequence_length=8),
    ).run_project_out_contract(tmp_path)


@pytest.mark.contract
@pytest.mark.integration_local
def test_toy_engine_satisfies_unpaired_patch_operator_contracts(tmp_path: Path) -> None:
    EngineRunnerContractSuite(
        runner_factory=LocalRunner,
        engine_factory=lambda: ToyEngine(sequence_length=8),
    ).run_unpaired_patch_operator_contracts(tmp_path)


@pytest.mark.contract
@pytest.mark.integration_local
def test_toy_engine_satisfies_paired_patch_operator_contracts(tmp_path: Path) -> None:
    EngineRunnerContractSuite(
        runner_factory=LocalRunner,
        engine_factory=lambda: ToyEngine(sequence_length=8),
    ).run_paired_patch_operator_contracts(tmp_path)
