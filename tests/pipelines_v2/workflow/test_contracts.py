from __future__ import annotations

from pathlib import Path

import pytest

from pipelines_v2.testing import WorkflowContractSuite


@pytest.mark.contract
@pytest.mark.integration_local
def test_workflow_orchestrator_satisfies_resume_and_reuse_contract(tmp_path: Path) -> None:
    WorkflowContractSuite().run_resume_and_reuse_contract(tmp_path)


@pytest.mark.contract
@pytest.mark.integration_local
def test_workflow_orchestrator_satisfies_failure_blocking_and_resume_contract(tmp_path: Path) -> None:
    WorkflowContractSuite().run_failure_blocks_downstream_contract(tmp_path)


@pytest.mark.contract
@pytest.mark.integration_local
def test_workflow_orchestrator_batches_compatible_ready_steps(tmp_path: Path) -> None:
    WorkflowContractSuite().run_ready_step_batching_contract(tmp_path)
