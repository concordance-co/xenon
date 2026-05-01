from __future__ import annotations

from pathlib import Path

import pytest

from pipelines_v2.core.types import TransferPolicy, TransferPolicyError
from pipelines_v2.storage import ArtifactManifest, FileCatalog, LocalArtifactStore, ModalVolumeStore
from pipelines_v2.storage.composite import CompositeCatalog, preferred_workflow_metadata_catalog
from pipelines_v2.testing import ArtifactStoreContractSuite, CatalogContractSuite
from pipelines_v2.workflow.records import WorkflowRunRecord


@pytest.mark.contract
@pytest.mark.unit
def test_local_artifact_store_satisfies_full_store_contract(tmp_path: Path) -> None:
    ArtifactStoreContractSuite(LocalArtifactStore).run_full_roundtrip(tmp_path)


@pytest.mark.contract
@pytest.mark.unit
def test_file_catalog_satisfies_workflow_index_contract(tmp_path: Path) -> None:
    CatalogContractSuite(FileCatalog).run_workflow_index_roundtrip(tmp_path)


@pytest.mark.unit
def test_local_artifact_store_rejects_absolute_and_parent_relative_writes(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    store.make_artifact_dir("artifact")

    with pytest.raises(ValueError, match="outside artifact root"):
        store.write_json("artifact", "../escape.json", {"bad": True})

    with pytest.raises(ValueError, match="outside artifact root"):
        store.write_json("artifact", str(tmp_path / "outside.json"), {"bad": True})

    assert not (tmp_path / "escape.json").exists()
    assert not (tmp_path / "outside.json").exists()


@pytest.mark.contract
@pytest.mark.integration_remote_fake
def test_modal_volume_store_satisfies_local_mode_store_contract(tmp_path: Path) -> None:
    def _store_factory(root: Path) -> ModalVolumeStore:
        return ModalVolumeStore(
            name="xenon-contract",
            root=str(root),
            local_cache_root=tmp_path / "cache",
        )

    ArtifactStoreContractSuite(_store_factory).run_full_roundtrip(tmp_path)


@pytest.mark.unit
def test_modal_volume_store_transfer_policy_blocks_unknown_or_large_remote_downloads(tmp_path: Path) -> None:
    strict = ModalVolumeStore(
        name="xenon-contract",
        root=str(tmp_path / "remote-root"),
        transfer_policy=TransferPolicy(max_download_bytes=10),
    )
    with pytest.raises(TransferPolicyError, match="Cannot estimate transfer size"):
        strict.validate_transfer(bytes=None, label="unknown remote ref")
    with pytest.raises(TransferPolicyError, match="exceeds max_download_bytes=10"):
        strict.validate_transfer(bytes=11, label="large remote ref")

    permissive = ModalVolumeStore(
        name="xenon-contract",
        root=str(tmp_path / "remote-root"),
        transfer_policy=TransferPolicy(allow_large_transfer=True, max_download_bytes=1),
    )
    permissive.validate_transfer(bytes=None, label="unknown remote ref")
    permissive.validate_transfer(bytes=10_000, label="large remote ref")


@pytest.mark.contract
@pytest.mark.unit
def test_composite_catalog_satisfies_workflow_index_contract(tmp_path: Path) -> None:
    CatalogContractSuite(
        lambda root: CompositeCatalog(
            (
                FileCatalog(root / "primary"),
                FileCatalog(root / "secondary"),
            )
        )
    ).run_workflow_index_roundtrip(tmp_path)


@pytest.mark.unit
def test_composite_catalog_reads_artifacts_in_configured_order_and_prefers_file_metadata(tmp_path: Path) -> None:
    primary = FileCatalog(tmp_path / "primary")
    secondary = FileCatalog(tmp_path / "secondary")
    composite = CompositeCatalog((secondary, primary))
    assert preferred_workflow_metadata_catalog(composite) is secondary

    CatalogContractSuite(lambda root: primary).run_record_smoke(tmp_path / "primary_contract")
    manifest = primary.load_artifact("contract_artifact")
    assert manifest is not None
    secondary.record_artifact(ArtifactManifest.from_dict({**manifest.to_dict(), "artifact_kind": "secondary_hit"}))

    loaded = composite.load_artifact("contract_artifact")
    assert loaded is not None
    assert loaded.artifact_kind == "secondary_hit"


@pytest.mark.unit
def test_composite_catalog_deduplicates_workflow_runs_across_backends(tmp_path: Path) -> None:
    first = FileCatalog(tmp_path / "first")
    second = FileCatalog(tmp_path / "second")
    duplicate = WorkflowRunRecord(
        run_id="wr_duplicate",
        workflow_name="wf",
        workflow_hash="hash",
        workflow_spec_hash="spec",
        workflow_payload={"kind": "workflow", "steps": []},
        status="completed",
        started_at="2026-01-01T00:00:00+00:00",
    )
    newer = WorkflowRunRecord(
        run_id="wr_newer",
        workflow_name="wf",
        workflow_hash="hash",
        workflow_spec_hash="spec",
        workflow_payload={"kind": "workflow", "steps": []},
        status="completed",
        started_at="2026-01-01T00:00:01+00:00",
    )
    first.record_workflow_run(duplicate)
    second.record_workflow_run(duplicate)
    second.record_workflow_run(newer)

    composite = CompositeCatalog((first, second))
    rows = composite.list_workflow_runs(workflow_name="wf")

    assert [row.run_id for row in rows] == ["wr_duplicate"]

    first_empty = FileCatalog(tmp_path / "first_empty")
    fallback = CompositeCatalog((first_empty, second))
    fallback_rows = fallback.list_workflow_runs(workflow_name="wf")

    assert [row.run_id for row in fallback_rows] == ["wr_newer", "wr_duplicate"]
