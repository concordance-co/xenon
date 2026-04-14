"""Contract-test helpers for v2 implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from pipelines_v2.core.types import EngineCapability
from pipelines_v2.data.datasets import Dataset, Example
from pipelines_v2.engine import Engine, ToyEngine
from pipelines_v2.operations.specs import CaptureSpec, GenerationSpec, ResidualSite, TokenSelector
from pipelines_v2.runtime import Runner
from pipelines_v2.storage.artifacts import ArtifactManifest
from pipelines_v2.storage import FileCatalog, LocalArtifactStore


def make_toy_dataset() -> Dataset:
    return Dataset.from_examples(
        [
            Example(
                key="ex_a",
                prompt="alpha beta gamma",
                labels={"class": "positive"},
                case_key="case_1",
            ),
            Example(
                key="ex_b",
                prompt="delta epsilon",
                labels={"class": "negative"},
                case_key="case_1",
            ),
        ],
        name="toy_dataset",
    )


def make_toy_capture_spec(dataset: Dataset | None = None) -> CaptureSpec:
    return CaptureSpec(
        engine=ToyEngine(),
        dataset=dataset or make_toy_dataset(),
        sites=[
            ResidualSite(
                name="resid_last",
                site="resid_post",
                layers=[0, 1],
                tokens=TokenSelector.last(),
            )
        ],
        generation=GenerationSpec(enabled=True, max_tokens=4),
    )


def assert_artifact_manifest_valid(manifest: ArtifactManifest | dict[str, Any]) -> None:
    data = manifest.to_dict() if isinstance(manifest, ArtifactManifest) else manifest
    required = {
        "artifact_id",
        "artifact_kind",
        "schema_version",
        "operation_spec_hash",
        "operation_semantic_hash",
        "created_at",
        "engine",
        "runner",
        "input_artifact_refs",
        "example_coverage",
        "storage_refs",
        "metadata",
        "workflow_context",
    }
    missing = required - set(data)
    assert not missing, f"Artifact manifest missing keys: {sorted(missing)}"
    assert data["artifact_id"]
    assert data["artifact_kind"]
    assert data["operation_spec_hash"]
    assert isinstance(data["engine"], dict)
    assert isinstance(data["runner"], dict)
    assert isinstance(data["storage_refs"], dict)


@dataclass(frozen=True, slots=True)
class EngineContractSuite:
    engine_factory: Callable[[], Engine]

    def run(
        self,
        *,
        required_capabilities: Iterable[EngineCapability] = (),
        optional_capabilities: Iterable[EngineCapability] = (),
    ) -> None:
        engine = self.engine_factory()
        caps = engine.capabilities()
        assert set(required_capabilities).issubset(caps)
        assert set(optional_capabilities).intersection(caps) <= caps
        identity = engine.identity()
        assert identity["kind"]


@dataclass(frozen=True, slots=True)
class RunnerContractSuite:
    runner_factory: Callable[..., Runner]

    def run_capture_smoke(self, tmp_path: str | Path) -> None:
        store = LocalArtifactStore(Path(tmp_path) / "artifacts")
        catalog = FileCatalog(Path(tmp_path) / "catalog")
        runner = self.runner_factory(artifacts=store, catalog=catalog)
        artifact = runner.run(make_toy_capture_spec())
        assert_artifact_manifest_valid(artifact.manifest())
        assert artifact.localize().exists()
        assert artifact.feature("resid_last").load()["kind"] == "residual"


@dataclass(frozen=True, slots=True)
class SourceContractSuite:
    source_factory: Callable[[], Any]

    def run_fetch_dataset_smoke(self) -> None:
        dataset = self.source_factory().fetch_dataset(
            prompt_column="prompt",
            example_key_column="example_id",
            label_columns=["class"],
            case_key_column="case_id",
        )
        assert dataset.examples
        assert dataset.labels("class").values
        assert dataset.cases("case_id").values


@dataclass(frozen=True, slots=True)
class ArtifactStoreContractSuite:
    store_factory: Callable[[Path], Any]

    def run_json_roundtrip(self, tmp_path: str | Path) -> None:
        store = self.store_factory(Path(tmp_path) / "artifacts")
        artifact_id = "contract_artifact"
        store.make_artifact_dir(artifact_id)
        ref = store.write_json(artifact_id, "payload.json", {"ok": True})
        assert store.read_json_ref(ref) == {"ok": True}


@dataclass(frozen=True, slots=True)
class CatalogContractSuite:
    catalog_factory: Callable[[Path], Any]

    def run_record_smoke(self, tmp_path: str | Path) -> None:
        manifest = ArtifactManifest(
            artifact_id="contract_artifact",
            artifact_kind="capture",
            schema_version=1,
            operation_spec_hash="abc",
            operation_semantic_hash="abc",
            created_at="2026-01-01T00:00:00+00:00",
            engine={"kind": "toy"},
            runner={"kind": "local"},
            input_artifact_refs=(),
            example_coverage={"example_count": 0},
            storage_refs={},
            metadata={},
            workflow_context={},
        )
        catalog = self.catalog_factory(Path(tmp_path) / "catalog")
        catalog.record_artifact(manifest)
