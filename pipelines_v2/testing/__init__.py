"""Contract-test helpers for v2 implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from pipelines_v2.core.types import EngineCapability, OperationSpec
from pipelines_v2.data.datasets import Dataset, Example
from pipelines_v2.engine import Engine, ToyEngine
from pipelines_v2.operations.derive import LabelMapSpec
from pipelines_v2.operations.interventions import (
    ActivationBankSpec,
    AddDirectionPatch,
    ExplicitPathEdge,
    ExplicitPathMaskSpec,
    InterchangePatch,
    PatchedGenerationSpec,
    ProjectOutPatch,
    RandomControlPatch,
    ResidualInterventionSite,
    ResidualPathPatch,
    SwapComponentsPatch,
    SwapMeanPatch,
)
from pipelines_v2.operations.representation import CentroidSpec, DirectionSpec, SubspaceSpec
from pipelines_v2.operations.specs import CaptureSpec, GenerationRunSpec, GenerationSpec, ResidualSite, TokenPooling, TokenSelector
from pipelines_v2.runtime import LocalRunner, Runner
from pipelines_v2.storage.artifacts import ArtifactManifest, InlineOperationArtifact
from pipelines_v2.storage import FileCatalog, LocalArtifactStore
from pipelines_v2.workflow.records import WorkflowRunRecord, WorkflowStepContext, WorkflowStepRecord
from pipelines_v2.workflow.specs import StepRef, WorkflowSpec, WorkflowStep
from pipelines_v2.workflow.orchestrator import WorkflowOrchestrator


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


def make_engine_contract_dataset() -> Dataset:
    return Dataset.from_examples(
        [
            Example(
                key="contract_positive",
                prompt="alpha beta gamma",
                labels={"class": "positive"},
                metadata={"token_sections": {"BODY": [0, 1, 2]}},
                case_key="contract_case",
            ),
            Example(
                key="contract_negative",
                prompt="delta epsilon zeta",
                labels={"class": "negative"},
                metadata={"token_sections": {"BODY": [0, 1, 2]}},
                case_key="contract_case",
            ),
        ],
        name="engine_contract_dataset",
    )


def make_engine_contract_capture_spec(
    *,
    engine: Engine,
    dataset: Dataset | None = None,
    capture_generation: bool = False,
) -> CaptureSpec:
    return CaptureSpec(
        engine=engine,
        dataset=dataset or make_engine_contract_dataset(),
        sites=[
            ResidualSite(
                name="contract_residual",
                site="resid_post",
                layers=[0],
                tokens=TokenSelector.section("BODY"),
            )
        ],
        generation=GenerationSpec(enabled=bool(capture_generation), max_tokens=2),
    )


def make_engine_contract_generation_spec(
    *,
    engine: Engine,
    dataset: Dataset | None = None,
) -> GenerationRunSpec:
    dataset = dataset or make_engine_contract_dataset()
    return GenerationRunSpec(
        engine=engine,
        dataset=dataset,
        select_when=dataset.labels("class").equals("positive"),
        generation=GenerationSpec(enabled=True, max_tokens=2, temperature=0.0),
    )


def _inline_project_out_subspace(*, engine: Engine, layer: int) -> InlineOperationArtifact:
    identity = dict(engine.identity())
    if identity.get("kind") == "toy":
        hidden_size = int(identity.get("hidden_size") or 4)
        component = [0.0] * hidden_size
        component[0] = 1.0
        mean: Any = [0.0] * hidden_size
        scale: Any = [1.0] * hidden_size
        safe_scale: Any = [1.0] * hidden_size
        components: Any = [component]
    else:
        mean = {"kind": "xenon_runtime_zeros"}
        scale = {"kind": "xenon_runtime_ones"}
        safe_scale = {"kind": "xenon_runtime_ones"}
        components = {"kind": "xenon_runtime_unit_basis", "indices": [0]}
    return InlineOperationArtifact(
        payload={
            "kind": "subspace_result",
            "feature": "runtime_contract_subspace",
            "layers": {
                str(int(layer)): {
                    "method": "runtime_unit_basis",
                    "mean": mean,
                    "scale": scale,
                    "safe_scale": safe_scale,
                    "components": components,
                    "explained_variance_ratio": [1.0],
                    "example_count": 0,
                    "component_count": 1,
                    "named_components": {},
                }
            },
            "summary": {
                "layer_count": 1,
                "component_count": 1,
                "method": "runtime_unit_basis",
            },
        },
        artifact_kind="subspace",
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


def _contract_manifest(
    artifact_id: str,
    *,
    created_at: str = "2026-01-01T00:00:00+00:00",
    run_id: str = "wr_contract",
    workflow_hash: str = "workflow_hash",
    step_name: str = "step",
    step_index: int = 0,
    operation_semantic_hash: str = "semantic_hash",
    input_artifact_refs: tuple[str, ...] = (),
) -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id=artifact_id,
        artifact_kind="contract",
        schema_version=1,
        operation_spec_hash="spec_hash",
        operation_semantic_hash=operation_semantic_hash,
        created_at=created_at,
        engine={},
        runner={"kind": "contract"},
        input_artifact_refs=input_artifact_refs,
        example_coverage={"materialized": True, "example_count": 0, "example_keys": []},
        storage_refs={},
        metadata={},
        workflow_context={
            "run_id": run_id,
            "workflow_name": "contract_workflow",
            "workflow_hash": workflow_hash,
            "workflow_spec_hash": "workflow_spec_hash",
            "step_name": step_name,
            "step_index": step_index,
            "runner": "contract",
            "workflow_step_key": f"{workflow_hash}.{step_name}",
            "step_semantic_hash": operation_semantic_hash,
            "step_spec_hash": "step_spec_hash",
        },
    )


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
class EngineRunnerContractSuite:
    runner_factory: Callable[..., Runner]
    engine_factory: Callable[[], Engine]
    expected_patch_dispatch: str | None = None

    def _runner(self, tmp_path: str | Path) -> Runner:
        store = LocalArtifactStore(Path(tmp_path) / "artifacts")
        catalog = FileCatalog(Path(tmp_path) / "catalog")
        return self.runner_factory(artifacts=store, catalog=catalog)

    def run_capture_contract(self, tmp_path: str | Path) -> None:
        runner = self._runner(tmp_path)
        dataset = make_engine_contract_dataset()
        self._run_capture_contract(runner=runner, dataset=dataset)

    def run_generation_contract(self, tmp_path: str | Path) -> None:
        runner = self._runner(tmp_path)
        dataset = make_engine_contract_dataset()
        self._run_generation_contract(runner=runner, dataset=dataset)

    def run_project_out_contract(self, tmp_path: str | Path) -> None:
        runner = self._runner(tmp_path)
        dataset = make_engine_contract_dataset()
        capture = self._run_capture_contract(runner=runner, dataset=dataset)
        subspace = self._run_subspace_contract(runner=runner, capture=capture)
        self._run_project_out_contract(runner=runner, dataset=dataset, subspace=subspace)

    def run_capture_generation_and_project_out_contract(self, tmp_path: str | Path) -> None:
        runner = self._runner(tmp_path)
        dataset = make_engine_contract_dataset()
        capture = self._run_capture_contract(runner=runner, dataset=dataset)
        self._run_generation_contract(runner=runner, dataset=dataset)
        subspace = self._run_subspace_contract(runner=runner, capture=capture)
        self._run_project_out_contract(runner=runner, dataset=dataset, subspace=subspace)

    def run_batched_capture_generation_and_project_out_contract(self, tmp_path: str | Path) -> None:
        runner = self._runner(tmp_path)
        dataset = make_engine_contract_dataset()
        engine = self.engine_factory()
        specs = (
            make_engine_contract_capture_spec(
                engine=engine,
                dataset=dataset,
            ),
            make_engine_contract_generation_spec(
                engine=engine,
                dataset=dataset,
            ),
            self._project_out_spec(
                engine=engine,
                dataset=dataset,
                subspace=_inline_project_out_subspace(engine=engine, layer=0),
            ),
        )
        run_many = getattr(runner, "run_many", None)
        if callable(run_many):
            capture, generation, patched = run_many(specs)
        else:
            capture, generation, patched = [runner.run(spec) for spec in specs]

        self._assert_capture_artifact(capture)
        self._assert_generation_artifact(generation)
        self._assert_patch_artifact(
            patched,
            expected_operator="project_out",
            expected_example_key="contract_positive",
            expected_target_tokens=[0, 1, 2],
            expected_dispatch=self.expected_patch_dispatch,
        )

    def run_requested_model_bound_contracts(
        self,
        tmp_path: str | Path,
        *,
        basic: bool = True,
        unpaired: bool = True,
        paired: bool = True,
    ) -> set[str]:
        """Run requested model-bound contracts with minimal vLLM session reloads."""

        requested = {
            name
            for name, enabled in {
                "basic": basic,
                "unpaired": unpaired,
                "paired": paired,
            }.items()
            if enabled
        }
        if not requested:
            return set()

        runner = self._runner(tmp_path)
        dataset = make_engine_contract_dataset()
        capture = None
        covered: set[str] = set()

        if basic:
            engine = self.engine_factory()
            specs = (
                make_engine_contract_capture_spec(
                    engine=engine,
                    dataset=dataset,
                ),
                make_engine_contract_generation_spec(
                    engine=engine,
                    dataset=dataset,
                ),
                self._project_out_spec(
                    engine=engine,
                    dataset=dataset,
                    subspace=_inline_project_out_subspace(engine=engine, layer=0),
                ),
            )
            capture, generation, patched = self._run_model_bound_group(runner=runner, specs=specs)
            self._assert_capture_artifact(capture)
            self._assert_generation_artifact(generation)
            self._assert_patch_artifact(
                patched,
                expected_operator="project_out",
                expected_example_key="contract_positive",
                expected_target_tokens=[0, 1, 2],
                expected_dispatch=self.expected_patch_dispatch,
            )
            covered.add("basic")

        patch_specs: list[PatchedGenerationSpec] = []
        expectations: list[tuple[str, str, bool]] = []
        if unpaired or paired:
            if capture is None:
                capture = self._run_capture_contract(runner=runner, dataset=dataset)

        if unpaired:
            subspace = self._run_subspace_contract(runner=runner, capture=capture)
            direction = runner.run(
                DirectionSpec(
                    feature=capture.feature("contract_residual"),
                    layers=[0],
                    positive=dataset.labels("class").equals("positive"),
                    negative=dataset.labels("class").equals("negative"),
                    tokens=TokenSelector.section("BODY"),
                    pooling=TokenPooling.mean(),
                    subspace=subspace,
                )
            )
            centroids = runner.run(
                CentroidSpec(
                    feature=capture.feature("contract_residual"),
                    by=dataset.labels("class"),
                    layers=[0],
                    tokens=TokenSelector.section("BODY"),
                    pooling=TokenPooling.mean(),
                    subspace=subspace,
                )
            )
            unpaired_patches = {
                "project_out": ProjectOutPatch(
                    subspace=subspace,
                    write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                    target_tokens=TokenSelector.section("BODY"),
                    component_indices_by_layer={0: (0,)},
                ),
                "random_control": RandomControlPatch(
                    subspace=subspace,
                    write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                    target_tokens=TokenSelector.section("BODY"),
                    component_indices_by_layer={0: (0,)},
                    random_seed=13,
                ),
                "add_direction": AddDirectionPatch(
                    direction=direction,
                    subspace=subspace,
                    write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                    target_tokens=TokenSelector.section("BODY"),
                    component_indices_by_layer={0: (0,)},
                ),
                "swap_mean": SwapMeanPatch(
                    centroids=centroids,
                    centroid_name="negative",
                    write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                    target_tokens=TokenSelector.section("BODY"),
                ),
                "swap_components": SwapComponentsPatch(
                    subspace=subspace,
                    centroids=centroids,
                    centroid_name="negative",
                    write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                    target_tokens=TokenSelector.section("BODY"),
                    component_indices_by_layer={0: (0,)},
                ),
            }
            for operator, patch in unpaired_patches.items():
                patch_specs.append(
                    PatchedGenerationSpec(
                        engine=self.engine_factory(),
                        dataset=dataset,
                        patch=patch,
                        select_when=dataset.labels("class").equals("positive"),
                        generation=GenerationSpec(enabled=True, max_tokens=2, temperature=0.0),
                    )
                )
                expectations.append(("unpaired", operator, False))

        if paired:
            activation_bank = runner.run(
                ActivationBankSpec(
                    feature=capture.feature("contract_residual"),
                    layers=[0],
                )
            )
            path_mask = runner.run(
                ExplicitPathMaskSpec(
                    edges=(ExplicitPathEdge(source_layer=0, write_layer=0, weight=1.0),),
                )
            )
            paired_patches = {
                "interchange": InterchangePatch(
                    activation_bank=activation_bank,
                    write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                    target_tokens=TokenSelector.section("BODY"),
                    donor_tokens=TokenSelector.section("BODY"),
                ),
                "residual_path": ResidualPathPatch(
                    activation_bank=activation_bank,
                    path_mask=path_mask,
                    write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                    target_tokens=TokenSelector.section("BODY"),
                    read_tokens=TokenSelector.section("BODY"),
                    transport="delta",
                ),
            }
            for operator, patch in paired_patches.items():
                patch_specs.append(
                    PatchedGenerationSpec(
                        engine=self.engine_factory(),
                        dataset=dataset,
                        patch=patch,
                        pair_by=dataset.cases("case_key"),
                        target_when=dataset.labels("class").equals("positive"),
                        donor_when=dataset.labels("class").equals("negative"),
                        generation=GenerationSpec(enabled=True, max_tokens=2, temperature=0.0),
                    )
                )
                expectations.append(("paired", operator, True))

        if patch_specs:
            artifacts = self._run_model_bound_group(runner=runner, specs=patch_specs)
            for (family, operator, expects_donor), patched in zip(expectations, artifacts, strict=True):
                row = self._assert_patch_artifact(
                    patched,
                    expected_operator=operator,
                    expected_example_key="contract_positive",
                    expected_target_tokens=[0, 1, 2],
                    expected_dispatch=self.expected_patch_dispatch,
                )
                covered.add(family)
                if expects_donor:
                    assert row["donor_example_key"] == "contract_negative"
                    donor_tokens = row.get("donor_tokens", row.get("read_tokens"))
                    assert donor_tokens == [0, 1, 2]

        assert requested <= covered
        return covered

    def run_unpaired_patch_operator_contracts(self, tmp_path: str | Path) -> None:
        runner = self._runner(tmp_path)
        dataset = make_engine_contract_dataset()
        capture = self._run_capture_contract(runner=runner, dataset=dataset)
        subspace = self._run_subspace_contract(runner=runner, capture=capture)
        direction = runner.run(
            DirectionSpec(
                feature=capture.feature("contract_residual"),
                layers=[0],
                positive=dataset.labels("class").equals("positive"),
                negative=dataset.labels("class").equals("negative"),
                tokens=TokenSelector.section("BODY"),
                pooling=TokenPooling.mean(),
                subspace=subspace,
            )
        )
        centroids = runner.run(
            CentroidSpec(
                feature=capture.feature("contract_residual"),
                by=dataset.labels("class"),
                layers=[0],
                tokens=TokenSelector.section("BODY"),
                pooling=TokenPooling.mean(),
                subspace=subspace,
            )
        )
        patch_specs = {
            "project_out": ProjectOutPatch(
                subspace=subspace,
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.section("BODY"),
                component_indices_by_layer={0: (0,)},
            ),
            "random_control": RandomControlPatch(
                subspace=subspace,
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.section("BODY"),
                component_indices_by_layer={0: (0,)},
                random_seed=13,
            ),
            "add_direction": AddDirectionPatch(
                direction=direction,
                subspace=subspace,
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.section("BODY"),
                component_indices_by_layer={0: (0,)},
            ),
            "swap_mean": SwapMeanPatch(
                centroids=centroids,
                centroid_name="negative",
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.section("BODY"),
            ),
            "swap_components": SwapComponentsPatch(
                subspace=subspace,
                centroids=centroids,
                centroid_name="negative",
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.section("BODY"),
                component_indices_by_layer={0: (0,)},
            ),
        }
        model_bound_specs = [
            PatchedGenerationSpec(
                engine=self.engine_factory(),
                dataset=dataset,
                patch=patch,
                select_when=dataset.labels("class").equals("positive"),
                generation=GenerationSpec(enabled=True, max_tokens=2, temperature=0.0),
            )
            for patch in patch_specs.values()
        ]
        artifacts = self._run_model_bound_group(runner=runner, specs=model_bound_specs)
        for operator, patched in zip(patch_specs, artifacts, strict=True):
            self._assert_patch_artifact(
                patched,
                expected_operator=operator,
                expected_example_key="contract_positive",
                expected_target_tokens=[0, 1, 2],
            )

    def run_paired_patch_operator_contracts(self, tmp_path: str | Path) -> None:
        runner = self._runner(tmp_path)
        dataset = make_engine_contract_dataset()
        capture = self._run_capture_contract(runner=runner, dataset=dataset)
        activation_bank = runner.run(
            ActivationBankSpec(
                feature=capture.feature("contract_residual"),
                layers=[0],
            )
        )
        path_mask = runner.run(
            ExplicitPathMaskSpec(
                edges=(ExplicitPathEdge(source_layer=0, write_layer=0, weight=1.0),),
            )
        )
        patch_specs = {
            "interchange": InterchangePatch(
                activation_bank=activation_bank,
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.section("BODY"),
                donor_tokens=TokenSelector.section("BODY"),
            ),
            "residual_path": ResidualPathPatch(
                activation_bank=activation_bank,
                path_mask=path_mask,
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.section("BODY"),
                read_tokens=TokenSelector.section("BODY"),
                transport="delta",
            ),
        }
        model_bound_specs = [
            PatchedGenerationSpec(
                engine=self.engine_factory(),
                dataset=dataset,
                patch=patch,
                pair_by=dataset.cases("case_key"),
                target_when=dataset.labels("class").equals("positive"),
                donor_when=dataset.labels("class").equals("negative"),
                generation=GenerationSpec(enabled=True, max_tokens=2, temperature=0.0),
            )
            for patch in patch_specs.values()
        ]
        artifacts = self._run_model_bound_group(runner=runner, specs=model_bound_specs)
        for operator, patched in zip(patch_specs, artifacts, strict=True):
            row = self._assert_patch_artifact(
                patched,
                expected_operator=operator,
                expected_example_key="contract_positive",
                expected_target_tokens=[0, 1, 2],
            )
            assert row["donor_example_key"] == "contract_negative"
            donor_tokens = row.get("donor_tokens", row.get("read_tokens"))
            assert donor_tokens == [0, 1, 2]

    def run_all_model_bound_contracts(self, tmp_path: str | Path) -> None:
        self.run_requested_model_bound_contracts(
            Path(tmp_path),
            basic=True,
            unpaired=True,
            paired=True,
        )

    def _run_capture_contract(self, *, runner: Runner, dataset: Dataset) -> Any:
        engine = self.engine_factory()
        capture = runner.run(
            make_engine_contract_capture_spec(
                engine=engine,
                dataset=dataset,
            )
        )
        self._assert_capture_artifact(capture)
        return capture

    def _assert_capture_artifact(self, capture: Any) -> None:
        assert_artifact_manifest_valid(capture.manifest())
        capture_payload = capture.feature("contract_residual").load()
        assert capture_payload["kind"] == "residual"
        assert sorted(capture_payload["layers"]) == ["0"]
        layer_payload = capture_payload["layers"]["0"]
        assert sorted(layer_payload) == ["contract_negative", "contract_positive"]
        for row in layer_payload.values():
            assert row["tokens"] == [0, 1, 2]
            assert row["token_sections"] == {"BODY": [0, 1, 2]}
            assert len(row["values"]) == 3
            assert len(row["values"][0]) > 0

    def _run_generation_contract(self, *, runner: Runner, dataset: Dataset) -> Any:
        engine = self.engine_factory()
        generation = runner.run(make_engine_contract_generation_spec(engine=engine, dataset=dataset))
        self._assert_generation_artifact(generation)
        return generation

    def _assert_generation_artifact(self, generation: Any) -> None:
        generation_payload = generation.result()
        assert generation_payload["kind"] == "generation_run_result"
        assert generation_payload["summary"]["example_count"] == 1
        assert [row["example_key"] for row in generation_payload["rows"]] == ["contract_positive"]
        assert generation_payload["rows"][0]["generated_text"] != ""

    def _run_subspace_contract(self, *, runner: Runner, capture: Any) -> Any:
        subspace = runner.run(
            SubspaceSpec(
                feature=capture.feature("contract_residual"),
                layers=[0],
                components=1,
                tokens=TokenSelector.section("BODY"),
                pooling=TokenPooling.mean(),
            )
        )
        subspace_payload = subspace.result()
        assert subspace_payload["kind"] == "subspace_result"
        assert subspace_payload["layers"]["0"]["component_count"] == 1
        return subspace

    def _run_project_out_contract(self, *, runner: Runner, dataset: Dataset, subspace: Any) -> Any:
        patched = runner.run(self._project_out_spec(engine=self.engine_factory(), dataset=dataset, subspace=subspace))
        self._assert_patch_artifact(
            patched,
            expected_operator="project_out",
            expected_example_key="contract_positive",
            expected_target_tokens=[0, 1, 2],
            expected_dispatch=self.expected_patch_dispatch,
        )
        return patched

    def _project_out_spec(self, *, engine: Engine, dataset: Dataset, subspace: Any) -> PatchedGenerationSpec:
        return PatchedGenerationSpec(
            engine=engine,
            dataset=dataset,
            patch=ProjectOutPatch(
                subspace=subspace,
                write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                target_tokens=TokenSelector.section("BODY"),
                component_indices_by_layer={0: (0,)},
                strength=1.0,
            ),
            select_when=dataset.labels("class").equals("positive"),
            generation=GenerationSpec(enabled=True, max_tokens=2, temperature=0.0),
        )

    def _run_model_bound_group(self, *, runner: Runner, specs: Iterable[OperationSpec]) -> list[Any]:
        specs_tuple = tuple(specs)
        run_many = getattr(runner, "run_many", None)
        if callable(run_many):
            return list(run_many(specs_tuple))
        return [runner.run(spec) for spec in specs_tuple]

    def _assert_patch_artifact(
        self,
        artifact: Any,
        *,
        expected_operator: str,
        expected_example_key: str,
        expected_target_tokens: list[int],
        expected_dispatch: str | None = None,
        ) -> dict[str, Any]:
        assert_artifact_manifest_valid(artifact.manifest())
        assert artifact.manifest().artifact_kind == "patched_generation"
        patch_payload = artifact.result()
        assert patch_payload["kind"] == "patched_generation_result"
        assert patch_payload["summary"]["patched_count"] == 1
        row = patch_payload["rows"][0]
        assert row["status"] == "ok"
        assert row["example_key"] == expected_example_key
        assert row["target_tokens"] == expected_target_tokens
        assert row["generated_text"] != ""
        stats = row["patch_stats"]["0"]
        assert stats["operator"] == expected_operator
        assert stats["status"] == "ok"
        assert int(stats["token_count"]) == len(expected_target_tokens)
        if expected_dispatch is not None:
            assert stats["dispatch"] == expected_dispatch
        return row


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
        assert catalog.load_artifact(artifact.id) is not None

    def run_capture_and_label_operation_smoke(self, tmp_path: str | Path) -> None:
        store = LocalArtifactStore(Path(tmp_path) / "artifacts")
        catalog = FileCatalog(Path(tmp_path) / "catalog")
        runner = self.runner_factory(artifacts=store, catalog=catalog)
        dataset = make_toy_dataset()
        capture = runner.run(make_toy_capture_spec(dataset))
        mapped = runner.run(
            LabelMapSpec(
                source=dataset.labels("class"),
                output_name="class_binary",
                mapping={"positive": 1, "negative": 0},
            )
        )
        assert_artifact_manifest_valid(mapped.manifest())
        assert mapped.label("class_binary").resolve_values() == {"ex_a": 1, "ex_b": 0}
        assert catalog.load_artifact(capture.id) is not None
        assert catalog.load_artifact(mapped.id) is not None


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

    def run_full_roundtrip(self, tmp_path: str | Path) -> None:
        store = self.store_factory(Path(tmp_path) / "artifacts")
        artifact_id = "contract_artifact"
        root = store.make_artifact_dir(artifact_id)
        assert root.exists()
        assert store.ensure_artifact_dir(artifact_id).exists()
        assert store.has_local_artifact(artifact_id)

        json_ref = store.write_json(
            artifact_id,
            "nested/payload.json",
            {"ok": True, "items": [3, 2, 1]},
        )
        assert json_ref["store"] == store.kind
        assert json_ref["format"] == "json"
        assert int(json_ref["bytes"]) > 0
        assert store.has_local_ref(json_ref)
        assert store.estimate_download_bytes(json_ref) == int(json_ref["bytes"])
        assert store.read_json_ref(json_ref) == {"ok": True, "items": [3, 2, 1]}

        tensor_ref = store.write_safetensors(
            artifact_id,
            "features/layer.safetensors",
            {
                "layer_0": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "layer_1": np.asarray([5.0, 6.0], dtype=np.float32),
            },
        )
        assert tensor_ref["store"] == store.kind
        assert tensor_ref["format"] == "safetensors"
        assert int(tensor_ref["bytes"]) > 0
        assert store.has_local_ref(tensor_ref)
        tensors = store.read_safetensors_ref(tensor_ref)
        np.testing.assert_allclose(tensors["layer_0"], np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        np.testing.assert_allclose(tensors["layer_1"], np.asarray([5.0, 6.0], dtype=np.float32))
        store.validate_transfer(bytes=store.estimate_download_bytes(tensor_ref), label="contract tensor")
        assert store.localize(artifact_id).exists()

        try:
            store.write_json(artifact_id, "../escape.json", {"bad": True})
        except ValueError:
            pass
        else:
            raise AssertionError("artifact stores must reject writes outside the artifact root")


@dataclass(frozen=True, slots=True)
class CatalogContractSuite:
    catalog_factory: Callable[[Path], Any]

    def run_record_smoke(self, tmp_path: str | Path) -> None:
        manifest = _contract_manifest("contract_artifact")
        catalog = self.catalog_factory(Path(tmp_path) / "catalog")
        catalog.record_artifact(manifest)
        assert catalog.load_artifact("contract_artifact") == manifest

    def run_workflow_index_roundtrip(self, tmp_path: str | Path) -> None:
        catalog = self.catalog_factory(Path(tmp_path) / "catalog")
        older = _contract_manifest(
            "contract_artifact_old",
            created_at="2026-01-01T00:00:00+00:00",
            step_name="capture",
            operation_semantic_hash="capture_semantic_hash",
        )
        newer = _contract_manifest(
            "contract_artifact_new",
            created_at="2026-01-01T00:00:01+00:00",
            step_name="capture",
            operation_semantic_hash="capture_semantic_hash",
        )
        downstream = _contract_manifest(
            "contract_artifact_downstream",
            created_at="2026-01-01T00:00:02+00:00",
            step_name="probe",
            step_index=1,
            operation_semantic_hash="probe_semantic_hash",
            input_artifact_refs=("contract_artifact_new",),
        )
        for manifest in (older, newer, downstream):
            catalog.record_artifact(manifest)
            assert catalog.load_artifact(manifest.artifact_id) == manifest

        latest = catalog.find_artifact_for_workflow_step(
            run_id="wr_contract",
            workflow_step_key="workflow_hash.capture",
        )
        assert latest is not None
        assert latest.artifact_id == "contract_artifact_new"

        first_run = WorkflowRunRecord(
            run_id="wr_contract",
            workflow_name="contract_workflow",
            workflow_hash="workflow_hash",
            workflow_spec_hash="workflow_spec_hash",
            workflow_payload={"kind": "workflow", "steps": [{"spec": {"kind": "capture"}}]},
            status="running",
            started_at="2026-01-01T00:00:00+00:00",
        )
        second_run = WorkflowRunRecord(
            run_id="wr_contract_second",
            workflow_name="contract_workflow",
            workflow_hash="workflow_hash",
            workflow_spec_hash="workflow_spec_hash",
            workflow_payload={"kind": "workflow", "steps": []},
            status="completed",
            started_at="2026-01-01T00:00:03+00:00",
            finished_at="2026-01-01T00:00:04+00:00",
        )
        catalog.record_workflow_run(first_run)
        catalog.record_workflow_run(second_run)

        assert catalog.load_workflow_run("wr_contract") == first_run
        completed = catalog.list_workflow_runs(workflow_name="contract_workflow", status="completed")
        assert [record.run_id for record in completed] == ["wr_contract_second"]
        limited = catalog.list_workflow_runs(workflow_hash="workflow_hash", limit=1)
        assert [record.run_id for record in limited] == ["wr_contract_second"]

        running_capture = WorkflowStepRecord(
            run_id="wr_contract",
            workflow_hash="workflow_hash",
            workflow_step_key="workflow_hash.capture",
            step_name="capture",
            step_index=0,
            runner="contract",
            status="running",
            step_semantic_hash="capture_semantic_hash",
            step_spec_hash="capture_spec_hash",
            started_at="2026-01-01T00:00:00+00:00",
        )
        completed_capture = WorkflowStepRecord(
            run_id="wr_contract",
            workflow_hash="workflow_hash",
            workflow_step_key="workflow_hash.capture",
            step_name="capture",
            step_index=0,
            runner="contract",
            status="completed",
            step_semantic_hash="capture_semantic_hash",
            step_spec_hash="capture_spec_hash",
            artifact_id="contract_artifact_new",
            artifact_kind="contract",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
        )
        completed_probe = WorkflowStepRecord(
            run_id="wr_contract",
            workflow_hash="workflow_hash",
            workflow_step_key="workflow_hash.probe",
            step_name="probe",
            step_index=1,
            runner="contract",
            status="completed",
            step_semantic_hash="probe_semantic_hash",
            step_spec_hash="probe_spec_hash",
            input_artifact_refs=("contract_artifact_new",),
            artifact_id="contract_artifact_downstream",
            artifact_kind="contract",
            started_at="2026-01-01T00:00:01+00:00",
            finished_at="2026-01-01T00:00:02+00:00",
        )
        catalog.record_workflow_step(running_capture)
        catalog.record_workflow_step(completed_capture)
        catalog.record_workflow_step(completed_probe)

        steps = {record.step_name: record for record in catalog.list_workflow_steps("wr_contract")}
        assert steps["capture"].status == "completed"
        assert steps["probe"].input_artifact_refs == ("contract_artifact_new",)
        reusable = catalog.find_latest_reusable_step(
            step_name="probe",
            step_semantic_hash="probe_semantic_hash",
            input_artifact_refs=("contract_artifact_new",),
        )
        assert reusable is not None
        assert reusable.artifact_id == "contract_artifact_downstream"

        if hasattr(catalog, "list_workflow_runs_light"):
            light_rows = catalog.list_workflow_runs_light(workflow_name="contract_workflow", limit=2)
            by_id = {row["run_id"]: row for row in light_rows}
            assert by_id["wr_contract"]["step_counts"] == {"completed": 2}


@dataclass(frozen=True, slots=True)
class WorkflowContractSuite:
    """Reusable workflow-orchestrator contracts over local deterministic specs."""

    def run_resume_and_reuse_contract(self, tmp_path: str | Path) -> None:
        root = Path(tmp_path)
        store = LocalArtifactStore(root / "artifacts")
        catalog = FileCatalog(root / "catalog")
        runner = LocalRunner(artifacts=store, catalog=catalog)
        workflow = _contract_workflow()
        orchestrator = WorkflowOrchestrator(
            runners={"local": runner},
            workflow_catalog=catalog,
            max_parallelism=1,
        )

        result = orchestrator.run(workflow)
        assert result.run_id is not None
        first_steps = {record.step_name: record for record in catalog.list_workflow_steps(result.run_id)}
        assert first_steps["map"].status == "completed"
        assert first_steps["remap"].input_artifact_refs == (first_steps["map"].artifact_id,)

        second = orchestrator.run(workflow, reuse_completed=True)
        assert second.run_id != result.run_id
        second_steps = {record.step_name: record for record in catalog.list_workflow_steps(second.run_id)}
        assert second_steps["map"].status == "reused"
        assert second_steps["map"].reused_from_artifact_id == first_steps["map"].artifact_id
        assert second_steps["remap"].status == "reused"
        assert second_steps["remap"].input_artifact_refs == (first_steps["map"].artifact_id,)

    def run_failure_blocks_downstream_contract(self, tmp_path: str | Path) -> None:
        root = Path(tmp_path)
        store = LocalArtifactStore(root / "artifacts")
        catalog = FileCatalog(root / "catalog")
        runner = _FailingRunner(
            LocalRunner(artifacts=store, catalog=catalog),
            fail_step="remap",
        )
        workflow = _contract_workflow()
        orchestrator = WorkflowOrchestrator(
            runners={"local": runner},
            workflow_catalog=catalog,
            max_parallelism=1,
        )

        try:
            orchestrator.run(workflow)
        except RuntimeError as exc:
            assert "contract failure at remap" in str(exc)
        else:
            raise AssertionError("workflow contract expected the injected runner failure")

        failed_runs = catalog.list_workflow_runs(workflow_name="contract_workflow", status="failed")
        assert len(failed_runs) == 1
        run_id = failed_runs[0].run_id
        steps = {record.step_name: record for record in catalog.list_workflow_steps(run_id)}
        assert steps["map"].status == "completed"
        assert steps["remap"].status == "failed"
        assert steps["final"].status == "blocked"
        assert steps["remap"].input_artifact_refs == (steps["map"].artifact_id,)

        recovered = WorkflowOrchestrator(
            runners={"local": LocalRunner(artifacts=store, catalog=catalog)},
            workflow_catalog=catalog,
            max_parallelism=1,
        ).run(workflow, resume_run_id=run_id)
        recovered_steps = {record.step_name: record for record in catalog.list_workflow_steps(run_id)}
        assert recovered.run_id == run_id
        assert recovered_steps["map"].status == "completed"
        assert recovered_steps["remap"].status == "completed"
        assert recovered_steps["final"].status == "completed"

    def run_ready_step_batching_contract(self, tmp_path: str | Path) -> None:
        root = Path(tmp_path)
        store = LocalArtifactStore(root / "artifacts")
        catalog = FileCatalog(root / "catalog")
        runner = _BatchingRunner(LocalRunner(artifacts=store, catalog=catalog))
        workflow = _batch_contract_workflow()
        orchestrator = WorkflowOrchestrator(
            runners={"local": runner},
            workflow_catalog=catalog,
            max_parallelism=2,
        )

        result = orchestrator.run(workflow)
        assert result.run_id is not None
        assert any(set(call) == {"map_negative", "map_positive"} for call in runner.batch_calls)
        steps = {record.step_name: record for record in catalog.list_workflow_steps(result.run_id)}
        assert steps["map_positive"].status == "completed"
        assert steps["map_negative"].status == "completed"
        assert steps["final"].status == "completed"
        assert steps["final"].input_artifact_refs == (steps["map_positive"].artifact_id,)


def _contract_workflow() -> WorkflowSpec:
    dataset = make_toy_dataset()
    return WorkflowSpec(
        name="contract_workflow",
        steps=(
            WorkflowStep(
                name="map",
                runner="local",
                spec=LabelMapSpec(
                    source=dataset.labels("class"),
                    output_name="class_binary",
                    mapping={"positive": 1, "negative": 0},
                ),
            ),
            WorkflowStep(
                name="remap",
                runner="local",
                spec=LabelMapSpec(
                    source=StepRef("map").label("class_binary"),
                    output_name="class_text",
                    mapping={"1": "pos", "0": "neg"},
                ),
            ),
            WorkflowStep(
                name="final",
                runner="local",
                spec=LabelMapSpec(
                    source=StepRef("remap").label("class_text"),
                    output_name="class_final",
                    mapping={"pos": "positive", "neg": "negative"},
                ),
            ),
        ),
    )


def _batch_contract_workflow() -> WorkflowSpec:
    dataset = make_toy_dataset()
    return WorkflowSpec(
        name="batch_contract_workflow",
        steps=(
            WorkflowStep(
                name="map_positive",
                runner="local",
                spec=LabelMapSpec(
                    source=dataset.labels("class"),
                    output_name="positive_flag",
                    mapping={"positive": 1, "negative": 0},
                ),
            ),
            WorkflowStep(
                name="map_negative",
                runner="local",
                spec=LabelMapSpec(
                    source=dataset.labels("class"),
                    output_name="negative_flag",
                    mapping={"positive": 0, "negative": 1},
                ),
            ),
            WorkflowStep(
                name="final",
                runner="local",
                spec=LabelMapSpec(
                    source=StepRef("map_positive").label("positive_flag"),
                    output_name="class_final",
                    mapping={"1": "positive", "0": "negative"},
                ),
            ),
        ),
    )


@dataclass(slots=True)
class _FailingRunner:
    wrapped: LocalRunner
    fail_step: str

    @property
    def artifacts(self) -> Any:
        return self.wrapped.artifacts

    @property
    def catalog(self) -> Any:
        return self.wrapped.catalog

    def identity(self) -> dict[str, Any]:
        return self.wrapped.identity()

    def plan(self, spec: OperationSpec) -> Any:
        return self.wrapped.plan(spec)

    def run(
        self,
        spec: OperationSpec,
        *,
        workflow_context: WorkflowStepContext | None = None,
        progress_callback: Any | None = None,
    ) -> Any:
        if workflow_context is not None and workflow_context.step_name == self.fail_step:
            raise RuntimeError(f"contract failure at {self.fail_step}")
        return self.wrapped.run(
            spec,
            workflow_context=workflow_context,
            progress_callback=progress_callback,
        )


@dataclass(slots=True)
class _BatchingRunner:
    wrapped: LocalRunner
    batch_calls: list[tuple[str, ...]] = field(default_factory=list)

    @property
    def artifacts(self) -> Any:
        return self.wrapped.artifacts

    @property
    def catalog(self) -> Any:
        return self.wrapped.catalog

    def identity(self) -> dict[str, Any]:
        return self.wrapped.identity()

    def plan(self, spec: OperationSpec) -> Any:
        return self.wrapped.plan(spec)

    def workflow_batch_key(self, spec: OperationSpec) -> str | None:
        return "label-map" if isinstance(spec, LabelMapSpec) else None

    def run(
        self,
        spec: OperationSpec,
        *,
        workflow_context: WorkflowStepContext | None = None,
        progress_callback: Any | None = None,
    ) -> Any:
        return self.wrapped.run(
            spec,
            workflow_context=workflow_context,
            progress_callback=progress_callback,
        )

    def run_many(
        self,
        specs: list[OperationSpec],
        *,
        workflow_contexts: list[WorkflowStepContext | None] | None = None,
        progress_callback: Any | None = None,
    ) -> list[Any]:
        del progress_callback
        contexts = workflow_contexts or [None] * len(specs)
        self.batch_calls.append(tuple(context.step_name for context in contexts if context is not None))
        return [
            self.wrapped.run(spec, workflow_context=context)
            for spec, context in zip(specs, contexts, strict=True)
        ]
