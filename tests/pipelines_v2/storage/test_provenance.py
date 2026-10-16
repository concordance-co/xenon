"""C01 scientific identity and publication boundaries, using deterministic fixtures."""
from dataclasses import replace
from pathlib import Path
import copy
import json

import pytest

from pipelines_v2.api import (CaptureSpec, Dataset, Example, GenerationRunSpec, GenerationSpec,
    LabelMapSpec, LocalArtifactStore, LocalRunner, PatchedGenerationSpec, ProjectOutPatch,
    ResidualInterventionSite, ResidualSite, TokenSelector, ToyEngine, VLLMEngine)
from pipelines_v2.core.provenance import (KEY, ProvenanceV1, bind_materialized_dataset,
    compare_provenance, content_hash, execution_context, resolve_provenance, source_snapshot)
from pipelines_v2.runtime.remote_executor import (_artifact_id_for, _bind_execution_dataset,
    _apply_execution_shard, execute_remote, merge_remote_shards)
from pipelines_v2.storage.artifacts import ArtifactManifest, InlineOperationArtifact, OperationArtifact


@pytest.fixture
def spec():
    return GenerationRunSpec(engine=ToyEngine(), dataset=Dataset.from_examples([
        Example(key="a", prompt="alpha", cases={"split": "train"}),
        Example(key="b", prompt="beta", cases={"split": "test"}),
    ]), generation=GenerationSpec(enabled=True, max_tokens=1))


@pytest.fixture
def provenance(spec):
    return resolve_provenance(spec, source={"repository_id": "fixture", "head": "a" * 40,
        "dirty_digest": "b" * 64, "mounted_source_digest": "c" * 64},
        runtime={"family": "local", "lock_digest": "d" * 64, "python": "3.13", "platform": "test",
            "installed_package_digest": "e" * 64, "image_identity": None, "image_reason": "local process"},
        model_components=[{"role": "model", "repository_id": "fixture/model", "requested_revision": "main",
            "resolved_revision": "a" * 40, "file_set_digest": "f" * 64}],
        numerics={"dtype": "float32", "attention_implementation": "eager", "device": "cpu",
            "determinism": True, "compile": False, "matmul": {"tf32": False}},
        methods={"operation_version": "fixture-v1", "hook_version": "hook-v1", "evaluator": {"version": "eval-v1"}},
        seeds={"generation": 12})


def rebuild(record, **changes):
    body = record.to_dict()
    body.pop("scientific_identity")
    body.pop("schema_version")
    body.pop("worker_compatibility")
    body.update(changes)
    return ProvenanceV1(**body)


@pytest.mark.parametrize(("section", "field", "value"), [
    ("source", "mounted_source_digest", "0" * 64), ("runtime", "lock_digest", "1" * 64),
    ("numerics", "dtype", "bfloat16"), ("numerics", "attention_implementation", "flash"),
    ("methods", "hook_version", "hook-v2"), ("methods", "evaluator", {"version": "eval-v2"}),
    ("data", "content_digest", "2" * 64), ("data", "split_digest", "3" * 64),
    ("seeds", "generation", 19),
])
def test_changed_scientific_fact_rejects_reuse(provenance, section, field, value):
    changed = {**provenance.to_dict()[section], field: value}
    other = rebuild(provenance, **{section: changed})
    decision = compare_provenance(other, provenance)
    assert not decision.reusable
    assert f"{section} changed" in decision.reasons


def test_changed_model_commit_rejects_reuse(provenance):
    components = provenance.to_dict()["model_components"]
    components[0]["resolved_revision"] = "b" * 40
    assert not compare_provenance(rebuild(provenance, model_components=components), provenance).reusable


def test_json_roundtrip_order_and_deep_immutability(provenance):
    original = provenance.to_dict()
    reordered = json.loads(json.dumps(original, sort_keys=True))
    assert ProvenanceV1.from_dict(reordered) == provenance
    detached = provenance.to_dict()
    detached["numerics"]["matmul"]["tf32"] = True
    assert provenance.to_dict() == original
    reordered["scientific_identity"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        ProvenanceV1.from_dict(reordered)


def test_unknown_schema_rejected(provenance):
    payload = provenance.to_dict()
    payload["schema_version"] = 2
    with pytest.raises(ValueError, match="Unsupported provenance schema_version"):
        ProvenanceV1.from_dict(payload)


def test_requested_alias_and_git_metadata_are_diagnostic(provenance):
    components = provenance.to_dict()["model_components"]
    components[0]["requested_revision"] = "release"
    source = {**provenance.to_dict()["source"], "head": "9" * 40, "dirty_digest": "8" * 64}
    other = rebuild(provenance, model_components=components, source=source)
    assert compare_provenance(other, provenance).reusable
    assert other.to_dict()["worker_compatibility"]["identity"] != other.scientific_identity


@pytest.mark.parametrize("revision", [None, "main", "latest", ""])
def test_mutable_or_missing_model_identity_cannot_reuse(provenance, revision):
    components = provenance.to_dict()["model_components"]
    components[0]["resolved_revision"] = revision
    unknown = rebuild(provenance, model_components=components)
    assert not unknown.identity_complete
    assert not compare_provenance(unknown, unknown).reusable


def test_missing_numeric_fact_and_remote_image_stay_unresolved(provenance):
    assert not rebuild(provenance, numerics={**provenance.to_dict()["numerics"], "dtype": None}).identity_complete
    assert not rebuild(provenance, runtime={**provenance.to_dict()["runtime"], "family": "modal", "image_reason": "unresolved"}).identity_complete


def test_source_bytes_not_location_docs_or_secrets(tmp_path):
    roots = [tmp_path / "one", tmp_path / "two"]
    for root in roots:
        (root / "package").mkdir(parents=True)
        (root / "package" / "code.py").write_text("x = 1\n")
        (root / "package" / "config.json").write_text('{"size": 4}')
        (root / "package" / "README.md").write_text(str(root))
    one = source_snapshot(roots[0], ("package",))
    two = source_snapshot(roots[1], ("package",))
    assert one["mounted_source_digest"] == two["mounted_source_digest"]
    (roots[1] / "package" / "config.json").write_text('{"size": 8}')
    assert source_snapshot(roots[1], ("package",))["mounted_source_digest"] != one["mounted_source_digest"]
    (roots[0] / "package" / ".env").write_text("TOKEN=do-not-record")
    secret = source_snapshot(roots[0], ("package",))
    assert "do-not-record" not in json.dumps(secret)
    assert secret["unresolved"]
    (roots[0] / "linked").symlink_to(roots[1] / "package", target_is_directory=True)
    linked = source_snapshot(roots[0], ("linked",))
    assert linked["mounted_source_digest"] is None
    assert linked["unresolved"]


def test_display_names_do_not_change_identity(spec):
    changed = replace(spec, dataset=replace(spec.dataset, name="new display", id="new label"))
    assert resolve_provenance(spec).scientific_identity == resolve_provenance(changed).scientific_identity


def test_actual_prompt_bytes_override_caller_prompt_hash(spec):
    dataset = replace(spec.dataset, examples=(replace(spec.dataset.examples[0], prompt="changed actual input"), *spec.dataset.examples[1:]))
    assert resolve_provenance(spec).scientific_identity != resolve_provenance(replace(spec, dataset=dataset)).scientific_identity


def test_credentials_do_not_enter_identity_but_numerical_env_does(spec, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "first-private-token")
    first = resolve_provenance(spec)
    monkeypatch.setenv("HF_TOKEN", "second-private-token")
    second = resolve_provenance(spec)
    assert first == second
    assert "private-token" not in json.dumps(second.to_dict())
    monkeypatch.setenv("OMP_NUM_THREADS", "987")
    assert resolve_provenance(spec).scientific_identity != first.scientific_identity


def operation_specs(spec):
    subspace = InlineOperationArtifact(payload={"kind": "subspace_result", "layers": {"0": {
        "mean": [0.] * 4, "scale": [1.] * 4, "safe_scale": [1.] * 4,
        "components": [[1., 0., 0., 0.]], "component_count": 1, "named_components": {}}}}, artifact_kind="subspace")
    return [CaptureSpec(engine=spec.engine, dataset=spec.dataset,
                sites=[ResidualSite(name="last", site="resid_post", layers=[0], tokens=TokenSelector.last())]),
            spec, PatchedGenerationSpec(engine=spec.engine, dataset=spec.dataset, generation=spec.generation,
                patch=ProjectOutPatch(subspace=subspace, write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                    target_tokens=TokenSelector.last(), component_indices_by_layer={0: (0,)})),
            LabelMapSpec(source=Dataset.from_examples([Example(key="x", prompt="x", labels={"label": "yes"})]).labels("label"),
                         mapping={"yes": 1}, output_name="mapped")]


@pytest.mark.parametrize("index", range(4))
def test_local_and_remote_producers_publish_valid_c01(tmp_path, spec, index):
    operation = operation_specs(spec)[index]
    store = LocalArtifactStore(tmp_path / "store")
    local = LocalRunner(artifacts=store).run(operation).manifest()
    remote = ArtifactManifest.from_dict(execute_remote(runner_config={"kind": "test"},
        store_config=store.identity(), spec_payload=operation.to_dict()))
    assert local.provenance() is not None
    assert remote.provenance() is not None
    assert local.provenance_status in {"resolved", "unresolved"}


@pytest.mark.parametrize("index", range(3))
def test_shard_specific_identity_and_all_merge_producers(tmp_path, spec, index):
    operation = operation_specs(spec)[index]
    store = LocalArtifactStore(tmp_path)
    context = {"run_id": "wr_shards", "workflow_step_key": "operation"}
    shards = [execute_remote(runner_config={"kind": "test"}, store_config=store.identity(), spec_payload=operation.to_dict(),
        workflow_context={**context, "execution_shard": {"index": i, "count": 3}}) for i in range(3)]
    child_ids = {item["metadata"][KEY]["scientific_identity"] for item in shards}
    assert len(child_ids) == 3  # empty partition still has explicit partition identity
    merged = merge_remote_shards(runner_config={"kind": "test"}, store_config=store.identity(),
        spec_payload=operation.to_dict(), shard_manifests=shards, workflow_context=context)
    assert ArtifactManifest.from_dict(merged).provenance() is not None
    damaged = copy.deepcopy(shards)
    old = ProvenanceV1.from_dict(damaged[0]["metadata"]["parent_provenance_v1"])
    changed = rebuild(old, source={**old.to_dict()["source"], "mounted_source_digest": "0" * 64})
    damaged[0]["metadata"]["parent_provenance_v1"] = changed.to_dict()
    with pytest.raises(ValueError, match="parent scientific provenance"):
        merge_remote_shards(runner_config={"kind": "test"}, store_config=store.identity(), spec_payload=operation.to_dict(),
            shard_manifests=damaged, workflow_context=context)


def test_unresolved_execution_remains_executable_without_reusing(tmp_path, spec):
    store = LocalArtifactStore(tmp_path)
    args = dict(runner_config={"kind": "modal"}, store_config=store.identity(), spec_payload=spec.to_dict(),
                workflow_context={"run_id": "wr_unknown", "workflow_step_key": "generate"})
    first, second = execute_remote(**args), execute_remote(**args)
    assert first["artifact_id"] != second["artifact_id"]
    assert ArtifactManifest.from_dict(first).provenance_status == "unresolved"


def test_legacy_and_stale_partial_rows_never_reused(tmp_path, spec):
    store = LocalArtifactStore(tmp_path)
    context = execution_context(spec, {"kind": "test"}, {"run_id": "wr_partial", "workflow_step_key": "generate"})
    _bind_execution_dataset(context, spec.dataset)
    artifact_id = _artifact_id_for(spec=spec, workflow_context=context)
    store.ensure_artifact_dir(artifact_id)
    for metadata in ({}, {KEY: rebuild(ProvenanceV1.from_dict(context[KEY]), seeds={"wrong": 22}).to_dict()}):
        store.write_json(artifact_id, "result.json", {"metadata": metadata, "rows": [{"example_key": "a", "generated_text": "stale"}]})
        manifest_path = tmp_path / artifact_id / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
        result = execute_remote(runner_config={"kind": "test"}, store_config=store.identity(), spec_payload=spec.to_dict(), workflow_context=context)
        payload = store.read_json_ref(result["storage_refs"]["result"])
        assert all(row["generated_text"] != "stale" for row in payload["rows"])


def test_relocated_artifact_and_timestamp_do_not_change_downstream_identity(tmp_path, spec):
    store = LocalArtifactStore(tmp_path / "one")
    operation = operation_specs(spec)[3]
    first = LocalRunner(artifacts=store).run(operation)
    # Relocate a coherent C03 descriptor, not just storage_refs. Real publication
    # rebuilds completion paths/hashes while preserving the actual payload bytes.
    old_root = str(Path(store.root) / first.id)
    new_root = str(tmp_path / "two" / "different_display_id")
    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        return value.replace(old_root, new_root) if isinstance(value, str) else value
    metadata = relocate(first.manifest().metadata)
    completion = metadata['completion_v1']
    completion['integrity_hash'] = content_hash({key:value for key,value in completion.items() if key != 'integrity_hash'})
    moved_manifest = replace(first.manifest(), artifact_id="different_display_id", created_at="2099-01-01",
                             runner={"resources": {"display": "changed"}}, metadata=metadata,
                             storage_refs=relocate(first.manifest().storage_refs))
    moved = OperationArtifact(_manifest=moved_manifest, store=LocalArtifactStore(tmp_path / "two"))
    before = LabelMapSpec(source=first.label("mapped"), mapping={"1": "yes"}, output_name="result")
    after = replace(before, source=moved.label("mapped"))
    assert resolve_provenance(before).scientific_identity == resolve_provenance(after).scientific_identity
    legacy = replace(first.manifest(), metadata={})
    assert legacy.provenance_status == "legacy"
    assert not compare_provenance(first.manifest().provenance(), legacy.provenance()).reusable


def test_incidental_torch_import_does_not_change_model_identity(spec, monkeypatch):
    import sys
    # A warm worker may import torch after the first spec. C01 must not infer
    # effective policy from that incidental import state.
    first = resolve_provenance(spec)
    monkeypatch.setitem(sys.modules, "torch", object())
    assert resolve_provenance(spec) == first


def test_merger_preserves_producer_runtime_instead_of_its_own(tmp_path, spec, monkeypatch):
    import pipelines_v2.core.provenance as module
    store = LocalArtifactStore(tmp_path)
    context = {"run_id": "wr_cold_merge", "workflow_step_key": "generate"}
    shards = [execute_remote(runner_config={"kind": "test"}, store_config=store.identity(), spec_payload=spec.to_dict(),
        workflow_context={**context, "execution_shard": {"index": i, "count": 2}}) for i in range(2)]
    producer_runtime = shards[0]["metadata"][KEY]["runtime"]
    original = module.observed_runtime
    monkeypatch.setattr(module, "observed_runtime", lambda **kwargs: {**original(**kwargs), "installed_package_digest": "0" * 64})
    merged = merge_remote_shards(runner_config={"kind": "test"}, store_config=store.identity(), spec_payload=spec.to_dict(),
        shard_manifests=shards, workflow_context=context)
    assert merged["metadata"][KEY]["runtime"] == producer_runtime


def test_credential_fields_are_rejected_without_echo(provenance):
    with pytest.raises(ValueError, match="Secret-bearing") as error:
        rebuild(provenance, methods={**provenance.to_dict()["methods"], "api_key": "never-print-this"})
    assert "never-print-this" not in str(error.value)


def test_unobserved_file_inputs_and_numerical_analysis_are_not_complete(spec):
    from pipelines_v2.api import CoordinateImportSpec, AssistantAxisPrecomputedCoordinateSpec
    for operation in (CoordinateImportSpec(path="mutable-file.json"),
                      AssistantAxisPrecomputedCoordinateSpec(model_id="google/gemma-2-27b-it")):
        assert not resolve_provenance(operation).identity_complete


@pytest.mark.parametrize("remote", [False, True])
def test_deferred_inputs_are_identified_at_execution_without_changing_spec_hash(tmp_path, remote):
    from pipelines_v2.api import InMemorySource
    dataset = Dataset.from_source(source=InMemorySource.from_records([{"example_id": "a", "prompt": "actual input"}]),
        defer=True, prompt_column="prompt", example_key_column="example_id")
    operation = CaptureSpec(engine=ToyEngine(), dataset=dataset,
        sites=[ResidualSite(name="last", site="resid_post", layers=[0])])
    before = resolve_provenance(operation)
    assert not before.identity_complete
    store = LocalArtifactStore(tmp_path)
    if remote:
        manifest = ArtifactManifest.from_dict(execute_remote(runner_config={"kind": "test"}, store_config=store.identity(), spec_payload=operation.to_dict()))
    else:
        manifest = LocalRunner(artifacts=store).run(operation).manifest()
    assert manifest.provenance().identity_complete
    assert manifest.provenance().to_dict()["data"]["content_digest"] == bind_materialized_dataset(before, operation.resolve_dataset().dataset).to_dict()["data"]["content_digest"]
    assert manifest.operation_spec_hash == operation.spec_hash()


def test_mounted_markdown_prompt_is_computational_content(tmp_path):
    (tmp_path / "package").mkdir()
    prompt = tmp_path / "package" / "system_prompt.md"
    prompt.write_text("first instruction")
    first = source_snapshot(tmp_path, ("package",))
    prompt.write_text("different instruction")
    second = source_snapshot(tmp_path, ("package",))
    assert first["mounted_source_digest"] != second["mounted_source_digest"]


def test_model_access_tokens_are_not_scientific_inputs(spec):
    first_spec = replace(spec, engine=VLLMEngine(model_id="fixture/model", extra={"hf_token": "dummy-one", "attention_backend": "FLASH_ATTN"}))
    second_spec = replace(first_spec, engine=replace(first_spec.engine, extra={"hf_token": "dummy-two", "attention_backend": "FLASH_ATTN"}))
    first, second = resolve_provenance(first_spec), resolve_provenance(second_spec)
    assert first.scientific_identity == second.scientific_identity
    assert "dummy-one" not in json.dumps(first.to_dict())
    assert "dummy-two" not in json.dumps(second.to_dict())
    changed_algorithm = replace(second_spec, engine=replace(second_spec.engine, extra={"hf_token": "dummy-two", "attention_backend": "FLASHINFER"}))
    assert resolve_provenance(changed_algorithm).scientific_identity != first.scientific_identity


@pytest.mark.parametrize("model_id", ["https://user:dummy@example.com/model", "https://example.com/model?hf_token=dummy"])
def test_credential_bearing_model_identifiers_are_rejected_without_echo(spec, model_id):
    with pytest.raises(ValueError, match="Credential-bearing") as error:
        resolve_provenance(replace(spec, engine=VLLMEngine(model_id=model_id)))
    assert "dummy" not in str(error.value)


def test_real_numerical_environment_change_between_shards_is_rejected(tmp_path, spec, monkeypatch):
    store = LocalArtifactStore(tmp_path)
    args = dict(runner_config={"kind": "test"}, store_config=store.identity(), spec_payload=spec.to_dict())
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    first = execute_remote(**args, workflow_context={"execution_shard": {"index": 0, "count": 2}})
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    second = execute_remote(**args, workflow_context={"execution_shard": {"index": 1, "count": 2}})
    assert first["metadata"][KEY]["numerics"]["environment"] != second["metadata"][KEY]["numerics"]["environment"]
    with pytest.raises(ValueError, match="mismatched parent scientific provenance"):
        merge_remote_shards(**args, shard_manifests=[first, second])


def test_merger_computation_source_change_is_rejected(tmp_path, spec, monkeypatch):
    import pipelines_v2.core.provenance as module
    store = LocalArtifactStore(tmp_path)
    args = dict(runner_config={"kind": "test"}, store_config=store.identity(), spec_payload=spec.to_dict())
    shards = [execute_remote(**args, workflow_context={"execution_shard": {"index": index, "count": 2}})
              for index in range(2)]
    original = module.source_snapshot
    monkeypatch.setattr(module, "source_snapshot", lambda *a, **kw: {**original(*a, **kw), "mounted_source_digest": "0" * 64})
    with pytest.raises(ValueError, match="submitted source/spec/input contract"):
        merge_remote_shards(**args, shard_manifests=shards)
