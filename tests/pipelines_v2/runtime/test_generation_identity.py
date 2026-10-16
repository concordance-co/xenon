import pytest

from pipelines_v2.api import Dataset, Example, GenerationRunSpec, GenerationSpec, ToyEngine
from pipelines_v2.core.provenance import KEY, execution_context
from pipelines_v2.runtime.remote_executor import _bind_execution_dataset
from pipelines_v2.engine.base import EngineGenerationResult
from pipelines_v2.runtime.remote_executor import _artifact_id_for, execute_remote, merge_remote_shards
from pipelines_v2.storage import LocalArtifactStore


def spec_and_context():
    dataset = Dataset.from_examples([
        Example(key="repeat_1", prompt="same prompt", labels={"keep": True}),
        Example(key="repeat_2", prompt="same prompt", labels={"keep": True}),
        Example(key="excluded", prompt="other prompt", labels={"keep": False}),
    ])
    return GenerationRunSpec(
        engine=ToyEngine(), dataset=dataset,
        select_when=dataset.labels("keep").equals(True),
        generation=GenerationSpec(enabled=True, max_tokens=1),
    ), {"run_id": "wr_identity", "workflow_step_key": "generate"}


def test_generation_resume_preserves_distinct_examples_with_identical_prompts(tmp_path):
    spec, context = spec_and_context()
    store = LocalArtifactStore(tmp_path)
    context = execution_context(spec, {"kind": "test"}, context)
    selected = Dataset.from_examples([row for row in spec.dataset.examples if row.labels["keep"]])
    _bind_execution_dataset(context, selected)
    artifact_id = _artifact_id_for(spec=spec, workflow_context=context)
    store.ensure_artifact_dir(artifact_id)
    from pipelines_v2.runtime.remote_executor import _write_generation_result_payload
    _write_generation_result_payload(store=store, artifact_id=artifact_id, total_example_count=2, partial=True, semantic_hash=spec.semantic_hash(), expected_keys=[str(row.key) for row in selected.examples], metadata={KEY: context[KEY]}, rows=[{
        "example_key": "repeat_1", "example": spec.dataset.examples[0].to_dict(),
        "generated_text": "checkpoint output",
    }])
    manifest = execute_remote(runner_config={"kind": "test"}, store_config=store.identity(),
                              spec_payload=spec.to_dict(), workflow_context=context)
    payload = store.read_json_ref(manifest["storage_refs"]["result"])
    assert [row["example_key"] for row in payload["rows"]] == ["repeat_1", "repeat_2"]
    assert payload["rows"][0]["generated_text"] == "checkpoint output"
    assert payload["rows"][1]["generated_text"] == "toy_generation:repeat_2"
    assert payload["summary"]["completed_example_count"] == 2
    assert payload["summary"]["total_example_count"] == 2


def test_sharded_generation_preserves_distinct_examples_with_identical_prompts(tmp_path):
    spec, context = spec_and_context()
    store = LocalArtifactStore(tmp_path)
    shards = [execute_remote(
        runner_config={"kind": "test"}, store_config=store.identity(), spec_payload=spec.to_dict(),
        workflow_context={**context, "execution_shard": {"index": index, "count": 2}},
    ) for index in range(2)]
    manifest = merge_remote_shards(
        runner_config={"kind": "test"}, store_config=store.identity(), spec_payload=spec.to_dict(),
        shard_manifests=shards, workflow_context=context,
    )
    payload = store.read_json_ref(manifest["storage_refs"]["result"])
    assert [row["example_key"] for row in payload["rows"]] == ["repeat_1", "repeat_2"]
    assert manifest["example_coverage"]["example_count"] == 2


def test_incomplete_generation_does_not_publish_complete_manifest(tmp_path, monkeypatch):
    spec, context = spec_and_context()
    store = LocalArtifactStore(tmp_path)
    monkeypatch.setattr(ToyEngine, "generate", lambda *_args: EngineGenerationResult(rows=[], metadata={}))
    with pytest.raises(ValueError, match="coverage mismatch"):
        execute_remote(runner_config={"kind": "test"}, store_config=store.identity(),
                       spec_payload=spec.to_dict(), workflow_context=context)
    artifact_id = _artifact_id_for(spec=spec, workflow_context=context)
    assert not (tmp_path / artifact_id / "manifest.json").exists()
