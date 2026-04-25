from __future__ import annotations

from pathlib import Path

import pytest

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    Example,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunner,
    ResidualSite,
    TokenSelector,
    ToyEngine,
)
from pipelines_v2.mechinterp.assistant_axis import (
    ASSISTANT_AXIS_PROMPT_DATASET_REPO,
    AssistantAxisPrecomputedCoordinateSpec,
    AssistantAxisScoreSpec,
    AssistantAxisVectorSpec,
    assistant_axis_model_config,
    assistant_axis_prompt_dataset,
)
from pipelines_v2.operations import operation_spec_from_dict


def test_assistant_axis_dataset_helper_points_at_canonical_hf_dataset() -> None:
    dataset = assistant_axis_prompt_dataset(limit=3, revision="abc123")

    assert dataset.is_deferred
    assert dataset.name == "assistant_axis_prompt_sources"
    assert dataset.source["path"] == ASSISTANT_AXIS_PROMPT_DATASET_REPO
    assert dataset.source["revision"] == "abc123"
    assert dataset.fetch["prompt_column"] == "name"
    assert dataset.fetch["prompt_template"] == "{source_type}:{name}"
    assert dataset.fetch["limit"] == 3
    assert "instructions" in dataset.fetch["metadata_columns"]
    assert "questions" in dataset.fetch["metadata_columns"]


def test_precomputed_assistant_axis_helper_resolves_known_hf_vector_and_roundtrips() -> None:
    spec = AssistantAxisPrecomputedCoordinateSpec(
        model_id="Qwen/Qwen3-32B",
        revision="abc123",
        token_env_var="HF_TOKEN",
    )
    loaded = operation_spec_from_dict(spec.to_dict())
    runtime_spec = spec.runtime_spec()

    assert isinstance(loaded, AssistantAxisPrecomputedCoordinateSpec)
    assert loaded.revision == "abc123"
    assert spec.resolved_layer() == 32
    assert spec.resolved_filename() == "qwen-3-32b/assistant_axis.pt"
    assert assistant_axis_model_config("gemma-2-27b")["target_layer"] == 22
    assert assistant_axis_model_config("meta-llama/Llama-3.3-70B-Instruct")["target_layer"] == 40
    assert runtime_spec is not None
    assert [secret.env_var for secret in runtime_spec.secrets] == ["HF_TOKEN"]


def test_assistant_axis_vector_generation_and_scoring(tmp_path: Path) -> None:
    dataset = _assistant_axis_training_dataset()
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=4, num_layers=1, sequence_length=5),
            dataset=dataset,
            generation=GenerationSpec(enabled=True, max_tokens=2, capture_generated_tokens=True),
            sites=[
                ResidualSite(
                    name="resid_post_full",
                    site="resid_post",
                    layers=[0],
                    tokens=TokenSelector.full_sequence(),
                )
            ],
        )
    )

    axis = runner.run(
        AssistantAxisVectorSpec(
            feature=capture.feature("resid_post_full"),
            role_by=dataset.labels("role"),
            default_when=dataset.labels("axis_kind").equals("default"),
            role_when=dataset.labels("axis_kind").equals("role"),
            score_by=dataset.labels("adherence_score"),
            score_values=(3,),
            min_role_examples_per_role=1,
            min_default_examples=1,
            layers=[0],
            model_id="google/gemma-2-27b-it",
        )
    )
    axis_payload = axis.result()

    assert axis_payload["kind"] == "coordinate_result"
    assert axis_payload["name"] == "assistant_axis"
    assert list(axis_payload["layers"]) == ["0"]
    assert axis_payload["layers"]["0"]["default_count"] == 2
    assert axis_payload["layers"]["0"]["role_vector_count"] == 2
    assert axis_payload["summary"]["score_filtered"] is True

    scores = runner.run(
        AssistantAxisScoreSpec(
            feature=capture.feature("resid_post_full"),
            axis=axis,
            layer=0,
            model_id="google/gemma-2-27b-it",
            summaries=("mean",),
            emit_labels=True,
        )
    )
    score_payload = scores.result()

    assert score_payload["kind"] == "assistant_axis_score_result"
    assert score_payload["assistant_axis"]["layer"] == 0
    assert score_payload["summary"]["slice_row_count"] == len(dataset.examples)
    assert scores.label("projection__assistant_axis__layer_0__mean").resolve_values()


def test_unknown_assistant_axis_model_warns() -> None:
    with pytest.warns(UserWarning, match="best layer"):
        AssistantAxisVectorSpec(model_id="acme/unknown-instruct-model")


def test_assistant_axis_score_from_dict_defaults_to_generated_slice() -> None:
    spec = AssistantAxisScoreSpec.from_dict({"kind": "assistant_axis_score", "model_id": "qwen-3-32b"})

    assert spec.slices.names == ("generated",)


def _assistant_axis_training_dataset() -> Dataset:
    examples = [
        Example(
            key="default_a",
            prompt="Default answer A",
            labels={"axis_kind": "default", "role": "default", "adherence_score": 0},
        ),
        Example(
            key="default_b",
            prompt="Default answer B",
            labels={"axis_kind": "default", "role": "default", "adherence_score": 0},
        ),
        Example(
            key="pirate_a",
            prompt="Pirate answer A",
            labels={"axis_kind": "role", "role": "pirate", "adherence_score": 3},
        ),
        Example(
            key="pirate_b",
            prompt="Pirate answer B",
            labels={"axis_kind": "role", "role": "pirate", "adherence_score": 2},
        ),
        Example(
            key="doctor_a",
            prompt="Doctor answer A",
            labels={"axis_kind": "role", "role": "doctor", "adherence_score": 3},
        ),
    ]
    return Dataset.from_examples(examples, name="assistant_axis_tiny")
