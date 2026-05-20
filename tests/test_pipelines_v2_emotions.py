from __future__ import annotations

import json
from pathlib import Path

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    EmotionDirectionSpec,
    EmotionGeometrySpec,
    EmotionPrecomputedVectorSpaceSpec,
    EmotionScoreSpec,
    EmotionVectorSpaceSpec,
    Example,
    LocalArtifactStore,
    LocalRunner,
    ResidualSite,
    SectionSelector,
    TokenSelector,
    ToyEngine,
)
from pipelines_v2.mechinterp.emotions import EMOTION_VECTOR_SPACE_KIND
from pipelines_v2.operations import operation_spec_from_dict


def test_emotion_vector_space_generation_scoring_direction_and_geometry(tmp_path: Path) -> None:
    dataset = _emotion_dataset()
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=4, num_layers=1, sequence_length=6),
            dataset=dataset,
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

    vector_space = runner.run(
        EmotionVectorSpaceSpec(
            feature=capture.feature("resid_post_full"),
            concept_by=dataset.labels("emotion"),
            layers=[0],
            tokens=TokenSelector.full_sequence(),
            min_examples_per_concept=1,
            metadata={"paper": "transformer-circuits-2026-emotions"},
        )
    )
    payload = vector_space.result()

    assert payload["kind"] == EMOTION_VECTOR_SPACE_KIND
    assert payload["vector_space_kind"] == "story"
    assert set(payload["layers"]["0"]["concepts"]) == {"happy", "sad", "self-conscious"}
    assert payload["summary"]["concept_count"] == 3
    assert payload["metadata"]["formula"] == "mean(concept examples) - mean(concept means)"

    scores = runner.run(
        EmotionScoreSpec(
            feature=capture.feature("resid_post_full"),
            vector_space=vector_space,
            concepts=("happy",),
            layers=[0],
            slices=SectionSelector.named("story"),
            summaries=("mean",),
            emit_labels=True,
        )
    )
    score_payload = scores.result()

    assert score_payload["kind"] == "emotion_score_result"
    assert score_payload["summary"]["slice_row_count"] == len(dataset.examples)
    assert {row["emotion"] for row in score_payload["rows"]} == {"happy"}
    assert scores.label("projection__emotion_happy__layer_0__mean").resolve_values()

    direction = runner.run(
        EmotionDirectionSpec(
            vector_space=vector_space,
            concept="happy",
            layers=[0],
            residual_norm_by_layer={0: 2.0},
        )
    )
    direction_payload = direction.result()
    assert direction_payload["kind"] == "direction_result"
    assert direction_payload["layers"]["0"]["emotion"] == "happy"
    assert direction_payload["layers"]["0"]["residual_norm"] == 2.0

    geometry = runner.run(
        EmotionGeometrySpec(
            vector_space=vector_space,
            layers=[0],
            pca_components=1,
            cluster_count=2,
        )
    ).result()
    assert geometry["kind"] == "emotion_geometry_result"
    assert geometry["summary"]["concept_count"] == 3
    assert geometry["layers"]["0"]["concepts"] == ["happy", "sad", "self-conscious"]
    assert len(geometry["layers"]["0"]["cosine_similarity"]) == 3
    assert set(geometry["layers"]["0"]["clusters"]) == {"happy", "sad", "self-conscious"}

    punctuation_scores = runner.run(
        EmotionScoreSpec(
            feature=capture.feature("resid_post_full"),
            vector_space=vector_space,
            concepts=("self-conscious",),
            layers=[0],
            slices=SectionSelector.named("story"),
            summaries=("mean",),
        )
    ).result()
    assert {row["emotion"] for row in punctuation_scores["rows"]} == {"self-conscious"}


def test_precomputed_emotion_vector_space_loads_compact_json_and_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "emotion_vectors.json"
    path.write_text(
        json.dumps(
            {
                "happy": [1.0, 0.0, 0.0, 0.0],
                "sad": [0.0, 1.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )
    spec = EmotionPrecomputedVectorSpaceSpec(
        path=str(path),
        select_layer=0,
        vector_space_kind="story",
    )
    loaded = operation_spec_from_dict(spec.to_dict())
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    artifact = runner.run(spec)
    payload = artifact.result()

    assert isinstance(loaded, EmotionPrecomputedVectorSpaceSpec)
    assert payload["kind"] == EMOTION_VECTOR_SPACE_KIND
    assert payload["layers"]["0"]["concepts"]["happy"]["vector"] == [1.0, 0.0, 0.0, 0.0]
    assert payload["summary"]["concept_count"] == 2


def _emotion_dataset() -> Dataset:
    metadata = {
        "token_sections": {"story": [0, 1, 2, 3, 4, 5]},
        "section_records": [
            {
                "name": "story",
                "unit": "story",
                "index": 0,
                "token_positions": [0, 1, 2, 3, 4, 5],
            }
        ],
    }
    return Dataset.from_examples(
        [
            Example(key="happy_a", prompt="happy story a", labels={"emotion": "happy"}, metadata=metadata),
            Example(key="happy_b", prompt="happy story b", labels={"emotion": "happy"}, metadata=metadata),
            Example(key="sad_a", prompt="sad story a", labels={"emotion": "sad"}, metadata=metadata),
            Example(key="sad_b", prompt="sad story b", labels={"emotion": "sad"}, metadata=metadata),
            Example(
                key="self_conscious_a",
                prompt="self conscious story a",
                labels={"emotion": "self-conscious"},
                metadata=metadata,
            ),
            Example(
                key="self_conscious_b",
                prompt="self conscious story b",
                labels={"emotion": "self-conscious"},
                metadata=metadata,
            ),
        ],
        name="emotion_stories",
    )
