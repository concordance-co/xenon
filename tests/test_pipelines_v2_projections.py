from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipelines_v2.api import (
    CaptureSpec,
    CoordinateImportSpec,
    Dataset,
    DirectionSpec,
    Example,
    LocalArtifactStore,
    LocalRunner,
    ProjectionCalibrationSpec,
    ProjectionSpec,
    PromptMetadataBuilder,
    ResidualSite,
    SectionSelector,
    TokenPooling,
    TokenSelector,
    ToyEngine,
)
from pipelines_v2.engine.prompt_metadata import resolve_prompt_metadata, section_records_from_metadata, token_sections_from_metadata
from pipelines_v2.operations.execution.common import routing_vector_from_record


def test_prompt_metadata_builder_chat_turns_produces_section_records() -> None:
    builder = PromptMetadataBuilder.chat_turns()
    prompt = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "What is two plus two?"},
        {"role": "assistant", "content": "It is four."},
    ]
    rendered = (
        "SYSTEM\nYou are terse.\n\n"
        "USER\nWhat is two plus two?\n\n"
        "ASSISTANT\nIt is four.\n"
    )
    offsets = [(index, index + 1) for index in range(len(rendered))]

    metadata = resolve_prompt_metadata(
        metadata={},
        rendered_prompt=rendered,
        builder=builder,
        prompt=prompt,
    )
    token_sections = token_sections_from_metadata(
        metadata=metadata,
        offsets=offsets,
        require_sections=False,
        allow_char_spans=True,
    )
    section_records = section_records_from_metadata(
        metadata=metadata,
        offsets=offsets,
        token_sections=token_sections,
        allow_char_spans=True,
    )

    names = [record["name"] for record in section_records]
    assert "user_turn_001" in names
    assert "assistant_turn_002" in names
    assistant_record = next(record for record in section_records if record["name"] == "assistant_turn_002")
    assert assistant_record["role"] == "assistant"
    assert assistant_record["unit"] == "turn"
    assert assistant_record["token_positions"]


def test_prompt_metadata_builder_chat_turns_can_mark_assistant_colon() -> None:
    builder = PromptMetadataBuilder.chat_turns(include_assistant_colon=True)
    prompt = [{"role": "user", "content": "How should I respond?"}]
    rendered = "Human: How should I respond?\n\nAssistant:"
    offsets = [(index, index + 1) for index in range(len(rendered))]

    metadata = resolve_prompt_metadata(
        metadata={},
        rendered_prompt=rendered,
        builder=builder,
        prompt=prompt,
    )
    token_sections = token_sections_from_metadata(
        metadata=metadata,
        offsets=offsets,
        require_sections=False,
        allow_char_spans=True,
    )
    section_records = section_records_from_metadata(
        metadata=metadata,
        offsets=offsets,
        token_sections=token_sections,
        allow_char_spans=True,
    )

    assert token_sections["assistant_colon"]
    marker = next(record for record in section_records if record["name"] == "assistant_colon")
    assert marker["role"] == "assistant"
    assert marker["unit"] == "marker"
    assert marker["tags"] == {"marker": "assistant_colon"}


def test_projection_scores_semantic_assistant_turn_slices_and_emits_labels(tmp_path: Path) -> None:
    metadata = _conversation_section_metadata()
    dataset = Dataset.from_examples(
        [
            Example(
                key="conv_positive",
                prompt="unused",
                labels={"style": "positive"},
                metadata=metadata,
            ),
            Example(
                key="conv_negative",
                prompt="unused",
                labels={"style": "negative"},
                metadata=metadata,
            ),
        ]
    )

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

    direction = runner.run(
        DirectionSpec(
            feature=capture.feature("resid_post_full"),
            positive=dataset.labels("style").equals("positive"),
            negative=dataset.labels("style").equals("negative"),
            layers=[0],
            tokens=TokenSelector.section("assistant_turn_001"),
            pooling=TokenPooling.mean(),
        )
    )

    projections = runner.run(
        ProjectionSpec(
            feature=capture.feature("resid_post_full"),
            coordinates=[direction],
            slices=SectionSelector.where(unit="turn", role="assistant"),
            metric="signed_dot",
            summaries=["mean", "trend", "first_last_delta"],
            emit_labels=True,
        )
    )

    payload = projections.result()
    assert payload["kind"] == "projection_result"
    assert payload["summary"]["slice_row_count"] == 4
    assert len(payload["example_summaries"]) == 2
    assert {row["coordinate"] for row in payload["rows"]} == {"coordinate_0"}
    assert {row["slice_name"] for row in payload["rows"]} == {"assistant_turn_001", "assistant_turn_002"}
    assert all(row["role"] == "assistant" for row in payload["rows"])

    mean_label = projections.label("projection__coordinate_0__layer_0__mean").resolve_values()
    assert set(mean_label) == {"conv_negative", "conv_positive"}


def test_coordinate_import_projection_and_quantile_calibration(tmp_path: Path) -> None:
    metadata = _conversation_section_metadata()
    dataset = Dataset.from_examples(
        [
            Example(
                key="conv_a",
                prompt="unused",
                labels={"split": "reference"},
                metadata=metadata,
            ),
            Example(
                key="conv_b",
                prompt="unused",
                labels={"split": "reference"},
                metadata=metadata,
            ),
        ]
    )

    axis_path = tmp_path / "assistant_axis.npy"
    np.save(axis_path, np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))

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
    imported = runner.run(
        CoordinateImportSpec(
            path=str(axis_path),
            format="npy",
            name="assistant_axis",
            select_layer=0,
        )
    )
    imported_payload = imported.result()
    assert imported_payload["kind"] == "coordinate_result"
    assert imported_payload["name"] == "assistant_axis"

    projections = runner.run(
        ProjectionSpec(
            feature=capture.feature("resid_post_full"),
            coordinates=[imported],
            slices=SectionSelector.where(role="assistant", unit="turn"),
            metric="signed_dot",
            summaries=["mean"],
        )
    )
    calibration = runner.run(
        ProjectionCalibrationSpec(
            projections=projections,
            fit_on=dataset.labels("split").equals("reference"),
            bands=["A", "B", "C"],
            orientation={"assistant_axis": "higher_is_more_assistant"},
        )
    )

    definitions = calibration.result()["definitions"]
    assert len(definitions) == 1
    assert definitions[0]["coordinate"] == "assistant_axis"
    assert definitions[0]["orientation"] == "higher_is_more_assistant"
    assert definitions[0]["bands"] == ["A", "B", "C"]
    assert len(definitions[0]["thresholds"]) == 2

    calibration_all = runner.run(ProjectionCalibrationSpec(projections=projections))
    assert calibration_all.result()["summary"]["fit_example_count"] == 2
    assert calibration_all.manifest().example_coverage["example_count"] == 2


def test_routing_projection_vector_reader_accepts_capture_record_schema() -> None:
    topk = routing_vector_from_record(
        {"topk_from_gate": {"expert_ids": [1, 3], "weights": [0.25, 0.75]}},
        routing_policy={"num_experts": 4},
    )
    expert_load = routing_vector_from_record(
        {"expert_load": {"counts": {"1": 2, "3": 1}}},
        routing_policy={"num_experts": 4},
    )

    assert topk.tolist() == [0.0, 0.25, 0.0, 0.75]
    assert expert_load.tolist() == [0.0, 2.0, 0.0, 1.0]


def test_coordinate_import_casts_torch_bfloat16_axes(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    axis_path = tmp_path / "axis.pt"
    torch.save(torch.ones((1, 4), dtype=torch.bfloat16), axis_path)

    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    imported = runner.run(
        CoordinateImportSpec(
            path=str(axis_path),
            format="torch_tensor_or_axis_dict",
            select_layer=0,
        )
    )

    assert imported.result()["layers"]["0"]["vector"] == [0.5, 0.5, 0.5, 0.5]


def _conversation_section_metadata() -> dict[str, object]:
    return {
        "token_sections": {
            "user_turn_000": [0, 1],
            "assistant_turn_001": [2, 3],
            "assistant_turn_002": [4, 5],
        },
        "section_records": [
            {
                "name": "user_turn_000",
                "role": "user",
                "unit": "turn",
                "index": 0,
                "token_positions": [0, 1],
            },
            {
                "name": "assistant_turn_001",
                "role": "assistant",
                "unit": "turn",
                "index": 1,
                "token_positions": [2, 3],
            },
            {
                "name": "assistant_turn_002",
                "role": "assistant",
                "unit": "turn",
                "index": 2,
                "token_positions": [4, 5],
            },
        ],
    }
