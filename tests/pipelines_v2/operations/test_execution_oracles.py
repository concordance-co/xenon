from __future__ import annotations

import numpy as np
import pytest

from pipelines_v2.api import (
    Dataset,
    DirectionSpec,
    Example,
    LabelFieldsSpec,
    LabelMapSpec,
    PairDeltaSpec,
    ProbeSpec,
    ProjectionSpec,
    SectionSelector,
    TokenPooling,
    TokenSelector,
)
from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.operations.execution.derive import run_label_fields, run_label_map, run_pair_delta
from pipelines_v2.operations.execution.projections import run_projection
from pipelines_v2.operations.execution.readouts import run_probe
from pipelines_v2.operations.execution.representation import run_direction
from tests.pipelines_v2.factories import feature_ref_from_payload, residual_feature_payload


@pytest.mark.unit
@pytest.mark.interp
def test_direction_and_pair_delta_have_known_vector_answers_without_toy_engine() -> None:
    dataset = Dataset.from_examples(
        [
            Example(
                key="case_a_negative",
                prompt="unused",
                labels={"side": "negative", "family": "size", "split": "train"},
                cases={"pair": "case_a"},
                case_key="case_a",
            ),
            Example(
                key="case_a_positive",
                prompt="unused",
                labels={"side": "positive", "family": "size", "split": "train"},
                cases={"pair": "case_a"},
                case_key="case_a",
            ),
            Example(
                key="case_b_negative",
                prompt="unused",
                labels={"side": "negative", "family": "activity", "split": "test"},
                cases={"pair": "case_b"},
                case_key="case_b",
            ),
            Example(
                key="case_b_positive",
                prompt="unused",
                labels={"side": "positive", "family": "activity", "split": "test"},
                cases={"pair": "case_b"},
                case_key="case_b",
            ),
        ]
    )
    feature = feature_ref_from_payload(
        residual_feature_payload(
            rows={
                "case_a_negative": [[0.0, 1.0]],
                "case_a_positive": [[2.0, 1.0]],
                "case_b_negative": [[1.0, -2.0]],
                "case_b_positive": [[3.0, -2.0]],
            }
        )
    )

    direction = run_direction(
        DirectionSpec(
            feature=feature,
            positive=dataset.labels("side").equals("positive"),
            negative=dataset.labels("side").equals("negative"),
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.mean(),
        )
    )

    layer = direction.payload["layers"]["0"]
    assert layer["raw_vector"] == pytest.approx([2.0, 0.0])
    assert layer["vector"] == pytest.approx([1.0, 0.0])
    assert layer["norm"] == pytest.approx(2.0)
    assert direction.example_coverage["example_keys"] == [
        "case_a_negative",
        "case_a_positive",
        "case_b_negative",
        "case_b_positive",
    ]

    pair_delta = run_pair_delta(
        PairDeltaSpec(
            feature=feature,
            case=dataset.cases("pair"),
            positive=dataset.labels("side").equals("positive"),
            negative=dataset.labels("side").equals("negative"),
            labels={"family": dataset.labels("family"), "split": dataset.labels("split")},
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.mean(),
        )
    )

    delta_layer = pair_delta.features["delta"]["layers"]["0"]
    assert sorted(delta_layer) == ["case_a", "case_b"]
    np.testing.assert_allclose(delta_layer["case_a"]["values"], [[2.0, 0.0]])
    np.testing.assert_allclose(delta_layer["case_b"]["values"], [[2.0, 0.0]])
    assert pair_delta.labels["family"]["values"] == {"case_a": "size", "case_b": "activity"}
    assert pair_delta.labels["split"]["values"] == {"case_a": "train", "case_b": "test"}


@pytest.mark.unit
@pytest.mark.interp
def test_probe_fixed_split_learns_separable_known_answer_and_persists_predictions_without_toy_engine() -> None:
    examples = []
    rows = {}
    test_keys = set()
    values = list(range(-20, 0)) + list(range(1, 21))
    for index, value in enumerate(values):
        split = "test" if abs(value) % 5 == 0 else "train"
        label = "negative" if value < 0 else "positive"
        key = f"ex_{index}"
        if split == "test":
            test_keys.add(key)
        examples.append(Example(key=key, prompt="unused", labels={"class": label, "split": split}))
        rows[key] = [[float(value), float(-value)]]
    dataset = Dataset.from_examples(examples)
    feature = feature_ref_from_payload(residual_feature_payload(rows=rows))

    result = run_probe(
        ProbeSpec(
            feature=feature,
            labels=dataset.labels("class"),
            split=dataset.labels("split"),
            train_values=("train",),
            test_values=("test",),
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.mean(),
            metrics=("accuracy", "balanced_accuracy", "auroc"),
            persist_predictions=True,
        )
    )

    layer = result.payload["layers"][0]
    assert layer["split_mode"] == "fixed"
    assert layer["accuracy"] == pytest.approx(1.0)
    assert layer["balanced_accuracy"] == pytest.approx(1.0)
    assert layer["auroc"] == pytest.approx(1.0)
    assert layer["test_prediction_count"] == len(test_keys)
    assert {row["example_key"] for row in layer["test_predictions"]} == test_keys
    assert result.payload["summary"]["best_layer"] == 0
    assert result.payload["summary"]["best_value"] == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.interp
def test_probe_fixed_split_supports_staged_finetuning_without_toy_engine() -> None:
    examples = []
    rows = {}
    split_values = {
        "synthetic_train": [-8, -7, -6, -5, 5, 6, 7, 8],
        "dev_domain": [-4, -3, 3, 4],
        "test_domain": [-10, -9, 9, 10],
    }
    for split, values in split_values.items():
        for value in values:
            key = f"{split}_{value}"
            label = "negative" if value < 0 else "positive"
            examples.append(Example(key=key, prompt="unused", labels={"class": label, "split": split}))
            rows[key] = [[float(value), float(-value)]]
    dataset = Dataset.from_examples(examples)
    feature = feature_ref_from_payload(residual_feature_payload(rows=rows))

    spec = ProbeSpec(
        feature=feature,
        labels=dataset.labels("class"),
        split=dataset.labels("split"),
        train_values=("synthetic_train", "dev_domain"),
        train_stages=(("synthetic_train",), ("dev_domain",)),
        stage_epochs=(1, 3),
        test_values=("test_domain",),
        tokens=TokenSelector.full_sequence(),
        pooling=TokenPooling.mean(),
        metrics=("accuracy", "balanced_accuracy", "auroc"),
    )
    serialized = spec.to_dict()
    assert serialized["train_stages"] == [["synthetic_train"], ["dev_domain"]]
    assert serialized["stage_epochs"] == [1, 3]

    result = run_probe(spec)

    layer = result.payload["layers"][0]
    assert layer["split_mode"] == "fixed"
    assert layer["training_mode"] == "staged_finetune"
    assert layer["train_stages"] == [["synthetic_train"], ["dev_domain"]]
    assert layer["stage_epochs"] == [1, 3]
    assert layer["accuracy"] == pytest.approx(1.0)
    assert layer["balanced_accuracy"] == pytest.approx(1.0)
    assert layer["auroc"] == pytest.approx(1.0)


@pytest.mark.unit
def test_label_map_and_label_fields_strict_modes_are_known_answer_checks() -> None:
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="unused", labels={"family": "trade_size", "json": '{"action":"buy","size":"large"}'}),
            Example(key="b", prompt="unused", labels={"family": "observe_only", "json": {"action": "observe", "size": "none"}}),
        ]
    )

    mapped = run_label_map(
        LabelMapSpec(
            source=dataset.labels("family"),
            output_name="mode",
            mapping={"trade_size": "trade", "observe_only": "monitor"},
            strict=True,
        )
    )
    assert mapped.labels["mode"]["values"] == {"a": "trade", "b": "monitor"}

    fields = run_label_fields(
        LabelFieldsSpec(
            source=dataset.labels("json"),
            fields={"action": "action", "size": "size"},
            strict=True,
        )
    )
    assert fields.labels["action"]["values"] == {"a": "buy", "b": "observe"}
    assert fields.labels["size"]["values"] == {"a": "large", "b": "none"}

    with pytest.raises(SpecValidationError, match="missing mappings"):
        run_label_map(
            LabelMapSpec(
                source=dataset.labels("family"),
                output_name="mode",
                mapping={"trade_size": "trade"},
                strict=True,
            )
        )
    with pytest.raises(SpecValidationError, match="missing requested fields"):
        run_label_fields(
            LabelFieldsSpec(
                source=dataset.labels("json"),
                fields={"missing": "missing"},
                strict=True,
            )
        )


@pytest.mark.unit
def test_projection_scores_section_slices_against_known_coordinate_without_toy_engine() -> None:
    section_records = [
        {
            "name": "user_turn",
            "index": 0,
            "role": "user",
            "unit": "turn",
            "token_positions": [0, 1],
            "tags": {"speaker": "user"},
        },
        {
            "name": "assistant_turn",
            "index": 1,
            "role": "assistant",
            "unit": "turn",
            "token_positions": [2, 3],
            "tags": {"speaker": "assistant"},
        },
    ]
    feature = feature_ref_from_payload(
        residual_feature_payload(
            rows={
                "conv_a": [[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]],
                "conv_b": [[-2.0, 0.0], [-4.0, 0.0], [0.0, -1.0], [0.0, -3.0]],
            },
            section_records=section_records,
        )
    )
    coordinate = {
        "kind": "coordinate_result",
        "name": "x_axis",
        "layers": {"0": {"vector": [1.0, 0.0]}},
    }

    result = run_projection(
        ProjectionSpec(
            feature=feature,
            coordinates=[coordinate],
            slices=SectionSelector.all(),
            pooling=TokenPooling.mean(),
            metric="signed_dot",
            summaries=("mean", "first_last_delta"),
            emit_labels=True,
        )
    )

    rows = {
        (row["example_key"], row["slice_name"]): row["score"]
        for row in result.payload["rows"]
    }
    assert set(rows) == {
        ("conv_a", "user_turn"),
        ("conv_a", "assistant_turn"),
        ("conv_b", "user_turn"),
        ("conv_b", "assistant_turn"),
    }
    assert rows[("conv_a", "user_turn")] == pytest.approx(2.0)
    assert rows[("conv_a", "assistant_turn")] == pytest.approx(0.0)
    assert rows[("conv_b", "user_turn")] == pytest.approx(-3.0)
    assert rows[("conv_b", "assistant_turn")] == pytest.approx(0.0)
    summaries = {
        row["example_key"]: row["metrics"]
        for row in result.payload["example_summaries"]
    }
    assert summaries["conv_a"]["mean"] == pytest.approx(1.0)
    assert summaries["conv_a"]["first_last_delta"] == pytest.approx(-2.0)
    assert summaries["conv_b"]["mean"] == pytest.approx(-1.5)
    assert summaries["conv_b"]["first_last_delta"] == pytest.approx(3.0)
    label_values = result.labels["projection__x_axis__layer_0__mean"]["values"]
    assert label_values["conv_a"] == pytest.approx(1.0)
    assert label_values["conv_b"] == pytest.approx(-1.5)
