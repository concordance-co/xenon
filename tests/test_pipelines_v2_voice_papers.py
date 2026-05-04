from __future__ import annotations

import importlib
from pathlib import Path

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    Example,
    LocalArtifactStore,
    LocalRunner,
    RefusalAblationSubspaceSpec,
    RefusalDirectionSelectionSpec,
    RefusalDirectionSpec,
    RefusalScoreSpec,
    ResidualSite,
    SectionSelector,
    TokenSelector,
    ToyEngine,
    TruthfulnessAblationSubspaceSpec,
    TruthfulnessDirectionSelectionSpec,
    TruthfulnessDirectionSpec,
    TruthfulnessScoreSpec,
    WorkflowOrchestrator,
    emotion_contrast_dataset,
    emotion_probe_story_dataset,
    refusal_direction_split_dataset,
    truthfulqa_answer_contrast_dataset,
)
from pipelines_v2.operations import operation_spec_from_dict
from papers.voice.common.smoke import token_metadata


def test_refusal_specs_roundtrip_execute_and_select_layer(tmp_path: Path) -> None:
    dataset = _refusal_dataset()
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=4, num_layers=2, sequence_length=6),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_post_full",
                    site="resid_post",
                    layers=[0, 1],
                    tokens=TokenSelector.full_sequence(),
                )
            ],
        )
    )

    direction_spec = RefusalDirectionSpec(
        feature=capture.feature("resid_post_full"),
        harmful_when=dataset.labels("direction_role").equals("harmful_train"),
        harmless_when=dataset.labels("direction_role").equals("harmless_train"),
        layers=[0, 1],
        tokens=TokenSelector.section("instruction"),
    )
    assert isinstance(operation_spec_from_dict(direction_spec.to_dict()), RefusalDirectionSpec)
    direction = runner.run(direction_spec)
    assert direction.result()["name"] == "refusal_direction"
    assert direction.result()["summary"]["positive_label"] == "harmful"

    scores = runner.run(
        RefusalScoreSpec(
            feature=capture.feature("resid_post_full"),
            direction=direction,
            layers=[0, 1],
            slices=SectionSelector.named("instruction"),
        )
    )
    assert scores.result()["kind"] == "refusal_score_result"

    selected = runner.run(
        RefusalDirectionSelectionSpec(
            direction=direction,
            scores=scores,
            harmful_when=dataset.labels("validation_role").equals("harmful_val"),
            harmless_when=dataset.labels("validation_role").equals("harmless_val"),
            layers=[0],
        )
    )
    assert selected.result()["summary"]["selected_layer"] == 0

    subspace = runner.run(RefusalAblationSubspaceSpec(direction=selected))
    assert subspace.result()["kind"] == "subspace_result"
    assert subspace.result()["layers"]["0"]["component_count"] == 1


def test_truthfulness_specs_roundtrip_execute_and_select_layer(tmp_path: Path) -> None:
    dataset = _truthfulness_dataset()
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / "artifacts"))
    capture = runner.run(
        CaptureSpec(
            engine=ToyEngine(hidden_size=4, num_layers=2, sequence_length=6),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_post_full",
                    site="resid_post",
                    layers=[0, 1],
                    tokens=TokenSelector.full_sequence(),
                )
            ],
        )
    )

    direction_spec = TruthfulnessDirectionSpec(
        feature=capture.feature("resid_post_full"),
        truthful_when=dataset.labels("direction_role").equals("truth_train"),
        untruthful_when=dataset.labels("direction_role").equals("false_train"),
        layers=[0, 1],
        tokens=TokenSelector.section("answer"),
    )
    assert isinstance(operation_spec_from_dict(direction_spec.to_dict()), TruthfulnessDirectionSpec)
    direction = runner.run(direction_spec)
    assert direction.result()["name"] == "truthfulness_direction"
    assert direction.result()["summary"]["positive_label"] == "truthful"

    scores = runner.run(
        TruthfulnessScoreSpec(
            feature=capture.feature("resid_post_full"),
            direction=direction,
            layers=[0, 1],
            slices=SectionSelector.named("answer"),
        )
    )
    selected = runner.run(
        TruthfulnessDirectionSelectionSpec(
            direction=direction,
            scores=scores,
            truthful_when=dataset.labels("validation_role").equals("truth_val"),
            untruthful_when=dataset.labels("validation_role").equals("false_val"),
            layers=[0],
        )
    )
    assert selected.result()["summary"]["selected_layer"] == 0

    subspace = runner.run(TruthfulnessAblationSubspaceSpec(direction=selected))
    assert subspace.result()["layers"]["0"]["named_components"]["truthfulness_direction_component"] == 0


def test_voice_dataset_helpers_are_lightweight_and_swappable() -> None:
    refusal = refusal_direction_split_dataset(split="train", revision="abc123", limit=4)
    truthfulqa = truthfulqa_answer_contrast_dataset(limit=5)
    emotion_probe = emotion_probe_story_dataset(limit=6)
    custom_emotions = emotion_contrast_dataset(
        [
            {"example_id": "a", "text": "The agent snapped back.", "emotion": "angry", "topic": "agent"},
            {"example_id": "b", "text": "The agent stayed steady.", "emotion": "calm", "topic": "agent"},
        ]
    )

    assert refusal.is_deferred
    assert refusal.source["kind"] == "url_json"
    assert refusal.fetch["prompt_column"] == "instruction"
    assert refusal.fetch["label_columns"] == ("harmtype", "split", "source_dataset", "category")
    assert "dataset/splits/harmful_train.json" in refusal.source["files"][0]["url"]

    assert truthfulqa.is_deferred
    assert truthfulqa.source["kind"] == "huggingface_list_contrast"
    assert truthfulqa.source["path"] == "truthful_qa"
    assert truthfulqa.fetch["label_name"] == "truthfulness"
    assert truthfulqa.fetch["prompt_template"] == "Question: {question}\nAnswer: {answer}"

    assert emotion_probe.is_deferred
    assert emotion_probe.source["kind"] == "huggingface"
    assert emotion_probe.source["path"] == "ryancodrai/emotion-probes"
    assert emotion_probe.fetch["label_columns"] == ("real_emotion",)

    assert not custom_emotions.is_deferred
    assert custom_emotions.labels("emotion").values == {"a": "angry", "b": "calm"}
    assert custom_emotions.cases("topic").values == {"a": "agent", "b": "agent"}


def test_voice_paper_workflows_plan_with_toyengine() -> None:
    module_names = [
        "papers.voice.assistant_axis.specs.workflow",
        "papers.voice.emotions.specs.workflow",
        "papers.voice.refusal_direction.specs.workflow",
        "papers.voice.honest_llama.specs.workflow",
    ]
    for module_name in module_names:
        module = importlib.import_module(module_name)
        workflow = module.build_workflow()
        runners = {name: spec.to_runner() for name, spec in module.build_runner_specs().items()}
        plan = WorkflowOrchestrator(runners=runners).plan(workflow)

        assert plan.steps
        assert not any(step.execution.errors for step in plan.steps)
        capture_steps = [step for step in workflow.steps if step.name == "capture"]
        assert capture_steps
        assert isinstance(capture_steps[0].spec.engine, ToyEngine)


def _refusal_dataset() -> Dataset:
    metadata = token_metadata("instruction", "completion")
    return Dataset.from_examples(
        [
            Example(key="harmful_train", prompt="bad instruction", labels={"direction_role": "harmful_train", "validation_role": "unused"}, metadata=metadata),
            Example(key="harmless_train", prompt="safe instruction", labels={"direction_role": "harmless_train", "validation_role": "unused"}, metadata=metadata),
            Example(key="harmful_val", prompt="bad validation", labels={"direction_role": "unused", "validation_role": "harmful_val"}, metadata=metadata),
            Example(key="harmless_val", prompt="safe validation", labels={"direction_role": "unused", "validation_role": "harmless_val"}, metadata=metadata),
        ],
        name="refusal_test",
    )


def _truthfulness_dataset() -> Dataset:
    metadata = token_metadata("question", "answer")
    return Dataset.from_examples(
        [
            Example(key="truth_train", prompt="true answer", labels={"direction_role": "truth_train", "validation_role": "unused"}, metadata=metadata),
            Example(key="false_train", prompt="false answer", labels={"direction_role": "false_train", "validation_role": "unused"}, metadata=metadata),
            Example(key="truth_val", prompt="true validation", labels={"direction_role": "unused", "validation_role": "truth_val"}, metadata=metadata),
            Example(key="false_val", prompt="false validation", labels={"direction_role": "unused", "validation_role": "false_val"}, metadata=metadata),
        ],
        name="truthfulness_test",
    )
