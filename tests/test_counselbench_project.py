from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pipelines_v2.api import Dataset, Example
from projects.COUNSELBENCH.advice_safety.phase_03.specs import workflow
from projects.COUNSELBENCH.advice_safety.phase_03_eval.specs import workflow as eval_workflow
from projects.COUNSELBENCH.advice_safety.phase_03b.specs import workflow as adv_03b_workflow
from projects.COUNSELBENCH.advice_safety.phase_04.specs import workflow as phase_04_workflow
from projects.COUNSELBENCH.shared.counselbench_dataset import (
    ADV_FAILURE_MODES,
    aggregate_eval_records,
    adv_generated_chat_prompt_sections,
    adv_prompt_chat_sections,
    adv_records_to_examples,
    build_eval_aggregated_dataset,
    build_phase4_pairing_candidates,
    build_raw_adv_source_dataset,
    build_raw_eval_source_dataset,
    build_successful_generation_capture_dataset,
    eval_chat_prompt_sections,
    evaluate_generation_quality_gate,
    question_response_key,
    run_eval_gated_readouts,
    summarize_eval_cheap_baselines,
    summarize_eval_confound_inventory,
    summarize_eval_label_support,
    summarize_geometry_metrics,
    summarize_generated_label_support,
)


def test_counselbench_adv_melts_wide_rows_to_balanced_examples() -> None:
    records = [
        {
            "source_row_id": "adv_row_000000",
            **{mode: f"{mode} prompt with therapy and medication wording" for mode in ADV_FAILURE_MODES},
        },
        {
            "source_row_id": "adv_row_000001",
            **{mode: f"second {mode} prompt" for mode in ADV_FAILURE_MODES},
        },
    ]

    examples = adv_records_to_examples(records, limit_per_mode=1)

    assert len(examples) == len(ADV_FAILURE_MODES)
    assert {example.labels["adv_failure_mode"] for example in examples} == set(ADV_FAILURE_MODES)
    assert all(example.labels["source_row_id"] == "adv_row_000000" for example in examples)
    assert all(example.labels["split"] == "test" for example in examples)
    assert all(example.labels["trigger_medication"] == "yes" for example in examples)
    assert all(example.prompt == [{"role": "user", "content": example.labels["prompt_text"]}] for example in examples)
    assert all(example.metadata["prompt_format"] == "chat_user_raw_question" for example in examples)
    assert "Assistant response:" not in str([example.prompt for example in examples])


def test_counselbench_eval_aggregation_freezes_stable_question_response_labels() -> None:
    base = {
        "questionID": "questionID_1",
        "questionTitle": "How can I get help?",
        "questionText": "I feel anxious and want support.",
        "response": "It sounds hard. Consider speaking with a licensed professional.",
        "topic": "anxiety",
        "responder": "gpt4",
    }
    records = [
        {
            **base,
            "survey_id": 1,
            "overall_score": 5,
            "empathy_score": 5,
            "specificity_score": 4,
            "medical_advice_score": "No",
            "factual_consistency_score": 4,
            "toxicity_score": 1,
            "toxicity_copy": "",
        },
        {
            **base,
            "survey_id": 2,
            "overall_score": 4,
            "empathy_score": 3,
            "specificity_score": 4,
            "medical_advice_score": "No",
            "factual_consistency_score": 4,
            "toxicity_score": 1,
            "toxicity_copy": "",
        },
        {
            **base,
            "survey_id": 3,
            "overall_score": 5,
            "empathy_score": 4,
            "specificity_score": 5,
            "medical_advice_score": "Yes",
            "factual_consistency_score": 2,
            "toxicity_score": 3,
            "toxicity_copy": "sounds judgmental",
        },
    ]

    rows = aggregate_eval_records(records)

    assert len(rows) == 1
    row = rows[0]
    assert row["question_response_key"] == question_response_key(base)
    assert row["annotator_count"] == 3
    assert row["empathy_high"] == "yes"
    assert row["specificity_high"] == "yes"
    assert row["medical_boundary_any_flag"] == "yes"
    assert row["medical_boundary_violation"] == "no"
    assert row["toxicity_or_judgmental"] == "yes"
    assert row["empathy_score_disagreement"] is not None


def test_counselbench_eval_aggregation_builds_dataset_with_question_grouping() -> None:
    raw = Dataset.from_examples(
        [
            Example(
                key="annotation_1",
                prompt="response text",
                labels={
                    "questionID": "q1",
                    "questionTitle": "Title",
                    "questionText": "Question text",
                    "response": "Response text",
                    "topic": "depression",
                    "responder": "human",
                    "overall_score": 4,
                    "empathy_score": 4,
                    "specificity_score": 4,
                    "medical_advice_score": "No",
                    "factual_consistency_score": 4,
                    "toxicity_score": 1,
                },
            )
        ]
    )

    result = build_eval_aggregated_dataset(raw_eval=raw)
    dataset = Dataset.from_dict(result["payload"]["dataset"])

    assert dataset.example_keys() == [question_response_key(raw.examples[0].labels)]
    assert dataset.examples[0].prompt == [{"role": "user", "content": "Question text"}, {"role": "assistant", "content": "Response text"}]
    assert dataset.labels("empathy_high").values == {dataset.examples[0].key: "yes"}
    assert dataset.cases("questionID").values == {dataset.examples[0].key: "q1"}
    assert dataset.examples[0].metadata["prompt_format"] == "chat_user_question_assistant_response"


def test_eval_chat_prompt_sections_find_question_and_response_spans() -> None:
    rendered = "<|im_start|>user\nQuestion text<|im_end|>\n<|im_start|>assistant\nResponse text<|im_end|>\n"

    result = eval_chat_prompt_sections(rendered)
    sections = result["token_sections"]

    assert rendered[sections["question"]["char_start"]:sections["question"]["char_end"]] == "Question text"
    assert rendered[sections["response"]["char_start"]:sections["response"]["char_end"]] == "Response text"
    assert sections["question_end"]["char_end"] <= sections["question"]["char_end"]
    assert sections["response_end"]["char_end"] <= sections["response"]["char_end"]


def test_adv_chat_prompt_sections_find_prompt_risk_and_generated_spans() -> None:
    prompt_rendered = "<|im_start|>user\nShould I take medication?<|im_end|>\n<|im_start|>assistant\n"
    generated_rendered = "<|im_start|>user\nShould I take medication?<|im_end|>\n<|im_start|>assistant\nTalk with a doctor.<|im_end|>\n"

    prompt_sections = adv_prompt_chat_sections(prompt_rendered)["token_sections"]
    generated_sections = adv_generated_chat_prompt_sections(generated_rendered)["token_sections"]

    assert prompt_rendered[prompt_sections["prompt"]["char_start"]:prompt_sections["prompt"]["char_end"]] == "Should I take medication?"
    assert prompt_rendered[prompt_sections["risk_span"]["char_start"]:prompt_sections["risk_span"]["char_end"]] == "medication"
    assert generated_rendered[generated_sections["generated"]["char_start"]:generated_sections["generated"]["char_end"]] == "Talk with a doctor."


def test_counselbench_generation_capture_dataset_adds_sections_and_smoke_summary() -> None:
    source = adv_records_to_examples(
        [
            {
                "source_row_id": "adv_row_000005",
                **{mode: f"{mode} prompt" for mode in ADV_FAILURE_MODES},
            }
        ],
        limit_per_mode=1,
    )[0]
    artifact = _FakeArtifact(
        {
            "rows": [
                {
                    "example_key": source.key,
                    "example": source.to_dict(),
                    "generated_text": "You may want to ask a doctor about medication options.",
                    "finish_reason": "stop",
                    "generated_token_ids": [1, 2, 3],
                }
            ]
        }
    )

    capture_result = build_successful_generation_capture_dataset(generation=artifact)
    dataset = Dataset.from_dict(capture_result["payload"]["dataset"])
    gate_result = evaluate_generation_quality_gate(generation_artifact=artifact)

    assert dataset.example_keys() == [source.key]
    assert dataset.examples[0].prompt == [
        {"role": "user", "content": source.labels["prompt_text"]},
        {"role": "assistant", "content": "You may want to ask a doctor about medication options."},
    ]
    assert "token_sections" not in dataset.examples[0].metadata
    assert dataset.labels("medical_boundary_violation").values == {source.key: "yes"}
    assert gate_result["payload"]["kind"] == "counselbench_adv_generation_quality_gate"
    assert gate_result["payload"]["summary"]["manual_review_required"] is True
    assert gate_result["payload"]["summary"]["failure_mode_counts"] == {"apathetic": 1}


def test_counselbench_generated_label_support_gates_one_class_readouts() -> None:
    examples = [
        Example(
            key=f"example_{index}",
            prompt=[
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": "safe response"},
            ],
            labels={
                "medical_boundary_violation": "no",
                "split": "train" if index < 2 else "test",
                "adv_failure_mode": "apathetic",
                "topic": "general_counseling",
                "response_length_bucket": "short",
            },
        )
        for index in range(4)
    ]
    result = summarize_generated_label_support(dataset=Dataset.from_examples(examples))
    summary = result["payload"]["summary"]

    assert summary["medical_boundary_violation_counts"] == {"no": 4}
    assert summary["generated_boundary_readout_ready"] is False
    assert summary["recommendation"] == "skip_generated_boundary_probe_until_min_class_support"


def test_eval_gated_readouts_skip_unsupported_labels() -> None:
    class FakeCapture:
        def feature(self, name: str) -> str:
            return f"feature:{name}"

    examples = [
        Example(
            key=f"eval_{index}",
            prompt=[
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "response"},
            ],
            labels={
                "questionID": f"question_{index}",
                "split": "train" if index < 3 else "test",
                "response_text": "safe response",
                "medical_boundary_violation": "no",
                "empathy_high": "no",
                "specificity_high": "no",
                "toxicity_or_judgmental": "no",
                "overall_quality_high": "no",
                "topic": "general_counseling",
            },
        )
        for index in range(6)
    ]

    result = run_eval_gated_readouts(
        dataset=Dataset.from_examples(examples),
        capture=FakeCapture(),
        cheap_baselines={"payload": {"summary": {"results": {}}}},
    )
    summary = result["payload"]["summary"]

    assert summary["completed_labels"] == []
    assert set(summary["skipped_labels"]) == {
        "medical_boundary_violation",
        "empathy_high",
        "specificity_high",
        "toxicity_or_judgmental",
        "overall_quality_high",
    }
    assert result["payload"]["labels"]["medical_boundary_violation"]["reason"] == "label_support_gate_failed"
    assert summary["decision"] == "CONTROL_INSUFFICIENT"


def test_counselbench_eval_label_support_and_phase4_pairing_candidates() -> None:
    examples = [
        Example(
            key=f"eval_{index}",
            prompt=[{"role": "user", "content": f"Question {index}"}, {"role": "assistant", "content": f"Response {index}"}],
            labels={
                "questionID": f"q{index // 2}",
                "topic": "medication",
                "responder": "model",
                "split": "train" if index < 4 else "test",
                "empathy_high": "yes" if index % 2 == 0 else "no",
                "specificity_high": "yes",
                "overall_quality_high": "yes" if index % 2 == 0 else "no",
                "medical_boundary_violation": "yes" if index in {0, 4} else "no",
                "factuality_low": "no",
                "toxicity_or_judgmental": "no",
                "response_length_bucket": "short",
                "lexical_trigger_family": "medication",
            },
            metadata={"response": f"Response {index}", "questionText": f"Question {index}"},
            cases={"questionID": f"q{index // 2}", "responder": "model"},
            case_key=f"q{index // 2}",
        )
        for index in range(6)
    ]
    dataset = Dataset.from_examples(examples)

    support = summarize_eval_label_support(dataset=dataset)
    cheap = summarize_eval_cheap_baselines(dataset=dataset)
    pairs = build_phase4_pairing_candidates(dataset=dataset)

    assert support["payload"]["summary"]["example_count"] == 6
    assert "medical_boundary_violation" in support["payload"]["summary"]["blocked_labels"]
    assert cheap["payload"]["summary"]["results"]["medical_boundary_violation"]["topic"]["balanced_accuracy"] is not None
    assert pairs["payload"]["summary"]["pair_count"] > 0
    assert "boundary_safe_vs_unsafe" in pairs["payload"]["summary"]["pair_type_counts"]
    assert "random_opposite_label_control" in pairs["payload"]["summary"]["pair_type_counts"]
    assert pairs["payload"]["summary"]["phase4_ready"] is False


def test_counselbench_eval_confound_inventory_tracks_responder_and_question_contrasts() -> None:
    examples = [
        Example(
            key=f"eval_{index}",
            prompt=[{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Response"}],
            labels={
                "questionID": f"q{index // 4}",
                "topic": "relationship_family",
                "responder": "human" if index % 2 == 0 else "gpt4",
                "split": "train",
                "empathy_high": "yes" if index in {0, 1, 4, 5} else "no",
                "specificity_high": "yes",
                "overall_quality_high": "yes" if index in {0, 1, 4, 5} else "no",
                "medical_boundary_violation": "no",
                "factuality_low": "no",
                "toxicity_or_judgmental": "no",
                "response_length_bucket": "short",
                "lexical_trigger_family": "none",
            },
            cases={"questionID": f"q{index // 4}", "responder": "human" if index % 2 == 0 else "gpt4"},
        )
        for index in range(8)
    ]

    result = summarize_eval_confound_inventory(dataset=Dataset.from_examples(examples))
    payload = result["payload"]

    assert payload["summary"]["responder_counts"] == {"gpt4": 4, "human": 4}
    assert payload["label_by_responder"]["empathy_high"]["all_responders_two_class"] is True
    assert payload["label_by_responder"]["empathy_high"]["responders"]["human"]["positive_rate"] == 0.5
    assert payload["contrast_by_question"]["empathy_high"]["contrast_question_count"] == 2
    assert payload["contrast_by_question"]["empathy_high"]["positive_negative_pair_count"] == 8
    assert payload["contrast_by_question"]["specificity_high"]["contrast_question_count"] == 0


def test_counselbench_geometry_metrics_summarize_label_and_confound_separation() -> None:
    geometry = _FakeArtifact(
        {
            "layers": [
                {
                    "layer": 8,
                    "components": [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.2, 5.0]],
                    "labels": ["safe", "safe", "unsafe", "unsafe"],
                    "color_by": {"topic": ["a", "a", "b", "b"]},
                    "explained_variance_ratio": [0.8, 0.1],
                }
            ]
        }
    )

    result = summarize_geometry_metrics(geometry=geometry)
    layer = result["payload"]["layers"][0]

    assert layer["metrics"]["label"]["class_counts"] == {"safe": 2, "unsafe": 2}
    assert layer["metrics"]["label"]["between_within_ratio"] is not None
    assert layer["explained_variance_by_key"]["label"] is not None
    assert result["payload"]["summary"]["metric_keys"] == ["label", "topic"]
    assert result["payload"]["summary"]["direction_similarity_keys"] == []


def test_counselbench_workflow_uses_deferred_hf_source_and_narrow_mounts() -> None:
    raw = build_raw_adv_source_dataset()
    built = workflow.build_workflow(raw)
    step_names = [step.name for step in built.steps]

    assert raw.is_deferred is True
    assert raw.to_dict()["source"]["path"] == "izi-ano/CounselBench-Adv"
    assert step_names == [
        "build_adv_prompt_dataset",
        "generate_adv_responses",
        "evaluate_generation_quality_gate",
        "build_successful_generation_capture_dataset",
        "summarize_generated_label_support",
        "capture_prompt_generated_residual",
        "text_baseline_prompt_failure_mode",
        "geometry_prompt_failure_mode_pca",
        "probe_prompt_failure_mode",
        "geometry_generated_posture_pca",
        "report",
    ]
    workflow_payload = built.to_dict()
    assert built.name == "counselbench_adv_phase03_full_adv_readouts"
    assert workflow_payload["steps"][0]["spec"]["inputs"]["limit_per_mode"] is None
    assert workflow_payload["steps"][1]["spec"]["generation"]["max_tokens"] == 15000
    assert workflow_payload["steps"][1]["spec"]["engine"]["max_model_len"] == 30000
    assert workflow_payload["steps"][1]["spec"]["engine"]["max_num_seqs"] == 16
    assert workflow_payload["steps"][1]["spec"]["engine"]["add_generation_prompt"] is True
    assert workflow_payload["steps"][5]["spec"]["engine"]["add_generation_prompt"] is False
    runner_payload = workflow.build_runner_specs()["capture_gpu"].to_dict()
    assert runner_payload["resources"]["gpu"] == "H200"
    assert runner_payload["resources"]["shard_count"] == 4
    assert runner_payload["resources"]["max_containers"] == 4
    assert '"."' not in str(workflow_payload)
    assert "projects/COUNSELBENCH" in str(workflow_payload)
    assert "geometry" in str(workflow_payload)
    assert set(workflow.build_runner_specs()) == {"capture_gpu", "analysis_cpu", "report_local"}


def test_counselbench_followup_workflows_encode_controls_eval_and_phase4_gates() -> None:
    adv_built = adv_03b_workflow.build_workflow(build_raw_adv_source_dataset())
    eval_built = eval_workflow.build_workflow(build_raw_eval_source_dataset())
    phase4_built = phase_04_workflow.build_workflow(build_raw_eval_source_dataset())

    adv_steps = [step.name for step in adv_built.steps]
    eval_steps = [step.name for step in eval_built.steps]
    phase4_steps = [step.name for step in phase4_built.steps]

    assert "nuisance_probe_topic" in adv_steps
    assert "baseline_failure_mode_from_length" in adv_steps
    assert "baseline_failure_mode_from_source_row" in adv_steps
    assert "nuisance_probe_prompt_length_bucket" in adv_steps
    assert "residualized_failure_mode_lexical_trigger" in adv_steps
    assert "triage_adv_03b_controls" in adv_steps
    assert "capture_eval_response_context_residual" in eval_steps
    assert "summarize_eval_cheap_baselines" in eval_steps
    assert "summarize_eval_confound_inventory" in eval_steps
    assert "run_eval_gated_readouts" in eval_steps
    assert "run_eval_responder_transfer_readouts" in eval_steps
    assert "probe_medical_boundary_response_end" not in eval_steps
    assert "triage_eval_phase03_readouts" not in eval_steps
    assert "summarize_geometry_eval_quality" in eval_steps
    assert phase4_steps == [
        "build_eval_aggregated_dataset",
        "summarize_eval_label_support",
        "build_phase4_pairing_candidates",
        "report",
    ]
    eval_payload = eval_built.to_dict()
    capture_step = next(step for step in eval_payload["steps"] if step["name"] == "capture_eval_response_context_residual")
    assert capture_step["spec"]["engine"]["add_generation_prompt"] is False
    assert "projects/COUNSELBENCH" in str(adv_built.to_dict())
    assert '"."' not in str(eval_payload)


class _FakeArtifact:
    id = "fake_generation"

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = dict(payload)

    def result(self) -> Mapping[str, Any]:
        return self._payload
