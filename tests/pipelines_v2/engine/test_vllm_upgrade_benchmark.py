from __future__ import annotations

import pytest

from scripts.pipelines_v2_vllm_upgrade_benchmark import (
    _benchmark_model_runner,
    build_dataset,
    build_runner_specs,
    build_workflow,
    summarize_generation,
)


def test_vllm_upgrade_benchmark_has_fixed_workload() -> None:
    dataset = build_dataset()
    workflow = build_workflow(dataset)

    assert len(dataset.examples) == 16
    assert [step.name for step in workflow.steps] == [
        "generation_throughput",
        "summarize_benchmark",
    ]
    generation = workflow.steps[0].spec
    assert generation.generation.max_tokens == 128
    assert generation.engine.max_num_seqs == 16


@pytest.mark.parametrize(
    ("model_runner", "expected_env"),
    (("v1", "0"), ("v2", "1")),
)
def test_vllm_upgrade_benchmark_selects_model_runner(
    monkeypatch: pytest.MonkeyPatch,
    model_runner: str,
    expected_env: str,
) -> None:
    monkeypatch.setenv("XENON_VLLM_BENCHMARK_MODEL_RUNNER", model_runner)

    runner = build_runner_specs()["capture_gpu"]

    assert _benchmark_model_runner() == model_runner
    assert runner.resources.env["VLLM_USE_V2_MODEL_RUNNER"] == expected_env


def test_vllm_upgrade_benchmark_rejects_unknown_model_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XENON_VLLM_BENCHMARK_MODEL_RUNNER", "v3")

    with pytest.raises(ValueError, match="either 'v1' or 'v2'"):
        _benchmark_model_runner()


def test_summarize_generation_emits_aggregate_metrics_and_digests() -> None:
    result = summarize_generation(
        generation={
            "rows": [
                {
                    "example_key": "a",
                    "generated_text": "one two",
                    "generated_token_ids": [1, 2],
                    "finish_reason": "stop",
                },
                {
                    "example_key": "b",
                    "generated_text": "three",
                    "generated_token_ids": [3],
                    "finish_reason": "length",
                },
            ],
            "metadata": {
                "performance": {
                    "generation_seconds": 0.5,
                    "generated_tokens_per_second": 6.0,
                }
            },
        }
    )

    summary = result["payload"]["summary"]
    assert summary["request_count"] == 2
    assert summary["nonempty_output_count"] == 2
    assert summary["generated_token_count"] == 3
    assert summary["generated_tokens_mean"] == 1.5
    assert summary["finish_reasons"] == {"length": 1, "stop": 1}
    assert summary["performance"]["generated_tokens_per_second"] == 6.0
    assert len(summary["output_digests"]["a"]) == 64
