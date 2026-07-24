from __future__ import annotations

from scripts.pipelines_v2_vllm_upgrade_benchmark import (
    build_dataset,
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
