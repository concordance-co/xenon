from dataclasses import replace

import pytest

from pipelines_v2.api import (
    CaptureSpec, Dataset, Example, GenerationSpec, ModalResources, ModalRunner,
    ModalVolumeMount, ModalVolumeStore, VLLMEngine,
)
from pipelines_v2.core.types import SpecValidationError


def generation(engine):
    return CaptureSpec(
        engine=engine, dataset=Dataset.from_examples([Example(key="one", prompt="hi")]),
        generation=GenerationSpec(enabled=True, max_tokens=1),
    )


@pytest.mark.parametrize("settings,match", [
    ({"tensor_parallel_size": 0}, "tensor_parallel_size"),
    ({"max_model_len": -1}, "max_model_len"),
    ({"max_num_seqs": "8"}, "max_num_seqs"),
    ({"gpu_memory_utilization": 1.5}, "gpu_memory_utilization"),
    ({"gpu_memory_utilization": float("nan")}, "gpu_memory_utilization"),
    ({"enable_prefix_caching": "false"}, "enable_prefix_caching"),
    ({"extra": {"llm_kwargs": {"dtype": "bfloat16"}}}, "Unsupported"),
    ({"extra": {"max_num_seq": 8}}, "max_num_seq"),
])
def test_invalid_engine_settings_fail_before_modal(monkeypatch, settings, match):
    def launch(**kwargs):
        pytest.fail("Invalid settings reached Modal")

    monkeypatch.setattr("pipelines_v2.runtime.modal.run_on_modal", launch)
    runner = ModalRunner(ModalResources(gpu="A100-80GB"), ModalVolumeStore("data", "/data/artifacts"))
    spec = generation(VLLMEngine(model_id="test/model", **settings))
    assert not runner.plan(spec).valid
    with pytest.raises(SpecValidationError, match=match):
        runner.run(spec)


@pytest.mark.parametrize("resources,match", [
    (ModalResources(), "requires ModalResources.gpu"),
    (ModalResources(gpu="A100-80GB", shard_count=8), "requires 2 GPUs"),
    (ModalResources(gpu="A100-80GB:0"), "GPU type"),
    (ModalResources(gpu="A100-80GB:2", timeout_seconds=0), "timeout_seconds"),
    (ModalResources(gpu="A100-80GB:2", cpu=-1), "cpu"),
    (ModalResources(gpu="A100-80GB:2", volumes=(ModalVolumeMount("models", "models"),)), "mount path"),
    (ModalResources(gpu="A100-80GB:2", volumes=(ModalVolumeMount("models", "/data/models"),)), "Overlapping"),
    (ModalResources(gpu="A100-80GB:2", volumes=(ModalVolumeMount("models", "/data/"),)), "Duplicate"),
])
def test_invalid_modal_resources_fail_planning(resources, match):
    runner = ModalRunner(resources, ModalVolumeStore("data", "/data/artifacts"))
    spec = generation(VLLMEngine(model_id="test/model", tensor_parallel_size=2))
    with pytest.raises(SpecValidationError, match=match):
        runner.plan(spec).validate()


def test_model_mount_and_topology_are_checked():
    spec = generation(VLLMEngine(model_id="/models/test", tensor_parallel_size=2))
    resources = ModalResources(gpu="A100-80GB:2")
    runner = ModalRunner(resources, ModalVolumeStore("data", "/data/artifacts"))
    with pytest.raises(SpecValidationError, match="not covered"):
        runner.plan(spec).validate()
    runner.resources = replace(resources, volumes=(ModalVolumeMount("models", "/models"),))
    assert runner.plan(spec).valid


def test_cli_plan_returns_failure_for_invalid_resources(monkeypatch, capsys):
    import argparse
    import json
    from pipelines_v2.cli import _workflow_plan
    from pipelines_v2.api import WorkflowSpec, WorkflowStep

    spec = generation(VLLMEngine(model_id="test/model"))
    workflow = WorkflowSpec(name="invalid", steps=(WorkflowStep(name="generate", runner="gpu", spec=spec),))
    runner = ModalRunner(ModalResources(), ModalVolumeStore("data", "/data/artifacts"))
    monkeypatch.setattr("pipelines_v2.cli.load_python_workflow_file", lambda **kwargs: (spec.dataset, workflow, {}))
    monkeypatch.setattr("pipelines_v2.cli._build_runners", lambda *args: {"gpu": runner})
    assert _workflow_plan(argparse.Namespace(file="unused", dataset_fn="unused", workflow_fn="unused")) == 1
    assert json.loads(capsys.readouterr().out)["steps"][0]["errors"]
