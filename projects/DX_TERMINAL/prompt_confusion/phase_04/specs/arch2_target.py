from __future__ import annotations

"""Runnable ARCH2 target for prompt_confusion phase_04.

This target is intentionally narrow. It mirrors the current Phase 04 direction:

- deferred Postgres dataset backed by ``conflict_probe_examples_v3``
- prompt-only capture first
- residual and MoE router capture split across separate GPU steps
- one first-pass prompt-state probe on the cheaper analysis runtime
- one local report step
"""

import json
from pathlib import Path
from typing import Any

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    MoERoutingSite,
    PostgresCatalog,
    PostgresSource,
    PromptMetadataBuilder,
    ProbeSpec,
    ReportSpec,
    ResidualSite,
    RoutingRecord,
    StepRef,
    TokenPooling,
    TokenSelector,
    VLLMEngine,
    WorkflowOrchestrator,
    WorkflowSpec,
    WorkflowStep,
)
from pipelines_v2.runtime.specs import runner_spec_from_dict


DB_ENV_VAR = "XENON_DATABASE_URL"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
PHASE_04_SQL = """
WITH ranked AS (
    SELECT row_number() OVER (ORDER BY example_id) AS log_id, src.*
    FROM conflict_probe_examples_v3 src
)
SELECT * FROM ranked
"""
DEFAULT_REPORT_DIR = "projects/DX_TERMINAL/prompt_confusion/phase_04/reports"
PROMPT_STATE_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)
PROMPT_SECTION_HEADERS = ("TASK", "STRATEGY", "SETTINGS", "PORTFOLIO", "MARKET")


def build_phase_04_prompt_metadata(rendered_prompt: str) -> dict[str, object]:
    return {
        "token_sections": {
            name: span
            for name, span in _explicit_header_spans(
                rendered_prompt,
                headers=PROMPT_SECTION_HEADERS,
            ).items()
        }
    }


def _explicit_header_spans(
    rendered_prompt: str,
    *,
    headers: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    spans: dict[str, dict[str, int]] = {}
    for index, header in enumerate(headers):
        marker = f"{header}\n"
        start = rendered_prompt.index(marker) + len(marker)
        if index + 1 < len(headers):
            next_header = headers[index + 1]
            end = rendered_prompt.index(f"\n\n{next_header}\n", start)
        else:
            end = len(rendered_prompt)
        spans[header] = {"char_start": start, "char_end": end}
    return spans


def _default_residual_engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=8192,
        enforce_eager=False,
        max_num_seqs=16,
    )


def _default_router_engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=8192,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
    )


def build_phase_04_dataset() -> Dataset:
    db = PostgresSource.from_env(DB_ENV_VAR)
    return Dataset.from_postgres(
        source=db,
        sql=PHASE_04_SQL,
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        label_columns=[
            "conflict_present",
            "pair_member",
            "strategy_family",
            "strategy_variant_id",
            "setting_family",
            "setting_lexical_family_id",
            "setting_variant_id",
            "setting_value",
            "setting_bucket",
            "conflict_strength",
            "environment_pressure_bucket",
            "context_family",
            "context_variant_id",
            "portfolio_state_family",
            "portfolio_variant_id",
            "lexical_split",
            "strategy_lexical_split",
            "setting_lexical_split",
            "market_expected_action",
            "market_expected_asset",
            "strategy_expected_action",
            "strategy_expected_asset",
            "strategy_expected_size",
            "setting_expected_action",
            "setting_expected_asset",
            "setting_expected_size",
            "expected_output_json",
        ],
        case_columns=[
            "matched_pair_id",
            "strategy_variant_id",
            "setting_lexical_family_id",
            "setting_variant_id",
            "context_variant_id",
        ],
        case_key_column="matched_pair_id",
        name="prompt_confusion_phase_04",
    )


def build_phase_04_runner_specs() -> dict[str, object]:
    db = PostgresSource.from_env(DB_ENV_VAR)
    artifact_store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/prompt_confusion_phase_04",
    )
    db_secret = ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-db")

    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="A100-80GB",
                secrets=(db_secret,),
                volumes=(
                    ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),
                ),
            ),
            artifacts=artifact_store,
            catalog=PostgresCatalog(source=db),
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=6,
                memory_mb=24 * 1024,
                secrets=(db_secret,),
            ),
            artifacts=artifact_store,
            catalog=PostgresCatalog(source=db),
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(Path("artifacts") / "prompt_confusion_phase_04")
        ),
    }


def build_phase_04_runners() -> dict[str, object]:
    return {
        name: spec.to_runner()
        for name, spec in build_phase_04_runner_specs().items()
    }


def build_phase_04_prompt_state_workflow(
    dataset: Dataset | None = None,
    *,
    residual_engine: object | None = None,
    router_engine: object | None = None,
    report_output_dir: str = DEFAULT_REPORT_DIR,
) -> WorkflowSpec:
    dataset = dataset or build_phase_04_dataset()
    residual_engine = residual_engine or _default_residual_engine()
    router_engine = router_engine or _default_router_engine()
    metadata_builder = PromptMetadataBuilder.from_function(build_phase_04_prompt_metadata)

    return WorkflowSpec(
        name="prompt_confusion_phase_04_prompt_state",
        steps=(
            WorkflowStep(
                name="capture_prompt_state_residual",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=residual_engine,
                    dataset=dataset,
                    prompt_metadata_builder=metadata_builder,
                    sites=[
                        ResidualSite(
                            name="resid_last_prompt_token",
                            site="resid_post",
                            layers=list(PROMPT_STATE_LAYERS),
                            tokens=TokenSelector.last(),
                        ),
                    ],
                    generation=GenerationSpec(enabled=False),
                ),
            ),
            WorkflowStep(
                name="capture_prompt_state_router",
                runner="capture_gpu",
                depends_on=("capture_prompt_state_residual",),
                spec=CaptureSpec(
                    engine=router_engine,
                    dataset=dataset,
                    prompt_metadata_builder=metadata_builder,
                    sites=[
                        MoERoutingSite(
                            name="router_last_prompt_token",
                            layers=list(PROMPT_STATE_LAYERS),
                            tokens=TokenSelector.last(),
                            record=[
                                RoutingRecord.gate_logits(dtype="float16"),
                                RoutingRecord.topk_from_gate(k=8, include_weights=True),
                                RoutingRecord.expert_load(source="topk_from_gate"),
                            ],
                        ),
                    ],
                    generation=GenerationSpec(enabled=False),
                ),
            ),
            WorkflowStep(
                name="probe_conflict_present",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("capture_prompt_state_residual").feature("resid_last_prompt_token"),
                    labels=dataset.labels("conflict_present"),
                    group_by=dataset.cases("matched_pair_id"),
                    split=dataset.labels("lexical_split"),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                    metrics=[
                        "accuracy",
                        "balanced_accuracy",
                        "auroc",
                        "selectivity",
                    ],
                    baselines=[
                        "majority",
                        "shuffled_label",
                    ],
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    template="prompt_confusion_phase_04_prompt_state",
                    output_dir=report_output_dir,
                    inputs=[
                        StepRef("capture_prompt_state_residual"),
                        StepRef("capture_prompt_state_router"),
                        StepRef("probe_conflict_present"),
                    ],
                ),
            ),
        ),
    )


def build_dataset() -> Dataset:
    return build_phase_04_dataset()


def build_runner_specs() -> dict[str, object]:
    return build_phase_04_runner_specs()


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    return build_phase_04_prompt_state_workflow(dataset)


def build_phase_04_orchestrator() -> WorkflowOrchestrator:
    return WorkflowOrchestrator(runners=build_phase_04_runners())


def build_phase_04_target_payload() -> dict[str, Any]:
    dataset = build_phase_04_dataset()
    workflow = build_phase_04_prompt_state_workflow(dataset)
    runner_specs = build_phase_04_runner_specs()
    return {
        "kind": "arch2_target",
        "schema_version": 1,
        "project_id": "DX_TERMINAL",
        "subproject": "prompt_confusion",
        "phase": "phase_04",
        "goal": (
            "Use phase_04 as the first end-to-end proof that the new library can run "
            "a remote dataset-backed GPU prompt-state capture workflow, fan out the "
            "first artifact-bound analysis step on a cheaper CPU runtime, and produce "
            "a local report for the prompt-only conflict benchmark."
        ),
        "dataset": dataset.to_dict(),
        "runners": {
            name: spec.to_dict()
            for name, spec in runner_specs.items()
        },
        "workflows": {
            "prompt_state": workflow.to_dict(),
        },
    }


def load_phase_04_target_json(
    path: str | Path = Path(__file__).with_name("arch2_target.json"),
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    dataset = Dataset.from_dict(dict(payload["dataset"]))
    runner_specs = {
        str(name): runner_spec_from_dict(dict(config))
        for name, config in dict(payload["runners"]).items()
    }
    runners = {
        name: spec.to_runner()
        for name, spec in runner_specs.items()
    }
    workflows = {
        str(name): WorkflowSpec.from_dict(dict(config))
        for name, config in dict(payload["workflows"]).items()
    }
    return {
        "dataset": dataset,
        "runner_specs": runner_specs,
        "runners": runners,
        "workflows": workflows,
        "orchestrator": WorkflowOrchestrator(runners=runners),
    }


def write_phase_04_target_json(
    path: str | Path = Path(__file__).with_name("arch2_target.json"),
) -> dict[str, Any]:
    payload = build_phase_04_target_payload()
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
