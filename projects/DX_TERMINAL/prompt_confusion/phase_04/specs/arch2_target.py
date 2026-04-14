from __future__ import annotations

"""Runnable ARCH2 workflow shape for prompt_confusion phase_04."""

import json
from pathlib import Path
from typing import Any

from pipelines_v2.api import (
    BasisSpec,
    CaptureSpec,
    Dataset,
    DirectionSpec,
    Example,
    GenerationSpec,
    LabelMapSpec,
    LocalArtifactStore,
    LocalRunner,
    ModalResources,
    ModalRunner,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    MoERoutingSite,
    PairDeltaSpec,
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
    TransformBuilder,
    TransformResult,
    TransformSpec,
    ToyEngine,
    VLLMEngine,
    WorkflowOrchestrator,
    WorkflowSpec,
    WorkflowStep,
)


DB_ENV_VAR = "XENON_DATABASE_URL"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
PHASE_04_TABLE = "conflict_probe_examples_v2"
DEFAULT_REPORT_DIR = "projects/DX_TERMINAL/prompt_confusion/phase_04/reports"


def build_phase_04_prompt_metadata(rendered_prompt: str) -> dict[str, object]:
    headers = ("TASK", "STRATEGY", "SETTINGS", "PORTFOLIO", "MARKET")
    return {
        "token_sections": {
            name: span
            for name, span in _explicit_header_spans(rendered_prompt, headers=headers).items()
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


def build_phase_04_behavior_labels(
    *,
    generations: Any,
    workflow_expected: Any,
    strategy_expected_action: Any,
    strategy_expected_asset: Any,
    strategy_expected_size: Any,
    setting_expected_action: Any,
    setting_expected_asset: Any,
    setting_expected_size: Any,
) -> TransformResult:
    generated_by_key = _generated_decisions(generations)
    workflow_by_key = _resolve_expected_output_map(workflow_expected)
    strategy_action = strategy_expected_action.resolve_values()
    strategy_asset = strategy_expected_asset.resolve_values()
    strategy_size = strategy_expected_size.resolve_values()
    setting_action = setting_expected_action.resolve_values()
    setting_asset = setting_expected_asset.resolve_values()
    setting_size = setting_expected_size.resolve_values()

    labels = {
        "generated_action": {key: value.get("action") for key, value in generated_by_key.items()},
        "generated_asset": {key: value.get("asset") for key, value in generated_by_key.items()},
        "generated_size": {key: value.get("size") for key, value in generated_by_key.items()},
        "matches_workflow_expected": {
            key: _matches_expected(generated_by_key[key], workflow_by_key.get(key, {}))
            for key in generated_by_key
        },
        "matches_strategy_expected": {
            key: _matches_expected(
                generated_by_key[key],
                {
                    "action": strategy_action.get(key),
                    "asset": strategy_asset.get(key),
                    "size": strategy_size.get(key),
                },
            )
            for key in generated_by_key
        },
        "matches_setting_expected": {
            key: _matches_expected(
                generated_by_key[key],
                {
                    "action": setting_action.get(key),
                    "asset": setting_asset.get(key),
                    "size": setting_size.get(key),
                },
            )
            for key in generated_by_key
        },
    }
    labels["source_following_side"] = {
        key: _source_following_side(
            matches_strategy=bool(labels["matches_strategy_expected"][key]),
            matches_setting=bool(labels["matches_setting_expected"][key]),
        )
        for key in generated_by_key
    }
    return TransformResult(
        payload={
            "kind": "phase_04_behavior_labels",
            "generated_count": len(generated_by_key),
            "label_names": sorted(labels),
        },
        labels=labels,
        example_keys=sorted(generated_by_key),
    )


def _generated_decisions(generations: Any) -> dict[str, dict[str, str]]:
    by_key: dict[str, dict[str, str]] = {}
    for item in generations.generations():
        example_key = str(item["example_key"])
        structured = item.get("structured_output")
        if isinstance(structured, dict):
            raw = structured
        else:
            text = str(item.get("text", "") or "").strip()
            raw = json.loads(text) if text.startswith("{") and text.endswith("}") else {}
        by_key[example_key] = {
            "action": str(raw.get("action") or "").lower(),
            "asset": str(raw.get("asset") or "").upper(),
            "size": str(raw.get("size") or "").lower(),
        }
    return by_key


def _resolve_expected_output_map(source: Any) -> dict[str, dict[str, Any]]:
    raw = source.resolve_values()
    resolved: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        payload = json.loads(value) if isinstance(value, str) else value
        resolved[str(key)] = dict(payload) if isinstance(payload, dict) else {}
    return resolved


def _matches_expected(generated: dict[str, str], expected: dict[str, Any]) -> bool:
    return (
        generated.get("action", "").lower() == str(expected.get("action") or "").lower()
        and generated.get("asset", "").upper() == str(expected.get("asset") or "").upper()
        and generated.get("size", "").lower() == str(expected.get("size") or "").lower()
    )


def _source_following_side(*, matches_strategy: bool, matches_setting: bool) -> str:
    if matches_strategy and matches_setting:
        return "both"
    if matches_strategy:
        return "strategy"
    if matches_setting:
        return "setting"
    return "neither"


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
        enforce_eager=False,
        max_num_seqs=1,
    )


def _default_generation_engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=8192,
        enforce_eager=False,
        max_num_seqs=16,
    )


def build_phase_04_dataset() -> Dataset:
    db = PostgresSource.from_env(DB_ENV_VAR)
    return Dataset.from_postgres(
        source=db,
        table=PHASE_04_TABLE,
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        label_columns=[
            "conflict_present",
            "pair_member",
            "strategy_family",
            "strategy_variant_id",
            "setting_family",
            "setting_lexical_family_id",
            "environment_pressure_bucket",
            "context_variant_id",
            "lexical_split",
            "strategy_lexical_split",
            "setting_lexical_split",
            "expected_output_json",
            "strategy_expected_action",
            "strategy_expected_asset",
            "strategy_expected_size",
            "setting_expected_action",
            "setting_expected_asset",
            "setting_expected_size",
        ],
        case_columns=[
            "matched_pair_id",
            "strategy_variant_id",
            "setting_lexical_family_id",
            "context_variant_id",
        ],
        case_key_column="matched_pair_id",
        name="prompt_confusion_phase_04",
    )


def build_phase_04_runners() -> dict[str, object]:
    db = PostgresSource.from_env(DB_ENV_VAR)
    artifact_store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/prompt_confusion_phase_04",
    )
    db_secret = ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-db")

    capture_runner = ModalRunner(
        resources=ModalResources(
            gpu="A100-80GB",
            secrets=(db_secret,),
            volumes=(
                ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),
            ),
        ),
        artifacts=artifact_store,
        catalog=PostgresCatalog(source=db),
    )

    analysis_runner = ModalRunner(
        resources=ModalResources(
            cpu=6,
            memory_mb=24 * 1024,
            secrets=(db_secret,),
        ),
        artifacts=artifact_store,
        catalog=PostgresCatalog(source=db),
    )

    report_runner = LocalRunner()

    return {
        "capture_gpu": capture_runner,
        "analysis_cpu": analysis_runner,
        "report_local": report_runner,
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

    return WorkflowSpec(
        name="prompt_confusion_phase_04_prompt_state",
        steps=(
            WorkflowStep(
                name="capture_prompt_state_residual",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=residual_engine,
                    dataset=dataset,
                    prompt_metadata_builder=PromptMetadataBuilder.from_function(build_phase_04_prompt_metadata),
                    sites=[
                        ResidualSite(
                            name="resid_last_prompt_token",
                            site="resid_post",
                            layers=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
                            tokens=TokenSelector.last(),
                        ),
                        ResidualSite(
                            name="resid_strategy_section",
                            site="resid_post",
                            layers=[12, 24, 36, 44],
                            tokens=TokenSelector.section("STRATEGY"),
                        ),
                        ResidualSite(
                            name="resid_settings_section",
                            site="resid_post",
                            layers=[12, 24, 36, 44],
                            tokens=TokenSelector.section("SETTINGS"),
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
                    sites=[
                        MoERoutingSite(
                            name="router_last_prompt_token",
                            layers=[12, 24, 36, 44],
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
                name="probe_conflict",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("capture_prompt_state_residual").feature("resid_last_prompt_token"),
                    labels=dataset.labels("conflict_present"),
                    group_by=dataset.cases("matched_pair_id"),
                    split=dataset.labels("lexical_split"),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.mean(),
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
                name="probe_strategy_family",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("capture_prompt_state_residual").feature("resid_last_prompt_token"),
                    labels=dataset.labels("strategy_family"),
                    group_by=dataset.cases("strategy_variant_id"),
                    split=dataset.labels("lexical_split"),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.mean(),
                    metrics=["accuracy", "balanced_accuracy"],
                    baselines=["majority"],
                ),
            ),
            WorkflowStep(
                name="probe_setting_lexical_family",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("capture_prompt_state_residual").feature("resid_last_prompt_token"),
                    labels=dataset.labels("setting_lexical_family_id"),
                    group_by=dataset.cases("setting_lexical_family_id"),
                    split=dataset.labels("lexical_split"),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.mean(),
                    metrics=["accuracy", "balanced_accuracy"],
                    baselines=["majority"],
                ),
            ),
            WorkflowStep(
                name="probe_environment_pressure",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("capture_prompt_state_residual").feature("resid_last_prompt_token"),
                    labels=dataset.labels("environment_pressure_bucket"),
                    group_by=dataset.cases("context_variant_id"),
                    split=dataset.labels("lexical_split"),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.mean(),
                    metrics=["accuracy", "balanced_accuracy"],
                    baselines=["majority"],
                ),
            ),
            WorkflowStep(
                name="label_conflict_dimension",
                runner="analysis_cpu",
                spec=LabelMapSpec(
                    source=dataset.labels("strategy_family"),
                    output_name="conflict_dimension",
                    mapping={
                        "trade_size_force_large": "size",
                        "trade_size_force_small": "size",
                        "activity_force_trade": "action",
                        "activity_force_observe": "action",
                        "holding_force_exit": "action",
                        "diversification_force_concentrate": "asset",
                    },
                ),
            ),
            WorkflowStep(
                name="pair_delta",
                runner="analysis_cpu",
                spec=PairDeltaSpec(
                    feature=StepRef("capture_prompt_state_residual").feature("resid_last_prompt_token"),
                    case=dataset.cases("matched_pair_id"),
                    positive=dataset.labels("pair_member").equals("strong_conflict"),
                    negative=dataset.labels("pair_member").equals("aligned"),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.mean(),
                    labels={
                        "matched_pair_id": dataset.cases("matched_pair_id"),
                        "conflict_dimension": StepRef("label_conflict_dimension").label("conflict_dimension"),
                        "lexical_split": dataset.labels("lexical_split"),
                    },
                ),
            ),
            WorkflowStep(
                name="probe_pair_delta_conflict_dimension",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("pair_delta").feature("delta"),
                    labels=StepRef("pair_delta").label("conflict_dimension"),
                    split=StepRef("pair_delta").label("lexical_split"),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.mean(),
                    metrics=[
                        "accuracy",
                        "balanced_accuracy",
                        "auroc",
                        "selectivity",
                    ],
                    baselines=["majority", "shuffled_label"],
                ),
            ),
            WorkflowStep(
                name="basis_pair_delta",
                runner="analysis_cpu",
                spec=BasisSpec(
                    feature=StepRef("pair_delta").feature("delta"),
                    method="pca",
                    by=StepRef("pair_delta").label("conflict_dimension"),
                    layers=[12, 24, 36, 44],
                    components=4,
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.mean(),
                ),
            ),
            WorkflowStep(
                name="direction_pair_delta_size_vs_action",
                runner="analysis_cpu",
                spec=DirectionSpec(
                    feature=StepRef("pair_delta").feature("delta"),
                    positive=StepRef("pair_delta").label("conflict_dimension").equals("size"),
                    negative=StepRef("pair_delta").label("conflict_dimension").equals("action"),
                    layers=[12, 24, 36, 44],
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.mean(),
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    template="prompt_confusion_phase_04_summary",
                    output_dir=report_output_dir,
                    inputs=[
                        StepRef("capture_prompt_state_residual"),
                        StepRef("capture_prompt_state_router"),
                        StepRef("probe_conflict"),
                        StepRef("probe_strategy_family"),
                        StepRef("probe_setting_lexical_family"),
                        StepRef("probe_environment_pressure"),
                        StepRef("label_conflict_dimension"),
                        StepRef("pair_delta"),
                        StepRef("probe_pair_delta_conflict_dimension"),
                        StepRef("basis_pair_delta"),
                        StepRef("direction_pair_delta_size_vs_action"),
                    ],
                ),
            ),
        ),
    )


def build_phase_04_behavioral_followon_workflow(
    dataset: Dataset | None = None,
    *,
    generation_engine: object | None = None,
    report_output_dir: str = DEFAULT_REPORT_DIR,
) -> WorkflowSpec:
    dataset = dataset or build_phase_04_dataset()
    generation_engine = generation_engine or _default_generation_engine()

    return WorkflowSpec(
        name="prompt_confusion_phase_04_behavioral_followon",
        steps=(
            WorkflowStep(
                name="capture_with_generation",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=generation_engine,
                    dataset=dataset,
                    sites=[
                        ResidualSite(
                            name="resid_prompt_boundary",
                            site="resid_post",
                            layers=[12, 24, 36, 44],
                            tokens=TokenSelector.last(),
                        ),
                    ],
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=128,
                        temperature=0.0,
                        structured_output={
                            "type": "json_schema",
                            "name": "phase_04_decision",
                        },
                    ),
                ),
            ),
            WorkflowStep(
                name="label_behavior",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(build_phase_04_behavior_labels),
                    inputs={
                        "generations": StepRef("capture_with_generation"),
                        "workflow_expected": dataset.labels("expected_output_json"),
                        "strategy_expected_action": dataset.labels("strategy_expected_action"),
                        "strategy_expected_asset": dataset.labels("strategy_expected_asset"),
                        "strategy_expected_size": dataset.labels("strategy_expected_size"),
                        "setting_expected_action": dataset.labels("setting_expected_action"),
                        "setting_expected_asset": dataset.labels("setting_expected_asset"),
                        "setting_expected_size": dataset.labels("setting_expected_size"),
                    },
                ),
            ),
            WorkflowStep(
                name="probe_behavior_source_following",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("capture_with_generation").feature("resid_prompt_boundary"),
                    labels=StepRef("label_behavior").label("source_following_side"),
                    group_by=dataset.cases("matched_pair_id"),
                    split=dataset.labels("lexical_split"),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.mean(),
                    metrics=["accuracy", "balanced_accuracy", "auroc", "selectivity"],
                    baselines=["majority", "shuffled_label"],
                ),
            ),
            WorkflowStep(
                name="behavior_report",
                runner="report_local",
                spec=ReportSpec(
                    template="prompt_confusion_phase_04_behavioral_followon",
                    output_dir=report_output_dir,
                    inputs=[
                        StepRef("capture_with_generation"),
                        StepRef("label_behavior"),
                        StepRef("probe_behavior_source_following"),
                    ],
                ),
            ),
        ),
    )


def build_phase_04_local_smoke_dataset() -> Dataset:
    examples: list[Example] = []
    token_sections = {"STRATEGY": [2, 3], "SETTINGS": [4, 5]}
    pair_specs = [
        {
            "split": "train",
            "family": "trade_size_force_large",
            "setting_family": "trade_size",
            "pressure": "balanced",
            "strategy_variant_id": "size_large_v0",
            "setting_lexical_family_id": "size_setting_phrase_v0",
            "context_variant_id": "ctx_size_train",
            "strategy_expected": {"action": "buy", "asset": "ALPHA", "size": "large"},
            "setting_expected": {"action": "buy", "asset": "ALPHA", "size": "small"},
        },
        {
            "split": "train",
            "family": "activity_force_trade",
            "setting_family": "trading_activity",
            "pressure": "strategy_favored",
            "strategy_variant_id": "activity_trade_v0",
            "setting_lexical_family_id": "activity_setting_phrase_v0",
            "context_variant_id": "ctx_action_train",
            "strategy_expected": {"action": "buy", "asset": "ALPHA", "size": "medium"},
            "setting_expected": {"action": "observe", "asset": "NONE", "size": "none"},
        },
        {
            "split": "train",
            "family": "diversification_force_concentrate",
            "setting_family": "diversification",
            "pressure": "setting_favored",
            "strategy_variant_id": "div_concentrate_v0",
            "setting_lexical_family_id": "div_setting_phrase_v0",
            "context_variant_id": "ctx_asset_train",
            "strategy_expected": {"action": "buy", "asset": "ALPHA", "size": "medium"},
            "setting_expected": {"action": "buy", "asset": "BETA", "size": "medium"},
        },
        {
            "split": "test",
            "family": "trade_size_force_large",
            "setting_family": "trade_size",
            "pressure": "balanced",
            "strategy_variant_id": "size_large_v1",
            "setting_lexical_family_id": "size_setting_phrase_v1",
            "context_variant_id": "ctx_size_test",
            "strategy_expected": {"action": "buy", "asset": "ALPHA", "size": "large"},
            "setting_expected": {"action": "buy", "asset": "ALPHA", "size": "small"},
        },
        {
            "split": "test",
            "family": "activity_force_trade",
            "setting_family": "trading_activity",
            "pressure": "strategy_favored",
            "strategy_variant_id": "activity_trade_v1",
            "setting_lexical_family_id": "activity_setting_phrase_v1",
            "context_variant_id": "ctx_action_test",
            "strategy_expected": {"action": "buy", "asset": "ALPHA", "size": "medium"},
            "setting_expected": {"action": "observe", "asset": "NONE", "size": "none"},
        },
        {
            "split": "test",
            "family": "diversification_force_concentrate",
            "setting_family": "diversification",
            "pressure": "setting_favored",
            "strategy_variant_id": "div_concentrate_v1",
            "setting_lexical_family_id": "div_setting_phrase_v1",
            "context_variant_id": "ctx_asset_test",
            "strategy_expected": {"action": "buy", "asset": "ALPHA", "size": "medium"},
            "setting_expected": {"action": "buy", "asset": "BETA", "size": "medium"},
        },
    ]

    for index, spec in enumerate(pair_specs, start=1):
        matched_pair_id = f"phase04_pair_{index}"
        for pair_member, conflict_present, expected_output in (
            ("aligned", False, spec["strategy_expected"]),
            ("strong_conflict", True, spec["setting_expected"]),
        ):
            example_key = f"{matched_pair_id}_{pair_member}"
            prompt = (
                "TASK\nChoose exactly one action.\n\n"
                f"STRATEGY\n{spec['family']} strategy text.\n\n"
                f"SETTINGS\n{pair_member} settings text.\n\n"
                "PORTFOLIO\nSimple portfolio state.\n\n"
                "MARKET\nALPHA leads BETA.\n"
            )
            examples.append(
                Example(
                    key=example_key,
                    prompt=prompt,
                    labels={
                        "conflict_present": conflict_present,
                        "pair_member": pair_member,
                        "strategy_family": spec["family"],
                        "strategy_variant_id": spec["strategy_variant_id"],
                        "setting_family": spec["setting_family"],
                        "setting_lexical_family_id": spec["setting_lexical_family_id"],
                        "environment_pressure_bucket": spec["pressure"],
                        "context_variant_id": spec["context_variant_id"],
                        "lexical_split": spec["split"],
                        "strategy_lexical_split": spec["split"],
                        "setting_lexical_split": spec["split"],
                        "expected_output_json": dict(expected_output),
                        "strategy_expected_action": spec["strategy_expected"]["action"],
                        "strategy_expected_asset": spec["strategy_expected"]["asset"],
                        "strategy_expected_size": spec["strategy_expected"]["size"],
                        "setting_expected_action": spec["setting_expected"]["action"],
                        "setting_expected_asset": spec["setting_expected"]["asset"],
                        "setting_expected_size": spec["setting_expected"]["size"],
                    },
                    metadata={"token_sections": token_sections},
                    cases={
                        "matched_pair_id": matched_pair_id,
                        "strategy_variant_id": spec["strategy_variant_id"],
                        "setting_lexical_family_id": spec["setting_lexical_family_id"],
                        "context_variant_id": spec["context_variant_id"],
                    },
                    case_key=matched_pair_id,
                )
            )

    return Dataset.from_examples(examples, name="prompt_confusion_phase_04_local_smoke")


def build_phase_04_local_smoke_runners(root: str | Path) -> dict[str, object]:
    root = Path(root)
    shared_store = LocalArtifactStore(root / "artifacts")
    return {
        "capture_gpu": LocalRunner(artifacts=shared_store),
        "analysis_cpu": LocalRunner(artifacts=shared_store),
        "report_local": LocalRunner(artifacts=shared_store),
    }


def build_phase_04_local_smoke_orchestrator(root: str | Path) -> WorkflowOrchestrator:
    return WorkflowOrchestrator(runners=build_phase_04_local_smoke_runners(root))


def build_phase_04_orchestrator() -> WorkflowOrchestrator:
    return WorkflowOrchestrator(runners=build_phase_04_runners())


def load_phase_04_target_json(
    path: str | Path = Path(__file__).with_name("arch2_target.json"),
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    dataset = _dataset_from_target_config(dict(payload["dataset"]))
    dataset_payload = dataset.to_dict()
    runners = {
        str(name): _runner_from_target_config(dict(config))
        for name, config in dict(payload["runners"]).items()
    }
    workflows = {
        str(name): WorkflowSpec.from_dict(_replace_dataset_refs(dict(config), dataset_payload))
        for name, config in dict(payload["workflows"]).items()
    }
    return {
        "dataset": dataset,
        "runners": runners,
        "workflows": workflows,
        "orchestrator": WorkflowOrchestrator(runners=runners),
    }


def _dataset_from_target_config(config: dict[str, Any]) -> Dataset:
    if str(config.get("kind")) != "dataset":
        raise ValueError(f"Unsupported target dataset config kind: {config.get('kind')!r}")
    if str(config.get("mode")) != "postgres":
        raise ValueError(f"Unsupported target dataset mode: {config.get('mode')!r}")
    return Dataset.from_postgres(
        source=PostgresSource.from_dict(dict(config["source"])),
        table=str(config["table"]) if config.get("table") is not None else None,
        sql=str(config["sql"]) if config.get("sql") is not None else None,
        prompt_column=str(config["prompt_column"]),
        example_key_column=str(config["example_key_column"]),
        label_columns=tuple(config.get("label_columns", ())),
        case_columns=tuple(config.get("case_columns", ())),
        case_key_column=config.get("case_key_column"),
        metadata_columns=tuple(config.get("metadata_columns", ())),
        name=config.get("name"),
    )


def _runner_from_target_config(config: dict[str, Any]) -> object:
    kind = str(config.get("kind") or "")
    if kind == "local":
        return LocalRunner()
    if kind != "modal":
        raise ValueError(f"Unsupported target runner kind: {kind!r}")

    resources_payload = dict(config.get("resources", {}))
    artifacts_payload = dict(config["artifacts"])
    catalog_payload = dict(config["catalog"])
    return ModalRunner(
        resources=ModalResources(
            gpu=resources_payload.get("gpu"),
            cpu=resources_payload.get("cpu"),
            memory_mb=resources_payload.get("memory_mb"),
            secrets=tuple(
                ModalSecret(
                    name=str(secret["name"]),
                    env_vars=tuple(str(env_var) for env_var in secret.get("env_vars", ())),
                )
                for secret in resources_payload.get("secrets", ())
            ),
            volumes=tuple(
                ModalVolumeMount(
                    name=str(volume["name"]),
                    mount_path=str(volume["mount_path"]),
                    create_if_missing=bool(volume.get("create_if_missing", False)),
                    commit_on_success=bool(volume.get("commit_on_success", False)),
                )
                for volume in resources_payload.get("volumes", ())
            ),
        ),
        artifacts=ModalVolumeStore(
            name=str(artifacts_payload["name"]),
            root=str(artifacts_payload["root"]),
        ),
        catalog=PostgresCatalog(source=PostgresSource.from_dict(dict(catalog_payload["source"]))),
    )


def _replace_dataset_refs(value: Any, dataset_payload: dict[str, Any]) -> Any:
    if isinstance(value, list):
        return [_replace_dataset_refs(item, dataset_payload) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("kind") == "dataset_ref":
        if value.get("ref") != "phase_dataset":
            raise ValueError(f"Unknown dataset ref: {value.get('ref')!r}")
        return dataset_payload
    return {str(key): _replace_dataset_refs(item, dataset_payload) for key, item in value.items()}
