from __future__ import annotations

"""pipelines_v2 synthetic-market patch smoke.

This keeps the workflow close to the old synthetic-market path-validation
shape while exercising the v2 patched-generation surface against a real remote
dataset and structured-output generation.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from pipelines_v2.api import (
    AddDirectionPatch,
    Dataset,
    GenerationRunSpec,
    GenerationSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PatchComparisonSpec,
    PatchedGenerationSpec,
    PostgresCatalog,
    PostgresSource,
    ProjectOutPatch,
    PromptMetadataBuilder,
    ResidualInterventionSite,
    StepRef,
    TokenSelector,
    TransformBuilder,
    TransformSpec,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
ARTIFACT_ROOT = "/data/artifacts/dx_terminal_synthetic_market/path_validation_v2_smoke"
PHASE_NAME = "phase15_market_basis_discovery_v1"
CONTEXT_VARIANT = "market_only"
DEFAULT_BASIS_STATE_KEY = "market_mean"
BASIS_NPZ_PATH = Path(
    "data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/pca_basis.npz"
)
BASIS_RESULTS_PATH = Path(
    "data/analysis_results/synthetic_market_axis_decomposition/phase17_market_axis_decomposition_v1/results.json"
)
DEFAULT_LIMIT = 16
PATCH_LAYER = 4
PATCH_COMPONENTS = 4
DEFAULT_GENERATION_MAX_TOKENS = 2500
DEFAULT_CAPTURE_GPU = "A100-80GB"
DEFAULT_PATCH_OPERATOR = "project_out"
DEFAULT_DIRECTION_TARGET = "leader_axis"

DATASET_SQL = f"""
WITH ranked AS (
    SELECT
        log_id,
        phase_name,
        example_id,
        family,
        family_variant,
        context_variant,
        prompt_messages_json,
        row_number() OVER (ORDER BY selection_rank ASC NULLS LAST, log_id) AS smoke_rank
    FROM synthetic_market_examples_v0
    WHERE phase_name = '{PHASE_NAME}'
      AND context_variant = '{CONTEXT_VARIANT}'
)
SELECT *
FROM ranked
ORDER BY smoke_rank
"""

_SECTION_HEADERS: tuple[tuple[str, str], ...] = (
    ("market", "## MARKET SNAPSHOT"),
    ("active_strategies", "## ACTIVE STRATEGIES"),
    ("active_settings", "## ACTIVE SETTINGS"),
    ("portfolio", "## PORTFOLIO CONTEXT"),
    ("constraints", "## CONSTRAINTS"),
    ("price_impact_limits", "## PRICE IMPACT LIMITS"),
    ("instruction", "Respond with the single best action for this tick:"),
)
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _trim_section_end_char(
    rendered_text: str,
    *,
    section_start_char: int,
    section_end_char: int,
) -> int:
    if section_end_char <= section_start_char:
        return int(section_end_char)
    section_text = rendered_text[int(section_start_char) : int(section_end_char)]
    section_text = re.sub(r"\s+\Z", "", section_text)
    section_text = re.sub(r"(?:\n-+[ \t]*)+\Z", "", section_text)
    section_text = re.sub(r"\s+\Z", "", section_text)
    trimmed_end = int(section_start_char) + len(section_text)
    return trimmed_end if trimmed_end > int(section_start_char) else int(section_end_char)


def _function_call_schema(*, name: str, arguments_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": [str(name)]},
            "arguments": dict(arguments_schema),
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }


def _trading_decision_chat_tools() -> tuple[dict[str, Any], ...]:
    shared_parameters = {
        "content": {
            "type": "string",
            "description": "Short description of your reasoning for this trade",
        },
        "strategy": {
            "type": "string",
            "description": (
                "Optional. If this action follows an active strategy from the "
                'ACTIVE STRATEGIES section, provide its label (e.g. "strategy1"). '
                "Omit if no active strategies exist or if this action is not strategy-driven."
            ),
        },
    }
    return (
        {
            "type": "function",
            "function": {
                "name": "buy_token",
                "parameters": {
                    "type": "object",
                    "required": ["token", "spend_pct"],
                    "properties": {
                        "token": {
                            "type": "string",
                            "description": "Counterparty token symbol or address.",
                        },
                        **shared_parameters,
                        "spend_pct": {
                            "type": "number",
                            "description": "Percent (0-100] of the source balance to allocate to this trade.",
                        },
                    },
                },
                "description": "Buy a token using ETH with a percentage of the available balance.",
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sell_token",
                "parameters": {
                    "type": "object",
                    "required": ["token", "spend_pct"],
                    "properties": {
                        "token": {
                            "type": "string",
                            "description": "Counterparty token symbol or address.",
                        },
                        **shared_parameters,
                        "spend_pct": {
                            "type": "number",
                            "description": "Percent (0-100] of the source balance to allocate to this trade.",
                        },
                    },
                },
                "description": "Sell a token back into ETH using a percentage of the token balance.",
            },
        },
        {
            "type": "function",
            "function": {
                "name": "record_observation",
                "parameters": {
                    "type": "object",
                    "required": ["content"],
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Body of the message to save",
                        },
                        "strategy": shared_parameters["strategy"],
                    },
                },
                "description": "Save a short note about your current market observations to aid you in future trades.",
            },
        },
    )


def _trading_decision_structured_output() -> dict[str, Any]:
    shared_properties = {
        "content": {
            "type": "string",
            "description": "Short description of your reasoning for this trade",
        },
        "strategy": {
            "type": "string",
            "description": (
                "Optional. If this action follows an active strategy from the "
                'ACTIVE STRATEGIES section, provide its label (e.g. "strategy1"). '
                "Omit if no active strategies exist or if this action is not strategy-driven."
            ),
        },
    }
    trade_arguments = {
        "type": "object",
        "required": ["token", "spend_pct"],
        "properties": {
            "token": {
                "type": "string",
                "description": "Counterparty token symbol or address.",
            },
            **shared_properties,
            "spend_pct": {
                "type": "number",
                "description": "Percent (0-100] of the source balance to allocate to this trade.",
            },
        },
    }
    observation_arguments = {
        "type": "object",
        "required": ["content"],
        "properties": {
            "content": {
                "type": "string",
                "description": "Body of the message to save",
            },
            "strategy": shared_properties["strategy"],
        },
    }
    return {
        "type": "object",
        "anyOf": [
            _function_call_schema(name="buy_token", arguments_schema=trade_arguments),
            _function_call_schema(name="sell_token", arguments_schema=trade_arguments),
            _function_call_schema(name="record_observation", arguments_schema=observation_arguments),
        ],
    }


def _dataset_limit() -> int:
    raw = os.environ.get("SYNTHETIC_MARKET_V2_SMOKE_LIMIT")
    if raw is None or not raw.strip():
        return DEFAULT_LIMIT
    value = int(raw)
    if value <= 0:
        raise ValueError("SYNTHETIC_MARKET_V2_SMOKE_LIMIT must be a positive integer")
    return value


def _generation_max_tokens() -> int:
    raw = os.environ.get("SYNTHETIC_MARKET_V2_SMOKE_MAX_TOKENS")
    if raw is None or not raw.strip():
        return DEFAULT_GENERATION_MAX_TOKENS
    value = int(raw)
    if value <= 0:
        raise ValueError("SYNTHETIC_MARKET_V2_SMOKE_MAX_TOKENS must be a positive integer")
    return value


def _capture_gpu() -> str:
    raw = os.environ.get("SYNTHETIC_MARKET_V2_SMOKE_GPU")
    if raw is None or not raw.strip():
        return DEFAULT_CAPTURE_GPU
    return str(raw).strip()


def _patch_operator() -> str:
    raw = os.environ.get("SYNTHETIC_MARKET_V2_SMOKE_PATCH_OPERATOR")
    value = DEFAULT_PATCH_OPERATOR if raw is None or not raw.strip() else str(raw).strip().lower()
    if value not in {"project_out", "add_direction"}:
        raise ValueError(
            "SYNTHETIC_MARKET_V2_SMOKE_PATCH_OPERATOR must be one of {'project_out', 'add_direction'}"
        )
    return value


def _direction_target() -> str:
    raw = os.environ.get("SYNTHETIC_MARKET_V2_SMOKE_DIRECTION_TARGET")
    return DEFAULT_DIRECTION_TARGET if raw is None or not raw.strip() else str(raw).strip()


def build_dataset(*, limit: int | None = None) -> Dataset:
    dataset = Dataset.from_postgres(
        source=PostgresSource.from_env(DB_ENV_VAR),
        sql=DATASET_SQL,
        prompt_column="prompt_messages_json",
        example_key_column="log_id",
        label_columns=[
            "phase_name",
            "example_id",
            "family",
            "family_variant",
            "context_variant",
        ],
        name="dx_terminal_synthetic_market_path_validation_v2_smoke",
    )
    actual_limit = _dataset_limit() if limit is None else int(limit)
    return dataset.select(limit=actual_limit)


def build_prompt_metadata(rendered_prompt: str) -> dict[str, object]:
    starts: list[tuple[str, int]] = []
    for name, marker in _SECTION_HEADERS:
        idx = rendered_prompt.find(marker)
        if idx >= 0:
            starts.append((name, idx))
    starts.sort(key=lambda item: item[1])
    token_sections: dict[str, dict[str, int]] = {}
    for index, (name, start) in enumerate(starts):
        raw_end = starts[index + 1][1] if index + 1 < len(starts) else len(rendered_prompt)
        end = _trim_section_end_char(
            rendered_prompt,
            section_start_char=int(start),
            section_end_char=int(raw_end),
        )
        token_sections[name] = {
            "char_start": int(start),
            "char_end": int(end),
        }
    return {"token_sections": token_sections}


def evaluate_patch_row(
    *,
    example: dict[str, Any],
    baseline: dict[str, Any],
    variants: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    patched = dict(variants or {}).get("patch") or dict(variants or {}).get("lesion", {})
    baseline_text = str(baseline.get("generated_text") or "")
    patched_text = str(patched.get("generated_text") or "")
    baseline_tool = _extract_tool_call(baseline_text)
    patched_tool = _extract_tool_call(patched_text)
    return {
        "metrics": {
            "text_changed": baseline_text != patched_text,
            "baseline_nonempty": bool(baseline_text.strip()),
            "patched_nonempty": bool(patched_text.strip()),
            "tool_call_changed": json.dumps(baseline_tool, sort_keys=True) != json.dumps(patched_tool, sort_keys=True),
        },
        "evaluation": {
            "example_key": str(example.get("key") or ""),
            "baseline_text": baseline_text,
            "patched_text": patched_text,
            "baseline_tool": baseline_tool,
            "patched_tool": patched_tool,
        },
    }


def _load_axis_targets() -> dict[str, Any]:
    from pipelines_v2.core.paths import find_workspace_root

    workspace_root = find_workspace_root()
    results_json_path = workspace_root / BASIS_RESULTS_PATH
    results_payload = json.loads(results_json_path.read_text()) if results_json_path.exists() else {}
    targets = results_payload.get("targets", {}) if isinstance(results_payload, dict) else {}
    return {str(name): payload for name, payload in dict(targets).items() if isinstance(payload, dict)}


def _operation_payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "result") and callable(value.result):
        value = value.result()
    payload = dict(value or {})
    return payload


def import_market_subspace(
    *,
    state_key: str,
    layers: list[int],
    components_per_layer: int,
) -> dict[str, Any]:
    from pipelines_v2.core.paths import find_workspace_root

    workspace_root = find_workspace_root()
    basis_npz_path = workspace_root / BASIS_NPZ_PATH
    basis_archive = np.load(basis_npz_path)
    targets = _load_axis_targets()
    layer_payloads: dict[str, Any] = {}
    for layer in sorted(int(value) for value in layers):
        prefix = f"{state_key}_layer_{layer}"
        mean_key = f"{prefix}__mean"
        scale_key = f"{prefix}__scale"
        components_key = f"{prefix}__components"
        if mean_key not in basis_archive or scale_key not in basis_archive or components_key not in basis_archive:
            raise KeyError(
                f"Missing basis payload for {prefix}. Expected keys {mean_key}, {scale_key}, {components_key}."
            )
        components = np.asarray(basis_archive[components_key], dtype=np.float32)
        take = min(max(1, int(components_per_layer)), int(components.shape[0]))
        named_components: dict[str, int] = {}
        for target_name, target_payload in targets.items():
            if not isinstance(target_payload, dict):
                continue
            if str(target_payload.get("state_key")) != str(state_key):
                continue
            if int(target_payload.get("layer", -1)) != int(layer):
                continue
            pc_index = int(target_payload.get("pc_index", 0))
            if 1 <= pc_index <= take:
                named_components[str(target_name)] = pc_index - 1
        layer_payloads[str(layer)] = {
            "mean": np.asarray(basis_archive[mean_key], dtype=np.float32).astype(float).tolist(),
            "scale": np.asarray(basis_archive[scale_key], dtype=np.float32).astype(float).tolist(),
            "components": components[:take].astype(float).tolist(),
            "explained_variance_ratio": [],
            "example_count": None,
            "component_count": int(take),
            "named_components": named_components,
        }
    return {
        "payload": {
            "kind": "subspace_result",
            "feature": "synthetic_market_precomputed_market_mean",
            "layers": layer_payloads,
            "summary": {
                "layer_count": len(layer_payloads),
                "component_count": int(components_per_layer),
                "source": "phase17_activation_patch_basis",
                "state_key": str(state_key),
            },
        }
    }


def import_market_direction(
    *,
    target_name: str,
    subspace_payload: Any,
) -> dict[str, Any]:
    payload = _operation_payload(subspace_payload)
    if str(payload.get("kind") or "") != "subspace_result":
        raise ValueError("import_market_direction requires a subspace_result payload")
    layers_payload = dict(payload.get("layers") or {})

    layer: int | None = None
    component_index: int | None = None
    selected_layer_payload: dict[str, Any] | None = None
    for raw_layer, raw_layer_payload in layers_payload.items():
        layer_payload = dict(raw_layer_payload or {})
        named_components = {
            str(name): int(index)
            for name, index in dict(layer_payload.get("named_components") or {}).items()
        }
        if str(target_name) in named_components:
            layer = int(raw_layer)
            component_index = int(named_components[str(target_name)])
            selected_layer_payload = layer_payload
            break
    if layer is None or component_index is None or selected_layer_payload is None:
        raise KeyError(f"Unknown synthetic-market direction target: {target_name!r}")

    scale = np.asarray(selected_layer_payload.get("scale") or (), dtype=np.float32)
    components = np.asarray(selected_layer_payload.get("components") or (), dtype=np.float32)
    if components.ndim == 1:
        components = components[None, :]
    if component_index >= int(components.shape[0]):
        raise ValueError(
            f"Direction target {target_name!r} requires component {component_index + 1}, "
            f"but only {int(components.shape[0])} components are available"
        )

    standardized_vector = components[component_index].astype(np.float32)
    raw_vector = (standardized_vector * scale).astype(np.float32)
    norm = float(np.linalg.norm(raw_vector))
    unit = (raw_vector / norm).astype(np.float32) if norm > 0 else raw_vector
    weights = np.zeros((int(components.shape[0]),), dtype=np.float32)
    weights[component_index] = 1.0
    return {
        "payload": {
            "kind": "direction_result",
            "feature": str(payload.get("feature") or "synthetic_market_precomputed_market_mean"),
            "layers": {
                str(layer): {
                    "vector": unit.astype(float).tolist(),
                    "raw_vector": raw_vector.astype(float).tolist(),
                    "norm": norm,
                    "subspace_weights": weights.astype(float).tolist(),
                    "subspace_component_count": int(components.shape[0]),
                    "target_name": str(target_name),
                }
            },
            "summary": {
                "layer_count": 1,
                "target_name": str(target_name),
                "source": "subspace_named_component",
            },
        }
    }


def build_engine(*, batch_size: int = 16) -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=40960,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        max_num_seqs=max(1, int(batch_size)),
        max_num_batched_tokens=max(40960, max(1, int(batch_size)) * 4096),
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        async_scheduling=False,
        add_generation_prompt=True,
        enable_thinking=None,
    )


def build_runner_specs() -> dict[str, object]:
    db = PostgresSource.from_env(DB_ENV_VAR)
    secret = ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon")
    artifact_store = ModalVolumeStore(name="xenon-data", root=ARTIFACT_ROOT)
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu=_capture_gpu(),
                timeout_seconds=60 * 60,
                secrets=(secret,),
                volumes=(
                    ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),
                    ModalVolumeMount(name="xenon-data", mount_path="/data"),
                ),
            ),
            artifacts=artifact_store,
            catalog=PostgresCatalog(source=db),
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=6,
                memory_mb=24 * 1024,
                timeout_seconds=60 * 60,
                secrets=(secret,),
                volumes=(ModalVolumeMount(name="xenon-data", mount_path="/data"),),
            ),
            artifacts=artifact_store,
            catalog=PostgresCatalog(source=db),
        ),
    }


def build_workflow(
    dataset: Dataset | None = None,
    *,
    patch_operator: str | None = None,
    direction_target: str | None = None,
) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    selected_patch_operator = _patch_operator() if patch_operator is None else str(patch_operator).strip().lower()
    selected_direction_target = _direction_target() if direction_target is None else str(direction_target).strip()
    prompt_metadata = PromptMetadataBuilder.from_function(
        build_prompt_metadata,
        local_python_sources=("projects",),
    )
    chat_tools = _trading_decision_chat_tools()
    structured_output = _trading_decision_structured_output()
    row_evaluator = TransformBuilder.from_function(
        evaluate_patch_row,
        local_python_sources=("projects",),
    )
    subspace_builder = TransformBuilder.from_function(
        import_market_subspace,
        local_python_sources=("projects", "data"),
    )
    direction_builder = TransformBuilder.from_function(
        import_market_direction,
        local_python_sources=("projects",),
    )
    engine = build_engine(batch_size=_dataset_limit())
    generation_max_tokens = _generation_max_tokens()
    if selected_patch_operator == "project_out":
        patch_spec = ProjectOutPatch(
            subspace=StepRef("import_market_subspace"),
            write_site=ResidualInterventionSite(site="resid_post", layers=(PATCH_LAYER,)),
            target_tokens=TokenSelector.section("market"),
            component_indices_by_layer={
                PATCH_LAYER: tuple(range(PATCH_COMPONENTS)),
            },
            strength=1.0,
        )
    elif selected_patch_operator == "add_direction":
        patch_spec = AddDirectionPatch(
            direction=StepRef("import_market_direction"),
            subspace=StepRef("import_market_subspace"),
            write_site=ResidualInterventionSite(site="resid_post", layers=(PATCH_LAYER,)),
            target_tokens=TokenSelector.section("market"),
            strength=1.0,
        )
    else:
        raise ValueError(f"Unsupported synthetic-market smoke patch operator: {selected_patch_operator!r}")
    return WorkflowSpec(
        name="dx_terminal_synthetic_market_path_validation_v2_smoke",
        steps=(
            WorkflowStep(
                name="import_market_subspace",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=subspace_builder,
                    inputs={
                        "state_key": DEFAULT_BASIS_STATE_KEY,
                        "layers": [PATCH_LAYER],
                        "components_per_layer": PATCH_COMPONENTS,
                    },
                    inline=True,
                ),
            ),
            WorkflowStep(
                name="import_market_direction",
                runner="analysis_cpu",
                depends_on=("import_market_subspace",),
                spec=TransformSpec(
                    builder=direction_builder,
                    inputs={
                        "target_name": selected_direction_target,
                        "subspace_payload": StepRef("import_market_subspace"),
                    },
                    inline=True,
                ),
            ),
            WorkflowStep(
                name="baseline_generation",
                runner="capture_gpu",
                spec=GenerationRunSpec(
                    engine=engine,
                    dataset=dataset,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=generation_max_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=-1,
                        capture_reasoning=False,
                        chat_tools=chat_tools,
                        tool_choice="required",
                        structured_output=structured_output,
                    ),
                ),
            ),
            WorkflowStep(
                name="patch_market",
                runner="capture_gpu",
                depends_on=("import_market_subspace", "import_market_direction", "baseline_generation"),
                spec=PatchedGenerationSpec(
                    engine=engine,
                    dataset=dataset,
                    patch=patch_spec,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=generation_max_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=-1,
                        capture_reasoning=False,
                        chat_tools=chat_tools,
                        tool_choice="required",
                        structured_output=structured_output,
                    ),
                    prompt_metadata_builder=prompt_metadata,
                ),
            ),
            WorkflowStep(
                name="compare_patch",
                runner="analysis_cpu",
                depends_on=("patch_market",),
                spec=PatchComparisonSpec(
                    baseline=StepRef("baseline_generation"),
                    variants={"patch": StepRef("patch_market")},
                    row_evaluator=row_evaluator,
                ),
            ),
        ),
    )


def _extract_tool_call(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    for candidate in (stripped, stripped.rsplit("</think>", 1)[-1].strip()):
        if candidate.startswith("{") or candidate.startswith("["):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            return _normalize_tool_payload(payload)
    match = _TOOL_CALL_PATTERN.search(stripped)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {"raw": match.group(1), "parse_ok": False}
    normalized = _normalize_tool_payload(payload)
    normalized["parse_ok"] = True
    return normalized


def _normalize_tool_payload(payload: Any) -> dict[str, Any]:
    item = payload[0] if isinstance(payload, list) and payload else payload
    if not isinstance(item, dict):
        return {"raw": payload}
    arguments = item.get("arguments", {}) if isinstance(item.get("arguments"), dict) else {}
    return {
        "name": item.get("name"),
        "token": arguments.get("token"),
        "strategy": arguments.get("strategy"),
        "spend_pct": arguments.get("spend_pct"),
        "content": arguments.get("content"),
    }
