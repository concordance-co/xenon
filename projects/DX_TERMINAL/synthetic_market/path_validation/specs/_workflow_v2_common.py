from __future__ import annotations

import json
import os
import re
from typing import Any

from pipelines_v2.api import (
    GenerationSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PostgresCatalog,
    PostgresSource,
    VLLMEngine,
)


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
DEFAULT_CAPTURE_GPU = "A100-80GB"
DEFAULT_GENERATION_MAX_TOKENS = 2500

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


def generation_max_tokens(*, env_var: str = "SYNTHETIC_MARKET_V2_SMOKE_MAX_TOKENS") -> int:
    raw = os.environ.get(env_var)
    if raw is None or not raw.strip():
        return DEFAULT_GENERATION_MAX_TOKENS
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{env_var} must be a positive integer")
    return value


def capture_gpu(*, env_var: str = "SYNTHETIC_MARKET_V2_SMOKE_GPU") -> str:
    raw = os.environ.get(env_var)
    if raw is None or not raw.strip():
        return DEFAULT_CAPTURE_GPU
    return str(raw).strip()


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


def build_runner_specs(*, artifact_root: str, gpu_env_var: str = "SYNTHETIC_MARKET_V2_SMOKE_GPU") -> dict[str, object]:
    db = PostgresSource.from_env(DB_ENV_VAR)
    secret = ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon")
    artifact_store = ModalVolumeStore(name="xenon-data", root=artifact_root)
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu=capture_gpu(env_var=gpu_env_var),
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


def trading_decision_chat_tools() -> tuple[dict[str, Any], ...]:
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


def trading_decision_structured_output() -> dict[str, Any]:
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


def build_generation_spec(*, max_tokens: int) -> GenerationSpec:
    return GenerationSpec(
        enabled=True,
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        capture_reasoning=False,
        chat_tools=trading_decision_chat_tools(),
        tool_choice="required",
        structured_output=trading_decision_structured_output(),
    )


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


__all__ = [
    "DB_ENV_VAR",
    "MODEL_ID",
    "build_engine",
    "build_generation_spec",
    "build_prompt_metadata",
    "build_runner_specs",
    "capture_gpu",
    "evaluate_patch_row",
    "generation_max_tokens",
    "trading_decision_chat_tools",
    "trading_decision_structured_output",
]
