from __future__ import annotations

import json
from typing import Any


def parse_json_payload(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        payload = json.loads(raw[start : end + 1])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def example_value(example: dict[str, Any], key: str) -> Any:
    if key in example:
        return example.get(key)
    labels = example.get("labels")
    if isinstance(labels, dict):
        return labels.get(key)
    return None


def evaluate_trade_size_patch_row(
    *,
    example: dict[str, Any],
    baseline: dict[str, Any],
    variants: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    strategy_size = str(example_value(example, "strategy_direction") or "")
    setting_size = str(example_value(example, "setting_implied_direction") or "")
    baseline_payload = parse_json_payload(str(baseline.get("generated_text") or ""))
    baseline_size = str((baseline_payload or {}).get("size") or "")
    results: dict[str, Any] = {}
    summary: dict[str, Any] = {
        "example_key": str(example.get("key") or example.get("example_key") or ""),
        "strategy_size": strategy_size,
        "setting_size": setting_size,
        "baseline_text": str(baseline.get("generated_text") or ""),
        "baseline_valid": baseline_payload is not None,
        "baseline_size": baseline_size,
        "baseline_follows_setting": baseline_size == setting_size,
        "baseline_follows_strategy": baseline_size == strategy_size,
    }
    for variant_name, row in dict(variants or {}).items():
        payload = parse_json_payload(str(row.get("generated_text") or ""))
        patched_size = str((payload or {}).get("size") or "")
        results[variant_name] = {
            "valid_json": payload is not None,
            "size_changed": patched_size != baseline_size,
            "patched_follows_setting": patched_size == setting_size,
            "patched_follows_strategy": patched_size == strategy_size,
            "intended_erasure_flip": baseline_size == setting_size and patched_size == strategy_size,
            "reverse_flip": baseline_size == strategy_size and patched_size == setting_size,
            "malformed": payload is None,
        }
        summary[f"{variant_name}_text"] = str(row.get("generated_text") or "")
        summary[f"{variant_name}_size"] = patched_size
        donor_key = row.get("donor_example_key")
        if donor_key is not None:
            summary[f"{variant_name}_donor_example_key"] = str(donor_key)
    return {"metrics": results, "evaluation": summary}
