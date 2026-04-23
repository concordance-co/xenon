from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import modal


APP_NAME = "xenon-prompt-confusion-phase9-behavior-smoke"
MODEL_ID = "Qwen/Qwen3-30B-A3B"
DEFAULT_INPUT = Path(
    "projects/DX_TERMINAL/prompt_confusion/phase_09/outputs/phase_09_dataset/phase_09_dataset.jsonl"
)
DEFAULT_OUTPUT = Path(
    "projects/DX_TERMINAL/prompt_confusion/phase_09/reports/behavior_smoke.json"
)
ALLOWED_ACTIONS = {"buy", "sell", "observe"}
ALLOWED_ASSETS = {"ALPHA", "BETA", "DELTA", "GAMMA", "NONE"}
ALLOWED_SIZES = {"small", "large", "none"}

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("xenon-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

gpu_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("numpy", "pyarrow", "torch", "transformers", "vllm", "huggingface_hub")
    .env({"VLLM_ALLOW_INSECURE_SERIALIZATION": "1"})
    .add_local_python_source("pipelines")
)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _classify(row: dict[str, Any], generated_text: str) -> dict[str, Any]:
    parsed = _parse_json_object(generated_text)
    result = {
        "valid_output": False,
        "exact_expected": False,
        "behavior_side": "neither",
        "action_label": "invalid",
        "asset_label": "invalid",
        "size_label": "invalid",
    }
    if parsed is None or set(parsed.keys()) != {"action", "asset", "size"}:
        return result

    action = parsed.get("action")
    asset = parsed.get("asset")
    size = parsed.get("size")
    if not (
        isinstance(action, str)
        and isinstance(asset, str)
        and isinstance(size, str)
        and action in ALLOWED_ACTIONS
        and asset in ALLOWED_ASSETS
        and size in ALLOWED_SIZES
    ):
        return result

    generated_triplet = {"action": action, "asset": asset, "size": size}
    strategy_expected = {
        "action": str(row["strategy_expected_action"]),
        "asset": str(row["strategy_expected_asset"]),
        "size": str(row["strategy_expected_size"]),
    }
    setting_expected = {
        "action": str(row["setting_expected_action"]),
        "asset": str(row["setting_expected_asset"]),
        "size": str(row["setting_expected_size"]),
    }
    expected_output = dict(row["expected_output_json"])
    matches_strategy = generated_triplet == strategy_expected
    matches_setting = generated_triplet == setting_expected
    if matches_strategy and matches_setting:
        behavior_side = "both"
    elif matches_strategy:
        behavior_side = "strategy"
    elif matches_setting:
        behavior_side = "setting"
    else:
        behavior_side = "neither"

    result.update(
        {
            "valid_output": True,
            "exact_expected": generated_triplet == expected_output,
            "behavior_side": behavior_side,
            "action_label": action,
            "asset_label": asset,
            "size_label": size,
        }
    )
    return result


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    if not rows:
        raise SystemExit(f"No rows loaded from {path}")
    return rows


def _sample_rows(rows: list[dict[str, Any]], max_per_cell: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_cell: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["target_dimension"]),
            str(row["strategy_direction"]),
            str(row["market_snapshot_json"]["evidence_tier"]),
            int(row["setting_value"]),
            str(row["lexical_split"]),
        )
        by_cell[key].append(row)
    for key, bucket in sorted(by_cell.items()):
        bucket_sorted = sorted(bucket, key=lambda r: str(r["example_id"]))
        selected.extend(bucket_sorted[:max_per_cell])
    return selected


@app.function(
    volumes={"/models": model_volume},
    image=gpu_image,
    gpu="H100",
    timeout=2 * 3600,
    cpu=4,
    memory=32 * 1024,
    secrets=[hf_secret],
)
def generate_rows(
    rows: list[dict[str, Any]],
    *,
    batch_size: int = 8,
    max_tokens: int = 128,
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    from pipelines.interp.modal_vllm_engine import (
        VLLMCaptureConfig,
        _cleanup_cuda_memory,
        _create_llm,
        _destroy_llm,
        _generate_batch_vllm,
    )

    local_model_path = f"/models/{MODEL_ID}"
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    cfg = VLLMCaptureConfig(
        output_dir=Path("/tmp/phase09_behavior_smoke"),
        model_id=local_model_path,
        add_generation_prompt=True,
        capture_router=False,
        capture_residual=False,
        capture_generation=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_num_seqs=max(1, int(batch_size)),
        max_num_batched_tokens=max(40960, max(1, int(batch_size)) * 4096),
        async_scheduling=False if int(batch_size) > 1 else None,
    )
    llm = _create_llm(cfg)
    outputs: list[dict[str, Any]] = []
    try:
        for offset in range(0, len(rows), max(1, int(batch_size))):
            chunk = rows[offset : offset + max(1, int(batch_size))]
            batch_requests = [{"messages": row["prompt_messages_json"]} for row in chunk]
            batch_outputs = _generate_batch_vllm(
                llm=llm,
                tokenizer=tokenizer,
                batch_requests=batch_requests,
                config=cfg,
                max_tokens=int(max_tokens),
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                chat_template_kwargs={"enable_thinking": False},
            )
            for row, output in zip(chunk, batch_outputs, strict=False):
                outputs.append(
                    {
                        "example_id": str(row["example_id"]),
                        "generated_text": str(output.get("generated_text") or ""),
                        "finish_reason": str(output.get("finish_reason") or ""),
                    }
                )
    finally:
        _destroy_llm(llm)
        _cleanup_cuda_memory()
    return outputs


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    behavior_counts = Counter(str(row["behavior_side"]) for row in rows)
    exact_true = sum(1 for row in rows if bool(row["exact_expected"]))
    valid_true = sum(1 for row in rows if bool(row["valid_output"]))
    summary: dict[str, Any] = {
        "rows": len(rows),
        "valid_output_rate": round(valid_true / len(rows), 4) if rows else 0.0,
        "exact_expected_rate": round(exact_true / len(rows), 4) if rows else 0.0,
        "behavior_side_counts": dict(behavior_counts),
        "by_cell": {},
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            f"{row['target_dimension']}::{row['strategy_direction']}::"
            f"{row['market_snapshot_json']['evidence_tier']}::value_{row['setting_value']}::"
            f"{row['lexical_split']}"
        )
        grouped[key].append(row)
    for key, bucket in sorted(grouped.items()):
        bucket_behavior = Counter(str(row["behavior_side"]) for row in bucket)
        bucket_exact = sum(1 for row in bucket if bool(row["exact_expected"]))
        bucket_valid = sum(1 for row in bucket if bool(row["valid_output"]))
        action_match = sum(1 for row in bucket if row["action_label"] == row["expected_output_json"]["action"])
        size_match = sum(1 for row in bucket if row["size_label"] == row["expected_output_json"]["size"])
        summary["by_cell"][key] = {
            "rows": len(bucket),
            "valid_output_rate": round(bucket_valid / len(bucket), 4) if bucket else 0.0,
            "exact_expected_rate": round(bucket_exact / len(bucket), 4) if bucket else 0.0,
            "action_match_rate": round(action_match / len(bucket), 4) if bucket else 0.0,
            "size_match_rate": round(size_match / len(bucket), 4) if bucket else 0.0,
            "behavior_side_counts": dict(bucket_behavior),
        }
    return summary


@app.local_entrypoint()
def main(
    input: str = str(DEFAULT_INPUT),
    output: str = str(DEFAULT_OUTPUT),
    max_per_cell: int = 1,
    batch_size: int = 8,
    max_tokens: int = 128,
) -> None:
    rows = _load_rows(Path(input))
    sampled = _sample_rows(rows, int(max_per_cell))
    generations = generate_rows.remote(sampled, batch_size=int(batch_size), max_tokens=int(max_tokens))
    generated_by_id = {str(item["example_id"]): item for item in generations}

    analyzed: list[dict[str, Any]] = []
    for row in sampled:
        output_row = generated_by_id[str(row["example_id"])]
        classified = _classify(row, str(output_row["generated_text"]))
        analyzed.append(
            {
                "example_id": str(row["example_id"]),
                "target_dimension": str(row["target_dimension"]),
                "strategy_direction": str(row["strategy_direction"]),
                "conflict_band": str(row["conflict_band"]),
                "conflict_present": row["conflict_present"],
                "setting_value": int(row["setting_value"]),
                "lexical_split": str(row["lexical_split"]),
                "market_snapshot_json": row["market_snapshot_json"],
                "expected_output_json": row["expected_output_json"],
                "generated_text": str(output_row["generated_text"]),
                "finish_reason": str(output_row["finish_reason"]),
                **classified,
            }
        )

    payload = {
        "max_per_cell": int(max_per_cell),
        "rows": analyzed,
        "sample_size": len(analyzed),
        "summary": _summarize(analyzed),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
