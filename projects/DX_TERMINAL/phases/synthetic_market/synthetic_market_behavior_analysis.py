from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(slots=True)
class SyntheticMarketBehaviorAnalysisConfig:
    baseline_dir: Path
    intervention_dir: Path
    output_dir: Path
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 0


def _load_rows(path: Path) -> list[dict[str, Any]]:
    table_path = path / "metadata.parquet"
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    return pq.read_table(table_path).to_pylist()


def _bootstrap_interval(
    values: list[float],
    *,
    reducer: Callable[[np.ndarray], float],
    samples: int,
    seed: int,
) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        value = float(reducer(arr))
        return {"low": value, "high": value}
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(samples),), dtype=np.float64)
    for idx in range(int(samples)):
        sample = arr[rng.integers(0, arr.size, size=arr.size)]
        draws[idx] = reducer(sample)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"low": float(low), "high": float(high)}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_patch_stats(raw: Any) -> list[dict[str, Any]]:
    if raw in (None, ""):
        return []
    payload = raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(payload, dict):
        return [dict(entry) for entry in payload.values() if isinstance(entry, dict)]
    if isinstance(payload, list):
        return [dict(entry) for entry in payload if isinstance(entry, dict)]
    return []


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _bucket_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        values.append(float(value))
    return values


def _summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count <= 0:
        return {"count": 0}
    return {
        "count": count,
        "tool_name_change_rate": float(np.mean(_bucket_values(rows, "tool_name_changed"))),
        "tool_token_change_rate": float(np.mean(_bucket_values(rows, "tool_token_changed"))),
        "mean_generated_token_count_delta": float(np.mean(_bucket_values(rows, "generated_token_count_delta"))),
        "mean_pair_metric_gap": _mean_or_none(_bucket_values(rows, "pair_metric_gap")),
        "source_tool_name_match_rate_baseline": _mean_or_none(_bucket_values(rows, "source_tool_name_match_baseline")),
        "source_tool_name_match_rate_intervention": _mean_or_none(
            _bucket_values(rows, "source_tool_name_match_intervention")
        ),
        "source_tool_name_restoration_rate": _mean_or_none(_bucket_values(rows, "source_tool_name_restored")),
        "source_tool_name_backfire_rate": _mean_or_none(_bucket_values(rows, "source_tool_name_backfire")),
        "source_tool_token_match_rate_baseline": _mean_or_none(
            _bucket_values(rows, "source_tool_token_match_baseline")
        ),
        "source_tool_token_match_rate_intervention": _mean_or_none(
            _bucket_values(rows, "source_tool_token_match_intervention")
        ),
        "source_tool_token_restoration_rate": _mean_or_none(_bucket_values(rows, "source_tool_token_restored")),
        "source_tool_token_backfire_rate": _mean_or_none(_bucket_values(rows, "source_tool_token_backfire")),
        "mean_source_tool_spend_pct_gap_baseline": _mean_or_none(
            _bucket_values(rows, "source_tool_spend_pct_gap_baseline")
        ),
        "mean_source_tool_spend_pct_gap_intervention": _mean_or_none(
            _bucket_values(rows, "source_tool_spend_pct_gap_intervention")
        ),
        "source_tool_spend_pct_improvement_rate": _mean_or_none(
            _bucket_values(rows, "source_tool_spend_pct_improved")
        ),
        "source_tool_spend_pct_backfire_rate": _mean_or_none(
            _bucket_values(rows, "source_tool_spend_pct_backfired")
        ),
        "mean_source_tool_spend_pct_normalized_restoration": _mean_or_none(
            _bucket_values(rows, "source_tool_spend_pct_normalized_restoration")
        ),
        "mean_source_generated_token_count_gap_baseline": _mean_or_none(
            _bucket_values(rows, "source_generated_token_count_gap_baseline")
        ),
        "mean_source_generated_token_count_gap_intervention": _mean_or_none(
            _bucket_values(rows, "source_generated_token_count_gap_intervention")
        ),
        "source_generated_token_count_improvement_rate": _mean_or_none(
            _bucket_values(rows, "source_generated_token_count_improved")
        ),
        "source_generated_token_count_backfire_rate": _mean_or_none(
            _bucket_values(rows, "source_generated_token_count_backfired")
        ),
        "mean_source_generated_token_count_normalized_restoration": _mean_or_none(
            _bucket_values(rows, "source_generated_token_count_normalized_restoration")
        ),
    }


def run_synthetic_market_behavior_analysis(config: SyntheticMarketBehaviorAnalysisConfig) -> dict[str, Any]:
    baseline_rows = {int(row["log_id"]): row for row in _load_rows(config.baseline_dir)}
    intervention_rows = {int(row["log_id"]): row for row in _load_rows(config.intervention_dir)}
    common_ids = sorted(set(baseline_rows) & set(intervention_rows))
    if not common_ids:
        raise ValueError("No shared log_ids between baseline and intervention behavior runs")

    first_token_flags: list[float] = []
    text_flags: list[float] = []
    tool_presence_flags: list[float] = []
    tool_name_flags: list[float] = []
    tool_token_flags: list[float] = []
    generated_token_deltas: list[float] = []
    spend_deltas: list[float] = []
    source_tool_name_match_baseline: list[float] = []
    source_tool_name_match_intervention: list[float] = []
    source_tool_name_restoration_flags: list[float] = []
    source_tool_name_backfire_flags: list[float] = []
    source_tool_token_match_baseline: list[float] = []
    source_tool_token_match_intervention: list[float] = []
    source_tool_token_restoration_flags: list[float] = []
    source_tool_token_backfire_flags: list[float] = []
    source_spend_delta_baseline: list[float] = []
    source_spend_delta_intervention: list[float] = []
    source_spend_improvement_flags: list[float] = []
    source_spend_full_restoration_flags: list[float] = []
    source_spend_backfire_flags: list[float] = []
    source_spend_normalized_restoration_values: list[float] = []
    source_generated_token_delta_baseline: list[float] = []
    source_generated_token_delta_intervention: list[float] = []
    source_generated_token_improvement_flags: list[float] = []
    source_generated_token_full_restoration_flags: list[float] = []
    source_generated_token_backfire_flags: list[float] = []
    source_generated_token_normalized_restoration_values: list[float] = []
    pair_metric_gap_values: list[float] = []

    patch_applied_flags: list[float] = []
    patch_skipped_flags: list[float] = []
    delta_norm_std_values: list[float] = []
    mean_norm_ratio_values: list[float] = []
    mean_std_norm_ratio_values: list[float] = []
    selected_proj_norm_values: list[float] = []

    rows: list[dict[str, Any]] = []
    family_variant_rows: dict[str, list[dict[str, Any]]] = {}
    pair_mode_rows: dict[str, list[dict[str, Any]]] = {}

    for log_id in common_ids:
        baseline = baseline_rows[log_id]
        intervention = intervention_rows[log_id]
        pair_mode = str(intervention.get("pair_mode") or baseline.get("pair_mode") or "").strip().lower() or None
        baseline_has_tool = bool(baseline.get("has_tool_call"))
        intervention_has_tool = bool(intervention.get("has_tool_call"))

        first_changed = baseline.get("first_generated_token_id") != intervention.get("first_generated_token_id")
        text_changed_flag = str(baseline.get("generated_text", "")) != str(intervention.get("generated_text", ""))
        tool_presence_changed_flag = baseline_has_tool != intervention_has_tool
        tool_name_changed_flag = baseline.get("first_tool_name") != intervention.get("first_tool_name")
        tool_token_changed_flag = baseline.get("first_tool_token") != intervention.get("first_tool_token")

        first_token_flags.append(float(first_changed))
        text_flags.append(float(text_changed_flag))
        tool_presence_flags.append(float(tool_presence_changed_flag))
        tool_name_flags.append(float(tool_name_changed_flag))
        tool_token_flags.append(float(tool_token_changed_flag))

        generated_delta = abs(
            int(intervention.get("generated_token_count", 0)) - int(baseline.get("generated_token_count", 0))
        )
        generated_token_deltas.append(float(generated_delta))

        baseline_spend = _safe_float(baseline.get("first_tool_spend_pct"))
        intervention_spend = _safe_float(intervention.get("first_tool_spend_pct"))
        source_tool_name = intervention.get("source_first_tool_name")
        source_tool_token = intervention.get("source_first_tool_token")
        source_spend = _safe_float(intervention.get("source_first_tool_spend_pct"))
        source_generated_token_count = intervention.get("source_generated_token_count")
        pair_metric_gap = _safe_float(intervention.get("pair_metric_gap"))
        if pair_metric_gap is not None:
            pair_metric_gap_values.append(pair_metric_gap)
        spend_delta = None
        if baseline_spend is not None and intervention_spend is not None:
            spend_delta = abs(intervention_spend - baseline_spend)
            spend_deltas.append(float(spend_delta))
        source_tool_name_match_baseline_value: float | None = None
        source_tool_name_match_intervention_value: float | None = None
        source_tool_name_restored_value: float | None = None
        source_tool_name_backfire_value: float | None = None
        if source_tool_name is not None:
            baseline_name_match = baseline.get("first_tool_name") == source_tool_name
            intervention_name_match = intervention.get("first_tool_name") == source_tool_name
            source_tool_name_match_baseline_value = float(baseline_name_match)
            source_tool_name_match_intervention_value = float(intervention_name_match)
            source_tool_name_match_baseline.append(source_tool_name_match_baseline_value)
            source_tool_name_match_intervention.append(source_tool_name_match_intervention_value)
            if not baseline_name_match:
                source_tool_name_restored_value = float(intervention_name_match)
                source_tool_name_restoration_flags.append(source_tool_name_restored_value)
            else:
                source_tool_name_backfire_value = float(not intervention_name_match)
                source_tool_name_backfire_flags.append(source_tool_name_backfire_value)
        source_tool_token_match_baseline_value: float | None = None
        source_tool_token_match_intervention_value: float | None = None
        source_tool_token_restored_value: float | None = None
        source_tool_token_backfire_value: float | None = None
        if source_tool_token is not None:
            baseline_token_match = baseline.get("first_tool_token") == source_tool_token
            intervention_token_match = intervention.get("first_tool_token") == source_tool_token
            source_tool_token_match_baseline_value = float(baseline_token_match)
            source_tool_token_match_intervention_value = float(intervention_token_match)
            source_tool_token_match_baseline.append(source_tool_token_match_baseline_value)
            source_tool_token_match_intervention.append(source_tool_token_match_intervention_value)
            if not baseline_token_match:
                source_tool_token_restored_value = float(intervention_token_match)
                source_tool_token_restoration_flags.append(source_tool_token_restored_value)
            else:
                source_tool_token_backfire_value = float(not intervention_token_match)
                source_tool_token_backfire_flags.append(source_tool_token_backfire_value)
        source_spend_gap_baseline: float | None = None
        source_spend_gap_intervention: float | None = None
        source_spend_improved_value: float | None = None
        source_spend_restored_value: float | None = None
        source_spend_backfire_value: float | None = None
        source_spend_normalized_restoration: float | None = None
        if source_spend is not None:
            if baseline_spend is not None:
                source_spend_gap_baseline = abs(baseline_spend - source_spend)
                source_spend_delta_baseline.append(source_spend_gap_baseline)
            if intervention_spend is not None:
                source_spend_gap_intervention = abs(intervention_spend - source_spend)
                source_spend_delta_intervention.append(source_spend_gap_intervention)
            if source_spend_gap_baseline is not None and source_spend_gap_intervention is not None:
                if source_spend_gap_baseline > 0.0:
                    source_spend_improved_value = float(source_spend_gap_intervention < source_spend_gap_baseline)
                    source_spend_restored_value = float(source_spend_gap_intervention == 0.0)
                    source_spend_backfire_value = float(source_spend_gap_intervention > source_spend_gap_baseline)
                    source_spend_normalized_restoration = float(
                        (source_spend_gap_baseline - source_spend_gap_intervention) / source_spend_gap_baseline
                    )
                    source_spend_improvement_flags.append(source_spend_improved_value)
                    source_spend_full_restoration_flags.append(source_spend_restored_value)
                    source_spend_backfire_flags.append(source_spend_backfire_value)
                    source_spend_normalized_restoration_values.append(source_spend_normalized_restoration)
                else:
                    source_spend_backfire_value = float(source_spend_gap_intervention > 0.0)
                    source_spend_backfire_flags.append(source_spend_backfire_value)
        source_generated_gap_baseline: float | None = None
        source_generated_gap_intervention: float | None = None
        source_generated_improved_value: float | None = None
        source_generated_restored_value: float | None = None
        source_generated_backfire_value: float | None = None
        source_generated_normalized_restoration: float | None = None
        if source_generated_token_count is not None:
            baseline_generated_count = int(baseline.get("generated_token_count", 0))
            intervention_generated_count = int(intervention.get("generated_token_count", 0))
            source_generated_count = int(source_generated_token_count)
            source_generated_gap_baseline = abs(baseline_generated_count - source_generated_count)
            source_generated_gap_intervention = abs(intervention_generated_count - source_generated_count)
            source_generated_token_delta_baseline.append(float(source_generated_gap_baseline))
            source_generated_token_delta_intervention.append(float(source_generated_gap_intervention))
            if source_generated_gap_baseline > 0:
                source_generated_improved_value = float(source_generated_gap_intervention < source_generated_gap_baseline)
                source_generated_restored_value = float(source_generated_gap_intervention == 0)
                source_generated_backfire_value = float(
                    source_generated_gap_intervention > source_generated_gap_baseline
                )
                source_generated_normalized_restoration = float(
                    (source_generated_gap_baseline - source_generated_gap_intervention) / source_generated_gap_baseline
                )
                source_generated_token_improvement_flags.append(source_generated_improved_value)
                source_generated_token_full_restoration_flags.append(source_generated_restored_value)
                source_generated_token_backfire_flags.append(source_generated_backfire_value)
                source_generated_token_normalized_restoration_values.append(source_generated_normalized_restoration)
            else:
                source_generated_backfire_value = float(source_generated_gap_intervention > 0)
                source_generated_token_backfire_flags.append(source_generated_backfire_value)

        patch_entries = _parse_patch_stats(intervention.get("patch_stats_json"))
        patch_applied = any(entry.get("status") != "skipped" for entry in patch_entries)
        patch_skipped = any(entry.get("status") == "skipped" for entry in patch_entries)
        patch_applied_flags.append(float(patch_applied))
        patch_skipped_flags.append(float(patch_skipped))
        for entry in patch_entries:
            delta_norm_std = _safe_float(entry.get("delta_norm_std"))
            if delta_norm_std is not None:
                delta_norm_std_values.append(delta_norm_std)
            selected_proj_norm = _safe_float(entry.get("selected_proj_norm_before"))
            if selected_proj_norm is not None:
                selected_proj_norm_values.append(selected_proj_norm)
            mean_norm_before = _safe_float(entry.get("mean_norm_before"))
            mean_norm_after = _safe_float(entry.get("mean_norm_after"))
            if mean_norm_before not in (None, 0.0) and mean_norm_after is not None:
                mean_norm_ratio_values.append(float(mean_norm_after / mean_norm_before))
            mean_std_before = _safe_float(entry.get("mean_std_norm_before"))
            mean_std_after = _safe_float(entry.get("mean_std_norm_after"))
            if mean_std_before not in (None, 0.0) and mean_std_after is not None:
                mean_std_norm_ratio_values.append(float(mean_std_after / mean_std_before))

        row_payload = {
            "log_id": int(log_id),
            "example_id": baseline.get("example_id"),
            "family": baseline.get("family"),
            "family_variant": baseline.get("family_variant"),
            "roster_key": baseline.get("roster_key") or intervention.get("roster_key") or "",
            "pair_mode": pair_mode,
            "pair_metric_name": intervention.get("pair_metric_name") or baseline.get("pair_metric_name"),
            "pair_metric_gap": pair_metric_gap,
            "baseline_has_tool_call": baseline_has_tool,
            "intervention_has_tool_call": intervention_has_tool,
            "baseline_first_tool_name": baseline.get("first_tool_name"),
            "intervention_first_tool_name": intervention.get("first_tool_name"),
            "baseline_first_tool_token": baseline.get("first_tool_token"),
            "intervention_first_tool_token": intervention.get("first_tool_token"),
            "source_first_tool_name": source_tool_name,
            "source_first_tool_token": source_tool_token,
            "baseline_first_tool_spend_pct": baseline_spend,
            "intervention_first_tool_spend_pct": intervention_spend,
            "source_first_tool_spend_pct": source_spend,
            "source_generated_token_count": (
                int(source_generated_token_count) if source_generated_token_count is not None else None
            ),
            "baseline_first_token_id": baseline.get("first_generated_token_id"),
            "intervention_first_token_id": intervention.get("first_generated_token_id"),
            "baseline_first_token_text": baseline.get("first_generated_token_text"),
            "intervention_first_token_text": intervention.get("first_generated_token_text"),
            "baseline_generated_text": baseline.get("generated_text", ""),
            "intervention_generated_text": intervention.get("generated_text", ""),
            "generated_token_count_delta": float(generated_delta),
            "tool_spend_pct_delta": spend_delta,
            "first_token_changed": bool(first_changed),
            "text_changed": bool(text_changed_flag),
            "tool_presence_changed": bool(tool_presence_changed_flag),
            "tool_name_changed": bool(tool_name_changed_flag),
            "tool_token_changed": bool(tool_token_changed_flag),
            "patch_applied": bool(patch_applied),
            "patch_skipped": bool(patch_skipped),
            "source_tool_name_match_baseline": source_tool_name_match_baseline_value,
            "source_tool_name_match_intervention": source_tool_name_match_intervention_value,
            "source_tool_name_restored": source_tool_name_restored_value,
            "source_tool_name_backfire": source_tool_name_backfire_value,
            "source_tool_token_match_baseline": source_tool_token_match_baseline_value,
            "source_tool_token_match_intervention": source_tool_token_match_intervention_value,
            "source_tool_token_restored": source_tool_token_restored_value,
            "source_tool_token_backfire": source_tool_token_backfire_value,
            "source_tool_spend_pct_gap_baseline": source_spend_gap_baseline,
            "source_tool_spend_pct_gap_intervention": source_spend_gap_intervention,
            "source_tool_spend_pct_improved": source_spend_improved_value,
            "source_tool_spend_pct_restored": source_spend_restored_value,
            "source_tool_spend_pct_backfired": source_spend_backfire_value,
            "source_tool_spend_pct_normalized_restoration": source_spend_normalized_restoration,
            "source_generated_token_count_gap_baseline": source_generated_gap_baseline,
            "source_generated_token_count_gap_intervention": source_generated_gap_intervention,
            "source_generated_token_count_improved": source_generated_improved_value,
            "source_generated_token_count_restored": source_generated_restored_value,
            "source_generated_token_count_backfired": source_generated_backfire_value,
            "source_generated_token_count_normalized_restoration": source_generated_normalized_restoration,
        }
        rows.append(row_payload)
        family_variant_rows.setdefault(str(row_payload["family_variant"]), []).append(row_payload)
        if pair_mode:
            pair_mode_rows.setdefault(pair_mode, []).append(row_payload)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), config.output_dir / "metadata.parquet")

    family_variant_summary: dict[str, dict[str, Any]] = {}
    for family_variant, bucket in sorted(family_variant_rows.items()):
        family_variant_summary[family_variant] = _summarize_bucket(bucket)

    pair_mode_summary: dict[str, dict[str, Any]] = {}
    for pair_mode, bucket in sorted(pair_mode_rows.items()):
        pair_mode_summary[pair_mode] = _summarize_bucket(bucket)

    result = {
        "baseline_dir": str(config.baseline_dir),
        "intervention_dir": str(config.intervention_dir),
        "output_dir": str(config.output_dir),
        "count": len(common_ids),
        "first_token_change_rate": float(np.mean(first_token_flags)),
        "first_token_change_rate_ci95": _bootstrap_interval(
            first_token_flags,
            reducer=lambda arr: float(np.mean(arr)),
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed,
        ),
        "text_change_rate": float(np.mean(text_flags)),
        "text_change_rate_ci95": _bootstrap_interval(
            text_flags,
            reducer=lambda arr: float(np.mean(arr)),
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed + 1,
        ),
        "tool_presence_change_rate": float(np.mean(tool_presence_flags)),
        "tool_presence_change_rate_ci95": _bootstrap_interval(
            tool_presence_flags,
            reducer=lambda arr: float(np.mean(arr)),
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed + 2,
        ),
        "tool_name_change_rate": float(np.mean(tool_name_flags)),
        "tool_name_change_rate_ci95": _bootstrap_interval(
            tool_name_flags,
            reducer=lambda arr: float(np.mean(arr)),
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed + 3,
        ),
        "tool_token_change_rate": float(np.mean(tool_token_flags)),
        "tool_token_change_rate_ci95": _bootstrap_interval(
            tool_token_flags,
            reducer=lambda arr: float(np.mean(arr)),
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed + 4,
        ),
        "mean_generated_token_count_delta": float(np.mean(generated_token_deltas)),
        "mean_generated_token_count_delta_ci95": _bootstrap_interval(
            generated_token_deltas,
            reducer=lambda arr: float(np.mean(arr)),
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed + 5,
        ),
        "median_generated_token_count_delta": float(np.median(generated_token_deltas)),
        "mean_tool_spend_pct_delta": float(np.mean(spend_deltas)) if spend_deltas else None,
        "mean_tool_spend_pct_delta_ci95": (
            _bootstrap_interval(
                spend_deltas,
                reducer=lambda arr: float(np.mean(arr)),
                samples=config.bootstrap_samples,
                seed=config.bootstrap_seed + 6,
            )
            if spend_deltas
            else None
        ),
        "median_tool_spend_pct_delta": float(np.median(spend_deltas)) if spend_deltas else None,
        "patch_applied_rate": float(np.mean(patch_applied_flags)) if patch_applied_flags else None,
        "patch_skipped_rate": float(np.mean(patch_skipped_flags)) if patch_skipped_flags else None,
        "rows_with_patch_stats": int(sum(bool(_parse_patch_stats(intervention_rows[log_id].get("patch_stats_json"))) for log_id in common_ids)),
        "mean_patch_delta_norm_std": float(np.mean(delta_norm_std_values)) if delta_norm_std_values else None,
        "mean_selected_proj_norm_before": (
            float(np.mean(selected_proj_norm_values)) if selected_proj_norm_values else None
        ),
        "mean_patch_mean_norm_ratio": float(np.mean(mean_norm_ratio_values)) if mean_norm_ratio_values else None,
        "mean_patch_mean_std_norm_ratio": (
            float(np.mean(mean_std_norm_ratio_values)) if mean_std_norm_ratio_values else None
        ),
        "paired_row_count": int(sum(1 for row in rows if row.get("pair_mode"))),
        "pair_modes_present": sorted(pair_mode_summary),
        "mean_pair_metric_gap": _mean_or_none(pair_metric_gap_values),
        "source_tool_name_match_rate_baseline": (
            float(np.mean(source_tool_name_match_baseline)) if source_tool_name_match_baseline else None
        ),
        "source_tool_name_match_rate_intervention": (
            float(np.mean(source_tool_name_match_intervention)) if source_tool_name_match_intervention else None
        ),
        "source_tool_name_match_rate_delta": (
            float(np.mean(source_tool_name_match_intervention) - np.mean(source_tool_name_match_baseline))
            if source_tool_name_match_baseline and source_tool_name_match_intervention
            else None
        ),
        "source_tool_name_restorable_count": len(source_tool_name_restoration_flags),
        "source_tool_name_restoration_rate": _mean_or_none(source_tool_name_restoration_flags),
        "source_tool_name_restoration_rate_ci95": (
            _bootstrap_interval(
                source_tool_name_restoration_flags,
                reducer=lambda arr: float(np.mean(arr)),
                samples=config.bootstrap_samples,
                seed=config.bootstrap_seed + 7,
            )
            if source_tool_name_restoration_flags
            else None
        ),
        "source_tool_name_backfire_rate": _mean_or_none(source_tool_name_backfire_flags),
        "source_tool_token_match_rate_baseline": (
            float(np.mean(source_tool_token_match_baseline)) if source_tool_token_match_baseline else None
        ),
        "source_tool_token_match_rate_intervention": (
            float(np.mean(source_tool_token_match_intervention)) if source_tool_token_match_intervention else None
        ),
        "source_tool_token_match_rate_delta": (
            float(np.mean(source_tool_token_match_intervention) - np.mean(source_tool_token_match_baseline))
            if source_tool_token_match_baseline and source_tool_token_match_intervention
            else None
        ),
        "source_tool_token_restorable_count": len(source_tool_token_restoration_flags),
        "source_tool_token_restoration_rate": _mean_or_none(source_tool_token_restoration_flags),
        "source_tool_token_restoration_rate_ci95": (
            _bootstrap_interval(
                source_tool_token_restoration_flags,
                reducer=lambda arr: float(np.mean(arr)),
                samples=config.bootstrap_samples,
                seed=config.bootstrap_seed + 8,
            )
            if source_tool_token_restoration_flags
            else None
        ),
        "source_tool_token_backfire_rate": _mean_or_none(source_tool_token_backfire_flags),
        "mean_source_tool_spend_pct_delta_baseline": (
            float(np.mean(source_spend_delta_baseline)) if source_spend_delta_baseline else None
        ),
        "mean_source_tool_spend_pct_delta_intervention": (
            float(np.mean(source_spend_delta_intervention)) if source_spend_delta_intervention else None
        ),
        "source_tool_spend_pct_improvement_rate": _mean_or_none(source_spend_improvement_flags),
        "source_tool_spend_pct_full_restoration_rate": _mean_or_none(source_spend_full_restoration_flags),
        "source_tool_spend_pct_backfire_rate": _mean_or_none(source_spend_backfire_flags),
        "mean_source_tool_spend_pct_normalized_restoration": _mean_or_none(
            source_spend_normalized_restoration_values
        ),
        "mean_source_tool_spend_pct_normalized_restoration_ci95": (
            _bootstrap_interval(
                source_spend_normalized_restoration_values,
                reducer=lambda arr: float(np.mean(arr)),
                samples=config.bootstrap_samples,
                seed=config.bootstrap_seed + 9,
            )
            if source_spend_normalized_restoration_values
            else None
        ),
        "median_source_tool_spend_pct_normalized_restoration": _median_or_none(
            source_spend_normalized_restoration_values
        ),
        "mean_source_generated_token_count_delta_baseline": _mean_or_none(source_generated_token_delta_baseline),
        "mean_source_generated_token_count_delta_intervention": _mean_or_none(
            source_generated_token_delta_intervention
        ),
        "source_generated_token_count_improvement_rate": _mean_or_none(source_generated_token_improvement_flags),
        "source_generated_token_count_full_restoration_rate": _mean_or_none(
            source_generated_token_full_restoration_flags
        ),
        "source_generated_token_count_backfire_rate": _mean_or_none(source_generated_token_backfire_flags),
        "mean_source_generated_token_count_normalized_restoration": _mean_or_none(
            source_generated_token_normalized_restoration_values
        ),
        "mean_source_generated_token_count_normalized_restoration_ci95": (
            _bootstrap_interval(
                source_generated_token_normalized_restoration_values,
                reducer=lambda arr: float(np.mean(arr)),
                samples=config.bootstrap_samples,
                seed=config.bootstrap_seed + 10,
            )
            if source_generated_token_normalized_restoration_values
            else None
        ),
        "median_source_generated_token_count_normalized_restoration": _median_or_none(
            source_generated_token_normalized_restoration_values
        ),
        "pair_mode_summary": pair_mode_summary,
        "family_variant_summary": family_variant_summary,
    }
    (config.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    return result
