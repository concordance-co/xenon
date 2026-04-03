from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from pipelines.db import connect_neon


VISIBLE_METRIC_NAMES = (
    "pct_5m",
    "pct_1h",
    "net_flow_5m",
    "vol_5m",
    "vol_1h",
    "unique_traders_5m",
    "top20_holder_pct",
)


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _safe_range(values: list[float]) -> float:
    return float(max(values) - min(values)) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _leader_gap(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values, reverse=True)
    return float(ordered[0] - ordered[1])


def _top2_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    return float(np.mean(ordered[: min(2, len(ordered))]))


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    center = float(np.mean(values))
    return float(np.mean(np.abs(np.asarray(values, dtype=np.float32) - center)))


def _max_minus_rest_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values, reverse=True)
    return float(ordered[0] - np.mean(ordered[1:]))


def _top1_minus_median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    return float(ordered[0] - np.median(ordered))


def _leader_zscore(values: list[float]) -> float:
    if not values:
        return 0.0
    std = float(np.std(values))
    if std == 0.0:
        return 0.0
    return float((max(values) - np.mean(values)) / std)


def _cv_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    denom = float(np.mean(np.abs(values)))
    if denom == 0.0:
        return 0.0
    return float(np.std(values) / denom)


def _aggregate_metric_family(values: list[float]) -> dict[str, float]:
    return {
        "mean": _safe_mean(values),
        "std": _safe_std(values),
        "max": float(max(values)) if values else 0.0,
        "min": float(min(values)) if values else 0.0,
        "range": _safe_range(values),
        "gap": _leader_gap(values),
        "mad": _mad(values),
        "median": _safe_median(values),
        "top2_mean": _top2_mean(values),
        "max_minus_rest_mean": _max_minus_rest_mean(values),
        "top1_minus_median": _top1_minus_median(values),
        "leader_zscore": _leader_zscore(values),
        "cv_abs": _cv_abs(values),
    }


def _group_asset_rows(asset_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in asset_rows:
        grouped.setdefault(int(row["log_id"]), []).append(dict(row))
    for log_id in grouped:
        grouped[log_id].sort(key=lambda row: int(row["row_index"]))
    return grouped


def load_prompt_visible_metric_map(
    *,
    phase_name: str,
    log_ids: list[int],
) -> dict[int, dict[str, float]]:
    if not log_ids:
        return {}
    conn = connect_neon()
    try:
        rows = conn.execute(
            """
            SELECT
                log_id,
                row_index,
                pct_5m,
                pct_1h,
                net_flow_5m,
                vol_5m,
                vol_1h,
                unique_traders_5m,
                top20_holder_pct
            FROM synthetic_market_assets_v0
            WHERE phase_name = %s AND log_id = ANY(%s)
            ORDER BY log_id, row_index
            """,
            (phase_name, log_ids),
        ).fetchall()
    finally:
        conn.close()

    asset_by_log = _group_asset_rows([dict(row) for row in rows])
    metric_map: dict[int, dict[str, float]] = {}
    for log_id, asset_list in asset_by_log.items():
        feature_row: dict[str, float] = {}
        for metric_name in VISIBLE_METRIC_NAMES:
            values = [float(row[metric_name]) for row in asset_list]
            for aggregate_name, aggregate_value in _aggregate_metric_family(values).items():
                feature_row[f"{metric_name}_{aggregate_name}"] = float(aggregate_value)
        metric_map[int(log_id)] = feature_row
    return metric_map


def build_matched_metric_examples(
    rows: list[dict[str, Any]],
    *,
    phase_name: str,
    pair_metric: str,
    pair_mode: str,
    min_metric_gap: float = 0.0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    metric_name = str(pair_metric).strip()
    if not metric_name:
        return rows
    mode = str(pair_mode).strip().lower()
    if mode not in {"denoise", "noise"}:
        raise ValueError(f"Unsupported pair_mode={pair_mode!r}; expected 'denoise' or 'noise'")

    metric_map = load_prompt_visible_metric_map(
        phase_name=phase_name,
        log_ids=[int(row["log_id"]) for row in rows],
    )

    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    bucket_order: list[tuple[str, str, str, str]] = []
    for row in rows:
        feature_row = metric_map.get(int(row["log_id"]))
        if feature_row is None or metric_name not in feature_row:
            continue
        enriched = dict(row)
        enriched["pair_metric_name"] = metric_name
        enriched["pair_metric_value"] = float(feature_row[metric_name])
        key = (
            str(row.get("family", "")),
            str(row.get("family_variant", "")),
            str(row.get("roster_key", "")),
            str(row.get("context_variant", "")),
        )
        if key not in buckets:
            bucket_order.append(key)
        buckets[key].append(enriched)

    selected: list[dict[str, Any]] = []
    pair_counter = 0
    pair_buckets: list[list[dict[str, Any]]] = []
    for key in bucket_order:
        bucket = sorted(
            buckets[key],
            key=lambda row: (float(row["pair_metric_value"]), int(row["log_id"])),
        )
        if len(bucket) < 2:
            continue
        local_pairs: list[dict[str, Any]] = []
        lo = 0
        hi = len(bucket) - 1
        while lo < hi:
            low = bucket[lo]
            high = bucket[hi]
            gap = float(high["pair_metric_value"]) - float(low["pair_metric_value"])
            lo += 1
            hi -= 1
            if gap < float(min_metric_gap):
                continue
            if mode == "denoise":
                base_row, source_row = low, high
            else:
                base_row, source_row = high, low
            local_pairs.append(
                {
                    **dict(base_row),
                    "pair_id": pair_counter,
                    "pair_mode": mode,
                    "source_log_id": int(source_row["log_id"]),
                    "source_example_id": str(source_row["example_id"]),
                    "source_family": str(source_row.get("family", "")),
                    "source_family_variant": str(source_row.get("family_variant", "")),
                    "source_roster_key": str(source_row.get("roster_key", "")),
                    "source_context_variant": str(source_row.get("context_variant", "")),
                    "source_prompt_messages_json": source_row["prompt_messages_json"],
                    "source_pair_metric_value": float(source_row["pair_metric_value"]),
                    "base_pair_metric_value": float(base_row["pair_metric_value"]),
                    "pair_metric_gap": float(gap),
                }
            )
            pair_counter += 1
        if local_pairs:
            pair_buckets.append(local_pairs)

    if not pair_buckets:
        return []

    indices = [0 for _ in pair_buckets]
    while True:
        progressed = False
        for bucket_idx, bucket in enumerate(pair_buckets):
            if limit is not None and limit > 0 and len(selected) >= limit:
                return selected
            idx = indices[bucket_idx]
            if idx >= len(bucket):
                continue
            selected.append(bucket[idx])
            indices[bucket_idx] += 1
            progressed = True
        if not progressed:
            break
    return selected
