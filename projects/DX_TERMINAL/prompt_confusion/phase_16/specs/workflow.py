from __future__ import annotations

"""Phase 16 split audit for Phase 13 real DX Terminal rows."""

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines_v2.api import (
    Dataset,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeStore,
    PostgresSource,
    ReportSpec,
    StepRef,
    TokenPooling,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)
from pipelines_v2.operations.execution.common import feature_matrices
from pipelines_v2.operations.execution.common import reference_example_keys, resolve_values_map
from projects.DX_TERMINAL.prompt_confusion.catalogs import build_prompt_confusion_catalog
from projects.DX_TERMINAL.prompt_confusion.phase_13.specs.workflow import CAPTURED_LAYERS
from projects.DX_TERMINAL.prompt_confusion.phase_15.specs.workflow import (
    BANK_SITES,
    DEFAULT_PHASE14_ARTIFACT_ROOT,
    DEFAULT_REAL_CAPTURE_ARTIFACT_ID,
    DEFAULT_REAL_CAPTURE_ARTIFACT_ROOT,
    DIRECTION_NAMES,
    MATCHED_SITE_MAP,
    REAL_POSITIONS,
    _build_phase14_direction_bank,
    _direction_for_layer,
    _load_capture_artifact,
)


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
DEFAULT_PHASE13_TABLE = os.environ.get(
    "PHASE16_PHASE13_TABLE",
    "dx_terminal_signal_discovery_phase13_v1",
)
DEFAULT_REPORT_DIR = "projects/DX_TERMINAL/prompt_confusion/phase_16/reports/split_audit"

LABEL_COLUMNS = (
    "row_example_id",
    "source_table",
    "source_example_id",
    "trace_id",
    "vault_address",
    "person_id",
    "stratum",
    "stratum_detail",
    "prompt_tier",
    "prompt_text",
    "prompt_char_count",
    "label",
    "fault",
    "root_cause",
    "agent_was_correct",
    "severity",
    "confidence",
    "urgency",
    "complaint_type",
    "complaint_text",
    "has_strategy",
    "slider_ta",
    "slider_arp",
    "slider_ts",
    "slider_hs",
    "slider_div",
    "tool",
    "size_relevant_complaint",
    "activity_relevant_complaint",
    "config_conflict_like",
    "system_fault",
    "adapter_alignment_label",
    "strategy_size_preference",
    "slider_size_bucket",
    "target_dimension",
    "synthetic_conflict_present",
    "size_allocation_risk_candidate",
)

HIGH_CONFIDENCE_THRESHOLD = 0.85
FOCUS_BUCKET_TRANSFER_CELLS = {
    (32, "settings_end", "settings_end", "shared_mean"),
    (36, "settings_end", "settings_end", "shared_mean"),
    (32, "market_end", "market_end", "shared_mean"),
    (36, "market_end", "market_end", "shared_mean"),
    (40, "market_end", "market_end", "shared_mean"),
    (44, "market_end", "market_end", "shared_mean"),
    (28, "portfolio_end", "portfolio_end", "shared_mean"),
    (36, "portfolio_end", "portfolio_end", "shared_mean"),
}
ACTION_COMPLAINT_TYPES = {
    "NOT_TRADING",
    "OVERTRADING",
    "STRATEGY_IGNORED",
    "UNWANTED_BUY",
    "UNWANTED_SELL",
    "UNWANTED_HOLD",
    "WRONG_SIZE",
    "HOLDING_VIOLATION",
}
STRICT_SYSTEM_ROOT_CAUSES = {
    "RULE_FABRICATION",
    "PROMPT_FAILURE",
    "OVERTRADING",
    "HOLDING_VIOLATION",
    "CHAT_AI_FABRICATION",
}


def _validate_table_name(value: str) -> str:
    if not value.replace("_", "").isalnum() or not value[0].isalpha():
        raise ValueError(f"Unsafe table name: {value}")
    return value


def build_dataset_sql(table_name: str = DEFAULT_PHASE13_TABLE) -> str:
    table_name = _validate_table_name(table_name)
    return f"""
        SELECT
            example_id AS row_example_id,
            *
        FROM {table_name}
        WHERE prompt_messages_json IS NOT NULL
        ORDER BY stratum, COALESCE(NULLIF(trace_id, ''), source_example_id, example_id), prompt_tier, example_id
    """


def build_dataset(*, limit: int | None = None) -> Dataset:
    dataset = Dataset.from_postgres(
        source=PostgresSource.from_env(DB_ENV_VAR),
        sql=build_dataset_sql(DEFAULT_PHASE13_TABLE),
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        label_columns=LABEL_COLUMNS,
        case_columns=("source_table", "source_example_id", "trace_id", "vault_address"),
        case_key_column="source_example_id",
        limit=limit,
        name="dx_terminal_phase16_phase13_split_audit",
    )
    return dataset


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "t", "true", "yes", "y"}


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dimension(row: Mapping[str, Any]) -> str:
    complaint_type = _string(row.get("complaint_type"))
    if _bool(row.get("size_relevant_complaint")) or complaint_type == "WRONG_SIZE":
        return "trade_size"
    if complaint_type == "HOLDING_VIOLATION":
        return "holding"
    if complaint_type in {"NOT_TRADING", "OVERTRADING", "STRATEGY_IGNORED", "UNWANTED_BUY", "UNWANTED_SELL", "UNWANTED_HOLD"}:
        return "strategy_lifecycle_activity"
    if complaint_type == "GENERAL_PERFORMANCE":
        return "general_performance"
    return "unknown"


def _action_polarity(row: Mapping[str, Any]) -> str:
    complaint_type = _string(row.get("complaint_type"))
    if complaint_type in ACTION_COMPLAINT_TYPES:
        return complaint_type.lower()
    if complaint_type:
        return complaint_type.lower()
    return "none"


def _canonical_case_id(row: Mapping[str, Any], example_key: str) -> str:
    stratum = _string(row.get("stratum"))
    if stratum == "complaint":
        trace_id = _string(row.get("trace_id"))
        return f"trace:{trace_id}" if trace_id else f"row:{example_key}"
    source_example_id = _string(row.get("source_example_id"))
    if source_example_id:
        return f"source:{source_example_id}"
    return f"row:{example_key}"


def _bucket(row: Mapping[str, Any]) -> str:
    stratum = _string(row.get("stratum"))
    label = _string(row.get("label"))
    root_cause = _string(row.get("root_cause"))
    complaint_type = _string(row.get("complaint_type"))
    confidence = _float(row.get("confidence"))
    config_conflict_like = _bool(row.get("config_conflict_like"))
    system_fault = _bool(row.get("system_fault"))
    alignment = _string(row.get("adapter_alignment_label"))

    if stratum == "structure_matched_control":
        return "synthetic_template_control"

    if stratum in {"anchor_positive", "anchor_positive_buy_only"}:
        if alignment == "aligned" and not system_fault:
            return "anchor_aligned_real"
        if alignment == "conflict" or system_fault or config_conflict_like:
            return "anchor_conflict_like"
        return "review_or_exclude"

    if stratum != "complaint":
        return "review_or_exclude"

    if confidence < HIGH_CONFIDENCE_THRESHOLD:
        return "review_or_exclude"
    if config_conflict_like and system_fault:
        return "ambiguous_mixed"
    if (
        label == "true_confusion"
        and system_fault
        and not config_conflict_like
        and complaint_type in ACTION_COMPLAINT_TYPES
        and root_cause in STRICT_SYSTEM_ROOT_CAUSES
    ):
        return "strict_system_conflict"
    if label == "user_confusion" and config_conflict_like and not system_fault:
        return "user_config_conflict_control"
    return "ambiguous_mixed"


def _row_preview(row: Mapping[str, Any], example_key: str) -> dict[str, Any]:
    return {
        "example_id": example_key,
        "case_id": _string(row.get("phase16_case_id")),
        "bucket": _string(row.get("phase16_bucket")),
        "stratum": _string(row.get("stratum")),
        "prompt_tier": _string(row.get("prompt_tier")),
        "label": _string(row.get("label")),
        "root_cause": _string(row.get("root_cause")),
        "complaint_type": _string(row.get("complaint_type")),
        "confidence": _float(row.get("confidence")),
        "system_fault": _bool(row.get("system_fault")),
        "config_conflict_like": _bool(row.get("config_conflict_like")),
        "dimension": _string(row.get("phase16_dimension")),
        "complaint_text": _string(row.get("complaint_text"))[:500],
    }


def _counter_rows(counter: Counter[tuple[Any, ...]], columns: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], tuple(str(part) for part in item[0]))):
        rows.append({column: key[index] for index, column in enumerate(columns)} | {"count": int(count)})
    return rows


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.asarray(values, dtype=np.float32).mean())


def _best(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    require_matched_site: bool | None = None,
    prompt_tier: str | None = None,
    direction: str | None = None,
    dimension: str | None = None,
) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(key) is not None]
    if require_matched_site is not None:
        candidates = [row for row in candidates if bool(row.get("matched_site")) is require_matched_site]
    if prompt_tier is not None:
        candidates = [row for row in candidates if row.get("prompt_tier") == prompt_tier]
    if direction is not None:
        candidates = [row for row in candidates if row.get("direction") == direction]
    if dimension is not None:
        candidates = [row for row in candidates if row.get("dimension") == dimension]
    if not candidates:
        return None
    return dict(max(candidates, key=lambda row: float(row[key])))


def _top_bottom_cases(
    case_scores: Mapping[str, float],
    case_examples: Mapping[str, Mapping[str, Any]],
    *,
    n: int = 8,
) -> dict[str, Any]:
    ordered = sorted(case_scores.items(), key=lambda item: item[1])

    def record(item: tuple[str, float]) -> dict[str, Any]:
        case_id, score = item
        example = case_examples.get(case_id, {})
        return {
            "case_id": case_id,
            "score": float(score),
            "bucket": _string(example.get("bucket")),
            "dimension": _string(example.get("dimension")),
            "complaint_type": _string(example.get("complaint_type")),
            "root_cause": _string(example.get("root_cause")),
            "complaint_text": _string(example.get("complaint_text"))[:500],
        }

    return {
        "top": [record(item) for item in ordered[-n:][::-1]],
        "bottom": [record(item) for item in ordered[:n]],
    }


def _case_averaged_bucket_transfer_builder(
    direction_bank: Any,
    phase16_bucket: Any,
    phase16_case_id: Any,
    phase16_dimension: Any,
    phase16_action_polarity: Any,
    prompt_tier: Any,
    complaint_text: Any,
    complaint_type: Any,
    root_cause: Any,
    real_capture_artifact_id: str = DEFAULT_REAL_CAPTURE_ARTIFACT_ID,
    real_capture_artifact_root: str = DEFAULT_REAL_CAPTURE_ARTIFACT_ROOT,
) -> TransformResult:
    capture = _load_capture_artifact(
        artifact_id=str(real_capture_artifact_id),
        artifact_root=str(real_capture_artifact_root),
    )
    direction_payload = direction_bank.result() if hasattr(direction_bank, "result") else direction_bank
    if not isinstance(direction_payload, Mapping):
        raise TypeError("direction_bank must resolve to a mapping payload")

    bucket_values = resolve_values_map(phase16_bucket, label="phase16_bucket")
    case_values = resolve_values_map(phase16_case_id, label="phase16_case_id")
    dimension_values = resolve_values_map(phase16_dimension, label="phase16_dimension")
    action_values = resolve_values_map(phase16_action_polarity, label="phase16_action_polarity")
    tier_values = resolve_values_map(prompt_tier, label="prompt_tier")
    complaint_values = resolve_values_map(complaint_text, label="complaint_text")
    complaint_type_values = resolve_values_map(complaint_type, label="complaint_type")
    root_cause_values = resolve_values_map(root_cause, label="root_cause")

    available_features = set(dict(capture.manifest().storage_refs.get("features", {})))
    selected_positions = [
        (position, feature_name)
        for position, feature_name in REAL_POSITIONS.items()
        if feature_name in available_features
    ]
    grid_rows: list[dict[str, Any]] = []
    cells: dict[str, Any] = {}
    bucket_case_counts: Counter[tuple[str, str, str]] = Counter()

    for real_position, feature_name in selected_positions:
        matrices, example_keys = feature_matrices(
            capture.feature(feature_name),
            layers=CAPTURED_LAYERS,
            token_pooling=TokenPooling.mean(),
        )
        ordered_buckets = np.asarray([bucket_values[key] for key in example_keys], dtype=object)
        ordered_cases = np.asarray([case_values[key] for key in example_keys], dtype=object)
        ordered_dimensions = np.asarray([dimension_values[key] for key in example_keys], dtype=object)
        ordered_tiers = np.asarray([tier_values[key] for key in example_keys], dtype=object)

        for bucket, dimension, case_id in zip(
            ordered_buckets.tolist(),
            ordered_dimensions.tolist(),
            ordered_cases.tolist(),
            strict=True,
        ):
            bucket_case_counts[(str(bucket), str(dimension), str(case_id))] += 1

        for layer in CAPTURED_LAYERS:
            X = matrices[int(layer)]
            for bank_site in BANK_SITES:
                matched_site = MATCHED_SITE_MAP.get(bank_site) == real_position
                for direction_name in DIRECTION_NAMES:
                    direction = _direction_for_layer(
                        direction_payload,
                        layer=int(layer),
                        bank_site=bank_site,
                        name=direction_name,
                    )
                    scores = X @ direction
                    for tier in sorted(set(str(value) for value in ordered_tiers.tolist())):
                        tier_indices = [index for index, value in enumerate(ordered_tiers.tolist()) if str(value) == tier]
                        tier_buckets = [str(ordered_buckets[index]) for index in tier_indices]
                        tier_dimensions = [str(ordered_dimensions[index]) for index in tier_indices]
                        tier_cases = [str(ordered_cases[index]) for index in tier_indices]
                        tier_scores = [float(scores[index]) for index in tier_indices]

                        for dimension in ("all", "strategy_lifecycle_activity", "trade_size", "holding"):
                            score_groups: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
                            case_examples: dict[str, Mapping[str, Any]] = {}
                            for local_index, index in enumerate(tier_indices):
                                bucket = tier_buckets[local_index]
                                row_dimension = tier_dimensions[local_index]
                                if dimension != "all" and row_dimension != dimension:
                                    continue
                                case_id = tier_cases[local_index]
                                score_groups[(bucket, case_id)].append(tier_scores[local_index])
                                case_examples.setdefault(
                                    case_id,
                                    {
                                        "bucket": bucket,
                                        "dimension": row_dimension,
                                        "action_polarity": action_values[example_keys[index]],
                                        "complaint_type": complaint_type_values[example_keys[index]],
                                        "root_cause": root_cause_values[example_keys[index]],
                                        "complaint_text": complaint_values[example_keys[index]],
                                    },
                                )

                            case_scores_by_bucket: defaultdict[str, dict[str, float]] = defaultdict(dict)
                            for (bucket, case_id), values in score_groups.items():
                                case_scores_by_bucket[bucket][case_id] = float(np.asarray(values, dtype=np.float32).mean())

                            bucket_means = {
                                bucket: _mean(list(case_scores.values()))
                                for bucket, case_scores in sorted(case_scores_by_bucket.items())
                            }
                            bucket_counts = {
                                bucket: len(case_scores)
                                for bucket, case_scores in sorted(case_scores_by_bucket.items())
                            }
                            strict_mean = bucket_means.get("strict_system_conflict")
                            user_control_mean = bucket_means.get("user_config_conflict_control")
                            synthetic_control_mean = bucket_means.get("synthetic_template_control")
                            anchor_aligned_mean = bucket_means.get("anchor_aligned_real")
                            strict_minus_user_control = (
                                strict_mean - user_control_mean
                                if strict_mean is not None and user_control_mean is not None
                                else None
                            )
                            strict_minus_synthetic_control = (
                                strict_mean - synthetic_control_mean
                                if strict_mean is not None and synthetic_control_mean is not None
                                else None
                            )
                            strict_minus_anchor_aligned = (
                                strict_mean - anchor_aligned_mean
                                if strict_mean is not None and anchor_aligned_mean is not None
                                else None
                            )
                            row = {
                                "layer": int(layer),
                                "bank_site": bank_site,
                                "real_position": real_position,
                                "matched_site": bool(matched_site),
                                "prompt_tier": tier,
                                "direction": direction_name,
                                "dimension": dimension,
                                "bucket_means": bucket_means,
                                "bucket_case_counts": bucket_counts,
                                "strict_minus_user_config_control": strict_minus_user_control,
                                "strict_minus_synthetic_template_control": strict_minus_synthetic_control,
                                "strict_minus_anchor_aligned_real": strict_minus_anchor_aligned,
                            }
                            grid_rows.append(row)
                            focus_key = (int(layer), bank_site, real_position, direction_name)
                            if focus_key in FOCUS_BUCKET_TRANSFER_CELLS and dimension in {"all", "strategy_lifecycle_activity"}:
                                cell_key = f"L{layer}:{bank_site}->{real_position}:{tier}:{direction_name}:{dimension}"
                                cells[cell_key] = {
                                    **row,
                                    "top_bottom_by_bucket": {
                                        bucket: _top_bottom_cases(case_scores, case_examples)
                                        for bucket, case_scores in case_scores_by_bucket.items()
                                        if bucket in {"strict_system_conflict", "user_config_conflict_control"}
                                    },
                                }

    summary = {
        "grid_cell_count": len(grid_rows),
        "real_capture_artifact_id": str(real_capture_artifact_id),
        "real_capture_artifact_root": str(real_capture_artifact_root),
        "case_averaging": "scores are averaged within phase16_case_id before bucket means",
        "best_matched_strict_minus_user_control": _best(
            grid_rows,
            key="strict_minus_user_config_control",
            require_matched_site=True,
            prompt_tier="aggressive",
            dimension="all",
        ),
        "best_matched_shared_mean_strict_minus_user_control": _best(
            grid_rows,
            key="strict_minus_user_config_control",
            require_matched_site=True,
            prompt_tier="aggressive",
            direction="shared_mean",
            dimension="all",
        ),
        "best_matched_strategy_lifecycle_shared_mean": _best(
            grid_rows,
            key="strict_minus_user_config_control",
            require_matched_site=True,
            prompt_tier="aggressive",
            direction="shared_mean",
            dimension="strategy_lifecycle_activity",
        ),
        "best_matched_trade_size_shared_mean": _best(
            grid_rows,
            key="strict_minus_user_config_control",
            require_matched_site=True,
            prompt_tier="aggressive",
            direction="shared_mean",
            dimension="trade_size",
        ),
    }
    return TransformResult(
        payload={
            "kind": "phase16_case_averaged_bucket_transfer",
            "layers": list(CAPTURED_LAYERS),
            "bank_sites": list(BANK_SITES),
            "real_positions": [position for position, _ in selected_positions],
            "directions": list(DIRECTION_NAMES),
            "summary": summary,
            "grid_rows": grid_rows,
            "cells": cells,
        },
    )


def _phase13_split_audit_builder(**labels: Any) -> TransformResult:
    example_keys = reference_example_keys(labels["row_example_id"], label="row_example_id")
    values = {name: resolve_values_map(source, label=name) for name, source in labels.items()}

    rows: list[dict[str, Any]] = []
    bucket_by_key: dict[str, str] = {}
    case_id_by_key: dict[str, str] = {}
    dimension_by_key: dict[str, str] = {}
    action_polarity_by_key: dict[str, str] = {}
    review_reason_by_key: dict[str, str] = {}
    for key in example_keys:
        row = {name: values[name].get(key) for name in values}
        case_id = _canonical_case_id(row, key)
        bucket = _bucket(row)
        dimension = _dimension(row)
        action_polarity = _action_polarity(row)
        review_reason = ""
        if bucket in {"ambiguous_mixed", "review_or_exclude"}:
            if _float(row.get("confidence")) < HIGH_CONFIDENCE_THRESHOLD:
                review_reason = "low_confidence"
            elif _bool(row.get("config_conflict_like")) and _bool(row.get("system_fault")):
                review_reason = "both_config_conflict_and_system_fault"
            elif _string(row.get("stratum")) == "complaint":
                review_reason = "complaint_not_strictly_classified"
            else:
                review_reason = "non_complaint_unclean"

        enriched = dict(row)
        enriched["phase16_case_id"] = case_id
        enriched["phase16_bucket"] = bucket
        enriched["phase16_dimension"] = dimension
        enriched["phase16_action_polarity"] = action_polarity
        enriched["phase16_review_reason"] = review_reason
        rows.append(enriched | {"example_key": key})
        bucket_by_key[key] = bucket
        case_id_by_key[key] = case_id
        dimension_by_key[key] = dimension
        action_polarity_by_key[key] = action_polarity
        review_reason_by_key[key] = review_reason

    by_bucket_tier: Counter[tuple[Any, ...]] = Counter()
    by_bucket_dimension_tier: Counter[tuple[Any, ...]] = Counter()
    by_stratum_bucket: Counter[tuple[Any, ...]] = Counter()
    by_bucket_root_cause_type: Counter[tuple[Any, ...]] = Counter()
    by_bucket_case: defaultdict[str, set[str]] = defaultdict(set)
    by_bucket_dimension_case: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    case_bucket_members: defaultdict[str, set[str]] = defaultdict(set)
    case_rows: defaultdict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)

    for row in rows:
        bucket = _string(row["phase16_bucket"])
        tier = _string(row.get("prompt_tier"))
        dimension = _string(row["phase16_dimension"])
        case_id = _string(row["phase16_case_id"])
        by_bucket_tier[(bucket, tier)] += 1
        by_bucket_dimension_tier[(bucket, dimension, tier)] += 1
        by_stratum_bucket[(_string(row.get("stratum")), bucket)] += 1
        by_bucket_root_cause_type[(bucket, _string(row.get("root_cause")), _string(row.get("complaint_type")))] += 1
        by_bucket_case[bucket].add(case_id)
        by_bucket_dimension_case[(bucket, dimension)].add(case_id)
        case_bucket_members[case_id].add(bucket)
        case_rows[case_id].append((_string(row["example_key"]), row))

    unique_case_rows = [
        {"bucket": bucket, "unique_cases": len(case_ids)}
        for bucket, case_ids in sorted(by_bucket_case.items(), key=lambda item: (-len(item[1]), item[0]))
    ]
    unique_case_dimension_rows = [
        {"bucket": bucket, "dimension": dimension, "unique_cases": len(case_ids)}
        for (bucket, dimension), case_ids in sorted(
            by_bucket_dimension_case.items(),
            key=lambda item: (-len(item[1]), item[0][0], item[0][1]),
        )
    ]

    mixed_case_ids = sorted(case_id for case_id, buckets in case_bucket_members.items() if len(buckets) > 1)
    review_examples = [
        _row_preview(row, key)
        for key, row in sorted(
            [
                (key, row)
                for row in rows
                for key in [_string(row["example_key"])]
                if row["phase16_bucket"] in {"ambiguous_mixed", "review_or_exclude"}
            ],
            key=lambda item: (
                _string(item[1].get("phase16_review_reason")),
                -_float(item[1].get("confidence")),
                _string(item[1].get("complaint_type")),
            ),
        )[:30]
    ]
    strict_examples = [
        _row_preview(row, key)
        for key, row in sorted(
            [
                (key, row)
                for row in rows
                for key in [_string(row["example_key"])]
                if row["phase16_bucket"] == "strict_system_conflict"
            ],
            key=lambda item: (
                _string(item[1].get("phase16_dimension")),
                _string(item[1].get("complaint_type")),
                _string(item[1].get("root_cause")),
            ),
        )[:30]
    ]
    user_control_examples = [
        _row_preview(row, key)
        for key, row in sorted(
            [
                (key, row)
                for row in rows
                for key in [_string(row["example_key"])]
                if row["phase16_bucket"] == "user_config_conflict_control"
            ],
            key=lambda item: (
                _string(item[1].get("phase16_dimension")),
                _string(item[1].get("complaint_type")),
                _string(item[1].get("root_cause")),
            ),
        )[:30]
    ]

    summary = {
        "row_count": len(rows),
        "unique_case_count": len(case_rows),
        "bucket_unique_cases": unique_case_rows,
        "mixed_case_count": len(mixed_case_ids),
        "strict_system_conflict_unique_cases": len(by_bucket_case.get("strict_system_conflict", set())),
        "user_config_conflict_control_unique_cases": len(by_bucket_case.get("user_config_conflict_control", set())),
        "ambiguous_or_review_unique_cases": len(by_bucket_case.get("ambiguous_mixed", set()))
        + len(by_bucket_case.get("review_or_exclude", set())),
        "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
    }

    return TransformResult(
        payload={
            "kind": "phase16_phase13_split_audit",
            "source_table": DEFAULT_PHASE13_TABLE,
            "summary": summary,
            "tables": {
                "by_bucket_tier": _counter_rows(by_bucket_tier, ("bucket", "prompt_tier")),
                "by_bucket_dimension_tier": _counter_rows(
                    by_bucket_dimension_tier,
                    ("bucket", "dimension", "prompt_tier"),
                ),
                "by_stratum_bucket": _counter_rows(by_stratum_bucket, ("stratum", "bucket")),
                "by_bucket_root_cause_type": _counter_rows(
                    by_bucket_root_cause_type,
                    ("bucket", "root_cause", "complaint_type"),
                ),
                "bucket_unique_cases": unique_case_rows,
                "bucket_dimension_unique_cases": unique_case_dimension_rows,
            },
            "examples": {
                "strict_system_conflict": strict_examples,
                "user_config_conflict_control": user_control_examples,
                "review": review_examples,
            },
            "mixed_case_ids": mixed_case_ids[:100],
        },
        labels={
            "phase16_case_id": {"kind": "label", "values": case_id_by_key},
            "phase16_bucket": {"kind": "label", "values": bucket_by_key},
            "phase16_dimension": {"kind": "label", "values": dimension_by_key},
            "phase16_action_polarity": {"kind": "label", "values": action_polarity_by_key},
            "phase16_review_reason": {"kind": "label", "values": review_reason_by_key},
        },
    )


def build_runner_specs() -> dict[str, object]:
    db_secret = ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon")
    artifact_store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/prompt_confusion_phase16_split_audit",
    )
    return {
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(secrets=(db_secret,)),
            artifacts=artifact_store,
            catalog=build_prompt_confusion_catalog(__file__),
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(Path("artifacts") / "prompt_confusion_phase16_split_audit"),
            catalog=build_prompt_confusion_catalog(__file__),
        ),
    }


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    split_audit_builder = TransformBuilder.from_function(
        _phase13_split_audit_builder,
        local_python_sources=("projects",),
    )
    direction_bank_builder = TransformBuilder.from_function(
        _build_phase14_direction_bank,
        local_python_sources=("projects",),
    )
    bucket_transfer_builder = TransformBuilder.from_function(
        _case_averaged_bucket_transfer_builder,
        local_python_sources=("projects",),
    )
    return WorkflowSpec(
        name="dx_terminal_phase16_phase13_split_audit",
        steps=(
            WorkflowStep(
                name="phase13_split_audit",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=split_audit_builder,
                    inputs={name: dataset.labels(name) for name in LABEL_COLUMNS},
                ),
            ),
            WorkflowStep(
                name="build_phase14_direction_bank",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=direction_bank_builder,
                    inputs={"phase14_artifact_root": DEFAULT_PHASE14_ARTIFACT_ROOT},
                ),
            ),
            WorkflowStep(
                name="case_averaged_bucket_transfer",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=bucket_transfer_builder,
                    inputs={
                        "direction_bank": StepRef("build_phase14_direction_bank"),
                        "phase16_bucket": StepRef("phase13_split_audit").label("phase16_bucket"),
                        "phase16_case_id": StepRef("phase13_split_audit").label("phase16_case_id"),
                        "phase16_dimension": StepRef("phase13_split_audit").label("phase16_dimension"),
                        "phase16_action_polarity": StepRef("phase13_split_audit").label("phase16_action_polarity"),
                        "prompt_tier": dataset.labels("prompt_tier"),
                        "complaint_text": dataset.labels("complaint_text"),
                        "complaint_type": dataset.labels("complaint_type"),
                        "root_cause": dataset.labels("root_cause"),
                        "real_capture_artifact_id": DEFAULT_REAL_CAPTURE_ARTIFACT_ID,
                        "real_capture_artifact_root": DEFAULT_REAL_CAPTURE_ARTIFACT_ROOT,
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(StepRef("phase13_split_audit"), StepRef("case_averaged_bucket_transfer")),
                    template="default",
                    output_dir=DEFAULT_REPORT_DIR,
                ),
            ),
        ),
    )
