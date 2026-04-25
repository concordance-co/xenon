from __future__ import annotations

"""Phase 13 real DX Terminal signal-discovery workflow."""

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines_v2.api import (
    ArtifactManifest,
    CaptureSpec,
    CaptureArtifact,
    Dataset,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PostgresSource,
    PromptMetadataBuilder,
    ReportSpec,
    ResidualSite,
    StepRef,
    TensorStorage,
    TokenPooling,
    TokenSelector,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)
from pipelines_v2.operations.execution.common import feature_matrices, ordered_values
from projects.DX_TERMINAL.prompt_confusion.catalogs import build_prompt_confusion_catalog


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
DEFAULT_CORPUS_TABLE = os.environ.get(
    "PHASE13_SIGNAL_DISCOVERY_TABLE",
    "dx_terminal_signal_discovery_phase13_v1",
)
DEFAULT_PHASE12_DIRECTION_CAPTURE_ARTIFACT_ID = os.environ.get(
    "PHASE13_PHASE12_DIRECTION_CAPTURE_ARTIFACT_ID",
    "capture_1_8b1f70f5",
)
DEFAULT_PHASE12_DIRECTION_ARTIFACT_ROOT = os.environ.get(
    "PHASE13_PHASE12_DIRECTION_ARTIFACT_ROOT",
    "/data/artifacts/prompt_confusion_three_family_geometry",
)
DEFAULT_REPORT_DIR = "projects/DX_TERMINAL/prompt_confusion/phase_13/reports/signal_discovery"
CAPTURED_LAYERS = (12, 16, 20, 24, 28, 32, 36, 40, 44)
DIRECTION_FAMILIES = ("trade_size", "risk_preference", "diversification_preference")
DIRECTION_NAMES = DIRECTION_FAMILIES + ("shared_mean",)
STRUCTURAL_SECTIONS = ("system", "strategies", "settings", "portfolio", "market")
POSITION_NAMES = (
    "system_end",
    "strategies_end",
    "settings_end",
    "portfolio_end",
    "market_end",
    "system_mean",
    "strategies_mean",
    "settings_mean",
    "portfolio_mean",
    "market_mean",
    "prompt_im_end",
    "full_sequence_max",
)

_ROLE_MARKERS = {
    "system": "<|im_start|>system\n",
    "user": "<|im_start|>user\n",
}
_IM_END = "<|im_end|>"
_SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "strategies": re.compile(r"^\s*(active\s+strategies|strategies?|strategy)\b.*$", re.IGNORECASE | re.MULTILINE),
    "settings": re.compile(
        r"^\s*(active\s+settings|settings?|configuration|risk settings|execution policy)\b.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "portfolio": re.compile(r"^\s*(portfolio|portfolio context|holdings|positions)\b.*$", re.IGNORECASE | re.MULTILINE),
    "market": re.compile(r"^\s*(market|market snapshot|market context|prices?|tokens?)\b.*$", re.IGNORECASE | re.MULTILINE),
}
_ANY_HEADING = re.compile(r"^\s*[A-Z][A-Z0-9 _/-]{2,}:?\s*$", re.MULTILINE)


def _dataset_limit_from_env() -> int | None:
    raw = os.environ.get("PHASE13_SIGNAL_DISCOVERY_LIMIT")
    if raw is None or not raw.strip():
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError("PHASE13_SIGNAL_DISCOVERY_LIMIT must be positive")
    return value


def _dataset_tiers_from_env() -> tuple[str, ...] | None:
    raw = os.environ.get("PHASE13_SIGNAL_DISCOVERY_TIERS")
    if raw is None or not raw.strip():
        return None
    tiers = tuple(part.strip() for part in raw.split(",") if part.strip())
    allowed = {"full", "light", "aggressive"}
    invalid = sorted(set(tiers) - allowed)
    if invalid:
        raise ValueError(f"Unsupported PHASE13_SIGNAL_DISCOVERY_TIERS values: {invalid}")
    return tiers or None


def _capture_site_set_from_env() -> str:
    raw = os.environ.get("PHASE13_CAPTURE_SITE_SET")
    value = "full" if raw is None or not raw.strip() else raw.strip().lower()
    if value not in {"full", "ends_only"}:
        raise ValueError("PHASE13_CAPTURE_SITE_SET must be one of {'full', 'ends_only'}")
    return value


def _validate_table_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe table name: {value}")
    return value


def build_dataset_sql(table_name: str = DEFAULT_CORPUS_TABLE, *, tiers: Sequence[str] | None = None) -> str:
    table_name = _validate_table_name(table_name)
    filters = ["prompt_messages_json IS NOT NULL"]
    if tiers:
        quoted = ", ".join("'" + tier.replace("'", "''") + "'" for tier in tiers)
        filters.append(f"prompt_tier IN ({quoted})")
    where_sql = "\n          AND ".join(filters)
    return f"""
        SELECT *
        FROM {table_name}
        WHERE {where_sql}
        ORDER BY stratum, source_table, source_example_id, prompt_tier
    """


def build_dataset(*, limit: int | None = None) -> Dataset:
    final_limit = _dataset_limit_from_env() if limit is None else limit
    return Dataset.from_postgres(
        source=PostgresSource.from_env(DB_ENV_VAR),
        sql=build_dataset_sql(DEFAULT_CORPUS_TABLE, tiers=_dataset_tiers_from_env()),
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        label_columns=[
            "source_table",
            "source_example_id",
            "trace_id",
            "vault_address",
            "person_id",
            "stratum",
            "stratum_detail",
            "prompt_tier",
            "prompt_text",
            "prompt_message_count",
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
            "tick_index",
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
        ],
        case_columns=["source_table", "source_example_id", "vault_address"],
        case_key_column="source_example_id",
        limit=final_limit,
        name="dx_terminal_phase13_signal_discovery",
    )


def _default_residual_engine() -> VLLMEngine:
    raw_max_num_seqs = os.environ.get("PHASE13_MAX_NUM_SEQS")
    max_num_seqs = 4 if raw_max_num_seqs is None or not raw_max_num_seqs.strip() else int(raw_max_num_seqs)
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=40960,
        enforce_eager=False,
        max_num_seqs=max_num_seqs,
    )


def _trim_end(text: str, *, start: int, end: int) -> int:
    while end > start and text[end - 1].isspace():
        end -= 1
    return max(start, end)


def _last_char_span(text: str, *, start: int, end: int) -> tuple[int, int]:
    end = _trim_end(text, start=start, end=end)
    if end <= start:
        return start, start + 1
    return end - 1, end


def _role_content_span(rendered_prompt: str, role: str) -> tuple[int, int] | None:
    marker = _ROLE_MARKERS[role]
    marker_start = rendered_prompt.find(marker)
    if marker_start < 0:
        return None
    start = marker_start + len(marker)
    end = rendered_prompt.find(_IM_END, start)
    if end < 0:
        return None
    return start, _trim_end(rendered_prompt, start=start, end=end)


def _prompt_im_end_span(rendered_prompt: str) -> tuple[int, int] | None:
    last = rendered_prompt.rfind(_IM_END)
    if last < 0:
        return None
    return last, last + len(_IM_END)


def _find_user_sections(rendered_prompt: str, user_span: tuple[int, int]) -> tuple[dict[str, tuple[int, int]], list[str]]:
    user_start, user_end = user_span
    user_text = rendered_prompt[user_start:user_end]
    headings: list[tuple[str, int, int]] = []
    for name, pattern in _SECTION_PATTERNS.items():
        match = pattern.search(user_text)
        if match:
            headings.append((name, user_start + match.start(), user_start + match.end()))
    headings.sort(key=lambda item: item[1])

    sections: dict[str, tuple[int, int]] = {}
    for index, (name, heading_start, heading_end) in enumerate(headings):
        next_start = headings[index + 1][1] if index + 1 < len(headings) else user_end
        sections[name] = (heading_start, _trim_end(rendered_prompt, start=heading_end, end=next_start))

    missing = [name for name in ("strategies", "settings", "portfolio", "market") if name not in sections]
    for name in missing:
        sections[name] = user_span
    return sections, missing


def build_signal_discovery_prompt_metadata(rendered_prompt: str) -> dict[str, Any]:
    token_sections: dict[str, dict[str, int]] = {}
    missing_sections: list[str] = []

    system_span = _role_content_span(rendered_prompt, "system")
    user_span = _role_content_span(rendered_prompt, "user")
    if system_span is not None:
        token_sections["system"] = {"char_start": system_span[0], "char_end": system_span[1]}
        start, end = _last_char_span(rendered_prompt, start=system_span[0], end=system_span[1])
        token_sections["system_end"] = {"char_start": start, "char_end": end}
    elif user_span is not None:
        token_sections["system"] = {"char_start": user_span[0], "char_end": user_span[1]}
        start, end = _last_char_span(rendered_prompt, start=user_span[0], end=user_span[1])
        token_sections["system_end"] = {"char_start": start, "char_end": end}
        missing_sections.append("system")

    if user_span is not None:
        user_sections, missing = _find_user_sections(rendered_prompt, user_span)
        missing_sections.extend(missing)
        for name, span in user_sections.items():
            token_sections[name] = {"char_start": span[0], "char_end": span[1]}
            start, end = _last_char_span(rendered_prompt, start=span[0], end=span[1])
            token_sections[f"{name}_end"] = {"char_start": start, "char_end": end}

    prompt_end = _prompt_im_end_span(rendered_prompt)
    if prompt_end is not None:
        token_sections["prompt_im_end"] = {"char_start": prompt_end[0], "char_end": prompt_end[1]}

    return {
        "token_sections": token_sections,
        "phase13_section_parse": {
            "missing_sections": sorted(set(missing_sections)),
            "prompt_last_target": "prompt_im_end" if prompt_end is not None else "missing",
        },
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _load_phase12_capture_artifact(
    *,
    artifact_id: str,
    artifact_root: str,
) -> CaptureArtifact:
    store = ModalVolumeStore(name="xenon-data", root=artifact_root)
    artifact_path = store.localize(artifact_id)
    manifest_path = artifact_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Phase 12 capture manifest not found: {manifest_path}")
    manifest = ArtifactManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    return CaptureArtifact(_manifest=manifest, store=store)


def _build_synthetic_direction_bank(
    phase12_capture_artifact_id: str = DEFAULT_PHASE12_DIRECTION_CAPTURE_ARTIFACT_ID,
    phase12_artifact_root: str = DEFAULT_PHASE12_DIRECTION_ARTIFACT_ROOT,
) -> TransformResult:
    from pipelines_v2.api import Dataset as RuntimeDataset
    from projects.DX_TERMINAL.prompt_confusion.phase_09.scripts.build_phase_09_dataset import (
        build_rows as build_phase09_rows,
    )
    from projects.DX_TERMINAL.prompt_confusion.phase_10.scripts.build_phase_10_dataset import (
        build_rows as build_phase10_rows,
    )
    from projects.DX_TERMINAL.prompt_confusion.phase_12.scripts.build_phase_12_dataset import (
        build_rows as build_phase12_rows,
    )

    capture = _load_phase12_capture_artifact(
        artifact_id=str(phase12_capture_artifact_id),
        artifact_root=str(phase12_artifact_root),
    )
    records = [
        row
        for row in build_phase09_rows()
        if row.get("target_dimension") == "trade_size"
        and bool(row.get("main_benchmark_row", True))
        and not bool(row.get("edge_conflict", False))
    ]
    records.extend(
        row
        for row in build_phase10_rows()
        if row.get("target_dimension") == "risk_preference"
        and bool(row.get("main_benchmark_row", True))
        and not bool(row.get("edge_conflict", False))
    )
    records.extend(
        row
        for row in build_phase12_rows()
        if row.get("target_dimension") == "diversification_preference"
        and bool(row.get("main_benchmark_row", True))
        and not bool(row.get("edge_conflict", False))
    )
    dataset = RuntimeDataset.from_records(
        records,
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        label_columns=["target_dimension", "conflict_present"],
        name="phase13_reconstructed_three_family_direction_labels",
    )
    matrices, example_keys = feature_matrices(
        capture.feature("residual_prompt_eos"),
        layers=CAPTURED_LAYERS,
        token_selector=TokenSelector.full_sequence(),
        token_pooling=TokenPooling.last(),
    )
    family_values = np.asarray(
        ordered_values(dataset.labels("target_dimension"), example_keys, label="target_dimension"),
        dtype=object,
    )
    conflict_values = np.asarray(
        [1 if bool(value) else 0 for value in ordered_values(dataset.labels("conflict_present"), example_keys, label="conflict_present")],
        dtype=np.int64,
    )

    layers_payload: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for layer in CAPTURED_LAYERS:
        X = matrices[int(layer)]
        layer_payload: dict[str, Any] = {}
        for family in DIRECTION_FAMILIES:
            family_mask = family_values == family
            pos_mask = family_mask & (conflict_values == 1)
            neg_mask = family_mask & (conflict_values == 0)
            if not pos_mask.any() or not neg_mask.any():
                raise ValueError(f"Cannot build {family} direction at L{layer}: missing positive or negative rows")
            vector = X[pos_mask].mean(axis=0) - X[neg_mask].mean(axis=0)
            unit = _unit(vector)
            scores = X @ unit
            if float(scores[pos_mask].mean()) < float(scores[neg_mask].mean()):
                unit = -unit
                scores = -scores
            layer_payload[family] = unit.astype(np.float32).tolist()
            summary_rows.append(
                {
                    "layer": int(layer),
                    "direction": family,
                    "positive_count": int(pos_mask.sum()),
                    "negative_count": int(neg_mask.sum()),
                    "vector_norm": float(np.linalg.norm(vector)),
                    "aligned_mean": float(scores[neg_mask].mean()),
                    "conflict_mean": float(scores[pos_mask].mean()),
                }
            )
        layers_payload[str(int(layer))] = layer_payload

    return TransformResult(
        payload={
            "kind": "phase13_synthetic_direction_bank",
            "source": {
                "phase12_capture_artifact_id": str(phase12_capture_artifact_id),
                "phase12_artifact_root": str(phase12_artifact_root),
                "feature": "residual_prompt_eos",
                "pooling": "last_token",
            },
            "layers": layers_payload,
            "summary": {
                "layers": list(CAPTURED_LAYERS),
                "directions": list(DIRECTION_FAMILIES),
                "shared_direction": "computed downstream as normalized mean of family directions",
                "rows": summary_rows,
            },
        },
    )


def _direction_for_layer(payload: Mapping[str, Any], *, layer: int, name: str) -> np.ndarray:
    layers = payload.get("layers", payload)
    if not isinstance(layers, Mapping) or str(layer) not in layers:
        raise KeyError(f"Direction payload missing layer {layer}")
    layer_payload = layers[str(layer)]
    if not isinstance(layer_payload, Mapping):
        raise TypeError(f"Direction layer {layer} must be an object")
    if name == "shared_mean":
        vectors = [_direction_for_layer(payload, layer=layer, name=family) for family in DIRECTION_FAMILIES]
        return _unit(np.stack(vectors, axis=0).mean(axis=0))
    raw = layer_payload.get(name)
    if raw is None and isinstance(layer_payload.get("directions"), Mapping):
        raw = layer_payload["directions"].get(name)
    if isinstance(raw, Mapping):
        raw = raw.get("vector", raw.get("coef", raw.get("weight", raw.get("raw_vector"))))
    if raw is None:
        raise KeyError(f"Direction payload missing {name!r} at layer {layer}")
    return _unit(np.asarray(raw, dtype=np.float32))


def _stratum_means(scores: np.ndarray, strata: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for stratum in sorted(set(str(value) for value in strata.tolist())):
        mask = strata == stratum
        if mask.any():
            result[stratum] = float(scores[mask].mean())
    return result


def _focused_prompt_excerpt(prompt: str, *, limit: int = 1800) -> str:
    markers = (
        "## ACTIVE STRATEGIES",
        "\nSTRATEGIES\n",
        "\nSETTINGS\n",
        "## ACTIVE SETTINGS",
        "[user]",
    )
    starts = [prompt.find(marker) for marker in markers if prompt.find(marker) >= 0]
    start = min(starts) if starts else 0
    return prompt[start : start + limit]


def _top_bottom(
    scores: np.ndarray,
    example_keys: Sequence[str],
    prompt_texts: Sequence[str],
    strata: Sequence[str],
    *,
    n: int = 10,
) -> dict[str, Any]:
    order = np.argsort(scores)
    bottom_indices = order[:n]
    top_indices = order[-n:][::-1]

    def record(index: int) -> dict[str, Any]:
        prompt = str(prompt_texts[index])
        return {
            "example_id": str(example_keys[index]),
            "stratum": str(strata[index]),
            "score": float(scores[index]),
            "prompt_preview": prompt[:1600],
            "focused_prompt_excerpt": _focused_prompt_excerpt(prompt),
        }

    return {
        "top": [record(int(index)) for index in top_indices],
        "bottom": [record(int(index)) for index in bottom_indices],
    }


def _max_projection_scores_from_payload(
    payload: Mapping[str, Any],
    *,
    layer: int,
    direction: np.ndarray,
    example_keys: Sequence[str],
) -> np.ndarray:
    layer_payload = payload["layers"][str(int(layer))]
    scores: list[float] = []
    for example_key in example_keys:
        record = layer_payload[str(example_key)]
        values = np.asarray(record["values"], dtype=np.float32)
        if values.ndim != 2:
            raise TypeError("Full-sequence residual payload values must be rank-2")
        token_scores = values @ direction
        scores.append(float(token_scores.max()))
    return np.asarray(scores, dtype=np.float32)


def _feature_width_from_payload(payload: Mapping[str, Any], *, layer: int, example_key: str) -> int:
    values = np.asarray(payload["layers"][str(int(layer))][str(example_key)]["values"], dtype=np.float32)
    if values.ndim != 2:
        raise TypeError("Residual payload values must be rank-2")
    return int(values.shape[1])


def _coarse_projection_grid_builder(
    capture: Any,
    direction_bank: Any,
    stratum: Any,
    prompt_tier: Any,
    prompt_text: Any,
) -> TransformResult:
    direction_payload = direction_bank.result() if hasattr(direction_bank, "result") else direction_bank
    if not isinstance(direction_payload, Mapping):
        raise TypeError("direction_bank must resolve to a mapping payload")
    strata: np.ndarray | None = None
    tiers: np.ndarray | None = None
    prompt_texts: list[str] | None = None
    grid_rows: list[dict[str, Any]] = []
    cell_payloads: dict[str, Any] = {}
    projection_labels: dict[str, dict[str, float]] = {}

    feature_specs = [
        ("system_end", "residual_system_end", TokenPooling.mean()),
        ("strategies_end", "residual_strategies_end", TokenPooling.mean()),
        ("settings_end", "residual_settings_end", TokenPooling.mean()),
        ("portfolio_end", "residual_portfolio_end", TokenPooling.mean()),
        ("market_end", "residual_market_end", TokenPooling.mean()),
        ("system_mean", "residual_system_span", TokenPooling.mean()),
        ("strategies_mean", "residual_strategies_span", TokenPooling.mean()),
        ("settings_mean", "residual_settings_span", TokenPooling.mean()),
        ("portfolio_mean", "residual_portfolio_span", TokenPooling.mean()),
        ("market_mean", "residual_market_span", TokenPooling.mean()),
        ("prompt_im_end", "residual_prompt_im_end", TokenPooling.mean()),
        ("full_sequence_max", "residual_full_sequence", TokenPooling.mean()),
    ]

    available_features = set()
    if hasattr(capture, "manifest"):
        available_features = set(dict(capture.manifest().storage_refs.get("features", {})))
    selected_feature_specs = [
        spec
        for spec in feature_specs
        if not available_features or spec[1] in available_features
    ]

    for position_name, feature_name, pooling in selected_feature_specs:
        feature_ref = capture.feature(feature_name)
        full_sequence_payload = None
        if position_name == "full_sequence_max":
            full_sequence_payload = feature_ref.load()
            example_keys = sorted(full_sequence_payload["layers"][str(CAPTURED_LAYERS[0])])
            matrices = {}
        else:
            matrices, example_keys = feature_matrices(
                feature_ref,
                layers=CAPTURED_LAYERS,
                token_pooling=pooling,
            )
        if strata is None:
            strata = np.asarray(ordered_values(stratum, example_keys, label="stratum"), dtype=object)
            tiers = np.asarray(ordered_values(prompt_tier, example_keys, label="prompt_tier"), dtype=object)
            prompt_texts = [str(value) for value in ordered_values(prompt_text, example_keys, label="prompt_text")]

        for layer in CAPTURED_LAYERS:
            X = matrices.get(int(layer))
            for direction_name in DIRECTION_NAMES:
                direction = _direction_for_layer(direction_payload, layer=int(layer), name=direction_name)
                feature_width = (
                    _feature_width_from_payload(
                        full_sequence_payload,
                        layer=int(layer),
                        example_key=str(example_keys[0]),
                    )
                    if full_sequence_payload is not None
                    else int(X.shape[1])
                )
                if feature_width != direction.shape[0]:
                    raise ValueError(
                        f"Direction width mismatch for {direction_name} L{layer}: "
                        f"capture={feature_width} direction={direction.shape[0]}"
                    )
                if position_name == "full_sequence_max":
                    scores = _max_projection_scores_from_payload(
                        full_sequence_payload,
                        layer=int(layer),
                        direction=direction,
                        example_keys=example_keys,
                    )
                else:
                    if X is None:
                        raise RuntimeError(f"Missing feature matrix for {feature_name} L{layer}")
                    scores = X @ direction
                for tier in sorted(set(str(value) for value in tiers.tolist())):
                    tier_mask = tiers == tier
                    tier_scores = scores[tier_mask]
                    tier_strata = strata[tier_mask]
                    tier_keys = [example_keys[i] for i, include in enumerate(tier_mask.tolist()) if include]
                    tier_prompts = [prompt_texts[i] for i, include in enumerate(tier_mask.tolist()) if include]
                    tier_strata_list = [str(strata[i]) for i, include in enumerate(tier_mask.tolist()) if include]
                    means = _stratum_means(tier_scores, tier_strata)
                    cell_key = f"L{layer}:{position_name}:{tier}:{direction_name}"
                    cell_payloads[cell_key] = {
                        "layer": int(layer),
                        "position": position_name,
                        "prompt_tier": tier,
                        "direction": direction_name,
                        "stratum_means": means,
                        "top_bottom": _top_bottom(tier_scores, tier_keys, tier_prompts, tier_strata_list),
                    }
                    grid_rows.append(
                        {
                            "layer": int(layer),
                            "position": position_name,
                            "prompt_tier": tier,
                            "direction": direction_name,
                            "anchor_positive_mean": means.get("anchor_positive"),
                            "anchor_positive_buy_only_mean": means.get("anchor_positive_buy_only"),
                            "complaint_mean": means.get("complaint"),
                            "structure_matched_control_mean": means.get("structure_matched_control"),
                            "baseline_control_mean": means.get("baseline_control"),
                            "obvious_aligned_mean": means.get("obvious_aligned"),
                            "anchor_minus_structure_matched_control": (
                                means["anchor_positive"] - means["structure_matched_control"]
                                if "anchor_positive" in means and "structure_matched_control" in means
                                else None
                            ),
                            "complaint_minus_structure_matched_control": (
                                means["complaint"] - means["structure_matched_control"]
                                if "complaint" in means and "structure_matched_control" in means
                                else None
                            ),
                            "anchor_minus_baseline": (
                                means["anchor_positive"] - means["baseline_control"]
                                if "anchor_positive" in means and "baseline_control" in means
                                else None
                            ),
                            "complaint_minus_baseline": (
                                means["complaint"] - means["baseline_control"]
                                if "complaint" in means and "baseline_control" in means
                                else None
                            ),
                        }
                    )
                label_name = f"projection__L{layer}__{position_name}__{direction_name}"
                projection_labels[label_name] = {
                    str(example_key): float(score)
                    for example_key, score in zip(example_keys, scores.tolist(), strict=True)
                }

    return TransformResult(
        payload={
            "kind": "phase13_signal_discovery_grid",
            "layers": list(CAPTURED_LAYERS),
            "positions": [position_name for position_name, _, _ in selected_feature_specs],
            "directions": list(DIRECTION_NAMES),
            "direction_bank_artifact_id": getattr(direction_bank, "id", None),
            "summary": {
                "grid_cell_count": len(grid_rows),
                "expected_grid_cell_count": len(grid_rows),
                "note": "No classifier, thresholds, or AUROC are computed in this coarse phase.",
            },
            "grid_rows": grid_rows,
            "cells": cell_payloads,
        },
        labels={
            name: {"kind": "label", "values": values}
            for name, values in projection_labels.items()
        },
    )


def build_runner_specs() -> dict[str, object]:
    db_secret = ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon")
    artifact_store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/prompt_confusion_phase13_signal_discovery",
    )
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H100",
                secrets=(db_secret,),
                volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
            ),
            artifacts=artifact_store,
            catalog=build_prompt_confusion_catalog(__file__),
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(secrets=(db_secret,)),
            artifacts=artifact_store,
            catalog=build_prompt_confusion_catalog(__file__),
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(Path("artifacts") / "prompt_confusion_phase13_signal_discovery"),
            catalog=build_prompt_confusion_catalog(__file__),
        ),
    }


def _residual_site(name: str, token_selector: TokenSelector) -> ResidualSite:
    return ResidualSite(
        name=name,
        site="resid_post",
        layers=list(CAPTURED_LAYERS),
        tokens=token_selector,
        storage=TensorStorage(dtype="float16", format="safetensors"),
    )


def build_workflow(
    dataset: Dataset | None = None,
    *,
    residual_engine: VLLMEngine | None = None,
) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    residual_engine = residual_engine or _default_residual_engine()
    prompt_metadata = PromptMetadataBuilder.from_function(
        build_signal_discovery_prompt_metadata,
        local_python_sources=("projects",),
    )
    projection_grid = TransformBuilder.from_function(
        _coarse_projection_grid_builder,
        local_python_sources=("projects",),
    )
    direction_bank = TransformBuilder.from_function(
        _build_synthetic_direction_bank,
        local_python_sources=("projects",),
    )

    sites = [
        _residual_site("residual_system_end", TokenSelector.section("system_end")),
        _residual_site("residual_strategies_end", TokenSelector.section("strategies_end")),
        _residual_site("residual_settings_end", TokenSelector.section("settings_end")),
        _residual_site("residual_portfolio_end", TokenSelector.section("portfolio_end")),
        _residual_site("residual_market_end", TokenSelector.section("market_end")),
        _residual_site("residual_prompt_im_end", TokenSelector.section("prompt_im_end")),
    ]
    if _capture_site_set_from_env() == "full":
        sites.extend(
            [
                _residual_site("residual_system_span", TokenSelector.section("system")),
                _residual_site("residual_strategies_span", TokenSelector.section("strategies")),
                _residual_site("residual_settings_span", TokenSelector.section("settings")),
                _residual_site("residual_portfolio_span", TokenSelector.section("portfolio")),
                _residual_site("residual_market_span", TokenSelector.section("market")),
                _residual_site("residual_full_sequence", TokenSelector.full_sequence()),
            ]
        )

    return WorkflowSpec(
        name="dx_terminal_phase13_signal_discovery",
        steps=(
            WorkflowStep(
                name="build_synthetic_direction_bank",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=direction_bank,
                    inputs={
                        "phase12_capture_artifact_id": DEFAULT_PHASE12_DIRECTION_CAPTURE_ARTIFACT_ID,
                        "phase12_artifact_root": DEFAULT_PHASE12_DIRECTION_ARTIFACT_ROOT,
                    },
                ),
            ),
            WorkflowStep(
                name="capture_real_signal_discovery_residuals",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=residual_engine,
                    dataset=dataset,
                    sites=sites,
                    prompt_metadata_builder=prompt_metadata,
                ),
            ),
            WorkflowStep(
                name="coarse_projection_grid",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=projection_grid,
                    inputs={
                        "capture": StepRef("capture_real_signal_discovery_residuals"),
                        "direction_bank": StepRef("build_synthetic_direction_bank"),
                        "stratum": dataset.labels("stratum"),
                        "prompt_tier": dataset.labels("prompt_tier"),
                        "prompt_text": dataset.labels("prompt_text"),
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(StepRef("coarse_projection_grid"),),
                    template="default",
                    output_dir=DEFAULT_REPORT_DIR,
                ),
            ),
        ),
    )
