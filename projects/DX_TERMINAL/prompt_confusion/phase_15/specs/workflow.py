from __future__ import annotations

"""Phase 15 real-transfer comparison for Phase 14 mid-prompt directions."""

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines_v2.api import (
    ArtifactManifest,
    CaptureArtifact,
    Dataset,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeStore,
    ReportSpec,
    StepRef,
    TokenPooling,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)
from pipelines_v2.operations.execution.common import feature_matrices, ordered_values
from projects.DX_TERMINAL.prompt_confusion.catalogs import build_prompt_confusion_catalog
from projects.DX_TERMINAL.prompt_confusion.phase_13.specs.workflow import (
    CAPTURED_LAYERS,
    DB_ENV_VAR,
    DIRECTION_FAMILIES,
    build_dataset as build_phase13_dataset,
)


DEFAULT_REAL_CAPTURE_ARTIFACT_ID = os.environ.get(
    "PHASE15_REAL_CAPTURE_ARTIFACT_ID",
    "capture_1_bbfed191794c",
)
DEFAULT_REAL_CAPTURE_ARTIFACT_ROOT = os.environ.get(
    "PHASE15_REAL_CAPTURE_ARTIFACT_ROOT",
    "/data/artifacts/prompt_confusion_phase13_signal_discovery",
)
DEFAULT_PHASE14_ARTIFACT_ROOT = os.environ.get(
    "PHASE15_PHASE14_ARTIFACT_ROOT",
    "/data/artifacts/prompt_confusion_phase_14_mid_prompt_geometry",
)
DEFAULT_REPORT_DIR = "projects/DX_TERMINAL/prompt_confusion/phase_15/reports/mid_prompt_real_transfer"

BANK_SITES = ("strategies_end", "settings_end", "portfolio_end", "market_end", "prompt_eos")
DIRECTION_NAMES = DIRECTION_FAMILIES + ("shared_mean",)
REAL_POSITIONS = {
    "system_end": "residual_system_end",
    "strategies_end": "residual_strategies_end",
    "settings_end": "residual_settings_end",
    "portfolio_end": "residual_portfolio_end",
    "market_end": "residual_market_end",
    "prompt_im_end": "residual_prompt_im_end",
}
MATCHED_SITE_MAP = {
    "strategies_end": "strategies_end",
    "settings_end": "settings_end",
    "portfolio_end": "portfolio_end",
    "market_end": "market_end",
    "prompt_eos": "prompt_im_end",
}
FOCUS_REVIEW_CELLS = {
    (28, "settings_end", "settings_end", "shared_mean"),
    (32, "settings_end", "settings_end", "shared_mean"),
    (36, "settings_end", "settings_end", "shared_mean"),
    (32, "prompt_eos", "settings_end", "shared_mean"),
    (36, "market_end", "market_end", "shared_mean"),
    (40, "market_end", "market_end", "shared_mean"),
    (44, "market_end", "market_end", "shared_mean"),
    (44, "market_end", "market_end", "risk_preference"),
    (44, "market_end", "market_end", "trade_size"),
    (28, "portfolio_end", "portfolio_end", "shared_mean"),
    (36, "portfolio_end", "portfolio_end", "shared_mean"),
}
PHASE14_DIRECTION_ARTIFACT_IDS = {
    ("strategies_end", "trade_size"): "direction_1_eb0d0e49",
    ("strategies_end", "risk_preference"): "direction_1_3911e60f",
    ("strategies_end", "diversification_preference"): "direction_1_8dc0fd83",
    ("settings_end", "trade_size"): "direction_1_145f2280",
    ("settings_end", "risk_preference"): "direction_1_daa2c368",
    ("settings_end", "diversification_preference"): "direction_1_1e07ba79",
    ("portfolio_end", "trade_size"): "direction_1_db2781ee",
    ("portfolio_end", "risk_preference"): "direction_1_ef9b2cd9",
    ("portfolio_end", "diversification_preference"): "direction_1_2b685fe8",
    ("market_end", "trade_size"): "direction_1_ebc4cb43",
    ("market_end", "risk_preference"): "direction_1_cfc59ff8",
    ("market_end", "diversification_preference"): "direction_1_e11d3886",
    ("prompt_eos", "trade_size"): "direction_1_8baa4187",
    ("prompt_eos", "risk_preference"): "direction_1_ebff8bb8",
    ("prompt_eos", "diversification_preference"): "direction_1_a5f94704",
}


def build_dataset() -> Dataset:
    return build_phase13_dataset()


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _load_capture_artifact(*, artifact_id: str, artifact_root: str) -> CaptureArtifact:
    store = ModalVolumeStore(name="xenon-data", root=artifact_root)
    artifact_path = store.localize(artifact_id)
    manifest_path = artifact_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Capture manifest not found: {manifest_path}")
    manifest = ArtifactManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    return CaptureArtifact(_manifest=manifest, store=store)


def _load_operation_result(*, artifact_id: str, artifact_root: str) -> Mapping[str, Any]:
    store = ModalVolumeStore(name="xenon-data", root=artifact_root)
    artifact_path = store.localize(artifact_id)
    result_path = artifact_path / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Operation result not found: {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected mapping payload for {artifact_id}, got {type(payload).__name__}")
    return payload


def _direction_vector(payload: Mapping[str, Any], *, layer: int) -> np.ndarray:
    layers = payload.get("layers")
    if not isinstance(layers, Mapping) or str(layer) not in layers:
        raise KeyError(f"Direction payload missing layer {layer}")
    raw = layers[str(layer)].get("vector")
    if raw is None:
        raise KeyError(f"Direction payload missing vector at layer {layer}")
    return _unit(np.asarray(raw, dtype=np.float32))


def _build_phase14_direction_bank(
    phase14_artifact_root: str = DEFAULT_PHASE14_ARTIFACT_ROOT,
) -> TransformResult:
    payloads: dict[tuple[str, str], Mapping[str, Any]] = {
        key: _load_operation_result(artifact_id=artifact_id, artifact_root=str(phase14_artifact_root))
        for key, artifact_id in PHASE14_DIRECTION_ARTIFACT_IDS.items()
    }
    layers_payload: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for layer in CAPTURED_LAYERS:
        layer_payload: dict[str, Any] = {}
        for site in BANK_SITES:
            site_payload: dict[str, Any] = {}
            family_vectors = []
            for family in DIRECTION_FAMILIES:
                vector = _direction_vector(payloads[(site, family)], layer=int(layer))
                site_payload[family] = vector.tolist()
                family_vectors.append(vector)
                rows.append(
                    {
                        "layer": int(layer),
                        "bank_site": site,
                        "direction": family,
                        "source_artifact_id": PHASE14_DIRECTION_ARTIFACT_IDS[(site, family)],
                    }
                )
            site_payload["shared_mean"] = _unit(np.stack(family_vectors, axis=0).mean(axis=0)).tolist()
            layer_payload[site] = site_payload
        layers_payload[str(int(layer))] = layer_payload
    return TransformResult(
        payload={
            "kind": "phase15_phase14_direction_bank",
            "source": {
                "phase14_artifact_root": str(phase14_artifact_root),
                "direction_artifact_ids": {
                    f"{site}__{family}": artifact_id
                    for (site, family), artifact_id in PHASE14_DIRECTION_ARTIFACT_IDS.items()
                },
            },
            "layers": layers_payload,
            "summary": {
                "layers": list(CAPTURED_LAYERS),
                "bank_sites": list(BANK_SITES),
                "directions": list(DIRECTION_NAMES),
                "rows": rows,
            },
        },
    )


def _direction_for_layer(payload: Mapping[str, Any], *, layer: int, bank_site: str, name: str) -> np.ndarray:
    layers = payload.get("layers")
    if not isinstance(layers, Mapping) or str(layer) not in layers:
        raise KeyError(f"Direction bank missing layer {layer}")
    site_payload = layers[str(layer)].get(bank_site)
    if not isinstance(site_payload, Mapping):
        raise KeyError(f"Direction bank missing site {bank_site!r} at layer {layer}")
    raw = site_payload.get(name)
    if raw is None:
        raise KeyError(f"Direction bank missing {name!r} for {bank_site} L{layer}")
    return _unit(np.asarray(raw, dtype=np.float32))


def _stratum_means(scores: np.ndarray, strata: np.ndarray) -> dict[str, float]:
    return {
        str(stratum): float(scores[strata == stratum].mean())
        for stratum in sorted(set(str(value) for value in strata.tolist()))
        if (strata == stratum).any()
    }


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
        return {
            "example_id": str(example_keys[index]),
            "stratum": str(strata[index]),
            "score": float(scores[index]),
            "prompt_preview": str(prompt_texts[index])[:1600],
        }

    return {
        "top": [record(int(index)) for index in top_indices],
        "bottom": [record(int(index)) for index in bottom_indices],
    }


def _top_bottom_by_stratum(
    scores: np.ndarray,
    example_keys: Sequence[str],
    prompt_texts: Sequence[str],
    strata: np.ndarray,
    *,
    n: int = 10,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stratum in sorted(set(str(value) for value in strata.tolist())):
        indices = [index for index, value in enumerate(strata.tolist()) if str(value) == stratum]
        if not indices:
            continue
        stratum_scores = scores[indices]
        order = np.argsort(stratum_scores)
        bottom = [indices[int(local_index)] for local_index in order[:n]]
        top = [indices[int(local_index)] for local_index in order[-n:][::-1]]

        def record(index: int) -> dict[str, Any]:
            return {
                "example_id": str(example_keys[index]),
                "stratum": str(strata[index]),
                "score": float(scores[index]),
                "prompt_preview": str(prompt_texts[index])[:2400],
            }

        result[stratum] = {
            "top": [record(index) for index in top],
            "bottom": [record(index) for index in bottom],
        }
    return result


def _best(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    require_matched_site: bool | None = None,
    direction: str | None = None,
) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(key) is not None]
    if require_matched_site is not None:
        candidates = [row for row in candidates if bool(row.get("matched_site")) is require_matched_site]
    if direction is not None:
        candidates = [row for row in candidates if row.get("direction") == direction]
    if not candidates:
        return None
    return dict(max(candidates, key=lambda row: float(row[key])))


def _phase14_real_transfer_builder(
    direction_bank: Any,
    stratum: Any,
    prompt_tier: Any,
    prompt_text: Any,
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

    grid_rows: list[dict[str, Any]] = []
    cell_payloads: dict[str, Any] = {}
    strata: np.ndarray | None = None
    tiers: np.ndarray | None = None
    prompt_texts: list[str] | None = None
    example_keys: Sequence[str] | None = None

    available_features = set(dict(capture.manifest().storage_refs.get("features", {})))
    selected_positions = [
        (position, feature_name)
        for position, feature_name in REAL_POSITIONS.items()
        if feature_name in available_features
    ]
    for real_position, feature_name in selected_positions:
        matrices, current_keys = feature_matrices(
            capture.feature(feature_name),
            layers=CAPTURED_LAYERS,
            token_pooling=TokenPooling.mean(),
        )
        if strata is None:
            example_keys = current_keys
            strata = np.asarray(ordered_values(stratum, current_keys, label="stratum"), dtype=object)
            tiers = np.asarray(ordered_values(prompt_tier, current_keys, label="prompt_tier"), dtype=object)
            prompt_texts = [str(value) for value in ordered_values(prompt_text, current_keys, label="prompt_text")]
        elif list(current_keys) != list(example_keys or ()):
            raise ValueError(f"Example key order mismatch for {feature_name}")

        for layer in CAPTURED_LAYERS:
            X = matrices[int(layer)]
            for bank_site in BANK_SITES:
                for direction_name in DIRECTION_NAMES:
                    direction = _direction_for_layer(
                        direction_payload,
                        layer=int(layer),
                        bank_site=bank_site,
                        name=direction_name,
                    )
                    if int(X.shape[1]) != int(direction.shape[0]):
                        raise ValueError(
                            f"Direction width mismatch for {bank_site}/{direction_name} L{layer}: "
                            f"capture={X.shape[1]} direction={direction.shape[0]}"
                        )
                    scores = X @ direction
                    for tier in sorted(set(str(value) for value in tiers.tolist())):
                        tier_mask = tiers == tier
                        tier_scores = scores[tier_mask]
                        tier_strata = strata[tier_mask]
                        means = _stratum_means(tier_scores, tier_strata)
                        matched_site = MATCHED_SITE_MAP.get(bank_site) == real_position
                        row = {
                            "layer": int(layer),
                            "bank_site": bank_site,
                            "real_position": real_position,
                            "matched_site": bool(matched_site),
                            "prompt_tier": tier,
                            "direction": direction_name,
                            "anchor_positive_mean": means.get("anchor_positive"),
                            "anchor_positive_buy_only_mean": means.get("anchor_positive_buy_only"),
                            "complaint_mean": means.get("complaint"),
                            "structure_matched_control_mean": means.get("structure_matched_control"),
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
                            "anchor_minus_complaint": (
                                means["anchor_positive"] - means["complaint"]
                                if "anchor_positive" in means and "complaint" in means
                                else None
                            ),
                        }
                        grid_rows.append(row)
                        cell_key = f"L{layer}:{bank_site}->{real_position}:{tier}:{direction_name}"
                        tier_keys = [current_keys[i] for i, include in enumerate(tier_mask.tolist()) if include]
                        tier_prompts = [prompt_texts[i] for i, include in enumerate(tier_mask.tolist()) if include]
                        tier_strata_list = [str(strata[i]) for i, include in enumerate(tier_mask.tolist()) if include]
                        cell_payloads[cell_key] = {
                            **row,
                            "stratum_means": means,
                            "top_bottom": _top_bottom(tier_scores, tier_keys, tier_prompts, tier_strata_list),
                        }
                        focus_key = (int(layer), bank_site, real_position, direction_name)
                        if focus_key in FOCUS_REVIEW_CELLS:
                            cell_payloads[cell_key]["top_bottom_by_stratum"] = _top_bottom_by_stratum(
                                tier_scores,
                                tier_keys,
                                tier_prompts,
                                tier_strata,
                            )

    summary = {
        "grid_cell_count": len(grid_rows),
        "real_capture_artifact_id": str(real_capture_artifact_id),
        "real_capture_artifact_root": str(real_capture_artifact_root),
        "best_matched_anchor_delta": _best(
            grid_rows,
            key="anchor_minus_structure_matched_control",
            require_matched_site=True,
        ),
        "best_matched_complaint_delta": _best(
            grid_rows,
            key="complaint_minus_structure_matched_control",
            require_matched_site=True,
        ),
        "best_overall_complaint_delta": _best(
            grid_rows,
            key="complaint_minus_structure_matched_control",
        ),
        "best_trade_size_matched_complaint_delta": _best(
            grid_rows,
            key="complaint_minus_structure_matched_control",
            require_matched_site=True,
            direction="trade_size",
        ),
        "best_shared_mean_matched_complaint_delta": _best(
            grid_rows,
            key="complaint_minus_structure_matched_control",
            require_matched_site=True,
            direction="shared_mean",
        ),
    }
    return TransformResult(
        payload={
            "kind": "phase15_mid_prompt_real_transfer",
            "layers": list(CAPTURED_LAYERS),
            "bank_sites": list(BANK_SITES),
            "real_positions": [position for position, _ in selected_positions],
            "directions": list(DIRECTION_NAMES),
            "summary": summary,
            "grid_rows": grid_rows,
            "cells": cell_payloads,
        },
    )


def build_runner_specs() -> dict[str, object]:
    db_secret = ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon")
    artifact_store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/prompt_confusion_phase15_mid_prompt_real_transfer",
    )
    return {
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(secrets=(db_secret,)),
            artifacts=artifact_store,
            catalog=build_prompt_confusion_catalog(__file__),
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(Path("artifacts") / "prompt_confusion_phase15_mid_prompt_real_transfer"),
            catalog=build_prompt_confusion_catalog(__file__),
        ),
    }


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    direction_bank_builder = TransformBuilder.from_function(
        _build_phase14_direction_bank,
        local_python_sources=("projects",),
    )
    transfer_builder = TransformBuilder.from_function(
        _phase14_real_transfer_builder,
        local_python_sources=("projects",),
    )
    return WorkflowSpec(
        name="dx_terminal_phase15_mid_prompt_real_transfer",
        steps=(
            WorkflowStep(
                name="build_phase14_direction_bank",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=direction_bank_builder,
                    inputs={"phase14_artifact_root": DEFAULT_PHASE14_ARTIFACT_ROOT},
                ),
            ),
            WorkflowStep(
                name="phase14_real_transfer_grid",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=transfer_builder,
                    inputs={
                        "direction_bank": StepRef("build_phase14_direction_bank"),
                        "stratum": dataset.labels("stratum"),
                        "prompt_tier": dataset.labels("prompt_tier"),
                        "prompt_text": dataset.labels("prompt_text"),
                        "real_capture_artifact_id": DEFAULT_REAL_CAPTURE_ARTIFACT_ID,
                        "real_capture_artifact_root": DEFAULT_REAL_CAPTURE_ARTIFACT_ROOT,
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(StepRef("phase14_real_transfer_grid"),),
                    template="default",
                    output_dir=DEFAULT_REPORT_DIR,
                ),
            ),
        ),
    )
