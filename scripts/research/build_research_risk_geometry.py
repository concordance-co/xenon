from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from pipelines.db import connect_neon
from pipelines.interp.counterfactual import build_settings_edited_variant, build_market_rows, parse_market_section
from research.research_rerun.core import _build_example_record, _load_source_examples, save_prompt_payload


RISK_LEVELS = (1, 2, 3, 4, 5)
BASE_CONTEXT = "risk_3"


def _zscore(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    std = float(arr.std())
    if std <= 1e-6:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / std


def _risk_alpha(level: int) -> float:
    return {
        1: 0.85,
        2: 0.45,
        3: 0.0,
        4: -0.35,
        5: -0.75,
    }[int(level)]


def _compute_symbol_geometry(rows: list[Any]) -> dict[str, dict[str, float]]:
    momentum = [0.55 * row.pct_5m + 0.45 * row.pct_1h for row in rows]
    participation = [0.18 * row.vol_5m + 0.06 * row.vol_1h + 0.22 * row.unique_traders_5m for row in rows]
    flow = [row.net_flow_5m for row in rows]
    traders = [row.unique_traders_5m for row in rows]
    vol_5m = [row.vol_5m for row in rows]
    vol_1h = [row.vol_1h for row in rows]
    abs_pct_5m = [abs(row.pct_5m) for row in rows]
    abs_pct_1h = [abs(row.pct_1h) for row in rows]
    abs_flow = [abs(row.net_flow_5m) for row in rows]

    z_momentum = _zscore(momentum)
    z_participation = _zscore(participation)
    z_flow = _zscore(flow)
    z_traders = _zscore(traders)
    z_vol_5m = _zscore(vol_5m)
    z_vol_1h = _zscore(vol_1h)
    z_abs_pct_5m = _zscore(abs_pct_5m)
    z_abs_pct_1h = _zscore(abs_pct_1h)
    z_abs_flow = _zscore(abs_flow)

    result: dict[str, dict[str, float]] = {}
    for idx, row in enumerate(rows):
        strength = (
            1.00 * float(z_momentum[idx])
            + 0.85 * float(z_flow[idx])
            + 0.35 * float(z_participation[idx])
        )
        stability = (
            0.75 * float(z_traders[idx])
            + 0.25 * float(z_vol_1h[idx])
            + 0.20 * float(z_vol_5m[idx])
            - 0.45 * float(z_abs_pct_5m[idx])
            - 0.25 * float(z_abs_pct_1h[idx])
            - 0.10 * float(z_abs_flow[idx])
        )
        result[row.symbol] = {
            "strength": strength,
            "stability": stability,
        }
    return result


def _select_four_asset_slice(rows: list[Any], symbol_geometry: dict[str, dict[str, float]]) -> list[int]:
    if len(rows) <= 4:
        return list(range(len(rows)))
    points = np.asarray(
        [[symbol_geometry[row.symbol]["strength"], symbol_geometry[row.symbol]["stability"]] for row in rows],
        dtype=np.float32,
    )
    centroid = points.mean(axis=0)
    selected = [int(np.argmax(np.linalg.norm(points - centroid, axis=1)))]
    while len(selected) < 4:
        best_idx = None
        best_score = None
        for idx in range(len(rows)):
            if idx in selected:
                continue
            dists = [float(np.linalg.norm(points[idx] - points[j])) for j in selected]
            score = min(dists)
            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            break
        selected.append(int(best_idx))
    return sorted(selected)


def _score_coords_for_level(
    *,
    selected_symbols: list[str],
    symbol_geometry: dict[str, dict[str, float]],
    risk_level: int,
) -> dict[str, list[float]]:
    alpha = _risk_alpha(risk_level)
    coords: dict[str, list[float]] = {}
    for symbol in selected_symbols:
        strength = float(symbol_geometry[symbol]["strength"])
        stability = float(symbol_geometry[symbol]["stability"])
        risk_adjusted = strength + alpha * stability
        coords[symbol] = [strength, risk_adjusted]
    return coords


def _load_candidate_rows(source_parquet: Path) -> list[dict[str, Any]]:
    table = pq.read_table(
        source_parquet,
        columns=[
            "example_id",
            "log_id",
            "created_at",
            "vault_address",
            "decision_type",
            "vault_risk_preference",
            "market_snapshot_json",
        ],
    )
    rows = table.to_pylist()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("decision_type") != "record_observation":
            continue
        market_json = row.get("market_snapshot_json")
        if isinstance(market_json, str):
            try:
                market_json = json.loads(market_json)
            except json.JSONDecodeError:
                continue
        tokens = list((market_json or {}).get("Tokens") or [])
        if len(tokens) < 5:
            continue
        roster = tuple(sorted((token.get("Symbol") or "") for token in tokens))
        candidates.append(
            {
                **row,
                "market_snapshot_json": market_json,
                "roster_key": roster,
                "n_tokens": len(tokens),
            }
        )
    return candidates


def _pick_diverse_rows(
    candidates: list[dict[str, Any]],
    *,
    top_rosters: int,
    per_roster: int,
) -> list[dict[str, Any]]:
    by_roster: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_roster[tuple(row["roster_key"])].append(row)

    chosen: list[dict[str, Any]] = []
    for roster_key, roster_rows in sorted(by_roster.items(), key=lambda kv: len(kv[1]), reverse=True)[:top_rosters]:
        roster_rows = sorted(roster_rows, key=lambda row: str(row["created_at"]))
        if len(roster_rows) <= per_roster:
            chosen.extend(roster_rows)
            continue
        idxs = np.linspace(0, len(roster_rows) - 1, per_roster, dtype=int)
        deduped = []
        seen = set()
        for idx in idxs.tolist():
            if idx in seen:
                continue
            seen.add(idx)
            deduped.append(roster_rows[idx])
        chosen.extend(deduped)
    return chosen


def build_risk_geometry_payload(
    *,
    source_parquet: Path,
    experiment_id: str,
    top_rosters: int,
    per_roster: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_rows = _load_candidate_rows(source_parquet)
    selected_rows = _pick_diverse_rows(candidate_rows, top_rosters=top_rosters, per_roster=per_roster)
    log_ids = sorted({int(row["log_id"]) for row in selected_rows})

    with connect_neon() as conn:
        source_examples = _load_source_examples(conn, log_ids)
    missing = [log_id for log_id in log_ids if log_id not in source_examples]
    if missing:
        raise RuntimeError(f"Missing source rows in interp_examples_v0 for log_ids={missing[:10]}")

    examples: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for selected in selected_rows:
        source_row = source_examples[int(selected["log_id"])]
        example_record = _build_example_record(source_row)
        market_json = example_record["market_json"]
        _, row_texts = parse_market_section(source_row["user_text"])
        market_rows = build_market_rows(market_json, row_texts)
        symbol_geometry = _compute_symbol_geometry(market_rows)
        selected_indices = _select_four_asset_slice(market_rows, symbol_geometry)
        selected_symbols = [market_rows[idx].symbol for idx in selected_indices]
        base_coords = {
            symbol: [
                float(symbol_geometry[symbol]["strength"]),
                float(symbol_geometry[symbol]["stability"]),
            ]
            for symbol in selected_symbols
        }

        example_record["metadata"] = {
            **(example_record.get("metadata") or {}),
            "experiment_group": "risk_geometry",
            "selected_symbols": selected_symbols,
            "selected_row_indices": selected_indices,
            "base_context": BASE_CONTEXT,
            "base_coords": base_coords,
            "roster_key": list(selected["roster_key"]),
        }
        examples.append(example_record)

        variant_texts: dict[str, str] = {}
        for risk_level in RISK_LEVELS:
            variant = f"risk_{risk_level}"
            edited_user = build_settings_edited_variant(
                source_row["user_text"],
                {"Asset Risk Preference": risk_level},
            )
            variant_texts[variant] = edited_user

            score_coords = _score_coords_for_level(
                selected_symbols=selected_symbols,
                symbol_geometry=symbol_geometry,
                risk_level=risk_level,
            )
            metadata = {
                "source_log_id": int(source_row["log_id"]),
                "risk_level": risk_level,
                "selected_symbols": selected_symbols,
                "selected_row_indices": selected_indices,
                "base_context": BASE_CONTEXT,
                "base_coords": base_coords,
                "score_coords": score_coords,
                "roster_key": list(selected["roster_key"]),
                "original_vault_risk_preference": selected.get("vault_risk_preference"),
            }
            prompts.append(
                {
                    "prompt_id": f"{experiment_id}:{source_row['log_id']}:risk_geometry:{variant}",
                    "base_example_id": source_row["example_id"],
                    "experiment_id": experiment_id,
                    "experiment_group": "risk_geometry",
                    "cohort_label": "risk_geometry",
                    "variant": variant,
                    "system_text": source_row["system_text"],
                    "user_text": edited_user,
                    "row_order": list(example_record["row_order"]),
                    "n_rows": len(example_record["row_order"]),
                    "target_asset": None,
                    "block_reason": None,
                    "settings_signature": variant,
                    "actionability_cell": None,
                    "metadata": metadata,
                }
            )
        if len(set(variant_texts.values())) < len(variant_texts):
            raise ValueError(f"Risk ladder collapsed for log_id={source_row['log_id']}")

        manifest_rows.append(
            {
                "base_example_id": source_row["example_id"],
                "log_id": int(source_row["log_id"]),
                "created_at": source_row.get("created_at"),
                "vault_address": source_row.get("vault_address"),
                "roster_key": list(selected["roster_key"]),
                "selected_symbols": selected_symbols,
                "selected_row_indices": selected_indices,
                "original_vault_risk_preference": selected.get("vault_risk_preference"),
            }
        )

    payload = {
        "experiment_id": experiment_id,
        "examples": examples,
        "prompts": prompts,
    }
    summary = {
        "experiment_id": experiment_id,
        "base_examples": len(examples),
        "prompts": len(prompts),
        "contexts": [f"risk_{level}" for level in RISK_LEVELS],
        "top_rosters": top_rosters,
        "per_roster": per_roster,
        "roster_counts": {
            ",".join(roster): sum(1 for row in manifest_rows if row["roster_key"] == list(roster))
            for roster in sorted({tuple(row["roster_key"]) for row in manifest_rows})
        },
        "manifest_rows": manifest_rows,
    }
    return payload, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real DX risk-geometry ladder prompts in Neon.")
    parser.add_argument(
        "--source-parquet",
        type=Path,
        default=Path("data/interp_exports/interp_examples_v0_high_quality.parquet"),
    )
    parser.add_argument("--experiment-id", default="real_risk_geometry_bridge_v1")
    parser.add_argument("--top-rosters", type=int, default=5)
    parser.add_argument("--per-roster", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis_results/real_risk_geometry_bridge"),
    )
    args = parser.parse_args()

    payload, summary = build_risk_geometry_payload(
        source_parquet=args.source_parquet,
        experiment_id=args.experiment_id,
        top_rosters=args.top_rosters,
        per_roster=args.per_roster,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.experiment_id}_manifest.json").write_text(json.dumps(summary, indent=2, default=str))

    with connect_neon(autocommit=False) as conn:
        save_prompt_payload(conn, payload)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
