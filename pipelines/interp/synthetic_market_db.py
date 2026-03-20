from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from pipelines.db import connect_neon
from pipelines.interp.decision_structure.cohorts import (
    validate_order_mode,
    validate_relation_name,
)


SCHEMA_SQL_PATH = Path(__file__).resolve().parents[2] / "sql" / "synthetic_market_phase1.sql"


def _family_priority(family: str) -> float:
    return {
        "pairwise_tradeoff": 300.0,
        "scalar_sweep": 200.0,
        "archetype_family": 100.0,
    }.get(family, 0.0)


def _load_rows(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tick_rows = pq.read_table(input_dir / "synthetic_market_tick_records.parquet").to_pylist()
    asset_rows = pq.read_table(input_dir / "synthetic_market_asset_records.parquet").to_pylist()
    pairwise_rows = pq.read_table(input_dir / "synthetic_market_pairwise_records.parquet").to_pylist()
    return tick_rows, asset_rows, pairwise_rows


def prepare_rows_for_upload(
    input_dir: Path,
    *,
    phase_name: str = "phase1",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tick_rows, asset_rows, pairwise_rows = _load_rows(input_dir)

    ordered_ticks = sorted(
        tick_rows,
        key=lambda row: (
            row["context_variant"] != "market_only",
            -_family_priority(row["family"]),
            row["family"],
            row["family_variant"],
            row["example_id"],
            row["context_variant"],
            row["log_id"],
        ),
    )
    selection_rank_by_log_id = {
        row["log_id"]: rank
        for rank, row in enumerate(ordered_ticks, start=1)
    }

    for row in tick_rows:
        row["phase_name"] = phase_name
        row["capture_enabled"] = True
        row["selection_rank"] = selection_rank_by_log_id[row["log_id"]]
        row["capture_priority"] = _family_priority(row["family"]) + (
            10.0 if row["context_variant"] == "market_only" else 0.0
        )

    for row in asset_rows:
        row["phase_name"] = phase_name
    for row in pairwise_rows:
        row["phase_name"] = phase_name

    return tick_rows, asset_rows, pairwise_rows


def apply_schema() -> None:
    sql = SCHEMA_SQL_PATH.read_text()
    conn = connect_neon()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
    finally:
        conn.close()


def upload_dataset(
    input_dir: Path,
    *,
    phase_name: str = "phase1",
    replace_phase: bool = True,
) -> dict[str, Any]:
    tick_rows, asset_rows, pairwise_rows = prepare_rows_for_upload(
        input_dir,
        phase_name=phase_name,
    )
    apply_schema()

    conn = connect_neon()
    try:
        with conn.transaction():
            if replace_phase:
                conn.execute("DELETE FROM synthetic_market_pairs_v0 WHERE phase_name = %s", (phase_name,))
                conn.execute("DELETE FROM synthetic_market_assets_v0 WHERE phase_name = %s", (phase_name,))
                conn.execute("DELETE FROM synthetic_market_examples_v0 WHERE phase_name = %s", (phase_name,))

            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO synthetic_market_examples_v0 (
                        log_id, phase_name, example_id, family, family_variant, context_variant,
                        system_prompt, user_prompt, prompt_messages_json, labels_json, num_assets,
                        best_asset, buy_any, observe_vs_act, capture_enabled, selection_rank,
                        capture_priority
                    ) VALUES (
                        %(log_id)s, %(phase_name)s, %(example_id)s, %(family)s, %(family_variant)s, %(context_variant)s,
                        %(system_prompt)s, %(user_prompt)s, %(prompt_messages_json)s::jsonb, %(labels_json)s::jsonb, %(num_assets)s,
                        %(best_asset)s, %(buy_any)s, %(observe_vs_act)s, %(capture_enabled)s, %(selection_rank)s,
                        %(capture_priority)s
                    )
                    """,
                    tick_rows,
                )
                cur.executemany(
                    """
                    INSERT INTO synthetic_market_assets_v0 (
                        log_id, phase_name, example_id, family, family_variant, context_variant,
                        row_index, symbol, archetype, pct_5m, pct_1h, net_flow_5m, vol_5m, vol_1h,
                        unique_traders_5m, top20_holder_pct, age_bucket, momentum_score, participation_score,
                        flow_score, concentration_penalty, riskiness_score, attractiveness_score,
                        risk_adjusted_score, edge_after_fee_score, edge_gt_fee, attractiveness_rank,
                        risk_adjusted_rank, is_best_asset, buyable_if_unconstrained,
                        acceptable_under_risk_setting
                    ) VALUES (
                        %(log_id)s, %(phase_name)s, %(example_id)s, %(family)s, %(family_variant)s, %(context_variant)s,
                        %(row_index)s, %(symbol)s, %(archetype)s, %(pct_5m)s, %(pct_1h)s, %(net_flow_5m)s, %(vol_5m)s, %(vol_1h)s,
                        %(unique_traders_5m)s, %(top20_holder_pct)s, %(age_bucket)s, %(momentum_score)s, %(participation_score)s,
                        %(flow_score)s, %(concentration_penalty)s, %(riskiness_score)s, %(attractiveness_score)s,
                        %(risk_adjusted_score)s, %(edge_after_fee_score)s, %(edge_gt_fee)s, %(attractiveness_rank)s,
                        %(risk_adjusted_rank)s, %(is_best_asset)s, %(buyable_if_unconstrained)s,
                        %(acceptable_under_risk_setting)s
                    )
                    """,
                    asset_rows,
                )
                cur.executemany(
                    """
                    INSERT INTO synthetic_market_pairs_v0 (
                        log_id, phase_name, example_id, family, family_variant, context_variant,
                        asset_a, asset_b, a_beats_b_on_attractiveness, a_beats_b_on_risk_adjusted,
                        delta_pct_5m, delta_pct_1h, delta_net_flow_5m, delta_vol_5m,
                        delta_unique_traders_5m, delta_top20_holder_pct
                    ) VALUES (
                        %(log_id)s, %(phase_name)s, %(example_id)s, %(family)s, %(family_variant)s, %(context_variant)s,
                        %(asset_a)s, %(asset_b)s, %(a_beats_b_on_attractiveness)s, %(a_beats_b_on_risk_adjusted)s,
                        %(delta_pct_5m)s, %(delta_pct_1h)s, %(delta_net_flow_5m)s, %(delta_vol_5m)s,
                        %(delta_unique_traders_5m)s, %(delta_top20_holder_pct)s
                    )
                    """,
                    pairwise_rows,
                )

        counts = {
            "examples": conn.execute(
                "SELECT COUNT(*) AS n FROM synthetic_market_examples_v0 WHERE phase_name = %s",
                (phase_name,),
            ).fetchone()["n"],
            "assets": conn.execute(
                "SELECT COUNT(*) AS n FROM synthetic_market_assets_v0 WHERE phase_name = %s",
                (phase_name,),
            ).fetchone()["n"],
            "pairs": conn.execute(
                "SELECT COUNT(*) AS n FROM synthetic_market_pairs_v0 WHERE phase_name = %s",
                (phase_name,),
            ).fetchone()["n"],
            "market_only": conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM synthetic_market_examples_v0
                WHERE phase_name = %s AND context_variant = 'market_only'
                """,
                (phase_name,),
            ).fetchone()["n"],
        }
    finally:
        conn.close()

    return counts


def _synthetic_order_sql(mode: str) -> str:
    mode = validate_order_mode(mode)
    if mode == "capture_priority_desc":
        return "capture_priority DESC NULLS LAST, selection_rank ASC NULLS LAST, log_id"
    if mode == "selection_rank_asc":
        return "selection_rank ASC NULLS LAST, log_id"
    if mode == "created_at_desc":
        return "created_at DESC NULLS LAST, log_id DESC"
    if mode == "hash":
        return "md5(log_id::text)"
    return "log_id"


def build_synthetic_example_query(
    *,
    select_columns: list[str],
    cohort_view: str | None,
    order_mode: str,
    limit: int | None,
) -> tuple[str, list[Any]]:
    if not select_columns:
        raise ValueError("select_columns must not be empty")

    relation = validate_relation_name(cohort_view) or "synthetic_market_phase1_capture_v0"
    order_sql = _synthetic_order_sql(order_mode)
    query = f"""
        SELECT {", ".join(select_columns)}
        FROM {relation}
        ORDER BY {order_sql}
    """
    params: list[Any] = []
    if limit is not None:
        query += "\n        LIMIT %s"
        params.append(limit)
    return query, params


def load_examples_from_neon(
    *,
    limit: int | None = None,
    cohort_view: str | None = None,
    order_mode: str = "selection_rank_asc",
) -> list[dict[str, Any]]:
    query, params = build_synthetic_example_query(
        select_columns=["log_id", "prompt_messages_json"],
        cohort_view=cohort_view,
        order_mode=order_mode,
        limit=limit,
    )
    conn = connect_neon()
    try:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def table_snapshot(phase_name: str = "phase1") -> dict[str, Any]:
    conn = connect_neon()
    try:
        return {
            "examples": conn.execute(
                "SELECT COUNT(*) AS n FROM synthetic_market_examples_v0 WHERE phase_name = %s",
                (phase_name,),
            ).fetchone()["n"],
            "market_only": conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM synthetic_market_examples_v0
                WHERE phase_name = %s AND context_variant = 'market_only'
                """,
                (phase_name,),
            ).fetchone()["n"],
            "families": json.loads(
                json.dumps(
                    conn.execute(
                        """
                        SELECT family, COUNT(*) AS n
                        FROM synthetic_market_examples_v0
                        WHERE phase_name = %s
                        GROUP BY family
                        ORDER BY family
                        """,
                        (phase_name,),
                    ).fetchall(),
                    default=str,
                )
            ),
        }
    finally:
        conn.close()
