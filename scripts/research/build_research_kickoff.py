#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.db import connect_neon
from pipelines.interp.decision_structure.manifest import manifest_summary, select_manifest_rows
from pipelines.interp.research_kickoff import (
    annotate_kickoff_row,
    blocked_valence_manifest_plan,
    roadmap_as_dicts,
    settings_twist_manifest_plan,
)


def _query_rows(conn, query: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    rows = conn.execute(query, params or []).fetchall()
    return [dict(row) for row in rows]


def _count_dict(rows: list[dict[str, Any]], key: str, *, top_n: int | None = None) -> dict[str, int]:
    counts = Counter(str(row.get(key) or "NONE") for row in rows)
    items = counts.most_common(top_n)
    return {name: int(count) for name, count in items}


def build_kickoff_payload(*, scan_limit: int) -> dict[str, Any]:
    with connect_neon() as conn:
        overall = conn.execute(
            """
            SELECT json_build_object(
                'decision_examples', (SELECT COUNT(*) FROM decision_capture_base_v1),
                'blocked_observe_candidates', (SELECT COUNT(*) FROM decision_blocked_observe_candidates_v1),
                'policy_tension_candidates', (SELECT COUNT(*) FROM decision_policy_tension_candidates_v1),
                'buy_candidates', (SELECT COUNT(*) FROM decision_trade_candidates_v1 WHERE trade_side = 'buy'),
                'sell_candidates', (SELECT COUNT(*) FROM decision_sell_candidates_v1)
            ) AS stats
            """
        ).fetchone()["stats"]
        blocked_reason_counts = _query_rows(
            conn,
            """
            SELECT block_reason, COUNT(*) AS count
            FROM decision_blocked_observe_candidates_v1
            GROUP BY 1
            ORDER BY count DESC, block_reason
            """,
        )
        blocked_actionability_counts = _query_rows(
            conn,
            """
            SELECT
                CASE
                    WHEN can_buy_any AND can_sell_any THEN 'buy+sell'
                    WHEN can_buy_any THEN 'buy_only'
                    WHEN can_sell_any THEN 'sell_only'
                    ELSE 'none'
                END AS actionability_cell,
                COUNT(*) AS count
            FROM decision_blocked_observe_candidates_v1
            GROUP BY 1
            ORDER BY count DESC, actionability_cell
            """,
        )
        policy_settings_signature_counts = _query_rows(
            conn,
            """
            SELECT
                concat_ws('/', trade_size, trading_activity, holding_style, diversification, risk_preference) AS settings_signature,
                COUNT(*) AS count
            FROM decision_policy_tension_candidates_v1
            GROUP BY 1
            ORDER BY count DESC, settings_signature
            LIMIT 12
            """,
        )
        policy_risk_activity_counts = _query_rows(
            conn,
            """
            SELECT
                concat('R', risk_preference, ':A', trading_activity) AS risk_activity_cell,
                COUNT(*) AS count
            FROM decision_policy_tension_candidates_v1
            GROUP BY 1
            ORDER BY count DESC, risk_activity_cell
            LIMIT 12
            """,
        )
        policy_actionability_counts = _query_rows(
            conn,
            """
            SELECT
                CASE
                    WHEN can_buy_any AND can_sell_any THEN 'buy+sell'
                    WHEN can_buy_any THEN 'buy_only'
                    WHEN can_sell_any THEN 'sell_only'
                    ELSE 'none'
                END AS actionability_cell,
                COUNT(*) AS count
            FROM decision_policy_tension_candidates_v1
            GROUP BY 1
            ORDER BY count DESC, actionability_cell
            """,
        )
        buy_asset_counts = _query_rows(
            conn,
            """
            SELECT target_asset, COUNT(*) AS count
            FROM decision_trade_candidates_v1
            WHERE trade_side = 'buy'
            GROUP BY 1
            ORDER BY count DESC, target_asset
            LIMIT 10
            """,
        )
        sell_asset_counts = _query_rows(
            conn,
            """
            SELECT target_asset, COUNT(*) AS count
            FROM decision_sell_candidates_v1
            GROUP BY 1
            ORDER BY count DESC, target_asset
            LIMIT 10
            """,
        )

        blocked_rows = _query_rows(
            conn,
            """
            SELECT
                log_id,
                created_at,
                vault_address,
                capture_priority,
                block_reason,
                trade_size,
                trading_activity,
                holding_style,
                diversification,
                risk_preference,
                can_buy_any,
                can_sell_any,
                zero_eth,
                high_strategy_count,
                blocks_all_buys,
                blocks_all_sells
            FROM decision_blocked_observe_candidates_v1
            ORDER BY capture_priority DESC NULLS LAST, created_at DESC NULLS LAST, log_id DESC
            LIMIT %s
            """,
            [scan_limit],
        )
        policy_rows = _query_rows(
            conn,
            """
            SELECT
                log_id,
                created_at,
                vault_address,
                capture_priority,
                trade_size,
                trading_activity,
                holding_style,
                diversification,
                risk_preference,
                can_buy_any,
                can_sell_any,
                zero_eth,
                extreme_settings_count,
                within_settings_rank
            FROM decision_policy_tension_candidates_v1
            ORDER BY capture_priority DESC NULLS LAST, created_at DESC NULLS LAST, log_id DESC
            LIMIT %s
            """,
            [scan_limit],
        )
        buy_rows = _query_rows(
            conn,
            """
            SELECT
                log_id,
                created_at,
                vault_address,
                capture_priority,
                target_asset,
                trade_size,
                trading_activity,
                holding_style,
                diversification,
                risk_preference,
                can_buy_any,
                can_sell_any,
                zero_eth
            FROM decision_trade_candidates_v1
            WHERE trade_side = 'buy'
            ORDER BY capture_priority DESC NULLS LAST, created_at DESC NULLS LAST, log_id DESC
            LIMIT %s
            """,
            [scan_limit],
        )
        sell_rows = _query_rows(
            conn,
            """
            SELECT
                log_id,
                created_at,
                vault_address,
                capture_priority,
                target_asset,
                trade_size,
                trading_activity,
                holding_style,
                diversification,
                risk_preference,
                can_buy_any,
                can_sell_any,
                zero_eth
            FROM decision_sell_candidates_v1
            ORDER BY capture_priority DESC NULLS LAST, created_at DESC NULLS LAST, log_id DESC
            LIMIT %s
            """,
            [scan_limit],
        )

    blocked_rows = [annotate_kickoff_row(row, cohort_label="blocked_observe") for row in blocked_rows]
    policy_rows = [annotate_kickoff_row(row, cohort_label="policy_tension_observe") for row in policy_rows]
    buy_rows = [annotate_kickoff_row(row, cohort_label="buy") for row in buy_rows]
    sell_rows = [annotate_kickoff_row(row, cohort_label="sell") for row in sell_rows]

    blocked_plan = blocked_valence_manifest_plan()
    settings_plan = settings_twist_manifest_plan()
    blocked_manifest_rows = select_manifest_rows(
        candidates_by_cohort={
            "blocked_observe": blocked_rows,
        },
        plan=blocked_plan,
    )
    settings_manifest_rows = select_manifest_rows(
        candidates_by_cohort={
            "policy_tension_observe": policy_rows,
            "buy": buy_rows,
            "sell": sell_rows,
        },
        plan=settings_plan,
    )

    audit = {
        "roadmap": roadmap_as_dicts(),
        "candidate_pool_counts": overall,
        "blocked_reason_counts": {
            row["block_reason"]: int(row["count"])
            for row in blocked_reason_counts
        },
        "blocked_actionability_counts": {
            row["actionability_cell"]: int(row["count"])
            for row in blocked_actionability_counts
        },
        "policy_settings_signature_counts": {
            row["settings_signature"]: int(row["count"])
            for row in policy_settings_signature_counts
        },
        "policy_risk_activity_counts": {
            row["risk_activity_cell"]: int(row["count"])
            for row in policy_risk_activity_counts
        },
        "policy_actionability_counts": {
            row["actionability_cell"]: int(row["count"])
            for row in policy_actionability_counts
        },
        "trade_asset_counts": {
            "buy": {
                row["target_asset"]: int(row["count"])
                for row in buy_asset_counts
            },
            "sell": {
                row["target_asset"]: int(row["count"])
                for row in sell_asset_counts
            },
        },
        "kickoff_manifest_summary": {
            "blocked_valence": manifest_summary(blocked_manifest_rows),
            "settings_twist": manifest_summary(settings_manifest_rows),
        },
        "kickoff_manifest_preview": {
            "blocked_valence": {
                "blocked_observe": [
                    {
                        "log_id": row["log_id"],
                        "group_key": row.get("group_key"),
                        "settings_signature": row.get("settings_signature"),
                        "actionability_cell": row.get("actionability_cell"),
                        "capture_priority": row.get("capture_priority"),
                    }
                    for row in blocked_manifest_rows
                ][:10]
            },
            "settings_twist": {
                label: [
                    {
                        "log_id": row["log_id"],
                        "group_key": row.get("group_key"),
                        "target_asset": row.get("target_asset"),
                        "settings_signature": row.get("settings_signature"),
                        "actionability_cell": row.get("actionability_cell"),
                        "capture_priority": row.get("capture_priority"),
                    }
                    for row in settings_manifest_rows
                    if row.get("cohort_label") == label
                ][:10]
                for label in ("policy_tension_observe", "buy", "sell")
            },
        },
    }

    return {
        "audit": audit,
        "manifests": {
            "blocked_valence": {
                "plan": {
                    "manifest_name": blocked_plan.manifest_name,
                    "per_vault_cap": blocked_plan.per_vault_cap,
                    "min_spacing_minutes": blocked_plan.min_spacing_minutes,
                    "cohort_rules": [
                        {
                            "label": rule.label,
                            "target_count": rule.target_count,
                            "group_field": rule.group_field,
                            "max_per_group": rule.max_per_group,
                            "max_per_asset": rule.max_per_asset,
                            "max_per_vault": rule.max_per_vault,
                        }
                        for rule in blocked_plan.cohort_rules
                    ],
                },
                "rows": blocked_manifest_rows,
            },
            "settings_twist": {
                "plan": {
                    "manifest_name": settings_plan.manifest_name,
                    "per_vault_cap": settings_plan.per_vault_cap,
                    "min_spacing_minutes": settings_plan.min_spacing_minutes,
                    "cohort_rules": [
                        {
                            "label": rule.label,
                            "target_count": rule.target_count,
                            "group_field": rule.group_field,
                            "max_per_group": rule.max_per_group,
                            "max_per_asset": rule.max_per_asset,
                            "max_per_vault": rule.max_per_vault,
                        }
                        for rule in settings_plan.cohort_rules
                    ],
                },
                "rows": settings_manifest_rows,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the ranked research kickoff audit and balanced kickoff manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "analysis_results" / "research_kickoff",
    )
    parser.add_argument("--scan-limit", type=int, default=50000)
    args = parser.parse_args()

    payload = build_kickoff_payload(scan_limit=args.scan_limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (args.output_dir / "audit.json").write_text(json.dumps(payload["audit"], indent=2, default=str))
    for name, manifest in payload["manifests"].items():
        (args.output_dir / f"{name}_manifest.json").write_text(json.dumps(manifest["rows"], indent=2, default=str))
        (args.output_dir / f"{name}_manifest_plan.json").write_text(json.dumps(manifest["plan"], indent=2, default=str))

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "summary": payload["audit"]["kickoff_manifest_summary"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
