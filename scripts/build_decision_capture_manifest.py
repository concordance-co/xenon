#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from psycopg import sql
from psycopg.types.json import Jsonb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.db import connect_neon
from pipelines.interp.decision_structure.manifest import (
    CohortRule,
    ManifestPlan,
    manifest_summary,
    plan_to_json,
    select_manifest_rows,
)


MANIFEST_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_manifest_name(name: str) -> str:
    if not MANIFEST_NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid manifest name: {name!r}")
    return name


def _json_safe(value):
    return json.loads(json.dumps(value, default=str))


def _ensure_tables(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_capture_manifests (
            manifest_name TEXT PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            plan_json JSONB NOT NULL,
            summary_json JSONB NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_capture_manifest_items (
            manifest_name TEXT NOT NULL,
            selection_rank INT NOT NULL,
            log_id BIGINT NOT NULL,
            cohort_label TEXT NOT NULL,
            cohort_rank INT NOT NULL,
            group_key TEXT,
            target_asset TEXT,
            vault_address TEXT,
            trade_side TEXT,
            created_at TIMESTAMPTZ,
            capture_priority INT,
            metadata_json JSONB NOT NULL,
            PRIMARY KEY (manifest_name, log_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_decision_capture_manifest_items_manifest_rank
            ON decision_capture_manifest_items (manifest_name, selection_rank)
        """
    )


def _build_plan(args: argparse.Namespace) -> ManifestPlan:
    return ManifestPlan(
        manifest_name=_validate_manifest_name(args.manifest_name),
        per_vault_cap=args.per_vault_cap,
        min_spacing_minutes=args.min_spacing_minutes,
        cohort_rules=[
            CohortRule(
                label="buy",
                target_count=args.buy_count,
                group_field="target_asset",
                max_per_group=args.max_per_trade_asset,
                max_per_asset=args.max_per_trade_asset,
                max_per_vault=args.max_per_trade_vault,
            ),
            CohortRule(
                label="sell",
                target_count=args.sell_count,
                group_field="target_asset",
                max_per_group=args.max_per_trade_asset,
                max_per_asset=args.max_per_trade_asset,
                max_per_vault=args.max_per_trade_vault,
            ),
            CohortRule(
                label="blocked_observe",
                target_count=args.blocked_observe_count,
                group_field="block_reason",
                max_per_group=args.max_per_block_reason,
                max_per_vault=args.max_per_observe_vault,
            ),
            CohortRule(
                label="policy_tension_observe",
                target_count=args.policy_tension_count,
                group_field="settings_cell",
                max_per_group=args.max_per_settings_cell,
                max_per_vault=args.max_per_observe_vault,
            ),
        ],
    )


def _fetch_candidates(conn, *, label: str, scan_limit: int) -> list[dict]:
    if label == "buy":
        query = """
            SELECT
                log_id,
                created_at,
                vault_address,
                target_asset,
                trade_side,
                capture_priority
            FROM decision_capture_priority_v1
            WHERE cohort_label = 'buy'
            ORDER BY capture_priority DESC NULLS LAST, created_at DESC NULLS LAST, log_id DESC
            LIMIT %s
        """
    elif label == "sell":
        query = """
            SELECT
                log_id,
                created_at,
                vault_address,
                target_asset,
                trade_side,
                capture_priority
            FROM decision_capture_priority_v1
            WHERE cohort_label = 'sell'
            ORDER BY capture_priority DESC NULLS LAST, created_at DESC NULLS LAST, log_id DESC
            LIMIT %s
        """
    elif label == "blocked_observe":
        query = """
            SELECT
                log_id,
                created_at,
                vault_address,
                target_asset,
                trade_side,
                capture_priority,
                block_reason
            FROM decision_blocked_observe_candidates_v1
            ORDER BY capture_priority DESC NULLS LAST, created_at DESC NULLS LAST, log_id DESC
            LIMIT %s
        """
    elif label == "policy_tension_observe":
        query = """
            SELECT
                log_id,
                created_at,
                vault_address,
                target_asset,
                trade_side,
                capture_priority,
                risk_preference,
                trading_activity,
                CONCAT(COALESCE(risk_preference::text, 'NA'), ':', COALESCE(trading_activity::text, 'NA')) AS settings_cell
            FROM decision_policy_tension_candidates_v1
            ORDER BY capture_priority DESC NULLS LAST, created_at DESC NULLS LAST, log_id DESC
            LIMIT %s
        """
    else:
        raise ValueError(f"Unsupported cohort label: {label!r}")
    return [dict(r) for r in conn.execute(query, [scan_limit]).fetchall()]


def _persist_manifest(conn, *, plan: ManifestPlan, selected_rows: list[dict]) -> None:
    summary = manifest_summary(selected_rows)
    conn.execute(
        "DELETE FROM decision_capture_manifest_items WHERE manifest_name = %s",
        [plan.manifest_name],
    )
    conn.execute(
        """
        INSERT INTO decision_capture_manifests (manifest_name, plan_json, summary_json)
        VALUES (%s, %s, %s)
        ON CONFLICT (manifest_name) DO UPDATE SET
            created_at = now(),
            plan_json = EXCLUDED.plan_json,
            summary_json = EXCLUDED.summary_json
        """,
        [plan.manifest_name, Jsonb(json.loads(plan_to_json(plan))), Jsonb(summary)],
    )

    rows = []
    for idx, row in enumerate(selected_rows, start=1):
        rows.append(
            (
                plan.manifest_name,
                idx,
                int(row["log_id"]),
                row["cohort_label"],
                int(row["cohort_rank"]),
                row.get("group_key"),
                row.get("target_asset"),
                row.get("vault_address"),
                row.get("trade_side"),
                row.get("created_at"),
                int(row.get("capture_priority") or 0),
                Jsonb(_json_safe(row)),
            )
        )
    if rows:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO decision_capture_manifest_items (
                    manifest_name,
                    selection_rank,
                    log_id,
                    cohort_label,
                    cohort_rank,
                    group_key,
                    target_asset,
                    vault_address,
                    trade_side,
                    created_at,
                    capture_priority,
                    metadata_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )

    view_sql = sql.SQL(
        """
        CREATE OR REPLACE VIEW decision_capture_manifest_v1 AS
        SELECT
            manifest_name,
            selection_rank,
            log_id,
            cohort_label,
            cohort_rank,
            group_key,
            target_asset,
            vault_address,
            trade_side,
            created_at,
            capture_priority
        FROM decision_capture_manifest_items
        WHERE manifest_name = {manifest_name}
        ORDER BY selection_rank
        """
    ).format(manifest_name=sql.Literal(plan.manifest_name))
    conn.execute(view_sql)


def _build_manifest(args: argparse.Namespace) -> None:
    plan = _build_plan(args)
    scan_multiplier = max(1, args.scan_multiplier)
    max_scan = max(1000, args.max_scan_per_cohort)

    with connect_neon() as conn:
        _ensure_tables(conn)
        candidates_by_cohort: dict[str, list[dict]] = {}
        for rule in plan.cohort_rules:
            scan_limit = min(max_scan, max(rule.target_count * scan_multiplier, rule.target_count))
            candidates_by_cohort[rule.label] = _fetch_candidates(conn, label=rule.label, scan_limit=scan_limit)

        selected = select_manifest_rows(candidates_by_cohort=candidates_by_cohort, plan=plan)
        _persist_manifest(conn, plan=plan, selected_rows=selected)

    payload = {
        "manifest_name": plan.manifest_name,
        "requested_total": sum(rule.target_count for rule in plan.cohort_rules),
        "selected_total": len(selected),
        "summary": manifest_summary(selected),
    }
    print(json.dumps(payload, indent=2, default=str))


def _stats(manifest_name: str) -> None:
    manifest_name = _validate_manifest_name(manifest_name)
    with connect_neon() as conn:
        row = conn.execute(
            """
            SELECT plan_json, summary_json, created_at
            FROM decision_capture_manifests
            WHERE manifest_name = %s
            """,
            [manifest_name],
        ).fetchone()
    if not row:
        raise ValueError(f"Manifest {manifest_name!r} not found")
    print(
        json.dumps(
            {
                "manifest_name": manifest_name,
                "created_at": row["created_at"],
                "plan": row["plan_json"],
                "summary": row["summary_json"],
            },
            indent=2,
            default=str,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced decision-capture manifest in Neon.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Create or replace a manifest and publish decision_capture_manifest_v1")
    p_build.add_argument("--manifest-name", default="balanced_v1")
    p_build.add_argument("--buy-count", type=int, default=300)
    p_build.add_argument("--sell-count", type=int, default=300)
    p_build.add_argument("--blocked-observe-count", type=int, default=200)
    p_build.add_argument("--policy-tension-count", type=int, default=200)
    p_build.add_argument("--per-vault-cap", type=int, default=4)
    p_build.add_argument("--min-spacing-minutes", type=int, default=0)
    p_build.add_argument("--max-per-trade-asset", type=int, default=60)
    p_build.add_argument("--max-per-trade-vault", type=int, default=1)
    p_build.add_argument("--max-per-observe-vault", type=int, default=1)
    p_build.add_argument("--max-per-block-reason", type=int, default=150)
    p_build.add_argument("--max-per-settings-cell", type=int, default=75)
    p_build.add_argument("--scan-multiplier", type=int, default=200)
    p_build.add_argument("--max-scan-per-cohort", type=int, default=70000)

    p_stats = sub.add_parser("stats", help="Show manifest plan and summary")
    p_stats.add_argument("--manifest-name", default="balanced_v1")

    args = parser.parse_args()

    if args.command == "build":
        _build_manifest(args)
    elif args.command == "stats":
        _stats(args.manifest_name)


if __name__ == "__main__":
    main()
