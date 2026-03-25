#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.db import connect_neon


DEFAULT_SQL = REPO_ROOT / "sql" / "decision_cohort_views.sql"
VIEW_NAMES = [
    "decision_capture_base_v1",
    "decision_trade_candidates_v1",
    "decision_sell_candidates_v1",
    "decision_blocked_observe_candidates_v1",
    "decision_policy_tension_candidates_v1",
    "decision_capture_priority_v1",
]
MATVIEW_NAMES = [
    "decision_capture_base_mv_v1",
    "decision_capture_priority_v1",
]
DROP_ORDER = [
    "decision_trade_candidates_v1",
    "decision_sell_candidates_v1",
    "decision_blocked_observe_candidates_v1",
    "decision_policy_tension_candidates_v1",
    "decision_capture_priority_v1",
    "decision_capture_base_v1",
    "decision_capture_base_mv_v1",
]


def _drop_relation(conn, name: str) -> None:
    row = conn.execute(
        """
        SELECT c.relkind
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relname = %s
        """,
        [name],
    ).fetchone()
    if not row:
        return
    relkind = row["relkind"]
    if relkind == "m":
        conn.execute(f"DROP MATERIALIZED VIEW IF EXISTS {name}")
    elif relkind == "v":
        conn.execute(f"DROP VIEW IF EXISTS {name}")


def _apply(sql_path: Path) -> None:
    sql = sql_path.read_text()
    with connect_neon() as conn:
        for name in DROP_ORDER:
            _drop_relation(conn, name)
        conn.execute(sql)
    print(f"Applied cohort views from {sql_path}")


def _stats() -> None:
    query = """
        SELECT json_build_object(
            'decision_capture_base_v1', (SELECT COUNT(*) FROM decision_capture_base_v1),
            'decision_trade_candidates_v1', (SELECT COUNT(*) FROM decision_trade_candidates_v1),
            'decision_sell_candidates_v1', (SELECT COUNT(*) FROM decision_sell_candidates_v1),
            'decision_blocked_observe_candidates_v1', (SELECT COUNT(*) FROM decision_blocked_observe_candidates_v1),
            'decision_policy_tension_candidates_v1', (SELECT COUNT(*) FROM decision_policy_tension_candidates_v1),
            'decision_capture_priority_v1', (SELECT COUNT(*) FROM decision_capture_priority_v1)
        ) AS stats
    """
    with connect_neon() as conn:
        row = conn.execute(query).fetchone()
    print(json.dumps(row["stats"], indent=2, default=str))


def _refresh() -> None:
    with connect_neon() as conn:
        for name in MATVIEW_NAMES:
            conn.execute(f"REFRESH MATERIALIZED VIEW {name}")
            print(f"Refreshed {name}")


def _sample(view_name: str, limit: int) -> None:
    if view_name not in VIEW_NAMES:
        raise ValueError(f"Unknown cohort view: {view_name!r}")
    query = f"SELECT * FROM {view_name} ORDER BY created_at DESC NULLS LAST, log_id DESC LIMIT %s"
    with connect_neon() as conn:
        rows = conn.execute(query, [limit]).fetchall()
    print(json.dumps([dict(r) for r in rows], indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply and inspect decision cohort views in Neon.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_apply = sub.add_parser("apply", help="Create or replace the cohort views")
    p_apply.add_argument("--sql-file", type=Path, default=DEFAULT_SQL)

    sub.add_parser("refresh", help="Refresh the materialized cohort relations")
    sub.add_parser("stats", help="Show row counts for each cohort view")

    p_sample = sub.add_parser("sample", help="Sample rows from a cohort view")
    p_sample.add_argument("view_name", choices=VIEW_NAMES)
    p_sample.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    if args.command == "apply":
        _apply(args.sql_file)
    elif args.command == "refresh":
        _refresh()
    elif args.command == "stats":
        _stats()
    elif args.command == "sample":
        _sample(args.view_name, args.limit)


if __name__ == "__main__":
    main()
