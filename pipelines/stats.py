"""Shared dashboard stats computation.

Single source of truth for the stats JSON structure consumed by the dashboard.
Used by both modal_ingest (snapshot to file) and backend/app (live endpoint).
"""

from __future__ import annotations

from typing import Any

import psycopg

# The canonical list of tables shown in the dashboard.
_DASHBOARD_TABLES = [
    "vaults", "strategies", "inference_logs", "full_logs",
    "swaps", "trade_outcomes", "interp_examples_v0",
]


def _safe_round(value: Any, decimals: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), decimals)


def _table_counts_exact(conn: psycopg.Connection) -> dict[str, int]:
    """Exact row counts via COUNT(*). Slower but accurate."""
    existing = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
    ).fetchall()
    existing_names = {r["table_name"] for r in existing}

    counts: dict[str, int] = {}
    for name in _DASHBOARD_TABLES:
        if name in existing_names:
            row = conn.execute(f'SELECT COUNT(*) AS n FROM "{name}"').fetchone()  # noqa: S608
            counts[name] = int(row["n"]) if row else 0
    return counts


def _table_counts_approximate(conn: psycopg.Connection) -> dict[str, int]:
    """Approximate row counts via pg_class statistics. Instant."""
    rows = conn.execute(
        "SELECT relname AS name, reltuples::bigint AS count "
        "FROM pg_class WHERE relname = ANY(%s) AND relkind = 'r'",
        [_DASHBOARD_TABLES],
    ).fetchall()
    return {r["name"]: max(int(r["count"]), 0) for r in rows}


def compute_stats(
    conn: psycopg.Connection,
    *,
    approximate_counts: bool = False,
) -> dict[str, Any]:
    """Compute the full dashboard stats dict from Neon.

    Args:
        conn: A psycopg connection with dict_row factory.
        approximate_counts: If True, use pg_class estimates for table counts
            (fast but approximate). If False, use exact COUNT(*).
    """
    counts = (
        _table_counts_approximate(conn) if approximate_counts
        else _table_counts_exact(conn)
    )

    tables = [{"name": t, "count": counts.get(t, 0)} for t in _DASHBOARD_TABLES if t in counts]

    # --- Ingest ---
    vc = counts.get("vaults", 0)
    sc = counts.get("strategies", 0)
    lc = counts.get("inference_logs", 0)
    flc = counts.get("full_logs", 0)
    coverage = round((flc / lc) * 100, 1) if lc else 0

    parse_errors = 0
    if "full_logs" in counts:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM full_logs "
            "WHERE parse_error IS NOT NULL AND parse_error != ''"
        ).fetchone()
        parse_errors = int(row["n"]) if row else 0

    ingest = {
        "vault_count": vc,
        "strategy_count": sc,
        "log_count": lc,
        "full_log_count": flc,
        "full_log_coverage_pct": coverage,
        "parse_error_count": parse_errors,
        "tables": tables,
    }

    # --- Outcomes ---
    total_swaps = counts.get("swaps", 0)
    outcomes: dict[str, Any] = {
        "total_outcomes": 0,
        "unlabeled_swaps": total_swaps,
        "total_swaps": total_swaps,
        "avg_pnl_1h": None,
        "avg_pnl_4h": None,
        "avg_pnl_1d": None,
        "win_rate_1h": None,
        "risk_breakdown": [],
    }

    if "trade_outcomes" in counts:
        oc = counts["trade_outcomes"]
        outcomes["total_outcomes"] = oc
        outcomes["unlabeled_swaps"] = max(0, total_swaps - oc)

        if oc > 0:
            agg = conn.execute(
                "SELECT "
                "  AVG(pnl_1h_pct) AS avg_1h, "
                "  AVG(pnl_4h_pct) AS avg_4h, "
                "  AVG(pnl_1d_pct) AS avg_1d, "
                "  AVG(CASE WHEN was_profitable_1h THEN 1.0 ELSE 0.0 END) AS wr_1h "
                "FROM trade_outcomes WHERE pnl_1h_pct IS NOT NULL"
            ).fetchone()
            if agg:
                outcomes["avg_pnl_1h"] = _safe_round(agg["avg_1h"])
                outcomes["avg_pnl_4h"] = _safe_round(agg["avg_4h"])
                outcomes["avg_pnl_1d"] = _safe_round(agg["avg_1d"])
                outcomes["win_rate_1h"] = _safe_round(agg["wr_1h"])

            risk_rows = conn.execute(
                "SELECT "
                "  COALESCE(v.asset_risk_preference, 0) AS risk_level, "
                "  COUNT(*) AS cnt, "
                "  AVG(t.pnl_1h_pct) AS avg_1h, "
                "  AVG(t.pnl_4h_pct) AS avg_4h, "
                "  AVG(t.pnl_1d_pct) AS avg_1d, "
                "  AVG(CASE WHEN t.was_profitable_1h THEN 1.0 ELSE 0.0 END) AS wr_1h "
                "FROM trade_outcomes t "
                "JOIN swaps s ON s.log_id = t.log_id "
                "JOIN vaults v ON v.vault_address = s.vault_address "
                "WHERE t.pnl_1h_pct IS NOT NULL "
                "GROUP BY risk_level ORDER BY risk_level"
            ).fetchall()
            outcomes["risk_breakdown"] = [
                {
                    "risk_level": int(r["risk_level"]),
                    "count": r["cnt"],
                    "avg_pnl_1h": _safe_round(r["avg_1h"]),
                    "avg_pnl_4h": _safe_round(r["avg_4h"]),
                    "avg_pnl_1d": _safe_round(r["avg_1d"]),
                    "win_rate_1h": _safe_round(r["wr_1h"]),
                }
                for r in risk_rows
            ]

    # --- Prep ---
    prep: dict[str, Any] = {
        "total_examples": 0,
        "high_quality": 0,
        "medium_quality": 0,
        "low_quality": 0,
        "trade_count": 0,
        "observation_count": 0,
        "label_distribution": [],
    }

    if "interp_examples_v0" in counts:
        quality_rows = conn.execute(
            "SELECT label_quality, COUNT(*) AS n "
            "FROM interp_examples_v0 GROUP BY label_quality"
        ).fetchall()
        quality_map = {r["label_quality"]: int(r["n"]) for r in quality_rows}
        prep["total_examples"] = sum(quality_map.values())
        prep["high_quality"] = quality_map.get("high", 0)
        prep["medium_quality"] = quality_map.get("medium", 0)
        prep["low_quality"] = quality_map.get("low", 0)

        dt_rows = conn.execute(
            "SELECT decision_type, COUNT(*) AS count, "
            "  STRING_AGG(DISTINCT trade_side, ',') AS trade_side, "
            "  AVG(vault_risk_preference) AS avg_risk "
            "FROM interp_examples_v0 GROUP BY decision_type"
        ).fetchall()
        label_dist = [
            {
                "decision_type": r["decision_type"],
                "count": r["count"],
                "trade_side": r["trade_side"],
                "avg_risk": _safe_round(r["avg_risk"]),
            }
            for r in dt_rows
        ]
        prep["label_distribution"] = label_dist
        prep["trade_count"] = sum(
            r["count"] for r in label_dist if r["decision_type"] == "trade"
        )
        prep["observation_count"] = sum(
            r["count"] for r in label_dist if r["decision_type"] == "record_observation"
        )

    return {"ingest": ingest, "outcomes": outcomes, "prep": prep}
