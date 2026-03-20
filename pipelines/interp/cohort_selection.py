from __future__ import annotations

import re
from typing import Any

_RELATION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ORDER_MODES = {
    "log_id",
    "created_at_desc",
    "capture_priority_desc",
    "selection_rank_asc",
    "hash",
}


def validate_relation_name(name: str | None) -> str | None:
    if name is None:
        return None
    text = name.strip()
    if not text:
        return None
    if not _RELATION_NAME_RE.fullmatch(text):
        raise ValueError(f"Invalid relation name: {name!r}")
    return text


def validate_order_mode(mode: str) -> str:
    text = mode.strip().lower()
    if text not in _ORDER_MODES:
        raise ValueError(
            f"Invalid order mode: {mode!r}. Expected one of {sorted(_ORDER_MODES)}"
        )
    return text


def cohort_order_by_sql(mode: str, *, has_cohort_view: bool) -> str:
    mode = validate_order_mode(mode)
    if mode == "capture_priority_desc" and has_cohort_view:
        return "c.capture_priority DESC NULLS LAST, c.created_at DESC NULLS LAST, ie.log_id DESC"
    if mode == "selection_rank_asc" and has_cohort_view:
        return "c.selection_rank ASC NULLS LAST, ie.log_id"
    if mode == "created_at_desc" and has_cohort_view:
        return "c.created_at DESC NULLS LAST, ie.log_id DESC"
    if mode == "hash":
        return "md5(ie.log_id::text)"
    return "ie.log_id"


def build_interp_example_query(
    *,
    select_columns: list[str],
    require_market_snapshot: bool,
    cohort_view: str | None,
    order_mode: str,
    limit: int | None,
) -> tuple[str, list[Any]]:
    if not select_columns:
        raise ValueError("select_columns must not be empty")

    validated_view = validate_relation_name(cohort_view)
    join_sql = f"\n        JOIN {validated_view} c ON c.log_id = ie.log_id" if validated_view else ""
    order_sql = cohort_order_by_sql(order_mode, has_cohort_view=bool(validated_view))

    where_clauses = [
        "ie.label_quality IN ('high', 'medium')",
        "ie.prompt_messages_json IS NOT NULL",
    ]
    if require_market_snapshot:
        where_clauses.append("ie.market_snapshot_json IS NOT NULL")

    query = f"""
        SELECT {", ".join(select_columns)}
        FROM interp_examples_v0 ie{join_sql}
        WHERE {" AND ".join(where_clauses)}
        ORDER BY {order_sql}
    """
    params: list[Any] = []
    if limit is not None:
        query += "\n        LIMIT %s"
        params.append(limit)
    return query, params
