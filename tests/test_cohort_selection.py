from __future__ import annotations

import pytest

from pipelines.datasets.interp_query import (
    cohort_order_by_sql,
    build_interp_example_query,
    validate_order_mode,
    validate_relation_name,
)


def test_validate_relation_name_accepts_simple_identifier() -> None:
    assert validate_relation_name("decision_capture_priority_v1") == "decision_capture_priority_v1"


def test_validate_relation_name_rejects_unsafe_input() -> None:
    with pytest.raises(ValueError):
        validate_relation_name("decision_capture_priority_v1; drop table full_logs")


def test_validate_order_mode_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        validate_order_mode("priority-ish")


def test_cohort_order_by_sql_uses_priority_when_cohort_present() -> None:
    sql = cohort_order_by_sql("capture_priority_desc", has_cohort_view=True)
    assert "capture_priority" in sql
    assert "created_at" in sql


def test_cohort_order_by_sql_uses_selection_rank_when_available() -> None:
    sql = cohort_order_by_sql("selection_rank_asc", has_cohort_view=True)
    assert "selection_rank" in sql


def test_cohort_order_by_sql_falls_back_without_cohort() -> None:
    assert cohort_order_by_sql("capture_priority_desc", has_cohort_view=False) == "ie.log_id"


def test_build_interp_example_query_adds_cohort_join_and_market_filter() -> None:
    query, params = build_interp_example_query(
        select_columns=["ie.log_id", "ie.prompt_messages_json"],
        require_market_snapshot=True,
        cohort_view="decision_capture_priority_v1",
        order_mode="capture_priority_desc",
        limit=25,
    )
    assert "JOIN decision_capture_priority_v1 c ON c.log_id = ie.log_id" in query
    assert "ie.market_snapshot_json IS NOT NULL" in query
    assert "c.capture_priority DESC" in query
    assert params == [25]


def test_build_interp_example_query_uses_selection_rank_order() -> None:
    query, params = build_interp_example_query(
        select_columns=["ie.log_id"],
        require_market_snapshot=False,
        cohort_view="decision_capture_manifest_v1",
        order_mode="selection_rank_asc",
        limit=10,
    )
    assert "JOIN decision_capture_manifest_v1 c ON c.log_id = ie.log_id" in query
    assert "c.selection_rank ASC" in query
    assert params == [10]
