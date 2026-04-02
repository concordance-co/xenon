from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pipelines.ingest.manifest import CohortRule, ManifestPlan, manifest_summary, select_manifest_rows


def _row(
    *,
    log_id: int,
    cohort: str,
    vault: str,
    minutes: int,
    asset: str | None = None,
    group: str | None = None,
    priority: int = 1000,
) -> dict:
    base = {
        "log_id": log_id,
        "vault_address": vault,
        "created_at": (datetime(2026, 3, 20, tzinfo=UTC) + timedelta(minutes=minutes)).isoformat(),
        "target_asset": asset,
        "capture_priority": priority,
    }
    if cohort == "blocked_observe":
        base["block_reason"] = group
    if cohort == "policy_tension_observe":
        base["settings_cell"] = group
    return base


def test_select_manifest_rows_respects_vault_spacing_and_asset_caps() -> None:
    plan = ManifestPlan(
        manifest_name="test_v1",
        per_vault_cap=2,
        min_spacing_minutes=20,
        cohort_rules=[
            CohortRule(label="buy", target_count=2, group_field="target_asset", max_per_group=1, max_per_asset=1),
            CohortRule(label="sell", target_count=1, group_field="target_asset", max_per_group=1, max_per_asset=1),
        ],
    )
    candidates = {
        "buy": [
            _row(log_id=1, cohort="buy", vault="v1", minutes=0, asset="POOPCOIN", priority=100),
            _row(log_id=2, cohort="buy", vault="v1", minutes=5, asset="POOPCOIN", priority=99),
            _row(log_id=3, cohort="buy", vault="v2", minutes=10, asset="HOTDOGZ", priority=98),
        ],
        "sell": [
            _row(log_id=4, cohort="sell", vault="v1", minutes=40, asset="AIGF", priority=97),
        ],
    }
    selected = select_manifest_rows(candidates_by_cohort=candidates, plan=plan)
    assert [row["log_id"] for row in selected] == [1, 4, 3]


def test_select_manifest_rows_uses_group_caps_for_observe_cohorts() -> None:
    plan = ManifestPlan(
        manifest_name="test_v1",
        per_vault_cap=10,
        min_spacing_minutes=0,
        cohort_rules=[
            CohortRule(label="blocked_observe", target_count=3, group_field="block_reason", max_per_group=1),
            CohortRule(label="policy_tension_observe", target_count=2, group_field="settings_cell", max_per_group=1),
        ],
    )
    candidates = {
        "blocked_observe": [
            _row(log_id=1, cohort="blocked_observe", vault="v1", minutes=0, group="no_eth", priority=100),
            _row(log_id=2, cohort="blocked_observe", vault="v2", minutes=1, group="no_eth", priority=99),
            _row(log_id=3, cohort="blocked_observe", vault="v3", minutes=2, group="high_strategy_present", priority=98),
            _row(log_id=4, cohort="blocked_observe", vault="v4", minutes=3, group="strategy_blocks_both", priority=97),
        ],
        "policy_tension_observe": [
            _row(log_id=5, cohort="policy_tension_observe", vault="v5", minutes=4, group="1:1", priority=96),
            _row(log_id=6, cohort="policy_tension_observe", vault="v6", minutes=5, group="1:1", priority=95),
            _row(log_id=7, cohort="policy_tension_observe", vault="v7", minutes=6, group="5:5", priority=94),
        ],
    }
    selected = select_manifest_rows(candidates_by_cohort=candidates, plan=plan)
    assert [row["log_id"] for row in selected] == [1, 5, 3, 7, 4]


def test_manifest_summary_counts_assets_and_cohorts() -> None:
    rows = [
        {"log_id": 1, "cohort_label": "buy", "target_asset": "POOPCOIN", "vault_address": "v1"},
        {"log_id": 2, "cohort_label": "buy", "target_asset": "POOPCOIN", "vault_address": "v1"},
        {"log_id": 3, "cohort_label": "sell", "target_asset": "HOTDOGZ", "vault_address": "v2"},
    ]
    summary = manifest_summary(rows)
    assert summary["row_count"] == 3
    assert summary["cohort_counts"] == {"buy": 2, "sell": 1}
    assert summary["top_assets"]["POOPCOIN"] == 2


def test_select_manifest_rows_round_robins_cohorts_under_shared_vault_caps() -> None:
    plan = ManifestPlan(
        manifest_name="test_v1",
        per_vault_cap=2,
        min_spacing_minutes=0,
        cohort_rules=[
            CohortRule(label="buy", target_count=2, group_field="target_asset", max_per_group=2, max_per_asset=2),
            CohortRule(label="sell", target_count=2, group_field="target_asset", max_per_group=2, max_per_asset=2),
            CohortRule(label="policy_tension_observe", target_count=2, group_field="settings_cell", max_per_group=2),
        ],
    )
    candidates = {
        "buy": [
            _row(log_id=1, cohort="buy", vault="v1", minutes=0, asset="POOPCOIN", priority=100),
            _row(log_id=2, cohort="buy", vault="v1", minutes=1, asset="HOTDOGZ", priority=99),
            _row(log_id=3, cohort="buy", vault="v2", minutes=2, asset="AIGF", priority=98),
        ],
        "sell": [
            _row(log_id=4, cohort="sell", vault="v1", minutes=3, asset="POOPCOIN", priority=97),
            _row(log_id=5, cohort="sell", vault="v2", minutes=4, asset="HOTDOGZ", priority=96),
            _row(log_id=6, cohort="sell", vault="v2", minutes=5, asset="AIGF", priority=95),
        ],
        "policy_tension_observe": [
            _row(log_id=7, cohort="policy_tension_observe", vault="v1", minutes=6, group="1:1", priority=94),
            _row(log_id=8, cohort="policy_tension_observe", vault="v3", minutes=7, group="5:5", priority=93),
            _row(log_id=9, cohort="policy_tension_observe", vault="v3", minutes=8, group="4:4", priority=92),
        ],
    }
    selected = select_manifest_rows(candidates_by_cohort=candidates, plan=plan)
    assert [row["cohort_label"] for row in selected[:5]] == [
        "buy",
        "sell",
        "policy_tension_observe",
        "buy",
        "sell",
    ]
    assert [row["log_id"] for row in selected] == [1, 4, 8, 3, 5, 9]
