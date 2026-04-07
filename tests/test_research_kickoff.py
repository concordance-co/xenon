from __future__ import annotations

from projects.DX_TERMINAL.research_kickoff.core import (
    actionability_cell,
    annotate_kickoff_row,
    blocked_valence_manifest_plan,
    ranked_research_tracks,
    research_kickoff_manifest_plan,
    risk_activity_cell,
    settings_signature,
    settings_twist_manifest_plan,
)


def test_ranked_research_tracks_sorted_and_top_track_is_blocked_valence() -> None:
    tracks = ranked_research_tracks()
    assert [track.rank for track in tracks] == [1, 2, 3, 4, 5]
    assert tracks[0].title == "Blocked Valence + Settings Twist"
    assert tracks[0].score > tracks[-1].score


def test_settings_signature_and_cells_are_derived_stably() -> None:
    row = {
        "trade_size": 5,
        "trading_activity": 1,
        "holding_style": 3,
        "diversification": 4,
        "risk_preference": 2,
        "can_buy_any": True,
        "can_sell_any": False,
    }
    assert settings_signature(row) == "5/1/3/4/2"
    assert risk_activity_cell(row) == "R2:A1"
    assert actionability_cell(row) == "buy_only"


def test_annotate_kickoff_row_preserves_original_fields_and_adds_derived_fields() -> None:
    row = {
        "log_id": 123,
        "trade_size": 1,
        "trading_activity": 5,
        "holding_style": 2,
        "diversification": 4,
        "risk_preference": 5,
        "can_buy_any": True,
        "can_sell_any": True,
    }
    annotated = annotate_kickoff_row(row, cohort_label="policy_tension_observe")
    assert annotated["log_id"] == 123
    assert annotated["cohort_label"] == "policy_tension_observe"
    assert annotated["settings_signature"] == "1/5/2/4/5"
    assert annotated["risk_activity_cell"] == "R5:A5"
    assert annotated["actionability_cell"] == "buy+sell"


def test_research_kickoff_manifest_plan_targets_balanced_kickoff_set() -> None:
    plan = research_kickoff_manifest_plan()
    assert plan.manifest_name == "research_kickoff_v1"
    assert plan.per_vault_cap == 2
    assert plan.min_spacing_minutes == 30
    assert [rule.label for rule in plan.cohort_rules] == [
        "blocked_observe",
        "policy_tension_observe",
        "buy",
        "sell",
    ]
    assert sum(rule.target_count for rule in plan.cohort_rules) == 216


def test_split_manifest_plans_match_top_ranked_tracks() -> None:
    blocked = blocked_valence_manifest_plan()
    settings = settings_twist_manifest_plan()
    assert blocked.manifest_name == "blocked_valence_kickoff_v1"
    assert settings.manifest_name == "settings_twist_kickoff_v1"
    assert [rule.label for rule in blocked.cohort_rules] == ["blocked_observe"]
    assert [rule.label for rule in settings.cohort_rules] == [
        "policy_tension_observe",
        "buy",
        "sell",
    ]
