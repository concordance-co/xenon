from __future__ import annotations

from research.research_rerun.core import (
    _build_blocked_valence_prompts,
    _build_settings_twist_prompts,
    _slice_manifest_rows,
    clear_active_strategies,
    replace_section_body,
)


USER_TEXT = (
    "## OPERATING RULES\n\nIntro.\n\n"
    "## MARKET SNAPSHOT\n\n"
    "  - Alpha (A) | Price: 1\n"
    "    stuff\n\n"
    "## ACTIVE STRATEGIES\n\n"
    "[high] do not buy\n\n"
    "## ACTIVE SETTINGS\n\n"
    "- Trade Size: 3/5\n"
    "- Trading Activity: 4/5\n"
    "- Holding Style: 2/5\n"
    "- Diversification: 1/5\n"
    "- Asset Risk Preference: 5/5\n\n"
    "## PORTFOLIO CONTEXT\n\n"
    "Portfolio.\n\n"
    "## PREVIOUS DECISIONS\n\n"
    "Prev.\n"
)


def test_replace_section_body_preserves_neighbor_sections() -> None:
    updated = replace_section_body(USER_TEXT, "## ACTIVE STRATEGIES", "No active strategies.")

    assert "No active strategies." in updated
    assert "## ACTIVE SETTINGS" in updated
    assert "## PORTFOLIO CONTEXT" in updated
    assert "[high] do not buy" not in updated


def test_clear_active_strategies_only_rewrites_strategy_section() -> None:
    updated = clear_active_strategies(USER_TEXT)

    assert "## ACTIVE STRATEGIES" in updated
    assert "No active strategies." in updated
    assert "- Trade Size: 3/5" in updated
    assert "- Asset Risk Preference: 5/5" in updated


def test_replace_section_body_matches_decorated_strategy_header() -> None:
    user_text = USER_TEXT.replace(
        "## ACTIVE STRATEGIES",
        "## ACTIVE STRATEGIES (CURRENT ONLY)",
    )

    updated = clear_active_strategies(user_text)

    assert "## ACTIVE STRATEGIES (CURRENT ONLY)" in updated
    assert "No active strategies." in updated


def test_slice_manifest_rows_applies_per_cohort_limits() -> None:
    rows = [
        {"log_id": 1, "cohort_label": "blocked_observe"},
        {"log_id": 2, "cohort_label": "blocked_observe"},
        {"log_id": 3, "cohort_label": "policy_tension_observe"},
        {"log_id": 4, "cohort_label": "buy"},
        {"log_id": 5, "cohort_label": "sell"},
        {"log_id": 6, "cohort_label": "sell"},
    ]

    sliced = _slice_manifest_rows(
        rows,
        blocked_limit=1,
        policy_limit=1,
        buy_limit=1,
        sell_limit=1,
    )

    assert [row["log_id"] for row in sliced] == [1, 3, 4, 5]


def test_blocked_valence_prompt_builder_creates_original_and_clear_strategies() -> None:
    prompts = _build_blocked_valence_prompts(
        "exp1",
        {
            "example_id": "ex1",
            "log_id": 10,
            "system_text": "system",
            "user_text": USER_TEXT,
        },
        {
            "cohort_label": "blocked_observe",
            "block_reason": "high_strategy_present",
            "settings_signature": "3/4/2/1/5",
            "actionability_cell": "sell_only",
        },
        ["A"],
    )

    assert [prompt["variant"] for prompt in prompts] == ["original", "clear_strategies"]
    assert prompts[1]["experiment_group"] == "blocked_valence"
    assert "No active strategies." in prompts[1]["user_text"]


def test_settings_twist_prompt_builder_creates_three_variants() -> None:
    prompts = _build_settings_twist_prompts(
        "exp2",
        {
            "example_id": "ex2",
            "log_id": 11,
            "system_text": "system",
            "user_text": USER_TEXT,
        },
        {
            "cohort_label": "buy",
            "target_asset": "A",
            "settings_signature": "3/4/2/1/5",
            "actionability_cell": "buy+sell",
        },
        ["A"],
    )

    assert [prompt["variant"] for prompt in prompts] == ["original", "settings_all1", "settings_all5"]
    assert "- Trade Size: 1/5" in prompts[1]["user_text"]
    assert "- Trading Activity: 5/5" in prompts[2]["user_text"]


def test_settings_twist_prompt_builder_rewrites_decorated_slider_labels() -> None:
    decorated_user_text = (
        "## ACTIVE SETTINGS\n\n"
        "- Trading Activity (TA): 1 / 5\n"
        "- Asset Risk Preference (Risk): 2 / 5\n"
        "- Trade Size (Size): 3 / 5\n"
        "- Holding Style (Hold): 4 / 5\n"
        "- Diversification (Div): 5 / 5\n\n"
        "## PORTFOLIO CONTEXT\n\n"
    )

    prompts = _build_settings_twist_prompts(
        "exp3",
        {
            "example_id": "ex3",
            "log_id": 12,
            "system_text": "system",
            "user_text": decorated_user_text,
        },
        {
            "cohort_label": "policy_tension_observe",
            "settings_signature": "1/2/3/4/5",
            "actionability_cell": "buy+sell",
        },
        ["A"],
    )

    settings_all1 = next(prompt for prompt in prompts if prompt["variant"] == "settings_all1")
    settings_all5 = next(prompt for prompt in prompts if prompt["variant"] == "settings_all5")

    assert "Trading Activity (TA): 1 / 5" in settings_all1["user_text"]
    assert "Trade Size (Size): 1 / 5" in settings_all1["user_text"]
    assert "Diversification (Div): 1 / 5" in settings_all1["user_text"]
    assert "Trading Activity (TA): 5 / 5" in settings_all5["user_text"]
    assert "Asset Risk Preference (Risk): 5 / 5" in settings_all5["user_text"]
    assert settings_all1["user_text"] != settings_all5["user_text"]
