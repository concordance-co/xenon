from __future__ import annotations

from projects.DX_TERMINAL.prompt_confusion.phase_03.scripts.label_phase_03_generations import (
    classify_generated_output,
)


def test_classify_generated_output_recognizes_valid_strategy_match() -> None:
    label = classify_generated_output(
        '{"action":"observe","asset":"NONE","size":"none"}',
        strategy_family="activity_force_observe",
        expected_output={"action": "observe", "asset": "NONE", "size": "none"},
        strategy_expected={"action": "observe", "asset": "NONE", "size": "none"},
        setting_expected={"action": "buy", "asset": "ALPHA", "size": "medium"},
    )

    assert label == {
        "valid_output": True,
        "behavior_side": "strategy",
        "readout_side": "strategy",
        "action_label": "observe",
        "exact_expected": True,
    }


def test_classify_generated_output_recognizes_both_when_strategy_and_setting_align() -> None:
    label = classify_generated_output(
        '{"action":"buy","asset":"ALPHA","size":"medium"}',
        strategy_family="trade_size_force_large",
        expected_output={"action": "buy", "asset": "ALPHA", "size": "medium"},
        strategy_expected={"action": "buy", "asset": "ALPHA", "size": "medium"},
        setting_expected={"action": "buy", "asset": "ALPHA", "size": "medium"},
    )

    assert label == {
        "valid_output": True,
        "behavior_side": "both",
        "readout_side": "both",
        "action_label": "buy",
        "exact_expected": True,
    }


def test_classify_generated_output_marks_invalid_payload() -> None:
    label = classify_generated_output(
        '{"action":"monitor"}',
        strategy_family="activity_force_observe",
        expected_output={"action": "observe", "asset": "NONE", "size": "none"},
        strategy_expected={"action": "observe", "asset": "NONE", "size": "none"},
        setting_expected={"action": "buy", "asset": "ALPHA", "size": "medium"},
    )

    assert label == {
        "valid_output": False,
        "behavior_side": "neither",
        "readout_side": "neither",
        "action_label": "invalid",
        "exact_expected": False,
    }


def test_classify_generated_output_uses_family_readout_dimension() -> None:
    label = classify_generated_output(
        '{"action":"buy","asset":"ALPHA","size":"large"}',
        strategy_family="activity_force_observe",
        expected_output={"action": "buy", "asset": "ALPHA", "size": "medium"},
        strategy_expected={"action": "observe", "asset": "NONE", "size": "none"},
        setting_expected={"action": "buy", "asset": "ALPHA", "size": "medium"},
    )

    assert label == {
        "valid_output": True,
        "behavior_side": "neither",
        "readout_side": "setting",
        "action_label": "buy",
        "exact_expected": False,
    }
