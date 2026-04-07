from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
ASSET_DIR = ROOT / "data" / "report_assets" / "synthetic_market_public_story"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _phase_rows() -> list[dict[str, Any]]:
    return [
        {
            "phase": "15",
            "title": "Build A Better Test Bed",
            "question": "Can we get the model to show a clean market read at all?",
            "what_we_tested": (
                "We replaced toy prompts with more realistic six-asset market prompts and looked for "
                "recurring internal market patterns."
            ),
            "what_held_up": (
                "Two promising patterns appeared: one linked to a standout asset and one linked to how "
                "uneven the market was. But some of the signal was still mixed with prompt-format effects."
            ),
            "confidence": "promising, but not yet clean",
        },
        {
            "phase": "15 rerun",
            "title": "Strip Out Formatting Effects",
            "question": "Do the same patterns survive after we remove prompt-shape effects?",
            "what_we_tested": (
                "We repeated the discovery step after removing the part of the signal that could be predicted "
                "from prompt length and layout."
            ),
            "what_held_up": (
                "The average market read got much cleaner and the two main patterns survived. The end-of-market "
                "read improved too, but it was still less trustworthy."
            ),
            "confidence": "strong for the average market read",
        },
        {
            "phase": "16",
            "title": "Move Context Before Or After The Market",
            "question": "Does earlier context change how the model reads the same market?",
            "what_we_tested": (
                "We took the same markets and moved risk or opportunity-setting context either before or after the market block."
            ),
            "what_held_up": (
                "Context placed before the market changed the model's market read. The same context placed after the market did not."
            ),
            "confidence": "strong",
        },
        {
            "phase": "17",
            "title": "Name The Two Strongest Patterns",
            "question": "What do the recurring internal patterns seem to mean in plain market terms?",
            "what_we_tested": (
                "We matched the model's internal patterns against visible market features only, without using hidden labels."
            ),
            "what_held_up": (
                "One pattern looked like a standout asset with strong activity. The other looked like the market's overall unevenness."
            ),
            "confidence": "good description, but not a full explanation",
        },
        {
            "phase": "18",
            "title": "First Intervention Test",
            "question": "If we weaken one of these signals, does the model behave differently?",
            "what_we_tested": (
                "We removed either one candidate signal at a time or a broader group of related signals and compared the result "
                "with matched random edits."
            ),
            "what_held_up": (
                "Single directions were weak. Broader groups mattered more. That pointed away from a one-switch story."
            ),
            "confidence": "suggestive, not decisive",
        },
        {
            "phase": "19",
            "title": "Run A Cleaner Joint Test",
            "question": "Do better controls and cleaner targeting change the story?",
            "what_we_tested": (
                "We used matched prompts, deterministic decoding, better controls, and cleaner targeting of the market span."
            ),
            "what_held_up": (
                "Targeted edits mattered slightly more than matched random edits, but the gap was still modest."
            ),
            "confidence": "real signal, but still exploratory",
        },
        {
            "phase": "20",
            "title": "Stress Test The Result",
            "question": "Do targeted edits still look special under a larger and more careful battery of tests?",
            "what_we_tested": (
                "We used matched pairs, both weaker-side and stronger-side prompts, several edit strengths, and matched random controls."
            ),
            "what_held_up": (
                "In all 12 main comparisons, targeted edits were less disruptive than matched random edits. The signal was real and selective."
            ),
            "confidence": "strong for selectivity, not yet enough for a causal claim",
        },
        {
            "phase": "21",
            "title": "Try To Move Behavior Back Toward The Source Example",
            "question": "If we put the matching signal back in, does behavior move back toward the source example?",
            "what_we_tested": (
                "We inserted source-side signal into base prompts and measured whether the model's choices moved toward the paired source example."
            ),
            "what_held_up": (
                "The stronger candidate helped a little. The weaker candidate did not. Neither looked like a decisive cause of the final choice."
            ),
            "confidence": "enough to rule out a strong causal claim",
        },
    ]


def _validated_claims(phase20: dict[str, Any], phase21: dict[str, Any], combined: dict[str, Any]) -> list[dict[str, str]]:
    leader = phase21["axes"]["leader"]["metrics"]
    dispersion = phase21["axes"]["dispersion"]["metrics"]
    return [
        {
            "claim": "The model builds a real internal picture of the market.",
            "support": "Strong",
            "evidence": (
                "This held up from the cleaned discovery rerun through the later intervention phases. "
                "The model repeatedly showed stable market-linked internal patterns."
            ),
        },
        {
            "claim": "What the model reads before the market changes how it interprets that market.",
            "support": "Strong",
            "evidence": (
                f"When context moved before the market, the market-reading gap reached {combined['context_order']['risk_gap']:.3f} "
                f"for risk and {combined['context_order']['aff_gap']:.3f} for opportunity framing. When the same context came after "
                "the market, the market read stayed effectively unchanged."
            ),
        },
        {
            "claim": "The two main internal patterns are meaningful, not random noise.",
            "support": "Strong",
            "evidence": (
                f"In Phase 20, targeted edits beat matched random edits in all "
                f"{phase20['overall']['tool_token_selectivity_gap_wins']} of "
                f"{phase20['overall']['total_gap_comparisons']} main choice comparisons."
            ),
        },
        {
            "claim": "The stronger candidate signal is the main cause of the final trading choice.",
            "support": "Not supported",
            "evidence": (
                f"In the clean source-matching phase, agreement with the source choice improved by only "
                f"{leader['source_tool_token_match_rate_delta']:.3f}, and the signal fixed mistakes on "
                f"{leader['source_tool_token_restoration_rate']:.3f} of rows that needed help."
            ),
        },
        {
            "claim": "The weaker candidate signal is a convincing driver of final action choice.",
            "support": "Not supported",
            "evidence": (
                f"In the same phase, agreement with the source choice fell by "
                f"{abs(dispersion['source_tool_token_match_rate_delta']):.3f}, while the rate at which it fixed mistakes "
                f"({dispersion['source_tool_token_restoration_rate']:.3f}) was slightly lower than the rate at which it created new ones "
                f"({dispersion['source_tool_token_backfire_rate']:.3f})."
            ),
        },
    ]


def _build_summary() -> dict[str, Any]:
    phase20 = _load_json(ROOT / "data" / "report_assets" / "synthetic_market_phase20_paired_robustness" / "summary.json")
    phase21 = _load_json(ROOT / "data" / "report_assets" / "synthetic_market_phase21_restoration" / "summary.json")
    combined = _load_json(ROOT / "data" / "report_assets" / "synthetic_market_phase16_17_combined" / "summary.json")
    phase17 = _load_json(ROOT / "data" / "report_assets" / "synthetic_market_phase17_axis_decomposition" / "summary.json")
    phase19 = _load_json(ROOT / "data" / "report_assets" / "synthetic_market_phase19_methodology_and_results" / "summary.json")
    phase18 = _load_json(ROOT / "data" / "report_assets" / "synthetic_market_phase18_causal_patching" / "summary.json")

    return {
        "date": "6 April 2026",
        "title": "Synthetic Market Research Story",
        "deck": (
            "A seven-phase attempt to trace how a trading model reads a market, what shaped that read, "
            "and whether any one internal signal actually drove the final choice."
        ),
        "top_line": (
            "We found a real internal market picture inside the model, and we showed that context can change that picture. "
            "But the two strongest candidate signals did not hold up as single clean causes of the final trading choice."
        ),
        "phases": _phase_rows(),
        "headline_numbers": {
            "risk_gap": combined["context_order"]["risk_gap"],
            "affordance_gap": combined["context_order"]["aff_gap"],
            "leader_feature": phase17["leader"]["best_single_feature"],
            "leader_feature_cv_r2": phase17["leader"]["best_single_feature"]["cv_r2"],
            "dispersion_feature": phase17["dispersion"]["best_single_feature"],
            "dispersion_feature_cv_r2": phase17["dispersion"]["best_single_feature"]["cv_r2"],
            "phase18_leader_4d_targeted": phase18["experiments"]["leader_4d"]["project_out"]["compare"]["tool_token_change_rate"],
            "phase18_leader_4d_control": phase18["experiments"]["leader_4d"]["random_control"]["compare"]["tool_token_change_rate"],
            "phase18_dispersion_4d_targeted": phase18["experiments"]["dispersion_4d"]["project_out"]["compare"]["tool_token_change_rate"],
            "phase18_dispersion_4d_control": phase18["experiments"]["dispersion_4d"]["random_control"]["compare"]["tool_token_change_rate"],
            "phase19_targeted_choice_change": phase19["comparisons"]["project_out"]["tool_token_change_rate"],
            "phase19_control_choice_change": phase19["comparisons"]["random_control"]["tool_token_change_rate"],
            "phase20_selectivity_wins": phase20["overall"]["tool_token_selectivity_gap_wins"],
            "phase20_total_comparisons": phase20["overall"]["total_gap_comparisons"],
            "phase21_leader_match_delta": phase21["axes"]["leader"]["metrics"]["source_tool_token_match_rate_delta"],
            "phase21_leader_fix_rate": phase21["axes"]["leader"]["metrics"]["source_tool_token_restoration_rate"],
            "phase21_leader_harm_rate": phase21["axes"]["leader"]["metrics"]["source_tool_token_backfire_rate"],
            "phase21_dispersion_match_delta": phase21["axes"]["dispersion"]["metrics"]["source_tool_token_match_rate_delta"],
            "phase21_dispersion_fix_rate": phase21["axes"]["dispersion"]["metrics"]["source_tool_token_restoration_rate"],
            "phase21_dispersion_harm_rate": phase21["axes"]["dispersion"]["metrics"]["source_tool_token_backfire_rate"],
        },
        "validated_claims": _validated_claims(phase20, phase21, combined),
        "final_conclusion": (
            "The model builds a meaningful internal market summary, and that summary is influenced by context. "
            "We can also find recurring internal signals that line up with visible market features. But the specific "
            "signals we isolated are not enough, on their own, to explain the model's final trading choice."
        ),
        "client_summary": {
            "what_we_now_know": [
                "The model's market read is real, measurable, and shaped by earlier context.",
                "The model appears to track both standout assets and overall market unevenness.",
                "Targeted edits behave differently from arbitrary matched edits, which shows the internal market signals are not random artifacts.",
            ],
            "what_we_do_not_know": [
                "We have not isolated one decisive internal lever behind the final choice.",
                "The weaker candidate signal did not survive the strongest test.",
                "Even the stronger candidate signal looks more like part of the story than the whole story.",
            ],
            "recommended_public_framing": (
                "The safest public claim is that we found a real internal picture of the market and learned how context can bend it, "
                "but we did not find a single clean internal cause of the final trading choice."
            ),
        },
        "figures": {
            "phase15_discovery": "../../../../data/report_assets/synthetic_market_phase15_discovery/phase15_discovery_summary.png",
            "phase15_rerun": "../../../../data/report_assets/synthetic_market_phase15_discovery_rerun/phase15_rerun_compare.png",
            "phase16_perception": "../../../../data/report_assets/synthetic_market_phase16_17_combined/combined_perception_curves.png",
            "phase17_breakdown": "../../../../data/report_assets/synthetic_market_phase16_17_combined/combined_axis_decomposition.png",
            "phase18_change_rates": "../../../../data/report_assets/synthetic_market_phase18_causal_patching/phase18_change_rates.png",
            "phase20_selectivity": "../../../../data/report_assets/synthetic_market_phase20_paired_robustness/phase20_lambda1_selectivity_gaps.png",
            "phase21_comparison": "../../../../data/report_assets/synthetic_market_public_story/phase21_choice_comparison.png",
        },
    }


def _build_phase21_chart(summary: dict[str, Any]) -> None:
    metrics = summary["headline_numbers"]
    labels = ["Stronger signal", "Weaker signal"]
    fix = np.array([metrics["phase21_leader_fix_rate"], metrics["phase21_dispersion_fix_rate"]], dtype=float)
    harm = np.array([metrics["phase21_leader_harm_rate"], metrics["phase21_dispersion_harm_rate"]], dtype=float)
    delta = np.array([metrics["phase21_leader_match_delta"], metrics["phase21_dispersion_match_delta"]], dtype=float)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), dpi=220)

    x = np.arange(len(labels))
    width = 0.34
    axes[0].bar(x - width / 2, fix, width=width, color="#2f6b4f", label="Helped on rows that needed fixing")
    axes[0].bar(x + width / 2, harm, width=width, color="#b6523a", label="Made already-correct rows worse")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, max(float(fix.max()), float(harm.max())) * 1.35)
    axes[0].set_ylabel("Share of rows")
    axes[0].set_title("How Often The Edit Helped Versus Hurt")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")

    axes[1].axhline(0.0, color="#555555", linewidth=1.0)
    colors = ["#2f6b4f" if value >= 0 else "#b6523a" for value in delta]
    axes[1].bar(x, delta, width=0.55, color=colors)
    axes[1].set_xticks(x, labels)
    bound = max(abs(float(delta.min())), abs(float(delta.max())), 0.05)
    axes[1].set_ylim(-bound * 1.4, bound * 1.4)
    axes[1].set_ylabel("Change in agreement with source choice")
    axes[1].set_title("Did The Edit Move The Model Toward The Source Choice?")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Phase 21: Final Source-Matching Test", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "phase21_choice_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    summary = _build_summary()
    (ASSET_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    _build_phase21_chart(summary)


if __name__ == "__main__":
    main()
