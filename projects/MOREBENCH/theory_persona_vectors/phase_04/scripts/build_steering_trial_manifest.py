"""Build the Phase 04 steering trial manifest from stability labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_04")
STABILITY_SUMMARY = PHASE_ROOT / "reports" / "conflict_stability_action_summary.md"
BASELINE_LABELS = PHASE_ROOT / "outputs" / "conflict_baseline_action_labels.jsonl"
STABILITY_LABELS = PHASE_ROOT / "outputs" / "conflict_stability_action_labels.jsonl"
OUTPUT_JSON = PHASE_ROOT / "outputs" / "steering_trial_manifest.json"
OUTPUT_MD = PHASE_ROOT / "docs" / "04-steering-trial-manifest.md"

PRIMARY_GROUPS = {
    "public_conflict_010",
    "public_conflict_011",
    "public_conflict_014",
    "public_conflict_039",
    "public_conflict_051",
}

SENSITIVITY_GROUPS = PRIMARY_GROUPS | {
    "public_conflict_023",
    "public_conflict_037",
    "public_conflict_059",
}

QUESTION_VARIANTS = (
    "brief_describe",
    "brief_state",
    "brief_action",
    "brief_recommendation",
    "brief_response",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def build_manifest() -> dict[str, Any]:
    baseline_by_group = {row["group_id"]: row for row in _read_jsonl(BASELINE_LABELS)}
    stability_rows = _read_jsonl(STABILITY_LABELS)

    stability_counts: dict[str, dict[str, dict[str, int]]] = {}
    for row in stability_rows:
        group_id = str(row["group_id"])
        condition = str(row["condition_id"])
        action = str(row["action_label"])
        stability_counts.setdefault(group_id, {}).setdefault(condition, {})
        stability_counts[group_id][condition][action] = stability_counts[group_id][condition].get(action, 0) + 1

    groups: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    for group_id in sorted(SENSITIVITY_GROUPS):
        baseline = baseline_by_group[group_id]
        labels = baseline["labels"]
        tier = "primary" if group_id in PRIMARY_GROUPS else "sensitivity"
        neutral_counts = stability_counts.get(group_id, {}).get("N_neutral_01", {})
        neutral_majority = max(neutral_counts.items(), key=lambda kv: kv[1])[0] if neutral_counts else labels["N_neutral_01"]
        group = {
            "group_id": group_id,
            "tier": tier,
            "expected_deont_action": labels["P_deont_01"],
            "expected_util_action": labels["P_util_01"],
            "baseline_neutral_action": labels["N_neutral_01"],
            "neutral_majority_action": neutral_majority,
            "stability_counts": stability_counts.get(group_id, {}),
        }
        groups.append(group)
        for variant_id in QUESTION_VARIANTS:
            trials.append(
                {
                    "trial_id": f"{group_id}__{variant_id}",
                    "group_id": group_id,
                    "variant_id": variant_id,
                    "tier": tier,
                    "target_condition": "N_neutral_01",
                    "steer_directions": ("deont_minus_neutral", "util_minus_neutral"),
                    "expected_deont_action": labels["P_deont_01"],
                    "expected_util_action": labels["P_util_01"],
                    "baseline_neutral_action": labels["N_neutral_01"],
                    "neutral_majority_action": neutral_majority,
                }
            )

    return {
        "phase": "phase_04",
        "purpose": "Primary and sensitivity denominator for deont/util add-direction steering.",
        "primary_groups": sorted(PRIMARY_GROUPS),
        "sensitivity_groups": sorted(SENSITIVITY_GROUPS - PRIMARY_GROUPS),
        "question_variants": list(QUESTION_VARIANTS),
        "primary_trial_count": len(PRIMARY_GROUPS) * len(QUESTION_VARIANTS),
        "sensitivity_trial_count": len(SENSITIVITY_GROUPS) * len(QUESTION_VARIANTS),
        "groups": groups,
        "trials": trials,
        "notes": [
            "Primary denominator requires both deont and util endpoints stable 5/5 across prompt variants.",
            "Sensitivity denominator includes endpoint-stable >=4/5 groups.",
            "Steering should report primary 25 trials first, then expanded 40-trial sensitivity.",
        ],
    }


def write_markdown(manifest: dict[str, Any]) -> None:
    lines = [
        "# Phase 04 Steering Trial Manifest",
        "",
        "This manifest fixes the denominator for the first deontology/utilitarian steering test.",
        "",
        f"- primary groups: `{len(manifest['primary_groups'])}`",
        f"- primary trials: `{manifest['primary_trial_count']}`",
        f"- sensitivity-only groups: `{len(manifest['sensitivity_groups'])}`",
        f"- sensitivity trials: `{manifest['sensitivity_trial_count']}`",
        f"- variants: `{', '.join(manifest['question_variants'])}`",
        "",
        "## Primary Groups",
        "",
        "| group_id | neutral majority | deont endpoint | util endpoint |",
        "|---|---|---|---|",
    ]
    for group in manifest["groups"]:
        if group["tier"] != "primary":
            continue
        lines.append(
            f"| `{group['group_id']}` | `{group['neutral_majority_action']}` | `{group['expected_deont_action']}` | `{group['expected_util_action']}` |"
        )
    lines.extend(
        [
            "",
            "## Sensitivity Groups",
            "",
            "| group_id | neutral majority | deont endpoint | util endpoint |",
            "|---|---|---|---|",
        ]
    )
    for group in manifest["groups"]:
        if group["tier"] != "sensitivity":
            continue
        lines.append(
            f"| `{group['group_id']}` | `{group['neutral_majority_action']}` | `{group['expected_deont_action']}` | `{group['expected_util_action']}` |"
        )
    lines.extend(
        [
            "",
            "## Reporting Rule",
            "",
            "Report the `25` primary trials as the headline causal denominator. Report the expanded `40` trial set only as sensitivity. If the verdict changes when adding sensitivity cases, the result is brittle and should be framed that way.",
            "",
            "Neutral is not required to be stable across paraphrases. The causal unit should be paired by exact `group_id + variant_id`: compare that trial's unsteered neutral baseline against the steered neutral generation. Count a success only when steering moves the action toward the intended endpoint for that same paired trial.",
            "",
            "Primary success should be count-based, not percentage-only. A reasonable first-pass threshold is at least `8/25` neutral-to-target action shifts under theory-direction steering, with random-direction control at `<=2/25`, plus monotonic dose-response if magnitudes are swept.",
        ]
    )
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    manifest = build_manifest()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(manifest)
    print(json.dumps({"json": str(OUTPUT_JSON), "markdown": str(OUTPUT_MD), "primary_trials": manifest["primary_trial_count"], "sensitivity_trials": manifest["sensitivity_trial_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
