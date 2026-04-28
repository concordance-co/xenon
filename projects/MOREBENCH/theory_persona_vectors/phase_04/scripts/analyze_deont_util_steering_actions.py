"""Semantic action scoring for Phase 04 deont/util steering runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from projects.MOREBENCH.theory_persona_vectors.phase_04.scripts.analyze_conflict_stability_actions import (
    CLASSIFIERS,
    _norm,
)


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_04")
REPORT_ROOT = PHASE_ROOT / "reports" / "deont_util_steering"
OUTPUT_ROOT = PHASE_ROOT / "reports" / "deont_util_steering_action_analysis"

VARIANTS = {
    "deont_1_0": "steer_deont_1_0_results.json",
    "util_1_0": "steer_util_1_0_results.json",
    "generic_1_0": "steer_generic_1_0_results.json",
    "random_1_0": "steer_random_1_0_results.json",
}


def _latest_report_dir(layer: int) -> Path:
    layer_root = REPORT_ROOT / f"L{layer}"
    candidates = sorted(layer_root.glob("report_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no report directory found under {layer_root}")
    return candidates[0]


def _rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise TypeError(f"{path} did not contain a JSON object with a rows list")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _labels(row: dict[str, Any]) -> dict[str, Any]:
    example = row.get("example") if isinstance(row.get("example"), dict) else {}
    labels = example.get("labels") if isinstance(example.get("labels"), dict) else {}
    return dict(labels)


def _text(row: dict[str, Any]) -> str:
    return str(row.get("generated_text") or "").strip()


def _case_key(labels: dict[str, Any]) -> tuple[str, str]:
    return str(labels.get("group_id") or ""), str(labels.get("variant_id") or "")


def _classify(group_id: str, text: str) -> str:
    classifier = CLASSIFIERS.get(group_id)
    if classifier is None:
        raise KeyError(f"no classifier for group_id={group_id!r}")
    return classifier(_norm(text))


def _load_layer(layer: int) -> tuple[Path, dict[str, list[dict[str, Any]]]]:
    report_dir = _latest_report_dir(layer)
    results_dir = report_dir / "results"
    loaded = {"baseline": _rows(results_dir / "baseline_generation_results.json")}
    for variant, filename in VARIANTS.items():
        loaded[variant] = _rows(results_dir / filename)
    return report_dir, loaded


def analyze_layer(layer: int) -> dict[str, Any]:
    report_dir, rows_by_variant = _load_layer(layer)
    baseline_by_case: dict[tuple[str, str], dict[str, Any]] = {}
    labeled_rows: list[dict[str, Any]] = []

    for row in rows_by_variant["baseline"]:
        labels = _labels(row)
        group_id, variant_id = _case_key(labels)
        action = _classify(group_id, _text(row))
        baseline_by_case[(group_id, variant_id)] = {
            "action": action,
            "text": _text(row),
            "labels": labels,
        }
        labeled_rows.append(
            {
                "layer": layer,
                "steering_variant": "baseline",
                "group_id": group_id,
                "variant_id": variant_id,
                "action_label": action,
                "expected_deont_action": labels.get("expected_deont_action"),
                "expected_util_action": labels.get("expected_util_action"),
                "text_preview": _text(row)[:500],
            }
        )

    trial_rows: list[dict[str, Any]] = []
    for steering_variant, rows in rows_by_variant.items():
        if steering_variant == "baseline":
            continue
        for row in rows:
            labels = _labels(row)
            group_id, variant_id = _case_key(labels)
            case = baseline_by_case[(group_id, variant_id)]
            action = _classify(group_id, _text(row))
            baseline_action = str(case["action"])
            expected_deont = str(labels.get("expected_deont_action") or "")
            expected_util = str(labels.get("expected_util_action") or "")
            trial = {
                "layer": layer,
                "steering_variant": steering_variant,
                "group_id": group_id,
                "variant_id": variant_id,
                "baseline_action": baseline_action,
                "steered_action": action,
                "changed_action": action != baseline_action,
                "expected_deont_action": expected_deont,
                "expected_util_action": expected_util,
                "baseline_is_deont": baseline_action == expected_deont,
                "baseline_is_util": baseline_action == expected_util,
                "steered_is_deont": action == expected_deont,
                "steered_is_util": action == expected_util,
                "changed_to_deont": baseline_action != expected_deont and action == expected_deont,
                "changed_to_util": baseline_action != expected_util and action == expected_util,
                "changed_away_from_deont": baseline_action == expected_deont and action != expected_deont,
                "changed_away_from_util": baseline_action == expected_util and action != expected_util,
                "ambiguous": action == "ambiguous",
                "baseline_text_preview": str(case["text"])[:500],
                "steered_text_preview": _text(row)[:500],
            }
            trial_rows.append(trial)
            labeled_rows.append(
                {
                    "layer": layer,
                    "steering_variant": steering_variant,
                    "group_id": group_id,
                    "variant_id": variant_id,
                    "action_label": action,
                    "expected_deont_action": expected_deont,
                    "expected_util_action": expected_util,
                    "text_preview": _text(row)[:500],
                }
            )

    summaries: dict[str, dict[str, Any]] = {}
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trial_rows:
        by_variant[str(trial["steering_variant"])].append(trial)

    for steering_variant, items in sorted(by_variant.items()):
        n = len(items)
        deont_opps = [item for item in items if not item["baseline_is_deont"]]
        util_opps = [item for item in items if not item["baseline_is_util"]]
        summaries[steering_variant] = {
            "n": n,
            "changed_action": sum(bool(item["changed_action"]) for item in items),
            "ambiguous": sum(bool(item["ambiguous"]) for item in items),
            "baseline_deont_hits": sum(bool(item["baseline_is_deont"]) for item in items),
            "baseline_util_hits": sum(bool(item["baseline_is_util"]) for item in items),
            "steered_deont_hits": sum(bool(item["steered_is_deont"]) for item in items),
            "steered_util_hits": sum(bool(item["steered_is_util"]) for item in items),
            "deont_opportunity_n": len(deont_opps),
            "util_opportunity_n": len(util_opps),
            "changed_to_deont": sum(bool(item["changed_to_deont"]) for item in items),
            "changed_to_util": sum(bool(item["changed_to_util"]) for item in items),
            "changed_away_from_deont": sum(bool(item["changed_away_from_deont"]) for item in items),
            "changed_away_from_util": sum(bool(item["changed_away_from_util"]) for item in items),
            "action_counts": dict(Counter(str(item["steered_action"]) for item in items)),
        }

    group_summaries: dict[str, dict[str, Any]] = {}
    baseline_items = [
        {
            "group_id": group_id,
            "variant_id": variant_id,
            "action": str(case["action"]),
        }
        for (group_id, variant_id), case in baseline_by_case.items()
    ]
    for group_id in sorted({str(item["group_id"]) for item in trial_rows}):
        group_items = [item for item in trial_rows if item["group_id"] == group_id]
        group_baselines = [item for item in baseline_items if item["group_id"] == group_id]
        group_summaries[group_id] = {
            "n": len(group_items),
            "baseline_actions": dict(Counter(str(item["action"]) for item in group_baselines)),
            "by_variant": {
                variant: {
                    "changed_action": sum(bool(item["changed_action"]) for item in group_items if item["steering_variant"] == variant),
                    "steered_actions": dict(Counter(str(item["steered_action"]) for item in group_items if item["steering_variant"] == variant)),
                }
                for variant in sorted(VARIANTS)
            },
        }

    return {
        "layer": layer,
        "report_dir": str(report_dir),
        "baseline_n": len(rows_by_variant["baseline"]),
        "trial_n": len(trial_rows),
        "summaries": summaries,
        "group_summaries": group_summaries,
        "trials": trial_rows,
        "labeled_rows": labeled_rows,
    }


def _pct(num: int, den: int) -> str:
    if den <= 0:
        return "n/a"
    return f"{num}/{den} ({num / den:.0%})"


def write_outputs(results: list[dict[str, Any]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    with (OUTPUT_ROOT / "trial_labels.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            for item in result["labeled_rows"]:
                handle.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")

    lines = [
        "# Phase 04 Deont/Util Steering Action Analysis",
        "",
        "Semantic labels are narrow, group-specific classifiers reused from the stability pass.",
        "Primary action-shift denominators are opportunity denominators: a deont shift is only possible where the paired neutral baseline was not already the deont endpoint, and likewise for util.",
        "",
    ]
    for result in results:
        layer = result["layer"]
        lines.extend(
            [
                f"## Layer {layer}",
                "",
                f"- report_dir: `{result['report_dir']}`",
                f"- baseline rows: `{result['baseline_n']}`",
                f"- steered rows: `{result['trial_n']}`",
                "",
                "| steering | changed action | changed to deont | changed to util | away from deont | away from util | ambiguous |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for steering_variant, summary in sorted(result["summaries"].items()):
            lines.append(
                "| "
                f"`{steering_variant}` | "
                f"{_pct(summary['changed_action'], summary['n'])} | "
                f"{_pct(summary['changed_to_deont'], summary['deont_opportunity_n'])} | "
                f"{_pct(summary['changed_to_util'], summary['util_opportunity_n'])} | "
                f"{summary['changed_away_from_deont']} | "
                f"{summary['changed_away_from_util']} | "
                f"{summary['ambiguous']} |"
            )
        lines.extend(["", "### Group Diagnostics", ""])
        lines.append("| group | baseline actions | deont steered | util steered | generic steered | random steered |")
        lines.append("|---|---|---|---|---|---|")
        for group_id, group in sorted(result["group_summaries"].items()):
            by_variant = group["by_variant"]
            lines.append(
                "| "
                f"`{group_id}` | "
                f"`{json.dumps(group['baseline_actions'], sort_keys=True)}` | "
                f"`{json.dumps(by_variant['deont_1_0']['steered_actions'], sort_keys=True)}` | "
                f"`{json.dumps(by_variant['util_1_0']['steered_actions'], sort_keys=True)}` | "
                f"`{json.dumps(by_variant['generic_1_0']['steered_actions'], sort_keys=True)}` | "
                f"`{json.dumps(by_variant['random_1_0']['steered_actions'], sort_keys=True)}` |"
            )
        lines.append("")

    (OUTPUT_ROOT / "report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=(16, 32))
    args = parser.parse_args()

    results = [analyze_layer(layer) for layer in args.layers]
    write_outputs(results)
    print(json.dumps({"layers": args.layers, "output": str(OUTPUT_ROOT / "report.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
