"""Analyze controlled deontology steering runs against stored deont references."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
STEERING_REPORT_ROOT = PHASE_ROOT / "reports" / "deont_prompt_isolation_steering"
SOURCE_REPORT_ROOT = PHASE_ROOT / "reports" / "deont_prompt_isolation_report"
OUTPUT_ROOT = PHASE_ROOT / "reports" / "deont_prompt_isolation_steering_analysis"

STEP_VARIANTS = (
    ("steer_deont01_on_neutral", "N_neutral_iso_01", "P_deont_iso_01"),
    ("steer_deont02_on_neutral", "N_neutral_iso_01", "P_deont_iso_02"),
    ("steer_generic_on_neutral", "N_neutral_iso_01", "N_generic_moral_iso_01"),
    ("steer_random_on_neutral", "N_neutral_iso_01", None),
    ("steer_deont01_on_generic", "N_generic_moral_iso_01", "P_deont_iso_01"),
    ("steer_deont02_on_generic", "N_generic_moral_iso_01", "P_deont_iso_02"),
    ("steer_random_on_generic", "N_generic_moral_iso_01", None),
)


def _latest_report_dir(root: Path) -> Path:
    candidates = sorted(
        (path for path in root.glob("report_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no report found under {root}")
    return candidates[0]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"{path} must contain a rows list")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    example = row.get("example") if isinstance(row.get("example"), Mapping) else {}
    labels = example.get("labels") if isinstance(example.get("labels"), Mapping) else {}
    return str(labels.get("dilemma_id") or ""), str(labels.get("condition_id") or "")


def _text_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for row in rows:
        out[_row_key(row)] = str(row.get("generated_text") or "")
    return out


def _recommendation_line(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("Recommendation:"):
            return line.removeprefix("Recommendation:").strip()
    return ""


def _ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(a=a, b=b).ratio())


def analyze(*, layer: int) -> dict[str, Any]:
    steering_dir = _latest_report_dir(STEERING_REPORT_ROOT / f"L{int(layer)}")
    source_dir = _latest_report_dir(SOURCE_REPORT_ROOT)

    baseline_rows = _load_rows(steering_dir / "results" / "baseline_generation_results.json")
    baseline_index = _text_index(baseline_rows)

    source_rows = _load_rows(source_dir / "results" / "generate_natural_responses_results.json")
    source_index = _text_index(source_rows)

    variant_metrics: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for step_name, target_condition, reference_condition in STEP_VARIANTS:
        variant_path = steering_dir / "results" / f"{step_name}_results.json"
        if not variant_path.exists():
            continue
        variant_rows = _load_rows(variant_path)
        changed = 0
        exact_ref = 0
        recommendation_gain_sum = 0.0
        fulltext_gain_sum = 0.0
        recommendation_gain_positive = 0
        records: list[dict[str, Any]] = []
        for row in variant_rows:
            dilemma_id, condition_id = _row_key(row)
            if condition_id != target_condition:
                continue
            patched_text = str(row.get("generated_text") or "")
            baseline_text = baseline_index.get((dilemma_id, target_condition), "")
            patched_rec = _recommendation_line(patched_text)
            baseline_rec = _recommendation_line(baseline_text)
            ref_text = source_index.get((dilemma_id, reference_condition), "") if reference_condition else ""
            ref_rec = _recommendation_line(ref_text)

            rec_gain = _ratio(patched_rec, ref_rec) - _ratio(baseline_rec, ref_rec) if reference_condition else 0.0
            text_gain = _ratio(patched_text, ref_text) - _ratio(baseline_text, ref_text) if reference_condition else 0.0
            changed_flag = patched_rec != baseline_rec
            exact_flag = bool(reference_condition) and patched_rec == ref_rec and bool(ref_rec)
            changed += int(changed_flag)
            exact_ref += int(exact_flag)
            recommendation_gain_sum += rec_gain
            fulltext_gain_sum += text_gain
            recommendation_gain_positive += int(rec_gain > 0.0)
            if changed_flag or exact_flag or rec_gain > 0.1:
                records.append(
                    {
                        "dilemma_id": dilemma_id,
                        "target_condition": target_condition,
                        "reference_condition": reference_condition,
                        "baseline_recommendation": baseline_rec,
                        "patched_recommendation": patched_rec,
                        "reference_recommendation": ref_rec,
                        "recommendation_gain": rec_gain,
                        "fulltext_gain": text_gain,
                    }
                )

        n = len(variant_rows)
        variant_metrics.append(
            {
                "step_name": step_name,
                "target_condition": target_condition,
                "reference_condition": reference_condition,
                "n_rows": n,
                "changed_rate": (changed / n) if n else 0.0,
                "exact_reference_recommendation_rate": (exact_ref / n) if n else 0.0,
                "mean_recommendation_similarity_gain": (recommendation_gain_sum / n) if n else 0.0,
                "mean_fulltext_similarity_gain": (fulltext_gain_sum / n) if n else 0.0,
                "recommendation_gain_positive_rate": (recommendation_gain_positive / n) if n else 0.0,
            }
        )
        sample_rows.extend(records[:5])

    return {
        "layer": int(layer),
        "steering_report_dir": str(steering_dir),
        "source_report_dir": str(source_dir),
        "variant_metrics": variant_metrics,
        "sample_rows": sample_rows[:30],
    }


def _write_report(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# Deont Prompt Isolation Steering Analysis L{summary['layer']}",
        "",
        f"- Steering report: `{summary['steering_report_dir']}`",
        f"- Source report: `{summary['source_report_dir']}`",
        "",
        "## Framing",
        "",
        "This phase is an intervention smoke, not a claim that the underlying deont signal is text-free.",
        "The main goal here is to test whether the controlled deont setup provides a writable causal substrate for patching.",
        "On that goal, substantial lexical leakage is acceptable: if the target behavior and generated activations are tightly coupled, that can still be useful for causal steering even when the representation is not purified away from surface text.",
        "Accordingly, the primary readouts in this report are writeability-oriented metrics like recommendation similarity gain and comparison against random-control change rates, not whether the setup fully eliminated textual confounds.",
        "",
        "## Variant Metrics",
        "",
        "| variant | target | reference | changed | exact-ref-rec | mean rec gain | mean text gain | rec gain >0 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["variant_metrics"]:
        lines.append(
            "| {step_name} | {target_condition} | {reference_condition} | {changed_rate:.3f} | "
            "{exact_reference_recommendation_rate:.3f} | {mean_recommendation_similarity_gain:.3f} | "
            "{mean_fulltext_similarity_gain:.3f} | {recommendation_gain_positive_rate:.3f} |".format(**row)
        )
    lines.extend(["", "## Sample Rows", ""])
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary["sample_rows"]:
        by_variant[str(row["target_condition"]) + ":" + str(row["reference_condition"])].append(row)
    for group, rows in by_variant.items():
        lines.append(f"### {group}")
        lines.append("")
        for row in rows:
            lines.append(f"- `{row['dilemma_id']}` baseline: {row['baseline_recommendation']}")
            lines.append(f"  patched: {row['patched_recommendation']}")
            lines.append(f"  reference: {row['reference_recommendation']}")
            lines.append(
                f"  rec gain `{row['recommendation_gain']:.3f}`, full-text gain `{row['fulltext_gain']:.3f}`"
            )
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()
    summary = analyze(layer=int(args.layer))
    out_dir = OUTPUT_ROOT / f"L{int(args.layer)}"
    _write_report(summary, out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
