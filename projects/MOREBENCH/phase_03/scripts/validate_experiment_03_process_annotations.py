#!/usr/bin/env python3
"""Validate, merge, and score process-supervision annotations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.metrics import cohen_kappa_score


ROOT = Path("projects/MOREBENCH/phase_03")
BASE_DIR = ROOT / "reports/experiment_03_process_supervision"
PACKET_DIR = BASE_DIR / "annotation_packet"
ANNOTATION_DIR = BASE_DIR / "annotations"
MERGED_PATH = BASE_DIR / "process_supervision_annotations.jsonl"
SUMMARY_PATH = BASE_DIR / "labelability_summary.json"
REPORT_PATH = BASE_DIR / "labelability_report.md"
RECALIBRATED_REVIEWER_PATH = ANNOTATION_DIR / "review_sample_30_annotations_recalibrated.jsonl"
DEFAULT_REVIEWER_PATH = ANNOTATION_DIR / "review_sample_30_annotations.jsonl"

F_KAPPA_THRESHOLD = 0.60
B_IOU_THRESHOLD = 0.50
B_SHARE_THRESHOLD = 0.80
C_KAPPA_THRESHOLD = 0.60


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _span(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, Mapping):
        return None
    start = value.get("char_start")
    end = value.get("char_end")
    if start is None or end is None:
        return None
    try:
        start_i = int(start)
        end_i = int(end)
    except (TypeError, ValueError):
        return None
    if start_i < 0 or end_i <= start_i:
        return None
    return start_i, end_i


def _span_iou(left: tuple[int, int] | None, right: tuple[int, int] | None) -> float | None:
    if left is None or right is None:
        return None
    overlap = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return float(overlap / union) if union > 0 else None


def _coverage_family_set(row: Mapping[str, Any]) -> set[str]:
    covered: set[str] = set()
    for item in row.get("criterion_coverage", []) if isinstance(row.get("criterion_coverage"), list) else []:
        if isinstance(item, Mapping) and bool(item.get("covered")):
            family_id = str(item.get("family_id") or "").strip()
            if family_id:
                covered.add(family_id)
    return covered


def _c_label(row: Mapping[str, Any]) -> str:
    consideration = row.get("consideration")
    if not isinstance(consideration, Mapping):
        return ""
    value = str(consideration.get("early_collapse_vs_sustained") or "").strip()
    return value


def _validate_row(
    row: dict[str, Any],
    *,
    source_rows: Mapping[str, Mapping[str, Any]],
    criterion_ids: set[str],
    family_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    row_id = str(row.get("row_id") or "").strip()
    if row_id not in source_rows:
        errors.append(f"unknown row_id {row_id!r}")
        return errors
    response_len = len(str(source_rows[row_id].get("response") or ""))

    coverage = row.get("criterion_coverage")
    if not isinstance(coverage, list):
        errors.append("criterion_coverage must be a list")
    else:
        for index, item in enumerate(coverage):
            if not isinstance(item, Mapping):
                errors.append(f"criterion_coverage[{index}] must be an object")
                continue
            criterion_id = str(item.get("criterion_id") or "").strip()
            family_id = str(item.get("family_id") or "").strip()
            if criterion_id and criterion_id not in criterion_ids:
                errors.append(f"unknown criterion_id {criterion_id!r}")
            if family_id not in family_ids:
                errors.append(f"unknown family_id {family_id!r}")

    claims = row.get("claims")
    if not isinstance(claims, list):
        errors.append("claims must be a list")
    else:
        for index, claim in enumerate(claims):
            span = _span(claim)
            if span is None:
                errors.append(f"claims[{index}] has invalid span")
            elif span[1] > response_len:
                errors.append(f"claims[{index}] span exceeds response length")

    commitment = row.get("commitment")
    if not isinstance(commitment, Mapping):
        errors.append("commitment must be an object")
    elif bool(commitment.get("has_commitment")):
        span = _span(commitment)
        if span is None:
            errors.append("commitment has invalid span")
        elif span[1] > response_len:
            errors.append("commitment span exceeds response length")

    controls = row.get("control_spans")
    if not isinstance(controls, Mapping):
        errors.append("control_spans must be an object")
    else:
        for name in ("matched_mid_reasoning", "same_position_noncommitment"):
            span = _span(controls.get(name))
            if span is None:
                errors.append(f"control_spans.{name} has invalid span")
            elif span[1] > response_len:
                errors.append(f"control_spans.{name} span exceeds response length")

    c_value = _c_label(row)
    if c_value not in {"early_collapse", "sustained_multi_consideration"}:
        errors.append("consideration.early_collapse_vs_sustained must be early_collapse or sustained_multi_consideration")
    return errors


def _load_primary_annotations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(ANNOTATION_DIR.glob("shard_*_annotations.jsonl")):
        rows.extend(_read_jsonl(path))
    return rows


def _agreement(primary_by_id: Mapping[str, Mapping[str, Any]], reviewer_rows: list[dict[str, Any]], family_ids: list[str]) -> dict[str, Any]:
    overlap = [row for row in reviewer_rows if str(row.get("row_id") or "") in primary_by_id]
    f_left: list[int] = []
    f_right: list[int] = []
    b_ious: list[float] = []
    c_left: list[str] = []
    c_right: list[str] = []

    for review in overlap:
        row_id = str(review["row_id"])
        primary = primary_by_id[row_id]
        primary_families = _coverage_family_set(primary)
        review_families = _coverage_family_set(review)
        for family_id in family_ids:
            f_left.append(1 if family_id in primary_families else 0)
            f_right.append(1 if family_id in review_families else 0)

        p_commit = primary.get("commitment") if isinstance(primary.get("commitment"), Mapping) else {}
        r_commit = review.get("commitment") if isinstance(review.get("commitment"), Mapping) else {}
        if bool(p_commit.get("has_commitment")) and bool(r_commit.get("has_commitment")):
            iou = _span_iou(_span(p_commit), _span(r_commit))
            if iou is not None:
                b_ious.append(iou)

        c_left.append(_c_label(primary))
        c_right.append(_c_label(review))

    f_kappa = float(cohen_kappa_score(f_left, f_right)) if len(set(f_left + f_right)) > 1 else None
    c_kappa = float(cohen_kappa_score(c_left, c_right)) if len(set(c_left + c_right)) > 1 else None
    b_share = float(np.mean([iou >= B_IOU_THRESHOLD for iou in b_ious])) if b_ious else None
    return {
        "overlap_count": len(overlap),
        "f_criterion_family_kappa": f_kappa,
        "b_commitment_span_iou_mean": float(np.mean(b_ious)) if b_ious else None,
        "b_commitment_span_iou_share_ge_0_5": b_share,
        "c_consideration_kappa": c_kappa,
        "gates": {
            "F": bool(f_kappa is not None and f_kappa >= F_KAPPA_THRESHOLD),
            "B": bool(b_share is not None and b_share >= B_SHARE_THRESHOLD),
            "C": bool(c_kappa is not None and c_kappa >= C_KAPPA_THRESHOLD),
        },
    }


def _reviewer_paths() -> list[Path]:
    recalibrated = sorted(ANNOTATION_DIR.glob("review_sample_30_annotations_recalibrated*.jsonl"))
    if recalibrated:
        return recalibrated
    return [DEFAULT_REVIEWER_PATH]


def _pairwise_reviewer_agreements(reviewer_payloads: Mapping[str, list[dict[str, Any]]], family_ids: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    items = list(reviewer_payloads.items())
    for index, (left_name, left_rows) in enumerate(items):
        left_by_id = {str(row.get("row_id") or ""): row for row in left_rows}
        for right_name, right_rows in items[index + 1 :]:
            key = f"{Path(left_name).name}__vs__{Path(right_name).name}"
            out[key] = _agreement(left_by_id, right_rows, family_ids)
    return out


def _aggregate_agreements(agreements: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    metric_keys = (
        "f_criterion_family_kappa",
        "b_commitment_span_iou_mean",
        "b_commitment_span_iou_share_ge_0_5",
        "c_consideration_kappa",
    )
    aggregate: dict[str, Any] = {}
    for metric in metric_keys:
        values = [
            float(record[metric])
            for record in agreements.values()
            if isinstance(record.get(metric), (int, float))
        ]
        aggregate[metric] = {
            "mean": float(np.mean(values)) if values else None,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
            "values": values,
        }
    aggregate["all_gates_pass"] = {
        "F": all(bool(record.get("gates", {}).get("F")) for record in agreements.values()) if agreements else False,
        "B": all(bool(record.get("gates", {}).get("B")) for record in agreements.values()) if agreements else False,
        "C": all(bool(record.get("gates", {}).get("C")) for record in agreements.values()) if agreements else False,
    }
    aggregate["any_gates_pass"] = {
        "F": any(bool(record.get("gates", {}).get("F")) for record in agreements.values()) if agreements else False,
        "B": any(bool(record.get("gates", {}).get("B")) for record in agreements.values()) if agreements else False,
        "C": any(bool(record.get("gates", {}).get("C")) for record in agreements.values()) if agreements else False,
    }
    return aggregate


def main() -> None:
    source_rows = {row["row_id"]: row for row in _read_jsonl(PACKET_DIR / "rows.jsonl")}
    criteria = _read_jsonl(PACKET_DIR / "criteria.jsonl")
    families_payload = _read_json(BASE_DIR / "criterion_families.json")
    family_ids = sorted(str(item["family_id"]) for item in families_payload.get("families", []))
    criterion_ids = {str(item["criterion_id"]) for item in criteria}

    primary_rows = _load_primary_annotations()
    reviewer_paths = _reviewer_paths()
    reviewer_payloads = {str(path): _read_jsonl(path) for path in reviewer_paths}

    errors: dict[str, list[str]] = {}
    seen = Counter(str(row.get("row_id") or "") for row in primary_rows)
    for row in primary_rows:
        row_id = str(row.get("row_id") or "")
        row_errors = _validate_row(row, source_rows=source_rows, criterion_ids=criterion_ids, family_ids=set(family_ids))
        if seen[row_id] > 1:
            row_errors.append("duplicate primary annotation")
        if row_errors:
            errors[row_id] = row_errors
    primary_by_id = {str(row.get("row_id")): row for row in primary_rows}
    agreements_by_reviewer = {
        path: _agreement(primary_by_id, rows, family_ids)
        for path, rows in reviewer_payloads.items()
    }
    aggregate_agreement = _aggregate_agreements(agreements_by_reviewer)
    pairwise_reviewer_agreements = _pairwise_reviewer_agreements(reviewer_payloads, family_ids)
    first_reviewer_path = str(reviewer_paths[0])
    agreement = agreements_by_reviewer[first_reviewer_path]

    summary = {
        "primary_annotation_count": len(primary_rows),
        "expected_primary_count": len(source_rows),
        "reviewer_annotation_count": len(reviewer_payloads[first_reviewer_path]),
        "reviewer_path": first_reviewer_path,
        "reviewer_paths": list(reviewer_payloads),
        "validation_error_count": sum(len(value) for value in errors.values()),
        "rows_with_errors": errors,
        "family_count": len(family_ids),
        "family_ids": family_ids,
        "coverage_family_counts": dict(Counter(family for row in primary_rows for family in _coverage_family_set(row))),
        "consideration_counts": dict(Counter(_c_label(row) for row in primary_rows)),
        "agreement": agreement,
        "agreements_by_reviewer": agreements_by_reviewer,
        "aggregate_agreement": aggregate_agreement,
        "pairwise_reviewer_agreements": pairwise_reviewer_agreements,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_jsonl(MERGED_PATH, sorted(primary_rows, key=lambda row: str(row.get("row_id") or "")))

    lines = [
        "# Experiment 03 Process-Supervision Labelability Report",
        "",
        f"- primary annotations: `{len(primary_rows)}` / `{len(source_rows)}`",
        f"- reviewer files: `{len(reviewer_payloads)}`",
        f"- first reviewer path: `{first_reviewer_path}`",
        f"- validation errors: `{summary['validation_error_count']}`",
        f"- F kappa: `{agreement['f_criterion_family_kappa']}`",
        f"- B span-IoU share >= 0.5: `{agreement['b_commitment_span_iou_share_ge_0_5']}`",
        f"- C kappa: `{agreement['c_consideration_kappa']}`",
        f"- gates: `{agreement['gates']}`",
        f"- aggregate all-gates-pass: `{aggregate_agreement['all_gates_pass']}`",
        f"- aggregate any-gates-pass: `{aggregate_agreement['any_gates_pass']}`",
        "",
        "## Reviewer Agreement Summary",
        "",
    ]
    for path, record in agreements_by_reviewer.items():
        lines.append(
            "- `{}`: F kappa `{}`, B IoU share `{}`, C kappa `{}`, gates `{}`".format(
                Path(path).name,
                record.get("f_criterion_family_kappa"),
                record.get("b_commitment_span_iou_share_ge_0_5"),
                record.get("c_consideration_kappa"),
                record.get("gates"),
            )
        )
    lines.extend([
        "",
        "## Reviewer-vs-Reviewer Summary",
        "",
    ])
    for key, record in pairwise_reviewer_agreements.items():
        lines.append(
            "- `{}`: F kappa `{}`, B IoU share `{}`, C kappa `{}`".format(
                key,
                record.get("f_criterion_family_kappa"),
                record.get("b_commitment_span_iou_share_ge_0_5"),
                record.get("c_consideration_kappa"),
            )
        )
    lines.extend([
        "",
        "## Top Covered Families",
        "",
    ])
    for family_id, count in Counter(summary["coverage_family_counts"]).most_common(30):
        lines.append(f"- `{family_id}`: {count}")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("primary_annotation_count", "validation_error_count")}, indent=2))


if __name__ == "__main__":
    main()
