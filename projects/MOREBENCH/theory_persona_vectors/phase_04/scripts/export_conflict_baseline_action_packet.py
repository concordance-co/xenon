"""Export Phase 04 conflict baselines for action-equivalence review."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_04")
REPORT_ROOT = PHASE_ROOT / "reports" / "conflict_baseline_report"
OUTPUT_JSONL = PHASE_ROOT / "outputs" / "conflict_baseline_action_review_packet.jsonl"
OUTPUT_MD = PHASE_ROOT / "reports" / "conflict_baseline_action_review_packet.md"

CONDITION_ORDER = ("N_neutral_01", "N_generic_moral_01", "P_deont_01", "P_util_01")


def _latest_generation_result() -> Path:
    candidates = sorted(
        REPORT_ROOT.glob("report_*/results/generate_conflict_baselines_results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no conflict baseline generation result under {REPORT_ROOT}")
    return candidates[0]


def _text(row: dict[str, Any]) -> str:
    return str(row.get("generated_text") or "").strip()


def _labels(row: dict[str, Any]) -> dict[str, Any]:
    example = row.get("example") if isinstance(row.get("example"), dict) else {}
    labels = example.get("labels") if isinstance(example.get("labels"), dict) else {}
    return dict(labels)


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    example = row.get("example") if isinstance(row.get("example"), dict) else {}
    metadata = example.get("metadata") if isinstance(example.get("metadata"), dict) else {}
    return dict(metadata)


def build_packet(result_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError("generation result must contain rows list")

    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    group_meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        labels = _labels(row)
        metadata = _metadata(row)
        group_id = str(labels.get("group_id") or "")
        condition_id = str(labels.get("condition_id") or "")
        if not group_id or not condition_id:
            continue
        group_meta.setdefault(
            group_id,
            {
                "group_id": group_id,
                "subset": labels.get("subset"),
                "source_family": labels.get("source_family"),
                "role_domain": labels.get("role_domain"),
                "split_shape": labels.get("split_shape"),
                "action_clusters": labels.get("action_clusters") or [],
                "minority_primes": labels.get("minority_primes") or [],
                "is_primary_steering_candidate": labels.get("is_primary_steering_candidate"),
                "is_tie_3_3": labels.get("is_tie_3_3"),
                "recommended_use": labels.get("recommended_use"),
                "dilemma_text": metadata.get("dilemma_text"),
            },
        )
        grouped[group_id][condition_id] = {
            "text": _text(row),
            "finish_reason": row.get("finish_reason"),
            "token_count": len(row.get("generated_token_ids") or []),
        }

    packet: list[dict[str, Any]] = []
    for group_id in sorted(grouped):
        conditions = {condition: grouped[group_id].get(condition, {}) for condition in CONDITION_ORDER}
        packet.append({**group_meta[group_id], "conditions": conditions})
    return packet


def write_jsonl(packet: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in packet:
            handle.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")


def write_markdown(packet: list[dict[str, Any]], path: Path, *, source_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Phase 04 Conflict Baseline Action Review Packet",
        "",
        f"- source: `{source_path}`",
        f"- groups: `{len(packet)}`",
        f"- conditions: `{', '.join(CONDITION_ORDER)}`",
        "",
        "Use this packet to assign coarse action-equivalence labels before steering. The primary causal denominator should exclude `is_tie_3_3=true` groups unless explicitly reported as diagnostic.",
        "",
    ]
    for item in packet:
        lines.extend(
            [
                f"## {item['group_id']}",
                "",
                f"- primary_steering_candidate: `{item['is_primary_steering_candidate']}`",
                f"- tie_3_3: `{item['is_tie_3_3']}`",
                f"- recommended_use: `{item['recommended_use']}`",
                f"- prior_split_shape: `{item['split_shape']}`",
                f"- prior_action_clusters: `{json.dumps(item['action_clusters'], ensure_ascii=False)}`",
                f"- prior_minority_primes: `{json.dumps(item['minority_primes'], ensure_ascii=False)}`",
                "",
                "### Dilemma",
                "",
                str(item.get("dilemma_text") or "").strip(),
                "",
            ]
        )
        conditions = item["conditions"]
        for condition_id in CONDITION_ORDER:
            condition = conditions.get(condition_id) or {}
            lines.extend(
                [
                    f"### {condition_id}",
                    "",
                    f"- finish_reason: `{condition.get('finish_reason')}`",
                    f"- token_count: `{condition.get('token_count')}`",
                    "",
                    str(condition.get("text") or "").strip(),
                    "",
                ]
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    result_path = _latest_generation_result()
    packet = build_packet(result_path)
    write_jsonl(packet, OUTPUT_JSONL)
    write_markdown(packet, OUTPUT_MD, source_path=result_path)
    primary = sum(1 for item in packet if item.get("is_primary_steering_candidate"))
    ties = sum(1 for item in packet if item.get("is_tie_3_3"))
    print(json.dumps({"groups": len(packet), "primary": primary, "ties": ties, "jsonl": str(OUTPUT_JSONL), "markdown": str(OUTPUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
