from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines_v2.storage.artifacts import ArtifactManifest, OperationArtifact
from pipelines_v2.storage.modal import ModalVolumeStore
from projects.DX_TERMINAL.prompt_confusion.neon import connect_neon, validate_table_name


DEFAULT_RESULT_JSON = (
    Path("projects/DX_TERMINAL/prompt_confusion/phase_13/reports/signal_discovery")
    / "report_922d1299ea2c_c7599a0a/results/coarse_projection_grid_results.json"
)
DEFAULT_TRANSFORM_MANIFEST = Path.home() / ".xenon/pipelines_v2/catalog/transform_1_0a089d56.json"
DEFAULT_OUTPUT_JSON = (
    Path("projects/DX_TERMINAL/prompt_confusion/phase_13/reports/signal_discovery")
    / "report_922d1299ea2c_c7599a0a/results/l44_strategies_topk_complaint_review.json"
)
DEFAULT_LAYER = 44
DEFAULT_POSITION = "strategies_end"
DEFAULT_CORPUS_TABLE = "dx_terminal_signal_discovery_phase13_smoke_v1"
DEFAULT_DIRECTIONS = (
    "trade_size",
    "risk_preference",
    "diversification_preference",
    "shared_mean",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_from_manifest(path: Path) -> OperationArtifact:
    manifest = ArtifactManifest.from_dict(_load_json(path))
    store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/prompt_confusion_phase13_signal_discovery",
    )
    return OperationArtifact(_manifest=manifest, store=store)


def _values_from_label(payload: Mapping[str, Any]) -> dict[str, float]:
    values = payload.get("values", payload)
    if not isinstance(values, Mapping):
        raise TypeError("Projection label payload must be a mapping or contain a values mapping")
    return {str(key): float(value) for key, value in values.items()}


def _section_between(text: str, start_markers: Sequence[str], end_markers: Sequence[str], *, limit: int = 2200) -> str:
    starts = [text.find(marker) for marker in start_markers if text.find(marker) >= 0]
    if not starts:
        return ""
    start = min(starts)
    ends = [text.find(marker, start + 1) for marker in end_markers if text.find(marker, start + 1) >= 0]
    end = min(ends) if ends else min(len(text), start + limit)
    section = text[start:end]
    return re.sub(r"[ \t\r\n]+", " ", section).strip()[:limit]


def _strategy_excerpt(text: str) -> str:
    return _section_between(
        text,
        start_markers=(
            "## ACTIVE STRATEGIES",
            "\nSTRATEGIES\n",
            "STRATEGIES\n",
        ),
        end_markers=(
            "## ACTIVE SETTINGS",
            "\nSETTINGS\n",
            "SETTINGS\n",
            "## PORTFOLIO",
            "\nPORTFOLIO\n",
        ),
    )


def _strategy_directives_excerpt(text: str) -> str:
    section = _section_between(
        text,
        start_markers=("## ACTIVE STRATEGIES",),
        end_markers=("## ACTIVE SETTINGS",),
        limit=8000,
    )
    if not section:
        return ""
    if "- No active strategies." in section:
        return "- No active strategies."
    directives = re.findall(r"- \[[^\]]+\] [^-]+ - [^-].*?(?= - \[[^\]]+\] [^-]+ - | ------------------------------|$)", section)
    if directives:
        return " ".join(directive.strip() for directive in directives)[:2200]
    return section[-2200:]


def _settings_excerpt(text: str) -> str:
    return _section_between(
        text,
        start_markers=(
            "## ACTIVE SETTINGS",
            "\nSETTINGS\n",
            "SETTINGS\n",
        ),
        end_markers=(
            "## PORTFOLIO",
            "\nPORTFOLIO\n",
            "PORTFOLIO\n",
            "## MARKET",
            "\nMARKET\n",
        ),
    )


def _decision_excerpt(source_row: Mapping[str, Any]) -> str:
    calls = source_row.get("completion_tool_calls_json")
    if isinstance(calls, list) and calls:
        return json.dumps(calls[0], ensure_ascii=False, sort_keys=True)[:1400]
    payload = source_row.get("raw_completion_payload_json")
    if isinstance(payload, Mapping):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            return json.dumps(choices[0], ensure_ascii=False, sort_keys=True)[:1400]
    return ""


def _row_payload(row: Mapping[str, Any], *, direction: str, rank_kind: str, rank: int, projection: float, control_mean: float) -> dict[str, Any]:
    source_row = row.get("source_row_json")
    if not isinstance(source_row, Mapping):
        source_row = {}
    prompt_text = str(source_row.get("prompt_text") or row.get("prompt_text") or "")
    return {
        "direction": direction,
        "rank_kind": rank_kind,
        "rank": rank,
        "projection": projection,
        "distance_to_structure_control_mean": abs(projection - control_mean),
        "example_id": str(row["example_id"]),
        "source_example_id": row.get("source_example_id"),
        "trace_id": row.get("trace_id"),
        "vault_address": row.get("vault_address"),
        "person_id": row.get("person_id"),
        "stratum": row.get("stratum"),
        "label": row.get("label"),
        "fault": row.get("fault"),
        "root_cause": row.get("root_cause"),
        "complaint_type": row.get("complaint_type"),
        "severity": row.get("severity"),
        "confidence": row.get("confidence"),
        "slider_ta": row.get("slider_ta"),
        "slider_arp": row.get("slider_arp"),
        "slider_ts": row.get("slider_ts"),
        "slider_hs": row.get("slider_hs"),
        "slider_div": row.get("slider_div"),
        "size_relevant_complaint": row.get("size_relevant_complaint"),
        "activity_relevant_complaint": row.get("activity_relevant_complaint"),
        "config_conflict_like": row.get("config_conflict_like"),
        "system_fault": row.get("system_fault"),
        "strategy_excerpt": _strategy_excerpt(prompt_text),
        "strategy_directives_excerpt": _strategy_directives_excerpt(prompt_text),
        "settings_excerpt": _settings_excerpt(prompt_text),
        "decision_excerpt": _decision_excerpt(source_row),
        "complaint_text": row.get("complaint_text"),
    }


def _fetch_rows(table: str, example_ids: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    if not example_ids:
        return {}
    table_name = validate_table_name(table)
    with connect_neon(autocommit=True) as conn:
        rows = conn.execute(
            f"""
            SELECT
              example_id,
              source_example_id,
              trace_id,
              vault_address,
              person_id,
              stratum,
              label,
              fault,
              root_cause,
              complaint_type,
              complaint_text,
              severity,
              confidence,
              slider_ta,
              slider_arp,
              slider_ts,
              slider_hs,
              slider_div,
              size_relevant_complaint,
              activity_relevant_complaint,
              config_conflict_like,
              system_fault,
              prompt_text,
              source_row_json
            FROM {table_name}
            WHERE example_id = ANY(%s)
            """,
            (list(example_ids),),
        ).fetchall()
    return {str(row["example_id"]): row for row in rows}


def build_review(
    *,
    result_json: Path,
    transform_manifest: Path,
    corpus_table: str,
    output_json: Path,
    layer: int,
    position: str,
    top_k: int,
    directions: Sequence[str],
) -> dict[str, Any]:
    result = _load_json(result_json)
    artifact = _artifact_from_manifest(transform_manifest)
    direction_payloads: dict[str, dict[str, Any]] = {}
    wanted_ids: set[str] = set()

    for direction in directions:
        label_name = f"projection__L{layer}__{position}__{direction}"
        scores = _values_from_label(artifact.load_label(label_name))
        complaint_scores = {
            example_id: value
            for example_id, value in scores.items()
            if ":complaint:" in example_id and example_id.endswith(":aggressive")
        }
        cell = result["cells"][f"L{layer}:{position}:aggressive:{direction}"]
        means = cell["stratum_means"]
        control_mean = float(means["structure_matched_control"])
        top_ids = [
            example_id
            for example_id, _ in sorted(complaint_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        ]
        bottom_ids = [
            example_id
            for example_id, _ in sorted(
                complaint_scores.items(),
                key=lambda item: (abs(item[1] - control_mean), item[1]),
            )[:top_k]
        ]
        wanted_ids.update(top_ids)
        wanted_ids.update(bottom_ids)
        direction_payloads[direction] = {
            "label_name": label_name,
            "stratum_means": means,
            "control_mean": control_mean,
            "scores": complaint_scores,
            "top_ids": top_ids,
            "bottom_ids": bottom_ids,
        }

    rows_by_id = _fetch_rows(corpus_table, sorted(wanted_ids))
    review: dict[str, Any] = {
        "kind": "phase13_topk_complaint_review",
        "source_result_json": str(result_json),
        "source_transform_artifact": artifact.id,
        "corpus_table": corpus_table,
        "cell": {
            "layer": layer,
            "position": position,
            "prompt_tier": "aggressive",
        },
        "top_k": top_k,
        "directions": {},
    }

    for direction, payload in direction_payloads.items():
        rows: dict[str, list[dict[str, Any]]] = {"top_complaints": [], "bottom_complaints_closest_to_control": []}
        for rank_kind, key_name in (
            ("top_complaints", "top_ids"),
            ("bottom_complaints_closest_to_control", "bottom_ids"),
        ):
            for rank, example_id in enumerate(payload[key_name], start=1):
                row = rows_by_id.get(example_id)
                if row is None:
                    raise KeyError(f"Missing Neon row for {example_id}")
                rows[rank_kind].append(
                    _row_payload(
                        row,
                        direction=direction,
                        rank_kind=rank_kind,
                        rank=rank,
                        projection=payload["scores"][example_id],
                        control_mean=payload["control_mean"],
                    )
                )
        review["directions"][direction] = {
            "label_name": payload["label_name"],
            "stratum_means": payload["stratum_means"],
            **rows,
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, default=DEFAULT_RESULT_JSON)
    parser.add_argument("--transform-manifest", type=Path, default=DEFAULT_TRANSFORM_MANIFEST)
    parser.add_argument("--corpus-table", default=DEFAULT_CORPUS_TABLE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--position", default=DEFAULT_POSITION)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--directions", nargs="+", default=list(DEFAULT_DIRECTIONS))
    args = parser.parse_args()
    review = build_review(
        result_json=args.result_json,
        transform_manifest=args.transform_manifest,
        corpus_table=args.corpus_table,
        output_json=args.output_json,
        layer=args.layer,
        position=args.position,
        top_k=args.top_k,
        directions=args.directions,
    )
    print(json.dumps({"output_json": str(args.output_json), "directions": list(review["directions"])}, indent=2))


if __name__ == "__main__":
    main()
