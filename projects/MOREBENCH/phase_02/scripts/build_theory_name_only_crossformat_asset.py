from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PHASE_02_OUTPUTS = ROOT / "projects" / "MOREBENCH" / "phase_02" / "outputs"
PHASE_02_REPORTS = ROOT / "projects" / "MOREBENCH" / "phase_02" / "reports"

SOURCE_EXAMPLES_PATH = PHASE_02_OUTPUTS / "theory_prompt_repair_examples.jsonl"

EXAMPLES_PATH = PHASE_02_OUTPUTS / "theory_name_only_crossformat_examples.jsonl"
CONTROLS_PATH = PHASE_02_OUTPUTS / "theory_name_only_crossformat_controls.jsonl"
SUMMARY_PATH = PHASE_02_OUTPUTS / "theory_name_only_crossformat_summary.json"
REVIEW_PATH = PHASE_02_REPORTS / "theory_name_only_crossformat_review.md"

THEORY_ORDER = (
    "Act Utilitarianism",
    "Aristotelian Virtue Ethics",
    "Gauthierian Contractarianism",
    "Kantian Deontology",
    "Scanlonian Contractualism",
)

THEORY_SHORT = {
    "Act Utilitarianism": "utilitarian",
    "Aristotelian Virtue Ethics": "virtue_ethics",
    "Gauthierian Contractarianism": "contractarianism",
    "Kantian Deontology": "deontology",
    "Scanlonian Contractualism": "contractualism",
}

GENERIC_CONTROL_LABEL = "generic_ethics_control"

NAME_PHRASE_BANKS: dict[str, dict[str, object]] = {
    "name_a": {
        "style_name": "direct_apply",
        "generic": "Apply ethical reasoning.",
        "Act Utilitarianism": "Apply act-utilitarian reasoning.",
        "Aristotelian Virtue Ethics": "Apply virtue ethics.",
        "Gauthierian Contractarianism": "Apply contractarian reasoning.",
        "Kantian Deontology": "Apply Kantian deontology.",
        "Scanlonian Contractualism": "Apply Scanlonian contractualism.",
    },
    "name_b": {
        "style_name": "persona",
        "generic": "Reason morally.",
        "Act Utilitarianism": "Reason as an act utilitarian.",
        "Aristotelian Virtue Ethics": "Reason as a virtue ethicist.",
        "Gauthierian Contractarianism": "Reason as a contractarian.",
        "Kantian Deontology": "Reason as a Kantian.",
        "Scanlonian Contractualism": "Reason as a contractualist.",
    },
    "name_c": {
        "style_name": "moral_reasoning",
        "generic": "Use ordinary moral judgment.",
        "Act Utilitarianism": "Use utilitarian moral reasoning.",
        "Aristotelian Virtue Ethics": "Use Aristotelian virtue reasoning.",
        "Gauthierian Contractarianism": "Use a Gauthier-style contractarian view.",
        "Kantian Deontology": "Use Kantian moral reasoning.",
        "Scanlonian Contractualism": "Use a Scanlon-style contractualist view.",
    },
}


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _stable_split(group_id: str) -> str:
    digest = hashlib.md5(group_id.encode("utf-8")).hexdigest()
    return "test" if int(digest, 16) % 5 == 0 else "train"


def _extract_dilemma(prompt_text: str) -> str:
    marker = "\n\nDILEMMA:\n"
    if marker not in prompt_text:
        raise ValueError("Expected source prompt to contain a DILEMMA section.")
    return prompt_text.split(marker, 1)[1].strip()


def build_rows() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    source_examples = _load_jsonl(SOURCE_EXAMPLES_PATH)
    name_rows = [row for row in source_examples if str(row.get("variant_family")) == "name_only"]

    by_group_theory: dict[tuple[str, str], dict[str, object]] = {}
    group_meta: dict[str, dict[str, str]] = {}
    for row in name_rows:
        group_id = str(row["group_id"])
        theory = str(row["theory"])
        by_group_theory[(group_id, theory)] = row
        group_meta[group_id] = {
            "source_family": str(row["source_family"]),
            "dilemma_type": str(row["dilemma_type"]),
            "context": str(row["context"]),
            "role_domain": str(row["role_domain"]),
        }

    example_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []

    for group_id in sorted(group_meta):
        split = _stable_split(group_id)
        meta = group_meta[group_id]
        for bank_id, payload in NAME_PHRASE_BANKS.items():
            generic_cue = str(payload["generic"])
            # Reuse any theory row for the dilemma body because name_only source rows share the same dilemma.
            source_row = by_group_theory[(group_id, THEORY_ORDER[0])]
            dilemma = _extract_dilemma(str(source_row["prompt"]))
            control_rows.append(
                {
                    "example_id": f"{group_id}__{GENERIC_CONTROL_LABEL}__{bank_id}",
                    "group_id": group_id,
                    "augmentation_family": "crossformat_name_only",
                    "control_type": GENERIC_CONTROL_LABEL,
                    "variant_family": "name_only_crossformat",
                    "cue_mode": "name_only",
                    "name_bank": bank_id,
                    "variant_style": str(payload["style_name"]),
                    "split": split,
                    "cue_text": generic_cue,
                    "prompt": f"{generic_cue}\n\nDILEMMA:\n{dilemma}",
                    **meta,
                }
            )
            for theory in THEORY_ORDER:
                source_row = by_group_theory[(group_id, theory)]
                dilemma = _extract_dilemma(str(source_row["prompt"]))
                cue_text = str(payload[theory])
                example_rows.append(
                    {
                        "example_id": f"{group_id}__{THEORY_SHORT[theory]}__{bank_id}",
                        "group_id": group_id,
                        "augmentation_family": "crossformat_name_only",
                        "variant_family": "name_only_crossformat",
                        "cue_mode": "name_only",
                        "theory": theory,
                        "prime_condition": THEORY_SHORT[theory],
                        "split": split,
                        "name_bank": bank_id,
                        "variant_style": str(payload["style_name"]),
                        "cue_text": cue_text,
                        "prompt": f"{cue_text}\n\nDILEMMA:\n{dilemma}",
                        **meta,
                    }
                )

    summary = {
        "benchmark": "morebench",
        "phase": "02",
        "artifact_family": "theory_name_only_crossformat",
        "status": "materialized_for_review",
        "purpose": (
            "Short imperative name-only family intended primarily as a held-out format for description->name "
            "or name->description transfer tests."
        ),
        "counts": {
            "matched_groups": len(group_meta),
            "theory_count": len(THEORY_ORDER),
            "name_bank_count": len(NAME_PHRASE_BANKS),
            "example_row_count": len(example_rows),
            "generic_control_row_count": len(control_rows),
            "total_prompt_count": len(example_rows) + len(control_rows),
        },
        "name_banks": {
            bank_id: {
                "style_name": str(payload["style_name"]),
                "generic": str(payload["generic"]),
                "theory_cues": {theory: str(payload[theory]) for theory in THEORY_ORDER},
            }
            for bank_id, payload in NAME_PHRASE_BANKS.items()
        },
        "intended_use": {
            "primary": "cross-format held-out evaluation against description_variant_sweep",
            "secondary": "within-name-only phrasing invariance sanity",
            "note": "within-format lexical baselines are expected to be near-ceiling because the theory names remain explicit",
        },
        "output_paths": {
            "examples": str(EXAMPLES_PATH.relative_to(ROOT)),
            "controls": str(CONTROLS_PATH.relative_to(ROOT)),
            "summary": str(SUMMARY_PATH.relative_to(ROOT)),
            "review": str(REVIEW_PATH.relative_to(ROOT)),
        },
    }
    return example_rows, control_rows, summary


def write_review(summary: dict[str, object]) -> None:
    lines: list[str] = []
    lines.append("# Theory Name-Only Cross-Format Review\n")
    lines.append(
        "This asset is intended mainly as a held-out format for cross-format transfer against the description "
        "variant sweep. It is not expected to beat within-format lexical baselines.\n\n"
    )
    lines.append("## Name Banks\n")
    for bank_id, payload in dict(summary["name_banks"]).items():
        lines.append(f"### `{bank_id}` / `{payload['style_name']}`\n")
        lines.append(f"- generic: {payload['generic']}\n")
        for theory in THEORY_ORDER:
            lines.append(f"- {theory}: {payload['theory_cues'][theory]}\n")
        lines.append("")
    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_PATH.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    example_rows, control_rows, summary = build_rows()
    _write_jsonl(EXAMPLES_PATH, example_rows)
    _write_jsonl(CONTROLS_PATH, control_rows)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_review(summary)
    print(str(EXAMPLES_PATH.relative_to(ROOT)))
    print(str(CONTROLS_PATH.relative_to(ROOT)))
    print(str(SUMMARY_PATH.relative_to(ROOT)))
    print(str(REVIEW_PATH.relative_to(ROOT)))


if __name__ == "__main__":
    main()
