from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
PHASE_02_OUTPUTS = ROOT / "projects" / "MECH_INTERP" / "morebench" / "phase_02" / "outputs"
PHASE_03_OUTPUTS = ROOT / "projects" / "MECH_INTERP" / "morebench" / "phase_03" / "outputs"

DIRECT_PATH = PHASE_02_OUTPUTS / "theory_prompt_augmentation_examples.jsonl"
VARIANT_PATH = PHASE_02_OUTPUTS / "theory_wording_variant_examples.jsonl"
NEUTRAL_PATH = PHASE_02_OUTPUTS / "theory_control_augmentation_examples.jsonl"

DATASET_PATH = PHASE_03_OUTPUTS / "experiment_01_prompt_dataset.jsonl"
SUMMARY_PATH = PHASE_03_OUTPUTS / "experiment_01_dataset_summary.json"


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _split_for_group(group_id: str) -> str:
    digest = hashlib.md5(group_id.encode("utf-8")).hexdigest()
    return "test" if int(digest, 16) % 5 == 0 else "train"


def _anchor_only_prompt(anchor: str, dilemma: str) -> str:
    return f"Analyze the dilemma. {anchor}\n\nDILEMMA:\n{dilemma}"


def build_records() -> list[dict[str, object]]:
    direct_rows = _load_jsonl(DIRECT_PATH)
    variant_rows = _load_jsonl(VARIANT_PATH)
    neutral_rows = _load_jsonl(NEUTRAL_PATH)

    records: list[dict[str, object]] = []

    for row in direct_rows:
        group_id = str(row["group_id"])
        theory = str(row["theory"])
        anchor = str(row["theory_anchor"])
        base_dilemma = str(row["base_dilemma"])
        common = {
            "group_id": group_id,
            "split": _split_for_group(group_id),
            "theory_identity": theory,
            "theory_or_control": theory,
            "source_family": str(row["source_family"]),
            "dilemma_type": str(row["dilemma_type"]),
            "context": str(row["context"]),
            "role_domain": str(row["role_domain"]),
            "is_theory_bearing": True,
            "is_named_theory": True,
            "has_theory_name": True,
            "anchor_text": anchor,
        }
        records.append(
            {
                "example_id": str(row["example_id"]),
                "prompt": str(row["augmented_prompt"]),
                "prompt_family": "theory_direct",
                **common,
            }
        )
        records.append(
            {
                "example_id": str(row["example_id"]).replace("__direct", "__anchor_only"),
                "prompt": _anchor_only_prompt(anchor, base_dilemma),
                "prompt_family": "anchor_only",
                "theory_identity": theory,
                "theory_or_control": theory,
                "source_family": str(row["source_family"]),
                "dilemma_type": str(row["dilemma_type"]),
                "context": str(row["context"]),
                "role_domain": str(row["role_domain"]),
                "group_id": group_id,
                "split": _split_for_group(group_id),
                "is_theory_bearing": True,
                "is_named_theory": False,
                "has_theory_name": False,
                "anchor_text": anchor,
            }
        )

    for row in variant_rows:
        group_id = str(row["group_id"])
        theory = str(row["theory"])
        records.append(
            {
                "example_id": str(row["example_id"]),
                "prompt": str(row["augmented_prompt"]),
                "prompt_family": "theory_wording_variant",
                "theory_identity": theory,
                "theory_or_control": theory,
                "source_family": str(row["source_family"]),
                "dilemma_type": str(row["dilemma_type"]),
                "context": str(row["context"]),
                "role_domain": str(row["role_domain"]),
                "group_id": group_id,
                "split": _split_for_group(group_id),
                "is_theory_bearing": True,
                "is_named_theory": True,
                "has_theory_name": True,
                "anchor_text": str(row["theory_anchor"]),
            }
        )

    for row in neutral_rows:
        group_id = str(row["group_id"])
        records.append(
            {
                "example_id": str(row["example_id"]),
                "prompt": str(row["prompt"]),
                "prompt_family": "neutral_control",
                "theory_identity": "neutral_control",
                "theory_or_control": "neutral_control",
                "source_family": str(row["source_family"]),
                "dilemma_type": str(row["dilemma_type"]),
                "context": str(row["context"]),
                "role_domain": str(row["role_domain"]),
                "group_id": group_id,
                "split": _split_for_group(group_id),
                "is_theory_bearing": False,
                "is_named_theory": False,
                "has_theory_name": False,
                "anchor_text": "",
            }
        )

    return records


def write_outputs() -> None:
    records = build_records()
    PHASE_03_OUTPUTS.mkdir(parents=True, exist_ok=True)

    with DATASET_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    family_counts = Counter(str(record["prompt_family"]) for record in records)
    split_counts = Counter(str(record["split"]) for record in records)
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        grouped[str(record["prompt_family"])][str(record["split"])] += 1

    summary = {
        "benchmark": "morebench",
        "phase": "03",
        "experiment": "experiment_01_theory_identity_prompt_readout",
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "row_count": len(records),
        "family_counts": dict(family_counts),
        "split_counts": dict(split_counts),
        "family_split_counts": {family: dict(counter) for family, counter in grouped.items()},
        "anti_shortcut_family": "anchor_only",
        "anti_shortcut_rule": "Remove the explicit theory name while preserving the framework anchor sentence and base dilemma.",
        "label_space": sorted({str(record["theory_or_control"]) for record in records}),
        "group_count": len({str(record["group_id"]) for record in records}),
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    write_outputs()
