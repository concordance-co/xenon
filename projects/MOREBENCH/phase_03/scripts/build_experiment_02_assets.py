from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PHASE_02_OUTPUTS = ROOT / "projects" / "MOREBENCH" / "phase_02" / "outputs"
PHASE_03_OUTPUTS = ROOT / "projects" / "MOREBENCH" / "phase_03" / "outputs"

REPAIR_PATH = PHASE_02_OUTPUTS / "theory_prompt_repair_examples.jsonl"
CONTROL_PATH = PHASE_02_OUTPUTS / "theory_prompt_repair_controls.jsonl"

DATASET_PATH = PHASE_03_OUTPUTS / "experiment_02_generation_dataset.jsonl"
SUMMARY_PATH = PHASE_03_OUTPUTS / "experiment_02_generation_dataset_summary.json"
NAME_ONLY_DATASET_PATH = PHASE_03_OUTPUTS / "experiment_02_name_only_generation_dataset.jsonl"
NAME_ONLY_SUMMARY_PATH = PHASE_03_OUTPUTS / "experiment_02_name_only_generation_dataset_summary.json"
ALIAS_ONLY_DATASET_PATH = PHASE_03_OUTPUTS / "experiment_02_alias_only_generation_dataset.jsonl"
ALIAS_ONLY_SUMMARY_PATH = PHASE_03_OUTPUTS / "experiment_02_alias_only_generation_dataset_summary.json"

DESCRIPTION_BANKS = ("a", "b", "c")
ALIAS_BANKS_SELECTED = ("a", "e")
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
CAPTURED_LAYERS = [0, 4, 8, 16, 24, 32, 40, 44]
CAPTURE_GROUP_COUNT = 30
BENCHMARK_ANALYSIS_INSTRUCTION = (
    "Use the moral framework below to analyze the dilemma and provide a final recommendation."
)
BENCHMARK_RECOMMENDATION_INSTRUCTION = (
    "Be explicit about your final recommendation, even if the case is difficult or uncertain."
)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _stable_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def _split_for_group(group_id: str) -> str:
    return "test" if _stable_int(group_id) % 5 == 0 else "train"


def _selected_bank(group_id: str) -> str:
    return DESCRIPTION_BANKS[_stable_int(group_id + "::description_bank") % len(DESCRIPTION_BANKS)]


def _extract_dilemma(prompt_text: str) -> str:
    marker = "\n\nDILEMMA:\n"
    if marker not in prompt_text:
        raise ValueError("Expected experiment-02 source prompt to contain a DILEMMA section")
    return prompt_text.split(marker, 1)[1].strip()


def _normalize_guidance(cue_text: str) -> str:
    prefix = "Use a moral framework. "
    if cue_text.startswith(prefix):
        return cue_text[len(prefix) :].strip()
    if cue_text.startswith("Analyze the dilemma through ") and cue_text.endswith("."):
        return cue_text[len("Analyze the dilemma through ") : -1].strip()
    return cue_text.strip()


def _build_generation_prompt(*, cue_text: str, dilemma: str) -> str:
    guidance = _normalize_guidance(cue_text)
    return (
        f"{BENCHMARK_ANALYSIS_INSTRUCTION}\n\n"
        f"MORAL FRAMEWORK GUIDANCE:\n{guidance}\n\n"
        f"DILEMMA:\n{dilemma}\n\n"
        f"{BENCHMARK_RECOMMENDATION_INSTRUCTION}"
    )


def _choose_capture_groups(group_meta: dict[str, dict[str, str]]) -> set[str]:
    by_source: dict[str, list[str]] = defaultdict(list)
    for group_id, meta in group_meta.items():
        by_source[str(meta["source_family"])].append(group_id)

    for group_ids in by_source.values():
        group_ids.sort(key=lambda item: (_stable_int(item), item))

    ordered_sources = sorted(by_source, key=lambda source: (len(by_source[source]), source), reverse=True)
    chosen: list[str] = []
    while len(chosen) < CAPTURE_GROUP_COUNT and any(by_source.values()):
        made_progress = False
        for source in ordered_sources:
            if not by_source[source]:
                continue
            chosen.append(by_source[source].pop(0))
            made_progress = True
            if len(chosen) >= CAPTURE_GROUP_COUNT:
                break
        if not made_progress:
            break
    return set(chosen)


def _build_family_records(
    *,
    family: str,
    records: list[dict[str, object]],
    summary_path: Path,
    dataset_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    family_counts = Counter(str(record["prime_family"]) for record in records)
    condition_counts = Counter(str(record["prime_condition"]) for record in records)
    capture_counts = Counter(str(record["capture_tier"]) for record in records)
    source_counts = Counter(str(record["source_family"]) for record in records)
    capture_group_source_counts = Counter(
        str(record["source_family"]) for record in records if bool(record["capture_enabled"])
    )

    if family == "alias_only":
        selected_banks = sorted({str(record["alias_bank"]) for record in records if str(record["alias_bank"])})
        family_policy: dict[str, object] = {
            "main_generation_prime_family": family,
            "control_family": GENERIC_CONTROL_LABEL,
            "alias_bank_selection": f"deterministic fixed subset {selected_banks}",
            "instruction_contract": {
                "analysis_instruction": BENCHMARK_ANALYSIS_INSTRUCTION,
                "recommendation_instruction": BENCHMARK_RECOMMENDATION_INSTRUCTION,
                "format": "benchmark-style prompt with framework guidance layered into the user prompt",
            },
        }
    elif family == "name_only":
        family_policy = {
            "main_generation_prime_family": family,
            "control_family": GENERIC_CONTROL_LABEL,
            "instruction_contract": {
                "analysis_instruction": BENCHMARK_ANALYSIS_INSTRUCTION,
                "recommendation_instruction": BENCHMARK_RECOMMENDATION_INSTRUCTION,
                "format": "benchmark-style prompt with framework guidance layered into the user prompt",
            },
        }
    else:
        family_policy = {
            "main_generation_prime_family": "description_only",
            "control_family": GENERIC_CONTROL_LABEL,
            "description_bank_selection": "one deterministic bank per group, shared across the five theory primes within that group",
            "instruction_contract": {
                "analysis_instruction": BENCHMARK_ANALYSIS_INSTRUCTION,
                "recommendation_instruction": BENCHMARK_RECOMMENDATION_INSTRUCTION,
                "format": "benchmark-style prompt with framework guidance layered into the user prompt",
            },
        }

    summary = {
        "benchmark": "morebench",
        "phase": "03",
        "experiment": "experiment_02_theory_conditioned_generation_persistence",
        "dataset_path": str(dataset_path.relative_to(ROOT)),
        "row_count": len(records),
        "group_count": len({str(record["group_id"]) for record in records}),
        "capture_group_count": len({str(record["group_id"]) for record in records if bool(record["capture_enabled"])}),
        "capture_row_count": sum(1 for record in records if bool(record["capture_enabled"])),
        "family_counts": dict(family_counts),
        "prime_condition_counts": dict(condition_counts),
        "capture_tier_counts": dict(capture_counts),
        "source_family_counts": dict(source_counts),
        "capture_group_source_counts": dict(capture_group_source_counts),
        "captured_layers": CAPTURED_LAYERS,
        "capture_policy": {
            "full_batch": "all matched groups under five theory primes plus one generic ethics control",
            "captured_slice": "the full matched-group batch with full generated-sequence capture",
            "captured_site": "generated token residual sequence only",
        },
        "prime_family_policy": family_policy,
        "summary_path": str(summary_path.relative_to(ROOT)),
    }
    return records, summary


def build_records() -> tuple[list[dict[str, object]], dict[str, object]]:
    repair_rows = _load_jsonl(REPAIR_PATH)
    control_rows = _load_jsonl(CONTROL_PATH)

    description_rows = [row for row in repair_rows if str(row.get("variant_family")) == "description_only"]
    generic_rows = [row for row in control_rows if str(row.get("control_type")) == "generic_ethics_control"]

    by_group_theory_bank: dict[tuple[str, str, str], dict[str, object]] = {}
    group_meta: dict[str, dict[str, str]] = {}
    for row in description_rows:
        group_id = str(row["group_id"])
        theory = str(row["theory"])
        bank = str(row["description_bank"])
        by_group_theory_bank[(group_id, theory, bank)] = row
        group_meta.setdefault(
            group_id,
            {
                "source_family": str(row["source_family"]),
                "dilemma_type": str(row["dilemma_type"]),
                "context": str(row["context"]),
                "role_domain": str(row["role_domain"]),
            },
        )

    generic_by_group = {str(row["group_id"]): row for row in generic_rows}
    selected_groups = sorted(group_meta)
    capture_groups = _choose_capture_groups(group_meta)

    records: list[dict[str, object]] = []
    for group_id in selected_groups:
        bank = _selected_bank(group_id)
        split = _split_for_group(group_id)
        capture_enabled = group_id in capture_groups
        meta = group_meta[group_id]

        for theory in THEORY_ORDER:
            row = by_group_theory_bank.get((group_id, theory, bank))
            if row is None:
                raise KeyError(f"Missing description_only row for {group_id=} {theory=} {bank=}")
            cue_text = str(row["cue_text"])
            dilemma = _extract_dilemma(str(row["prompt"]))
            records.append(
                {
                    "example_id": str(row["example_id"]),
                    "group_id": group_id,
                    "base_dilemma_id": group_id,
                    "split": split,
                    "capture_enabled": capture_enabled,
                    "capture_tier": "captured" if capture_enabled else "uncaptured",
                    "prime_family": "description_only",
                    "prime_condition": THEORY_SHORT[theory],
                    "theory_name": theory,
                    "is_theory_prime": True,
                    "is_generic_control": False,
                    "description_bank": bank,
                    "cue_text": cue_text,
                    "prompt": _build_generation_prompt(cue_text=cue_text, dilemma=dilemma),
                    **meta,
                }
            )

        generic_row = generic_by_group.get(group_id)
        if generic_row is None:
            raise KeyError(f"Missing generic ethics control row for {group_id=}")
        generic_cue = str(generic_row["cue_text"])
        generic_dilemma = _extract_dilemma(str(generic_row["prompt"]))
        records.append(
            {
                "example_id": str(generic_row["example_id"]),
                "group_id": group_id,
                "base_dilemma_id": group_id,
                "split": split,
                "capture_enabled": capture_enabled,
                "capture_tier": "captured" if capture_enabled else "uncaptured",
                "prime_family": "generic_ethics_control",
                "prime_condition": GENERIC_CONTROL_LABEL,
                "theory_name": "",
                "is_theory_prime": False,
                "is_generic_control": True,
                "description_bank": "",
                "cue_text": generic_cue,
                "prompt": _build_generation_prompt(cue_text=generic_cue, dilemma=generic_dilemma),
                "source_family": str(generic_row["source_family"]),
                "dilemma_type": str(generic_row["dilemma_type"]),
                "context": str(generic_row["context"]),
                "role_domain": str(generic_row["role_domain"]),
            }
        )

    return _build_family_records(
        family="description_only",
        records=records,
        summary_path=SUMMARY_PATH,
        dataset_path=DATASET_PATH,
    )


def build_name_only_records() -> tuple[list[dict[str, object]], dict[str, object]]:
    repair_rows = _load_jsonl(REPAIR_PATH)
    control_rows = _load_jsonl(CONTROL_PATH)
    name_rows = [row for row in repair_rows if str(row.get("variant_family")) == "name_only"]
    generic_rows = [row for row in control_rows if str(row.get("control_type")) == "generic_ethics_control"]

    generic_by_group = {str(row["group_id"]): row for row in generic_rows}
    capture_groups = {str(row["group_id"]) for row in name_rows if bool(row.get("split"))}
    del capture_groups

    group_meta: dict[str, dict[str, str]] = {}
    for row in name_rows:
        group_meta.setdefault(
            str(row["group_id"]),
            {
                "source_family": str(row["source_family"]),
                "dilemma_type": str(row["dilemma_type"]),
                "context": str(row["context"]),
                "role_domain": str(row["role_domain"]),
            },
        )

    records: list[dict[str, object]] = []
    for row in name_rows:
        group_id = str(row["group_id"])
        meta = group_meta[group_id]
        cue_text = str(row["cue_text"])
        dilemma = _extract_dilemma(str(row["prompt"]))
        records.append(
            {
                "example_id": str(row["example_id"]),
                "group_id": group_id,
                "base_dilemma_id": group_id,
                "split": str(row["split"]),
                "capture_enabled": True,
                "capture_tier": "captured",
                "prime_family": "name_only",
                "prime_condition": THEORY_SHORT[str(row["theory"])],
                "theory_name": str(row["theory"]),
                "is_theory_prime": True,
                "is_generic_control": False,
                "description_bank": "",
                "alias_bank": "",
                "cue_text": cue_text,
                "prompt": _build_generation_prompt(cue_text=cue_text, dilemma=dilemma),
                **meta,
            }
        )

    for group_id, generic_row in sorted(generic_by_group.items()):
        meta = group_meta[group_id]
        generic_cue = str(generic_row["cue_text"])
        generic_dilemma = _extract_dilemma(str(generic_row["prompt"]))
        records.append(
            {
                "example_id": f"{group_id}__generic_ethics_control__name_only",
                "group_id": group_id,
                "base_dilemma_id": group_id,
                "split": _split_for_group(group_id),
                "capture_enabled": True,
                "capture_tier": "captured",
                "prime_family": "name_only",
                "prime_condition": GENERIC_CONTROL_LABEL,
                "theory_name": "",
                "is_theory_prime": False,
                "is_generic_control": True,
                "description_bank": "",
                "alias_bank": "",
                "cue_text": generic_cue,
                "prompt": _build_generation_prompt(cue_text=generic_cue, dilemma=generic_dilemma),
                **meta,
            }
        )

    records.sort(key=lambda record: (str(record["group_id"]), str(record["prime_condition"])))
    return _build_family_records(
        family="name_only",
        records=records,
        summary_path=NAME_ONLY_SUMMARY_PATH,
        dataset_path=NAME_ONLY_DATASET_PATH,
    )


def build_alias_only_records() -> tuple[list[dict[str, object]], dict[str, object]]:
    repair_rows = _load_jsonl(REPAIR_PATH)
    control_rows = _load_jsonl(CONTROL_PATH)
    alias_rows = [
        row
        for row in repair_rows
        if str(row.get("variant_family")) == "alias_only" and str(row.get("alias_bank")) in ALIAS_BANKS_SELECTED
    ]
    generic_rows = [row for row in control_rows if str(row.get("control_type")) == "generic_ethics_control"]

    generic_by_group = {str(row["group_id"]): row for row in generic_rows}
    group_meta: dict[str, dict[str, str]] = {}
    for row in alias_rows:
        group_meta.setdefault(
            str(row["group_id"]),
            {
                "source_family": str(row["source_family"]),
                "dilemma_type": str(row["dilemma_type"]),
                "context": str(row["context"]),
                "role_domain": str(row["role_domain"]),
            },
        )

    records: list[dict[str, object]] = []
    for row in alias_rows:
        group_id = str(row["group_id"])
        meta = group_meta[group_id]
        cue_text = str(row["cue_text"])
        dilemma = _extract_dilemma(str(row["prompt"]))
        records.append(
            {
                "example_id": str(row["example_id"]),
                "group_id": group_id,
                "base_dilemma_id": group_id,
                "split": str(row["split"]),
                "capture_enabled": True,
                "capture_tier": "captured",
                "prime_family": "alias_only",
                "prime_condition": THEORY_SHORT[str(row["theory"])],
                "theory_name": str(row["theory"]),
                "is_theory_prime": True,
                "is_generic_control": False,
                "description_bank": "",
                "alias_bank": str(row["alias_bank"]),
                "cue_text": cue_text,
                "prompt": _build_generation_prompt(cue_text=cue_text, dilemma=dilemma),
                **meta,
            }
        )

    for group_id, generic_row in sorted(generic_by_group.items()):
        meta = group_meta[group_id]
        generic_cue = str(generic_row["cue_text"])
        generic_dilemma = _extract_dilemma(str(generic_row["prompt"]))
        records.append(
            {
                "example_id": f"{group_id}__generic_ethics_control__alias_only",
                "group_id": group_id,
                "base_dilemma_id": group_id,
                "split": _split_for_group(group_id),
                "capture_enabled": True,
                "capture_tier": "captured",
                "prime_family": "alias_only",
                "prime_condition": GENERIC_CONTROL_LABEL,
                "theory_name": "",
                "is_theory_prime": False,
                "is_generic_control": True,
                "description_bank": "",
                "alias_bank": "",
                "cue_text": generic_cue,
                "prompt": _build_generation_prompt(cue_text=generic_cue, dilemma=generic_dilemma),
                **meta,
            }
        )

    records.sort(key=lambda record: (str(record["group_id"]), str(record["prime_condition"]), str(record["alias_bank"])))
    return _build_family_records(
        family="alias_only",
        records=records,
        summary_path=ALIAS_ONLY_SUMMARY_PATH,
        dataset_path=ALIAS_ONLY_DATASET_PATH,
    )


def write_outputs() -> None:
    records, summary = build_records()
    PHASE_03_OUTPUTS.mkdir(parents=True, exist_ok=True)

    with DATASET_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    name_records, name_summary = build_name_only_records()
    with NAME_ONLY_DATASET_PATH.open("w", encoding="utf-8") as handle:
        for record in name_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with NAME_ONLY_SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(name_summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    alias_records, alias_summary = build_alias_only_records()
    with ALIAS_ONLY_DATASET_PATH.open("w", encoding="utf-8") as handle:
        for record in alias_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with ALIAS_ONLY_SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(alias_summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    write_outputs()
