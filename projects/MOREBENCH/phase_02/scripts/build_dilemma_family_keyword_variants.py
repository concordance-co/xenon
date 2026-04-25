#!/usr/bin/env python3
"""Apply variant keyword banks to the 500 dilemmas and materialize per-row labels.

Reads the spec at phase_02/specs/dilemma_family_keyword_variants.json, applies
each variant's regex patterns to the dilemma text of each of the 500 public
rows, and writes fully-instantiated binary labels to
phase_02/outputs/dilemma_family_keyword_variant_labels.jsonl.

Output has one row per dilemma with columns:
  row_id, source_family, context, role_domain, dilemma_type,
  for each label: <label>__variant_a, <label>__variant_b, <label>__variant_c
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "projects/MOREBENCH/phase_02/specs/dilemma_family_keyword_variants.json"
GENERATION_RESULT = ROOT / "artifacts/_modal_cache/generation_run_1_d6e12a467208/result.json"
OUTPUT = ROOT / "projects/MOREBENCH/phase_02/outputs/dilemma_family_keyword_variant_labels.jsonl"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(
            str(item.get("content", item)) if isinstance(item, dict) else str(item)
            for item in value
        )
    return str(value)


def pattern_flag(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def compile_patterns(raw: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pat, re.IGNORECASE) for pat in raw]


def load_dilemmas() -> list[dict[str, Any]]:
    generation = json.loads(GENERATION_RESULT.read_text())
    rows = []
    for item in generation["rows"]:
        ex = item["example"]
        row_id = ex["key"]
        labels = dict(ex.get("labels", {}))
        metadata = dict(ex.get("metadata", {}))
        dilemma_text = normalize_text(metadata.get("dilemma_text", ""))
        if not dilemma_text:
            prompt_text = normalize_text(ex.get("prompt", ""))
            if "DILEMMA:" in prompt_text:
                dilemma_text = prompt_text.split("DILEMMA:", 1)[1].strip()
            else:
                dilemma_text = prompt_text
        rows.append(
            {
                "row_id": row_id,
                "source_family": labels.get("source_family"),
                "context": labels.get("context"),
                "role_domain": labels.get("role_domain"),
                "dilemma_type": labels.get("dilemma_type"),
                "dilemma_text": dilemma_text,
            }
        )
    return rows


def main() -> None:
    spec = json.loads(SPEC.read_text())
    labels_spec = spec["labels"]

    compiled: dict[str, dict[str, list[re.Pattern[str]]]] = {}
    for label, block in labels_spec.items():
        compiled[label] = {
            variant_key: compile_patterns(patterns)
            for variant_key, patterns in block.items()
            if variant_key.startswith("variant_")
        }

    dilemmas = load_dilemmas()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    coverage_counts: dict[str, dict[str, int]] = {
        label: {variant: 0 for variant in compiled[label]} for label in compiled
    }
    total = 0
    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in dilemmas:
            out: dict[str, Any] = {
                "row_id": row["row_id"],
                "source_family": row["source_family"],
                "context": row["context"],
                "role_domain": row["role_domain"],
                "dilemma_type": row["dilemma_type"],
            }
            for label, variants in compiled.items():
                for variant_key, patterns in variants.items():
                    flag = pattern_flag(row["dilemma_text"].lower(), patterns)
                    out[f"{label}__{variant_key}"] = bool(flag)
                    if flag:
                        coverage_counts[label][variant_key] += 1
            f.write(json.dumps(out) + "\n")
            total += 1

    print(f"wrote {total} rows to {OUTPUT.relative_to(ROOT)}")
    print("True counts per (label, variant):")
    for label, variants in coverage_counts.items():
        print(f"  {label}:")
        for variant, count in variants.items():
            pct = 100.0 * count / total
            print(f"    {variant}: {count}/{total} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
