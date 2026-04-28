"""Behavior-only smoke analysis for phase 03 natural prompt generations."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
DEFAULT_REPORT_ROOT = PHASE_ROOT / "reports" / "all_theories_natural_prompt_behavior_smoke"
DEFAULT_OUTPUT_DIR = PHASE_ROOT / "reports" / "all_theories_natural_prompt_behavior_analysis"

PRIMARY_CONDITIONS = (
    "N_neutral_01",
    "N_neutral_02",
    "N_generic_moral_01",
    "P_deont_01",
    "P_util_01",
    "P_virtue_01",
    "P_contract_01",
)


def _latest_generation_rows_path(report_root: Path) -> Path:
    candidates = sorted(
        report_root.glob("report_*/results/generate_natural_responses_results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no generate_natural_responses result found under {report_root}")
    return candidates[0]


def _rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"{path} must contain a rows list")
    return [r for r in rows if isinstance(r, Mapping)]


def _labels(row: Mapping[str, Any]) -> Mapping[str, Any]:
    example = row.get("example")
    if not isinstance(example, Mapping):
        return {}
    labels = example.get("labels")
    return labels if isinstance(labels, Mapping) else {}


def _token_count(row: Mapping[str, Any]) -> int:
    token_ids = row.get("generated_token_ids")
    if isinstance(token_ids, list):
        return len(token_ids)
    return len(str(row.get("generated_text") or "").split())


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z']+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    wa = _words(a)
    wb = _words(b)
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def _condition_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        condition = str(_labels(row).get("condition_id") or "")
        if condition:
            by_condition[condition].append(row)

    out: dict[str, Any] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        toks = np.asarray([_token_count(row) for row in condition_rows], dtype=np.float32)
        chars = np.asarray([len(str(row.get("generated_text") or "")) for row in condition_rows], dtype=np.float32)
        out[condition] = {
            "n": int(len(condition_rows)),
            "tokens_mean": float(toks.mean()),
            "tokens_median": float(np.median(toks)),
            "tokens_min": int(toks.min()),
            "tokens_max": int(toks.max()),
            "share_ge_20_tokens": float(np.mean(toks >= 20)),
            "chars_mean": float(chars.mean()),
            "chars_median": float(np.median(chars)),
        }
    return out


def _overall_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    toks = np.asarray([_token_count(row) for row in rows], dtype=np.float32)
    chars = np.asarray([len(str(row.get("generated_text") or "")) for row in rows], dtype=np.float32)
    return {
        "n": int(len(rows)),
        "tokens_mean": float(toks.mean()),
        "tokens_median": float(np.median(toks)),
        "tokens_p10": float(np.percentile(toks, 10)),
        "tokens_p90": float(np.percentile(toks, 90)),
        "share_lt_10_tokens": float(np.mean(toks < 10)),
        "share_lt_20_tokens": float(np.mean(toks < 20)),
        "chars_mean": float(chars.mean()),
        "chars_median": float(np.median(chars)),
    }


def _examples(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        condition = str(_labels(row).get("condition_id") or "")
        if condition in PRIMARY_CONDITIONS:
            by_condition[condition].append(row)

    out: dict[str, list[dict[str, str]]] = {}
    for condition in PRIMARY_CONDITIONS:
        picked = sorted(by_condition.get(condition, []), key=lambda r: _token_count(r), reverse=True)[:3]
        out[condition] = [
            {
                "dilemma_id": str(_labels(row).get("dilemma_id") or ""),
                "tokens": str(_token_count(row)),
                "text": str(row.get("generated_text") or "").strip(),
            }
            for row in picked
        ]
    return out


def _pair_divergence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index: dict[tuple[str, str], str] = {}
    for row in rows:
        labels = _labels(row)
        dilemma = str(labels.get("dilemma_id") or "")
        condition = str(labels.get("condition_id") or "")
        if dilemma and condition:
            index[(dilemma, condition)] = str(row.get("generated_text") or "")

    pairs = (
        ("P_deont_01", "N_neutral_01"),
        ("P_deont_01", "N_generic_moral_01"),
        ("P_deont_01", "P_util_01"),
        ("P_util_01", "N_neutral_01"),
        ("P_virtue_01", "N_neutral_01"),
        ("P_contract_01", "N_neutral_01"),
        ("N_generic_moral_01", "N_neutral_01"),
    )
    dilemmas = sorted({d for d, _ in index})
    out: list[dict[str, Any]] = []
    for a, b in pairs:
        vals: list[float] = []
        for dilemma in dilemmas:
            if (dilemma, a) in index and (dilemma, b) in index:
                vals.append(1.0 - _jaccard(index[(dilemma, a)], index[(dilemma, b)]))
        out.append(
            {
                "condition_a": a,
                "condition_b": b,
                "n_pairs": len(vals),
                "mean_text_divergence": float(mean(vals)) if vals else float("nan"),
                "median_text_divergence": float(np.median(vals)) if vals else float("nan"),
            }
        )
    return out


def _write_report(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Natural-Prompt Behavior Smoke")
    lines.append("")
    lines.append(f"- generation rows: `{summary['generation_rows_path']}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    overall = summary["overall"]
    lines.append(
        f"- n={overall['n']}, mean tokens={_fmt(overall['tokens_mean'])}, "
        f"median tokens={_fmt(overall['tokens_median'])}, "
        f"share <10={_fmt(overall['share_lt_10_tokens'])}, share <20={_fmt(overall['share_lt_20_tokens'])}"
    )
    lines.append("")
    lines.append("## By Condition")
    lines.append("")
    lines.append("| condition | n | mean tok | med tok | min | max | share >=20 | mean chars |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for condition, stats in summary["by_condition"].items():
        lines.append(
            f"| {condition} | {stats['n']} | {_fmt(stats['tokens_mean'])} | "
            f"{_fmt(stats['tokens_median'])} | {stats['tokens_min']} | {stats['tokens_max']} | "
            f"{_fmt(stats['share_ge_20_tokens'])} | {_fmt(stats['chars_mean'])} |"
        )
    lines.append("")
    lines.append("## Crude Text Divergence")
    lines.append("")
    lines.append("| A | B | n | mean 1-Jaccard | median 1-Jaccard |")
    lines.append("|---|---|---:|---:|---:|")
    for row in summary["pair_divergence"]:
        lines.append(
            f"| {row['condition_a']} | {row['condition_b']} | {row['n_pairs']} | "
            f"{_fmt(row['mean_text_divergence'])} | {_fmt(row['median_text_divergence'])} |"
        )
    lines.append("")
    lines.append("## Longest Examples")
    for condition, examples in summary["examples"].items():
        lines.append("")
        lines.append(f"### {condition}")
        lines.append("")
        for example in examples:
            lines.append(f"- `{example['dilemma_id']}` ({example['tokens']} tok): {example['text']}")
    lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-rows", default=None)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    generation_rows = Path(args.generation_rows) if args.generation_rows else _latest_generation_rows_path(Path(args.report_root))
    rows = _rows(generation_rows)
    summary = {
        "generation_rows_path": str(generation_rows),
        "overall": _overall_stats(rows),
        "by_condition": _condition_stats(rows),
        "pair_divergence": _pair_divergence(rows),
        "examples": _examples(rows),
    }
    _write_report(summary, Path(args.output_dir))
    print(f"wrote {Path(args.output_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
