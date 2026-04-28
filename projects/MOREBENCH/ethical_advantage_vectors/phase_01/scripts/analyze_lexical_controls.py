"""Lexical controls for ethical-vs-self-advantage activation probes."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from projects.MOREBENCH.ethical_advantage_vectors.phase_01.scripts import (
    analyze_activation_probes as probes,
)


PHASE_ROOT = Path("projects/MOREBENCH/ethical_advantage_vectors/phase_01")
DEFAULT_GENERATION_ROWS = probes.DEFAULT_GENERATION_ROWS
DEFAULT_ACTION_ROWS = probes.DEFAULT_ACTION_ROWS
DEFAULT_REPORT_DIR = PHASE_ROOT / "reports" / "lexical_controls" / "v2_full40"


def _strip_generated(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _first_words(text: str, n: int = 16) -> str:
    return " ".join(_strip_generated(text).split()[:n])


def _render_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, (list, tuple)):
        parts: list[str] = []
        for msg in prompt:
            if isinstance(msg, Mapping):
                role = str(msg.get("role") or "")
                content = msg.get("content") or ""
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(msg))
        return "\n\n".join(parts)
    return str(prompt)


def _rows(generation_rows_path: Path, action_rows_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(generation_rows_path.read_text(encoding="utf-8"))
    action_by_key = probes._action_by_key(action_rows_path)
    out: list[dict[str, Any]] = []
    for row in payload["rows"]:
        example = row.get("example") or {}
        labels = example.get("labels") or {}
        key = str(row.get("example_key") or example.get("key") or "")
        action = action_by_key.get(key)
        out.append(
            {
                "key": key,
                "dilemma_id": str(labels.get("dilemma_id") or ""),
                "condition_id": str(labels.get("condition_id") or ""),
                "sample_index": int(labels.get("sample_index") or 0),
                "is_negative_prompt": str(labels.get("condition_id") or "") in probes.NEGATIVE_CONDITIONS,
                "is_ethical_prompt": str(labels.get("condition_id") or "") in probes.ETHICAL_CONDITIONS,
                "action_label": action,
                "prompt_text": _render_prompt(example.get("prompt")),
                "generated_text": _strip_generated(str(row.get("generated_text") or "")),
                "generated_prefix16": _first_words(str(row.get("generated_text") or ""), 16),
            }
        )
    return out


def _text_auc(
    rows: list[dict[str, Any]],
    *,
    text_field: str,
    target: str,
    negative_only: bool = False,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if negative_only and not row["is_negative_prompt"]:
            continue
        if target == "prompt_pole":
            if row["is_negative_prompt"] or row["is_ethical_prompt"]:
                selected.append(row)
        elif target == "observed_action":
            if row["action_label"] in {"self_advantage", "ethical"}:
                selected.append(row)
        else:
            raise ValueError(target)
    if not selected:
        return {"n": 0, "positive": 0, "auc_mean": float("nan"), "fold_aucs": []}
    texts = [str(row[text_field]) for row in selected]
    if target == "prompt_pole":
        y = np.asarray([1 if row["is_negative_prompt"] else 0 for row in selected], dtype=np.int64)
    else:
        y = np.asarray([1 if row["action_label"] == "self_advantage" else 0 for row in selected], dtype=np.int64)
    groups = np.asarray([row["dilemma_id"] for row in selected], dtype=object)
    if len(set(y.tolist())) < 2:
        return {"n": len(selected), "positive": int(y.sum()), "auc_mean": float("nan"), "fold_aucs": []}

    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    fold_aucs: list[float] = []
    for train_idx, test_idx in splitter.split(texts, y, groups):
        if len(set(y[train_idx].tolist())) < 2 or len(set(y[test_idx].tolist())) < 2:
            continue
        model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=20000),
            Ridge(alpha=10.0),
        )
        model.fit([texts[i] for i in train_idx], y[train_idx].astype(float))
        score = model.predict([texts[i] for i in test_idx])
        fold_aucs.append(float(roc_auc_score(y[test_idx], score)))
    return {
        "n": len(selected),
        "positive": int(y.sum()),
        "auc_mean": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
        "auc_std": float(np.std(fold_aucs)) if fold_aucs else float("nan"),
        "fold_aucs": fold_aucs,
    }


def _activation_table(report_summary_path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    summary = json.loads(report_summary_path.read_text(encoding="utf-8"))
    return {(row["target"], row["slice"], int(row["layer"])): row for row in summary["rows"]}


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def analyze(
    *,
    generation_rows_path: Path,
    action_rows_path: Path,
    activation_summary_path: Path,
    report_dir: Path,
) -> None:
    rows = _rows(generation_rows_path, action_rows_path)
    activation = _activation_table(activation_summary_path)
    text_rows: list[dict[str, Any]] = []
    for target in ("prompt_pole", "observed_action"):
        for text_field in ("prompt_text", "generated_prefix16", "generated_text"):
            text_rows.append(
                {
                    "target": target,
                    "scope": "all",
                    "text_field": text_field,
                    **_text_auc(rows, text_field=text_field, target=target),
                }
            )
    for text_field in ("prompt_text", "generated_prefix16", "generated_text"):
        text_rows.append(
            {
                "target": "observed_action",
                "scope": "negative_prompts_only",
                "text_field": text_field,
                **_text_auc(rows, text_field=text_field, target="observed_action", negative_only=True),
            }
        )

    summary = {
        "generation_rows_path": str(generation_rows_path),
        "action_rows_path": str(action_rows_path),
        "activation_summary_path": str(activation_summary_path),
        "text_rows": text_rows,
        "activation_reference": {
            "prompt_pole_prompt_end_L32": activation.get(("prompt_pole", "prompt_end", 32), {}),
            "prompt_pole_first16_L32": activation.get(("prompt_pole", "generated_first_16", 32), {}),
            "observed_action_first16_L32": activation.get(("observed_action", "generated_first_16", 32), {}),
            "observed_action_full_L32": activation.get(("observed_action", "generated_full", 32), {}),
            "observed_action_within_negative_first16_L32": activation.get(("observed_action_within_negative", "generated_first_16", 32), {}),
            "observed_action_within_negative_full_L32": activation.get(("observed_action_within_negative", "generated_full", 32), {},
            ),
        },
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Ethical Advantage Lexical Controls",
        "",
        "## Text Baselines",
        "",
        "| target | scope | text field | n | positives | text AUROC |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in text_rows:
        lines.append(
            f"| {row['target']} | {row['scope']} | {row['text_field']} | {row['n']} | {row['positive']} | {_fmt(row['auc_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Activation Reference",
            "",
            "| target/slice | activation AUROC | note |",
            "|---|---:|---|",
        ]
    )
    refs = summary["activation_reference"]
    ref_specs = [
        ("prompt_pole prompt_end L32", refs["prompt_pole_prompt_end_L32"], "Instruction/template separability."),
        ("prompt_pole first16 L32", refs["prompt_pole_first16_L32"], "Early generated prompt-conditioned state."),
        ("observed_action first16 L32", refs["observed_action_first16_L32"], "Action label across all conditions."),
        ("observed_action full L32", refs["observed_action_full_L32"], "Full response, lexically exposed."),
        (
            "observed_action within-negative first16 L32",
            refs["observed_action_within_negative_first16_L32"],
            "Harder: same negative-prompt family.",
        ),
        (
            "observed_action within-negative full L32",
            refs["observed_action_within_negative_full_L32"],
            "Harder but full text exposed.",
        ),
    ]
    for label, rec, note in ref_specs:
        lines.append(f"| {label} | {_fmt(float(rec.get('centroid_auc_mean', float('nan'))))} | {note} |")
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-rows", type=Path, default=DEFAULT_GENERATION_ROWS)
    parser.add_argument("--action-rows", type=Path, default=DEFAULT_ACTION_ROWS)
    parser.add_argument(
        "--activation-summary",
        type=Path,
        default=PHASE_ROOT / "reports" / "activation_probe_analysis" / "v2_full40_controls" / "summary.json",
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    analyze(
        generation_rows_path=args.generation_rows,
        action_rows_path=args.action_rows,
        activation_summary_path=args.activation_summary,
        report_dir=args.report_dir,
    )
    print(f"wrote {args.report_dir / 'report.md'}")


if __name__ == "__main__":
    main()
