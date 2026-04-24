from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from projects.MOREBENCH.phase_03.scripts.analyze_experiment_02_extended_metrics import _load_operation_artifact
from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as workflow


ROOT = Path(__file__).resolve().parents[4]
PHASE_02_EXAMPLES_PATH = ROOT / "projects" / "MOREBENCH" / "phase_02" / "outputs" / "theory_prompt_variant_sweep_examples.jsonl"
GENERATION_ARTIFACT_ID = "generation_run_1_3d4009fb21d8"
TARGET_PRIMES = ("deontology", "virtue_ethics")
TARGET_FAMILY = "description_only"
REPORT_DIR = ROOT / "projects" / "MOREBENCH" / "phase_03" / "reports" / "experiment_02_deont_vs_virtue_type_holdout"
REPORT_PATH = REPORT_DIR / "report.md"
SUMMARY_PATH = REPORT_DIR / "summary.json"
RANDOM_GROUP_HOLDOUT_TEXT_BASELINE_AUROC = 1.0
RESPONSE_MIN_PER_CLASS = 5
VIEWPORTS = ("full", "last_75", "last_25", "mid_50")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _binary_metrics(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, preds)), 4),
        "auroc": round(float(roc_auc_score(labels, probs)), 4),
        "positive_count": int(labels.sum()),
        "negative_count": int((1 - labels).sum()),
    }


def _label_value(prime_condition: str) -> int:
    if prime_condition == "deontology":
        return 1
    if prime_condition == "virtue_ethics":
        return 0
    raise ValueError(f"Unexpected prime_condition={prime_condition!r}")


def _fit_char_tfidf(train_texts: list[str], train_y: np.ndarray, test_texts: list[str], test_y: np.ndarray) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    clf = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, train_y)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)
    return _binary_metrics(test_y, probs, preds)


def _coverage_by(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, int]]:
    coverage: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        coverage[str(row[field])][str(row["prime_condition"])] += 1
    return {
        key: {prime: int(count) for prime, count in sorted(counter.items())}
        for key, counter in sorted(coverage.items())
    }


def _evaluate_holdouts(
    rows: list[dict[str, Any]],
    *,
    holdout_field: str,
    min_test_per_class: int | None,
) -> dict[str, Any]:
    holdout_values = sorted({str(row[holdout_field]) for row in rows})
    evaluated: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for holdout_value in holdout_values:
        train_rows = [row for row in rows if str(row[holdout_field]) != holdout_value]
        test_rows = [row for row in rows if str(row[holdout_field]) == holdout_value]
        train_counts = Counter(str(row["prime_condition"]) for row in train_rows)
        test_counts = Counter(str(row["prime_condition"]) for row in test_rows)

        train_min = min(int(train_counts.get(prime, 0)) for prime in TARGET_PRIMES)
        test_min = min(int(test_counts.get(prime, 0)) for prime in TARGET_PRIMES)
        if train_min == 0 or test_min == 0:
            skipped.append(
                {
                    "holdout_value": holdout_value,
                    "reason": "missing_class",
                    "train_counts": {prime: int(train_counts.get(prime, 0)) for prime in TARGET_PRIMES},
                    "test_counts": {prime: int(test_counts.get(prime, 0)) for prime in TARGET_PRIMES},
                }
            )
            continue
        if min_test_per_class is not None and test_min < min_test_per_class:
            skipped.append(
                {
                    "holdout_value": holdout_value,
                    "reason": f"test_class_count_below_{min_test_per_class}",
                    "train_counts": {prime: int(train_counts.get(prime, 0)) for prime in TARGET_PRIMES},
                    "test_counts": {prime: int(test_counts.get(prime, 0)) for prime in TARGET_PRIMES},
                }
            )
            continue

        train_texts = [str(row["text"]) for row in train_rows]
        test_texts = [str(row["text"]) for row in test_rows]
        train_y = np.asarray([_label_value(str(row["prime_condition"])) for row in train_rows], dtype=np.int32)
        test_y = np.asarray([_label_value(str(row["prime_condition"])) for row in test_rows], dtype=np.int32)
        metrics = _fit_char_tfidf(train_texts, train_y, test_texts, test_y)
        metrics.update(
            {
                "holdout_value": holdout_value,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "train_counts": {prime: int(train_counts.get(prime, 0)) for prime in TARGET_PRIMES},
                "test_counts": {prime: int(test_counts.get(prime, 0)) for prime in TARGET_PRIMES},
            }
        )
        evaluated.append(metrics)

    mean_auroc = None
    if evaluated:
        mean_auroc = round(float(sum(item["auroc"] for item in evaluated) / len(evaluated)), 4)

    return {
        "holdout_field": holdout_field,
        "model": "tfidf_char_wb_logreg",
        "vectorizer": {"analyzer": "char_wb", "ngram_range": [3, 5]},
        "min_test_per_class": min_test_per_class,
        "evaluated": evaluated,
        "skipped": skipped,
        "mean_auroc": mean_auroc,
    }


def _viewport_text(text: str, viewport: str) -> str:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return ""
    n = len(tokens)
    q1 = int(np.floor(0.25 * n))
    q3 = int(np.ceil(0.75 * n))
    if viewport == "full":
        chosen = tokens
    elif viewport == "last_75":
        chosen = tokens[q1:]
    elif viewport == "last_25":
        chosen = tokens[q3:]
    elif viewport == "mid_50":
        chosen = tokens[q1:q3]
    else:
        raise ValueError(f"Unexpected viewport={viewport!r}")
    if not chosen:
        chosen = tokens[-1:]
    return " ".join(chosen)


def _load_cue_rows() -> list[dict[str, Any]]:
    rows = _load_jsonl(PHASE_02_EXAMPLES_PATH)
    return [
        {
            "group_id": str(row["group_id"]),
            "prime_condition": str(row["prime_condition"]),
            "source_family": str(row["source_family"]),
            "context": str(row["context"]),
            "text": str(row["cue_text"]),
        }
        for row in rows
        if str(row.get("prime_condition")) in TARGET_PRIMES
    ]


def _load_generation_rows() -> list[dict[str, Any]]:
    artifact = _load_operation_artifact(GENERATION_ARTIFACT_ID)
    payload = artifact.result()
    raw_rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(raw_rows, list):
        raise TypeError(f"Generation artifact {GENERATION_ARTIFACT_ID!r} missing rows")

    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        example = row.get("example")
        if not isinstance(example, dict):
            continue
        labels = dict(example.get("labels", {}))
        prime_condition = str(labels.get("prime_condition") or "")
        if prime_condition not in TARGET_PRIMES:
            continue
        if str(labels.get("prime_family") or "") != TARGET_FAMILY:
            continue
        generated_text = str(row.get("generated_text") or row.get("text") or "")
        finish_reason = str(row.get("finish_reason") or "")
        source_prompt = workflow._render_prompt_text(example.get("prompt") or "")
        capture_enabled = bool(labels.get("capture_enabled"))
        if not capture_enabled:
            continue
        if finish_reason == "length":
            continue
        if not generated_text.strip() or not source_prompt.strip():
            continue

        theory_name = str(labels.get("theory_name") or "")
        is_theory_prime = bool(labels.get("is_theory_prime"))
        cue_text = str(labels.get("cue_text") or "")
        theory_name_metrics = (
            workflow._theory_name_copy_metrics(theory_name=theory_name, generated_text=generated_text)
            if is_theory_prime and theory_name
            else {"theory_name_mention_count": 0, "repeated_theory_name_copy": False}
        )
        cue_overlap = False
        cue_overlap_fraction = 0.0
        cue_longest_run = 0
        if is_theory_prime:
            cue_overlap, cue_metrics = workflow._near_verbatim_cue_copy(
                cue_text=cue_text,
                generated_text=generated_text,
            )
            cue_overlap_fraction = float(cue_metrics["cue_overlap_fraction"])
            cue_longest_run = int(cue_metrics["cue_longest_run"])
        direct_copy = bool(theory_name_metrics["repeated_theory_name_copy"]) or cue_overlap

        rows.append(
            {
                "group_id": str(labels.get("group_id") or ""),
                "prime_condition": prime_condition,
                "source_family": str(labels.get("source_family") or ""),
                "context": str(labels.get("context") or ""),
                "full_text": generated_text,
                "direct_copy": direct_copy,
                "cue_overlap_fraction": round(cue_overlap_fraction, 4),
                "cue_longest_run": cue_longest_run,
                "theory_name_copy": bool(theory_name_metrics["repeated_theory_name_copy"]),
            }
        )
    return rows


def _response_variant_rows(raw_rows: list[dict[str, Any]], *, filter_mode: str, viewport: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if filter_mode == "filter_on" and bool(row["direct_copy"]):
            continue
        rows.append(
            {
                "group_id": str(row["group_id"]),
                "prime_condition": str(row["prime_condition"]),
                "source_family": str(row["source_family"]),
                "context": str(row["context"]),
                "text": _viewport_text(str(row["full_text"]), viewport),
            }
        )
    return rows


def _table_lines(title: str, results: dict[str, Any]) -> list[str]:
    lines = [f"### {title}", ""]
    lines.append("| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |")
    lines.append("| --- | ---: | ---: | --- | --- | ---: | ---: |")
    for item in results["evaluated"]:
        train_counts = f"d={item['train_counts']['deontology']}, v={item['train_counts']['virtue_ethics']}"
        test_counts = f"d={item['test_counts']['deontology']}, v={item['test_counts']['virtue_ethics']}"
        lines.append(
            f"| `{item['holdout_value']}` | `{item['train_count']}` | `{item['test_count']}` | "
            f"`{train_counts}` | `{test_counts}` | `{item['auroc']}` | `{item['balanced_accuracy']}` |"
        )
    if not results["evaluated"]:
        lines.append("| _none_ |  |  |  |  |  |  |")
    lines.append("")
    if results["skipped"]:
        lines.append("Skipped holdouts:")
        for item in results["skipped"]:
            lines.append(
                f"- `{item['holdout_value']}`: `{item['reason']}` "
                f"(train d/v `{item['train_counts']['deontology']}/{item['train_counts']['virtue_ethics']}`, "
                f"test d/v `{item['test_counts']['deontology']}/{item['test_counts']['virtue_ethics']}`)"
            )
        lines.append("")
    lines.append(f"- mean AUROC across evaluated holdouts: `{results['mean_auroc']}`")
    lines.append("")
    return lines


def _response_decision_flags(response_results: dict[str, Any]) -> dict[str, bool]:
    evaluated = [
        item
        for filter_bucket in response_results.values()
        for viewport_bucket in filter_bucket.values()
        for axis_bucket in viewport_bucket.values()
        for item in axis_bucket["evaluated"]
    ]
    any_le_085 = any(item["auroc"] <= 0.85 for item in evaluated)
    all_ge_095 = all(item["auroc"] >= 0.95 for item in evaluated) if evaluated else False
    return {
        "any_response_cell_le_085": any_le_085,
        "all_response_cells_ge_095": all_ge_095,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cue_rows = _load_cue_rows()
    raw_response_rows = _load_generation_rows()

    cue_source = _evaluate_holdouts(cue_rows, holdout_field="source_family", min_test_per_class=None)
    cue_context = _evaluate_holdouts(cue_rows, holdout_field="context", min_test_per_class=None)

    response_results: dict[str, dict[str, dict[str, Any]]] = {}
    response_coverage: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    response_row_counts: dict[str, dict[str, int]] = {}
    for filter_mode in ("filter_on", "filter_off"):
        response_results[filter_mode] = {}
        response_coverage[filter_mode] = {}
        response_row_counts[filter_mode] = {}
        for viewport in VIEWPORTS:
            viewport_rows = _response_variant_rows(raw_response_rows, filter_mode=filter_mode, viewport=viewport)
            response_row_counts[filter_mode][viewport] = len(viewport_rows)
            response_coverage[filter_mode][viewport] = {
                "source_family_prime_counts": _coverage_by(viewport_rows, "source_family"),
                "context_prime_counts": _coverage_by(viewport_rows, "context"),
            }
            response_results[filter_mode][viewport] = {
                "source_family_loo": _evaluate_holdouts(
                    viewport_rows,
                    holdout_field="source_family",
                    min_test_per_class=RESPONSE_MIN_PER_CLASS,
                ),
                "context_loo": _evaluate_holdouts(
                    viewport_rows,
                    holdout_field="context",
                    min_test_per_class=RESPONSE_MIN_PER_CLASS,
                ),
            }

    any_cue_below_085 = any(
        item["auroc"] <= 0.85
        for bucket in (cue_source["evaluated"], cue_context["evaluated"])
        for item in bucket
    )
    all_cue_ge_095 = all(
        item["auroc"] >= 0.95
        for bucket in (cue_source["evaluated"], cue_context["evaluated"])
        for item in bucket
    ) if (cue_source["evaluated"] or cue_context["evaluated"]) else False
    response_flags = _response_decision_flags(response_results)

    summary = {
        "analysis": "experiment_02_deont_vs_virtue_type_holdout",
        "target_primes": list(TARGET_PRIMES),
        "random_group_holdout_text_baseline_auroc": RANDOM_GROUP_HOLDOUT_TEXT_BASELINE_AUROC,
        "cue_rows": len(cue_rows),
        "raw_response_rows": len(raw_response_rows),
        "cue_source_family_prime_counts": _coverage_by(cue_rows, "source_family"),
        "cue_context_prime_counts": _coverage_by(cue_rows, "context"),
        "cue_diagnostic": {
            "source_family_loo": cue_source,
            "context_loo": cue_context,
        },
        "response_diagnostic": {
            "source_generation_artifact_id": GENERATION_ARTIFACT_ID,
            "filter_modes": response_results,
            "response_row_counts": response_row_counts,
            "coverage": response_coverage,
            "response_min_test_per_class": RESPONSE_MIN_PER_CLASS,
        },
        "decision_rule": {
            "response_warranted_if_any_cue_holdout_le_085": any_cue_below_085,
            "stop_after_step1_if_all_cue_holdouts_ge_095": all_cue_ge_095,
            **response_flags,
            "note": "Step 2 was executed regardless because the user explicitly requested both steps.",
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# Experiment 02 Deontology vs Virtue Type Holdout\n")
    lines.append(
        "Type-stratified text diagnostic on `deontology` vs `virtue_ethics`, comparing source-family and context holdouts "
        "against the earlier random dilemma-group holdout text baseline of `1.00` AUROC.\n\n"
    )
    lines.append("## Coverage\n")
    lines.append(f"- cue rows: `{len(cue_rows)}`\n")
    lines.append(f"- raw response rows from generation artifact `{GENERATION_ARTIFACT_ID}`: `{len(raw_response_rows)}`\n")
    lines.append(f"- response minimum test count per class: `{RESPONSE_MIN_PER_CLASS}`\n")
    lines.append("")
    lines.append("### Cue Source-Family x Prime\n")
    lines.append("| Source family | Deontology | Virtue |")
    lines.append("| --- | ---: | ---: |")
    for key, counts in summary["cue_source_family_prime_counts"].items():
        lines.append(f"| `{key}` | `{counts.get('deontology', 0)}` | `{counts.get('virtue_ethics', 0)}` |")
    lines.append("")
    lines.append("### Cue Context x Prime\n")
    lines.append("| Context | Deontology | Virtue |")
    lines.append("| --- | ---: | ---: |")
    for key, counts in summary["cue_context_prime_counts"].items():
        lines.append(f"| `{key}` | `{counts.get('deontology', 0)}` | `{counts.get('virtue_ethics', 0)}` |")
    lines.append("")
    lines.extend(_table_lines("Cue Source-Family Holdout", cue_source))
    lines.extend(_table_lines("Cue Context Holdout", cue_context))

    for filter_mode in ("filter_on", "filter_off"):
        lines.append(f"## Response Diagnostic: `{filter_mode}`\n")
        for viewport in VIEWPORTS:
            lines.append(f"- viewport `{viewport}` row count: `{response_row_counts[filter_mode][viewport]}`")
        lines.append("")
        for viewport in VIEWPORTS:
            coverage = response_coverage[filter_mode][viewport]
            lines.append(f"### `{filter_mode}` / `{viewport}` Source-Family x Prime\n")
            lines.append("| Source family | Deontology | Virtue |")
            lines.append("| --- | ---: | ---: |")
            for key, counts in coverage["source_family_prime_counts"].items():
                lines.append(f"| `{key}` | `{counts.get('deontology', 0)}` | `{counts.get('virtue_ethics', 0)}` |")
            lines.append("")
            lines.append(f"### `{filter_mode}` / `{viewport}` Context x Prime\n")
            lines.append("| Context | Deontology | Virtue |")
            lines.append("| --- | ---: | ---: |")
            for key, counts in coverage["context_prime_counts"].items():
                lines.append(f"| `{key}` | `{counts.get('deontology', 0)}` | `{counts.get('virtue_ethics', 0)}` |")
            lines.append("")
            lines.extend(_table_lines(f"`{filter_mode}` / `{viewport}` Response Source-Family Holdout", response_results[filter_mode][viewport]["source_family_loo"]))
            lines.extend(_table_lines(f"`{filter_mode}` / `{viewport}` Response Context Holdout", response_results[filter_mode][viewport]["context_loo"]))

    lines.append("## Interpretation\n")
    lines.append(
        f"The comparison point is the earlier random dilemma-group holdout text baseline of `{RANDOM_GROUP_HOLDOUT_TEXT_BASELINE_AUROC}` AUROC. "
        f"Cue-side type holdouts {'do' if any_cue_below_085 else 'do not'} produce any AUROC at or below `0.85`. "
        f"Across response-side diagnostics, any viewport/holdout/filter cell at or below `0.85` is `{response_flags['any_response_cell_le_085']}`. "
        f"If every evaluated response cell stays at or above `0.95` (`{response_flags['all_response_cells_ge_095']}`), then neither dropping the strict copy filter nor shifting to later windows meaningfully lowers the deontology-vs-virtue text ceiling on the existing response data."
    )
    lines.append("")
    lines.append("## Decision Surface\n")
    lines.append(f"- any cue holdout <= `0.85`: `{any_cue_below_085}`")
    lines.append(f"- all cue holdouts >= `0.95`: `{all_cue_ge_095}`")
    lines.append(f"- any evaluated response cell <= `0.85`: `{response_flags['any_response_cell_le_085']}`")
    lines.append(f"- all evaluated response cells >= `0.95`: `{response_flags['all_response_cells_ge_095']}`")
    lines.append("- per user request, step 2 was executed regardless of the cue result.")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
        "any_cue_below_085": any_cue_below_085,
        "all_cue_ge_095": all_cue_ge_095,
        **response_flags,
    }, indent=2))


if __name__ == "__main__":
    main()
