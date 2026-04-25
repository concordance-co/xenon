#!/usr/bin/env python3
"""Lexical gate for benchmark-native MoReBench prompt/response labels."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = ROOT / "projects/MOREBENCH/phase_03/reports/experiment_03_label_lexical_gate"
GENERATION_RESULT = ROOT / "artifacts/_modal_cache/generation_run_1_d6e12a467208/result.json"
LABEL_ROWS = ROOT / "projects/MOREBENCH/phase_03/reports/experiment_03_full_public_generation_labelpilot/label_rows.jsonl"
RUBRIC_SCORES = ROOT / "projects/MOREBENCH/phase_03/reports/experiment_03_full_public_rubric_judge/manual_scores_merged.jsonl"

MIN_BINARY_CLASS = 50
MIN_MULTICLASS_CLASS = 20
MIN_HOLDOUT_TEST_N = 25
RANDOM_STATE = 7

# Per-label overrides that lower the minority-class floor. Used for
# scientifically interesting small-N labels like helpful_harmless_off_diagonal
# (minority class N~33), which the default floor would silently drop.
SMALL_N_BINARY_OVERRIDES: dict[str, int] = {
    "helpful_harmless_off_diagonal": 25,
}


FAMILY_PATTERNS = {
    "close_relationship_obligation": [
        r"\bfamily\b",
        r"\bfriend(s)?\b",
        r"\bpartner\b",
        r"\bspouse\b",
        r"\bparent(s)?\b",
        r"\bmother\b",
        r"\bfather\b",
        r"\bsibling\b",
        r"\bchild(ren)?\b",
        r"\bdaughter\b",
        r"\bson\b",
        r"\broommate\b",
        r"\brelative\b",
    ],
    "institutional_policy_constraint": [
        r"\bpolicy\b",
        r"\bpolicies\b",
        r"\blaw\b",
        r"\blegal\b",
        r"\bcontract\b",
        r"\bregulation(s)?\b",
        r"\brule(s)?\b",
        r"\bboard\b",
        r"\bcommittee\b",
        r"\bcommission\b",
        r"\binstitution\b",
        r"\bcompany\b",
        r"\bemployer\b",
        r"\buniversity\b",
        r"\bschool\b",
        r"\bagency\b",
        r"\bprotocol\b",
        r"\bprocedure\b",
    ],
    "privacy_monitoring_conflict": [
        r"\bprivacy\b",
        r"\bprivate\b",
        r"\bconfidential\b",
        r"\bconsent\b",
        r"\bsurveillance\b",
        r"\bmonitor(ing)?\b",
        r"\btracking\b",
        r"\brecord(s)?\b",
        r"\bdata\b",
        r"\bgenetic\b",
        r"\bcamera(s)?\b",
        r"\bpersonal information\b",
    ],
    "public_harm_resource_tradeoff": [
        r"\bpublic\b",
        r"\bcommunity\b",
        r"\bemergency\b",
        r"\bhospital\b",
        r"\bdisaster\b",
        r"\binfrastructure\b",
        r"\bbudget\b",
        r"\bresource(s)?\b",
        r"\ballocat(e|ion|ing)\b",
        r"\btriage\b",
        r"\brisk to\b",
        r"\bpublic safety\b",
        r"\bharm to others\b",
    ],
    "disclosure_transparency_conflict": [
        r"\bdisclose\b",
        r"\bdisclosure\b",
        r"\breveal\b",
        r"\breport\b",
        r"\bleak\b",
        r"\btransparency\b",
        r"\bhonest(y)?\b",
        r"\btell\b",
        r"\binform\b",
        r"\bconceal\b",
        r"\bwithhold\b",
        r"\bpublicly\b",
    ],
    "fairness_access_conflict": [
        r"\bfair(ness)?\b",
        r"\bbias(ed)?\b",
        r"\bdiscrimination\b",
        r"\binequal(ity)?\b",
        r"\bequity\b",
        r"\binclusion\b",
        r"\baccess\b",
        r"\badvantage\b",
        r"\bdisparit(y|ies)\b",
        r"\bmarginalized\b",
        r"\bunderrepresented\b",
    ],
    "authority_constraint": [
        r"\bauthority\b",
        r"\bmanager\b",
        r"\bsupervisor\b",
        r"\bleader\b",
        r"\bofficial\b",
        r"\bexpert\b",
        r"\bdoctor\b",
        r"\btherapist\b",
        r"\bteacher\b",
        r"\bprofessor\b",
        r"\bpolice\b",
        r"\bcourt\b",
    ],
    "loyalty_relationship": [
        r"\bloyal(ty)?\b",
        r"\bobligation\b",
        r"\bduty\b",
        r"\btrust\b",
        r"\bpromise\b",
        r"\bowe\b",
        r"\bfamily\b",
        r"\bfriend(s)?\b",
        r"\bpartner\b",
    ],
    "public_safety": [
        r"\bsafety\b",
        r"\bsafe\b",
        r"\bdanger\b",
        r"\brisk\b",
        r"\bharm\b",
        r"\binjury\b",
        r"\bdeath\b",
        r"\bpublic\b",
        r"\bemergency\b",
        r"\bsecurity\b",
    ],
    "autonomy_boundary": [
        r"\bautonomy\b",
        r"\bconsent\b",
        r"\bchoice\b",
        r"\bpersonal\b",
        r"\bfreedom\b",
        r"\bindependent\b",
        r"\bagency\b",
        r"\bright to\b",
        r"\bvalues\b",
        r"\bbeliefs\b",
    ],
    "uncertainty_incomplete_info": [
        r"\buncertain\b",
        r"\buncertainty\b",
        r"\bunknown\b",
        r"\bno guarantee\b",
        r"\bincomplete\b",
        r"\bambiguous\b",
        r"\bunclear\b",
        r"\bpossibly\b",
        # "may/might/could" removed — too common as neutral English modals;
        # they flagged nearly every dilemma and produced a near-constant label.
    ],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item.get("content", item)) if isinstance(item, dict) else str(item) for item in value)
    return str(value)


def final_window(text: str, fraction: float = 0.25) -> str:
    words = text.split()
    if not words:
        return ""
    n = max(80, math.ceil(len(words) * fraction))
    return " ".join(words[-n:])


def pattern_flag(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def load_rows() -> list[dict[str, Any]]:
    generation = json.loads(GENERATION_RESULT.read_text())
    label_by_id = {row["row_id"]: row for row in read_jsonl(LABEL_ROWS)}
    rubric_by_id = {row["row_id"]: row for row in read_jsonl(RUBRIC_SCORES)}

    gen_ids = {item["example"]["key"] for item in generation["rows"]}
    missing_labels = gen_ids - set(label_by_id)
    missing_rubric = gen_ids - set(rubric_by_id)
    print(
        f"[load_rows] generation={len(gen_ids)} labels={len(label_by_id)} rubric={len(rubric_by_id)} "
        f"missing_labels={len(missing_labels)} missing_rubric={len(missing_rubric)}"
    )
    if missing_labels:
        print(f"[load_rows] WARN missing label rows (first 5): {sorted(missing_labels)[:5]}")
    if missing_rubric:
        print(f"[load_rows] WARN missing rubric rows (first 5): {sorted(missing_rubric)[:5]}")

    rows: list[dict[str, Any]] = []
    for item in generation["rows"]:
        ex = item["example"]
        row_id = ex["key"]
        labels = dict(ex.get("labels", {}))
        metadata = dict(ex.get("metadata", {}))
        generated_text = normalize_text(item.get("generated_text", ""))
        prompt_text = normalize_text(ex.get("prompt", ""))
        dilemma_text = normalize_text(metadata.get("dilemma_text", ""))
        label_row = label_by_id.get(row_id, {})
        rubric = rubric_by_id.get(row_id, {})

        row = {
            "row_id": row_id,
            "source_family": labels.get("source_family") or label_row.get("source_family"),
            "context": labels.get("context") or label_row.get("context"),
            "role_domain": labels.get("role_domain") or label_row.get("role_domain"),
            "dilemma_type": labels.get("dilemma_type") or label_row.get("dilemma_type"),
            "prompt_text": prompt_text,
            "dilemma_text": dilemma_text,
            "response_text": generated_text,
            "final_span": label_row.get("final_span") or final_window(generated_text),
        }
        row.update({f"heuristic_{k}": v for k, v in label_row.items() if k not in row})
        row.update({f"rubric_{k}": v for k, v in rubric.items() if k != "row_id"})

        helpful_score = rubric.get("helpful_score")
        harmless_score = rubric.get("harmless_score")
        if helpful_score is not None:
            row["strong_helpful"] = int(helpful_score) == 3
            row["helpful_binary"] = int(helpful_score) >= 2
            row["helpful_low_or_mid"] = int(helpful_score) < 3
        if harmless_score is not None:
            row["strong_harmless"] = int(harmless_score) == 3
            row["harmless_binary"] = int(harmless_score) >= 2
            row["harmless_low_or_mid"] = int(harmless_score) < 3
        if helpful_score is not None and harmless_score is not None:
            h = int(helpful_score)
            s = int(harmless_score)
            if h == 3 and s < 3:
                row["helpful_harmless_off_diagonal"] = "helpful_over_harmless"
            elif h < 3 and s == 3:
                row["helpful_harmless_off_diagonal"] = "harmless_over_helpful"
            else:
                row["helpful_harmless_off_diagonal"] = None
            row["helpful_harmless_cell"] = f"h{h}_s{s}"

        for label, patterns in FAMILY_PATTERNS.items():
            row[label] = pattern_flag(dilemma_text, patterns)

        rows.append(row)
    return rows


def class_counts(values: list[Any]) -> Counter:
    return Counter(v for v in values if v is not None and v != "")


def is_supported(counts: Counter, label: str | None = None) -> bool:
    if len(counts) == 2:
        floor = SMALL_N_BINARY_OVERRIDES.get(label or "", MIN_BINARY_CLASS)
        return min(counts.values()) >= floor
    if len(counts) > 2:
        return min(counts.values()) >= MIN_MULTICLASS_CLASS
    return False


def make_text_pipeline(feature: str):
    if feature == "char":
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=12000)
    elif feature == "word":
        vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, lowercase=True, max_features=12000
        )
    else:
        raise ValueError(feature)
    return make_pipeline(
        vectorizer,
        LinearSVC(class_weight="balanced", dual="auto", random_state=RANDOM_STATE),
    )


def make_length_features(texts: list[str]) -> np.ndarray:
    return np.array([[len(text), len(text.split())] for text in texts], dtype=float)


def fit_predict(
    texts: list[str],
    y: np.ndarray,
    feature: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    binary: bool,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.empty_like(y)
    scores = np.zeros((len(y), 2 if binary else len(set(y))), dtype=float)
    classes_all = np.array(sorted(set(y)))
    for train_idx, test_idx in splits:
        if len(set(y[train_idx])) < 2:
            raise ValueError("train split has fewer than two classes")
        if feature == "length":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced"),
            )
            x_train = make_length_features([texts[i] for i in train_idx])
            x_test = make_length_features([texts[i] for i in test_idx])
        else:
            clf = make_text_pipeline(feature)
            x_train = [texts[i] for i in train_idx]
            x_test = [texts[i] for i in test_idx]
        clf.fit(x_train, y[train_idx])
        pred[test_idx] = clf.predict(x_test)
        local_scores = decision_scores(clf, x_test)
        local_classes = clf.classes_
        for col, cls in enumerate(local_classes):
            dest = int(np.where(classes_all == cls)[0][0])
            scores[test_idx, dest] = local_scores[:, col]
    return pred, scores


def decision_scores(clf: Any, x_test: Any) -> np.ndarray:
    if hasattr(clf, "decision_function"):
        raw = clf.decision_function(x_test)
        if raw.ndim == 1:
            return np.column_stack([-raw, raw])
        return raw
    proba = clf.predict_proba(x_test)
    if proba.shape[1] == 2:
        return np.column_stack([-proba[:, 1], proba[:, 1]])
    return proba


def score_predictions(y: np.ndarray, pred: np.ndarray, scores: np.ndarray, binary: bool) -> dict[str, float | None]:
    result: dict[str, float | None] = {"balanced_accuracy": float(balanced_accuracy_score(y, pred))}
    try:
        if binary:
            result["auroc"] = float(roc_auc_score(y, scores[:, 1]))
        else:
            result["auroc"] = float(roc_auc_score(y, scores, multi_class="ovr", average="weighted"))
    except ValueError:
        result["auroc"] = None
    return result


def stratified_splits(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    counts = Counter(y)
    n_splits = min(5, min(counts.values()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return [(train, test) for train, test in cv.split(np.zeros(len(y)), y)]


def group_holdout_splits(groups: list[str], y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, str]]:
    splits = []
    arr_groups = np.array(groups)
    for group in sorted(set(groups)):
        test_idx = np.where(arr_groups == group)[0]
        train_idx = np.where(arr_groups != group)[0]
        if len(test_idx) < MIN_HOLDOUT_TEST_N:
            continue
        if len(set(y[test_idx])) < 2 or len(set(y[train_idx])) < 2:
            continue
        if not set(y[test_idx]).issubset(set(y[train_idx])):
            continue
        splits.append((train_idx, test_idx, group))
    return splits


def evaluate_one(
    rows: list[dict[str, Any]],
    label: str,
    text_field: str,
    feature: str,
    split_kind: str,
) -> dict[str, Any] | None:
    filtered = [row for row in rows if row.get(label) is not None and row.get(label) != ""]
    values = [row[label] for row in filtered]
    counts = class_counts(values)
    if not is_supported(counts, label=label):
        return None
    encoder = LabelEncoder()
    y = encoder.fit_transform([str(v) for v in values])
    binary = len(encoder.classes_) == 2
    texts = [normalize_text(row.get(text_field, "")) for row in filtered]

    try:
        if split_kind == "cv":
            splits = stratified_splits(y)
            pred, scores = fit_predict(texts, y, feature, splits, binary)
            metrics = score_predictions(y, pred, scores, binary)
            return {
                "split": "cv",
                "n_splits": len(splits),
                **metrics,
            }
        group_name = split_kind.removesuffix("_holdout")
        splits_with_names = group_holdout_splits([str(row.get(group_name, "")) for row in filtered], y)
        if not splits_with_names:
            return None
        fold_metrics = []
        for train_idx, test_idx, group in splits_with_names:
            pred, scores = fit_predict(texts, y, feature, [(train_idx, test_idx)], binary)
            fold_y = y[test_idx]
            fold_pred = pred[test_idx]
            fold_scores = scores[test_idx]
            metrics = score_predictions(fold_y, fold_pred, fold_scores, binary)
            metrics["heldout"] = group
            metrics["n_test"] = int(len(test_idx))
            fold_metrics.append(metrics)
        return {
            "split": split_kind,
            "n_splits": len(fold_metrics),
            "balanced_accuracy": float(mean(m["balanced_accuracy"] for m in fold_metrics)),
            "auroc": float(mean(m["auroc"] for m in fold_metrics if m["auroc"] is not None))
            if any(m["auroc"] is not None for m in fold_metrics)
            else None,
            "folds": fold_metrics,
        }
    except Exception as exc:
        return {"split": split_kind, "error": str(exc)}


def evaluation_plan(label_type: str) -> list[tuple[str, str, str]]:
    """Focused gate: primary text surface, final-span check, and source-family generalization."""
    if label_type == "prompt":
        return [
            ("prompt_text", "char", "cv"),
            ("prompt_text", "char", "source_family_holdout"),
            ("prompt_text", "length", "source_family_holdout"),
        ]
    return [
        ("response_text", "char", "cv"),
        ("response_text", "char", "source_family_holdout"),
        ("response_text", "length", "source_family_holdout"),
        ("final_span", "char", "source_family_holdout"),
        ("prompt_text", "char", "source_family_holdout"),
    ]


def evaluate_label(rows: list[dict[str, Any]], label: str, label_type: str) -> dict[str, Any]:
    filtered_values = [row.get(label) for row in rows if row.get(label) is not None and row.get(label) != ""]
    counts = class_counts(filtered_values)
    result = {
        "label": label,
        "label_type": label_type,
        "counts": dict(counts),
        "n": int(sum(counts.values())),
        "supported": is_supported(counts, label=label),
        "evaluations": {},
    }
    if not result["supported"]:
        return result

    for text_field, feature, split_kind in evaluation_plan(label_type):
        key = f"{text_field}:{feature}:{split_kind}"
        value = evaluate_one(rows, label, text_field, feature, split_kind)
        if value is not None:
            result["evaluations"][key] = value
    return result


def primary_metric(record: dict[str, Any], key: str) -> float | None:
    value = record["evaluations"].get(key, {})
    metric = value.get("balanced_accuracy")
    return float(metric) if metric is not None else None


def triage(record: dict[str, Any]) -> str:
    if not record["supported"]:
        return "unsupported"
    if record["label_type"] == "prompt":
        key = "prompt_text:char:source_family_holdout"
    else:
        key = "response_text:char:source_family_holdout"
    holdout = primary_metric(record, key)
    cv_key = key.replace("source_family_holdout", "cv")
    cv = primary_metric(record, cv_key)
    if holdout is None:
        return "needs_split_review"
    if holdout >= 0.75:
        return "blocked_text_solvable"
    if cv is not None and cv - holdout >= 0.10 and holdout <= 0.70:
        return "augmentation_candidate"
    if holdout <= 0.65:
        return "probe_candidate"
    return "borderline"


def write_report(records: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for record in records:
        record["triage"] = triage(record)

    summary = {
        "row_count": 500,
        "thresholds": {
            "min_binary_class": MIN_BINARY_CLASS,
            "min_multiclass_class": MIN_MULTICLASS_CLASS,
            "min_holdout_test_n": MIN_HOLDOUT_TEST_N,
            "probe_candidate": "primary source-family char baseline balanced_accuracy <= 0.65",
            "augmentation_candidate": "primary CV - source-family holdout >= 0.10 and holdout <= 0.70",
            "blocked_text_solvable": "primary source-family char baseline balanced_accuracy >= 0.75",
        },
        "records": records,
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    supported = [r for r in records if r["supported"]]
    supported.sort(
        key=lambda r: (
            {"probe_candidate": 0, "augmentation_candidate": 1, "borderline": 2, "blocked_text_solvable": 3}.get(
                r["triage"], 4
            ),
            primary_metric(
                r,
                ("prompt_text" if r["label_type"] == "prompt" else "response_text")
                + ":char:source_family_holdout",
            )
            or 99,
        )
    )

    lines = [
        "# Experiment 03 Label Lexical Gate",
        "",
        "Exploratory lexical preflight for benchmark-native MoReBench labels. Theory-augmentation labels are excluded.",
        "",
        "Important: the dilemma-family and structural-axis labels in this report are heuristic prompt-text flags, not gold annotations. Their purpose is to map leakage and split viability before deciding whether to invest in real annotation.",
        "",
        "## Triage Summary",
        "",
    ]
    by_triage = Counter(r["triage"] for r in records)
    for bucket, count in sorted(by_triage.items()):
        lines.append(f"- `{bucket}`: `{count}`")
    lines.extend(["", "## Supported Labels", ""])
    lines.append(
        "| label | type | triage | counts | primary BA | primary AUROC | CV BA | final/prompt BA |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in supported:
        primary_key = (
            "prompt_text:char:source_family_holdout"
            if r["label_type"] == "prompt"
            else "response_text:char:source_family_holdout"
        )
        cv_key = primary_key.replace("source_family_holdout", "cv")
        aux_key = (
            "dilemma_text:char:source_family_holdout"
            if r["label_type"] == "prompt"
            else "final_span:char:source_family_holdout"
        )
        primary = r["evaluations"].get(primary_key, {})
        cv = r["evaluations"].get(cv_key, {})
        aux = r["evaluations"].get(aux_key, {})
        counts = ", ".join(f"{k}={v}" for k, v in r["counts"].items())
        lines.append(
            "| {label} | {typ} | {triage} | {counts} | {ba} | {auroc} | {cvba} | {auxba} |".format(
                label=r["label"],
                typ=r["label_type"],
                triage=r["triage"],
                counts=counts,
                ba=format_metric(primary.get("balanced_accuracy")),
                auroc=format_metric(primary.get("auroc")),
                cvba=format_metric(cv.get("balanced_accuracy")),
                auxba=format_metric(aux.get("balanced_accuracy")),
            )
        )

    unsupported = [r for r in records if not r["supported"]]
    lines.extend(["", "## Unsupported / Low-Support Labels", ""])
    for r in unsupported:
        counts = ", ".join(f"{k}={v}" for k, v in r["counts"].items())
        lines.append(f"- `{r['label']}` ({r['label_type']}): {counts}")

    lines.extend(["", "## Notes", ""])
    lines.append("- `primary BA` is source-family holdout balanced accuracy for char-TFIDF on the relevant text surface: prompt text for prompt labels, response text for generation labels.")
    lines.append("- `CV BA` is random stratified CV on the same surface and feature class.")
    lines.append("- `final/prompt BA` is final-span char-TFIDF for generation labels, and dilemma-text char-TFIDF for prompt labels.")
    lines.append("- Labels marked `probe_candidate` still need human review of the label definition before activation capture if they are heuristic or support annotations.")
    (REPORT_DIR / "report.md").write_text("\n".join(lines) + "\n")


def format_metric(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def main() -> None:
    rows = load_rows()
    prompt_labels = [
        "source_family",
        "context",
        "role_domain",
        "dilemma_type",
        *FAMILY_PATTERNS.keys(),
    ]
    generation_labels = [
        "strong_helpful",
        "strong_harmless",
        "helpful_binary",
        "harmless_binary",
        "helpful_harmless_off_diagonal",
        "helpful_harmless_cell",
        "heuristic_tradeoff_engagement",
        "heuristic_refuses_or_hedges",
        "heuristic_helpfulness_invoked",
        "heuristic_harm_avoidance_invoked",
        "heuristic_uncertainty_and_scope_calibration",
        "heuristic_commitment_style",
        "heuristic_helpful_harm_dynamic",
        "rubric_recommendation_clarity",
        "rubric_tradeoff_balance",
        "rubric_tailored_actionability",
        "rubric_harm_vector_addressed",
    ]

    records = []
    for label in prompt_labels:
        records.append(evaluate_label(rows, label, "prompt"))
    for label in generation_labels:
        records.append(evaluate_label(rows, label, "generation"))
    write_report(records)
    print(REPORT_DIR)


if __name__ == "__main__":
    main()
