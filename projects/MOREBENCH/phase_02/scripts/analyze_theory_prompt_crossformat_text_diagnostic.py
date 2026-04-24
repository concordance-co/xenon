from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[4]
DESC_SUMMARY_PATH = ROOT / "projects" / "MOREBENCH" / "phase_02" / "outputs" / "theory_prompt_variant_sweep_summary.json"
NAME_SUMMARY_PATH = ROOT / "projects" / "MOREBENCH" / "phase_02" / "outputs" / "theory_name_only_crossformat_summary.json"
OUTPUT_PATH = ROOT / "projects" / "MOREBENCH" / "phase_02" / "outputs" / "theory_prompt_crossformat_text_diagnostic.json"
REPORT_PATH = ROOT / "projects" / "MOREBENCH" / "phase_02" / "reports" / "theory_prompt_crossformat_text_diagnostic.md"

THEORY_ORDER = (
    "Act Utilitarianism",
    "Aristotelian Virtue Ethics",
    "Gauthierian Contractarianism",
    "Kantian Deontology",
    "Scanlonian Contractualism",
)


def _fit_predict(train_texts: list[str], train_labels: list[int], test_texts: list[str], test_labels: list[int]) -> float:
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, train_labels)
    probs = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(test_labels, probs))


def main() -> None:
    desc = json.loads(DESC_SUMMARY_PATH.read_text())
    name = json.loads(NAME_SUMMARY_PATH.read_text())
    desc_theory = desc["cue_catalog"]
    desc_generic = desc["generic_cue_catalog"]
    name_banks = name["name_banks"]

    output: dict[str, object] = {
        "benchmark": "morebench",
        "phase": "02",
        "artifact_family": "theory_prompt_crossformat_text_diagnostic",
        "vectorizer": {"analyzer": "char", "ngram_range": [3, 5]},
        "model": "tfidf_char_logreg",
        "description_to_name": {},
        "name_to_description": {},
    }

    lines: list[str] = []
    lines.append("# Theory Prompt Cross-Format Text Diagnostic\n")
    lines.append(
        "Char-TF-IDF + logistic regression on cue strings only. Trains on one format and tests on the other "
        "for theory-vs-generic discrimination.\n\n"
    )

    for theory in THEORY_ORDER:
        train_desc = []
        train_desc_labels = []
        for bank, text in desc_theory[theory].items():
            train_desc.append(text)
            train_desc_labels.append(1)
            train_desc.append(desc_generic[bank])
            train_desc_labels.append(0)
        test_name = []
        test_name_labels = []
        for bank_name, bank_payload in name_banks.items():
            test_name.append(bank_payload["theory_cues"][theory])
            test_name_labels.append(1)
            test_name.append(bank_payload["generic"])
            test_name_labels.append(0)
        desc_to_name = _fit_predict(train_desc, train_desc_labels, test_name, test_name_labels)

        train_name = []
        train_name_labels = []
        for bank_name, bank_payload in name_banks.items():
            train_name.append(bank_payload["theory_cues"][theory])
            train_name_labels.append(1)
            train_name.append(bank_payload["generic"])
            train_name_labels.append(0)
        test_desc = []
        test_desc_labels = []
        for bank, text in desc_theory[theory].items():
            test_desc.append(text)
            test_desc_labels.append(1)
            test_desc.append(desc_generic[bank])
            test_desc_labels.append(0)
        name_to_desc = _fit_predict(train_name, train_name_labels, test_desc, test_desc_labels)

        output["description_to_name"][theory] = round(desc_to_name, 4)
        output["name_to_description"][theory] = round(name_to_desc, 4)
        lines.append(f"## {theory}\n")
        lines.append(f"- description -> name AUROC: `{desc_to_name:.4f}`\n")
        lines.append(f"- name -> description AUROC: `{name_to_desc:.4f}`\n\n")

    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(str(OUTPUT_PATH.relative_to(ROOT)))
    print(str(REPORT_PATH.relative_to(ROOT)))


if __name__ == "__main__":
    main()
