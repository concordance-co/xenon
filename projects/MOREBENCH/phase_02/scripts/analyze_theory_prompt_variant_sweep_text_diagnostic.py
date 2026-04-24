from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[4]
SUMMARY_PATH = ROOT / "projects" / "MOREBENCH" / "phase_02" / "outputs" / "theory_prompt_variant_sweep_summary.json"
REPORT_PATH = ROOT / "projects" / "MOREBENCH" / "phase_02" / "reports" / "theory_prompt_variant_sweep_text_diagnostic.md"
OUTPUT_PATH = ROOT / "projects" / "MOREBENCH" / "phase_02" / "outputs" / "theory_prompt_variant_sweep_text_diagnostic.json"

THEORY_ORDER = (
    "Act Utilitarianism",
    "Aristotelian Virtue Ethics",
    "Gauthierian Contractarianism",
    "Kantian Deontology",
    "Scanlonian Contractualism",
)


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    cue_catalog = summary["cue_catalog"]
    generic_catalog = summary["generic_cue_catalog"]
    banks = list(generic_catalog.keys())

    results: dict[str, object] = {
        "benchmark": "morebench",
        "phase": "02",
        "artifact_family": "theory_prompt_variant_sweep_text_diagnostic",
        "vectorizer": {
            "analyzer": "char",
            "ngram_range": [3, 5],
        },
        "model": "tfidf_char_logreg",
        "holdout_unit": "variant_bank",
        "theory_vs_generic": {},
    }

    lines: list[str] = []
    lines.append("# Theory Prompt Variant Sweep Text Diagnostic\n")
    lines.append(
        "Char-TF-IDF + logistic regression on cue text only, trained on five banks and tested on the held-out bank "
        "for each theory-vs-generic pair.\n\n"
    )

    for theory in THEORY_ORDER:
        holdouts: list[dict[str, object]] = []
        aucs: list[float] = []
        for held_out_bank in banks:
            train_texts: list[str] = []
            train_labels: list[int] = []
            test_texts = [cue_catalog[theory][held_out_bank], generic_catalog[held_out_bank]]
            test_labels = [1, 0]
            for bank in banks:
                if bank == held_out_bank:
                    continue
                train_texts.extend([cue_catalog[theory][bank], generic_catalog[bank]])
                train_labels.extend([1, 0])
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
            X_train = vectorizer.fit_transform(train_texts)
            X_test = vectorizer.transform(test_texts)
            model = LogisticRegression(max_iter=2000, class_weight="balanced")
            model.fit(X_train, train_labels)
            probs = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(test_labels, probs))
            aucs.append(auc)
            holdouts.append(
                {
                    "held_out_bank": held_out_bank,
                    "auroc": round(auc, 4),
                    "theory_probability": round(float(probs[0]), 4),
                    "generic_probability": round(float(probs[1]), 4),
                }
            )
        mean_auc = sum(aucs) / len(aucs)
        worst_auc = min(aucs)
        best_auc = max(aucs)
        results["theory_vs_generic"][theory] = {
            "mean_auroc": round(mean_auc, 4),
            "min_auroc": round(worst_auc, 4),
            "max_auroc": round(best_auc, 4),
            "holdouts": holdouts,
        }
        lines.append(f"## {theory}\n")
        lines.append(f"- mean AUROC: `{mean_auc:.4f}`\n")
        lines.append(f"- min/max AUROC: `{worst_auc:.4f}` / `{best_auc:.4f}`\n")
        for item in holdouts:
            lines.append(
                f"- holdout `{item['held_out_bank']}`: AUROC `{item['auroc']}`, "
                f"theory prob `{item['theory_probability']}`, generic prob `{item['generic_probability']}`\n"
            )
        lines.append("")

    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(str(OUTPUT_PATH.relative_to(ROOT)))
    print(str(REPORT_PATH.relative_to(ROOT)))


if __name__ == "__main__":
    main()
