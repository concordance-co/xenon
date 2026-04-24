from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import modal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


GENERATION_ARTIFACT_ID = "generation_run_1_6e2b8f5a2902"
VOLUME_NAME = "xenon-data"
VOLUME_RESULT_PATH = (
    "artifacts/morebench_phase_03_experiment02_cross_language_pilot/"
    f"{GENERATION_ARTIFACT_ID}/result.json"
)

REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_cross_language_pilot")
REPORT_PATH = REPORT_DIR / "report.md"
SUMMARY_PATH = REPORT_DIR / "summary.json"

LANGUAGE_ORDER = ("en", "es", "zh")
LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "zh": "Simplified Chinese"}


def _load_generation_rows() -> list[dict[str, Any]]:
    volume = modal.Volume.from_name(VOLUME_NAME)
    payload = json.loads(b"".join(volume.read_file(VOLUME_RESULT_PATH)))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("Generation payload rows missing or malformed.")
    return rows


def _response_text(row: dict[str, Any]) -> str:
    return str(row.get("generated_text") or row.get("text") or "")


def _labels(row: dict[str, Any]) -> dict[str, Any]:
    example = row.get("example") or {}
    labels = example.get("labels") or {}
    if not isinstance(labels, dict):
        return {}
    return labels


def _script_purity(language_code: str, text: str) -> float:
    if language_code == "en":
        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            return 0.0
        ascii_letters = [ch for ch in letters if "a" <= ch.lower() <= "z"]
        return len(ascii_letters) / len(letters)
    if language_code == "es":
        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            return 0.0
        latinish = [ch for ch in letters if ("a" <= ch.lower() <= "z") or ch.lower() in "áéíóúüñ"]
        return len(latinish) / len(letters)
    if language_code == "zh":
        chars = [ch for ch in text if not ch.isspace()]
        if not chars:
            return 0.0
        han = [ch for ch in chars if "\u4e00" <= ch <= "\u9fff"]
        return len(han) / len(chars)
    return 0.0


def _clean_markdown(text: str) -> str:
    text = re.sub(r"[*#>`_~-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _letters_only(text: str) -> str:
    chars: list[str] = []
    for ch in text:
        category = unicodedata.category(ch)
        chars.append(ch if category.startswith("L") or ch.isspace() else " ")
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def _strip_ascii_word_tokens(text: str) -> str:
    text = re.sub(r"\b[A-Za-z][A-Za-z'_-]*\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _fit_auc(train_texts: list[str], train_labels: list[int], test_texts: list[str], test_labels: list[int]) -> float:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    model = LogisticRegression(max_iter=4000, class_weight="balanced")
    model.fit(x_train, train_labels)
    probs = model.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(test_labels, probs))


def _matrix(
    rows_by_language: dict[str, list[dict[str, Any]]],
    cleaner,
) -> tuple[dict[str, dict[str, float | None]], float | None]:
    matrix: dict[str, dict[str, float | None]] = {}
    cross_values: list[float] = []
    for train_lang in LANGUAGE_ORDER:
        train_rows = rows_by_language.get(train_lang, [])
        train_texts = [cleaner(row) for row in train_rows]
        train_labels = [1 if _labels(row).get("prime_condition") == "deontology" else 0 for row in train_rows]
        matrix[train_lang] = {}
        for test_lang in LANGUAGE_ORDER:
            test_rows = rows_by_language.get(test_lang, [])
            test_texts = [cleaner(row) for row in test_rows]
            test_labels = [1 if _labels(row).get("prime_condition") == "deontology" else 0 for row in test_rows]
            if len(set(train_labels)) < 2 or len(set(test_labels)) < 2:
                matrix[train_lang][test_lang] = None
                continue
            try:
                auc = round(_fit_auc(train_texts, train_labels, test_texts, test_labels), 4)
            except ValueError:
                matrix[train_lang][test_lang] = None
                continue
            matrix[train_lang][test_lang] = auc
            if train_lang != test_lang:
                cross_values.append(auc)
    mean_cross = round(sum(cross_values) / len(cross_values), 4) if cross_values else None
    return matrix, mean_cross


def _zh_code_switch_summary(rows_by_language: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    theory_terms = ("Phronesis", "Temperance", "Virtue", "Categorical", "Imperative")
    rows = rows_by_language.get("zh", [])
    details = []
    for row in rows:
        text = _response_text(row)
        hits = [term for term in theory_terms if term.lower() in text.lower()]
        details.append(
            {
                "group_id": _labels(row).get("group_id"),
                "prime_condition": _labels(row).get("prime_condition"),
                "script_purity": round(_script_purity("zh", text), 4),
                "english_theory_terms": hits,
            }
        )
    return {
        "rows_with_english_theory_terms": sum(1 for item in details if item["english_theory_terms"]),
        "details": details,
    }


def _prompt_theory_term_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    theory_terms = ("Phronesis", "Temperance", "Virtue", "Categorical", "Imperative")
    details = []
    for row in rows:
        labels = _labels(row)
        language_code = str(labels.get("language_code") or "")
        if language_code == "en":
            continue
        example = row.get("example") or {}
        prompt_messages = example.get("prompt") or []
        prompt_text = "\n".join(
            str(message.get("content") or "")
            for message in prompt_messages
            if isinstance(message, dict)
        )
        hits = [term for term in theory_terms if term.lower() in prompt_text.lower()]
        details.append(
            {
                "group_id": labels.get("group_id"),
                "prime_condition": labels.get("prime_condition"),
                "language_code": language_code,
                "english_theory_terms": hits,
            }
        )
    return {
        "rows_with_english_theory_terms": sum(1 for item in details if item["english_theory_terms"]),
        "details": details,
    }


def _tail_preview(text: str, limit: int = 320) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[-limit:]


def _write_report(summary: dict[str, Any]) -> None:
    raw = summary["raw_matrix"]
    clean_md = summary["clean_markdown_matrix"]
    ascii_ablate = summary["non_english_ascii_stripped_matrix"]
    report = f"""# Experiment 02 Cross-Language Pilot

Small fully translated diagonal pilot testing whether `English / Spanish / Simplified Chinese` output breaks the response-side lexical ceiling for `deontology` vs `virtue_ethics`.

## Run
- workflow run id: `wr_74c38b01483d_7e3195fe`
- generation artifact: `{GENERATION_ARTIFACT_ID}`
- transform artifact: `transform_ca6b7a420e0a_c939a86f`
- runtime app id: `ap-mmO4OzSUvrr0k4sTT05AdK`

## Pilot Shape
- `5` theory groups:
  - `theory_group_005`
  - `theory_group_009`
  - `theory_group_013`
  - `theory_group_015`
  - `theory_group_022`
- `2` theories:
  - `deontology`
  - `virtue_ethics`
- `3` fully translated language conditions:
  - `English in / English out`
  - `Spanish in / Spanish out`
  - `Simplified Chinese in / Simplified Chinese out`
- `30` prompts total
- all `30/30` generations completed with `stop`

## Main Result
The cross-language move helped, but not enough to cleanly reopen response-side probing.

Raw cross-language char-TF-IDF AUROC matrix:

| train -> test | `en` | `es` | `zh` |
| --- | ---: | ---: | ---: |
| `en` | `{raw['en']['en']:.2f}` | `{raw['en']['es']:.2f}` | `{raw['en']['zh']:.2f}` |
| `es` | `{raw['es']['en']:.2f}` | `{raw['es']['es']:.2f}` | `{raw['es']['zh']:.2f}` |
| `zh` | `{raw['zh']['en']:.2f}` | `{raw['zh']['es']:.2f}` | `{raw['zh']['zh']:.2f}` |

- mean cross-language AUROC: `{summary['mean_cross_language_auroc']:.4f}`

Markdown-stripped cross-language AUROC matrix:

| train -> test | `en` | `es` | `zh` |
| --- | ---: | ---: | ---: |
| `en` | `{clean_md['en']['en']:.2f}` | `{clean_md['en']['es']:.2f}` | `{clean_md['en']['zh']:.2f}` |
| `es` | `{clean_md['es']['en']:.2f}` | `{clean_md['es']['es']:.2f}` | `{clean_md['es']['zh']:.2f}` |
| `zh` | `{clean_md['zh']['en']:.2f}` | `{clean_md['zh']['es']:.2f}` | `{clean_md['zh']['zh']:.2f}` |

- mean cross-language markdown-stripped AUROC: `{summary['mean_cross_language_markdown_stripped_auroc']:.4f}`

Non-English ASCII-token ablation matrix:

| train -> test | `en` | `es` | `zh` |
| --- | ---: | ---: | ---: |
| `en` | `{ascii_ablate['en']['en']:.2f}` | `{ascii_ablate['en']['es']:.2f}` | `{ascii_ablate['en']['zh']:.2f}` |
| `es` | `{ascii_ablate['es']['en']:.2f}` | `{ascii_ablate['es']['es']:.2f}` | `{ascii_ablate['es']['zh']:.2f}` |
| `zh` | `{ascii_ablate['zh']['en']:.2f}` | `{ascii_ablate['zh']['es']:.2f}` | `{ascii_ablate['zh']['zh']:.2f}` |

- mean cross-language AUROC after ablating ASCII tokens from non-English outputs: `{summary['mean_cross_language_non_english_ascii_stripped_auroc']:.4f}`

## Interpretation
- This is the first response-side intervention that clearly reduced the lexical ceiling relative to the English-only pilots.
- The strongest drop was `Spanish -> Chinese`, which fell to `{raw['es']['zh']:.2f}` on raw text.
- But the overall cross-language ceiling stayed high: mean cross-language AUROC remained `{summary['mean_cross_language_auroc']:.4f}`, well above the intended `<= 0.70` scale-up gate.
- The remaining signal is not just Markdown/header leakage: stripping Markdown still leaves high transfer in most cells.
- The asymmetry is important: `Chinese -> English` stayed at `{raw['zh']['en']:.2f}` on raw text, but after ablating ASCII tokens from non-English outputs it collapsed to `{ascii_ablate['zh']['en']:.2f}`; `English -> Chinese` likewise fell from `{raw['en']['zh']:.2f}` to `{ascii_ablate['en']['zh']:.2f}`.
- That pattern strongly supports the code-switching explanation: the surviving `en <-> zh` ceiling was largely carried by English theory terms embedded inside the Chinese outputs, not by genuine cross-script char-ngram transfer.

## Language / Fidelity Notes
- Script purity was strong for English and Spanish (`1.0` mean each).
- Chinese stayed mostly Chinese but not perfectly pure (`{summary['length_summary']['zh']['mean_script_purity']:.4f}` mean Han-script share).
- Several Chinese rows still contained English philosophical terms like `Phronesis`, `Temperance`, `Virtue`, or `Categorical Imperative`, especially on `virtue_ethics` rows. That is a real residual leakage path.
- Manual spot-check of the response tails suggests theory fidelity mostly held: deontology rows still argued in principle / standing terms, and virtue rows still argued in practical-wisdom / balance terms, but the Chinese virtue outputs were the most likely to code-switch into English philosophical vocabulary.
- Prompt audit was clean on the specific theory terms: non-English prompts contained `0` rows with English philosophical anchor terms from the audit set.

## What This Means
- Full translation made an impact. This was not another pure `1.0` ceiling result.
- The English-token ablation clarifies the story: language variation *does* break the baseline when code-switching is controlled, but the current response-side outputs still reintroduce English lexical anchors.
- That is a real methodological win for the translation strategy, but still not a clean response-side win.
- The honest read is:
  - `not zero`: yes, the text ceiling moved
  - `methodological validation of cross-language variation`: yes, partial
  - `clean room for a probe`: not yet
  - `ready to scale response-side activation capture`: no

## Recommendation
- Do **not** scale response-side cross-language capture yet.
- If we stay on this line, the cleaner next move is prompt-side / pre-generation state on the same translated prompts.
- If we revisit response-side later, we should first tighten the non-English outputs further, especially Chinese code-switching on theory-specific terms.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_generation_rows()

    finish_reason_counts: Counter[str] = Counter()
    rows_by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lengths_by_language: dict[str, list[int]] = defaultdict(list)
    purity_by_language: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        finish_reason_counts[str(row.get("finish_reason") or "")] += 1
        labels = _labels(row)
        language_code = str(labels.get("language_code") or "")
        text = _response_text(row)
        rows_by_language[language_code].append(row)
        lengths_by_language[language_code].append(len(text))
        purity_by_language[language_code].append(_script_purity(language_code, text))

    raw_matrix, mean_cross = _matrix(rows_by_language, cleaner=lambda row: _response_text(row))
    clean_md_matrix, mean_cross_clean_md = _matrix(
        rows_by_language,
        cleaner=lambda row: _clean_markdown(_response_text(row)),
    )
    letters_only_matrix, mean_cross_letters = _matrix(
        rows_by_language,
        cleaner=lambda row: _letters_only(_response_text(row)),
    )
    non_english_ascii_stripped_matrix, mean_cross_non_english_ascii = _matrix(
        rows_by_language,
        cleaner=lambda row: (
            _response_text(row)
            if str(_labels(row).get("language_code") or "") == "en"
            else _strip_ascii_word_tokens(_response_text(row))
        ),
    )

    length_summary = {
        lang: {
            "count": len(lengths_by_language[lang]),
            "mean_char_length": round(sum(lengths_by_language[lang]) / len(lengths_by_language[lang]), 1),
            "min_char_length": min(lengths_by_language[lang]),
            "max_char_length": max(lengths_by_language[lang]),
            "mean_script_purity": round(sum(purity_by_language[lang]) / len(purity_by_language[lang]), 4),
        }
        for lang in LANGUAGE_ORDER
    }

    sample_tails = []
    for row in rows[:12]:
        labels = _labels(row)
        sample_tails.append(
            {
                "example_key": row.get("example_key"),
                "language_code": labels.get("language_code"),
                "prime_condition": labels.get("prime_condition"),
                "tail_preview": _tail_preview(_response_text(row)),
            }
        )

    summary = {
        "generation_artifact_id": GENERATION_ARTIFACT_ID,
        "row_count": len(rows),
        "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
        "language_counts": {lang: len(rows_by_language[lang]) for lang in LANGUAGE_ORDER},
        "raw_matrix": raw_matrix,
        "mean_cross_language_auroc": mean_cross,
        "clean_markdown_matrix": clean_md_matrix,
        "mean_cross_language_markdown_stripped_auroc": mean_cross_clean_md,
        "letters_only_matrix": letters_only_matrix,
        "mean_cross_language_letters_only_auroc": mean_cross_letters,
        "non_english_ascii_stripped_matrix": non_english_ascii_stripped_matrix,
        "mean_cross_language_non_english_ascii_stripped_auroc": mean_cross_non_english_ascii,
        "length_summary": length_summary,
        "zh_code_switch_summary": _zh_code_switch_summary(rows_by_language),
        "prompt_theory_term_audit": _prompt_theory_term_audit(rows),
        "sample_tails": sample_tails,
        "decision": {
            "mean_cross_language_auroc_le_070": bool(mean_cross is not None and mean_cross <= 0.70),
            "any_raw_cross_language_cell_le_070": any(
                value is not None and value <= 0.70
                for train_lang, test_map in raw_matrix.items()
                for test_lang, value in test_map.items()
                if train_lang != test_lang
            ),
            "any_non_english_ascii_stripped_cross_language_cell_le_070": any(
                value is not None and value <= 0.70
                for train_lang, test_map in non_english_ascii_stripped_matrix.items()
                for test_lang, value in test_map.items()
                if train_lang != test_lang
            ),
            "recommended_next_step": "prompt_side_next",
        },
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(summary)


if __name__ == "__main__":
    main()
