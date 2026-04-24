from __future__ import annotations

import hashlib
import json
import re
from itertools import combinations
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[4]
PHASE_02_OUTPUTS = ROOT / "projects" / "MOREBENCH" / "phase_02" / "outputs"
PHASE_02_REPORTS = ROOT / "projects" / "MOREBENCH" / "phase_02" / "reports"

SOURCE_EXAMPLES_PATH = PHASE_02_OUTPUTS / "theory_prompt_repair_examples.jsonl"
SOURCE_CONTROLS_PATH = PHASE_02_OUTPUTS / "theory_prompt_repair_controls.jsonl"

EXAMPLES_PATH = PHASE_02_OUTPUTS / "theory_prompt_variant_sweep_examples.jsonl"
CONTROLS_PATH = PHASE_02_OUTPUTS / "theory_prompt_variant_sweep_controls.jsonl"
SUMMARY_PATH = PHASE_02_OUTPUTS / "theory_prompt_variant_sweep_summary.json"
REVIEW_PATH = PHASE_02_REPORTS / "theory_prompt_variant_sweep_review.md"

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

VARIANT_BANKS: dict[str, dict[str, str]] = {
    "analytic": {
        "style_name": "analytic",
        "style_rationale": (
            "Uses abstract comparative language and compact theory-specific terminology."
        ),
        "generic": (
            "Assess the morally relevant considerations in each option and compare how well each one is "
            "supported overall. Prefer the option with the strongest all-things-considered case."
        ),
        "Act Utilitarianism": (
            "Assess each option by its total effects on everyone's well-being. Prefer the option whose "
            "expected pattern of benefits and burdens is best overall once every affected party is counted."
        ),
        "Aristotelian Virtue Ethics": (
            "Assess each option by the quality of judgment and character it expresses in this situation. "
            "Prefer the option that reflects practical wisdom, proper balance, and well-formed virtue."
        ),
        "Gauthierian Contractarianism": (
            "Assess each option by what rational parties could agree to when each seeks reliable terms of "
            "advantage. Prefer the option that best supports fair bargaining, reciprocal restraint, and "
            "a stable arrangement people have reason to keep."
        ),
        "Kantian Deontology": (
            "Assess each option by the principle it follows and whether that principle could be willed "
            "consistently in like cases. Prefer the option that honors each person's standing rather than "
            "overriding it for convenience."
        ),
        "Scanlonian Contractualism": (
            "Assess each option by the complaints it would leave each affected person with under the "
            "governing principle. Prefer the option supported by principles that no one could reasonably "
            "reject."
        ),
    },
    "everyday": {
        "style_name": "everyday",
        "style_rationale": (
            "Uses plain-language paraphrases meant to sound like ordinary decision advice rather than theory prose."
        ),
        "generic": (
            "Look at what matters most, who is affected, and what seems fairest or most responsible overall. "
            "Go with the choice that makes the best sense once the full situation is considered."
        ),
        "Act Utilitarianism": (
            "Look at who ends up better off and worse off under each option. Go with the choice that leaves "
            "the full set of people affected in the best overall position."
        ),
        "Aristotelian Virtue Ethics": (
            "Ask what a mature and well-balanced person would do here. Go with the choice that shows good "
            "judgment, steadiness, and the right kind of character."
        ),
        "Gauthierian Contractarianism": (
            "Ask what deal people could reasonably live with when everyone wants workable terms and no one "
            "wants to be cornered. Go with the choice that best protects each side's interests while keeping "
            "the arrangement workable."
        ),
        "Kantian Deontology": (
            "Ask what rule you would be prepared for anyone to follow in the same situation, and whether the "
            "choice honors each person's standing instead of sacrificing them for convenience. Go with the "
            "option that keeps that standard intact."
        ),
        "Scanlonian Contractualism": (
            "Ask what objection each person could make to a rule allowing this choice, and whether that "
            "objection would be hard to answer fairly. Go with the option backed by a principle each person "
            "could live with once those objections are faced directly."
        ),
    },
    "checklist": {
        "style_name": "checklist",
        "style_rationale": (
            "Uses an explicit decision procedure with stepwise instructions instead of a single condensed cue."
        ),
        "generic": (
            "For each option: identify the main considerations, note who is affected, and compare the reasons "
            "for and against. Choose the option that is best supported once the key factors are weighed together."
        ),
        "Act Utilitarianism": (
            "For each option: list the people affected, note the main gains and harms for each, then compare "
            "the totals. Choose the option with the strongest aggregate outcome."
        ),
        "Aristotelian Virtue Ethics": (
            "For each option: note what it reveals about character, whether it shows excess or deficiency, "
            "and whether it fits the demands of the situation. Choose the option that best reflects wise and "
            "balanced judgment."
        ),
        "Gauthierian Contractarianism": (
            "For each option: identify the parties, ask what terms each could accept under mutual concession, "
            "and compare whether the arrangement gives everyone reason to keep to it. Choose the option with "
            "the strongest case for a bargain people would maintain."
        ),
        "Kantian Deontology": (
            "For each option: state the guiding rule, test whether it could hold for anyone in the same kind "
            "of case, and check whether the choice could be openly justified to every person it affects. Choose "
            "the option that best passes those tests."
        ),
        "Scanlonian Contractualism": (
            "For each option: identify who bears the burdens, state the likely complaints they could raise, "
            "and compare whether the governing principle answers those complaints one by one. Choose the "
            "option with the strongest answer to the most serious complaint."
        ),
    },
    "comparative": {
        "style_name": "comparative",
        "style_rationale": (
            "Uses side-by-side comparison language and relative evaluation rather than rule-like instructions."
        ),
        "generic": (
            "Set the options side by side and compare the strongest reasons for each. Favor the option that "
            "has the best overall justification."
        ),
        "Act Utilitarianism": (
            "Set the options side by side and ask which one produces the better overall result for the group "
            "as a whole. Favor the option with the best combined welfare profile."
        ),
        "Aristotelian Virtue Ethics": (
            "Set the options side by side and ask which one embodies the better pattern of character in "
            "context. Favor the option that most clearly displays practical wisdom and virtuous balance."
        ),
        "Gauthierian Contractarianism": (
            "Set the options side by side and ask which one better fits an arrangement people could bargain "
            "into for mutual benefit. Favor the option that better preserves terms people have reason to keep."
        ),
        "Kantian Deontology": (
            "Set the options side by side and ask which one rests on a principle that could be followed "
            "consistently by all. Favor the option that best preserves equal moral standing."
        ),
        "Scanlonian Contractualism": (
            "Set the options side by side and ask which one is better supported by a principle that affected "
            "people could not reasonably reject. Favor the option that leaves the least forceful complaint."
        ),
    },
    "stakeholder": {
        "style_name": "stakeholder",
        "style_rationale": (
            "Keeps the theory stance intact while foregrounding the affected parties and the social viewpoints "
            "inside the case."
        ),
        "generic": (
            "Consider the different people affected, the pressures in the case, and the main reasons pulling "
            "in each direction. Choose the option that best addresses the case overall."
        ),
        "Act Utilitarianism": (
            "Treat every affected person's welfare as part of the same calculation and do not let one "
            "perspective crowd out the others. Choose the option that best balances benefits and burdens across "
            "all of them."
        ),
        "Aristotelian Virtue Ethics": (
            "Consider how a person of good character would respond to the people and pressures present in this "
            "case. Choose the option that expresses sound judgment, proportion, and stable virtue toward those "
            "involved."
        ),
        "Gauthierian Contractarianism": (
            "Consider what each side could accept if all were looking for terms that protect their interests "
            "without leaving anyone open to exploitation. Choose the option that best fits a durable bargain "
            "among them."
        ),
        "Kantian Deontology": (
            "Consider each affected person as someone with standing who cannot simply be traded off for "
            "convenience. Choose the option whose guiding principle could be affirmed for everyone in the case."
        ),
        "Scanlonian Contractualism": (
            "Take each affected person's standpoint seriously and compare the burdens each would have to accept "
            "under the relevant principle. Choose the option whose principle best addresses each person's "
            "strongest complaint."
        ),
    },
    "policy": {
        "style_name": "policy",
        "style_rationale": (
            "Frames the choice as a standing policy or decision rule, which changes rhetoric without changing "
            "the underlying theory stance."
        ),
        "generic": (
            "Use a general moral-decision lens: compare the relevant considerations across the options, weigh "
            "how much each should matter here, and prefer the one with the strongest overall justification."
        ),
        "Act Utilitarianism": (
            "Use an outcome-focused policy lens: compare the consequences of each option for the whole "
            "community of affected parties. Prefer the option with the most favorable overall payoff once all "
            "impacts are included."
        ),
        "Aristotelian Virtue Ethics": (
            "Use a character-and-judgment lens rather than a payoff tally. Prefer the option that a practically "
            "wise agent could stand behind as balanced, fitting, and well-formed in context."
        ),
        "Gauthierian Contractarianism": (
            "Use a bargaining-and-cooperation lens: compare which option could anchor durable terms among "
            "affected parties without giving anyone reason to defect. Prefer the option that most strongly "
            "supports stable cooperation."
        ),
        "Kantian Deontology": (
            "Use a principle-and-respect lens rather than a results tally. Prefer the option grounded in a rule "
            "that can be upheld consistently while honoring each person's standing as a rational chooser."
        ),
        "Scanlonian Contractualism": (
            "Use a complaints-and-justifiability lens: compare which option could be governed by a principle "
            "that each affected party could accept after fair discussion. Prefer the option least exposed to "
            "unresolved complaint."
        ),
    },
}


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _stable_split(group_id: str) -> str:
    digest = hashlib.md5(group_id.encode("utf-8")).hexdigest()
    return "test" if int(digest, 16) % 5 == 0 else "train"


def _extract_dilemma(prompt_text: str) -> str:
    marker = "\n\nDILEMMA:\n"
    if marker not in prompt_text:
        raise ValueError("Expected source prompt to contain a DILEMMA section.")
    return prompt_text.split(marker, 1)[1].strip()


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower())
        if len(token) >= 5
    }


def _char_ngrams(text: str, low: int = 3, high: int = 5) -> set[str]:
    compact = re.sub(r"\s+", " ", text.lower())
    grams: set[str] = set()
    for n in range(low, high + 1):
        for start in range(0, max(0, len(compact) - n + 1)):
            grams.add(compact[start : start + n])
    return grams


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _theory_names_present(text: str) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for theory in THEORY_ORDER:
        if theory.lower() in lowered:
            found.append(theory)
    return found


def build_rows() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    source_examples = _load_jsonl(SOURCE_EXAMPLES_PATH)
    source_controls = _load_jsonl(SOURCE_CONTROLS_PATH)

    description_rows = [
        row
        for row in source_examples
        if str(row.get("variant_family")) == "description_only"
        and str(row.get("description_bank") or "") == "a"
    ]
    generic_rows = [
        row for row in source_controls if str(row.get("control_type")) == "generic_ethics_control"
    ]

    by_group_theory: dict[tuple[str, str], dict[str, object]] = {}
    group_meta: dict[str, dict[str, str]] = {}
    for row in description_rows:
        group_id = str(row["group_id"])
        theory = str(row["theory"])
        by_group_theory[(group_id, theory)] = row
        group_meta[group_id] = {
            "source_family": str(row["source_family"]),
            "dilemma_type": str(row["dilemma_type"]),
            "context": str(row["context"]),
            "role_domain": str(row["role_domain"]),
        }

    generic_by_group = {str(row["group_id"]): row for row in generic_rows}

    example_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []

    for group_id in sorted(group_meta):
        split = _stable_split(group_id)
        meta = group_meta[group_id]
        generic_source = generic_by_group[group_id]
        generic_dilemma = _extract_dilemma(str(generic_source["prompt"]))

        for bank_id, bank_payload in VARIANT_BANKS.items():
            generic_cue = str(bank_payload["generic"])
            control_rows.append(
                {
                    "example_id": f"{group_id}__{GENERIC_CONTROL_LABEL}__variant_{bank_id}",
                    "group_id": group_id,
                    "augmentation_family": "lexical_confound_variant_sweep",
                    "control_type": GENERIC_CONTROL_LABEL,
                    "variant_family": "description_variant_sweep",
                    "cue_mode": "description_only",
                    "variant_bank": bank_id,
                    "variant_style": str(bank_payload["style_name"]),
                    "split": split,
                    "cue_text": generic_cue,
                    "prompt": f"{generic_cue}\n\nDILEMMA:\n{generic_dilemma}",
                    **meta,
                }
            )

        for theory in THEORY_ORDER:
            source_row = by_group_theory[(group_id, theory)]
            dilemma = _extract_dilemma(str(source_row["prompt"]))
            for bank_id, bank_payload in VARIANT_BANKS.items():
                cue_text = str(bank_payload[theory])
                example_rows.append(
                    {
                        "example_id": f"{group_id}__{THEORY_SHORT[theory]}__variant_{bank_id}",
                        "group_id": group_id,
                        "augmentation_family": "lexical_confound_variant_sweep",
                        "variant_family": "description_variant_sweep",
                        "cue_mode": "description_only",
                        "theory": theory,
                        "prime_condition": THEORY_SHORT[theory],
                        "split": split,
                        "alias_bank": None,
                        "description_bank": bank_id,
                        "variant_bank": bank_id,
                        "variant_style": str(bank_payload["style_name"]),
                        "cue_text": cue_text,
                        "prompt": f"{cue_text}\n\nDILEMMA:\n{dilemma}",
                        **meta,
                    }
                )

    cue_catalog: dict[str, dict[str, str]] = {}
    generic_catalog: dict[str, str] = {}
    within_theory_token_jaccard: dict[str, float] = {}
    within_theory_char_jaccard: dict[str, float] = {}

    for theory in THEORY_ORDER:
        theory_cues = {bank_id: str(VARIANT_BANKS[bank_id][theory]) for bank_id in VARIANT_BANKS}
        cue_catalog[theory] = theory_cues
        token_scores = []
        char_scores = []
        for left_bank, right_bank in combinations(sorted(theory_cues), 2):
            left_text = theory_cues[left_bank]
            right_text = theory_cues[right_bank]
            token_scores.append(_jaccard(_content_tokens(left_text), _content_tokens(right_text)))
            char_scores.append(_jaccard(_char_ngrams(left_text), _char_ngrams(right_text)))
        within_theory_token_jaccard[theory] = round(mean(token_scores), 4)
        within_theory_char_jaccard[theory] = round(mean(char_scores), 4)

    for bank_id, bank_payload in VARIANT_BANKS.items():
        generic_catalog[bank_id] = str(bank_payload["generic"])

    cue_length_by_bank = {
        bank_id: {
            "generic_words": len(_normalize_space(str(bank_payload["generic"])).split()),
            "mean_theory_words": round(
                mean(len(_normalize_space(str(bank_payload[theory])).split()) for theory in THEORY_ORDER),
                2,
            ),
        }
        for bank_id, bank_payload in VARIANT_BANKS.items()
    }

    theory_name_leakage = {
        theory: {
            bank_id: _theory_names_present(text)
            for bank_id, text in theory_cues.items()
        }
        for theory, theory_cues in cue_catalog.items()
    }

    summary = {
        "benchmark": "morebench",
        "phase": "02",
        "artifact_family": "theory_prompt_variant_sweep",
        "status": "materialized_for_review",
        "purpose": (
            "Systematic training-distribution variation for theory prompts: six style-diverse description-only "
            "banks per theory plus matched generic controls, intended for held-out-variant prompt-side testing."
        ),
        "counts": {
            "matched_groups": len(group_meta),
            "theory_count": len(THEORY_ORDER),
            "variant_bank_count": len(VARIANT_BANKS),
            "example_row_count": len(example_rows),
            "generic_control_row_count": len(control_rows),
            "total_prompt_count": len(example_rows) + len(control_rows),
        },
        "bank_styles": {
            bank_id: {
                "style_name": str(bank_payload["style_name"]),
                "style_rationale": str(bank_payload["style_rationale"]),
            }
            for bank_id, bank_payload in VARIANT_BANKS.items()
        },
        "cue_catalog": cue_catalog,
        "generic_cue_catalog": generic_catalog,
        "review_gate": {
            "status": "human_review_required_before_any_run",
            "checks": [
                "semantic fidelity to intended theory",
                "no repeated anchor phrase across all banks",
                "no accidental theory-name leakage",
                "bank styles are visibly different in tone and syntax",
                "generic variants remain theory-neutral",
            ],
        },
        "lexical_diversity_heuristics": {
            "within_theory_mean_content_token_jaccard": within_theory_token_jaccard,
            "within_theory_mean_char_ngram_jaccard": within_theory_char_jaccard,
            "cue_length_by_bank": cue_length_by_bank,
            "theory_name_leakage": theory_name_leakage,
        },
        "intended_eval_shape": {
            "primary_phase": "prompt_side_final_token_or_prompt_end_capture",
            "cross_variant_cv": "train on 5 banks, test on held-out bank",
            "scaling_study": [1, 3, 6],
            "primary_metric": "probe minus text AUROC on held-out variant banks",
        },
        "output_paths": {
            "examples": str(EXAMPLES_PATH.relative_to(ROOT)),
            "controls": str(CONTROLS_PATH.relative_to(ROOT)),
            "summary": str(SUMMARY_PATH.relative_to(ROOT)),
            "review": str(REVIEW_PATH.relative_to(ROOT)),
        },
    }
    return example_rows, control_rows, summary


def write_review(summary: dict[str, object]) -> None:
    lines: list[str] = []
    lines.append("# Theory Prompt Variant Sweep Review\n")
    lines.append("## Why This Exists\n")
    lines.append(
        "This asset is a strategic augmentation pass for beating lexical confounds rather than enlarging the "
        "dataset. It creates six style-diverse description-only banks per theory, plus matched generic controls, "
        "so later prompt-side analyses can use held-out-variant transfer instead of a single fixed theory cue.\n"
    )
    lines.append("## Counts\n")
    counts = dict(summary["counts"])
    lines.append(f"- matched groups: `{counts['matched_groups']}`\n")
    lines.append(f"- theory rows: `{counts['example_row_count']}`\n")
    lines.append(f"- generic control rows: `{counts['generic_control_row_count']}`\n")
    lines.append(f"- total prompts: `{counts['total_prompt_count']}`\n")
    lines.append("")
    lines.append("## Bank Styles\n")
    for bank_id, payload in dict(summary["bank_styles"]).items():
        style_name = payload["style_name"]
        rationale = payload["style_rationale"]
        lines.append(f"- `{bank_id}` / `{style_name}`: {rationale}\n")
    lines.append("")
    lines.append("## Review Checklist\n")
    for item in dict(summary["review_gate"])["checks"]:
        lines.append(f"- {item}\n")
    lines.append("")

    generic_catalog = dict(summary["generic_cue_catalog"])
    cue_catalog = dict(summary["cue_catalog"])
    bank_order = list(VARIANT_BANKS)
    lines.append("## Generic Control Cues\n")
    for bank_id in bank_order:
        lines.append(f"### `{bank_id}`\n")
        lines.append(f"- {generic_catalog[bank_id]}\n")
    lines.append("")

    lines.append("## Theory Cues By Bank\n")
    for theory in THEORY_ORDER:
        lines.append(f"### {theory}\n")
        for bank_id in bank_order:
            lines.append(f"- `{bank_id}`: {cue_catalog[theory][bank_id]}\n")
        lines.append("")

    heuristics = dict(summary["lexical_diversity_heuristics"])
    lines.append("## Lexical Diversity Heuristics\n")
    for theory in THEORY_ORDER:
        token_j = heuristics["within_theory_mean_content_token_jaccard"][theory]
        char_j = heuristics["within_theory_mean_char_ngram_jaccard"][theory]
        lines.append(
            f"- `{theory}`: mean content-token Jaccard `{token_j}`, mean char-ngram Jaccard `{char_j}`\n"
        )

    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_PATH.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    example_rows, control_rows, summary = build_rows()
    _write_jsonl(EXAMPLES_PATH, example_rows)
    _write_jsonl(CONTROLS_PATH, control_rows)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_review(summary)
    print(str(EXAMPLES_PATH.relative_to(ROOT)))
    print(str(CONTROLS_PATH.relative_to(ROOT)))
    print(str(SUMMARY_PATH.relative_to(ROOT)))
    print(str(REVIEW_PATH.relative_to(ROOT)))


if __name__ == "__main__":
    main()
