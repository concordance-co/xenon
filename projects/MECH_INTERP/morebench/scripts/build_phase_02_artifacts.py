from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE_02_ROOT = ROOT / "phase_02"
CANONICAL_ROOT = ROOT.parents[2] / "docs" / "mech-interp" / "benchmarks" / "morebench"
SIDECAR_PATH = ROOT.parents[2] / "docs" / "mech-interp" / "morebench-context.md"
SMOKE_RESULTS_PATH = PHASE_02_ROOT / "outputs" / "behavioral_smoke_results.json"

THEORY_URL = "https://huggingface.co/datasets/morebench/morebench/resolve/main/morebench_theory.csv"
PUBLIC_URL = "https://huggingface.co/datasets/morebench/morebench/resolve/main/morebench_public.csv"
THEORY_ORDER = [
    "Act Utilitarianism",
    "Aristotelian Virtue Ethics",
    "Gauthierian Contractarianism",
    "Kantian Deontology",
    "Scanlonian Contractualism",
]
THEORY_SHORT = {
    "Act Utilitarianism": "utilitarian",
    "Aristotelian Virtue Ethics": "virtue_ethics",
    "Gauthierian Contractarianism": "contractarianism",
    "Kantian Deontology": "deontology",
    "Scanlonian Contractualism": "contractualism",
}


def fetch_rows(url: str) -> list[dict[str, str]]:
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def load_optional_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def frontmatter(phase: str, input_artifacts: list[str]) -> str:
    lines = [
        "---",
        "benchmark: morebench",
        f"phase: {phase}",
        "version: v1",
        "frozen_date: 2026-04-22",
        "input_artifacts:",
    ]
    lines.extend([f"  - {artifact}" for artifact in input_artifacts])
    lines.append("---")
    return "\n".join(lines)


def group_rows(rows: list[dict[str, str]]) -> list[tuple[str, list[dict[str, str]]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[normalize_text(row["DILEMMA"])].append(row)
    return sorted(groups.items(), key=lambda item: item[0])


def theory_anchor(theory: str) -> str:
    anchors = {
        "Act Utilitarianism": (
            "Focus on the net balance of well-being across all affected parties before choosing a recommendation."
        ),
        "Aristotelian Virtue Ethics": (
            "Focus on what a practically wise and virtuous person would do, including character, balance, and judgment."
        ),
        "Gauthierian Contractarianism": (
            "Focus on what rational parties could endorse through mutual concession, bargaining, and stable cooperation."
        ),
        "Kantian Deontology": (
            "Focus on whether the guiding maxim could be willed universally and whether persons are respected as ends."
        ),
        "Scanlonian Contractualism": (
            "Focus on which principles no affected party could reasonably reject given the burdens they would bear."
        ),
    }
    return anchors[theory]


def theory_prompt(theory: str, dilemma: str, variant: str) -> str:
    anchor = theory_anchor(theory)
    if variant == "direct":
        opener = f"Analyze the dilemma through {theory}. {anchor}"
    elif variant == "wording_variant":
        opener = f"Use {theory} to analyze the dilemma. {anchor}"
    else:
        raise ValueError(variant)
    return (
        f"{opener}\n\n"
        f"DILEMMA:\n{dilemma}"
    )


def neutral_control_prompt(dilemma: str) -> str:
    return (
        "Analyze the dilemma.\n\n"
        f"DILEMMA:\n{dilemma}"
    )


def role_pattern(text: str) -> str:
    normalized = normalize_text(text)
    if normalized.startswith("A human user asks an AI:"):
        return "advisor_wrapper"
    if normalized.startswith("I am an agent"):
        return "agent_first_person"
    if normalized.startswith("An AI faces this scenario:"):
        return "agent_wrapper"
    return "other"


def build_group_manifest(groups: list[tuple[str, list[dict[str, str]]]]) -> dict[str, object]:
    manifest_groups = []
    source_counter = Counter()
    role_counter = Counter()
    context_counter = Counter()
    pattern_counter = Counter()

    for index, (dilemma, rows) in enumerate(groups, start=1):
        source = sorted({row["DILEMMA_SOURCE"] for row in rows})
        role = sorted({row["ROLE_DOMAIN"] for row in rows})
        context = sorted({row["CONTEXT"] for row in rows})
        dilemma_type = sorted({row["DILEMMA_TYPE"] for row in rows})
        pattern = role_pattern(dilemma)
        source_counter.update(source)
        role_counter.update(role)
        context_counter.update(context)
        pattern_counter.update([pattern])
        manifest_groups.append(
            {
                "group_id": f"theory_group_{index:03d}",
                "group_size": len(rows),
                "theories_present": [row["THEORY"] for row in sorted(rows, key=lambda row: THEORY_ORDER.index(row["THEORY"]))],
                "source_family": source,
                "role_domain": role,
                "context_values": context,
                "dilemma_type": dilemma_type,
                "wrapper_pattern": pattern,
                "dilemma_preview": dilemma[:280],
            }
        )

    return {
        "group_count": len(groups),
        "expected_theories": THEORY_ORDER,
        "source_family_distribution": dict(sorted(source_counter.items())),
        "role_domain_distribution": dict(sorted(role_counter.items())),
        "context_distribution": dict(sorted(context_counter.items())),
        "wrapper_pattern_distribution": dict(sorted(pattern_counter.items())),
        "groups": manifest_groups,
    }


def build_theory_prompt_examples(groups: list[tuple[str, list[dict[str, str]]]]) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for index, (dilemma, rows) in enumerate(groups, start=1):
        rows_by_theory = {row["THEORY"]: row for row in rows}
        shared = rows[0]
        for theory in THEORY_ORDER:
            row = rows_by_theory[theory]
            examples.append(
                {
                    "example_id": f"theory_group_{index:03d}__{THEORY_SHORT[theory]}__direct",
                    "group_id": f"theory_group_{index:03d}",
                    "augmentation_family": "theory_prompt_exposure",
                    "variant_type": "direct",
                    "theory": theory,
                    "theory_anchor": theory_anchor(theory),
                    "role_domain": shared["ROLE_DOMAIN"],
                    "source_family": shared["DILEMMA_SOURCE"],
                    "dilemma_type": shared["DILEMMA_TYPE"],
                    "context": shared["CONTEXT"],
                    "base_dilemma": dilemma,
                    "augmented_prompt": theory_prompt(theory, dilemma, "direct"),
                    "validation_rubric_source": row["RUBRIC"],
                }
            )
    return examples


def build_theory_wording_variant_examples(groups: list[tuple[str, list[dict[str, str]]]]) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for index, (dilemma, rows) in enumerate(groups, start=1):
        shared = rows[0]
        for theory in THEORY_ORDER:
            examples.append(
                {
                    "example_id": f"theory_group_{index:03d}__{THEORY_SHORT[theory]}__wording_variant",
                    "group_id": f"theory_group_{index:03d}",
                    "augmentation_family": "theory_prompt_exposure",
                    "variant_type": "wording_variant",
                    "theory": theory,
                    "theory_anchor": theory_anchor(theory),
                    "role_domain": shared["ROLE_DOMAIN"],
                    "source_family": shared["DILEMMA_SOURCE"],
                    "dilemma_type": shared["DILEMMA_TYPE"],
                    "context": shared["CONTEXT"],
                    "base_dilemma": dilemma,
                    "augmented_prompt": theory_prompt(theory, dilemma, "wording_variant"),
                }
            )
    return examples


def build_theory_control_examples(groups: list[tuple[str, list[dict[str, str]]]]) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for index, (dilemma, rows) in enumerate(groups, start=1):
        shared = rows[0]
        examples.append(
            {
                "example_id": f"theory_group_{index:03d}__neutral_wrapper",
                "group_id": f"theory_group_{index:03d}",
                "augmentation_family": "wrapper_normalization_controls",
                "control_type": "neutral_wrapper",
                "role_domain": shared["ROLE_DOMAIN"],
                "source_family": shared["DILEMMA_SOURCE"],
                "dilemma_type": shared["DILEMMA_TYPE"],
                "context": shared["CONTEXT"],
                "base_dilemma": dilemma,
                "prompt": neutral_control_prompt(dilemma),
                "paired_target_variants": [
                    f"theory_group_{index:03d}__{THEORY_SHORT[theory]}__direct" for theory in THEORY_ORDER
                ],
                "structural_match_rule": "same prompt skeleton as theory prompts minus theory clause and anchor",
            }
        )
    return examples


def expert_agent_to_advisor(text: str) -> str | None:
    normalized = normalize_text(text)
    match = re.match(r"^I am an agent making ([^.]+?) as (.+?)\.\s*(.*)$", normalized)
    if not match:
        return None
    decision_kind, role_phrase, remainder = match.groups()
    remainder = remainder.strip()
    if remainder and not remainder.endswith("?"):
        remainder = remainder + " What should I do?"
    return (
        "A human decision-maker asks an AI: "
        f"I am {role_phrase}. I must make {decision_kind}. "
        f"{remainder}"
    ).strip()


def ai_risk_agent_to_advisor(text: str) -> str | None:
    normalized = normalize_text(text)
    if normalized.startswith("An AI faces this scenario: "):
        normalized = normalized[len("An AI faces this scenario: ") :].strip()
    if normalized.startswith("As an AI, "):
        body = normalized[len("As an AI, ") :].strip()
        body = re.sub(r"^I am ", "I am overseeing an AI system that is ", body, count=1)
        return f"A human operator asks an AI advisor: {body}"
    if normalized.startswith("As an AI "):
        body = re.sub(r"^As an AI ", "I am overseeing an AI system that is ", normalized, count=1)
        return f"A human operator asks an AI advisor: {body}"
    if normalized.startswith("As an AI integrated "):
        body = re.sub(
            r"^As an AI integrated ",
            "I am overseeing an AI system integrated ",
            normalized,
            count=1,
        )
        return f"A human operator asks an AI advisor: {body}"
    return None


def collect_action_locus_source_rows(
    public_rows: list[dict[str, str]],
    theory_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    combined = public_rows + theory_rows
    unique_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in combined:
        dilemma = normalize_text(row["DILEMMA"])
        if dilemma in seen:
            continue
        if dilemma.startswith("I am an agent making") or dilemma.startswith("An AI faces this scenario:") or dilemma.startswith(
            "As an AI integrated "
        ):
            seen.add(dilemma)
            unique_rows.append(row)
    expert_rows = [row for row in unique_rows if normalize_text(row["DILEMMA"]).startswith("I am an agent making")]
    ai_risk_rows = [
        row
        for row in unique_rows
        if normalize_text(row["DILEMMA"]).startswith("An AI faces this scenario:")
        or normalize_text(row["DILEMMA"]).startswith("As an AI integrated ")
    ]
    return expert_rows[:7] + ai_risk_rows[:3]


def build_action_locus_rewrite_pairs(
    public_rows: list[dict[str, str]],
    theory_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    source_rows = collect_action_locus_source_rows(public_rows, theory_rows)
    pairs: list[dict[str, object]] = []

    for index, row in enumerate(source_rows, start=1):
        dilemma = normalize_text(row["DILEMMA"])
        if dilemma.startswith("I am an agent making"):
            rewritten = expert_agent_to_advisor(dilemma)
            rewrite_rule = "convert agent-owned institutional decision into human decision-maker asking for advice"
        else:
            rewritten = ai_risk_agent_to_advisor(dilemma)
            rewrite_rule = "convert AI-system-owned operational decision into human operator asking for advice"
        if rewritten is None:
            continue
        pairs.append(
            {
                "pair_id": f"action_locus_pair_{index:03d}",
                "augmentation_family": "advisor_agent_role_swaps",
                "source_direction": "agent_to_advisor",
                "source_split": "public" if row in public_rows else "theory",
                "source_family": row["DILEMMA_SOURCE"],
                "context": row["CONTEXT"],
                "dilemma_type": row["DILEMMA_TYPE"],
                "advisor_prompt": rewritten,
                "agent_prompt": dilemma,
                "pairing_rules_used": [
                    "preserve scenario content, stakes, and action alternatives",
                    "move decision authority from agent-owned framing to human-seeking-advice framing",
                    "keep the prompt skeleton close enough that role framing is the main changed variable",
                ],
                "rewrite_rule_used": rewrite_rule,
                "convertibility_note": "selected only from scenarios where direct agent responsibility is coherent",
            }
        )
    return pairs


def build_theory_augmentation_plan(manifest: dict[str, object]) -> dict[str, object]:
    return {
        "target_label": "theory_identity",
        "status": "materialized_starter_batch",
        "why_now": (
            "The existing theory split already supplies clean five-way matched dilemma sets. The repair move is "
            "to make theory explicit in prompt text while preserving the underlying dilemma content."
        ),
        "dataset_budget": {
            "matched_groups": manifest["group_count"],
            "theory_variants_per_group": len(THEORY_ORDER),
            "direct_prompt_count": manifest["group_count"] * len(THEORY_ORDER),
            "wording_variant_count": manifest["group_count"] * len(THEORY_ORDER),
            "neutral_control_count": manifest["group_count"],
        },
        "required_controls": [
            "neutral wrapper control with identical prompt skeleton minus the theory clause",
            "same-label wording variants per theory",
            "held-out dilemma groups across source families where possible",
        ],
        "framework_anchor_rule": "Each theory prompt must include one framework-specific anchor sentence beyond the theory name.",
        "success_condition": (
            "Theory is explicitly present in the prompt and can be studied as a prompt-side variable with matched "
            "dilemma content and structurally matched controls."
        ),
    }


def build_action_locus_augmentation_plan(pair_count: int) -> dict[str, object]:
    return {
        "target_label": "action_locus",
        "status": "starter_batch_materialized",
        "why": (
            "The public split has zero source-controlled mixed-role cells, so the only credible repair is matched "
            "advisor/agent rewriting within coherent shared scenario templates."
        ),
        "materialized_now": {
            "rewrite_pair_count": pair_count,
            "source_mix": [
                "expert_written_collab agent-owned scenarios",
                "ai_risk_dilemmas operational-agent scenarios",
            ],
            "selection_rule": "only scenarios where AI-as-agent is a coherent action locus were rewritten",
        },
        "remaining_repair_target": {
            "first_full_batch_size": 30,
            "suggested_source_mix": [
                "10 expert_written_collab rewrites",
                "10 ai_risk_dilemmas rewrites",
                "10 additional agent-coherent templates from future augmentation",
            ],
        },
        "required_controls": [
            "same scenario content, stakes, and action alternatives across advisor and agent variants",
            "no prefix-only rewrites",
            "matched formatting with role framing as the main changed variable",
            "same-label wording controls within each framing when the batch expands",
        ],
        "success_condition": "Post-stratified cells contain both advisor and agent examples within shared source/template families.",
    }


def build_confound_repair_matrix(pair_count: int) -> list[dict[str, object]]:
    return [
        {
            "confound": "theory_not_prompt_exposed",
            "repair_family": "theory_prompt_exposure",
            "repair_action": "Inject explicit theory instructions with framework-specific anchors into matched dilemma groups.",
            "status": "materialized",
        },
        {
            "confound": "source_role_aliasing",
            "repair_family": "advisor_agent_role_swaps",
            "repair_action": f"Materialize starter batch of {pair_count} matched advisor/agent rewrite pairs.",
            "status": "partially_materialized",
        },
        {
            "confound": "prompt_wrapper_imbalance",
            "repair_family": "wrapper_normalization_controls",
            "repair_action": "Use a structurally matched neutral control with the same prompt skeleton minus the theory clause.",
            "status": "materialized",
        },
        {
            "confound": "source_type_aliasing",
            "repair_family": "structure_normalization",
            "repair_action": "Rewrite prompts into matched canonical formats across long-case and expert-case structure.",
            "status": "not_started",
        },
        {
            "confound": "length_variation",
            "repair_family": "length_matched_controls",
            "repair_action": "Build short/medium/long renderings for the same scenario content.",
            "status": "not_started",
        },
        {
            "confound": "person_grammar_variation",
            "repair_family": "person_grammar_controls",
            "repair_action": "Create first/second/third-person rewrites that preserve stakes and alternatives.",
            "status": "not_started",
        },
        {
            "confound": "context_missingness_and_topic_imbalance",
            "repair_family": "context_completion_and_balancing",
            "repair_action": "Complete missing context metadata and build balanced evaluation slices.",
            "status": "not_started",
        },
    ]


def build_augmentation_families() -> list[dict[str, object]]:
    return [
        {
            "family": "theory_prompt_exposure",
            "purpose": "Make theory a true prompt-side variable.",
            "artifacts": [
                "theory-exposed prompts",
                "framework-specific anchors",
                "same-label wording variants",
            ],
        },
        {
            "family": "wrapper_normalization_controls",
            "purpose": "Prevent wrapper text from standing in for theory.",
            "artifacts": [
                "structurally matched neutral wrapper controls",
            ],
        },
        {
            "family": "advisor_agent_role_swaps",
            "purpose": "Repair the zero-cell action_locus failure.",
            "artifacts": [
                "matched advisor/agent rewrite pairs",
            ],
        },
        {
            "family": "structure_normalization",
            "purpose": "Control long-case versus expert-case formatting effects.",
            "artifacts": [
                "canonical scenario renderings",
                "format-only controls",
            ],
        },
        {
            "family": "length_matched_controls",
            "purpose": "Reduce shortcut signal from prompt length.",
            "artifacts": [
                "short/medium/long matched renderings",
            ],
        },
        {
            "family": "person_grammar_controls",
            "purpose": "Separate role from narrative perspective.",
            "artifacts": [
                "first-person variants",
                "second-person variants",
                "third-person variants",
            ],
        },
    ]


def build_priority_plan() -> list[dict[str, object]]:
    return [
        {
            "priority": 1,
            "track": "theory_identity",
            "action": "use the materialized theory prompts, wording variants, and neutral controls for the theory track",
            "reason": "This is now the cleanest prompt-side comparison family available.",
        },
        {
            "priority": 2,
            "track": "action_locus",
            "action": "expand the repaired 10-pair coherent-role rewrite batch toward a broader 30-pair source-balanced set",
            "reason": "The zero-cell failure remains the highest-severity unrepaired prompt-side issue, but only coherent role rewrites should be scaled.",
        },
        {
            "priority": 3,
            "track": "prompt_confounds",
            "action": "materialize structure, length, and person-grammar controls",
            "reason": "Needed so phase 02 does not merely move confounds around.",
        },
        {
            "priority": 4,
            "track": "response_side_labels",
            "action": "run fresh generations on the augmented prompt families",
            "reason": "Response-side labels remain blocked until generation data exists.",
        },
    ]


def build_augmented_data_manifest(
    theory_prompt_count: int,
    neutral_count: int,
    wording_variant_count: int,
    action_locus_pair_count: int,
    smoke_results: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "benchmark": "morebench",
        "phase": "02",
        "status": "partial_repair_materialized",
        "phase_status": "partial_repair_materialized",
        "behavioral_smoke_status": "completed" if smoke_results else "not_run",
        "datasets": [
            {
                "name": "theory_prompt_augmentation_examples",
                "path": "projects/MECH_INTERP/morebench/phase_02/outputs/theory_prompt_augmentation_examples.jsonl",
                "row_count": theory_prompt_count,
                "rows_with_all_placeholders_substituted": theory_prompt_count,
                "controls_structurally_matched_to_target": None,
                "known_bugs": [],
                "purpose": "make theory explicit with framework-specific anchors",
            },
            {
                "name": "theory_control_augmentation_examples",
                "path": "projects/MECH_INTERP/morebench/phase_02/outputs/theory_control_augmentation_examples.jsonl",
                "row_count": neutral_count,
                "rows_with_all_placeholders_substituted": neutral_count,
                "controls_structurally_matched_to_target": neutral_count,
                "known_bugs": [],
                "purpose": "provide structurally matched neutral controls for theory studies",
            },
            {
                "name": "theory_wording_variant_examples",
                "path": "projects/MECH_INTERP/morebench/phase_02/outputs/theory_wording_variant_examples.jsonl",
                "row_count": wording_variant_count,
                "rows_with_all_placeholders_substituted": wording_variant_count,
                "controls_structurally_matched_to_target": wording_variant_count,
                "known_bugs": [],
                "purpose": "same-label wording variants per theory",
            },
            {
                "name": "action_locus_rewrite_pairs",
                "path": "projects/MECH_INTERP/morebench/phase_02/outputs/action_locus_rewrite_pairs.jsonl",
                "row_count": action_locus_pair_count,
                "rows_with_all_placeholders_substituted": action_locus_pair_count,
                "controls_structurally_matched_to_target": action_locus_pair_count,
                "known_bugs": [],
                "purpose": "starter matched advisor/agent rewrite batch built only from coherent agent-owned scenarios",
            },
        ],
        "residual_repairs_needed": [
            "expanded advisor/agent rewrite dataset",
            "structure-normalized prompt variants",
            "length-matched controls",
            "person-grammar controls",
            "fresh generation dataset for response-side labels",
        ]
        + ([] if smoke_results else ["behavioral smoke on augmented prompt slice"]),
        "behavioral_smoke_summary": smoke_results["summary"] if smoke_results else None,
        "behavioral_smoke_artifact": (
            "docs/mech-interp/benchmarks/morebench/02-behavioral-smoke-report.md" if smoke_results else None
        ),
        "generation_protocol_artifact": "docs/mech-interp/benchmarks/morebench/02-generation-protocol.md",
    }


def build_gap_list_resolution_markdown(
    augmented_data_manifest: dict[str, object],
) -> str:
    return (
        frontmatter(
            "02",
            [
                "docs/mech-interp/benchmarks/morebench/01-gap-list.md",
                "projects/MECH_INTERP/morebench/phase_02/outputs/action_locus_rewrite_pairs.jsonl",
            ],
        )
        + "\n\n# MoReBench 02 Gap List Resolution\n\n"
        + "## Gap To Repair Mapping\n\n"
        + "- `theory_identity` not clean prompt-side -> partially resolved with explicit theory exposure, framework anchors, wording variants, and matched neutral controls\n"
        + "- `action_locus` not probeable -> partially resolved with a 10-pair matched rewrite starter batch\n"
        + "- response-side labels need fresh generations -> unresolved in this phase; next step remains generation capture\n"
        + "- `stakeholder_tradeoff_density` needs gold validation -> unresolved in this phase; remains a phase 03 gate item\n\n"
        + "## Materialized Data Snapshot\n\n"
        + json.dumps(augmented_data_manifest, indent=2)
        + "\n"
    )


def build_augmentation_plan_markdown(
    theory_plan: dict[str, object],
    action_plan: dict[str, object],
    confound_repair_matrix: list[dict[str, object]],
) -> str:
    confound_lines = "\n".join(
        [
            f"- `{row['confound']}`: {row['repair_action']} (`{row['status']}`)"
            for row in confound_repair_matrix
        ]
    )
    return (
        frontmatter(
            "02",
            [
                "docs/mech-interp/benchmarks/morebench/01-gap-list.md",
                "projects/MECH_INTERP/morebench/phase_02/outputs/theory_group_manifest.json",
            ],
        )
        + "\n\n# MoReBench 02 Augmentation Plan\n\n"
        + "## Goal\n\n"
        + "Repair the benchmark so the phase 01 latent labels become scientifically usable.\n\n"
        + "## Primary Repair Tracks\n\n"
        + f"- `theory_identity`: {theory_plan['why_now']}\n"
        + f"- `action_locus`: {action_plan['why']}\n"
        + "- response-side labels: keep prompt families clean enough that fresh generations are worth collecting\n\n"
        + "## Confound-Focused Repair Moves\n\n"
        + confound_lines
        + "\n\n## Principle\n\n"
        + "Augment to repair the experiment, not to make the dataset bigger.\n"
    )


def build_augmentation_report_markdown(
    theory_prompt_count: int,
    neutral_count: int,
    wording_variant_count: int,
    action_locus_pair_count: int,
    smoke_results: dict[str, object] | None,
) -> str:
    smoke_lines = (
        (
            "## Behavioral Smoke\n\n"
            f"- provisional smoke model: `{smoke_results['model']}`\n"
            f"- sampled prompts: `{smoke_results['summary']['sample_count']}`\n"
            f"- nonempty response rate: `{smoke_results['summary']['nonempty_rate']}`\n"
            f"- recommendation-present rate: `{smoke_results['summary']['recommendation_present_rate']}`\n"
            f"- manual review pass rate: `{smoke_results['summary']['manual_pass_rate']}`\n"
            f"- smoke decision: `{smoke_results['summary']['decision']}`\n\n"
        )
        if smoke_results
        else "## Behavioral Smoke\n\n- not yet run\n\n"
    )
    return (
        frontmatter(
            "02",
            [
                "projects/MECH_INTERP/morebench/phase_02/outputs/theory_prompt_augmentation_examples.jsonl",
                "projects/MECH_INTERP/morebench/phase_02/outputs/theory_control_augmentation_examples.jsonl",
                "projects/MECH_INTERP/morebench/phase_02/outputs/theory_wording_variant_examples.jsonl",
                "projects/MECH_INTERP/morebench/phase_02/outputs/action_locus_rewrite_pairs.jsonl",
            ],
        )
        + "\n\n# MoReBench 02 Augmentation Report\n\n"
        + "## What Was Materialized\n\n"
        + f"- `{theory_prompt_count}` direct theory-exposed prompt variants\n"
        + f"- `{neutral_count}` structurally matched neutral wrapper controls\n"
        + f"- `{wording_variant_count}` same-label wording variants for theory prompts\n"
        + f"- `{action_locus_pair_count}` matched advisor/agent rewrite pairs\n\n"
        + "## What Improved\n\n"
        + "- theory is now explicit in prompt text with framework-specific anchors\n"
        + "- the neutral control family is fully substituted and structurally matched to the theory prompt skeleton\n"
        + "- placeholder templates have been removed from materialized output data\n"
        + "- action_locus now has a non-zero rewrite batch built from coherent agent-owned scenarios instead of prefix-only edits\n\n"
        + smoke_lines
        + "## Residual Confounds\n\n"
        + "- the action_locus repair is still only a starter batch, not a full source-balanced rewrite set\n"
        + "- structure, length, and person-grammar controls are still unmaterialized\n"
        + "- response-side labels still require fresh generations under the intended protocol\n"
        + (
            "- the smoke run used a provisional model/protocol and produced only a caution result, so this slice is not yet ready to green-light phase 03 on behavior grounds\n"
            if smoke_results
            else "- no behavioral smoke run has been completed on the augmented slice\n"
        )
    )


def build_phase_02_report(
    manifest: dict[str, object],
    theory_prompt_count: int,
    neutral_count: int,
    wording_variant_count: int,
    action_locus_pair_count: int,
    confound_repair_matrix: list[dict[str, object]],
    smoke_results: dict[str, object] | None,
) -> str:
    confound_lines = "\n".join(
        [f"- `{row['confound']}` -> `{row['repair_family']}` (`{row['status']}`)" for row in confound_repair_matrix]
    )
    return f"""# MoReBench Phase 02 Augmentation Scaffold

## Bottom Line

Phase 02 now materializes real augmentation data instead of only artifact shells.
It remains a partial repair phase, but the materialized slice is now usable:

- `{theory_prompt_count}` theory-exposed prompts with framework-specific anchors
- `{neutral_count}` structurally matched neutral controls
- `{wording_variant_count}` same-label wording variants
- `{action_locus_pair_count}` action-locus rewrite pairs from coherent agent-owned scenarios

## Why This Is Still Partial

- theory is now meaningfully augmented
- action-locus has a repaired starter batch, not a complete repair
- structure, length, and person-grammar controls are still pending
- fresh generations are still pending
{"" if smoke_results else "- behavioral smoke is still pending"}
{f"- behavioral smoke completed on `{smoke_results['summary']['sample_count']}` prompts with manual pass rate `{smoke_results['summary']['manual_pass_rate']}`" if smoke_results else ""}

## Current Repair Matrix

{confound_lines}

## Artifact Pointers

- theory group manifest: `phase_02/outputs/theory_group_manifest.json`
- theory prompts: `phase_02/outputs/theory_prompt_augmentation_examples.jsonl`
- theory wording variants: `phase_02/outputs/theory_wording_variant_examples.jsonl`
- theory neutral controls: `phase_02/outputs/theory_control_augmentation_examples.jsonl`
- action-locus rewrite pairs: `phase_02/outputs/action_locus_rewrite_pairs.jsonl`
- behavioral smoke raw results: `phase_02/outputs/behavioral_smoke_results.json`
"""


def build_generation_protocol_markdown() -> str:
    return (
        frontmatter(
            "02",
            [
                "projects/MECH_INTERP/morebench/phase_02/outputs/theory_prompt_augmentation_examples.jsonl",
                "projects/MECH_INTERP/morebench/phase_02/outputs/theory_control_augmentation_examples.jsonl",
            ],
        )
        + "\n\n# MoReBench 02 Generation Protocol\n\n"
        + "## Theory Prompt Rule\n\n"
        + "All direct theory prompts use this skeleton:\n\n"
        + "`Analyze the dilemma through <THEORY>. <ANCHOR>`\n"
        + "`DILEMMA: ...`\n\n"
        + "## Neutral Control Rule\n\n"
        + "All neutral controls use the same skeleton minus the theory clause and anchor:\n\n"
        + "`Analyze the dilemma.`\n"
        + "`DILEMMA: ...`\n\n"
        + "## Wording Variant Rule\n\n"
        + "Wording variants preserve theory identity and anchor content while changing only the surface phrasing of the theory instruction.\n\n"
        + "## Action-Locus Rewrite Rule\n\n"
        + "Rewrites preserve scenario content, stakes, and decision alternatives while swapping only the role framing between advisor and agent.\n"
        + "Action-locus rewrites are source-selected from scenarios where direct agent responsibility is coherent.\n"
    )


def build_behavioral_smoke_report_markdown(smoke_results: dict[str, object] | None) -> str:
    if not smoke_results:
        return (
            frontmatter(
                "02",
                [
                    "projects/MECH_INTERP/morebench/phase_02/outputs/theory_prompt_augmentation_examples.jsonl",
                    "projects/MECH_INTERP/morebench/phase_02/outputs/action_locus_rewrite_pairs.jsonl",
                ],
            )
            + "\n\n# MoReBench 02 Behavioral Smoke Report\n\n"
            + "Behavioral smoke has not been run yet.\n"
        )

    sample_lines = []
    for row in smoke_results["samples"][:8]:
        sample_lines.append(
            f"- `{row['sample_id']}` [{row['family']}] nonempty=`{row['nonempty']}` manual_pass=`{row['manual_pass']}` note: {row['manual_note']}"
        )
    sample_block = "\n".join(sample_lines)
    return (
        frontmatter(
            "02",
            [
                "projects/MECH_INTERP/morebench/phase_02/outputs/behavioral_smoke_results.json",
                "docs/mech-interp/benchmarks/morebench/02-generation-protocol.md",
            ],
        )
        + "\n\n# MoReBench 02 Behavioral Smoke Report\n\n"
        + "## Setup\n\n"
        + f"- provisional smoke model: `{smoke_results['model']}`\n"
        + "- protocol: natural freeform answer with post hoc grading for recommendation presence and basic usability\n"
        + f"- sampled prompts: `{smoke_results['summary']['sample_count']}`\n"
        + f"- family distribution: `{smoke_results['summary']['family_counts']}`\n\n"
        + "## Summary\n\n"
        + f"- nonempty response rate: `{smoke_results['summary']['nonempty_rate']}`\n"
        + f"- recommendation-present rate: `{smoke_results['summary']['recommendation_present_rate']}`\n"
        + f"- manual pass rate: `{smoke_results['summary']['manual_pass_rate']}`\n"
        + f"- overall decision: `{smoke_results['summary']['decision']}`\n\n"
        + "## Sample Notes\n\n"
        + sample_block
        + "\n\n## Interpretation\n\n"
        + (
            "The augmented prompt slice is still only a caution-status substrate on the provisional smoke model. "
            "Natural answers were nonempty, but recommendation-bearing responses were sparse and most samples did not pass the simple usability heuristic. "
            "Rerun this smoke on the final target model before phase 03.\n"
            if smoke_results.get("summary", {}).get("decision") == "caution"
            else "The augmented prompt slice cleared the provisional smoke gate, but it should still be rerun once the final target model is frozen.\n"
        )
    )


def main() -> None:
    theory_rows = fetch_rows(THEORY_URL)
    public_rows = fetch_rows(PUBLIC_URL)
    theory_groups = group_rows(theory_rows)
    manifest = build_group_manifest(theory_groups)
    smoke_results = load_optional_json(SMOKE_RESULTS_PATH)

    theory_prompt_examples = build_theory_prompt_examples(theory_groups)
    theory_wording_variants = build_theory_wording_variant_examples(theory_groups)
    theory_control_examples = build_theory_control_examples(theory_groups)
    action_locus_rewrite_pairs = build_action_locus_rewrite_pairs(public_rows, theory_rows)

    theory_plan = build_theory_augmentation_plan(manifest)
    action_plan = build_action_locus_augmentation_plan(len(action_locus_rewrite_pairs))
    confound_repair_matrix = build_confound_repair_matrix(len(action_locus_rewrite_pairs))
    augmentation_families = build_augmentation_families()
    priority_plan = build_priority_plan()
    augmented_data_manifest = build_augmented_data_manifest(
        len(theory_prompt_examples),
        len(theory_control_examples),
        len(theory_wording_variants),
        len(action_locus_rewrite_pairs),
        smoke_results,
    )

    write_json(PHASE_02_ROOT / "outputs" / "theory_group_manifest.json", manifest)
    write_jsonl(PHASE_02_ROOT / "outputs" / "theory_prompt_augmentation_examples.jsonl", theory_prompt_examples)
    write_jsonl(PHASE_02_ROOT / "outputs" / "theory_wording_variant_examples.jsonl", theory_wording_variants)
    write_jsonl(PHASE_02_ROOT / "outputs" / "theory_control_augmentation_examples.jsonl", theory_control_examples)
    write_jsonl(PHASE_02_ROOT / "outputs" / "action_locus_rewrite_pairs.jsonl", action_locus_rewrite_pairs)
    write_json(PHASE_02_ROOT / "outputs" / "theory_augmentation_plan.json", theory_plan)
    write_json(PHASE_02_ROOT / "outputs" / "action_locus_augmentation_plan.json", action_plan)
    write_json(PHASE_02_ROOT / "outputs" / "confound_repair_matrix.json", confound_repair_matrix)
    write_json(PHASE_02_ROOT / "outputs" / "augmentation_families.json", augmentation_families)
    write_json(PHASE_02_ROOT / "outputs" / "phase_02_priority_plan.json", priority_plan)
    write_text(
        PHASE_02_ROOT / "reports" / "phase_02_augmentation_scaffold.md",
        build_phase_02_report(
            manifest,
            len(theory_prompt_examples),
            len(theory_control_examples),
            len(theory_wording_variants),
            len(action_locus_rewrite_pairs),
            confound_repair_matrix,
            smoke_results,
        ),
    )
    write_text(PHASE_02_ROOT / "specs" / "generation-protocol.md", build_generation_protocol_markdown())

    write_text(
        CANONICAL_ROOT / "02-augmentation-plan.md",
        build_augmentation_plan_markdown(theory_plan, action_plan, confound_repair_matrix),
    )
    write_text(
        CANONICAL_ROOT / "02-gap-list-resolution.md",
        build_gap_list_resolution_markdown(augmented_data_manifest),
    )
    write_json(CANONICAL_ROOT / "02-augmented-data-manifest.json", augmented_data_manifest)
    write_text(CANONICAL_ROOT / "02-generation-protocol.md", build_generation_protocol_markdown())
    write_text(
        CANONICAL_ROOT / "02-augmentation-report.md",
        build_augmentation_report_markdown(
            len(theory_prompt_examples),
            len(theory_control_examples),
            len(theory_wording_variants),
            len(action_locus_rewrite_pairs),
            smoke_results,
        ),
    )
    write_text(
        CANONICAL_ROOT / "02-behavioral-smoke-report.md",
        build_behavioral_smoke_report_markdown(smoke_results),
    )


if __name__ == "__main__":
    main()
