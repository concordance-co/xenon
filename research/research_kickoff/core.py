from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from pipelines.ingest.manifest import CohortRule, ManifestPlan


@dataclass(slots=True)
class RoadmapTrack:
    rank: int
    title: str
    score: float
    why_now: str
    key_hypotheses: list[str]
    first_experiments: list[str]
    success_criteria: list[str]
    deprioritize: list[str]


def ranked_research_tracks() -> list[RoadmapTrack]:
    return [
        RoadmapTrack(
            rank=1,
            title="Blocked Valence + Settings Twist",
            score=9.7,
            why_now=(
                "This is the shortest path from localization results to an actual decision "
                "mechanism: what the model wants to do, what stops it, and whether settings "
                "reweight that preference or merely gate execution."
            ),
            key_hypotheses=[
                "Early preference, late permission: market preference forms before downstream legality gating.",
                "Observe contains hidden bullish and bearish states, not just neutral non-action.",
                "Settings reweight an existing preference space rather than creating preference from scratch.",
            ],
            first_experiments=[
                "Build blocked-observe rerun cohorts stratified by block reason and actionability regime.",
                "Build settings-tension cohorts with legal action available but extreme settings pressure.",
                "Run deconstraint and settings-rewrite reruns on those cohorts and compare downstream states.",
            ],
            success_criteria=[
                "Blocked reruns reveal stable bullish/bearish latent preference on a nontrivial fraction of observe cases.",
                "Settings rewrites produce mostly parallel drift in downstream states rather than total replacement.",
                "Asset-conditioned valence becomes more predictable than raw buy/sell/observe labels.",
            ],
            deprioritize=[
                "Broad manifold searching without a behavioral decomposition.",
                "More generic buy/sell probes without blocked-valence labels.",
            ],
        ),
        RoadmapTrack(
            rank=2,
            title="Causal Necessity of Market Variables",
            score=9.2,
            why_now=(
                "The strongest synthetic variables are now clear enough that the next gain comes "
                "from causal tests, not prettier geometry plots."
            ),
            key_hypotheses=[
                "`pct_5m` and momentum × flow are not just decodable; they are necessary inputs to preference formation.",
                "Participation acts as a confidence modulator rather than a primary axis.",
                "Concentration matters later or through policy, not as a clean early perceptual variable.",
            ],
            first_experiments=[
                "Patch or ablate row states along the strongest scalar and coupled directions.",
                "Run rank-preserving and magnitude-preserving corruptions on synthetic market rows.",
                "Measure whether best-asset and pairwise preference collapse in predictable ways.",
            ],
            success_criteria=[
                "Causal interventions shift preference in the predicted direction.",
                "Momentum × flow perturbations outperform matched control directions.",
            ],
            deprioritize=[
                "More passive decoding work on the same synthetic slices.",
            ],
        ),
        RoadmapTrack(
            rank=3,
            title="Real-Data Decision Decomposition",
            score=8.8,
            why_now=(
                "The synthetic work is useful only if it transfers back to DX-terminal-style prompts. "
                "This track reconnects the clean synthetic factors to real decision traces."
            ),
            key_hypotheses=[
                "Asset-conditioned valence transfers better than pooled buy/sell probes.",
                "Real observe cases split into neutral, blocked bullish, and blocked bearish subtypes.",
                "Late sections sharpen actionability more than raw preference.",
            ],
            first_experiments=[
                "Train asset-valence probes on rerun-labeled blocked cases and validate on held-out real trades.",
                "Compare row_mean_i versus active_settings_eos / constraints_eos for blocked-valence decoding.",
                "Measure how much real decision performance survives after regressing out simple market heuristics.",
            ],
            success_criteria=[
                "Blocked-valence labels improve transfer to real action outcomes.",
                "Preference and legality become separable in downstream states.",
            ],
            deprioritize=[
                "Treating observe as a uniformly neutral class.",
            ],
        ),
        RoadmapTrack(
            rank=4,
            title="Policy vs Perception Routing",
            score=7.9,
            why_now=(
                "If experts split along market perception versus policy gating, routing may provide a cleaner "
                "mechanistic handle than more generic residual-stream probing."
            ),
            key_hypotheses=[
                "Some experts specialize in market parsing, others in policy or affordance handling.",
                "Settings-twist effects should be concentrated in a narrower routing subset than raw market reading.",
            ],
            first_experiments=[
                "Compare router specialization across market-only, settings, and blocked-observe cohorts.",
                "Test whether blocked-valence reruns shift routing more than row-level market perception.",
            ],
            success_criteria=[
                "Routing clusters separate perception and policy contexts in a stable way.",
            ],
            deprioritize=[
                "Assuming routing is just an implementation detail.",
            ],
        ),
        RoadmapTrack(
            rank=5,
            title="Geometry Support Track",
            score=6.6,
            why_now=(
                "Geometry remains useful as a simplifying lens, but it should support the main behavioral "
                "and causal questions rather than define them."
            ),
            key_hypotheses=[
                "A few variables admit low-dimensional geometric structure.",
                "The joint market state is composed from several simpler pieces rather than one universal manifold.",
            ],
            first_experiments=[
                "Use geometry to simplify interpretation of the strongest causal variables.",
                "Stop expanding geometry breadth until the top behavioral hypotheses are tested.",
            ],
            success_criteria=[
                "Geometry clarifies or compresses a causal story rather than replacing one.",
            ],
            deprioritize=[
                "Manifold-first broad search as the main scientific program.",
            ],
        ),
    ]


def roadmap_as_dicts() -> list[dict[str, Any]]:
    return [asdict(track) for track in ranked_research_tracks()]


def settings_signature(row: dict[str, Any]) -> str:
    return "/".join(
        str(int(row.get(field) or 0))
        for field in (
            "trade_size",
            "trading_activity",
            "holding_style",
            "diversification",
            "risk_preference",
        )
    )


def risk_activity_cell(row: dict[str, Any]) -> str:
    risk = int(row.get("risk_preference") or 0)
    activity = int(row.get("trading_activity") or 0)
    return f"R{risk}:A{activity}"


def actionability_cell(row: dict[str, Any]) -> str:
    can_buy = bool(row.get("can_buy_any"))
    can_sell = bool(row.get("can_sell_any"))
    if can_buy and can_sell:
        return "buy+sell"
    if can_buy:
        return "buy_only"
    if can_sell:
        return "sell_only"
    return "none"


def annotate_kickoff_row(row: dict[str, Any], *, cohort_label: str) -> dict[str, Any]:
    out = dict(row)
    out["settings_signature"] = settings_signature(out)
    out["risk_activity_cell"] = risk_activity_cell(out)
    out["actionability_cell"] = actionability_cell(out)
    out["cohort_label"] = cohort_label
    return out


def research_kickoff_manifest_plan() -> ManifestPlan:
    return ManifestPlan(
        manifest_name="research_kickoff_v1",
        per_vault_cap=2,
        min_spacing_minutes=30,
        cohort_rules=[
            CohortRule(
                label="blocked_observe",
                target_count=72,
                group_field="block_reason",
                max_per_group=18,
                max_per_vault=1,
            ),
            CohortRule(
                label="policy_tension_observe",
                target_count=72,
                group_field="settings_signature",
                max_per_group=10,
                max_per_vault=1,
            ),
            CohortRule(
                label="buy",
                target_count=36,
                group_field="target_asset",
                max_per_group=12,
                max_per_asset=12,
                max_per_vault=1,
            ),
            CohortRule(
                label="sell",
                target_count=36,
                group_field="target_asset",
                max_per_group=12,
                max_per_asset=12,
                max_per_vault=1,
            ),
        ],
    )


def blocked_valence_manifest_plan() -> ManifestPlan:
    return ManifestPlan(
        manifest_name="blocked_valence_kickoff_v1",
        per_vault_cap=1,
        min_spacing_minutes=15,
        cohort_rules=[
            CohortRule(
                label="blocked_observe",
                target_count=48,
                group_field="block_reason",
                max_per_group=24,
                max_per_vault=1,
            ),
        ],
    )


def settings_twist_manifest_plan() -> ManifestPlan:
    return ManifestPlan(
        manifest_name="settings_twist_kickoff_v1",
        per_vault_cap=2,
        min_spacing_minutes=15,
        cohort_rules=[
            CohortRule(
                label="policy_tension_observe",
                target_count=72,
                group_field="risk_activity_cell",
                max_per_group=36,
                max_per_vault=1,
            ),
            CohortRule(
                label="buy",
                target_count=24,
                group_field="target_asset",
                max_per_group=8,
                max_per_asset=8,
                max_per_vault=1,
            ),
            CohortRule(
                label="sell",
                target_count=24,
                group_field="target_asset",
                max_per_group=8,
                max_per_asset=8,
                max_per_vault=1,
            ),
        ],
    )
