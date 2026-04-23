#set page(
  paper: "us-letter",
  margin: (x: 0.9in, y: 0.85in),
  numbering: "1 / 1",
)
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.6em)
#set heading(numbering: "1.")

#show heading.where(level: 1): it => [
  #v(0.6em)
  #set text(size: 15pt, weight: "bold")
  #block(it.body)
  #v(0.2em)
]
#show heading.where(level: 2): it => [
  #v(0.4em)
  #set text(size: 12pt, weight: "bold")
  #block(it.body)
]
#show heading.where(level: 3): it => [
  #v(0.2em)
  #set text(size: 11pt, weight: "bold", style: "italic")
  #block(it.body)
]

#align(center)[
  #text(size: 18pt, weight: "bold")[
    Phase 06 / v4 Report \
    Prompt-Level Conflict Detection in Qwen3-30B-A3B
  ]
  #v(0.4em)
  #text(size: 10pt)[
    DX Terminal / prompt_confusion / phase 06 \
    2026-04-15
  ]
]

#v(1em)

#align(center)[
  #block(width: 90%, inset: 0.6em, stroke: (left: 1.5pt + rgb("#333")))[
    #align(left)[
      #text(weight: "bold")[Dataset.] `conflict_probe_examples_v4` (Neon), 768 rows,
      size-axis only. #text(weight: "bold")[Model.] Qwen/Qwen3-30B-A3B (vLLM,
      `resid_post` at last prompt token, 12 captured layers). #text(weight: "bold")[Run.]
      `wr_cc10418ff064_f8b538db`. Capture on Modal A100-80GB, analyses on Modal CPU,
      reporting local. End-to-end wall clock: \~10 min.
    ]
  ]
]

#v(0.6em)

= Abstract

We rebuilt the prompt-confusion dataset around a single conflict axis
(trade-size directives), with four lexical variants each for STRATEGY
and SETTINGS wordings and a 50/50 STRATEGY-first / SETTINGS-first order
swap. Against this dataset, a linear probe on last-token residual
activations decodes `conflict_present` at balanced accuracy $>= 0.80$
under every holdout condition tested, including a strict both-axes
lexical holdout, while CountVectorizer + LogisticRegression on the raw
prompt text is at chance. The probe signal shows a classic
constructed-feature depth profile --- chance at L0, rising through
L16--L28, plateau from L28 onward.

A behavioral audit of the model's generations surfaces a separate and
important finding: the model's *conflict resolution* (which side it
follows) is dominated by surface wording format, not by a deeper
arbitration mechanism. Settings written as a numeric scale
(`"Trade size: N/5..."`) carry disproportionate authority compared to
verbal-only setting wordings (`"execution size can use the large tier"`),
and conditional strategy preambles interact with hedged market language
to drive \~20% of aligned rows into refusal.

The detection result is robust and non-trivial. The resolution story is
interesting as a descriptive artifact of v4 and motivates Phase 07
iteration, not a mechanistic claim in its own right.

= Dataset

`conflict_probe_examples_v4` was a rebuild of v3 with four goals:
restrict to one conflict axis (size) to eliminate the family-vocabulary
confound of Phase 05; provide 4 lexical variants per axis (strategy and
settings), up from 2 in v3; include a 50/50 STRATEGY-first /
SETTINGS-first order split; and label refusal as a first-class outcome
rather than dropping it.

Structural summary:

- *768 rows* --- 384 conflict, 384 aligned matched pairs.
- *Families (2):* `trade_size_force_large`, `trade_size_force_small`.
  On conflict rows, SETTINGS prescribes the opposite size from
  STRATEGY.
- *Strategy variants (4 per direction):* e.g., `size_small_v0..v3`.
- *Setting phrase variants (4):* `size_setting_phrase_v0..v3`. Setting
  value 1 renders as "small" wording, value 5 as "large" wording.
- *Market pressure buckets (3):* `balanced`, `strategy_favored`,
  `setting_favored`, 2 context variants per bucket.
- *Section order:* 50/50 `strategy_first` / `setting_first`.

Lexical holdouts: variants v0/v1 are train, v2/v3 are test. Combined
strict holdout tests both axes simultaneously.

Pre-capture QA ran text baselines on the generator output. All three
text-baseline conditions hit bal_acc $= 0.50$, confirming no surface
lexical leakage. Matched-pair integrity (384 pairs of 2) and per-cell
class support ($>= 30$ rows) passed.

= Detection Results

Feature: `resid_post` at last prompt token (float16) across 12 layers.
Probe: `SGDClassifier(loss="log_loss")` (`pipelines_v2.ProbeSpec`).
Target: `conflict_present` (boolean). Baseline:
`CountVectorizer(ngram 1-2) + LogisticRegression` on `user_text`.

#figure(
  image("figures/fig1_depth_sweep.png", width: 100%),
  caption: [
    Probe performance on `conflict_present` across layers for four
    holdout conditions. Dotted lines mark the corresponding text
    baseline. Dashed gray line is chance.
  ],
) <fig-depth>

#v(0.4em)

#align(center)[
#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, center),
  stroke: 0.4pt,
  table.header(
    [*Condition*], [*Text bal / AUROC*], [*Probe bal / AUROC*], [*Peak layer*],
  ),
  [Grouped-CV (no holdout)],            [0.43 / 0.39], [*0.952 / 0.978*], [L40],
  [Strategy-holdout (v0/v1 → v2/v3)],   [0.50 / 0.50], [*0.867 / 0.941*], [L28–L40],
  [Settings-holdout (v0/v1 → v2/v3)],   [0.50 / 0.50], [*0.805 / 0.859*], [L36–L40],
  [*Combined strict holdout*],          [0.50 / 0.50], [*0.849 / 0.876*], [L40],
)
]

#v(0.4em)

== What this supports

The model builds a *linearly-decodable representation of STRATEGY /
SETTINGS directive disagreement* at the last prompt token. The
representation is *non-lexical* (text baselines at chance across all
conditions, including the strict combined holdout where both strategy
and setting wordings are unseen). It is a *constructed feature*: no
signal at L0, steady ramp through L16--L28, plateau from L28. The model
is computing the comparison, not pattern-matching.

Single-axis holdouts (0.81--0.87 peak) drop only modestly to the strict
combined holdout (0.85). Both-axis leakage is a small effect, not a
large inflation.

== What this does not support

Nothing about how the model arbitrates conflicts --- the probe target is
`conflict_present`, not resolution direction. No causal claim that the
detection representation drives downstream behavior. And no spatial
localization: capture is at last prompt token, so the signal could be
"detection," "output-prep" content correlated with detection, or any
linear combination.

= Behavioral Audit

All 768 generations parsed cleanly as JSON. Labels derived by comparing
output `size` against each row's `strategy_expected_size` and
`setting_expected_size`.

#figure(
  image("figures/fig2_behavior_by_family_pair.png", width: 95%),
  caption: [
    Behavioral outcome by (family, pair). Each column is 192 rows.
  ],
) <fig-behavior>

Overall distribution: aligned_match 298 (39%), follow_strategy 171
(22%), follow_setting 127 (17%), refuse 164 (21%), other 8 (1%).

== Refusal is structured, not random

#figure(
  image("figures/fig3_aligned_refusal_by_pressure.png", width: 65%),
  caption: [
    force_small aligned refusal rate by market-pressure context.
  ],
) <fig-refusal>

35% of force_small aligned rows refuse. These rows have STRATEGY and
SETTINGS agreeing ("both say small"), so the refusal is not about
conflict. It concentrates on the `setting_favored` market-pressure
bucket, which has language like "usable edge with some confirmation,
but one caution remains." The combination of (a) conditional strategy
wording ("#emph[when an entry is justified], use the small size tier")
and (b) explicit caution language in the market section is interpreted
as "not justified" by the model. This is real multi-source reasoning,
just on an axis different from the one we designed.

22% of conflict rows also refuse (15% force_large, 30% force_small).
The system prompt explicitly allows refusal, so this is partly an
instructed escape hatch.

== Resolution is driven by wording format, not arbitration

#figure(
  image("figures/fig4_variant_heatmap.png", width: 100%),
  caption: [
    follow_setting rate by (strategy variant $times$ setting variant)
    for conflict rows, computed over non-refuse outputs. Blue cells:
    model follows STRATEGY. Red cells: model follows SETTINGS.
  ],
) <fig-heatmap>

The family-aggregate numbers (force_large: 52% follow_setting / 33%
follow_strategy; force_small: 14% / 56%) average over variant
combinations with qualitatively different behaviors.

#block(inset: 0.4em, fill: rgb("#f4f4f4"), width: 100%)[
  *Finding 1 (numeric-scale authority).* On force_small conflict,
  `setting_v0` ("Trade size: N/5. Use the [small|large] allocation
  tier.") drives follow_setting in 22/24 non-refuse cases. The other
  three setting variants (verbal-only) drive follow_strategy in 86/92
  non-refuse cases.
]

#block(inset: 0.4em, fill: rgb("#f4f4f4"), width: 100%)[
  *Finding 2 (variant v3 is soft).* "On a clear setup, size up/down
  rather than scale down/up" is the softest strategy wording. On
  force_large conflict, strategy_v3 yields 0/46 follow_strategy across
  all four setting variants --- the wording is effectively ignored.
]

#block(inset: 0.4em, fill: rgb("#f4f4f4"), width: 100%)[
  *Finding 3 (system prompt's "SETTINGS still constrain" does not
  consistently win).* Despite explicit instruction to prefer SETTINGS,
  STRATEGY wins in most verbal-setting cells. Resolution depends on
  whether SETTINGS is formatted as numeric authority or soft verbal
  guidance, not on the sys prompt directive.
]

= What the Evidence Supports

== Confident claims

The model builds a linearly-decodable representation of whether
STRATEGY and SETTINGS contain contradictory sizing directives. This
representation is non-lexical (survives combined holdout where both
strategy and setting wordings are unseen), emerges in middle layers
(L16--28 ramp), and stabilizes in upper layers (L28+ plateau). It is a
*constructed feature*: the model is doing semantic comparison across
two prompt sections, not pattern-matching on surface tokens. This is a
real and non-trivial finding about prompt comprehension in
Qwen3-30B-A3B.

== Defensible, requires careful framing

The model's conflict *resolution* behavior is dominated by wording
format --- numeric-scale settings carry disproportionate authority
over verbal settings, and permissive/hedging language gets treated as
optional. The defensible claim is *"on this dataset, resolution is
driven by surface formatting cues rather than source identity."* What
we cannot yet claim is *"the model lacks a deeper arbitration
mechanism"* --- we have not tested a dataset where surface cues are
controlled for.

== Interesting, preliminary

The 77% aligned refusal on force_small $times$ setting_favored shows
genuine cross-section composition: the model is combining "#emph[when
an entry is justified]" with "#emph[one caution remains]" and
concluding no trade is justified. Multi-source reasoning is happening,
just on an axis different from the one we designed for.

== Not yet claimable

Nothing about how the model resolves conflicts internally --- whether
there is a representational signature that distinguishes
"follow strategy" from "follow setting" outcomes. The probe target was
`conflict_present`, not resolution direction. Nor can we say whether
the detection representation *causes* downstream behavior or is a
byproduct of comprehension the model routes through separately.

== Novelty

The thread opened here --- showing that format-mismatched sources
produce shallow (surface-cue-driven) resolution while format-matched
sources produce richer arbitration, and finding the mechanistic
signature of that difference --- is a genuinely novel claim if it
holds. "Interpretability of policy conflict resolution as a function
of prompt formatting" is directly relevant to real agent deployments
where system prompts, tool outputs, and user instructions compete for
behavioral authority in different formats.

= Known Design Issues and Phase 07 Iteration

In rough priority for v5 iteration:

+ *Setting wordings duplicate scale semantics.* `"Trade size: 1/5.
  Use the small allocation tier."` embeds the scale gloss in every
  row. Scale should live in the system prompt once;
  SETTINGS lean (`"Trade size: 1/5"`). The disproportionate
  authority of `setting_v0` is partly driven by this in-line
  reinforcement.

+ *Conditional strategy preambles.* "When an entry is justified,"
  "if one asset clearly leads" give the model a conditional to
  evaluate against market context. On hedged markets this flips
  aligned rows into refusals. Decide deliberately whether
  conditionals are in scope (studying "does the entry fire at all?"
  is a different question from conflict resolution).

+ *System prompt pre-resolves the conflict.* `"SETTINGS still
  constrain the final execution"` pre-answers the question we want
  to study. Remove or neutralize if resolution is a target claim.

+ *Refusal escape hatch.* `"If no trade should be made, return
  observe..."` interacts badly with conditional strategies on hedged
  markets. Either keep and make hedged-market contexts unambiguously
  tradeable on aligned rows, or remove for a cleaner outcome space.

+ *Variant v3 is systematically softer than v0--v2.* Variant
  authoring for v5 should include a pre-publication behavioral check
  that all four variants produce similar aligned-match rates.

+ *Aggregated follow_strategy / follow_setting numbers are
  misleading.* Any resolution discussion must be broken out by
  (strategy variant $times$ setting variant), because the
  variant-pair effect is the primary signal.

= Operational Notes

End-to-end wall clock was \~10 min: 8m46s capture, 4 text baselines
plus 4 probes in parallel on Modal CPU (\~15--60s each), local report
17s. Capture config: `enforce_eager=False`, `max_num_seqs=16`,
`enable_prefix_caching=True`, `enable_thinking=False`,
`add_generation_prompt=True`, residual only.

Two repo fixes landed during this phase, documented in
`docs/PIPELINES_V2_API.md` and the `constructing-workflows` skill:
lazy-import `pipelines_v2.reporting` on the Modal import path so
capture containers do not need matplotlib; and an `enable_thinking`
kwarg on `VLLMEngine` to control the Qwen3 reasoning-mode chat
template.

Reuse hazard observed: adding a derived label column to the dataset
SQL changed the dataset semantic hash and forced `--reuse-completed`
to re-run capture. Derived labels should land pre-capture or via
`TransformSpec`, not via SQL column additions mid-phase.

= Artifacts

- *Dataset generator:* `projects/DX_TERMINAL/prompt_confusion/phase_06/scripts/build_phase_06_dataset.py`
- *Workflow:* `specs/workflow.py` + `workflow.json` (same dir)
- *Auto-generated report (per-step charts):* `outputs/report/report_cc46450ff515_c31e2c79/`
- *Modal run:* `wr_cc10418ff064_f8b538db`. Capture artifact under
  `/artifacts/prompt_confusion/phase_06/capture_1_d5683901/` on the
  `xenon-data` volume.
- *Running notes:* `phase_06/notes.md`
- *Review snapshot:* `reports/review_2026-04-15.md`
