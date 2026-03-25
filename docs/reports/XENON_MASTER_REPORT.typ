#set page(
  paper: "us-letter",
  margin: (top: 2.4cm, bottom: 2.4cm, left: 2.6cm, right: 2.6cm),
  numbering: "1",
  number-align: right,
)

#set text(font: "Georgia", size: 10.5pt)
#set par(justify: true, leading: 0.7em)
#set heading(numbering: none)

#show heading.where(level: 1): it => {
  set text(size: 13pt, weight: "bold")
  v(1.2em)
  it
  v(0.4em)
}

#show heading.where(level: 2): it => {
  set text(size: 11pt, weight: "bold")
  v(0.8em)
  it
  v(0.3em)
}

#show heading.where(level: 3): it => {
  set text(size: 10pt, weight: "bold")
  v(0.5em)
  it
  v(0.2em)
}

#align(left)[
  #text(size: 9pt, fill: rgb("#b33a2a"), tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold")[Xenon Master Report]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Consolidated synthesis of the Typst reporting corpus through 22 March 2026. This report integrates 21 report files spanning
    decision structure, synthetic market geometry, representation controls, context ladders, actionability algebra, and real-data reruns.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[22 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[SOURCE REPORTS]\ #text(size: 9pt)[21 Typst files]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[MAIN SYNTHESIS]\ #text(size: 9pt)[Early preference, preserved market frame, late context deformation]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[MAIN CHALLENGE]\ #text(size: 9pt)[No universal manifold or clean monolithic permission code]],
  )
  #v(0.3em)
  #line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
]

#v(1em)

#block(
  width: 100%,
  inset: (left: 14pt, top: 12pt, bottom: 12pt, right: 12pt),
  stroke: (left: 3pt + rgb("#b33a2a"), top: none, right: none, bottom: none),
  fill: rgb("#faf5f3"),
)[
  #text(size: 7.5pt, fill: rgb("#b33a2a"), weight: "bold", tracking: 0.08em)[MAIN READ]
  #v(0.3em)
  #text(size: 12.5pt, weight: "medium")[The strongest result across the corpus is not “one market manifold.” It is a staged decision-and-representation story: primitive market variables and asset preference are encoded early and very robustly; a shared multi-asset market frame survives multiple context families; later states deform that frame in context-specific ways; and real-data settings reruns already support downstream gating more than upstream reranking.]
]

#v(1.2em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[STRONGEST POSITIVE]\ #text(size: 16pt, weight: "bold")[Early market frame] #text(size: 8pt, fill: rgb("#888"))[\ repeated across synthetic and real reports]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[STRONGEST NEGATIVE]\ #text(size: 16pt, weight: "bold")[No universal 1D manifold] #text(size: 8pt, fill: rgb("#888"))[\ scalar and coupled phases never closed it]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[REAL-DATA READ]\ #text(size: 16pt, weight: "bold")[Stable symbols, moving actions] #text(size: 8pt, fill: rgb("#888"))[\ settings flips with zero top-symbol reranks]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST NEXT TARGET]\ #text(size: 16pt, weight: "bold")[Preference vs permission] #text(size: 8pt, fill: rgb("#888"))[\ factorized, causal, and real-rerun follow-up]],
)

= How To Read Strength

This master report uses four evidence labels:

- *Very strong*: repeated across multiple reports, or across both synthetic and real-data legs, with hard controls that did not overturn the claim.
- *Strong*: supported by multiple targeted reports, but still missing causal validation or a broader real-data bridge.
- *Moderate*: supported by one focused line of work, but a nearby harder control narrows the claim materially.
- *Challenged*: later robustness checks or harder controls weakened the earlier claim enough that it should not be treated as a current headline.

Because the corpus mixes interleaved tracks, the chronology below is reconstructed by dependency, not by filename order.


= Executive Synthesis

== Most Supported Discoveries

#table(
  columns: (1.9fr, 0.8fr, 3.3fr),
  align: (left, center, left),
  table.hline(stroke: 1pt),
  table.header([*Discovery*], [*Strength*], [*Why this rating is justified*]),
  table.hline(stroke: 0.5pt),
  [Early market variables and latent coordinates are explicit in row states.], [*Very strong*], [Primitive variables repeatedly decode near ceiling, often with `R²` around `0.997–1.000`, and 4-asset latent coordinates transfer above `0.99` across several context families.],
  [The model forms asset preference very early.], [*Very strong*], [Decision-structure, synthetic policy, actionability, and real rerun reports all preserve early preference or top-symbol identity even when later action state moves.],
  [A shared multi-asset market frame survives risk, portfolio, and affordance contexts early.], [*Very strong*], [Phases 10-14 all show early coordinate transfer with only small degradation, typically above `0.99 R²`.],
  [Later states deform the shared market frame rather than erase it.], [*Strong*], [Phases 10-14 repeatedly find positive late realignment toward context-adjusted score geometry, with context-specific transform structure on top of preserved early coordinates.],
  [Risk behaves like the cleanest globally coherent ladder.], [*Strong*], [Phases 11-12 show positive realignment across all risk settings and an end-to-end late ladder that composes best under near-rigid maps.],
  [Portfolio behaves more like local reallocation inside the shared frame.], [*Moderate*], [Phase 13 keeps the shared frame intact but finds middle-heavy local fits and weak end-to-end global warps.],
  [Affordance behaves more like route masking than smooth preference reweighting.], [*Moderate*], [Phase 14 keeps the shared frame intact but shows sharper local late fits, stronger severe-step effects, and only moderate end-to-end recoverability.],
  [Settings and policy mostly act downstream of stable asset preference on real data.], [*Strong*], [The corrected rerun report finds `17 / 120` settings valence flips and `13 / 120` strong shifts with `0 / 120` settings-symbol reranks and almost unchanged row states.],
  [Actionability looks more factorized than fused.], [*Moderate*], [Across hardened `v3` and `v4`, `can_buy`, `can_sell`, and `observe_vs_act` outperform the fused `permission_mode` label.],
  [The best current research agenda is preference vs permission rather than manifold-first search.], [*Strong*], [The roadmap, free-ranch sweep, actionability reports, and real reruns all converge on the same narrower decomposition.],
  table.hline(stroke: 1pt),
)

== Best-Supported Negative Results

#table(
  columns: (1.9fr, 0.8fr, 3.3fr),
  align: (left, center, left),
  table.hline(stroke: 1pt),
  table.header([*Claim That Is Not Supported*], [*Strength*], [*Why it should be rejected or narrowed*]),
  table.hline(stroke: 0.5pt),
  [A universal one-dimensional market manifold.], [*Challenged*], [Phases 1-3 improve scalar and coupled geometry only partially; the best scalar or coupled scores never justify a single clean manifold story across variable families.],
  [Stable row-identity profile abstraction under nuisance variation.], [*Challenged*], [Representation-control and Phase 6 results show that layout changes destroy row-profile retrieval far more than style changes do.],
  [Fixed anchor-pair contextual relation as a generally robust object.], [*Challenged*], [Phase 7 looked strong, but Phase 8 largely collapsed the effect once contextual roster pressure was introduced.],
  [Whole-shape family codes for same-rank 4-asset markets.], [*Challenged*], [Phase 9 preserves coordinates and some geometry, but whole-shape identity margins stay tiny and layout-sensitive.],
  [A crisp wording-robust fused permission circuit.], [*Challenged*], [Actionability `v2` looked clean, but `v3` and `v4` weakened the downstream permission story sharply.],
  [A broad hidden directional pool inside generic blocked-observe rows.], [*Challenged*], [The corrected blocked-valence rerun reveals only `3 / 34` directional cases after strategy clearing.],
  [A single generic “settings transform” shared by all context families.], [*Challenged*], [Risk, portfolio, and affordance all preserve the same base frame, but the late transform taxonomy differs substantially across families.],
  table.hline(stroke: 1pt),
)


= Program Arc

Across the corpus, the program moves through six clear turns.

First, the work localized a real early market-preference signal and then tried to explain it with a manifold-first synthetic program.
Second, the scalar and coupled geometry phases showed that the representation is structured, but not by one universal low-dimensional market curve.
Third, harder controls showed that raw row identity was the wrong object and pushed the work toward relations and then set-level geometry.
Fourth, the 4-asset set program found the cleanest representational object so far: a reusable shared market frame whose late geometry is deformed differently by risk, portfolio, and affordance.
Fifth, the actionability line narrowed the downstream claim: early preference is robust, but permission is not one crisp fused code and is better modeled through factorized affordance bits.
Sixth, the real-data reruns and broader research sweeps converged on the same operational agenda: preference vs permission, targeted direct-block cohorts, and causal follow-up on settings-sensitive examples.


= Step-by-Step Experiment Summary

== 1. Market Decision Structure

The 918-tick decision-structure report established the first real-data baseline for staged decision formation.
Buy-target signal was strongest in pre-context row states (`AUROC 0.878`), while target-asset and sell-target received only small downstream boosts (`+0.019` and `+0.018`) from constraints and settings.
The right reading was already “early read, later reweighting,” not “all preference early” and not “all policy late.”
*Evidence read:* strong.

== 2. Synthetic Market Phase 1

Phase 1 validated the synthetic pipeline and showed that clean market variables are linearly present in row states, with latent regressions around `R² 0.989–0.990`.
The geometry result was much weaker: scalar ordering existed, but the best manifold-like read was only moderate (`0.659` on `net_flow_5m`), so the one-dimensional manifold hypothesis was not closed.
*Evidence read:* very strong for variable storage, challenged for a clean 1D manifold.

== 3. Synthetic Market Phase 2 Geometry

Phase 2 densified the scalar sweeps and improved the strongest scalar geometry, especially `pct_5m` (`0.741` vs `0.637` in Phase 1) and concentration in minimal prompts.
That rescued the hypothesis from a pure data-resolution failure, but still did not produce a universal scalar manifold across all variable families.
The result narrowed the search toward family-specific geometry rather than one shared scalar axis.
*Evidence read:* strong for family-specific scalar structure, challenged for a universal manifold.

== 4. Synthetic Market Phase 3 Coupled Geometry

Phase 3 tested whether the right object was a small coupled space instead of an isolated scalar.
Momentum × flow produced a stable coupled geometry (`0.708` dense, `0.693` minimal) that survived within-template checks, but it still did not beat the best scalar baseline from Phase 2.
This shifted the story from “one dominant manifold” to “a family of partial scalar and coupled structures.”
*Evidence read:* moderate.

== 5. Synthetic Market Representation Controls

This checkpoint, covering the Phase 4 and Phase 5 control work, showed that primitive market factors are explicit in row states with near-perfect fidelity, but profile-level abstraction is selective.
Participation/concentration survives the hard symbol-and-row control meaningfully better than momentum/flow, while momentum/flow mostly collapses once superficial shortcuts are removed.
This is where the program stopped treating pairwise AUROC wins as the main abstraction result.
*Evidence read:* very strong for primitive-factor storage, moderate for profile abstraction, challenged for a uniform profile representation.

== 6. Synthetic Market Phase 6

Phase 6 decomposed nuisance variation and found the key failure mode: layout, not wording.
Primitive factors stayed near ceiling, style-only retrieval remained relatively healthy, but layout-only margins fell to around `0.006–0.011`.
That directly argued against more paraphrase-heavy row-retrieval work and set up the shift toward relational objects.
*Evidence read:* strong.

== 7. Synthetic Market Relational Representation

This report asked whether the earlier abstraction failure came from probing the wrong object.
The answer was yes: single-row identity was weak, pairwise relation invariance was much stronger (layout-only margins roughly `0.127–0.162`), and whole-snapshot geometry survived more weakly but above collapse.
The best upstream representation target became comparative structure, not row identity.
*Evidence read:* moderate to strong.

== 8. Synthetic Market Phase 7

Phase 7 validated the relation-first shift under stronger nuisance axes: layout, roster rank, and magnitude.
Relation identity beat both rank-bucket and scale-bucket confounds with margins around `0.24–0.27`, and every best read still achieved `1.0` nearest-neighbor accuracy.
But because the entire task peaked at `row_mean @ layer 1`, the result looked real but too easy.
*Evidence read:* moderate.

== 9. Synthetic Market Phase 8

Phase 8 applied harder contextual pressure by keeping the anchor pair numerically fixed while changing what the surrounding roster made that pair mean.
Most of the Phase 7 win collapsed under this design: hard-control margins fell close to zero almost everywhere, with only `paired_cluster_context` retaining a modest residual signal.
This was an important negative result because it prevented overclaiming fixed-pair relation identity as the main robust object.
*Evidence read:* strong negative.

== 10. XENON Research Roadmap & Kickoff

After the early synthetic geometry phases, the roadmap report argued that the center of gravity should move away from manifold-first work and toward blocked valence, settings twist, and decision decomposition.
It made that argument operational by showing that the live candidate pool was already large enough to run the top-ranked program immediately.
This report matters less as a measurement result than as the first coherent re-ranking of the whole research tree.
*Evidence read:* strong strategic synthesis, not a new mechanistic claim.

== 11. Synthetic Market Phase 9

Phase 9 introduced the cleanest set-level object so far: a 4-asset latent market with fixed rank order but different shapes.
The decisive result was that latent asset coordinates were almost perfectly recoverable (`latent_x 0.99967`, `latent_y 0.99977`), while within-snapshot geometry survived only moderately and whole-shape family identity was nearly flat.
From here onward, the best object became “shared coordinates plus partial geometry,” not family retrieval.
*Evidence read:* very strong for coordinates, challenged for family identity.

== 12. Synthetic Market Phase 10

Phase 10 asked what stronger context does to the shared 4-asset frame.
Market-only probes transferred into low-risk and high-risk prompts with almost no degradation, while later `row_eos` states became slightly closer to context-adjusted score geometry than to the raw latent layout.
This is the first clear preserve-then-deform result: context did not overwrite the market frame, it warped it.
*Evidence read:* strong.

== 13. Synthetic Market Phase 11

Phase 11 upgraded the context axis from toy low/high risk to the actual DX-native `risk_1..risk_5` ladder.
The base market frame survived every risk level above `0.995 R²`, and late realignment toward score geometry appeared at nearly the same depth (`L13–L14`) across the full ladder.
The missing piece was smoothness: local risk-step deformations were clearly structured, but one clean global rotation over the whole ladder was still not supported.
*Evidence read:* strong.

== 14. Synthetic Market Phase 12

Phase 12 fit explicit transform families to the risk ladder and clarified the Phase 11 ambiguity.
Early risk changes were almost rigid or trivial, some late adjacent steps preferred strongly anisotropic local linear fits, but those flexible local maps failed to compose, while the late orthogonal chain matched the direct end-to-end transform almost perfectly (`matrix cosine 0.99998`).
The best description became “global near-rigid ladder with locally anisotropic late steps.”
*Evidence read:* strong.

== 15. Synthetic Market Phase 13

Phase 13 reused the same 4-asset frame for a portfolio ladder.
Early coordinate transfer again stayed near ceiling, but later effects were middle-heavy and local: the strongest fits were on `portfolio_1→2`, `2→3`, and `3→4`, while `market_only→portfolio_5` remained weak.
Unlike risk, portfolio looked like asset-relative redistribution inside the shared frame rather than one coherent global warp.
*Evidence read:* moderate to strong.

== 16. Synthetic Market Phase 14

Phase 14 repeated the same test for an affordance ladder built from progressively stronger route caps and blocks.
The shared frame again survived almost perfectly early, but late transforms were sharper, more local, and more flexible: adjacent steps preferred late linear fits, strong severe-step margins appeared, and the end-to-end `market_only→affordance_5` fit stayed only moderate.
That made affordance look more mask-like than risk or portfolio.
*Evidence read:* moderate.

== 17. Actionability Algebra Findings

The actionability sequence compared `v1`, `v2`, and `v3` synthetic policy datasets.
The stable result was early market preference at ceiling across all phases, while the clean late permission result from `v2` collapsed in `v3` once paraphrase and section-order variation were introduced.
This kept the “early preference” half of the preference-vs-permission story and directly weakened the “clean wording-robust permission circuit” half.
*Evidence read:* very strong for early preference, challenged for fused permission.

== 18. Actionability Next Steps Memo

This memo is not a new empirical report, but it recorded the correct interpretation after `v3`.
The decision was to push one harder synthetic step and bridge only the surviving claim back to real data.
That choice proved important because later work did in fact favor narrower, factorized downstream targets.
*Evidence read:* planning step, included here because it explains the branch point.

== 19. Actionability Affordance Factorization

This follow-up compared hardened `v3` and `v4` variants and asked whether the fused permission label was even the right target.
It was probably not: `can_buy`, `can_sell`, and `observe_vs_act` all decoded materially better than `permission_mode`, while top-symbol invariance stayed perfect.
This is the first actionability report that positively narrows the downstream mechanism instead of only weakening an earlier result.
*Evidence read:* moderate.

== 20. Blocked Valence + Settings Twist

The corrected real-data kickoff tested the top-ranked roadmap track on paired real DX-style prompts.
The blocked-valence half came back weak (`3 / 34` reveals after strategy clearing), but the settings half strongly supported stable preference with downstream state movement (`17 / 120` valence flips, `13 / 120` strong shifts, `0 / 120` settings-symbol reranks, and nearly unchanged row states).
This is the cleanest real-data support in the corpus for “early preference, late permission.”
*Evidence read:* strong.

== 21. Free-Ranch Research Sweep

The free-ranch sweep stepped back and ranked ten broader research programs against natural task structure, synthetic isolatability, real-data validation path, and mechanistic leverage.
Preference vs permission algebra won because it fit the prompt structure, the real corpus, the synthetic design space, and the corrected rerun results better than another open-ended manifold hunt.
The report also made the biggest cautionary note explicit: the risk-gate branch was still unstable and not ready to headline.
*Evidence read:* strong strategic synthesis.


= What The Corpus Now Supports Overall

The reports now support a narrower and better-grounded theory of Xenon than the original manifold-first framing.

The early part of the model appears to preserve a reusable market description very explicitly: primitive variables, latent coordinates, and asset preference all appear early and remain unusually stable under many prompt changes.
Later states do add real structure, but that structure is not best described as a single global market manifold or a single late permission circuit.
Instead, later states look like context operators over a shared market frame:

- risk acts most like a globally coherent near-rigid ladder
- portfolio acts more like local redistribution inside the same frame
- affordance acts more like increasingly sharp route masking

On the decision side, the best current real-data read is the same one suggested by the stronger synthetic reports:

- the preferred asset identities are comparatively stable
- the action state is the part that moves downstream
- generic blocked-observe is too noisy, so direct-block cohorts are the better real-data follow-up

On the actionability side, the evidence no longer supports a monolithic permission representation.
The factorized affordance bits look like the better mechanistic target.


= Highest-Value Open Questions

Four questions would most change the current picture.

- Can causal interventions on the settings-sensitive real subset show that late state changes are sufficient to flip action while leaving early asset ranking intact?
- Do direct block cohorts (`buy_blocked_only`, `sell_blocked_only`, `zero_eth`, `hold_floor`) reveal a much cleaner hidden-valence pool than generic blocked-observe?
- Does the risk / portfolio / affordance transform taxonomy survive a small matched real-DX bridge using the same coordinate-and-deformation lens?
- Do factorized affordance probes (`can_buy`, `can_sell`, `observe_vs_act`) remain robust under a still harder synthetic prompt-hardening pass?


= Bottom Line

The master conclusion is straightforward.

The corpus does not justify a clean universal market manifold story.
It does justify a staged representation-and-decision story:

- early layers preserve a stable market frame and early asset preference
- later layers apply context-family-specific deformations to that frame
- real settings mostly move downstream action state without reranking the preferred assets
- downstream actionability is better modeled as factorized affordance structure than as one fused permission code

That is the strongest current synthesis of the data, and it is a better foundation for the next experiments than any broader “manifold fishing” frame.

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[
  Xenon Master Report — compiled from the Typst reporting corpus through 22 March 2026.
]
