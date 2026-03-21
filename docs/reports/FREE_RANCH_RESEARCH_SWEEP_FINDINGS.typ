// ── Page Setup ──────────────────────────────────────────────────
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

// ── Title Block ─────────────────────────────────────────────────
#align(left)[
  #text(size: 9pt, fill: rgb("#b33a2a"), tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold")[Free-Ranch Research Sweep]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[A broad research scan across Xenon’s decision domain, culminating in a synthetic and real-data test of the strongest candidate: preference vs permission algebra. Source notes and candidate ranking are recorded in `docs/FREE_RANCH_RESEARCH_SWEEP.md`.]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[21 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[CANDIDATES]\ #text(size: 9pt)[10 ranked questions]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[SYNTHETIC RUN]\ #text(size: 9pt)[120 captures]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[REAL LEG]\ #text(size: 9pt)[154 paired reruns]],
  )
  #v(0.3em)
  #line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
]

#v(1em)

// ── Verdict ─────────────────────────────────────────────────────
#block(
  width: 100%,
  inset: (left: 14pt, top: 12pt, bottom: 12pt, right: 12pt),
  stroke: (left: 3pt + rgb("#b33a2a"), top: none, right: none, bottom: none),
  fill: rgb("#faf5f3"),
)[
  #text(size: 7.5pt, fill: rgb("#b33a2a"), weight: "bold", tracking: 0.08em)[MAIN READ]
  #v(0.3em)
  #text(size: 12.5pt, weight: "medium")[The best current research path is preference vs permission algebra, not another open-ended manifold hunt.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[Across the free-ranch sweep, this was the most natural, data-rich, and synthetically isolatable question. The synthetic policy dataset supports a stable market-preference signal with downstream policy gating, and the corrected real-data reruns already validate the same qualitative split. The weak spot is risk gating, which needs a harder synthetic design before it becomes a central claim.]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[TOP SYNTHETIC RESULT]\ #text(size: 16pt, weight: "bold")[1.000] #text(size: 8pt, fill: rgb("#888"))[\ market-best asset hit-at-1]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[POLICY-ASSET READOUT]\ #text(size: 16pt, weight: "bold")[0.967] #text(size: 8pt, fill: rgb("#888"))[\ last_token accuracy]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[ROBUST INVARIANCE]\ #text(size: 16pt, weight: "bold")[1.000] #text(size: 8pt, fill: rgb("#888"))[\ permission + strategy means]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[REAL SETTINGS LEG]\ #text(size: 16pt, weight: "bold")[17 / 120] #text(size: 8pt, fill: rgb("#888"))[\ valence flips, 0 reranks]],
)


= Scope

This report intentionally steps back from the recent manifold-heavy line and asks a broader question: *what is the most promising research program, given the actual Xenon prompt structure, the real DB distribution, and what synthetic datasets we can build cleanly?*

The free-ranch sweep used four filters:

- *Natural task structure.* The question should be explicit in the prompt and common in the real corpus.
- *Synthetic isolatability.* We should be able to build a clean synthetic dataset around it.
- *Real-data validation path.* The synthetic result should map back to actual DX-style prompts.
- *Mechanistic leverage.* The answer should advance the decision decomposition, not just produce another probe score.


= Candidate Sweep

The sweep considered ten candidates, ranging from strategy compliance and observe taxonomy to slider semantics and memory inertia. The winning candidate was *Preference vs Permission Algebra*:

- the model forms a market preference,
- then permissions, strategy overrides, and settings determine what is executable,
- and the final action is selected downstream of that decomposition.

#align(center)[#image("./assets/free_ranch_research_sweep/candidate_ranking.png", width: 94%)]
#text(size: 8pt, fill: rgb("#888"))[Ranking from `docs/FREE_RANCH_RESEARCH_SWEEP.md`. The winning candidate subsumes several narrower alternatives rather than competing with them.]

Why this won:

- it directly targets the model’s decision algorithm,
- it is abundant in the real corpus,
- it can be isolated cleanly with synthetic prompts,
- and existing corrected settings reruns already hint that it is true.


= Domain Space

The DB audit made the choice relatively clear.

- `121,352` candidate decision rows are already available in the base cohort.
- Observe dominates the corpus, but the actionability regimes are broad enough to study:
  - `92,244` fully actionable rows
  - `15,116` zero-ETH sell-only rows
  - `9,660` buy-only rows
- `30,115` policy-tension observe candidates already exist.
- `59,468` blocked-observe candidates already exist, but they are dominated by the noisy `high_strategy_present` bucket.
- More specific direct block modes are still large enough to matter:
  - `strategy_blocks_buys`: `1,998`
  - `strategy_blocks_sells`: `1,445`
  - `strategy_blocks_both`: `2,466`

This is exactly the shape of domain that makes preference-vs-permission attractive: the corpus is messy, but the key ingredients are present at scale.


= Synthetic Policy Algebra

I built a fresh synthetic dataset, `policy_algebra_v1`, on the dedicated synthetic Modal volume rather than the crowded legacy volume.

- `permission_grid`: `48` prompts
- `strategy_override_grid`: `48` prompts
- `risk_gate_grid`: `24` prompts
- total: `120` prompts, all captured successfully

The synthetic task holds the market rows fixed while changing only policy text:

- explicit permission mode
- explicit strategy override mode
- explicit risk mode

The labels are constructed directly:

- `market_best_asset`
- `policy_best_asset`
- `expected_action_type`
- `expected_action_asset`
- `scenario_group`

#align(center)[#image("./assets/free_ranch_research_sweep/synthetic_dataset.png", width: 72%)]
#text(size: 8pt, fill: rgb("#888"))[The synthetic suite is intentionally small and algebraic. It is designed to isolate preference vs permission cleanly, not to be difficult in a semantic sense.]


= Synthetic Findings

The synthetic result is easy, but useful.

#align(center)[#image("./assets/free_ranch_research_sweep/synthetic_decomposition.png", width: 86%)]
#text(size: 8pt, fill: rgb("#888"))[The market-best asset is perfectly recoverable from row states, while the policy-adjusted chosen asset is strongest later from `last_token`. Permission and action type are trivially encoded in the policy text itself.]

Key findings:

- `market_best_asset` is perfectly recoverable from row states across the full layer range.
  - best: `row_mean`, layer `0`, `AUROC=1.0`, `hit-at-1=1.0`
- `permission_mode` is perfectly recoverable from `active_settings_eos`.
  - best: layer `0`, `accuracy=1.0`
- `expected_action_type` is perfectly recoverable from the settings / downstream section.
  - best: layer `0`, `accuracy=1.0`
- `policy_best_asset` is *not* best read directly from the settings section.
  - `active_settings_eos` best: `0.833`
  - `last_token` best: `0.967` at layer `27`

#align(center)[#image("./assets/free_ranch_research_sweep/policy_best_asset_curves.png", width: 90%)]
#text(size: 8pt, fill: rgb("#888"))[The policy-adjusted asset choice sharpens later than the raw permission labels. That is the most interesting synthetic result in the suite.]

Interpretation:

- The market preference signal is already stable and easily decodable from the rows.
- The permission and action labels are explicit and therefore trivial to read.
- The composition “which asset should be acted on under policy?” is the later step.

That is the right decomposition for this line of research, even though the synthetic prompts are intentionally easy.


= Robustness Read

The permission and strategy branches remain stable under repeated random train/test splits. The risk branch does not.

#align(center)[#image("./assets/free_ranch_research_sweep/synthetic_invariance.png", width: 78%)]
#text(size: 8pt, fill: rgb("#888"))[Repeated-split means over the held-out synthetic groups. Permission and strategy invariance are perfectly stable; the risk branch is not.]

Repeated-split summary (`16` repeats):

- permission top-symbol invariance:
  - mean `1.000`, std `0.000`
- strategy top-symbol invariance:
  - mean `1.000`, std `0.000`
- risk-pair policy accuracy:
  - mean `0.416`, std `0.368`

This is the one place where I would actively *not* overclaim.

- The permission and strategy results are robust.
- The risk-gate result is unstable.
- That looks more like a synthetic design problem than a deep challenge to the overall preference-vs-permission framing.

So the right read is: the candidate won, but the synthetic risk branch is not ready to carry a headline result.


= Real Validation

The strongest part of this project is that the synthetic direction already maps onto existing real-data reruns.

The corrected `blocked valence + settings twist v2` report already showed:

- `17 / 120` settings valence flips
- `13 / 120` strong trade-probability shifts
- `0 / 154` top-symbol reranks across blocked pairs and settings triplets
- `3 / 34` blocked reveals after clearing strategies

#align(center)[#image("./assets/free_ranch_research_sweep/real_validation.png", width: 84%)]
#text(size: 8pt, fill: rgb("#888"))[The real-data leg is more interesting than the synthetic leg: asset identity stays stable, but downstream action state moves. That is exactly the preference-vs-permission split.]

Interpretation:

- The real-data settings leg already supports the same qualitative picture:
  - stable asset preference
  - downstream permission / policy movement
- The generic blocked pool remains noisy and under-targeted.
- The right real follow-up is *not* more generic blocked observe. It is a focused rerun on:
  - `strategy_blocks_buys`
  - `strategy_blocks_sells`
  - `strategy_blocks_both`


#pagebreak()

= Overall Interpretation

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#2e7d32"), top: none, right: none, bottom: none),
  fill: rgb("#e8f5e9"),
)[
  #text(size: 7.5pt, fill: rgb("#2e7d32"), weight: "bold", tracking: 0.08em)[SUPPORTED NOW]
  #v(0.2em)
  Preference vs permission is the strongest current research path. The stable market-preference signal and the downstream action-state shift both show up in the synthetic and real legs.
]

#v(0.5em)

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#f57f17"), top: none, right: none, bottom: none),
  fill: rgb("#fff8e1"),
)[
  #text(size: 7.5pt, fill: rgb("#f57f17"), weight: "bold", tracking: 0.08em)[NOT SUPPORTED YET]
  #v(0.2em)
  The risk-gate branch is not yet robust, and the generic blocked-observe pool is still too noisy to act as the primary real-data validation set.
]

#v(0.6em)

The practical implication is important: the best research program from here is not “more manifold fishing.” It is a cleaner decision-decomposition program:

1. stronger synthetic permission / strategy algebra,
2. redesigned synthetic risk gating,
3. targeted real direct-block reruns,
4. only then broader causal or geometric elaboration.


= What I Would Do Next

1. Build `policy_algebra_v2`.
   - Make the risk branch harder and less lexically obvious.
   - Distribute policy text instead of concentrating it in one settings block.
2. Run a direct-block real rerun.
   - prioritize `strategy_blocks_buys`, `strategy_blocks_sells`, `strategy_blocks_both`
3. Keep the corrected settings rerun as the main real anchor.
   - It is already the strongest evidence that preference is stable while downstream action state moves.
4. Use geometry only where it pays down complexity.
   - The best current use of geometry is explanatory support, not the main objective.


= Operational Notes

- The synthetic capture stayed comfortably within the requested compute budget.
- The initial H200 smoke stalled at startup with no useful work completed.
- The successful synthetic capture ran on A100 after capping `--max-model-len 16384`.
- No large extra real-data capture was required for this sweep because the corrected real rerun results already provided the necessary first validation leg.


#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[Free-Ranch Research Sweep — 21 March 2026. Synthetic policy-algebra capture plus real rerun validation. Ranking, audit notes, and next-step recommendations are recorded in `docs/FREE_RANCH_RESEARCH_SWEEP.md`.]
