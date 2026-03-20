#set page(
  paper: "us-letter",
  margin: (x: 0.75in, y: 0.8in),
)

#set par(justify: false, leading: 0.6em)
#set text(size: 10pt)

#show heading.where(level: 1): it => block(above: 1.1em, below: 0.5em, text(15pt, weight: "bold")[#it.body])
#show heading.where(level: 2): it => block(above: 0.9em, below: 0.4em, text(12pt, weight: "bold")[#it.body])

= Market Decision-Structure Findings

_Date:_ March 19, 2026

_Primary references at repo root:_

- `MARKET_MANIFOLD_RESEARCH_PLAN.md`
- `MARKET_MANIFOLD_IMPLEMENTATION_PLAN.md`

This report summarizes the first end-to-end execution of the decision-structure and asset-valence work described in the research and implementation plans. The focus of this pass was narrow: determine whether target-asset and buy/sell asset binding are already visible in the market-row manifold, or whether they only become decodable after downstream sections such as `ACTIVE SETTINGS`, `PORTFOLIO CONTEXT`, `CONSTRAINTS`, and `PREVIOUS DECISIONS` are integrated.

== Executive Summary

- The full-sequence capture, pooling, and probe pipeline now works end-to-end on real prompts.
- We captured and analyzed `101` pooled decision examples with usable row and downstream section states.
- On this sample, the best target-asset and buy-target probes are already present in pre-settings market-row representations.
- Downstream sections do not outperform the best pre-row states on the current sample.
- The evidence currently supports an "early asset binding" interpretation: the model appears to form asset-conditioned preference while reading the market, with later sections acting more like overlay, gating, or preservation than initial construction.
- This result is useful but not final. The current sample is observation-heavy and strongly imbalanced on sells.

== What We Accomplished

Relative to `MARKET_MANIFOLD_IMPLEMENTATION_PLAN.md`, this pass completed the first concrete execution of the asset-valence and decision-manifold path.

- Added a working full-sequence vLLM capture path for real decision prompts.
- Fixed the Modal capture path so it distinguishes legacy pooled residual artifacts from true full-sequence residual captures.
- Fixed the Modal analysis wrapper to mount the model volume correctly for tokenizer-based pooling.
- Implemented and validated real-prompt decision-structure pooling.
- Replaced the original real-prompt row-boundary logic with rendered-prompt offset mapping so row and downstream section spans are aligned to the actual tokenizer view of the prompt.
- Captured `100` new full-sequence residual examples on Modal, in addition to the earlier smoke example.
- Re-pooled the relevant Neon slice and reran the decision-structure analysis on corrected `27`-tensor pooled artifacts per snapshot.

Operationally, this also de-risked the pipeline.

- Large worker fan-out was unstable for this workload. Single-worker capture batches were stable.
- The original pooling path silently produced invalid artifacts with only `preamble_*` and `last_token`; this is now fixed.
- The current decision-structure analysis can now run on real pooled row states rather than on degenerate fallback tensors.

== Dataset Used In This Pass

- Usable pooled ticks: `101`
- Pooled tensors per snapshot: `27`
- Decision mix:
  - `54` `record_observation`
  - `38` buys
  - `9` sells
- Asset-row labels:
  - `47` positive `is_target_asset` rows
  - `38` positive `is_buy_target` rows
  - `9` positive `is_sell_target` rows

Interpretation of the dataset:

- The target-asset and buy-target probes have enough positive examples to be suggestive.
- The sell-target probe is underpowered and should be treated as provisional.
- The current sample is still not appropriate for any strong claim about settings-induced reinterpretation under balanced action regimes.

== Main Quantitative Findings

#table(
  columns: (1.5fr, 2.0fr, auto, 2.2fr, auto, auto),
  align: (left, left, center, left, center, center),
  inset: 5pt,
  stroke: 0.4pt,

  [*Target*], [*Best pre state*], [*AUROC*], [*Best post state*], [*AUROC*], [*Post - Pre*],

  [`is_target_asset`], [`row_mean` @ layer 47], [0.900], [`row_mean + portfolio_eos` @ layer 5], [0.867], [-0.033],
  [`is_buy_target`], [`row_eos` @ layer 1], [0.975], [`row_mean + active_settings_eos` @ layer 3], [0.963], [-0.013],
  [`is_sell_target`], [`row_mean` @ layer 15], [0.900], [`row_mean + portfolio_eos` @ layer 1], [0.900], [0.000],
)

Additional readout from the same result file:

- `is_target_asset`
  - strongest pre representation: `row_mean` at layer `47`
  - strongest downstream contender: `row_mean + portfolio_eos` at layer `5`
  - pre remained better than post on the best score
- `is_buy_target`
  - strongest pre representation: `row_eos` at layer `1`
  - strongest downstream contenders included `row_mean + active_settings_eos`, `row_mean + constraints_eos`, `row_mean + prev_decisions_eos`, and `row_mean + last_token`
  - none exceeded the best pre-row score
- `is_sell_target`
  - best pre and post AUROC were equal on this sample
  - because only `9` sells are present, this should not be overinterpreted

== Interpretation

This pass supports the following narrow interpretation.

- Asset binding is already strongly decodable from market-row states.
- Buy-side asset valence is especially early: the strongest signal appears in `row_eos` at layer `1`.
- Later policy/state sections do not appear to create a new stronger asset-binding signal on this sample.
- Portfolio, settings, constraints, and previous decisions can preserve or slightly reshape the signal, but they do not beat the best pre-row representation in the current run.

In plain terms: the model seems to know which asset it likes while it is reading the market rows. The later sections look more like overlay and gating than the first point where the model decides which asset is good.

This is aligned with the representation stack in `MARKET_MANIFOLD_RESEARCH_PLAN.md`, especially:

- the `Asset-in-Context Manifold`
- the `Asset-Valence Manifold`
- the `Decision Manifold`

It is also consistent with the hypothesis that settings and downstream context may modulate an already-formed market preference rather than constructing that preference from scratch.

== What This Does Not Yet Prove

This pass does *not* yet prove that settings never matter. It only shows that, for the current decision-structure sample, the best asset-binding probes are already present before downstream policy/context sections are read.

Important limitations:

- The sample is still small for this question.
- The sample is not balanced by action type.
- Sell examples are especially sparse.
- The current selection process is not optimized for "hard" policy interactions where downstream constraints should matter more.
- These results are correlational probe results, not causal necessity results.

== What We Need More Data For

Per the research plan, the next meaningful improvement is not more infrastructure. It is better data selection.

We need more data for:

- trade-heavy slices rather than observation-heavy slices
- many more sells
- more examples with genuine tension between market preference and policy/constraint state
- more examples where `ACTIVE SETTINGS` or `ACTIVE STRATEGIES` plausibly change the final choice
- matched same-market comparisons under stronger settings or policy interventions

Without that, the pipeline will keep learning mostly the easiest part of the problem: market-driven target preference under relatively ordinary conditions.

== Recommended Next Steps

The next steps below follow directly from both root plans.

1. Reweight capture selection toward trades.
   - Oversample trade ticks from Neon instead of taking the early generic slice.
   - Explicitly oversample sells until the `sell` class is no longer the bottleneck.

2. Build a larger balanced decision-structure set.
   - Target at least `500-1000` full-sequence pooled decision examples.
   - Track class counts during capture so the dataset does not drift back toward observations.

3. Rerun the same asset-binding probes with the larger set.
   - Keep the same `is_target_asset`, `is_buy_target`, and `is_sell_target` targets so the next run is directly comparable.
   - Add confidence intervals once the sample is large enough.

4. Move from executed action labels toward true asset-valence labels.
   - Add "blocked bullish" and "blocked bearish" pseudo-labels for observe cases where policy or affordance prevents action.
   - This is the next step toward the `Asset-Valence Manifold` described in the research plan.

5. Connect this to stronger settings experiments.
   - The current decision-structure result suggests preference is already present in row states.
   - The next question is whether stronger settings interventions reweight that existing signal or leave it mostly unchanged.

6. Run causal necessity tests after the larger dataset is in place.
   - row masking
   - rank-preserving vs magnitude-preserving corruption
   - pairwise swap interventions
   - downstream section ablations

== Status Against The Plans

Completed or substantially completed:

- raw-payload manifold export groundwork
- legality-aware and executed-valence probe targets
- full-sequence decision-structure capture path
- real-prompt decision-structure pooling
- first pre-vs-post asset-binding probe run on real pooled data

Not yet complete:

- trade-balanced full-sequence dataset
- blocked-valence pseudo-label program
- stronger settings intervention family for the real pre/post test
- causal intervention suite
- broad market-geometry analysis on the same full-sequence decision dataset

== Current Bottom Line

The first real run is scientifically useful.

- We now have a functioning pipeline for asking where target-asset and buy/sell binding appear in the model.
- On the current sample, the best evidence points to early market-row binding rather than late downstream construction.
- The next bottleneck is data composition, not missing infrastructure.

The right next move is to capture a larger, trade-heavy, sell-enriched decision dataset and rerun the same analysis before making any stronger claim about settings, policy overlay, or late reinterpretation.
