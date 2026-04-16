# Phase 12 Methodology Notes

## Initial rationale

Phase 12 is the third conflict-family scaffold.

The reason to try diversification next is that it is:

- meaningfully different from `trade_size`
- meaningfully different from `risk_preference`
- still plausibly prompt-local if the portfolio block carries the right
  information

## Initial design choice

The first version of the phase treats diversification as a
portfolio-conditioned asset-selection problem.

The current book already overlaps with `ALPHA`, while `BETA` broadens
the book into a more distinct sleeve.

That gives a crisp intended mapping:

- `diversified` -> `BETA`
- `concentrated` -> `ALPHA`

## Known risk before smoke

The likely behavioral failure mode is simple:

- the model may still choose `ALPHA` too often because it reads as the
  stronger immediate opportunity

If that happens, the right interpretation will be:

- portfolio semantics are too weak relative to raw setup quality

The first smoke should therefore focus on:

- aligned aggressive concentration rows
- aligned diversification rows
- whether `BETA` is chosen reliably when both strategy and settings say
  to broaden the book

## First behavior smoke

Report:

- `reports/behavior_smoke.json`

Headline:

- valid JSON: `1.0`
- exact expected: `0.9271`
- action match: `1.0`
- asset match: `0.9271`
- size match: `1.0`

By conflict band:

- aligned rows: `1.0` exact
- strong-conflict rows: `0.8542` exact

Conflict-direction asymmetry:

- `setting_value = 1` conflict rows
  - expected diversified resolution -> `BETA`
  - exact / asset match: `0.9062`
- `setting_value = 5` conflict rows
  - expected concentrated resolution -> `ALPHA`
  - exact / asset match: `0.8021`

Interpretation:

- the phase is behaviorally much cleaner than the Phase 10 / Phase 11
  risk family
- aligned rows are fully stable
- portfolio-conditioned diversification semantics are real enough to
  drive the model off the strongest raw sleeve in many conflict rows
- the remaining asymmetry is the reverse of the old risk problem:
  concentration overrides are weaker than diversification overrides

This is a promising first-pass benchmark because:

- aligned behavior is clean
- conflict behavior is good enough to be interpretable
- the likely failure mode we worried about did happen somewhat, but not
  enough to break the family

## Initial capture / probe run

Run:

- `wr_05c165379b02_b5a78196`

Report:

- `reports/pipelines_v2/report_27dde9ff9c93_8aa7cdc8/report.md`

### Formal text baseline

Both formal lexical holdouts are at exact chance:

- strategy lexical split:
  - balanced accuracy `0.500`
  - AUROC `0.500`
- settings lexical split:
  - balanced accuracy `0.500`
  - AUROC `0.500`

### Probe results

XOR-style lexical split:

- best layer: `L28`
- balanced accuracy: `0.9896`
- AUROC: `0.9995`

Strategy holdout:

- best layer: `L28`
- balanced accuracy: `1.0000`
- AUROC: `1.0000`

Settings holdout:

- best layer: `L36`
- balanced accuracy: `0.9792`
- AUROC: `0.9957`

### Layer shape

Representative layers:

- `L0`: `0.500 / 0.500`
- `L24`:
  - XOR `0.8958 / 0.9752`
  - strategy `0.9323 / 0.9939`
  - settings `0.8854 / 0.9732`
- `L28`:
  - XOR `0.9896 / 0.9995`
  - strategy `1.0000 / 1.0000`
  - settings `0.9375 / 0.9964`
- `L36`:
  - XOR `0.9427 / 0.9803`
  - strategy `0.9688 / 0.9934`
  - settings `0.9792 / 0.9957`

### Current interpretation after the first formal run

Phase 12 looks like a genuinely strong third family.

Why:

- aligned behavior is fully clean
- conflict behavior is strong enough to be interpretable
- cheap and formal text gates are both dead at chance
- probe performance is near-ceiling under the standard holdouts

The main comparison point to keep in mind is:

- this looks behaviorally cleaner than `risk_preference`
- and representationally it is already in the same general quality band
  as the earlier strong families

## Three-family geometry run

Run:

- `wr_409531658949_7852bd1d`

Report:

- `reports/three_family_geometry/report_deb4d3acab0e_0248c967/report.md`

### High-level outcome

The three-family geometry result is not the simple extension of the
earlier two-family result.

What still holds:

- `trade_size` and `risk_preference` reproduce the earlier shared
  structure
- their strongest overlap is still around `L36`

What is new:

- `diversification_preference` shows only modest cosine similarity to
  either existing family
- but some transfer AUROC values are still meaningfully above chance
- cross-family transfer involving diversification is strongly asymmetric

### Pairwise direction similarity

Representative layers:

- `L28`
  - diversification vs risk: `0.3246`
  - diversification vs size: `0.2991`
  - risk vs size: `0.3743`
- `L36`
  - diversification vs risk: `0.2996`
  - diversification vs size: `0.2846`
  - risk vs size: `0.5341`
- `L40`
  - diversification vs risk: `0.3144`
  - diversification vs size: `0.2661`
  - risk vs size: `0.4370`

Current interpretation:

- diversification shares some structure with the family
- but it is not nearly as aligned with the earlier pair as
  `trade_size` and `risk_preference` are with each other

### Pairwise transfer highlights

At `L36`:

- `trade_size -> risk_preference`
  - balanced accuracy `0.9062`
  - AUROC `0.9832`
- `risk_preference -> trade_size`
  - balanced accuracy `0.8568`
  - AUROC `0.9773`
- `trade_size -> diversification_preference`
  - balanced accuracy `0.7422`
  - AUROC `0.8255`
- `risk_preference -> diversification_preference`
  - balanced accuracy `0.5964`
  - AUROC `0.8492`
- `diversification_preference -> risk_preference`
  - balanced accuracy `0.5000`
  - AUROC `0.8879`
- `diversification_preference -> trade_size`
  - balanced accuracy `0.5000`
  - AUROC `0.9120`

At `L40`:

- `risk_preference -> diversification_preference`
  - balanced accuracy `0.8047`
  - AUROC `0.8923`
- `trade_size -> risk_preference`
  - balanced accuracy `0.8958`
  - AUROC `0.9706`

### What this likely means

The diversification family appears to participate in the broader conflict
family, but with a much stronger calibration mismatch than the original
`trade_size` / `risk_preference` pair.

The most striking pattern is:

- diversification-trained probes rank conflict well in the other
  families (`AUROC ~0.88-0.91`)
- but default threshold transfer collapses to `0.5` balanced accuracy in
  several directions

## Projection / threshold diagnostic

Run:

- `wr_d26774c32aa0_1c8bb1ed`

Manual report:

- `reports/three_family_projection_diagnostic/manual_wr_d26774c32aa0_1c8bb1ed/report.md`

Result JSON:

- `reports/three_family_projection_diagnostic/manual_wr_d26774c32aa0_1c8bb1ed/results/projection_diagnostic_result.json`

### Why this run mattered

The earlier three-family transfer workflow mixed together:

- genuine direction similarity
- probe-threshold transfer

For diversification, that produced the confusing pattern:

- high outgoing AUROC
- but `0.500` balanced accuracy

This diagnostic recomputed family-specific conflict directions inside the
same capture and projected all three families onto each direction with the
source family's own midpoint threshold.

### Main result

The diversification weirdness is mostly real baseline offset plus moderate
shared geometry, not "no relation."

At `L36`, same-capture direction similarity is:

- diversification vs risk: `0.4684`
- diversification vs trade size: `0.4883`
- risk vs trade size: `0.6449`

This is materially higher than the earlier probe-weight cross-family
cosines (`~0.28-0.31`) and much closer to the conceptual picture we
expected.

### Threshold-offset evidence

At `L36`, diversification's score band is far more negative than the other
families:

- diversification aligned mean: `-2.1717`
- diversification conflict mean: `-1.6130`
- diversification threshold: `-1.8923`

Applying the diversification direction to the other families gives:

- diversification -> trade size
  - balanced accuracy `0.5000`
  - AUROC `0.8471`
  - FNR `1.0000`
- diversification -> risk
  - balanced accuracy `0.5599`
  - AUROC `0.8165`
  - FNR `0.8802`

The reverse direction shows the complementary offset:

- risk -> diversification
  - balanced accuracy `0.5000`
  - AUROC `0.8185`
  - FPR `1.0000`
- trade size -> diversification
  - balanced accuracy `0.5859`
  - AUROC `0.8302`
  - FPR `0.8281`

Current read:

- diversification really does live on a shifted score baseline
- that baseline shift is large enough to break cross-family threshold
  transfer
- but the underlying direction overlap is still moderate and clearly
  non-zero
- so the most honest read is:
  - calibration / offset issue first
  - genuine but weaker family overlap second

This makes the last-token `PORTFOLIO` hypothesis more plausible:

- diversification may share the conflict-family subspace
- while the extra portfolio-conditioned text shifts the prompt-EOS
  baseline enough to distort thresholded transfer metrics

## Directed visualization pass

Script:

- `scripts/generate_three_family_visuals.py`

Outputs:

- `reports/three_family_visuals/directed_subspace_scatter_by_family_conflict.png`
- `reports/three_family_visuals/directed_subspace_scatter_by_conflict.png`
- `reports/three_family_visuals/shared_axis_distributions.png`
- `reports/three_family_visuals/summary.json`

### What the first figures show

The most useful figure is the shared-axis distribution plot.

It makes two things visually obvious:

- each family has an internal aligned -> conflict separation on the shared
  axis
- the family baselines are shifted relative to each other

That matches the projection-diagnostic table read:

- shared conflict signal is real across all three families
- family-specific offsets are large enough to break direct threshold
  transfer

The directed 2D scatter is also useful, but more as a communication
figure than as the cleanest analytic view:

- families form partially separated clouds
- within each family, conflict points tend to shift in a similar local
  direction
- `L36` remains the best single layer for the shared-geometry story

Current visual take:

- the offset story is real
- diversification does not look like an isolated outlier
- the shared-axis figure should likely become the anchor figure for a
  shareable team report

## Shareable checkpoint report

Team-facing Typst checkpoint:

- `reports/PROMPT_CONFLICT_FAMILY_CHECKPOINT_2026_04_16.typ`
- `reports/PROMPT_CONFLICT_FAMILY_CHECKPOINT_2026_04_16.pdf`

Scope:

- brief background on the process
- primary focus on datasets, analysis, results, and implications
- anchored by the new directed-subspace and shared-axis visuals

That suggests:

- some shared ranking structure is present
- but the signed boundary / offset learned from diversification does not
  transfer cleanly to the other families

So the current best read is:

- `diversification_preference` is a real third family
- it is not just noise or an outlier
- but it does not slot into the original two-family geometry as cleanly
  as a simple "three equally aligned families" story would predict
