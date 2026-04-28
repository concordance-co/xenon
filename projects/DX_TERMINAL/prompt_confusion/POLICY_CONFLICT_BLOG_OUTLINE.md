# Policy Conflict Blog Outline

Writer-neutral scaffold. Bullets only. Use as source material, not prose.

## Possible Titles

- Finding Policy Conflict Inside Real LLM Trading Prompts
- From Synthetic Conflicts to Real-World Model Internals
- Policy Conflict, Synthetic Probes, And Real Agent Prompts

## Core Thesis Options

- synthetic policy-conflict probes
- real DX Terminal prompts
- transfer only when conflict shape is analogous
- strongest current claim:
  - `trade_size` -> current-prefix concrete action/size conflict
  - `shared_mean` -> broader policy tension, still under-audited
- not:
  - final detector
  - deployment-ready monitor
  - causal mechanism claim

## High-Level Post Shape

- concrete product/system problem first
- synthetic abstraction second
- transfer failure / bridge work third
- Phase 13 top/bottom read as core result
- broader "real data -> synthetic abstraction -> real data" loop woven in
- avoid paper-style tone
- emphasize process and ontology refinement

## Claims To Keep

- clean synthetic policy-conflict directions are strongly readable
- validated synthetic families:
  - `trade_size`
  - `risk_preference`
  - `diversification_preference`
- families share meaningful geometry
- not collinear
- direct full-prompt real transfer was unclear / mostly null
- bridge experiments:
  - real but weak evidence
  - noisy Stage 1b
  - buy-only filtering helped
  - ontology / representation mismatch remained
- Phase 13:
  - positive transfer evidence
  - L32 `settings_end`
  - `trade_size`
  - `shared_mean`
- root-cause labels were the wrong top/bottom ontology
- action-shape labels better match the probe target

## Claims To Avoid

- solved policy conflict detection
- probe classifies all complaints
- bottom rows are non-conflicts
- shared direction already proves broad conflict detection
- production monitor ready
- causal intervention result
- all synthetic directions transfer equally

## 1. Problem: Conflicting Policy Sources

What to cover:

- DX Terminal as concrete system
- model sees multiple policy sources:
  - system rules
  - user strategies
  - active settings
  - portfolio state
  - market data
  - prior decisions / logs
- conflict examples:
  - strategy says buy aggressively
  - settings cap trade size
  - strategy says sell all
  - sell sizing/settings imply partial exits
  - user complaint references directive no longer active
- core question:
  - readable internal signal for policy-source conflict?

Useful sources:

- [Phase 13 notes](phase_13/notes.md)
- [Phase 13 signal brief PDF](phase_13/reports/PHASE13_REAL_TRANSFER_SIGNAL_BRIEF_2026_04_24.pdf)
- [Phase 13 signal brief Typst](phase_13/reports/PHASE13_REAL_TRANSFER_SIGNAL_BRIEF_2026_04_24.typ)

## 2. Why This Matters

What to cover:

- not just "bad trade" diagnosis
- internal representation of competing constraints
- possible downstream uses:
  - monitoring
  - auditability
  - prompt / UX redesign
  - strategy lifecycle debugging
  - settings semantics debugging
- broader thesis, lightly:
  - bringing mech interp into real systems
  - real data -> synthetic abstractions -> real data
  - pipeline viability for monitoring / auditability / improvements

## 3. Synthetic Abstraction

What to cover:

- real prompts too entangled as starting point
- controlled prompt families
- purpose:
  - isolate policy-source conflict
  - known labels
  - clean readout target
  - not meant to replace production

Families:

- `trade_size`
  - buy small vs large
  - respect size constraints
  - output size/action axis
- `risk_preference`
  - asset selection by allowed risk posture
- `diversification_preference`
  - asset selection by concentration vs broadening
  - portfolio-conditioned

Sources:

- [Phase 12 checkpoint PDF](phase_12/reports/PROMPT_CONFLICT_FAMILY_CHECKPOINT_2026_04_16.pdf)
- [Phase 12 checkpoint Typst](phase_12/reports/PROMPT_CONFLICT_FAMILY_CHECKPOINT_2026_04_16.typ)
- [Phase 12 notes](phase_12/notes.md)
- [Phase 09 strict battery README](phase_09/reports/marshalls_battery/README.md)

Data to include:

| Family | Results |
| --- | --- |
| `trade_size` | XOR `0.9948 / 1.0000`; strategy holdout `1.0000 / 1.0000`; settings holdout `0.9948 / 1.0000`; strict combined `0.990 / 1.000` at L40 |
| `risk_preference` | XOR `0.9635 / 0.9766`; strategy holdout `0.9844 / 0.9937`; settings holdout `0.9740 / 0.9839`; strict both-axes `0.8854 / 0.9119` |
| `diversification_preference` | aligned behavior `1.0000`; conflict behavior `0.8542`; XOR `0.9896 / 0.9995`; strategy holdout `1.0000 / 1.0000`; settings holdout `0.9792 / 0.9957`; strict both-axes `0.8333 / 0.8819` |

Note:

- values are `balanced accuracy / AUROC` where paired

Figures:

- [family_within_auroc_by_layer.png](phase_12/reports/dx_terminal_brief_assets/family_within_auroc_by_layer.png)
- [strict_family_auroc_by_layer.png](phase_12/reports/dx_terminal_brief_assets/strict_family_auroc_by_layer.png)
- [strict_asset_family_auroc_by_layer.png](phase_12/reports/dx_terminal_brief_assets/strict_asset_family_auroc_by_layer.png)

## 4. Clean-Setting Result

What to cover:

- strong linear readability
- cheap text baselines near chance
- three-family geometry
- shared conflict structure
- family-specific offsets
- threshold transfer vs ranking transfer distinction

Geometry data:

| Pair | L36 same-capture cosine |
| --- | ---: |
| `risk_preference` vs `trade_size` | `0.6449` |
| `diversification_preference` vs `risk_preference` | `0.4684` |
| `diversification_preference` vs `trade_size` | `0.4883` |

Interpretation bullets:

- risk and size = tightest pair
- diversification = real third member
- moderate alignment, not collinearity
- baseline offsets can break threshold transfer
- AUROC can remain meaningful when balanced accuracy collapses

Figures:

- [shared_axis_distributions.png](phase_12/reports/three_family_visuals/shared_axis_distributions.png)
  - use for baseline-offset story
- [directed_subspace_scatter_by_family_conflict_v2.png](phase_12/reports/three_family_visuals/directed_subspace_scatter_by_family_conflict_v2.png)
  - use for three-family geometry
- [three_family_visuals README](phase_12/reports/three_family_visuals/README.md)
- [three_family_visuals summary JSON](phase_12/reports/three_family_visuals/summary.json)

## 5. Optional: Joint Prompt Result

Use if needed:

- simultaneous conflict readout
- same activation stream
- both size and risk readable

Data:

| Readout | Balanced accuracy / AUROC |
| --- | ---: |
| `size_conflict_present` | `0.9414 / 0.9862` |
| `risk_conflict_present` | `0.9388 / 0.9871` |
| `any_conflict_present` | `0.9306 / 0.9503` |
| `double_conflict_present` | `0.8898 / 0.9687` |

Figure:

- [phase11_joint_prompt_auroc_by_layer.png](phase_12/reports/dx_terminal_brief_assets/phase11_joint_prompt_auroc_by_layer.png)

## 6. First Real Transfer Attempt

What to cover:

- direct projection onto full real prompts
- unclear / mostly null discrimination
- reason this matters:
  - synthetic success not automatically production success
  - real prompt template mismatch
  - real label mismatch
  - real evidence location mismatch

Failure modes to mention:

- different templates
- noisy complaint labels
- present-tense vs retrospective complaints
- user confusion vs model failure
- market losses
- strategy lifecycle issues
- conflict evidence outside current prefix

Source:

- [REAL_TRANSFER_BRIDGE_PLAN_2026_04_22.md](phase_12/reports/REAL_TRANSFER_BRIDGE_PLAN_2026_04_22.md)

Source note:

- bridge plan says first real-data pass did not separate at coarse global sites

## 7. Bridge Program

What to cover:

- purpose:
  - debug transfer failure
  - separate template mismatch from content/ontology mismatch
- Stage 1a:
  - synthetic content in real DX Terminal template
- Stage 1b:
  - real production content in synthetic benchmark shape
- buy-only filter:
  - helped
  - did not solve all mismatch

Bridge dataset counts:

| Dataset | Table | Rows | Aligned | Conflict |
| --- | --- | ---: | ---: | ---: |
| Stage 1a template control | `dx_terminal_trade_size_stage1a_template_control_v1` | 768 | 384 | 384 |
| Stage 1b loose adapter | `dx_terminal_trade_size_stage1b_adapter_loose_v1` | 258 | 168 | 90 |
| Stage 1b strict adapter | `dx_terminal_trade_size_stage1b_adapter_strict_v1` | 118 | 81 | 37 |
| Stage 1b strict buy-only | `dx_terminal_trade_size_stage1b_adapter_strict_buy_only_v1` | 33 | 27 | 6 |

Interpretation bullets:

- bridge evidence real but weak
- Stage 1b noisy
- buy-only filtering improved AUROC/read
- probe-to-synthetic cosine near zero
- sell/liquidation contamination = part of issue
- not whole issue
- ontology / representation mismatch unresolved

Sources:

- [Bridge plan](phase_12/reports/REAL_TRANSFER_BRIDGE_PLAN_2026_04_22.md)
- [Stage 1a summary](phase_12/outputs/transfer_bridge/trade_size_stage1a_template_control_summary.json)
- [Stage 1b strict summary](phase_12/outputs/transfer_bridge/trade_size_stage1b_adapter_strict_summary.json)
- [Stage 1b strict buy-only summary](phase_12/outputs/transfer_bridge/trade_size_stage1b_adapter_strict_buy_only_summary.json)
- [Stage 1b loose summary](phase_12/outputs/transfer_bridge/trade_size_stage1b_adapter_loose_summary.json)

## 8. Phase 13 Real Signal Discovery

Question to cover:

- does synthetic conflict direction fire anywhere on real prompts?

Design bullets:

- no classifier
- no thresholds
- fixed synthetic directions
- projection over real prompt captures
- inspect cells
- read top/bottom rows

Run:

- run id: `wr_14f78308dbac_dbc78513`
- table: `dx_terminal_signal_discovery_phase13_v1`
- tier: aggressive
- site set: ends-only
- primary cell: L32 `settings_end`

Primary result:

| Direction | Anchor | Complaint | Structure control | Anchor-control | Complaint-control |
| --- | ---: | ---: | ---: | ---: | ---: |
| `trade_size` | 4.425 | 3.803 | 3.278 | +1.147 | +0.526 |
| `shared_mean` | 3.462 | 3.137 | 2.760 | +0.703 | +0.377 |

Family-specific caveat:

- `risk_preference` weaker at this cell
- `diversification_preference` not clean at this cell
- not generic complaint direction

Sources:

- [Phase 13 notes](phase_13/notes.md)
- [Phase 13 signal brief PDF](phase_13/reports/PHASE13_REAL_TRANSFER_SIGNAL_BRIEF_2026_04_24.pdf)
- [Phase 13 signal brief Typst](phase_13/reports/PHASE13_REAL_TRANSFER_SIGNAL_BRIEF_2026_04_24.typ)
- [medium_settings_validation_prereg.md](phase_13/medium_settings_validation_prereg.md)
- [running_log.md](phase_13/running_log.md)

Expected medium-run artifacts:

- `phase_13/reports/signal_discovery/report_dd8c8ac3385c_7e82ff1b/report.md`
- `phase_13/reports/signal_discovery/report_dd8c8ac3385c_7e82ff1b/results/l32_settings_top25_complaint_review.json`
- `phase_13/reports/signal_discovery/report_dd8c8ac3385c_7e82ff1b/results/l32_settings_trade_size_audit_packet.json`

Note:

- deeper Phase 13 result JSONs not present in current checkout
- key values preserved in [Phase 13 notes](phase_13/notes.md)
- key values preserved in [Phase 13 signal brief](phase_13/reports/PHASE13_REAL_TRANSFER_SIGNAL_BRIEF_2026_04_24.pdf)

## 9. Row Reading / Ontology Correction

What to cover:

- most important interpretability move
- read top/bottom rows
- preregistered proxy was wrong
- root-cause labels != probe target

Wrong proxy:

- high expected:
  - `USER_CONFIG_CONFLICT`
  - `config_conflict_like`
- low expected:
  - `RULE_FABRICATION`
  - non-config

Why wrong:

- root-cause labels diagnose why complaint happened
- `trade_size` probe target = visible conflict shape
- `RULE_FABRICATION` can include wrong-size trade
- `USER_CONFIG_CONFLICT` can include stale-history complaint
- current prompt may show no active strategy

Root-cause mismatch:

- `trade_size` top-25:
  - 17/25 `USER_CONFIG_CONFLICT`
- `trade_size` bottom-25:
  - 20/25 `USER_CONFIG_CONFLICT`

Better top/bottom shape:

| Direction | Top action/size | Top strategy ignored | Bottom action/size | Bottom strategy ignored |
| --- | ---: | ---: | ---: | ---: |
| `trade_size` | 20/25 | 5/25 | 15/25 | 10/25 |
| `shared_mean` | 20/25 | 5/25 | 9/25 | 16/25 |

Top `trade_size` complaint types:

- `UNWANTED_BUY`: 10/25
- `UNWANTED_SELL`: 6/25
- `WRONG_SIZE`: 4/25
- combined concrete action/size: 20/25

Point to make:

- internal signal forced label ontology refinement
- "complaint" too broad
- "root cause" too broad
- "current-prefix action/size conflict" closer to probe target

## 10. What The Probe Seems To Read

High projection examples:

- "why did you buy HOTDOGZ?"
- "why did you buy so much POOPCOIN?"
- "Buy available balance 30%, not 10 ETH"
- "Quit buying tokens. Liquidate..."
- "You are under allocated to POOPCOIN"

High projection shape:

- current
- concrete
- action/size-shaped
- visible in prompt prefix
- closer to synthetic `trade_size`

Low projection shape:

- still possibly valid complaint
- evidence elsewhere
- history
- lifecycle state
- interpretation
- bookkeeping

Bottom patterns:

1. no active strategy visible
   - complaint references old/expected strategy
   - example:
     - complaint: "why didn't you lock in some gains when i asked you to?"
     - prompt: `No active strategies.`
     - decision: sells HOTDOGZ 100%

2. agent already taking requested action
   - retrospective conflict
   - example:
     - complaint: "why you not buying AIGF as I said?"
     - active strategy: allocate all available ETH to AIGF
     - decision: buys AIGF 100%

3. multi-step execution
   - current action = phase one
   - example:
     - complaint wants HOLE
     - strategy: sell POOPCOIN first, then deploy into HOLE
     - decision: sells POOPCOIN

4. strategy/rule interpretation
   - examples:
     - "sell every time you have a profit"
     - "why is strategy 30 minutes, not 30 seconds"
     - "you bought higher than my entry"
   - issue type:
     - thresholds
     - timing
     - entry price
     - fulfillment
     - prior decisions

5. partial execution
   - exists but not dominant
   - example:
     - complaint: "SELL IT FULL"
     - decision: sells 50%

Point to make:

- low projection != no conflict
- low projection often != no action
- low projection = less clean current-prefix `trade_size` target

## 11. Claim Boundary

Current strong claim:

- fixed synthetic directions recover real production signal
- L32 `settings_end`
- `trade_size` selective for current-prefix concrete sized-action conflict
- `shared_mean` tracks broader policy tension
- both fire less on temporal/bookkeeping/interpretation complaints

Caveats:

- projection extremes, not gold dataset
- `structure_matched_control`, not true non-complaint production baseline
- high `shared_mean` overlaps high `trade_size`
- broader shared-family top-k not proven yet
- no causal result

## 12. Larger Thesis / Why It Matters

What to weave in:

- practical mech interp workflow
- not one benchmark / one probe
- loop:
  1. real data exposes messy failure mode
  2. synthetic prompts isolate abstraction
  3. probes find candidate signal
  4. bridge tests expose transfer mismatch
  5. real-data projection finds narrower shape-specific signal
  6. row reading improves ontology

Potential phrase fragments:

- clean abstraction
- real system pressure test
- labels changed by looking at internals
- synthetic geometry survives only where shape is analogous
- monitoring / auditability / improvement pipeline

## 13. Next Steps

Concrete next steps:

- hand-label top/bottom rows by conflict shape
- label schema:
  - `current_action_size_conflict`
  - `retrospective_history_conflict`
  - `strategy_fulfillment_conflict`
  - `interpretation_or_rule_conflict`
  - `unclear_or_label_mismatch`
- inspect neighboring settings cells only after hand-labeling:
  - L28 `settings_end`
  - L32 `settings_end`
  - L36 `settings_end`
- find true non-complaint production controls
- later:
  - causal tests
  - interventions
  - monitoring prototype

## Figure / Artifact Index

Synthetic performance:

- [family_within_auroc_by_layer.png](phase_12/reports/dx_terminal_brief_assets/family_within_auroc_by_layer.png)
- [strict_family_auroc_by_layer.png](phase_12/reports/dx_terminal_brief_assets/strict_family_auroc_by_layer.png)
- [strict_asset_family_auroc_by_layer.png](phase_12/reports/dx_terminal_brief_assets/strict_asset_family_auroc_by_layer.png)

Three-family geometry:

- [shared_axis_distributions.png](phase_12/reports/three_family_visuals/shared_axis_distributions.png)
- [directed_subspace_scatter_by_family_conflict_v2.png](phase_12/reports/three_family_visuals/directed_subspace_scatter_by_family_conflict_v2.png)
- [three_family_visuals README](phase_12/reports/three_family_visuals/README.md)
- [three_family_visuals summary JSON](phase_12/reports/three_family_visuals/summary.json)

Joint prompt:

- [phase11_joint_prompt_auroc_by_layer.png](phase_12/reports/dx_terminal_brief_assets/phase11_joint_prompt_auroc_by_layer.png)

Real transfer:

- [PHASE13_REAL_TRANSFER_SIGNAL_BRIEF_2026_04_24.pdf](phase_13/reports/PHASE13_REAL_TRANSFER_SIGNAL_BRIEF_2026_04_24.pdf)
- [PHASE13_REAL_TRANSFER_SIGNAL_BRIEF_2026_04_24.typ](phase_13/reports/PHASE13_REAL_TRANSFER_SIGNAL_BRIEF_2026_04_24.typ)

Primary docs:

- [Phase 13 notes](phase_13/notes.md)
- [Phase 13 running log](phase_13/running_log.md)
- [Phase 13 prereg](phase_13/medium_settings_validation_prereg.md)
- [Real transfer bridge plan](phase_12/reports/REAL_TRANSFER_BRIDGE_PLAN_2026_04_22.md)
- [Phase 12 checkpoint PDF](phase_12/reports/PROMPT_CONFLICT_FAMILY_CHECKPOINT_2026_04_16.pdf)
- [Phase 12 geometry brief PDF](phase_12/reports/DX_TERMINAL_CONFLICT_GEOMETRY_BRIEF_2026_04_17.pdf)
- [Phase 09 strict battery README](phase_09/reports/marshalls_battery/README.md)

## Possible Post Skeleton

- intro:
  - DX Terminal policy-source conflict
  - real-system motivation
  - not final detector
- synthetic setup:
  - three families
  - strong probes
  - shared geometry
- transfer failure:
  - direct full-prompt transfer unclear
  - bridge program
- Phase 13:
  - simpler projection question
  - L32 `settings_end`
  - cohort means
- row read:
  - wrong root-cause proxy
  - top/bottom action-shape contrast
  - concrete examples
- broader lesson:
  - internal signal refines ontology
  - practical mech interp loop
- next steps:
  - hand labels
  - true controls
  - neighboring cells
  - causal later
