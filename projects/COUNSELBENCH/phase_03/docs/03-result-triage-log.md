---
benchmark: counselbench
phase: 03
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/COUNSELBENCH/phase_03/docs/03-analysis-plan.md
---

# CounselBench Result Triage Log

## 2026-04-23 Runs

### Adv 03b Prompt-Family Readout

Workflow run: `wr_32d55c484739_e7536498`

Report: `projects/COUNSELBENCH/advice_safety/phase_03b/reports/controls/report_0c12d35be277_dc2bffc3/report.md`

Result:

- `adv_failure_mode` is decodable from prompt-end residuals under source-row grouping: best balanced accuracy `0.7667`.
- The strongest cheap baseline is prompt text at `0.5750`, so the activation margin is `+0.1917`.
- The signal weakens but does not vanish after nuisance residualization: topic residualized best BA about `0.6667`; lexical-trigger residualized best BA about `0.6167`.
- Geometry is not cleanly class-separated. Failure-mode PCA between/within ratios are near `1.0`, while lexical-trigger/topic structure is comparable or stronger in several layers.

Verdict: `LOCALIZED_REPRESENTATIONAL` candidate, not causal-ready.

Routing:

- Keep Adv as a prompt-family/control substrate.
- Do not enter Phase 04 from this alone.
- Use earlier/mid-layer bands as localization candidates, but treat late layer `44` as a readout layer until write-site tests exist.

### Eval Phase 03 Expert-Label Readouts

Workflow run: `wr_9848bc77ae2a_c8e8a267`

Report: `projects/COUNSELBENCH/advice_safety/phase_03_eval/reports/expert_labels/report_0a01ef6ad435_e2dd029f/report.md`

Result:

- `empathy_high`, `specificity_high`, `toxicity_or_judgmental`, and `overall_quality_high` pass the support gate.
- `medical_boundary_violation` and `factuality_low` are blocked by class support (`16` and `10` positives out of `400`, respectively).
- Expert quality readouts beat named cheap baselines, but margins are modest: empathy `0.8167` vs `0.7667`; specificity `0.7524` vs `0.7357`; overall quality `0.7837` vs `0.7783`.
- `toxicity_or_judgmental` is support-ready but weak: best BA `0.5836` vs strongest cheap baseline `0.5000`.
- Direction geometry shows expert positive-quality labels are almost the same direction: empathy vs specificity mean abs cosine `0.9553`; overall vs specificity `0.9848`; empathy vs overall `0.9506`.
- Toxicity points partly opposite to the positive-quality cluster: mean abs cosine `0.6038` to empathy, `0.6853` to specificity, and `0.7078` to overall quality, with negative cosine at all inspected layers.

Verdict: `CONTROL_INSUFFICIENT` for strong response-quality claims; `REPRESENTATIONAL_ONLY` for a broad quality/posture axis.

Routing:

- Treat Eval as useful for response-quality readout, but not yet for safety-boundary causality.
- The next Eval analysis should control responder more directly with responder-balanced or within-question contrasts, not only projection-style residualization.
- Do not use `medical_boundary_violation` from Eval for probes without augmentation or threshold repair.

### Adv Generated-Response Boundary Readout

Workflow run: `wr_0d127d437b3e_e979170f`

Report: `projects/COUNSELBENCH/advice_safety/phase_03/reports/full_adv/report_05ad61e0ca9f_74ff3b95/report.md`

Result:

- Generation parseability is good at the tripwire level: nonempty rate `1.0` over `120` prompts.
- The corrected raw-question chat prompt no longer hits the previous length-cap failure mode: `120/120` generations finished with `stop` under `max_tokens=15000`.
- Replay contexts are now stored as chat messages `[{"role": "user"}, {"role": "assistant"}]`, not as literal `User:` / `Assistant:` strings.
- The provisional generated-response medical-boundary heuristic is still not probeable under the stricter support gate: `110` no / `10` yes overall, with only `1` positive in the test split.

Verdict: `AUGMENTATION_NEEDED` for generated-response boundary readouts.

Routing:

- Do not run E2 as a probe on the current heuristic labels.
- Repair options are: curated/manual unsafe-response labels, targeted contrast-set augmentation, or a separate benchmark-faithful protocol for eliciting unsafe vs safe boundary behavior.
- Keep this result as evidence that Qwen is mostly boundary-conservative on the current Adv prompt set and that the earlier cap issue was prompt-format related, not as evidence that no boundary representation exists.

## Current Evidence Level

- Adv failure-mode signal: Level 2 to weak Level 3 representational evidence.
- Eval positive-quality signal: Level 2 representational evidence, strongly entangled with broad quality/responder/style structure.
- Eval/Adv medical-boundary signal: blocked for current probe claims.
- Phase 04: still blocked. Required missing pieces are robust safety-boundary readout, matched donor-target pairs with supported labels, and stronger localization/control evidence.
