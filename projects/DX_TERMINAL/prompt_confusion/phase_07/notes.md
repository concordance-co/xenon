# Phase 07 Notes

Running log of limitations, gotchas, and learnings as Phase 07
progresses. Keep entries dated. Add things as they come up -- do not
wait until the end of the phase.

---

## Known Limitations Going In

### Carried from Phase 06 (still load-bearing)

- **AUROC and balanced accuracy disagree under covariate shift.** The
  probe direction transfers even when the calibrated threshold does
  not. Always report both metrics on any lexical-holdout probe.
- **Probe weight vectors across layers are not comparable.** Cosine
  similarity is only meaningful within a layer. Phase 07 Move 2's
  probe-direction cosine comparison between v4 and v5 must be done
  per-layer.
- **Capture stores at last prompt token by default.** Any spatial /
  section-level attribution would require `PromptMetadataBuilder`
  metadata and `TokenSelector.section(...)` selection, which v4 did
  not set up. If Phase 07 needs that, add it deliberately.
- **Reuse hazard.** Adding derived label columns to the dataset SQL
  changes the dataset semantic hash and invalidates capture reuse
  (`--reuse-completed`). Land new labels pre-capture in Neon (either
  as source-table columns or as a SQL-level derivation baked in from
  the start) or produce them downstream via `TransformSpec`. Do not
  extend `DATASET_SQL` with new columns mid-run.
- **Reports run locally, never on Modal.** Capture/analysis runners on
  Modal do not have matplotlib. Keep `pipelines_v2.reporting` imports
  lazy on any Modal-loaded import path.
- **vLLM engine config for residual-only capture:** `enforce_eager=False`,
  `max_num_seqs=16`, `enable_prefix_caching=True`,
  `enable_thinking=False`, `add_generation_prompt=True`. MoE router
  capture is not in scope unless resurrected -- constraints there
  (`max_num_seqs=1`, `enforce_eager=True`) make it too slow for a
  full-dataset run.

### Phase 07-specific starting state

- **v4 dataset lives at Neon table `conflict_probe_examples_v4` and is
  frozen.** Phase 07 publishes `conflict_probe_examples_v5` as a new
  table. Do not mutate v4.
- **v5 is a minimal diff from v4, not a rewrite.** Per design.md Move
  1: rewrite setting variants to verbal-imperative matching strategy
  format, drop `setting_v0`'s numeric scale, drop `strategy_v3` and
  `setting_v3`. Three variants per side. Everything else unchanged
  (contexts, pressures, section order, matched pairs, system prompt
  including the "SETTINGS still constrain" hint).
- **Behavioral pre-screen is a hard gate on capture.** Before spending
  GPU on v5 activations, generate on all 3x3 variant pairs and verify:
  (a) aligned rows produce expected behavior at >90%; (b) no single
  variant dominates conflict resolution the way `setting_v0` did on
  v4. Kill bad variants before capture.
- **v4 vs v5 probe-direction cosine requires same-layer, same-probe
  class pipeline.** Use the same `ProbeSpec` surface
  (`SGDClassifier(loss="log_loss")`) with the same hyperparameters on
  both captures so the comparison is meaningful.
- **Move 2.5 hint removal is gated on v5 behavioral outcome.** Do not
  remove the system-prompt resolution hint if v5 still shows the v4
  variant-pair asymmetry -- that would muddle attribution across two
  simultaneous changes.
- **Move 3 resolution probe is gated on clean resolution behavior.**
  The probe target shifts from `conflict_present` to
  `resolution_direction` (follow_strategy vs follow_setting) and only
  runs on conflict rows. If v5 / v5-no-hint resolution is dominated by
  one format, one variant, or one section order, the labels are
  meaningless and this move should wait.

### Framing consistent with Phase 06 report

- **Detection claim on v4 was:** "the model builds a linearly-decodable
  representation of STRATEGY/SETTINGS directive disagreement,
  non-lexical, constructed-feature depth profile at L28+ plateau." That
  is the baseline result Phase 07 extends, not replaces.
- **Resolution claim on v4 was scoped as descriptive of the dataset,
  not mechanistic.** Phase 07 Move 3 is the path to a mechanistic
  resolution claim, but only if the dataset supports it.

---

## Learnings Log

### YYYY-MM-DD -- template entry

- Context: what were we doing
- What we found: the result or surprise
- Implication: what this changes for subsequent work
- Reference: paths to outputs, run IDs, report sections
