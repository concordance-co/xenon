# Config Entanglement vs. Late Policy: Counterfactual Experiment

## Context

**Three questions**, tested separately:

1. **Does pre-existing policy context change how the model reads the market while it is reading the market?**
2. **After the model later sees ACTIVE SETTINGS, does it reinterpret the already-read market differently?**
3. **At the final decision point, is config mostly an additive policy layer, or does it interact with market content?**

**Why this matters:** If config warps market perception early or causes reinterpretation after settings → the single-prompt agent design should separate market analysis from policy injection. If config is purely a late additive layer → current design is sound.

**Model:** Qwen3-30B-A3B surrogate on A100-80GB. Results describe the surrogate only. 235B validation is a separate follow-up.

---

## 1. Verified Prompt Structure

Universal section order (verified on 50 random HQ observation prompts, all configs):

```
[SYSTEM MESSAGE — fixed]
[USER MESSAGE:
  ## OPERATING RULES (preamble)     ← CONFIG-CONDITIONAL, varies by risk bucket
  ## MARKET SNAPSHOT                 ← SHARED across vaults at same timestamp
  ## ACTIVE STRATEGIES               ← VAULT-SPECIFIC
  ## ACTIVE SETTINGS                 ← CONFIG-SPECIFIC (5 slider values + conditional notes)
  ## PORTFOLIO CONTEXT               ← VAULT-SPECIFIC
  ## CONSTRAINTS                     ← VAULT-SPECIFIC
  ## PREVIOUS DECISIONS              ← VAULT-SPECIFIC, config-conditioned reasoning
]
```

**Causal decoder consequence:** Market tokens attend to preamble (config-conditional) but NOT to ACTIVE SETTINGS (which comes after market). Config enters via two causal pathways:
1. **Pre-market:** Preamble instructions (different sell rules, market scanning heuristics per risk bucket)
2. **Post-market:** Slider values in ACTIVE SETTINGS (market tokens cannot see these, but downstream positions can attend to both market AND settings)

### Key data facts (from DB queries)

- Preamble lengths: 1/1 ~13794, 3/3 ~13654, 5/5 ~13736 chars in Mar 12-14 window (58-140 char differences)
- Preamble templates differ by risk bucket (1/1 ≠ 3/3 ≠ 5/5) but are shared across activity levels (1/1 = 2/1)
- Template version drift exists — multiple preamble hashes per config over the 3-week period
- Market snapshot: 6-10 tokens per snapshot, each with PctChange (1m/5m/1h/6h/24h/7d), Volume (5m/1h/6h/24h/7d), NetFlow (5m/1h), HolderCount, UniqueTraders5m, Top20HolderPct
- High-quality observation prompts: 1/1=5888 (8 vaults), 3/3=4882 (7 vaults), 5/5=1857 (1 vault)

---

## 2. Datasets

### Dataset A: Canonical Mechanism Set

The clean causal mechanism test for Question A. Stripped of all vault-specific context.

**Sampling:**
- 120 market snapshots from `interp_examples_v0`, one per `vault_address × UTC day`, stratified across all 3 weeks
- 24 snapshots held out as fixed test split

**Prompt structure per snapshot:**
```
[fixed system message]
[pre-market policy block — low or high risk preamble]
[## MARKET SNAPSHOT — canonical table rendered from market_snapshot_json]
[## END + generation prompt token — minimal suffix only]
```

No strategies, no portfolio, no constraints, no history. Do NOT add a long "neutral" suffix. Do NOT interpret Dataset A `last_token` as a production decision point. The main readout is market-row positions.

**4 variants per snapshot:**
- `low_raw` — low-risk preamble, natural length
- `high_raw` — high-risk preamble, natural length
- `low_pad` — low-risk preamble, padded to `L = max(len(low_raw), len(high_raw))` tokens
- `high_pad` — high-risk preamble, padded to same `L`

`low_pad` vs `high_pad` is the key comparison. `low_pad` and `high_pad` exist for position-control decomposition.

**Decomposition:**
- `low_pad` vs `high_pad` = **content effect** at matched positions
- `low_raw` vs `low_pad` = **position effect** (same content, different length)
- `high_raw` vs `high_pad` = **position effect** (same content, different length)
- `low_raw` vs `high_raw` = **combined effect**

**Total:** 120 × 4 = 480 captures (section-pooled)

### Dataset B: Real Prompt Set

Ecological validation for Questions B and C. Full production-length prompts.

**Sampling:**
- 30 prompts from 1/1 — **powered validation arm** (8 vaults available)
- 30 prompts from 3/3 — **powered validation arm** (7 vaults available)
- 30 prompts from 5/5 — **exploratory** (treat as pilot if effectively one-vault)
- Cap at 2 prompts per `vault × UTC day`
- For 1/1 and 3/3: require dominant preamble hash within narrow time window

**Variants generated:**
- Settings-numbers-only edits (change 5 slider digits, keep everything else fixed) for Questions B & C
- Preamble-swapped (raw + padded) for ecological validation of Question A

**Total:** ~90 × 2-3 (settings) + ~90 × 4 (preamble) ≈ 630 captures (section-pooled)

---

## 3. Activation Pooling

### Market-row level (for Question A)

Parse market block into per-asset-token rows. Each row starts with `  - TOKEN_NAME (SYMBOL) |` and contains multiple metric lines.

**Save per capture, per layer:**
- `row_mean[token]` — mean activation over content tokens in that asset's row (excluding symbol prefix)
- `row_eos[token]` — activation at last token of that asset's row
- `market_mean` — mean over entire market section
- `market_eos` — last token of market section
- `last_token` — generation prompt position

### Downstream section level (for Questions B & C)

For Dataset B captures, additionally pool at positions that can attend to both market AND settings:

- `settings_eos` — last token of ACTIVE SETTINGS section
- `portfolio_mean` / `portfolio_eos` — PORTFOLIO CONTEXT section
- `constraints_mean` / `constraints_eos` — CONSTRAINTS section
- `prev_decisions_mean` / `prev_decisions_eos` — PREVIOUS DECISIONS section
- `last_token` — generation prompt position

These downstream positions are where the model integrates market state with config. Market-row activations themselves CANNOT change after settings (causal decoder), but the model's downstream use of market information can.

---

## 4. Labels

Labels are **per-asset-row** (one label per asset in the market snapshot), matching the market-row representation units. Used in Questions A and B for probe training.

### Sanity labels (reading comprehension, per row)
- `is_top_5m_gainer` — binary: is this asset the highest `PctChange5m` in its snapshot?
- `is_top_net_flow` — binary: is this asset the highest `NetFlowInEth5m` in its snapshot?

### Synthesis labels (reconnaissance-gated, per row)

Run a reconnaissance pass on the 120 snapshots BEFORE committing to synthesis labels.

**Preregistered retention criteria:**
- At least 80 positive rows across the Dataset A sample
- Top-winner token share < 0.45
- At least 5 distinct winning tokens
- Deterministic from `market_snapshot_json`
- Depends on at least 2 fields and/or cross-token comparison

**Candidates:**
- `is_momentum_divergence_leader` — highest `(PctChange5m − PctChange1h)`
- `is_flow_surprise` — highest `NetFlowInEth5m / (VolumeInEth1h / 12 + eps)`
- `has_short_term_reversal` — `PctChange5m > 0 AND PctChange1h < 0 AND VolumeInEth5m > 0`
- `is_participation_momentum_leader` — highest `UniqueTraders5m × PctChange5m`

**Reconnaissance results (120 snapshots):**
- `is_top_5m_gainer`: PASS (120 positives, 9 distinct, top share 0.358)
- `is_top_net_flow`: PASS (120 positives, 9 distinct, top share 0.408)
- `is_momentum_divergence_leader`: PASS (120 positives, 8 distinct, top share 0.258) — **best diversity**
- `is_flow_surprise`: PASS (120 positives, 9 distinct, top share 0.358)
- `has_short_term_reversal`: FAIL (58 positives) — **dropped**
- `is_participation_momentum_leader`: PASS (120 positives, 9 distinct, top share 0.392)

### Symbol-leakage control
- Primary row pooling (`row_mean`, `row_eos`) excludes the symbol prefix tokens. Row boundaries start AFTER the `|` delimiter.
- **Symbol-only baseline:** Train a probe using ONLY the symbol prefix tokens. The full-row probe must beat this baseline.

### Row-position leakage control
- **Randomize row order per snapshot** during canonical prompt rendering (Dataset A). Deterministic seed per `snapshot_id`, identical across all 4 variants.
- **Row-index-only baseline:** Train a probe using only the row index (0..N-1). The symbol-masked row probe must beat both baselines.

---

## 5. Questions

### Question A: Pre-market entanglement

**Does pre-existing policy context change how the model reads the market while it is reading the market?**

Run on Dataset A (canonical). The clean causal mechanism test.

**Metrics at market-row positions (`row_mean`, `row_eos`), per layer:**
1. **Probe transfer:** Train on `low_pad` train split, evaluate within-variant and cross-variant (on `high_pad`). Report transfer gap.
2. **Linear CKA** between `low_pad` and `high_pad` market-row activations.
3. **Router Jaccard / JSD** at market-row positions.

**Position-only controls:**
- `low_raw` vs `low_pad` — position effect for low-risk
- `high_raw` vs `high_pad` — position effect for high-risk
- Report position-induced transfer gap as baseline noise

**Procrustes disambiguator (preregistered):**
- Fit orthogonal Procrustes map from `low_pad` to `high_pad` on train activations
- `rotated/preserved`: CKA ≥ 0.90 AND alignment largely restores transfer (aligned gap ≤ 50% of unaligned)
- `entangled`: CKA < 0.85 AND alignment does not restore transfer (≥ 75% of unaligned gap remains)
- `mixed`: triggers v2 follow-up, not an immediate hard conclusion

**Interpretation:**
- If market-row content effects are near the position-only control through early/mid layers → market reading is config-invariant.
- If they exceed position-only controls in early/mid layers → early entanglement.

### Question B: Post-market reinterpretation

**After the model sees ACTIVE SETTINGS, does it reinterpret the already-read market differently?**

Run on Dataset B. Numbers-only edit in ACTIVE SETTINGS (change 5 slider digits, keep everything else fixed).

**Key insight:** Market-row activations themselves CANNOT change retroactively in a causal decoder. But downstream positions (portfolio, constraints, previous decisions, last_token) CAN attend to both market and settings. The model's *use* of market information can change even if the market tokens themselves don't.

**At downstream positions that can attend to both market and settings:**
- `settings_eos`, `portfolio_eos`, `constraints_eos`, `prev_decisions_eos`, `last_token`

**Metrics:**
1. **Probe transfer on market-feature labels:** Train probes to decode the same market-feature labels used in Question A at these downstream positions. Compare within-setting accuracy vs cross-setting transfer.
2. **CKA across settings** at downstream positions.
3. **Delta consistency:** `Δ_p(l) = h^{all5}(p, l) − h^{all1}(p, l)`. Compute mean pairwise cosine of Δ vectors across prompts.

**Interpretation:**
- Stable market-feature decoding with high delta consistency → settings don't reinterpret market. Late additive policy.
- Config-dependent downstream market decoding → the model IS reinterpreting market state after seeing settings. That is real entanglement in the integrated state, even though the original market tokens didn't change.

### Question C: Final decision-layer interaction

**At the final decision point, is config mostly an additive policy layer, or does it interact with market content?**

Run on same Dataset B captures from Question B.

**Focus on `last_token` and final downstream sections.**

**Metrics:**
- Mean pairwise cosine of the per-prompt config effect vectors (`Δ = h^{settings_a} - h^{settings_b}`).
- High consistency → the config effect is largely shared across prompts (consistent with a late additive policy layer).
- Low consistency → the effect depends heavily on market content (policy-content interaction at the decision stage).

---

## 6. Evaluation

- Split by `snapshot_id`, never by individual rows.
- Compute metrics within each snapshot over its rows, then average per-snapshot metrics across snapshots.
- Bootstrap by `vault_day`, not raw prompt. 1000 bootstrap reps.
- Report `balanced_accuracy` and `AUROC`; for leader-style row tasks also report `Hit@1`.
- Use probe transfer, CKA, and routing together.
- Correct across 48 layers with both FDR and Bonferroni (report both).
- Dataset B 5/5 arm: report but flag as exploratory.
- Procrustes is only a disambiguator, not a primary metric.

---

## 7. Decision Rules

**`Objective market first, policy later`:**
- Question A shows market-row invariance through early/mid layers (near position-only controls).
- Question B shows only downstream reinterpretation or no reinterpretation.
- Question C shows mostly additive late effects (high delta consistency).

**`Early entanglement`:**
- Question A shows market-row divergence beyond position controls in early/mid layers.

**`Late reinterpretation`:**
- Question A is stable (market reading is config-invariant).
- But Question B shows config-dependent downstream market decoding. The model reinterprets market state after seeing settings.
- This is not "early perception warp," but it is still real entanglement in the later integrated state.

**`Mixed`:**
- Metrics disagree or land in the preregistered gray zone. Triggers a targeted v2.

### Realness vs cleanliness
- Dataset A is the clean causal mechanism test.
- Dataset B is the ecological validation on the actual prompt structure.
- If Dataset A and Dataset B agree, the conclusion is strong.
- If they disagree, the right conclusion is not "one is wrong," but "deployment prompt composition is changing the result."

---

## 8. Capture

- Deterministic forward passes on Qwen3-30B-A3B (A100-80GB)
- Save section-pooled residuals and router indices for all 48 layers
- Dataset A: market-row level pooling
- Dataset B: market-row level + downstream section level pooling
- 10 determinism controls (same prompt captured twice)
- **No sparse attention capture in v1.** Deferred to v2 if primary metrics disagree.

### Storage estimate

Section-pooled: ~(6 asset rows × 2 poolings + 5 downstream sections × 2 poolings + 3 section-level) × 48 layers × 2048 dim × 2 bytes ≈ **4.5 MB** per capture. Total for ~1100 captures: **~5 GB**.

Router data: per layer, top-k expert indices at section positions. ~200 KB per capture. Total: **~220 MB**.

---

## 9. Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `pipelines/interp/counterfactual.py` | **Create** | Dataset construction: snapshot sampling, canonical prompt rendering, preamble swap, settings number edit, padding logic, section boundary parsing, per-asset-row tokenization, label extraction, row randomization |
| `pipelines/interp/counterfactual_capture.py` | **Create** | Per-row + per-section activation pooling, counterfactual capture orchestration, safetensors I/O |
| `pipelines/interp/counterfactual_analysis.py` | **Create** | Probe training/transfer, CKA, Procrustes, router JSD/Jaccard, delta consistency, bootstrap CIs, symbol/row-index baselines, label reconnaissance |
| `pipelines/interp/modal_capture.py` | **Modify** | Add `run_counterfactual_capture` Modal function + `CounterfactualCaptureWorker` |
| `pipelines/interp/modal_analysis.py` | **Modify** | Add `run_counterfactual_analysis` Modal function |

### Existing code reused (unchanged)
- `capture.py:_capture_one()` — forward pass + activation extraction via hooks
- `capture.py:_parse_messages()` — message parsing
- `db.py:connect_neon()` — database connection
- `analysis.py` probe infrastructure pattern (SGDClassifier + StandardScaler)

---

## 10. Execution Order

### Phase 0: Label Reconnaissance — DONE
- 120 snapshots sampled, 5 of 6 labels pass retention criteria
- `has_short_term_reversal` dropped (only 58 positives)
- Best diversity: `is_momentum_divergence_leader` (0.258 top share)

### Phase 1: Dataset Construction
1. Extract preamble templates for low-risk and high-risk from dominant hashes in narrow time window
2. Implement canonical prompt renderer: system + preamble + market table + `## END`
3. Implement padding logic (match token counts, identical padding tokens)
4. Implement per-asset-row tokenization and boundary detection (with deterministic row-order randomization per snapshot_id, identical across variants)
5. Implement ACTIVE SETTINGS number replacement for Dataset B
6. Implement downstream section boundary detection for Dataset B
7. Build Dataset A: 120 snapshots × 4 variants = 480 prompt sets
8. Build Dataset B: 90 prompts × settings variants + preamble variants

### Phase 2: Capture Pipeline
9. Per-row activation pooling (market-row level for Dataset A)
10. Downstream section pooling (settings_eos, portfolio, constraints, prev_decisions, last_token for Dataset B)
11. `run_counterfactual_capture()` with new metadata schema
12. Modal function wrappers
13. Test: 5 captures locally, verify shapes and section boundaries

### Phase 3: Run Captures (~4-6 hours on Modal A100)
14. Dataset A: 480 section-pooled captures
15. Dataset B: ~630 section-pooled captures
16. Determinism controls: 10 duplicate captures
17. Verify metadata completeness

### Phase 4: Analysis
18. Question A: probe transfer, CKA, router divergence, Procrustes, position-only controls
19. Question B: downstream market-feature probes, cross-setting transfer, delta consistency
20. Question C: config effect vector consistency at last_token
21. Bootstrap CIs, statistical tests
22. Baseline probes (symbol-only, row-index-only)

### Phase 5: Interpret
23. Apply decision rules
24. If `mixed` → scope v2 attention investigation

---

## 11. What v1 Does NOT Include

- No sparse attention capture (deferred to v2)
- No long synthetic suffix in Dataset A
- No 235B validation (separate follow-up if 30B result is strong)
- Dataset B 5/5 arm is exploratory
- Binary probes only (Trent flagged potential need for more sophisticated probes — revisit in v2 if binary probes are insufficient)
