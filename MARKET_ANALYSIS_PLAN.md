# Config Entanglement vs. Late Policy: Counterfactual Experiment (Final)

## Context

**Core question:** Does the model form an objective market understanding first, then apply config as a late decision layer? Or does the config warp the model's actual perception of market state from early on?

**Why this matters:** If config warps market perception early → the single-prompt agent design should separate market analysis from policy injection. If config only modulates late → current design is sound.

**Scope narrowing:** In the deployed prompt, config enters via two causal pathways: (1) risk-bucket-specific instruction text in the preamble (before market), and (2) numerical slider values in ACTIVE SETTINGS (after market). Pathway 1 is where early entanglement can occur in a causal decoder. This experiment primarily tests **whether pre-market policy instructions distort market processing at early/mid layers**. That is the strongest causal claim the design can support. The broader question "does config warp market perception?" reduces to this pathway for the current prompt architecture, since pathway 2 cannot affect market tokens by construction.

**Approach:** Two datasets (canonical mechanism + real prompt validation), three analysis lenses (probe transfer, CKA, routing), explicit decomposition of content effect vs position effect.

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
2. **Post-market:** Slider values in ACTIVE SETTINGS (market tokens cannot see these)

### Key data facts (from DB queries)

- Preamble lengths: 1/1 ~13794, 3/3 ~13654, 5/5 ~13736 chars in Mar 12-14 window (58-140 char differences)
- Preamble templates differ by risk bucket (1/1 ≠ 3/3 ≠ 5/5) but are shared across activity levels (1/1 = 2/1)
- Template version drift exists — multiple preamble hashes per config over the 3-week period
- PREVIOUS DECISIONS: 1/1 has ~80 slider refs, 5/5 has ~20. All 19,929 HQ observation prompts have non-empty history.
- Market snapshot: 6-10 tokens per snapshot, each with PctChange (1m/5m/1h/6h/24h/7d), Volume (5m/1h/6h/24h/7d), NetFlow (5m/1h), HolderCount, UniqueTraders5m, Top20HolderPct
- High-quality observation prompts: 1/1=5888 (8 vaults), 3/3=4882 (7 vaults), 5/5=1857 (1 vault)

---

## 2. Datasets

### Dataset A: Canonical Mechanism Set

The primary causal test. Stripped of all vault-specific context.

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

Padding tokens are identical across both padded variants.

**Decomposition:**
- `low_pad` vs `high_pad` = **content effect** at matched positions
- `low_raw` vs `low_pad` = **position effect** (same content, different length)
- `high_raw` vs `high_pad` = **position effect** (same content, different length)
- `low_raw` vs `high_raw` = **combined effect**

**Total:** 120 × 4 = 480 captures (section-pooled)

### Dataset B: Real Prompt Validation Set

Ecological validity check. Not the primary causal test.

**Sampling:**
- 30 prompts from 1/1 — **powered validation arm** (8 vaults available)
- 30 prompts from 3/3 — **powered validation arm** (7 vaults available)
- 30 prompts from 5/5 — **pilot only** (1 vault, insufficient for reliable bootstrap; if a signal appears only in 5/5, do NOT treat it as confirmatory)
- Cap at 2 prompts per `vault × UTC day`
- For 1/1 and 3/3: require dominant preamble hash within narrow time window
- 1/1 vs 3/3 is the primary real-prompt comparison. 5/5 is supplementary.

**Variants generated:**
- Preamble-swapped (raw + padded) for Experiment C
- Settings-numbers-only for Experiment B

**Total:** ~90 × 4 (preamble) + ~90 × 2-3 (numbers) ≈ 630 captures (section-pooled)

---

## 3. Market Representation Units

Parse market block into per-asset-token rows. Each row starts with `  - TOKEN_NAME (SYMBOL) |` and contains multiple metric lines.

**Save per capture, per layer:**
- `row_mean[token]` — mean activation over all tokens in that asset's row
- `row_eos[token]` — activation at last token of that asset's row
- `market_mean` — mean over entire market section
- `market_eos` — last token of market section
- `last_token` — generation prompt position

This avoids washing out local effects across the whole market section.

---

## 4. Labels

Labels are **per-asset-row** (one label per asset in the market snapshot), matching the primary representation units (`row_mean`, `row_eos`). This avoids mixing row identity, symbol identity, and snapshot composition.

### Sanity labels (reading comprehension, per row)
- `is_top_5m_gainer` — binary: is this asset the highest `PctChange5m` in its snapshot?
- `is_top_net_flow` — binary: is this asset the highest `NetFlowInEth5m` in its snapshot?

### Synthesis labels (reconnaissance-gated, per row)

Run a reconnaissance pass on the 120 snapshots BEFORE committing to synthesis labels.

**Preregistered retention criteria:**
- At least 80 positive rows across the Dataset A sample (for leader labels with ~1 positive per snapshot, this requires ~80 snapshots with a clear winner)
- Top-winner token share < 0.45 (no single symbol dominates the positive class)
- At least 5 distinct winning tokens over the Dataset A sample
- Deterministic from `market_snapshot_json`
- Depends on at least 2 fields and/or cross-token comparison within the row or across rows

**Candidate synthesis labels to evaluate (all per row):**
- `is_momentum_divergence_leader` — binary: is this the asset with highest `(PctChange5m − PctChange1h)` in its snapshot?
- `is_flow_surprise` — binary: is this the asset with highest `NetFlowInEth5m / (VolumeInEth1h / 12 + eps)`?
- `has_short_term_reversal` — binary: does this asset have `PctChange5m > 0 AND PctChange1h < 0 AND VolumeInEth5m > 0`?
- `is_participation_momentum_leader` — binary: is this the asset with highest `UniqueTraders5m × PctChange5m`?

If no synthesis labels pass retention criteria: run experiment with sanity labels only. Report as "market-read stability" result, not full "market-understanding" result.

### Probe task

The supervised task is binary classification per asset-row: given `row_mean(l)` or `row_eos(l)` at layer `l`, predict the binary label for that row. Train on one preamble variant, test transfer to the other. This directly tests whether the model's per-asset representation encodes the same market features regardless of upstream instructions.

**Metrics:**
- **Binary leader labels:** Primary metric is `AUROC` (computed within each snapshot over its rows, then averaged across snapshots). Secondary ranking metrics: `Hit@1` (does the top-predicted row match the true leader?) and `MRR` (mean reciprocal rank of the true leader). `balanced_accuracy` is tertiary.
- **Multiclass labels** (if any): `balanced_accuracy` primary, `macro one-vs-rest AUROC` secondary.
- Do NOT average per-row predictions within a snapshot — compute the metric within each snapshot over its rows, then average per-snapshot metrics across snapshots. Bootstrap those per-snapshot results by `vault_day`.

**Symbol-leakage control:** The `TOKEN_NAME (SYMBOL)` prefix tokens at the start of each row carry symbol identity, which correlates with label (e.g., POOPCOIN may often be the gainer). Two mitigations:
- **Primary row pooling** (`row_mean`, `row_eos`) excludes the symbol prefix tokens. Row boundaries start AFTER the `|` delimiter following the symbol name.
- **Symbol-only baseline:** Also train a probe using ONLY the symbol prefix tokens. The full-row probe must beat this baseline to claim it's encoding market features, not just symbol identity.

**Row-position leakage control:** If asset rows are rendered in a fixed or predictable order within the market snapshot, row position can partially proxy symbol identity even after masking symbol tokens. Mitigation:
- **Randomize row order per snapshot** during canonical prompt rendering (Dataset A). Use a deterministic seed per `snapshot_id` so the order is reproducible, but keep it **identical across all 4 variants** for that snapshot. This ensures position-based shortcuts are broken while the content comparison between variants remains paired.
- **Row-index-only baseline:** Train a probe using only the row index (0..N-1) as feature. The symbol-masked row probe must beat both the symbol-only and row-index-only baselines.

**Train/test grouping:** Split by `snapshot_id`, never by individual rows. Each snapshot's rows go entirely into train or test. Compute metrics within each snapshot over its rows, then average the per-snapshot metrics across snapshots. Bootstrap by `vault_day` on those per-snapshot results. This prevents row-level leakage within a snapshot from inflating accuracy.

---

## 5. Capture

- Deterministic forward passes on Qwen3-30B-A3B (A100-80GB)
- Save section-pooled residuals and router indices for all 48 layers
- Save full-sequence residuals for 20 snapshots from Dataset A + 15 prompts from Dataset B = 35 detail captures
- 10 determinism controls (same prompt captured twice)
- **No sparse attention capture in v1.** Deferred to v2 if primary metrics disagree.

### Storage estimate

Section-pooled: ~(6 asset rows × 2 poolings + 3 section-level) × 48 layers × 2048 dim × 2 bytes ≈ **2.8 MB** per capture. Total for ~1100 captures: **~3 GB**.

Full-sequence detail: 35 × ~2 GB = **~70 GB**.

Router data: per layer, top-k expert indices at each section position. ~200 KB per capture. Total: **~220 MB**.

---

## 6. Experiments

### Experiment A: Pre-Market Instruction Effect (Primary)

Run on Dataset A (canonical). Primary causal test of whether risk-bucket preamble instructions distort market-token processing.

**Framing:** The `low_pad` vs `high_pad` comparison measures **content effect under matched market-start position**, not "pure content effect." Padding matches market-start token index but does not hold constant the distance from meaningful instruction tokens to market tokens. The raw-vs-pad controls quantify how much of the observed difference is attributable to position vs content.

**Metrics at market-row positions (`row_mean`, `row_eos`), per layer:**
1. **Within-variant probe accuracy** — train probe on `low_pad` train split, evaluate on `low_pad` test split
2. **Cross-variant transfer accuracy** — train on `low_pad`, evaluate on `high_pad` (and vice versa)
3. **Transfer gap** — mean within-variant accuracy minus mean cross-variant accuracy
4. **Linear CKA** between `low_pad` and `high_pad` market-row activations
5. **Router Jaccard / JSD** at market-row positions

**Position-only controls:**
- `low_raw` vs `low_pad` — position effect for low-risk
- `high_raw` vs `high_pad` — position effect for high-risk
- Report position-induced transfer gap as baseline noise

**Procrustes disambiguator (preregistered):**
- Fit orthogonal Procrustes map from `low_pad` to `high_pad` on train activations
- Evaluate whether probe transfer is restored after alignment

**Procrustes decision thresholds:**
- CKA ≥ 0.90 AND Procrustes-aligned transfer gap ≤ 50% of unaligned gap → `rotated/preserved`
- CKA < 0.85 AND aligned transfer gap ≥ 75% of unaligned gap → `entangled`
- Otherwise → `mixed` (triggers v2 attention investigation as decision point)

### Experiment B: Post-Market Slider Value Effect (Sanity)

Run on Dataset B only. Change only the 5 slider digits in ACTIVE SETTINGS.

**Checks:**
- Market-row activations MUST be invariant (same preamble, market before settings). Any deviation = bug.
- Settings-position activations must change (model reads the numbers).
- Downstream positions may change.

**Primary downstream metric:**
```
Δ_p(l) = h^{all5}(p, l) − h^{all1}(p, l)
```
Compute mean pairwise cosine of Δ vectors across prompts at: `portfolio`, `constraints`, `prev_decisions`, `last_token`.

**Interpretation:**
- High Δ consistency → additive late policy
- Low Δ consistency → policy interacts with content at downstream positions

### Experiment C: Real Prompt Ecological Validation

Run on Dataset B preamble-swap variants.

**This is NOT the primary causal test.** It answers whether the effect from Dataset A survives in full production-length prompts.

**Report separately:**
- Content effect at matched positions (low_pad vs high_pad)
- Position effect from raw-vs-padded controls
- Whether the sign and layer onset match Dataset A

---

## 7. Statistics

- **Bootstrap unit:** `vault_day` (NOT individual prompts)
- 1000 bootstrap reps
- Report 95% CI per layer
- Paired Wilcoxon for cosine deviations and transfer-gap differences
- Correct across 48 layers with both FDR and Bonferroni (report both)
- Do NOT report prompt-level SEs as if prompts were iid
- Dataset B 5/5 arm: report but flag as exploratory (one-vault, insufficient vault-day units for reliable bootstrap)

---

## 8. Decision Rules

### Interpretive asymmetry

Dataset A (canonical) is a mechanism test — it strips away deployment context to isolate the instruction pathway. Dataset B (real prompts) tests ecological validity.

- **Dataset A positive** (market-row content effect at early/mid layers) = **strong evidence** of instruction-pathway entanglement. The stripped context makes this a clean signal.
- **Dataset A null alone is NOT sufficient** to conclude late policy. The canonical prompt is ~60% shorter than production; strategies, portfolio, constraints, and history could create entanglement pathways that Dataset A cannot detect. Dataset B is required to rule out deployment-induced effects.
- **Dataset A null + Dataset B null** = late policy conclusion is well-supported.
- **Dataset A null + Dataset B positive** = deployment-induced entanglement (the bare instruction signal doesn't distort, but the full prompt composition does).

### Conclusions

**`Late policy` (objective market first):** Dataset A shows small market-row content effect through layers 0-31. Transfer gap stays near position-only control. CKA stays near same-variant baseline. Routing is stable. Dataset B Experiment B shows only downstream position changes. Dataset B Experiment C shows no instruction-pathway effect in real prompts.

**`Early entanglement`:** Dataset A shows market-row content effect that exceeds position-only controls at early or mid layers. AND Dataset B preamble edits (1/1 vs 3/3 powered arms) show the same direction in real prompts.

**`Rotated but preserved`:** Transfer drops, but CKA stays high (≥ 0.90) and Procrustes largely restores transfer (aligned gap ≤ 50% of unaligned). Information is preserved in a different encoding.

**`Deployment-induced entanglement`:** Dataset A is stable (canonical prompts show late policy), but Dataset B real prompts show entanglement. The deployed prompt composition — not the bare policy signal — is causing it.

**`Mixed`:** CKA between 0.85-0.90, or Procrustes partially restores. Decision point: proceed to v2 with targeted attention hooks on a small subset.

---

## 9. Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `pipelines/interp/counterfactual.py` | **Create** | Dataset construction: market snapshot sampling, canonical prompt rendering, preamble swap, ACTIVE SETTINGS number edit, padding logic, section boundary parsing, per-asset-row tokenization, label extraction from market_snapshot_json |
| `pipelines/interp/capture.py` | **Modify** | Add `run_counterfactual_capture()`: accepts pre-built messages + section boundaries, computes section-pooled + per-row activations post-forward-pass. New save path `/data/activations/counterfactual/{experiment_id}/`. New metadata: `base_log_id`, `config_tag`, `experiment`, `section_boundaries_json`, `row_boundaries_json` |
| `pipelines/interp/modal_capture.py` | **Modify** | Add `run_counterfactual` Modal function. Separate from existing `run_capture`. |
| `pipelines/interp/counterfactual_analysis.py` | **Create** | Probe training/transfer, CKA computation, Procrustes alignment, router JSD/Jaccard, Δ consistency, per-token heatmaps, bootstrap infrastructure, label reconnaissance pass |
| `pipelines/interp/modal_analysis.py` | **Modify** | Add `mode="counterfactual"` dispatch |

### Existing code reused (unchanged)
- `capture.py:_capture_one()` / `vllm_capture.py:_capture_one_vllm()` — forward pass + activation extraction
- `capture.py:_parse_messages()` — message parsing
- `db.py:connect_neon()` — database connection
- `analysis.py` probe infrastructure pattern (SGDClassifier + StandardScaler + StratifiedKFold)

### New dataset type
Counterfactual captures keyed by `(snapshot_id, config_tag)` for Dataset A and `(base_log_id, config_tag)` for Dataset B. Separate metadata parquet. Does NOT modify existing log_id-keyed pipeline.

---

## 10. Execution Order

### Phase 0: Label Reconnaissance (~1 hour)
1. Sample 120 market snapshots (1 per vault × UTC day, stratified across weeks)
2. Extract `market_snapshot_json` for each
3. Compute all candidate labels
4. Evaluate against retention criteria
5. Commit final label set before any captures

### Phase 1: Dataset Construction (~3 hours)
6. Extract preamble templates for low-risk and high-risk from the dominant hashes in a narrow time window
7. Implement canonical prompt renderer: system + preamble + market table + `## END`
8. Implement padding logic (match token counts, identical padding tokens)
9. Implement per-asset-row tokenization and boundary detection (with deterministic row-order randomization per snapshot_id, identical across variants)
10. Implement ACTIVE SETTINGS number replacement for Dataset B
11. Implement preamble swap for Dataset B
12. Build Dataset A: 120 snapshots × 4 variants = 480 prompt sets
13. Build Dataset B: 90 prompts × 4-7 variants each

### Phase 2: Capture Pipeline (~3 hours)
14. Add per-row activation pooling to capture.py (post-forward-pass: for each asset row, compute row_mean and row_eos)
15. Add `run_counterfactual_capture()` with new metadata schema
16. Add Modal function wrapper
17. Test: 5 captures locally, verify shapes and section boundaries

### Phase 3: Run Captures (~4-6 hours on Modal A100)
18. Dataset A: 480 section-pooled captures
19. Dataset B: ~630 section-pooled captures
20. Detail set: 35 full-sequence captures
21. Determinism controls: 10 duplicate captures
22. Verify metadata completeness

### Phase 4: Analysis (~3-4 hours)
23. Experiment A: probe training, transfer, CKA, router divergence, Procrustes
24. Position-only controls
25. Experiment B: market invariance check, Δ consistency
26. Experiment C: real-prompt validation
27. Per-token heatmaps for detail subset
28. Bootstrap CIs, statistical tests

### Phase 5: Interpret
29. Apply decision rules
30. If `mixed` → scope v2 attention investigation

---

## 11. What v1 Does NOT Include

- No sparse attention capture (deferred to v2)
- No long synthetic suffix in Dataset A
- No heavy interpretation of Dataset A downstream positions
- No 235B validation (separate follow-up if 30B result is strong)
- Dataset B 5/5 arm is exploratory (one-vault)

---

## 12. Minimum Viable Run

| Component | Count | Storage |
|-----------|-------|---------|
| Dataset A pooled | 480 | ~1.3 GB |
| Dataset B pooled | ~630 | ~1.8 GB |
| Detail full-sequence | 35 | ~70 GB |
| Determinism controls | 10 | ~28 MB |
| Router data | ~1120 | ~220 MB |
| **Total** | **~1155 captures** | **~73 GB** |

Forward pass time estimate: ~1155 captures × ~30s per capture on A100 ≈ **~10 hours**. Can parallelize across multiple Modal containers.
