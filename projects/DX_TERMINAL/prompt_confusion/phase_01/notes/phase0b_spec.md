# Conflict Probe — Phase 0b: Behavioral Validation & Reframe

Phase 0 returned a clean null. The linear probe learned majority class at every layer. Diagnosis: we labeled by input properties (does a conflict exist?) but never verified the model processes those inputs differently. The Zeng (2025) paper on role conflicts shows probes work when you label by **what the model did**, not what you put in.

This phase has three workstreams.

---

## Workstream 1: Behavioral Inference on Synthetic Prompts

**Question:** Does the 30B model's behavior actually change across the slider sweep?

**Method:**
- Take all 375 prompts from conflict_probe_v0
- Run full inference (generate responses, not just forward pass) on Qwen3-30B-A3B
- Parse each response: what action did the model take? (buy/sell/observe, which token, what spend_pct)

**Analysis:**
- For each strategy, plot the behavioral output across slider values 1→5
- Key test: does "go all-in" + trade_size=1 produce different spend_pct than trade_size=5?
- Compute behavioral variance per strategy — which strategies show slider sensitivity?

**If behavior varies:** Relabel the existing activations by behavioral output (followed strategy, followed slider, mixed/neither). Re-run the probe with behavioral labels.

**If behavior doesn't vary:** The 30B model isn't sensitive to sliders in this prompt structure. Either make sliders louder in the prompt, or move to workstream 3 on real data where we know the 235B is sensitive.

---

## Workstream 2: PCA / UMAP on Existing Activations

**Question:** Is there any structure in the activations correlated with our experimental variables, even though the binary probe failed?

**Method:**
- Load the 375 × 48-layer activations already captured
- At each layer, run PCA (keep top 10 components) and UMAP (2D projection)
- Color points by: strategy_key (15 categories), slider_value (1–5), conflict_label (binary), swept_slider (which slider was varied)

**What we're looking for:**
- Clustering by strategy type = model encodes strategy identity (interesting but expected)
- Gradient by slider value = model encodes slider magnitude (means signal exists, probe was wrong approach)
- No structure at all = activations at last token are dominated by market context noise

This is free — we already have the activations. Run it before anything else.

---

## Workstream 3: Source-Following Probe on Real Data

**Question:** Can we detect which instruction source the model followed from the residual stream?

This is the Zeng reframe. Instead of "is there conflict?", ask "did the model follow the strategy or the sliders?"

**Dataset:**
- The original strategy alignment report has 5,903 trades from 306 agents on Qwen3-235B with Claude Sonnet judge labels (aligned / unaligned / partial / baseline)
- These have full prompts in `interp_examples_v0` with strategy snapshots and slider configs
- Cross-reference to get the subset that exists in our interp_examples table

**Activation capture:**
- Run the matched prompts through Qwen3-30B-A3B with full inference (generate + capture)
- Capture: last-token residual stream, all 48 layers
- Also capture at post-strategy and post-slider token positions if feasible (test whether conflict signal is stronger at the point where both inputs have been read vs. at the last token after 5k tokens of market data)

**Probe design:**
- Three-class: aligned / unaligned / partial (drop the 3 baseline examples)
- Per-layer logistic regression, same as phase 0 but with behavioral labels
- Also try binary: aligned vs unaligned (drop partial for cleaner separation)
- Evaluate with balanced accuracy and AUROC given class imbalance (72% / 16% / 11%)

**Cross-validation:**
- 5-fold stratified by vault_address (no vault appears in both train and test) to prevent probe from learning vault-specific patterns rather than decision patterns

**Stretch analysis (if probe works):**
- Train probe on buy trades only, test on sell trades (buy/sell asymmetry from the report)
- Train on high-priority strategies, test on low-priority
- Check whether the probe direction correlates with strategy category (momentum vs all-in vs restrictive)
- Compare probe weights to market representation directions from phases 15-16

---

## Priority Order

1. **Workstream 2** (PCA) — do this first, it's free and informs everything else
2. **Workstream 1** (behavioral inference) — fast, answers the critical diagnostic question
3. **Workstream 3** (source-following probe) — the real experiment, but needs the most setup

## Success Criteria

- Workstream 1: Behavioral variance across slider sweep for ≥ 5 of 15 strategies
- Workstream 2: Visible clustering or gradient in PCA/UMAP by any experimental variable
- Workstream 3: Balanced accuracy > 60% at any layer on the three-class probe, or > 70% on binary aligned/unaligned
