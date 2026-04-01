# Conflict Probe Experiment — Phase 0: Synthetic Signal Search

## Goal

Can we detect slider-vs-strategy conflict in the residual stream using a linear probe?

## Approach

Use the counterfactual pipeline to generate synthetic prompt variants with **known conflict labels by construction**. No LLM judge needed — conflict is defined by the experimental design.

## Dataset Construction

### Strategy templates (pick 10–20 spanning the spectrum)

Examples:

| Strategy Text | Conflicting Slider | Conflict Condition |
|---------------|-------------------|-------------------|
| "Go all-in on {token}" | trade_size | trade_size ≤ 2 = conflict |
| "Hold forever, ride to 10x" | holding_style | holding_style ≤ 2 = conflict |
| "Only trade {token}, ignore everything else" | diversification | diversification ≥ 4 = conflict |
| "Trade aggressively, catch every move" | trading_activity | trading_activity ≤ 2 = conflict |
| "Buy the dip on momentum tokens" | risk_preference | risk_preference ≤ 2 = conflict (only safe tokens available) |
| "Observe only. No trades." | trading_activity | trading_activity ≥ 4 = conflict |
| "Diversify across top 5 performers" | diversification | diversification = 1 = conflict |
| "Take partial profits at 2% gain" | holding_style | holding_style = 5 = conflict |

### Sweep design

For each strategy:
- Take a real prompt from `interp_examples_v0` via the counterfactual pipeline
- Lock the strategy text
- Sweep the relevant conflicting slider across all 5 values (1–5)
- Hold all other sliders and market context constant
- This gives you a clean gradient from "harmonious" to "conflicting" per strategy

### Volume

- 15 strategies × 5 slider values × ~5 base prompts (different market contexts) = **375 prompts**
- Enough to probe for signal

### Labels

No judge. Labels are determined by the sweep design:
- `conflict = True` when slider value falls in the conflict condition for that strategy
- `conflict_strength` can be ordinal (1–5) if you want to test gradient separability

## Activation Capture

- Model: Qwen3-30B-A3B on Modal (existing pipeline)
- Pooling: **last_token across all 48 layers** (not just 4 — we need the full layer sweep to find the peak)
- Also capture router logits (free, already in pipeline)

## Probe Playbook

1. Per-layer logistic regression (L1 regularized) on conflict vs. no-conflict
2. Plot accuracy / AUROC by layer — look for mid-layer peak (Zhao et al. found layer 14 of 32 on Llama3-8B)
3. If signal exists: check whether the learned direction is consistent across strategy types (train on "all-in" conflicts, test on "hold forever" conflicts)
4. If direction transfers across conflict types → evidence for a general conflict representation

## Success Criteria

- Probe accuracy meaningfully above 50% at any layer = signal exists, keep going
- Probe accuracy above 70% = strong signal, proceed to real data validation
- Cross-strategy transfer above 60% = evidence for general conflict direction

## What Comes Next (only if Phase 0 shows signal)

- Validate on real trade data (build the LLM judge, label real trades, test probe generalization)
- Check whether observations under conflict look different from observations without conflict
- Test whether the conflict direction predicts decision outcomes (the 2×2 matrix)
