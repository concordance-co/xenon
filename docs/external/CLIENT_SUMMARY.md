# Concordance Research Summary
## Understanding the Internal Reasoning of an Autonomous Trading Agent

**Prepared by:** Concordance
**Date:** April 2026
**Subject:** Synthetic market interpretability program, findings and implications
**Model under study:** Qwen3-30B-A3B (30 billion parameter mixture-of-experts language model)

---

## Executive Summary

We conducted a systematic investigation into how an AI trading agent internally processes market data and arrives at trading decisions. The goal was to move beyond observing the model's outputs ("it bought Token X") and understand what the model actually represents internally when it makes that choice.

**What we established:**

- The model builds a precise internal picture of the market. Individual metrics (price movement, volume, capital flows, trader participation, ownership concentration) are preserved in the model's internal state with near-perfect fidelity, exceeding 99% accuracy across all tested factors.

- The model reasons comparatively. Its internal representation is organized around relationships between assets rather than independent assessments of each one. These relational representations are approximately 20 times more stable than single-asset representations under formatting and presentation changes.

- Pre-market context shapes interpretation. Risk framing and operational constraints presented before market data measurably change how the model encodes that data.

- We identified two primary internal signals: one tracking the standout asset in the market (by volume and momentum), and another tracking market unevenness (how spread apart the field is). Both are confirmed as real, meaningful features of the model's processing.

- These signals are real but not sufficient. When we attempted to use them to steer the model's final trading decision, the stronger signal produced a modest positive effect (+4.2% improvement in choice agreement). The weaker signal did not produce a net positive effect. The model's complete decision process involves additional factors beyond what these two signals capture.

---

## Research Approach

### Methodology

We examined the model's internal states directly: the numerical representations it builds at each stage of processing a prompt. By training simple mathematical probes on these states, we can ask precise questions like "Does the model's state at this point contain enough information to reconstruct a particular market metric?" or "Does editing this specific internal signal change the model's behavior?"

All experiments used carefully controlled synthetic market scenarios where we know the ground truth. We designed the markets, so we know exactly what the model was looking at. This allows us to measure accuracy and isolate variables in ways that are not possible with live market data alone.

### Controls and Validation

Every finding was validated against matched controls:

- **Formatting controls:** We varied prompt layout, ticker symbols, row ordering, and text styling to ensure findings reflect genuine market content rather than surface-level text patterns.
- **Random controls:** Every targeted intervention was compared against a matched random edit of the same magnitude, to distinguish specific signals from general disruption.
- **Statistical rigor:** All behavioral metrics include bootstrap confidence intervals (2,000 samples). Selectivity claims were tested across multiple intervention strengths and conditions.
- **Shuffle baselines:** Key correlations were verified against shuffled data to rule out spurious statistical relationships.

### Scale

| Dimension | Value |
|---|---|
| Model parameters | 30 billion (3 billion active per token) |
| Model layers examined | 48 |
| Synthetic market scenarios | 184 to 920 per experiment (varied by phase) |
| Real inference logs (reference) | 203,292 |
| Real activation captures (bridge tests) | 11,579 |
| Intervention experiments | 48 scenarios per condition, 12 conditions |

---

## Findings

### 1. The Model Preserves Market Data with High Fidelity

We tested whether the model's internal state retains the raw market data it reads. Using linear probes on early-layer representations, we measured recovery accuracy for nine categories of market data.

| Market Factor | Recovery Accuracy (R²) |
|---|---|
| Price change (5-minute) | 0.997 |
| Price change (1-hour) | 0.998 |
| Net capital flow (5-minute) | 0.994 |
| Volume (5-minute) | 0.997 |
| Volume (1-hour) | 0.998 |
| Unique traders (5-minute) | 0.998 |
| Holder concentration | 0.999 |
| Composite attractiveness score | 0.998 |
| Risk-adjusted score | 0.998 |

*R² of 1.000 would be perfect recovery. All factors exceed 0.994.*

**Implication:** The model does not discard or heavily compress raw market data. It maintains a faithful internal copy that downstream processing draws on. The model's choices are grounded in the actual data it received.

### 2. The Model Reasons Through Comparisons

When the model reads a table of six to ten assets, it builds a relational representation: tracking how assets compare to each other rather than encoding each row independently.

We measured resilience to prompt reformatting (changing row order, swapping ticker names, altering text style). Individual asset representations broke easily. Pairwise comparisons between assets were approximately 20 times more robust.

This was confirmed across 384 controlled prompt variations spanning four scenario families, multiple formatting styles, different roster compositions, and three magnitude scales.

**Implication:** The model's market understanding is inherently comparative. When it evaluates a token, it does so relative to the full field. This suggests the model develops a genuine market map.

### 3. Pre-Market Context Shapes Market Interpretation

We tested whether surrounding information (risk preferences, operational constraints) changes how the model interprets market data. We showed the model identical markets with different contextual framing and measured the shift in internal market representations.

| Context Type | Representation Shift |
|---|---|
| Risk framing | 0.061 |
| Constraint framing | 0.070 |

We validated this across five risk levels (conservative to aggressive). At every level, the model's base market coordinate system remained almost perfectly intact (R² > 0.995 for cross-context coordinate recovery), but its interpretation of what the market data *means* shifted in context-appropriate ways.

**Implication:** The model reads context first and uses it as a lens for interpreting the market. This has practical implications for prompt structure: the relevant framing must precede the market data to influence interpretation.

### 4. Two Primary Market Signals Identified

Through statistical decomposition of the model's market-section representations (after controlling for prompt-formatting artifacts), we identified two dominant internal signals.

**Signal A: "Leader"** (early processing layers)

Tracks the standout asset: the one that dominates the market by activity and momentum.

| Predictor | Accuracy (R²) |
|---|---|
| Top asset's 1-hour volume (single feature) | 0.459 |
| 1-hour price change + 5-minute volume (pair) | 0.672 |

**Signal B: "Dispersion"** (later processing layers)

Tracks how uneven the market is: whether one asset is far ahead or the field is tightly bunched.

| Predictor | Accuracy (R²) |
|---|---|
| 1-hour price deviation across assets (single feature) | 0.523 |
| 5-minute volume mean + 1-hour volume median (pair) | 0.843 |

Both signals survived shuffled-data controls (R² collapsed below 0.06), confirming they reflect genuine market content.

**Implication:** The model builds higher-level summaries beyond raw numbers. The leader signal answers "is there a clear standout?" The dispersion signal answers "how competitive is the field?" Both are reasonable strategic considerations for a trading agent.

### 5. The Signals Are Real and Specific

To verify these signals play a functional role, we ran intervention experiments. We edited the model's internal state at the location of each signal and measured behavioral consequences, comparing targeted edits against matched random edits of the same magnitude.

| Condition | Targeted Edit | Random Edit | Gap |
|---|---|---|---|
| Leader, constructive | 43.8% disruption | 68.8% disruption | 25.0 pp |
| Leader, destructive | 56.3% | 68.8% | 12.5 pp |
| Dispersion, constructive | 40.6% | 75.0% | 34.4 pp |
| Dispersion, destructive | 31.3% | 65.6% | 34.4 pp |

*Shown at intervention strength 1.0. All 12 comparisons across three strength levels showed the same pattern.*

**Implication:** Both signals carry specific, non-redundant information. Editing them produces focused behavioral changes; random edits of equal magnitude produce more widespread disruption. This is the signature of a meaningful internal feature.

### 6. Causal Contribution to Final Decisions: Partial

We transplanted each signal from one market scenario into another and measured whether the model shifted its decision toward the donor's choice.

**Leader signal transplant:**
- Choice agreement improved by **+4.2 percentage points**
- Fix rate (correcting wrong choices): **25%** of fixable cases
- Backfire rate (breaking correct choices): **6.3%**
- Spending pattern improvement: **66.7%** of cases

**Dispersion signal transplant:**
- Choice agreement decreased by **-2.1 percentage points**
- Fix rate: **13.6%**
- Backfire rate: **15.4%** (exceeds fix rate)
- Spending pattern improvement: **60.0%** of cases

**Implication:** The leader signal has a genuine, partial causal influence on the model's final decision. The dispersion signal does not have sufficient causal weight to steer decisions on its own. The complete decision pathway involves additional factors, likely the interaction between market perception and user-specific context (portfolio, constraints, strategy history), that we have not yet isolated.

---

## What Was Not Supported

**Full decision explanation from market signals alone.**
The transplant experiments show these signals contribute to but do not determine the final decision. The model's decision process involves additional stages that integrate market perception with user-specific factors. Identifying those stages is the clear next step.

---

## Implications and Recommendations

### For model oversight
The model builds precise, verifiable internal representations of market data. Its decisions are grounded in the actual data it was given. There is no evidence of hallucinated market information or loss of input fidelity.

### For prompt design
Pre-market context measurably shifts market interpretation.

### For future research
The gap between "we can identify what the model sees" and "we can explain why it chose what it chose" is the clearest next target. The transplant experiments point toward the integration points where the model combines its market read with portfolio state, constraint awareness, and strategy framing. Those integration points are where the next phase of research should focus.

---

## Technical Details

| Parameter | Value |
|---|---|
| Model | Qwen3-30B-A3B (mixture of experts) |
| Active parameters per token | ~3 billion |
| Total layers | 48 |
| Expert routing | Top-8 of 60 experts per MoE layer |
| Capture infrastructure | Modal (NVIDIA A100-80GB) |
| Inference engine | vLLM with custom activation capture hooks |
| Probe methods | Ridge regression, L1-regularized logistic regression, SGD with balanced class weights |
| Statistical validation | Bootstrap confidence intervals (2,000 samples), matched random controls, shuffle baselines |
| Synthetic market scenarios | 184-920 per experiment |
| Real inference log population | 203,292 decisions |
| Real activation captures | 11,579 |
| Intervention conditions | 12 (2 signals x 2 directions x 3 strengths), plus matched random controls |
| Intervention sample size | 48 paired scenarios per condition |


