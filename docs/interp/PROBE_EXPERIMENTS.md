# Mechanistic Interpretability Probe Experiments

## What We Have to Work With

**Activation sources** (Qwen3-30B-A3B):
- Residual stream: 48 layers x 2048 hidden_dim (fp16)
- Router logits: 48 layers x 128 experts (fp16)
- Router indices: 48 layers x top-8 (int16)
- Pooling: last_token, mean_pool, or full sequence

**Existing probe targets**: `decision_type`, `trade_side`, `was_profitable_1h`, `risk_tolerance`, `asset`

**Dataset features not yet probed**: PnL continuous values (1h/4h/1d), trade size, vault config dimensions (trade_size, trading_activity, holding_style, diversification), swap execution price, context completeness, model source, sequence length, observation text content

**Sampling**: ~150 trades (balanced buy/sell), ~150 observations, ~100 paired examples (same vault, opposite decisions)

---

## Tier 1: High-Signal, Low-Effort (use existing infrastructure)

### 1. Router Logits vs Residual Stream Showdown
Run every existing target with `--data-source router` and `--data-source residual` side by side. The literature (MOEE, 2024) shows router logits encode meaningful semantic information -- they may outperform residual stream for decision-level features while being 16x smaller (128 dims vs 2048). Plot selectivity curves across layers for both. **Hypothesis**: Router logits will match or beat residual stream for `decision_type` and `trade_side` because routing *is* the model's computational strategy.

### 2. Concept Depth Profiling
Sweep all 48 layers for each target to find the "concept depth" -- the earliest layer where selectivity exceeds a threshold. Recent work (Exploring Concept Depth, 2024) shows basic concepts resolve early while complex ones need depth. **Hypothesis**: `decision_type` (structural -- "am I trading or observing?") resolves by layer 12-16. `was_profitable_1h` (requires market reasoning) won't resolve until layer 35+. `risk_tolerance` (injected via system prompt) resolves very early since it's in the context.

### 3. Pooling Strategy Comparison
For each target, compare `last_token` vs `mean_pool` on both data sources. The literature (Pooling and Attention, 2024) says mean pooling is most robust, but last-token captures the accumulated decision representation in decoder models. **Hypothesis**: `last_token` wins for decision targets (the model's final representation carries the decision); `mean_pool` wins for contextual targets like `asset` and `risk_tolerance` (information spread across the sequence).

---

## Tier 2: New Targets (need small prepare.py additions)

### 4. Continuous PnL Regression Probes
Replace the binary `was_profitable_1h` classification with **ridge regression** on `pnl_1h_pct`, `pnl_4h_pct`, and `pnl_1d_pct` as continuous targets. This is richer signal. Report R^2 per layer. **Hypothesis**: If the model encodes any market-predictive information, we'd see positive R^2 in late layers. If R^2 is near zero at all layers, the model's "profitable" trades aren't based on internal price prediction -- they're based on heuristics.

### 5. Trade Size Probe
Probe for `size` (the spend_pct field -- how much of the portfolio the model risks). Bin into small (<10%), medium (10-30%), large (>30%). **Hypothesis**: Size decisions correlate with router entropy -- larger trades should show more concentrated routing (the model is "more certain"). This connects to the risk/confidence literature.

### 6. Multi-Dimensional Vault Personality Probe
The vault config has 5 dimensions (trade_size, trading_activity, holding_style, diversification, risk_preference) each on a 1-5 scale. Train separate probes for each. **Hypothesis**: These are injected via system prompt, so they should be linearly readable from early layers. The interesting question is whether they *persist* through to late layers or get consumed and transformed into behavioral representations.

### 7. Context Completeness Probe
Probe for `context_complete` -- whether the model had full market/portfolio/strategy context. **Hypothesis**: Missing context changes the model's internal processing strategy (different expert routing). This is a sanity check: if we can't detect missing context, our probes may lack sensitivity.

---

## Tier 3: Novel MoE-Specific Experiments

### 8. Router Entropy as Uncertainty Signal
For each example, compute per-layer router entropy: `H = -sum(softmax(logits) * log(softmax(logits)))` across the 128 experts. Compare entropy distributions between:
- Trades vs observations (is trading "harder"?)
- Profitable vs unprofitable trades (does the model "know" when it's uncertain?)
- Different assets (are some tokens processed with more expert consensus?)

**Hypothesis**: Based on recent MoE research, routing entropy decreases in later layers as representations sharpen. Profitable trades should show *lower* late-layer entropy (more decisive routing), while unprofitable trades show higher entropy (the model is uncertain but acts anyway).

### 9. Expert Specialization Clusters
Go beyond the existing `experts` analysis mode. For each layer, build a 128-dim vector per example from expert selection frequencies (how often each expert is chosen across the sequence). Cluster these vectors with k-means or HDBSCAN. **Hypothesis**: Clusters will correspond to decision types or asset classes -- revealing that the MoE routing network has implicitly learned to partition its experts by trading task.

### 10. Discriminative Expert Identification
For binary targets (trade/observe, buy/sell, profitable/unprofitable), compute Cohen's d per expert per layer. Identify the top-5 "decision experts" -- experts whose activation frequency differs most between classes. Then ablate: what happens to probe accuracy when we mask those experts' logits from the feature vector? **Hypothesis**: A small subset of experts (5-10 out of 128) will carry most of the discriminative signal, consistent with the sparse specialization literature.

### 11. Router Logit Difference-in-Means Probes
The simplest possible probe: no training. For each layer, compute `mean(router_logits[trade]) - mean(router_logits[observe])`. This 128-dim direction *is* the probe. Project test examples onto it and threshold at 0. Compare accuracy to trained linear probes. Recent work (Mass-Mean Probing, Marks & Tegmark 2024) shows this often matches or beats trained probes. **Hypothesis**: For `decision_type`, difference-in-means will be within 2-3% of trained probes, proving the signal is strongly linearly encoded.

---

## Tier 4: Advanced / Research-Grade

### 12. Paired Vault Contrastive Analysis
The `interp_sample_paired_v0` sample has examples from the *same vault* making opposite decisions (trade vs observe). For each pair, compute `activation[trade] - activation[observe]`. Average these difference vectors across pairs. This isolates the "decision direction" while controlling for vault personality, strategy, and market regime. **Hypothesis**: The average difference vector is a high-quality linear probe direction -- potentially better than a trained probe because it controls for confounds.

### 13. Causal Validation via LEACE Erasure
Train the best probe for `decision_type`. Then use LEACE (Linear Erasure, NeurIPS 2023) to surgically remove the concept from activations. Re-run the probe -- accuracy should drop to chance. Then check: does erasing `decision_type` information also destroy `trade_side` information? (It should, since trade_side only exists for trades.) Does it destroy `was_profitable_1h`? (Partially -- profitability is downstream of the trade decision.) This maps the **causal dependency structure** between concepts.

### 14. Cross-Layer Router Pattern Trajectories
For each example, concatenate router logit vectors across all 48 layers into a single 48x128 = 6144-dim vector. This captures the model's full "routing strategy." Train probes on this concatenated space. **Hypothesis**: Cross-layer patterns will dramatically outperform single-layer probes for complex targets like `was_profitable_1h`, because profitable trading requires integrating information across multiple processing stages.

### 15. Sparse Probe Feature Selection
Train L1-regularized probes (varying sparsity from k=5 to k=100 non-zero weights) on both residual stream and router logits. For residual stream: which 10 out of 2048 dimensions encode "is this a buy?" For router logits: which 5 out of 128 experts differentiate profitable from unprofitable? **Hypothesis**: Router logits will need fewer features (sparser signal) than residual stream, because expert specialization naturally concentrates information.

### 16. Temporal Position Probing (Full Sequence)
Capture without pooling. Instead of probing the last token, probe at every 10th position through the sequence. Plot selectivity as a function of relative position. **Hypothesis**: `decision_type` information appears abruptly near the end of the sequence (when the model processes its tool call), while `risk_tolerance` and `asset` information appear early (when the model processes the context/prompt).

---

## Suggested Execution Order

**Phase 1** (experiments 1-3): Run with existing infrastructure, no code changes. Just different CLI flags. Gets baseline understanding of what's where.

**Phase 2** (experiments 4-7, 8, 11): Add continuous targets + new features to prepare.py, add entropy computation and difference-in-means to analysis.py. Medium effort, high insight.

**Phase 3** (experiments 9-10, 12, 14-15): Requires new analysis modes. Router clustering, paired contrastive, cross-layer concatenation, sparse probes.

**Phase 4** (experiments 13, 16): LEACE integration, full-sequence position probing. Most complex but highest scientific value.

---

## Key References

- MOEE (2024): MoE router weights as zero-shot text embeddings
- Exploring Concept Depth (2024): Formal measure of concept resolution layer
- Mass-Mean Probing (Marks & Tegmark 2024): Difference-in-means outperforms trained probes
- LEACE (Belrose et al., NeurIPS 2023): Closed-form linear concept erasure
- DeepSeekMoE (ACL 2024): Fine-grained expert specialization patterns
- Pooling and Attention (2024): Comparison of LLM pooling strategies
- How Reliable are Causal Probing Interventions? (2024): Nonlinear probes in early layers, linear in late
- Adaptive Temperature Scaling (ICLR 2025): Per-token calibration
- No Answer Needed (2025): Probes predict LLM correctness from question alone
- LLM Probing with Contrastive Eigenproblems (2025): CCS as eigenproblem
