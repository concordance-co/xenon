# Prompt Confusion Methodology Problems Before Phase 04

This note records the main methodological issues surfaced by the latest
Phase 03 review so the next dataset iteration has a clear target.

## 1. The grouped source probe is not really grouped

The source probe only uses one `strong_conflict` row per `matched_pair_id`.
That means:

- one row per pair
- one group per row
- grouped CV is almost the same as ordinary row-level CV

This makes the "grouped" source result much weaker than it sounds. The next
iteration needs a stricter independence unit, such as:

- setting lexical family
- strategy lexical family
- context template family
- or a broader prompt-template id

## 2. The mechanistic slice is dominated by one family

The current mechanistic slice defaults to only two families and then keeps only
quality-filtered pairs. In practice:

- `trade_size_force_large` survives the filter at very high rates
- `activity_force_observe` survives poorly

So the resulting slice is mostly a `trade_size_force_large` analysis rather
than a balanced prompt-confusion analysis.

## 3. Capture-generation may be making the probe problem too easy

The latest workflow captures with generation enabled while still pooling the
`last_token`. If the pooled token comes from generated output rather than the
prompt boundary, then the probe can partly read out the model's answer instead
of the pre-decision latent state.

Phase 04 should start with:

- `capture_generation = false`
- prompt-only activations
- explicit verification of which token is being pooled

## 4. The benchmark may still be too semantically loaded

Even in the minimal dataset, many strategies and settings carry very strong
polarity:

- aggressive
- patient
- churn
- maximum
- minimal

This creates an easy shortcut where the model or probe can key on sentiment or
semantic polarity rather than the intended structured conflict.

Phase 04 should reduce that sentiment gradient and use more neutral wording.

## 5. The task semantics may still be underspecified to the model

The prompt shell shows `STRATEGY` and `SETTINGS`, but the system prompt does
not clearly explain what each source is for or how they jointly constrain the
decision.

Phase 04 should make explicit that:

- `STRATEGY` describes the directional plan
- `SETTINGS` describe execution policy and constraints
- the model must consider both
- conflict between them is part of the task

## 6. The dataset is too large for first-pass debugging

Phase 03 jumped to a large dataset and then had to debug methodology on top of
that scale.

Phase 04 should start with a smaller reviewed dataset:

- target size: roughly `200-400` rows
- manually inspect examples before large capture
- only scale up once the benchmark is behaviorally sane and methodologically
  legible

## 7. The right next move

Phase 04 should be an iterative rebuild with:

- a smaller dataset
- explicit system guidance about strategy vs settings
- more neutral wording
- prompt-only capture first
- stronger split keys for source-following evaluation

The goal is not to make the benchmark harder for its own sake. The goal is to
make the measured variable more clearly match the intended computation.
