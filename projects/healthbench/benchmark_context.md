# HealthBench Context

**Version:** `v0`
**Freeze Date:** `2026-04-28`

Canonical shared rules live in:

- [methodology/PRINCIPLES.md](../../methodology/PRINCIPLES.md)

## 1. Benchmark Snapshot

- Name: HealthBench.
- Source repo: https://github.com/openai/simple-evals
- Benchmark article: https://openai.com/index/healthbench/
- Primary eval file: `healthbench_eval.py`.
- Public scale reported by OpenAI: 5,000 realistic health conversations and
  48,562 unique rubric criteria.
- Initial project slice: HealthBench Consensus.
- Consensus scale reported by OpenAI: 3,671 examples with heavily filtered
  criteria validated against physician consensus.
- Hard scale reported by OpenAI: 1,000 examples selected for frontier-model
  difficulty.
- Initial probe-target model: `/models/Qwen/Qwen3-30B-A3B`.
- Scope convention for this context: unless otherwise noted, claims are about
  source documentation and schema-level inspection available on 2026-04-28,
  before a Neon ingest or row-level ontology freeze.

## 2. Why It Matters

HealthBench is product-relevant because it targets high-impact health assistant
behavior rather than exam-style medical QA. The benchmark asks for a best
assistant response to realistic individual-user or clinician conversations and
grades each response using physician-written, example-specific rubrics.

For mech-interp work, its value is the combination of:

- high-stakes user intent and safety requirements
- multi-turn and multilingual conversations
- layperson and clinician personas
- weighted positive and negative rubric criteria
- theme-level and axis-level evaluation structure
- a consensus subset with stronger physician-validation signal

## 3. Native Label Surfaces

Schema surfaces visible from the public implementation:

- `prompt_id`
- `prompt`: chat-message list
- `example_tags`: example-level theme/category tags
- `rubrics`: criterion, point value, and rubric-item tags
- `ideal_completions_data`: optional physician completion metadata

Rubric tags include axis-style structure in the public eval code path. Positive
criteria define desired response properties; negative point criteria define
undesired behavior and must be interpreted as criterion-met, not goodness-met.

## 4. Refined Latent Label Spec

Not frozen yet. Phase 01 should convert native fields into a minimal first-pass
ontology.

Promising candidates:

- prompt-side: underspecification/context-seeking pressure, emergency referral
  pressure, clinician-vs-layperson framing, specialty/domain, language/locale
- response-side: safety escalation, calibrated uncertainty, context-seeking
  before advice, answer-depth calibration, expertise-tailored communication
- objective orientation: immediate helpfulness vs safety/triage caution,
  confident direct answer vs scoped guidance
- nuisance set: topic, language, prompt length, turn count, persona, rubric
  count, source subset, obvious emergency keywords

## 5. Known Confounds

- Health topic and safety label are likely correlated: emergency-referral rows
  can be text-recoverable from surface medical terms.
- Rubric wording may leak labels if included in model prompts or response-side
  labelers.
- Response-side labels will be lexically confounded unless the probe viewport,
  training variation, or residualization strategy is designed explicitly.
- Consensus filtering improves label trust but may narrow the behavioral
  distribution.
- Hard filtering is partly model-performance-conditioned and should not be
  treated as an independent health-risk label.
- Example-specific rubric criteria are outcome-grade surfaces, not automatically
  mechanistic latent labels.

## 6. Behavioral Sanity Notes

Checked so far:

- public source links exist
- `healthbench_eval.py` exposes full, consensus, and hard blob paths
- schema-level row structure was inspected without saving raw data locally
- initial project slice and probe-target model were chosen
- HealthBench Consensus was uploaded to Neon table
  `healthbench_consensus_v1` with 3,671 rows

Not checked yet:

- Qwen generation quality on HealthBench Consensus
- parser stability for Qwen responses
- HealthBench grader agreement on Qwen responses
- label derivability and cheap text baselines
- nuisance-stratified cell sizes after the first label ontology

## 6.1 Benchmark-Specific Gotchas

- OpenAI asks researchers not to reveal dataset examples in plain text or
  images online. Reports should use aggregate summaries and internal row ids.
- The eval scorer normalizes achieved positive and negative rubric points by
  total positive points, then clips aggregate means to `[0, 1]`.
- The public `simple-evals` repo is a reference implementation and its README
  says it is no longer maintained for new model results after July 2025.
- Consensus and Hard are source subsets, not separate task schemas.

## 7. Strong Candidate Feature Hypotheses

To be discussed before Phase 01 closes. Plausible starting families:

- context-needed state before answering underspecified health questions
- emergency escalation / triage urgency state
- scope-calibration state for medical advice under uncertainty
- persona-aware communication state for clinician vs layperson prompts
- response-depth control state for complete-but-not-oververbose answers

## 8. Methods That Look Promising

- text baselines before activation probes, especially for topic and emergency
  shortcut checks
- linear residual probes over prompt-end and early response positions
- layer and token-position sweeps before choosing a measurement locus
- transfer probes between Consensus and Hard once labels are frozen
- contrastive steering only after a clean behavior-divergence target exists

## 9. Methods Or Hypotheses To Be Careful About

- treating overall HealthBench score as a single latent
- probing response text for rubric axes without lexical controls
- using Hard membership as if it were a native clinical-risk label
- interpreting late-layer probe strength as causal leverage
- mixing rubric text into generation prompts in a way that leaks target labels

## 10. Data Gap List

- frozen row-level latent labels for Consensus
- Qwen generations under a fixed protocol
- HealthBench grader outputs for those generations
- cheap text-baseline reports for proposed labels
- candidate matched or counterbalanced subsets if native Consensus confounds are
  too strong

## 11. Open Questions

- Which latent family should be the first causal candidate?
- Should Phase 01 freeze a prompt-side label first, or generate Qwen responses
  and freeze a response-side objective-orientation label?
- Which loci are visible enough to transfer from Consensus to Hard and later
  product-like traces?
