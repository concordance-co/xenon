---
project: MOREBENCH
subproject: theory_persona_vectors
artifact: hypothesis_catalog
created: 2026-04-27
last_updated: 2026-04-27
status: living_document
---

# Theory Persona Vectors — Hypothesis Catalog

Cross-phase reference. Generated as a step-back after phase_01 (deont-only
null), phase_02 (4-theory null with locus + paired-analysis corrections),
and phase_03 (response-mean directions stable but mutually high-cosine,
prompt-end null at L32).

This doc lists candidate hypotheses worth running through the flywheel
(see `xenon/methodology/FLYWHEEL.md`). It is a living document — append
new hypotheses, update status as phases close, mark closed-out ones with
their phase-exit reference.

## Status table

| ID | Hypothesis | Status | Cost | Information | Last touched |
|---|---|---|---|---|---|
| H1 | Moral *considerations* are the right latents, not moral *theories* | proposed | medium | high | 2026-04-27 |
| H2 | Moral-conflict detection is the load-bearing latent | proposed | low | very high | 2026-04-27 |
| H3 | Justification-style features generate theory expression as downstream effect | proposed | low | high | 2026-04-27 |
| H4 | PEFT-anchored direction is cleaner than ICL-anchored | proposed | medium | very high | 2026-04-27 |
| H5 | SAE feature mining > residual direction extraction | proposed | high (or low if SAE available) | high | 2026-04-27 |

Closed hypotheses (carried forward from phase exits):

| ID | Hypothesis | Phase that closed it | Verdict |
|---|---|---|---|
| H0a | Persona-vectors primed-vs-default recipe extracts deontology direction at L32 prompt_end | phase_01, phase_02 | rejected at chosen locus; theory directions exist but are entangled and live in response-mean residuals, not prompt-final |
| H0b | Pivot to theory-vs-theory contrastive directions | phase_02 (under paired analysis) | rejected; alt constructions have negative gaps at L32 generated |
| H0c | Substrate too terse to test the recipe | phase_03 | partially rejected; brief-recommendation substrate (median 80 tokens) shows stable response-mean directions but mutual cosines remain high |

## Orientation

After three phases, the consistent finding is: **theory primes produce
real, stable activation differences (paired sign-flip null gap 0.42–0.45),
but those differences are mutually high-cosine across theories
(0.71–0.90), absent at L32 prompt-end, present at L32 response-mean, and
do not separate cleanly into theory-specific axes.**

Three reads from the literature inform the next iteration:

- **"Personality directions are entangled. Trait purification is needed."**
  ([Personality as a Probe](https://arxiv.org/abs/2509.04794)) The
  cross-theory cosine clique is the entanglement signature. ICL primes
  give entangled directions; PEFT or contrastive subspace methods give
  cleaner ones. We have only tried ICL.
- **"Emotion concepts are locally scoped functional variables, not just
  style."** ([Anthropic emotions](https://transformer-circuits.pub/2026/emotions/index.html);
  [Valence-Arousal Subspace](https://arxiv.org/abs/2604.03147)) The
  right framing for moral content may be the same: not "deontology
  persona" but "constraint-respecting / welfare-aggregating /
  character-evaluating" as locally-scoped functional variables. Low-D
  geometry via PCA, validated by monotonic steering.
- **"Truth has a domain-general direction plus domain-specific
  directions, with Mahalanobis cosine predicting cross-domain transfer
  at R² 0.98."** ([Truthfulness Spectrum](https://arxiv.org/abs/2602.20273))
  If moral concerns work the same way, our cross-theory clique is the
  domain-general moral-prime direction with theory-specific residuals.
  Mahalanobis cosine becomes a tool for predicting which directions
  transfer across substrates.

The thread these share: **the primitive concept may be more granular
than "moral theory" — moral considerations, justification styles, or
conflict signals — and the named theories are aggregations over those
primitives.** That reframing is consistent with the cross-theory cosine
clique, with paper 17's trait-purification motif, and with the
Anthropic emotions paper's locally-scoped-functional-variables framing.
It is the strongest single shift available to the research program
right now.

---

## H1: Moral *considerations* are the right latents, not moral *theories*

**Status:** proposed
**Cost:** medium
**Information value:** high
**Last touched:** 2026-04-27

### Paper anchors

- [Valence-Arousal Subspace in LLMs](https://arxiv.org/abs/2604.03147) —
  211k-labeled-text PCA + ridge regression for low-D geometry; method
  template
- [Anthropic emotion concepts](https://transformer-circuits.pub/2026/emotions/index.html) —
  171 emotion concepts as locally-scoped functional variables
- [Personality as a Probe](https://arxiv.org/abs/2509.04794) — trait
  purification for entangled directions
- [Constrained Belief Updates / Simplex](https://arxiv.org/abs/2502.01954) —
  geometric structures arise from constrained update problems

### Claim

Qwen3-30B-A3B does not represent moral theories as such. It represents a
small set of more primitive moral considerations
(welfare-aggregation, constraint-respect, character-integrity,
fairness-justifiability, possibly autonomy-respect), and theory primes
reweight these considerations during generation.

### Flywheel path

1. **Stage 1 (real-data grounding):** read MoReBench rubric dimensions
   directly. Identify which rubric items map to internal model state
   versus external fact-checking. **MoReBench rubric is the natural
   source of latents and we have not actually used it.**
2. **Stage 2 (synth representation):** prime with single-consideration
   emphases at varying strengths. Five primes × three strength levels
   × 30 dilemmas = 450 conditions. Synth designed to break the
   named-theory framing.
3. **Stage 3 (synth validation):** cheap baseline check, lexical
   confound check per consideration. The strength axis gives a
   built-in dose-response control.
4. **Stage 4 (smoke):** L24 / L32 / L40, both prompt-end and
   response-mean, paired sign-flip null. Test: do
   single-consideration directions have *lower* mutual cosine than
   theory directions did?

### Pass / fail

- Pass: 5 single-consideration directions with mutual cosines ≤ 0.50,
  each with paired-null gap ≥ 0.20 at the chosen locus.
- Fail: same entanglement pattern as theories (cosines 0.7+) → the
  entanglement is a fact about the model, not about the named-theory
  framing.

---

## H2: Moral-conflict detection is the load-bearing latent

**Status:** proposed
**Cost:** low (mostly reuses phase_04 conflict-set work)
**Information value:** very high
**Last touched:** 2026-04-27

### Paper anchors

- [Detecting High-Stakes Interactions with Activation Probes](https://arxiv.org/abs/2506.10805) —
  high-stakes triage layer
- [Can You Trust an LLM with Your Life-Changing Decision?](https://arxiv.org/abs/2507.21132) —
  cautiousness vector, clarifying-question behavior correlates with
  judged safety
- [Constrained Belief Updates / Simplex](https://arxiv.org/abs/2502.01954)

### Claim

There is a "moral difficulty" signal in residual space that is much
cleaner than any theory-specific signal. Theory primes work in part by
amplifying this signal more than by conjuring a theory. A
"conflict-axis" signal (welfare-vs-rights, individual-vs-collective,
short-vs-long-term) may also be present as a more granular structure.

### Flywheel path

1. **Stage 1:** from MoReBench, partition dilemmas into (clear-answer,
   low-conflict) versus (high-conflict). Behavioral check: does the
   model's response style change between them — hedging, clarifying,
   asserting?
2. **Stage 2:** synth dilemmas at 4 conflict-difficulty levels (none,
   mild, strong, no-clear-answer). Four 30-dilemma blocks, single
   neutral prime each.
3. **Stage 4:** extract directions for "moral difficulty" and (if
   warranted) "conflict-axis." Test against null and against existing
   theory-prime directions for separability.

### Pass / fail

- Pass: moral-difficulty direction has paired-null gap ≥ 0.30, and its
  cosine with the existing theory-prime directions is ≤ 0.40 (i.e.,
  it is a separate axis from theory choice).
- Fail: moral-difficulty direction is itself entangled with the
  moral-prime clique. Then this is just the same shared moral-prime
  direction we have been measuring, recharacterized.

### Why this is high-value

A clean moral-difficulty direction would be a much better Concordance
product surface than a theory direction. The conflict-set labeling
work in phase_04 is already half of Stage 1 for this hypothesis.

---

## H3: Justification-style features generate theory expression as downstream effect

**Status:** proposed
**Cost:** low (hand-coding step is ~$5 in API)
**Information value:** high
**Last touched:** 2026-04-27

### Paper anchors

- [LLM Assertiveness can be Mechanistically Decomposed](https://arxiv.org/abs/2508.17182) —
  assertiveness splits into emotional + logical components
- [Personality as a Probe](https://arxiv.org/abs/2509.04794) —
  entanglement and purification methods
- [Anatomy of Alignment / FSRL](https://arxiv.org/abs/2509.12934) —
  steering sparse features rather than holistic preferences

### Claim

The model has primitive features for "appeals to outcomes," "appeals to
rules," "appeals to character," and "appeals to consensus." Theory
primes do not write a "theory" into the residual stream; they reweight
these primitive features. The theory direction is an aggregation over
them, not a feature in its own right.

This is a more aggressive version of H1 in that it commits to a
specific decomposition (4 justification-style primitives) rather than
letting PCA find it.

### Flywheel path

1. **Stage 1:** hand-code (or LLM-judge with fixed rubric) every
   phase_03 response on the 4 justification-style axes. Cheap; ~$5 in
   API; ~1 hour. **This is also the missing behavioral-sanity layer
   that everything downstream needs.**
2. **Stage 2:** 4 single-feature prompts, crossed against the existing
   30 dilemmas. Test whether the existing theory primes' behavioral
   signatures decompose as predicted (e.g., util ≈ high welfare + low
   duty + low character + low fairness).
3. **Stage 4:** extract single-feature directions. Test whether
   theory-vs-neutral directions equal weighted sums of single-feature
   directions in residual space.

### Pass / fail

- Pass: theory-prime activations are well-modelled (R² ≥ 0.70) by a
  weighted combination of single-feature directions. Theory persona
  vectors as such don't exist; four interpretable justification-style
  vectors do.
- Fail: theory-prime activations contain substantial residual variance
  not explained by the four justification-style features, suggesting
  theory-specific representation beyond these primitives.

---

## H4: PEFT-anchored direction as the cleaner reference

**Status:** proposed
**Cost:** medium (~$10 in compute for 4 small LoRA fits)
**Information value:** very high (methodologically critical control)
**Last touched:** 2026-04-27

### Paper anchors

- [Personality as a Probe for LLM Evaluation](https://arxiv.org/abs/2509.04794) —
  compares ICL, PEFT, mech steering for personality control; finds
  PEFT-controlled traits give cleaner reference directions

### Claim

ICL (prompt priming) and PEFT (fine-tuning a tiny adapter to behave
deontologically) extract different directions in residual space.
Without comparing them, every "we extracted a deontology direction"
claim is conditioned on ICL being a faithful elicitation method, which
paper 17 says it often isn't.

### Flywheel path

1. **Stage 2:** train a tiny LoRA on Qwen3-30B for each of the 4
   theories on a small targeted training set. ~30 min × 4 on Modal
   H200.
2. **Stage 4:** extract residuals from the LoRA-adapted model on the
   same dilemmas with neutral prompts. Compare LoRA-extracted
   "deontology direction" to ICL-extracted "deontology direction."

### Pass / fail

- Pass: cosine ≥ 0.50 between PEFT-direction and ICL-direction → ICL
  was finding the same thing PEFT does. The persona-vector recipe is
  faithful to internal model representation.
- Fail: cosine ≤ 0.30 → ICL was finding something else (probably
  entangled with prompt-token residue or instruction-following
  artifacts). All prior phases need recharacterization, and the PEFT
  direction becomes the new reference.

### Note

This is the methodologically critical control we have been missing.
Worth running as a separate parallel track because it requires
substantially different infrastructure (LoRA training rather than
direction extraction). Does not block H1–H3.

---

## H5: SAE feature mining instead of residual direction extraction

**Status:** proposed
**Cost:** high if no SAE available, low if one exists
**Information value:** high
**Last touched:** 2026-04-27

### Paper anchors

- [Mechanistic Knobs in LLMs](https://arxiv.org/abs/2601.02978) —
  contrastive semantic-retrieval over SAE features
- [Faithful RAG with SAEs / RAGLens](https://arxiv.org/abs/2512.08892) —
  mid-layer SAE features beat residual baselines for unfaithfulness
  detection
- [Anatomy of Alignment / FSRL](https://arxiv.org/abs/2509.12934) —
  steering SAE features for preference optimization
- [Goodfire trillion-parameter SAE infra](https://www.goodfire.ai/blog/interpretability-infra-at-frontier-scale) —
  scale infrastructure for SAE-based work

### Claim

When residual-direction methods give entangled or non-transferable
directions, SAE features tend to be more interpretable and more
cleanly separable. Our entanglement problem is exactly the regime where
SAE features tend to outperform.

### Flywheel path

1. **Stage 1:** check whether a Qwen3-30B-A3B SAE exists publicly
   (Goodfire, Anthropic, EleutherAI, or other). 30-minute scout.
2. **Stage 2 (if SAE available):** mine the SAE feature space for
   moral-concept features using contrastive retrieval — theory-primed
   activations vs neutral activations → which features fire
   differentially?
3. **Stage 4:** use the mined features instead of residual directions.
   Steer via feature activation rather than direction injection.

### Pass / fail

- Pass: small set of SAE features (≤ 20) cleanly differentiate theory
  primes from neutral, with cosine separability that residual
  directions lacked. Steering via these features produces clean
  behavioral effects.
- Fail: SAE features for moral concerns are not strongly active or
  cleanly separable on this substrate. Either the SAE is wrong scope,
  or moral concepts are not represented as SAE features in this model.

### Note

Longer-horizon. Worth a 30-minute scout to check existing SAE
availability before committing. If we end up training one ourselves,
treat as a separate subproject — not a phase here.

---

## Cross-cutting methodology lessons

These bind every hypothesis above. Derived from `xenon/methodology/PRINCIPLES.md`
and `xenon/methodology/CHECKS.md` plus the phase_01–03 corrections.

1. **Behavioral sanity first** (PRINCIPLES.md §1). Every hypothesis
   needs a Stage 1 read-the-real-data pass before Stage 2. Phase_01
   and phase_02 were weak on this; the conflict-set work in phase_04
   was a partial fix; the rubric-dimension read is still pending and
   it is what H1 and H2 both need.

2. **Stack two confound-reduction techniques** (PRINCIPLES.md §12).
   The four techniques are: viewport reduction, training-distribution
   variation, lexical-subspace subtraction, target reformulation.
   We have done viewport reduction (first_16, pre-theory-window
   filter). We must add at least one more for any response-side claim.
   **Lexical-subspace subtraction (residualize against TFIDF
   prediction) is the cheapest one we haven't done.**

3. **Read layer ≠ write layer** (PRINCIPLES.md §3). Phase_03 found
   theory state at L32 generated. Steering should not assume L32 is
   the right *write* layer; sweep multiple layers for steering
   injection separately.

4. **Nuisance-stratified cell size matters** (PRINCIPLES.md §8). The
   N=10 conflict cases is at the edge of what supports a 30%-effect
   pre-reg. Any new hypothesis must front-load the cell-size check;
   H1 with 5 considerations × 3 strengths × 30 dilemmas = 450 cells
   but per-condition N is still only 30.

5. **One-split success is not transfer** (PRINCIPLES.md §11).
   Whatever passes synth smoke must transfer to MoReBench held-out,
   not just within-substrate. Any direction we promote should be
   checked against Mahalanobis-cosine prediction (Truthfulness
   Spectrum paper) for cross-substrate transferability before being
   advanced up the evidence ladder.

6. **Locus discipline.** Phase_01 / phase_02 inherited L32 prompt_end
   from prior MoReBench prompt-side work without re-deriving from
   the persona-vectors method. Future pre-regs in this subproject
   should derive the primary locus from the *method* (persona-vectors
   → response tokens) and report cross-locus consistency as a
   geometry/mechanistic discriminator (prompt-final cross-locus
   cosine ≥ 0.50 with response-mean is required for the
   genuine-state claim).

## Sequencing recommendation

Run **H3** and **H2** first. Both can be initiated within the next week
with existing artifacts. H3 is the cheapest path to a real result;
hand-coding phase_03 responses on 4 justification-style axes produces a
behavioral-sanity layer that supports everything downstream. If theory
primes decompose cleanly into the 4 axes, the entire research program
reframes from "find theory persona vectors" to "find justification-style
features and validate them as Concordance behavioral-integrity surfaces."

H2 reuses the conflict set already labeled in phase_04. The Stage-1
read of MoReBench rubric dimensions is overdue and informs whether H1
is testable as designed.

H4 is the highest-information control if any future claim about
ICL-extracted directions is to be defensible. Worth scheduling as a
separate phase_04-parallel track.

H1 stays as-is — formalize as a phase_05 or phase_06 candidate after
H2 and H3 inform the design.

H5 is a longer-horizon option pending SAE availability.

---

## Decision log

| Date | Event | Phase / artifact |
|---|---|---|
| 2026-04-27 | Initial catalog created after phase_01–03 step-back | this doc |

