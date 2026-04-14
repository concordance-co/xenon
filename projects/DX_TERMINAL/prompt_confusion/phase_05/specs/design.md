# Prompt Confusion Phase 05 Design

## Framing

Phase 04 established that the model builds a strong conflict detection
signal (92% balanced accuracy at layer 36) and that arbitration is
family-conditioned: `strategy_family + environment_pressure_bucket`
predicts resolution behavior at 80% balanced accuracy, and per-family
behavioral splits are strongly family-conditioned (two of four families
near-deterministic).

The Phase 05 working hypothesis reframes the Phase 04 "arbitration
confound" as a mechanistic finding: **the model resolves conflicts by
first identifying what type of conflict it is in, then applying a
family-specific resolution policy.** Phase 05 investigates the structure
of this branching mechanism:

- **Direction 1** tests whether conflict detection is a shared
  representation that generalizes across families, or is itself
  family-specific.
- **Direction 2** characterizes where and how the model branches from
  detection into family-specific resolution.
- **Direction 3** tests whether STRATEGY and SETTINGS token spans compete
  differently per family in the attention pathway.

This is a working hypothesis, not an established finding. Phase 05 is
designed to be able to falsify it: if detection does not transfer across
families (Direction 1), or families do not separate cleanly in geometry
(Direction 2), or competition patterns are uniform across families
(Direction 3), the branching interpretation does not hold and the Phase 04
confound reading remains the honest story.

---

## Direction 1: Cross-Family Probe Transfer

**Question.** Split into two sub-questions, which can have different
answers:

- **1a (detection transfer):** Does the aligned-vs-conflict signal transfer
  across families? Is "something is wrong" a shared representation?
- **1b (arbitration transfer):** Does the strategy-vs-setting resolution
  direction transfer across families? Is "follow this source" a shared
  representation?

The likely outcome under the branching hypothesis is **detection
transfers, arbitration does not** -- a shared "conflict exists" signal
followed by family-local resolution. That outcome would be the cleanest
version of "shared awareness, local resolution."

### Method

For both sub-questions:

1. Train a linear probe on one family grouping (size:
   `trade_size_force_large` + `trade_size_force_small`). Evaluate on the
   other grouping (activity: `activity_force_trade` +
   `activity_force_observe`). Reverse and repeat.
2. Within-training: stratified group k-fold by `matched_pair_id`, matching
   Phase 04 probe setup.
3. Run at all 12 captured layers. Interpret at layer 36 (Phase 04 detection
   peak) and layers 20/24 (arbitration-relevant).
4. Compute cosine similarity of probe weight vectors at each layer.
   **Same-layer comparisons only** -- probe weights at different layers
   live in different semantic spaces.
5. Run on **both residual stream and router logits** as parallel probes.
   Residual vs router can dissociate (e.g., detection transfers in residual
   but not router, or vice versa), which is itself informative about where
   the shared representation lives.

### Measuring "transfer" quantitatively

Transfer must be reported as a **delta against within-family baseline**,
not an absolute number. An absolute 65% cross-family accuracy is
ambiguous without knowing what within-family training achieves at the
same layer.

For each (data source, layer, direction) combination, report:

- **Within-family baseline:** train and evaluate within size families
  (stratified group k-fold). Same for activity families.
- **Cross-family transfer:** train on one family grouping, evaluate on the
  other (as described above).
- **Transfer delta:** cross-family minus within-family. A small delta
  means the signal generalizes; a large delta means the representation
  is family-specific.

This framing avoids committing to an arbitrary absolute threshold while
still producing a clear quantitative claim.

### What results mean

- High transfer accuracy + small transfer delta + high cosine similarity
  → shared representation at that layer
- High transfer accuracy + small transfer delta + low cosine similarity
  → both families have decodable signal but encoded in different
  directions
- Large transfer delta → family-specific encoding; the Phase 04 result
  was leaning on family identity at that layer

---

## Direction 2: Branching Structure

Characterize how family identity emerges in activation space and where the
model branches from shared detection into family-specific resolution.

### 2a. PCA on conflict-only and full dataset

Take the 144 conflict rows. PCA on residual stream activations at layer 36.
Visualize PC1 vs PC2 colored by family. Compare against PCA on all 288
rows (aligned + conflict) to see whether alignment status or family
identity is the dominant axis of variation.

**Normalization:** RMS-normalize activations before PCA. Residual-stream
norms grow with depth; unnormalized PCA risks having PC1 dominated by norm
rather than content.

**Interpretation:**

- Families separating on PC1/PC2 in the conflict-only space → family
  structure dominates within the conflict subspace
- Alignment status dominating the all-rows PCA and family dominating the
  conflict-only PCA → the shared-then-branching story is visible directly

### 2b. Layer-wise family-identity probe (branching depth)

**Question:** At which layer does family identity become dominant in the
residual stream?

**Method:** Fit a 4-class linear probe (`strategy_family` as label) at each
captured layer on all 288 rows. Plot accuracy over depth. The saturation
point identifies where the branching mechanism becomes locally linearly
readable.

Run this probe on **both residual stream and router logits**, matching
Direction 1. If family identity saturates at a different layer in router
vs residual, that tells us whether branching happens via expert routing
or residual content (or both).

**Why this replaces "per-family arbitration probes":** Earlier drafts
proposed per-family arbitration probes with cosine-similarity comparison
across families. That design is untrainable on the Phase 04 dataset -- two
of four families have near-one-sided behavioral splits
(e.g., `trade_size_force_large`: 0 strategy / 21 setting), which leaves
per-family arbitration probes with no usable positive class. Under the
branching hypothesis, the right question is not "can we probe arbitration
per family" (family identity already determines resolution) but "where
does family identity become dominant" -- which this probe answers
directly and is feasible on existing data.

### 2c. LDA on conflict families

Fit a 4-class LDA on conflict-only activations with family as the label.
The discriminant axes define the "family subspace."

**Power caveat:** 36 rows per class is near the floor for stable LDA.
Treat the discriminant axes as **visualization-quality structure, not
quantitative claims.** The method is most useful to set up eventual
novelty detection -- a new conflict family introduced in a future phase
should project outside the hull of existing families if the branching
hypothesis holds.

**Normalization:** Same RMS normalization as 2a.

### Interpretation note: Phase 04 causal asymmetry

Phase 04 Stage 2 found `strategy_push` shifts behavior 9pp but
`setting_push` only 3pp, and size families move more than activity
families. Direction 2's geometry should be interpreted against this: if
activity families cluster far from the SETTINGS direction used for
patching, their non-response to `setting_push` has a geometric
explanation. This does not require a dedicated causal workstream in
Phase 05 -- it is an interpretation overlay on the geometry results.

---

## Direction 3: Logit Attribution for Constraint Competition

**Question:** Do STRATEGY and SETTINGS token spans exert opposing
influences on the model's output logit at the decisive token, and does the
balance vary by family?

This is the most ambitious direction and is scoped as a gated sub-project
within Phase 05.

### Infrastructure status

Phase 04 already built a Modal HF Transformers run in eager attention mode
with `output_attentions=True`
(`phase_04/scripts/modal_conflict_arbitration_analysis.py`, line 1227+).
That captures per-head attention weights. For full logit attribution we
additionally need:

- **Per-head value vectors V**: not exposed by `output_attentions`. Add
  forward hooks on the attention submodule's `v_proj` (or architecture
  equivalent).
- **W_O per head**: model weight, trivial to access.
- **Unembedding vector**: model weight, trivial to access.

The lift is an extension to the existing eager HF pass, not a new capture
pipeline.

### Method (adapted from Zeng et al. Section 2.2)

1. For each conflict row, identify two disjoint token spans: `T_strategy`
   (tokens within the STRATEGY section) and `T_settings` (tokens within
   the SETTINGS section).
2. Run the extended forward pass. For each attention head at each layer,
   compute the contribution of each source token to the **decisive
   token's** logit:
   - `c_t = Attn[p, t] · V_t · W_O`
   - `LA_t = ⟨w_y, c_t⟩`
3. Aggregate: `C_strategy = Σ_{t ∈ T_strategy} LA_t`, same for settings.
   Compute signed share: `S_strategy = C_strategy / (C_strategy + C_settings)`.
4. Analyze:
   - **Conflict detection via LA:** fraction of conflict rows with
     `sign(S_strategy) ≠ sign(S_settings)`. Opposing contributions =
     internally-represented competition. Compare against aligned rows.
   - **Resolution prediction:** among opposing-sign rows, does the winning
     share predict behavioral side-following?
   - **Per-family breakdown:** does opposing-sign rate and resolution
     reliability differ across families? Under the branching hypothesis,
     yes.

### Decisive-token selection

Output format is
`{"action": "...", "asset": "...", "size": "..."}`. The first generated
token is `{`, which is formatting, not decision. The decisive token
depends on family:

- Size families (`trade_size_force_*`): token inside the `"size"` field
  value (`small` / `medium` / `large`)
- Activity families (`activity_force_*`): token inside the `"action"`
  field value (`buy` / `sell` / `observe`)

LA must be computed at the decisive token per row, selected by family.

### Known limitations (in-scope to address in Phase 05)

**Section-order position bias.** All Phase 04 prompts have STRATEGY before
SETTINGS. Recency effects can contaminate attention-based LA results.
Phase 05 Direction 3 includes a **section-order swap condition** -- a
subset of rows re-rendered with SETTINGS before STRATEGY -- as a control.
Without the swap, attention-based conclusions are not defensible.

Note that the swap requires **new behavioral generation runs on the
swapped rows**, not just new LA captures. The model may resolve
differently when section order changes (plausible given recency bias),
so we need to re-label which side the model followed on the swapped
prompts before LA results on those rows can be interpreted. Concretely:
generate outputs on the swapped subset, re-publish a conflict-readout
view for the swapped condition, then run LA against that view's labels.

**MoE logit attribution gap.** Qwen3-30B-A3B has MoE routing. The
attention-based LA decomposition captures only the attention pathway's
contribution to the output logit. MLP/expert contributions are invisible
to this method. Phase 05 Direction 3 claims must be framed as "how
STRATEGY and SETTINGS tokens compete in the attention pathway," not "in
the model's computation."

---

## Sequencing

| Priority | Work | Dependency |
|---|---|---|
| First | Direction 1a/1b (cross-family transfer, residual + router) | None -- existing activations |
| Parallel | Direction 2a, 2b, 2c (geometry + branching depth) | None -- existing activations |
| Gated | Direction 3 (logit attribution + section-order swap) | Infrastructure extension: V-vector hooks on existing HF eager pass |

Causal patching follow-ups (bidirectional patching, per-family deltas,
push-strength calibration) are deferred to a later phase after these
structural results are in.

---

## Dataset Expansion (Gated Follow-Up)

Results from Directions 1 and 2 determine whether expansion is needed and
what shape it takes:

- **If detection transfers but arbitration does not** (likely under
  branching hypothesis) → add new families (timing, concentration, risk)
  to test generalization of the branching mechanism with held-out-family
  evaluation.
- **If neither transfers** → expansion may not help; reconsider benchmark
  design.
- **If both transfer** → branching hypothesis fails; evidence of a
  cross-family arbitration mechanism; focus shifts to mechanistic
  follow-ups on that.

Any expansion also doubles conflict rows per family (~60), which would
make per-family arbitration probes feasible -- an analysis we cannot run
on the Phase 04 data.
