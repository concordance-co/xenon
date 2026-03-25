# Action-Framing Diagnostic Review

## Context

This note reviews the remote branch `origin/codex/action-framing-diagnostic`
without switching branches locally.

I did **not** audit all `331` changed files. I focused on the pieces that seem
to carry the actual scientific claim:

- `TRANSFER_ONTOLOGY_SUMMARY_2026-03-20.md`
- `docs/transfer_policy_dissociation_20260323.md`
- the cited behavioral summary JSONs
- the cited representation summary JSONs
- `pipelines/interp/synth/render_transfer_ontology.py`
- `pipelines/interp/behavioral.py`

The goal here is not to restate everything in the branch. It is to answer:

1. What does the experiment actually support?
2. What is likely overclaimed?
3. Is this already something people would want to hear/read about?

## Strongest Honest Claim

The strongest version of the result is:

> In this synthetic transfer benchmark, prompt shell / action framing changes
> behavior a lot, while resource family remains decodable in hidden states under
> those different shells.

More concretely:

- `direct_request`, `review_request`, and `ops_ticket` produce materially
  different behavioral rates on the same narrow ambiguous third-party transfer
  slice.
- the model still linearly encodes resource family strongly under both
  `direct_request` and `ops_ticket`.

That is real and useful.

## What It Does **Not** Yet Show

I do **not** think the current evidence supports the stronger phrasing:

> the model applies different policies over one stable resource manifold

Why not:

1. the bridge evidence is separate decodability, not shared-subspace evidence
2. the bridge is not causal
3. the representation benchmark still has lexical leakage

So the safe claim is:

> framing changes behavior without erasing decodable resource information

That is weaker than:

> the same stable internal manifold is being reused differently by policy

## Main Review Findings

### 1. The bridge result is weaker than the write-up makes it sound

In `docs/transfer_policy_dissociation_20260323.md`, the write-up moves from:

- both framings have high resource-family decodability

to:

- framing changes how a stable resource representation is used behaviorally

That step is too strong for the evidence currently shown.

What the bridge actually contains:

- one probe suite on `direct_request`
- one probe suite on `ops_ticket`
- both on `last_token`
- both with `144` examples
- both using separate probes

So the bridge shows:

- resource family is still easy to decode under both shells

It does **not** show:

- same coordinates
- same subspace
- same representation geometry
- or causal use of that representation

Relevant refs:

- `origin/codex/action-framing-diagnostic:docs/transfer_policy_dissociation_20260323.md:120-149`
- `origin/codex/action-framing-diagnostic:data/analysis_results_transfer_representation/transfer_behavioral_rep_bridge_direct_request_20260323/summary.json:1-33,122-133`
- `origin/codex/action-framing-diagnostic:data/analysis_results_transfer_representation/transfer_behavioral_rep_bridge_ops_ticket_20260323/summary.json:1-33,122-133`

### 2. The "neutralized" representation benchmark is not very neutral

The representation spec correctly says that if `resource_type` is decodable only
because the prompt literally says things like `$500` or `files`, that is not a
strong semantic result.

But the actual `neutralized_carrier` render still includes very strong
family-specific language such as:

- `authorization handling`
- `credential handling`
- `contract handling`
- `policy handling`
- `customer record handling`

That means the lexical-control story is weaker than intended.

This does **not** kill the representation result entirely. But it does mean the
benchmark is still leaking resource-family semantics through surface language.

Relevant refs:

- `origin/codex/action-framing-diagnostic:TRANSFER_ONTOLOGY_REPRESENTATION_SPEC.md:92-101`
- `origin/codex/action-framing-diagnostic:pipelines/interp/synth/render_transfer_ontology.py:320-369`

### 3. Cross-style transfer is only partial, not a clean stable-manifold result

The Phase A v2 style-projected benchmark is the best evidence that resource
family representation is real.

But the strongest cross-style `resource_type` results I found were:

- `typed_to_neutralized_cross_style`: accuracy `0.3000`, selectivity `0.2333`
- `neutralized_to_typed_cross_style`: accuracy `0.5708`, selectivity `0.4750`

That is clearly above chance, so there is signal.

But it is not close to the kind of saturated cross-style invariance that would
justify a very strong "stable manifold" framing.

Relevant refs:

- `origin/codex/action-framing-diagnostic:data/analysis_results_transfer_representation/transfer_representation_phase_a_v2_balanced_style_projected_20260322/summary.json:761-970`

### 4. The behavioral action-framing result is real, but narrow

The template-matrix benchmark is a narrow slice:

- `render_style = typed_specific`
- `action_mode = execute`
- `ownership = third_party`
- `authorization = ambiguous`
- `reversibility in {reversible, irreversible}`
- `instruction_style = naturalized_v3`

So the strongest behavioral result is not:

> framing changes transfer policy in general

It is:

> framing changes behavior a lot on this specific ambiguous third-party execute
> slice

That is still interesting. But it should stay narrow unless broader slices also
check out.

Relevant refs:

- `origin/codex/action-framing-diagnostic:data/synth/templates/transfer_ontology_behavioral_family_v3_ambiguous_template_matrix.json:5-46`
- `origin/codex/action-framing-diagnostic:docs/transfer_policy_dissociation_20260323.md:69-118`

### 5. The behavioral headline depends on freeform regex scoring

The cited action-framing summaries are from the `freeform` surface, and the
behavioral scorer uses regex-based pattern matching for:

- confirmation
- clarification
- refusal
- execute-like language
- warning intensity

That is fine for an internal diagnostic.

But it means the strongest headline should not pretend this is a perfectly clean
behavioral measurement surface. Template changes can also change response style,
which can interact with regex scoring.

This is a caution, not a fatal flaw. The effect sizes are large enough that I do
not think the result is *only* a scoring artifact. But it does reduce how hard
I would lean on the behavioral exactness of the comparison.

Relevant refs:

- `origin/codex/action-framing-diagnostic:pipelines/interp/behavioral.py:66-130`
- `origin/codex/action-framing-diagnostic:docs/transfer_policy_dissociation_20260323.md:69-118`

### 6. The older summary reads closer to the truth than the newer dissociation note

The branch-level summary says:

- authorization is the dominant stable behavioral axis
- reversibility is second
- action framing is real
- resource-type effects are too weak / unstable / entangled to treat as a clean
  probe target

That feels like the most defensible overall reading of the branch.

The newer dissociation note is not wrong, but it is a more optimistic
reinterpretation of the evidence than I think the current artifacts fully earn.

Relevant refs:

- `origin/codex/action-framing-diagnostic:TRANSFER_ONTOLOGY_SUMMARY_2026-03-20.md:40-59`
- `origin/codex/action-framing-diagnostic:TRANSFER_ONTOLOGY_SUMMARY_2026-03-20.md:61-90`
- `origin/codex/action-framing-diagnostic:TRANSFER_ONTOLOGY_SUMMARY_2026-03-20.md:146-161`

## Bottom Line

My blunt take:

- as an **internal research result**, this is useful and real
- as an **external / talk-worthy mech-interp result**, it is not there yet

What I think people would reasonably want to hear:

- we probably killed the strong "money is special" story in this benchmark
- authorization and reversibility dominate
- framing matters a lot
- resource identity remains decodable even when framing changes behavior

What I do **not** think is ready yet:

- a strong "stable resource manifold plus policy selection" claim

## Best Use Of This Branch Right Now

I think this branch is strongest as:

1. a useful anti-result on the original resource-specialness question
2. a benchmark-design lesson about framing effects
3. a setup for the next, stronger experiment

## What I Would Check Next

If the team wants to decide whether this can graduate from "interesting internal
note" to "real result," I would check:

1. **Cross-frame probe transfer directly**
   - Train on `direct_request`, test on `ops_ticket`
   - Train on `ops_ticket`, test on `direct_request`
   - Not just separate within-frame probes

2. **Subspace alignment**
   - Compare whether the same resource-family directions appear in both shells
   - RSA / CKA / subspace overlap would already be a step up

3. **Tool-call behavioral measurement**
   - Re-run the key framing comparison with tool-call outputs rather than
     regex-scored freeform text

4. **Actually neutralize the carrier**
   - Remove family-specific words like `authorization`, `credential`,
     `contract`, `policy`, `customer record`

5. **Causal patching**
   - If the team wants the "policy over stable representation" claim, patching
     is what would start to make that real

## Short Version To Share Verbally

If I had to summarize this branch in a few sentences:

> The branch does not rescue a strong money-special story. What it does show is
> that action framing changes behavior a lot on a controlled synthetic transfer
> slice, while resource family remains decodable in hidden states under those
> framings. That is interesting, but the current bridge is still decodability,
> not mechanistic evidence that one stable resource manifold is being reused by
> different policies.
