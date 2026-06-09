# Benchmark Craft Reference

Use this reference when the synthetic task needs stronger controls, clearer
splits, or a deterministic generator contract.

## Optimization Targets

Synthetic data should be:

- behaviorally sane
- experimentally valid
- auditable by humans
- controllable along the dimensions that matter
- hard to solve with trivial lexical shortcuts
- easy to analyze later

The craft target is not "remove all easy words." It is:

- preserve the semantic distinction
- break the cheapest shortcut
- verify that a careful human could still recover the intended label

## Anti-Shortcut Menu

Prefer repairs like:

- paraphrased-content variants that keep semantics fixed while varying surface
  form
- held-out alias generalization, such as canonical name vs descriptive alias
- cross-label hard negatives where surface cues pull one way and the intended
  semantic label pulls another
- factorial surface designs, such as `name x anchor x position`
- shared-vocabulary anchors across labels
- content-removed or name-removed variants that preserve enough information for
  a careful human to recover the intended label

## Workflow

### 1. Start from a real workflow, then simplify

The best synthetic benchmarks often start from a real dataset, workflow, or
failure mode, then abstract it into a smaller controlled task.

Record:

- the real pattern that inspired the benchmark
- core elements preserved
- messy details removed
- why the simplified task still tests the same decision

Simplify toward the question, not toward convenience. If simplification makes
the task solvable from one keyword, it went too far.

### 2. Write the latent variable first

Before writing prompts, define:

- the exact variable the benchmark is trying to isolate
- what counts as positive and negative
- what should not change between labels

If the latent variable is relational, the prompt must force a relational
computation.

### 3. Define the minimal computation

Ask:

- what information must the model combine?
- in what order does it see that information?
- what trivial shortcuts would avoid the intended computation?

For authority-applicability tasks, for example, the intended computation might
be: read the control map, read the claim, read the requested action, compare the
claim against the governing rule.

### 4. Design the negative class carefully

Avoid lazy negatives:

- random wrong strings
- obviously irrelevant domains
- malformed or unnatural text

Use plausible but wrong negatives in the same broad domain and format. The
wrong answer should still be believable inside the prompt world.

### 5. Add lexical controls early

At minimum vary:

- entity surface forms
- action surface forms
- template families
- instruction wording

Reserve at least one lexical bundle entirely for test when claiming abstraction
over vocabulary.

### 6. Make the split reflect the claim

Common split types:

- lexical holdout
- action holdout
- domain holdout
- carrier holdout

Do not claim abstraction across a dimension that was never held out.

### 7. Build matched pairs when causal work is likely

Ideal matched pairs share:

- template family
- carrier family
- action family
- control map or environment structure

Only the decisive target variable should change.

For instruction-conflict data, distinguish aligned controls from
source-disambiguation supervision. Useful audit labels include:

- `aligned_agreement`
- `strategy_followed`
- `setting_followed`
- `mixed_or_neither`

### 8. Freeze inventories before writing the generator

Lock named inventories for:

- strategy wording
- setting wording
- environment templates
- lexical split assignments
- output schemas

Useful generator sequence:

1. write the generator contract
2. write a hand-audited example bank
3. define the exact row shape
4. implement the full generator

### 9. Keep prompts auditable by eye

A human should be able to inspect one row and answer:

- what is the correct label?
- why?
- what changed relative to its paired opposite-label row?

If a human cannot audit rows quickly, the dataset will be painful to debug and
easy to misinterpret.

## Behavioral Sanity Checklist

Before probing:

- inspect prompt examples from each class
- run the base model on a small sample
- verify outputs are parseable
- verify the model is solving the intended task at a reasonable rate
- inspect failures manually
- confirm failures are about the target variable, not formatting

If behavior is wrong, ask:

- is the prompt ambiguous?
- is role placement unnatural?
- is the requested output underspecified?
- is the model confused by carrier formatting?
- did the design accidentally create label leakage?
