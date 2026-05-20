---
name: synthetic-data-generation
description: Use when designing or repairing benchmark and prompt data so later experiments are behaviorally sane and experimentally valid. Especially useful for lexical controls, split schemes, matched pairs, prompt role placement, relational task design, and preserving the real decision bottleneck.
---

# Synthetic Data Generation

Use this skill when the user wants to:

- design a new synthetic benchmark
- repair a synthetic benchmark that is behaving strangely
- create controlled prompt variants for probing or patching
- add lexical, carrier, or domain controls
- build synthetic datasets for agent monitoring, policy adherence, or mechanistic interpretability

This skill is for **experiment design**, not just text generation. The goal is to create data that the model can solve for the *right reasons*, and that later analysis can interpret cleanly.

For benchmark-first mechanistic interpretability, this skill is also the main craft library for breaking shortcut channels in imported benchmarks.

## Core rule

`Behavioral sanity comes before interpretability.`

Before treating probe results, patching results, or attention results as meaningful, first verify that the benchmark itself is behaviorally sane.

If the base task is malformed, ambiguous, or solved via role-format artifacts, later interpretability work will be misleading.

## What to optimize for

Synthetic data should be:

- behaviorally sane
- experimentally valid
- auditable by humans
- controllable along the dimensions you care about
- hard to solve with trivial lexical shortcuts
- easy to analyze later

## Primary benchmark-repair use case

Imported benchmarks leak by construction.
If a benchmark was designed so humans or graders could apply labels reliably, those labels are usually recoverable from surface semantics too.

One of the main jobs of synthetic-data-generation is therefore:

- break surface-label correlation
- preserve the semantic content a good human reader would still recover
- leave behind data where later probes must solve something more interesting than a keyword lookup

This is not downstream cleanup.
It is one of the primary crafts of benchmark-to-mech-interp work.

## Anti-shortcut craft menu

When a target family is shortcut-satisfiable, prefer repairs like these:

### 1. Paraphrased-content variants

Keep the semantic content fixed while varying the surface form.

### 2. Held-out alias generalization

Train on one surface realization of a concept and test on another:

- canonical name vs descriptive alias
- short alias vs long-form alias

### 3. Cross-label hard negatives

Create examples where easy surface cues pull one way while the intended semantic label pulls another.

### 4. Factorial surface designs

For labels with multiple possible surface carriers, vary them independently:

- name x anchor x position
- wrapper x content x role framing

### 5. Shared-vocabulary anchors

Use overlapping vocabulary across labels so a small keyword list cannot solve the task.

### 6. Content-removed or name-removed variants

Redact one shortcut channel while preserving the rest of the prompt:

- name removed, anchor preserved
- anchor removed, name preserved

These are good repairs only if a careful human could still recover the intended label.

## Worked examples

### Theory-label benchmarks

Weak anti-shortcut control:

- remove the theory name but keep a theory-specific anchor sentence that still identifies the framework perfectly

Stronger anti-shortcut design:

- factorial family with:
  - name + anchor
  - anchor only
  - name only
  - alias-based paraphrases
  - held-out aliases or masked-name evaluation

`Anchor only` is useful here only as one cell in a larger factorial decomposition.
By itself it is not a credible anti-shortcut test if the anchor sentence still identifies the framework cleanly.

### Counseling-style empathy labels

Weak anti-shortcut setup:

- positive class contains obvious crisis or reassurance vocabulary

Stronger anti-shortcut design:

- paraphrase the emotional language
- create same-topic hard negatives
- hold out obvious empathy phrases at test time
- preserve the counseling situation while weakening direct lexical markers

### General rule

The craft target is not "remove all easy words."
It is:

- keep the semantic distinction
- break the cheapest shortcut
- verify that a careful human could still recover the intended label

## Workflow

### 0. Start from a real workflow, then simplify

The best synthetic benchmarks usually do **not** start from nowhere.
They start from a real dataset, real workflow, or real failure mode, and then abstract it into a smaller controlled task.

The right question is:

- what is the smallest synthetic environment that still preserves the core decision the model must make?

This is often better than trying to synthesize "realism" directly.

Good process:

1. inspect real examples
2. identify the recurring decision structure
3. strip away details that are noisy but not essential
4. keep the relational or procedural core intact
5. rebuild that core as a controlled synthetic benchmark

Examples:

- messy real workflow:
  - long Slack thread, multiple people, mixed policies, changing approvals
- useful synthetic abstraction:
  - one control map, one approval claim, one requested action, one binary decision

Why this works:

- the synthetic task becomes auditable
- the latent variable is easier to isolate
- later mechanistic claims become cleaner

### 0.1 Ask what must be preserved

When abstracting from real data, explicitly separate:

- **core elements to preserve**
- **messy details to remove**

Core elements to preserve are the pieces without which the target question changes.

Examples:

- who is claiming authority
- what action is requested
- what policy or rule governs that action
- whether the claim actually matches that rule

Messy details to remove are things that make the prompt realistic but do not matter to the core computation.

Examples:

- greetings and filler language
- timestamps
- repeated context
- irrelevant tool chatter
- extra organizational politics

### 0.2 Simplify toward the question, not toward convenience

There are two bad simplifications:

- simplifying until the task is trivial
- simplifying until the task no longer reflects the real question

The right simplification preserves the *decision bottleneck*.

Ask:

- after simplification, is the model still solving the same fundamental problem?

If the answer becomes "the model can now solve this from one keyword," you simplified too far.

### 0.3 Keep a traceable link to the real-world source

When possible, document:

- what real pattern or workflow inspired the benchmark
- what was removed
- what was preserved
- why the simplified version is still faithful to the original question

This makes the benchmark easier to defend later, especially in research writeups.

### 1. Write the latent variable first

Before writing prompts, define:

- what exact variable the benchmark is trying to isolate
- what counts as a positive and negative example
- what should *not* change between labels

Examples:

- good: `claimed authority applies to requested action`
- weaker: `prompt looks security-related`

If the latent variable is relational, the prompts must force a relational computation.

### 2. Define the minimal computation the model must perform

Ask:

- what information must the model combine?
- in what order does it see that information?
- what trivial shortcuts would let it avoid the intended computation?

For authority-applicability, the intended computation was:

1. read the control map
2. read the claim
3. read the requested action
4. compare claim vs governing rule

That is much better than simply presenting a good holder in one class and a nonsense string in the other.

### 2.1 Treat the environment as part of the measurement instrument

In many synthetic benchmarks, the surrounding "world state" is not neutral background.
It is part of how the latent variable becomes behaviorally visible.

This includes things like:

- market state
- portfolio state
- current permissions or affordances
- currently held objects or positions

Rule:

- if the benchmark is about instruction conflict, design the environment so the conflict shows up on the intended behavioral dimension

Examples:

- if testing size conflict, make action and target asset obvious so only size is live
- if testing trade-vs-observe conflict, make action selection live rather than target-asset selection
- if testing concentration-vs-diversification conflict, include current holdings so spreading vs adding is a real choice
- if testing hold-vs-exit conflict, include an existing position so hold/exit is behaviorally meaningful

Bad pattern:

- one generic context pool reused across all conflict families

Better pattern:

- family-specific environment contracts

### 2.3 Repo-specific workflow operations live elsewhere

If the task turns into:

- workflow file design
- workflow execution
- runtime bookkeeping
- run triage

use the repo-operational skills instead:

- [constructing-workflows](../constructing-workflows/SKILL.md)
- [pipelines-v2-run-ops](../pipelines-v2-run-ops/SKILL.md)

### 2.2 Operationalize the same latent variable differently by family

The benchmark-level latent variable can stay constant while the readout changes by family.

Example:

- benchmark latent variable: `instruction_source_followed`

Possible family-specific readouts:

- size family: same action, same asset, different size
- activity family: same market, different action
- diversification family: different chosen asset under the same market and portfolio
- holding family: keep position vs reduce position

Do not force every family into the same behavioral readout if that makes the task unnatural or ambiguous.

### 3. Design the negative class carefully

Avoid lazy negatives.

Bad negative design:

- random wrong strings
- obviously irrelevant domains
- malformed or unnatural text

Good negative design:

- plausible but wrong
- same broad domain
- same format
- differs on the exact target relation

Examples:

- good: `treasury_officer` vs `audit_controller` for a finance action
- good: `repository_maintainer` vs `deployment_operator` for a code action
- weak: `treasury_officer` vs `banana`

The wrong answer should still be *believable* inside the world of the prompt.

### 4. Get prompt-role semantics right

Prompt placement is part of the experiment, not wrapper text.

Be deliberate about:

- what goes in the system message
- what goes in the user message
- what goes in tool outputs
- whether the carrier itself is part of the variable

Rules:

- put task-defining rules in the system message if they are meant to function as agent instructions
- put evidence in the user/tool/document sections where the model would naturally encounter it
- if testing carrier effects, keep semantic content constant while moving only the carrier
- do not accidentally make the label recoverable from role placement alone

When using system, user, and tool carriers:

- preserve comparable semantics across carriers
- preserve comparable output constraints
- keep the model's decision rule stable

### 5. Respect agent logic, not just surface text

Synthetic prompts should make sense to the *agent* being modeled.

Ask:

- what would an operations agent infer here?
- what would a coding agent treat as authoritative?
- what would a tool-calling agent consider evidence vs instruction?

A prompt can be lexically balanced and still unrealistic from an agent-logic perspective.

Examples of agent-logic failures:

- tool output phrased like a system policy
- user content pretending to redefine the agent's role without a believable reason
- approval claims that no real workflow would express that way

When in doubt, write prompts that feel like something the target agent could actually receive in a workflow.

### 6. Constrain outputs hard enough to audit behavior

For early experiments, prefer tightly constrained outputs over freeform answers.

Good:

- `DECISION: COMPLY or ESCALATE`
- explicit structured labels

Why:

- easier to parse
- easier to spot malformed behavior
- easier to compare across conditions
- reduces ambiguity during sanity checks

Do not start with open-ended natural language outputs unless the generation style is itself part of the experiment.

### 7. Add lexical controls early

If you skip lexical controls, your benchmark will almost always overstate abstraction.

At minimum vary:

- holder surface forms
- action surface forms
- template families
- map / instruction wording

Then split train and test so that held-out lexical families appear only at test.

Best practice:

- have multiple lexical bundles per entity
- reserve at least one bundle entirely for test

### 8. Add vocab-control if the claim is abstraction

If the intended claim is that the model learned a relation rather than a familiar phrase, create a vocab-controlled variant.

That means:

- preserve the underlying relational structure
- replace intuitive surfaces with less obvious but still coherent surfaces

### 9. Make the split reflect the generalization claim

Your split scheme should match the claim you want to make.

Common split types:

- lexical holdout
- action holdout
- domain holdout
- carrier holdout

Do not claim abstraction across a dimension you never actually held out.

### 10. Build matched pairs when causal work is likely

If you may later do activation patching or interchange, design the dataset so paired examples exist naturally.

Ideal matched pair:

- same template family
- same carrier family
- same action family
- same control map
- label differs only in the decisive variable

For instruction-conflict datasets, this usually means:

- same strategy wording
- same setting wording family
- same environment template
- same lexical split assignment
- only the target setting value changes

Important:

- aligned control rows are still part of the dataset, but they should not be mislabeled as one side "winning"
- if strategy and setting agree, treat that row as an aligned control state rather than source-disambiguation supervision

Good audit scheme:

- `aligned_agreement`
- `strategy_followed`
- `setting_followed`
- `mixed_or_neither`

Then derive a smaller training target later if needed.

### 10.1 Freeze inventories before writing the generator

If the user wants a benchmark that is reproducible and auditable, do not let the generator improvise.

Lock named inventories for:

- strategy wording
- setting wording
- environment templates
- portfolio templates
- market templates
- lexical split assignments

Then make the generator instantiate only those inventories.

This avoids a common failure mode where the markdown spec looks precise but the actual script reintroduces ambiguity by inventing details on the fly.

Useful pattern:

1. write the generator contract
2. write a hand-audited example bank from that contract
3. define the exact dataset row shape
4. only then implement the full generator

### 11. Keep prompts auditable by eye

A human should be able to inspect one row and answer:

- what is the correct label?
- why?
- what changed relative to its paired opposite-label row?

If a human cannot audit rows quickly, the dataset will be painful to debug and easy to misinterpret.

## Behavioral sanity checklist

Before probing:

- inspect real prompt examples from each class
- run the base model on a small sample
- verify outputs are parseable
- verify the model is solving the benchmark at a reasonable rate
- inspect failures manually
- confirm the failures are about the target variable, not formatting

If behavior is wrong, ask:

- is the prompt ambiguous?
- is role placement unnatural?
- is the requested output underspecified?
- is the model confused by carrier formatting?
- did we accidentally create label leakage?

Do not move on to probing until this is addressed.

## Default stance

Synthetic data generation is not "make realistic-looking text."

It is:

`designing a controlled environment where later behavioral and mechanistic claims will still be trustworthy.`
