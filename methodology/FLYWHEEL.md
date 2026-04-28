# The Flywheel

How a subproject moves from question to finding. This is the methodology spine. Every project, every subproject, every phase lives inside this model.

## Structure

Work is organized in three levels:

- **Project** — one dataset and one system under study. The real-data context lives here.
- **Subproject** — one research question about that system. Owns its synthetic seed, probes, findings.
- **Phase** — a chapter of work inside a subproject. A new phase begins when a non-negligible change commits the subproject to a new direction.

The flywheel describes the stages a subproject can move through. Stages vary in size, subprojects visit different subsets of them, and the loopbacks between stages are where the research happens.

### Layout

```
projects/<name>/
  phase_00/
    PHASE.md           initialized from templates/PHASE.md
    specs/
      workflow.py      blank — fills in as the phase commits to a direction
    docs/              phase-internal artifacts
    reports/           local report outputs
```

Optional, added when the work needs them:

- `projects/<name>/REAL_DATA.md` — living real-data context doc (stage-1 artifact). See `templates/REAL_DATA.md`.
- `projects/<name>/shared/` — project-local helpers (`paths.py`, `neon.py`, etc.)
- `projects/<name>/scripts/` — project-level builders
- `projects/<name>/<subproject>/phase_XX/` — same shape, when a project has multiple distinct research questions
- `projects/<name>/benchmark_context.md` — frozen sidecar for benchmark projects

## Starting a new project

When the user opens with a concrete task — "do benchmark X", "analyze model Y", "investigate behavior Z" — go to work:

1. Read `PRINCIPLES.md`, `CHECKS.md`, and `operations/LOCALITY.md`. Skim `ROSTER.md`.
2. If the task names a repo, paper, or dataset, scan it.
3. Scaffold the project per the layout above. The first phase is `phase_00` — the framing pass: orientation, source scan, project-shape decisions. Hypothesis sketches and substrate analysis live here. Capture and synth design come later.
4. Report back with what the substrate looks like, what scaffold went up, and what framings the data invites.

The user takes it into hypothesis mode from there. Phase 0 closes when the project has a defensible first research question and the next phase has its premise.

## The stages

### 1. Real-data grounding

Goal: understand the system and its data deeply enough to know what's worth asking.

Moves: read prompts, traces, logs directly. Look for failure modes that are hard to catch from the outside. Note strange behaviors. Sketch abstractions that might correspond to something hidden in activation space.

Exit when: you can name specific candidate latents worth studying and point to concrete real-world examples that motivate each one.

Failure mode: jumping to synth before strong priors have been built. The resulting synth won't track the real system.

Artifact: the real-data context doc for this project, versioned and dated. Living document — revisit across the subproject's lifetime. See `templates/REAL_DATA.md`.

### 2. Synthetic representation

Goal: construct a tightly-controlled synthetic dataset that isolates the abstraction from stage 1.

Moves: design the prompt template. Define the labels. Bake in the contrast structure you want to probe. Keep nuisance axes controllable. Start small and iterate.

Exit when: you can point to rows of the synth dataset and argue the target abstraction varies across them while nuisances are controlled.

Failure mode: synth that encodes the lexical shadow of the abstraction, not the abstraction itself. Labels that look clean but are recoverable from surface form alone.

Artifact: the synthetic seed and dataset spec. Versioned — a redesigned synth is a new version, not an in-place edit.

### 3. Synth dataset validation

Goal: confirm the synth dataset is ready to learn things about before spending compute on captures.

Moves: lexical confound checks, n-gram classifiers, cheap baselines, direct inspection of rows across labels. Read the data. Make sure the label isn't trivially predictable from surface features you didn't intend.

Exit when: cheap baselines fail to recover the label above chance on the channels you care about, and you've eyeballed enough rows to trust the contrast is real.

Failure mode: overdoing this. Perfect lexical separation is not achievable and chasing it indefinitely is a trap. Goal is "no unintended shortcut," not "no signal at all."

Artifact: validation report — cheap-baseline numbers, confound check results, sampled rows with commentary.

### 4. Smoke tests

Goal: small-slice capture plus probe to check where signal might live and whether the shape of the results is consistent with the hypothesis.

Moves: capture on a small slice. Sweep layers and token positions. Inspect AUROC curves. Make pooling decisions. Run behavioral sanity on the generations, not just the probes.

Exit when: the curves look roughly like the hypothesis predicts and pooling decisions are anchored in something defensible. Especially: measurement loci chosen here must transfer to real data later. Prefer positions identifiable in production traces.

Failure mode: reading probes without reading generations. A probe on a prompt-leaky setup can look great while the model is doing nothing hypothesis-relevant.

Artifact: smoke report — curves, pooling rationale, behavioral sanity notes.

### 5. Scale

Goal: run on the full synth set now that smoke has validated the approach.

Moves: scale captures. Rerun probes at the chosen loci. Confirm the smoke pattern holds at scale and that previously-unseen examples behave the same as smoke examples.

Exit when: signal holds at scale without qualitative change from smoke.

Failure mode: assuming scale confirms smoke when it doesn't. If scaled results diverge, treat that as a loopback, not a weaker version of the same finding.

Artifact: scaled capture and probe results with explicit comparison to smoke.

### 6. Discovery / signal mining

Goal: find structure in the captured activations that the supervised probes didn't directly train on.

Moves: unsupervised decomposition, subspace analysis, clustering, direction discovery, SAE features if in scope. Ask what else is there.

Exit when: you have hypotheses about additional structure worth formal probe work, or enough evidence that the candidate representation is cleanly one thing.

Failure mode: confusing exploratory structure for confirmed structure. Discovery generates hypotheses; it does not confirm them.

Artifact: discovery notes — directions, subspaces, hypothesized structure with caveats.

### 7. Supervised probe work

Goal: with labels and signal confirmed, train probes and characterize the representation precisely.

Moves: probe architectures, cross-validation, transfer tests, geometry analysis. Characterize what the probe reads and what it doesn't.

Exit when: probes at the chosen loci are well-characterized — their strengths, failure modes, and transfer behavior are understood.

Failure mode: promoting probe readouts to causal or mechanistic claims. See `PRINCIPLES.md` on the evidence ladder.

Artifact: probes, directions, characterization report.

### 8. Real-data incremental return

Goal: move from synth-validated signal to real-data transfer. Ask whether the thing you built survives on real data from the original system.

Moves: apply probes to real data at the matched loci. Inspect high-projection rows. Inspect low-projection rows. Understand what the probe reads on real data — it will differ from what it reads on synth.

Exit when: you can state clearly what the probe reads on real data and what it doesn't, with the claim boundary matching the evidence.

Failure mode: expecting a 1:1 mapping from synth performance to real-data performance. Real data is messier; the probe may fire on a subset of the intended target and miss related cases.

Artifact: phase report with explicit claim boundary. See `templates/PHASE.md`.

## The interventions track

Stages 1–8 are the diagnostic spine. Once a candidate representation exists (usually by stage 6 or 7), causal work branches off as a parallel track: patching, steering, ablation, erasure. This is a different mode — causal rather than diagnostic — and its claims live higher on the evidence ladder.

Interventions are rarely a clean "next step" after stage 8. They run alongside supervised probe work and real-data return, and their conclusions should be integrated into the phase report alongside diagnostic findings.

## Loopbacks

Loopbacks are the research. They are triggered by specific conditions:

- **8 → 1.** Real-data return reshapes what we believed was hidden. The phase closes; a new phase opens, possibly in a new subproject.
- **7 → 2.** Labels don't separate as expected. The abstraction was wrong. Redesign synth.
- **6 → 3.** Discovery surfaces structure that implies new confounds in the dataset. Revalidate before trusting it.
- **5 → 4.** Scaled results diverge from smoke. Something about scale broke assumptions. Reduce and isolate.
- **4 → 2.** Smoke curves don't look like the hypothesis predicts. Probably the synth doesn't encode what was intended.
- **4 → 3.** Smoke is suspiciously clean. Confound check failed to catch something. Rerun validation with the suspected shortcut explicit.

Each loopback is a phase boundary. When you hit one, write the phase exit artifact per `templates/PHASE.md`. Start a new phase.

## Flywheel vs evidence ladder

The flywheel is the process of moving from question to finding.

The evidence ladder (see `PRINCIPLES.md`) is the claim hygiene that runs alongside.

They are orthogonal. The flywheel says what moves are available next. The ladder says what claims the current evidence earns.

Typical correspondence:

- Stage 4 smoke and stage 5 scale usually earn Level 2 representational claims.
- Stage 6 and stage 7 with localization earn Level 3 localized representational claims.
- The interventions track is required for Level 4 causal claims.
- Level 5 mechanistic claims require both intervention evidence and a plausible computation path.

Never advance a claim ahead of its evidence just because the flywheel has moved forward.

## Crosscutting concerns

`PRINCIPLES.md` holds the always-true research values. They apply at every stage.

`CHECKS.md` holds the decision-point triggers. They fire when you are about to commit to a measurement locus, promote a claim, design a synth, cross into real data, or close a phase.

Both are default context for any agent working inside the flywheel.

## Phase transitions

A new phase begins when:

- A loopback is triggered.
- The subproject commits to a meaningfully new direction.
- A major artifact (scaled capture, first real-data run, first intervention) changes the center of gravity.

At every phase boundary, the closing phase writes its exit artifact per `templates/PHASE.md`. That artifact is what the next phase inherits.
