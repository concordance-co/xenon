# Phase Template

Every phase has a `PHASE.md` following this shape. It is the canonical
orientation, design, running-state, and closure artifact for the phase.

Initialize it at the start of the phase but **leave most sections empty
until the phase has earned them**. Pre-work, only `Premise` and
`Artifacts` carry content (and `Open threads` if real questions exist).
Everything else — `What we ran`, `Primary result`, `Qualitative
inspection`, `Corrections`, `Running hypothesis`, `Claim boundary` —
stays empty until runs, data reading, or evidence updates have produced
something to record.

Filling sections prospectively defeats the point. PHASE.md anchors what
the phase actually showed against what was expected.

The `Claim boundary` section tracks **mechanistic claims** — what the
phase's work supports about model internals, representations,
localization, or causality. Operational facts (the data is accessible,
the benchmark is runnable, a model is selected) belong in `Premise` or
`Artifacts`.

File: `workspaces/<workspace>/[<thread>/]<phase>/PHASE.md` (thread optional)

---

## Premise

What we believed coming in. The specific hypothesis or direction this phase is
exploring. For a new phase, include the intended measurement loci, design
rationale, and success / failure criteria.

## What we ran

Run ids, datasets, loci, sites. Enough to reproduce or re-enter.

Empty until runs exist. Planned workflow and data sources belong in `Premise`.

## Primary result

The headline numbers or observations. Tables, not prose, where possible.

Empty until a run has produced a primary readout.

## Qualitative inspection

High rows, low rows, rows that surprise. What reading the data actually looks like.

Empty until you've inspected real run data.

## Corrections

Beliefs held going in that this phase revised. Preregistered proxies that didn't work. Framings that turned out to be wrong.

Load-bearing section. Do not omit to avoid the appearance of backtracking.

Empty until the phase has actually revised a belief against evidence.

## Running hypothesis

What we now believe after this phase. Specific. Avoid restating the primary result.

Empty until evidence has updated belief. Pre-work belief lives in `Premise`.

## Claim boundary

Mechanistic claims earned by this phase's work, separated by support level.

Safe to claim right now:

- ...

Not supported yet, avoid claiming:

- ...

Preferred phrasing for the current state:

- ...

Empty until the phase has produced mechanistic evidence.

## Artifacts

Paths to workflows, reports, review JSONs, result files, prereg docs, and other
phase-local orientation artifacts.

## Open threads

What's unresolved. What the next phase might pick up. What controls or labels are still missing.

Handoff section. The next phase's premise starts from these threads.
