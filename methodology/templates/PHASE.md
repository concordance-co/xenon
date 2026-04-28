# Phase Template

Every phase has a `PHASE.md` following this shape. It is the canonical
orientation, design, running-state, and closure artifact for the phase.

Initialize it when the phase starts. Early sections may say "not run yet" or
name planned datasets, loci, controls, and success criteria. Keep it current as
work runs. When the phase closes, finalize the same file so the next phase's
premise can inherit from its open threads.

File: `projects/<project>/[<subproject>/]<phase>/PHASE.md` (subproject optional)

---

## Premise

What we believed coming in. The specific hypothesis or direction this phase is
exploring. For a new phase, include the intended measurement loci, design
rationale, and success / failure criteria.

## What we ran

Run ids, datasets, loci, sites. Enough to reproduce or re-enter. Before runs
exist, state the planned workflow, data source, controls, and execution
sequence.

## Primary result

The headline numbers or observations. Tables, not prose, where possible. Before
runs exist, state the planned primary readout and what would count as a useful
result.

## Qualitative inspection

High rows, low rows, rows that surprise. What reading the data actually looks like.

## Corrections

Beliefs held going in that this phase revised. Preregistered proxies that didn't work. Framings that turned out to be wrong.

Load-bearing section. Do not omit to avoid the appearance of backtracking.

## Running hypothesis

What we now believe after this phase. Specific. Avoid restating the primary result.

## Claim boundary

Safe to claim right now:

- ...

Not supported yet, avoid claiming:

- ...

Preferred phrasing for the current state:

- ...

## Artifacts

Paths to workflows, reports, review JSONs, result files, prereg docs, and other
phase-local orientation artifacts.

## Open threads

What's unresolved. What the next phase might pick up. What controls or labels are still missing.

Handoff section. The next phase's premise starts from these threads.
