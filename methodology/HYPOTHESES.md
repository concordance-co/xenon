# Hypotheses

A hypothesis catalog is a subproject-level cross-phase reference that
collects candidate research directions worth running through the flywheel
but not yet committed to a phase. One per subproject, lives at the
subproject root, append-only with explicit closure semantics.

This doc explains the why and how. The instance for any given subproject
lives at `projects/<project>/<subproject>/HYPOTHESES.md`.

## Purpose

Open threads accumulate faster than phases can absorb them. They come from
three sources:

- **Phase closures.** Every PHASE.md is finalized with open threads at
  the end of its phase. Some get picked up by the next phase; many do not.
- **External reading.** Papers, talks, conversations generate candidates
  that don't map to a current phase but should not be lost.
- **Step-back analyses.** A periodic review across phases surfaces
  candidates the per-phase view misses.

Without a catalog, these candidates live scattered across PHASE.md files,
chat logs, and memory. They get rediscovered, re-debated, and sometimes
re-run. The catalog is the durable home that prevents this.

## What it is not

A hypothesis catalog is not a backlog, not a roadmap, not a wishlist.

- A backlog is task-shaped — work to be done.
- A roadmap is sequence-shaped — order of work.
- A hypothesis catalog is **claim-shaped** — falsifiable propositions about
  the system under study, each with the evidence it would take to confirm
  or refute, and the flywheel path that produces that evidence.

If an entry can't be turned into a sentence of the form "We believe X;
to confirm we would need Y; here is the cheapest path to Y" — it's not a
hypothesis. It's a vibe or a task. Move it elsewhere.

## What goes in

A candidate belongs in the catalog when it is:

- **Specific enough to falsify.** "Maybe theories live in a low-dim
  subspace" is not a hypothesis. "Cross-theory cosines drop below 0.50
  after within-dilemma centering and PCA on response-mean L32 residuals,
  with three dominant eigenvalues explaining ≥ 60% of variance" is.
- **Anchored to evidence already seen or to a paper.** A free-floating
  speculation without an anchor doesn't earn a slot. Either prior
  phases or external work should make this candidate worth taking
  seriously.
- **Mappable to flywheel stages.** You should be able to write down the
  Stage 1 / Stage 2 / Stage 4 / etc. moves for the hypothesis, even if
  rough. If you can't, it's not yet ready for the catalog — it needs
  more thinking first.

## What does not go in

- Hypotheses with no falsifiable pass/fail criterion. These rot.
- Hypotheses whose evidence would require a different model, dataset, or
  subproject. Those belong to that other subproject's catalog, not this
  one.
- Already-confirmed claims. Those go into the claim ladder of the
  relevant phase report.
- Speculation about general questions in the field. The catalog is
  scoped to *this subproject's* questions.

## Structure of an entry

Each hypothesis entry has the same fields. Consistency matters because
the catalog is read by future-you under context loss.

Required:

- **ID.** Stable identifier (`H1`, `H2`, ...). Never reused after a
  hypothesis closes.
- **Status.** One of: `proposed`, `active`, `closed`, `superseded`.
- **Cost.** Rough estimate of compute, time, or capital required.
- **Information value.** What this hypothesis decides if confirmed or
  refuted. High info-value hypotheses change the next experiment;
  low ones don't.
- **Last touched.** Date.
- **Paper anchors.** Specific external work this hypothesis derives
  from or is informed by. URLs.
- **Claim.** The falsifiable proposition, in one or two sentences.
- **Flywheel path.** Stages it would visit and the moves at each.
- **Pass / fail criteria.** Concrete numbers or behaviors that would
  confirm or refute.

Optional:

- **Why this is high-value.** Justification beyond what the claim
  itself says.
- **Notes.** Anything else load-bearing — known confounds, blocking
  dependencies, infrastructure concerns.

## Status lifecycle

A hypothesis moves through statuses in one direction:

- `proposed` — entered the catalog, not yet started.
- `active` — currently being run by a phase.
- `closed` — confirmed, refuted, or judged not-worth-pursuing. Phase that
  closed it must be cited.
- `superseded` — replaced by a sharper version. Cite the successor.

Never delete a closed hypothesis. Move it to the closed-hypotheses table.
The audit trail matters — repeated rediscovery of the same dead end is a
real failure mode in long-running subprojects.

## Relationship to PHASE.md

`PHASE.md` (per `templates/PHASE.md`) is the canonical orientation,
design, running-state, and closure artifact for each phase. Open threads
accumulate in its `Open threads` section as the phase runs and are
finalized at closure. The catalog is where those threads live between
phases. The flow:

1. Phase runs → open threads accumulated in its PHASE.md throughout, finalized at closure.
2. At phase boundary, open threads are reviewed against the catalog:
   - new ones get added as new hypothesis entries
   - threads that match existing entries update those entries' status or
     refine their claim
   - threads that contradict existing entries trigger a status change
     (often `superseded`)
3. The next phase's premise (in its own PHASE.md) is built from the
   catalog plus the prior phase's PHASE.md.

The catalog is therefore the persistent store across phases; the
`Open threads` section of each PHASE.md is the source feed.

## Relationship to other methodology docs

- `FLYWHEEL.md` — the process. Tells you what stage a phase is in and
  what loopback to take.
- `PRINCIPLES.md` — the claim hygiene. Tells you what evidence earns
  what claim level.
- `CHECKS.md` — the decision-point triggers. Tells you what to ask
  before committing to a measurement, claim, or design.
- `HYPOTHESES.md` (subproject) — the candidates. Tells you what's worth
  running next.

The flywheel says how to move; the principles say what to claim; the
checks say when to pause; the catalog says where to go.

## Closed hypotheses

A closed hypothesis carries a verdict and the phase that produced it.
Format:

```
| ID | Hypothesis | Phase that closed it | Verdict |
```

Verdicts are short — "rejected at chosen locus," "confirmed under paired
analysis," "superseded by H7," "abandoned, see decision log." If the
verdict needs more than ten words, the catalog is the wrong place; cite
the relevant PHASE.md and link.

Closed hypotheses are a feature, not a bug. They prevent repeated
rediscovery and they are the longest-lived honest output of the
subproject.

## Decision log

The catalog has an append-only decision log at the bottom. Each entry is
date + event + reference. Examples of what to log:

- Catalog created.
- New hypothesis added (which one, why now).
- Hypothesis status change (proposed → active, active → closed).
- Hypothesis superseded by another (cite both).
- Major reframing of the catalog itself.

The log is the catalog's history. When a future maintainer or new agent
opens the catalog cold, the log explains how it got to its current
state.

## When to create one

Create the catalog the first time both of the following are true:

- The subproject has at least two closed phases with non-trivial open
  threads.
- A step-back analysis or external read has generated at least three
  candidate hypotheses worth tracking.

Before that, open threads can live in the closing-state of each phase's
PHASE.md and in the next phase's PHASE.md premise. After that, they
need a durable home.

## Common failure modes

- **Vibes-as-hypotheses.** Entries that are not falsifiable. Symptom:
  no pass/fail criterion, or pass/fail criterion that is a synonym for
  "the thing happened." Fix: rewrite or delete.
- **Hypothesis sprawl.** Twenty entries, none of which is being worked.
  Symptom: the catalog has not been touched in months. The catalog has
  become a graveyard. Fix: prune to the top three by information value
  and move the rest to a numbered archive section.
- **ID reuse.** Closed H1 gets recycled as a new H1. Breaks the
  decision log. Don't.
- **Catalog as roadmap.** Entries written as tasks, not claims. Fix:
  rewrite with explicit pass/fail criteria. If you can't, the entry
  doesn't belong here.
- **Stale entries.** `last_touched` six months old, status still
  `proposed`. Either run it, supersede it, or close it as
  not-worth-pursuing. Don't leave it.
- **Closed hypotheses deleted.** Loses the audit trail. Move them to
  the closed table; never delete.

## Templates

A starter template lives at `templates/HYPOTHESES.md` (TODO if not
present). For an example of a populated catalog, see
`projects/MOREBENCH/theory_persona_vectors/HYPOTHESES.md`.
