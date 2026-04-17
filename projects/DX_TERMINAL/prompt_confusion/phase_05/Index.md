# Prompt Confusion Phase 05 -- Index

Status: **family-identity claim retracted; conflict-detection claim retained as the main surviving result from this phase**

Continues from [Phase 04](../phase_04/Index.md). Phase 05 is kept here as part
of the active prompt-confusion lineage because its workflow spec and notes still
anchor later `pipelines_v2` work.

## What Still Matters

- [design.md](specs/design.md)
  Full Phase 05 framing and decision logic.
- [workflow.py](specs/workflow.py)
  `pipelines_v2` workflow builder retained as a real library-backed fixture.
- [workflow.json](specs/workflow.json)
  Checked-in snapshot of the workflow.
- [notes.md](notes.md)
  The main record of what held up and what collapsed.

## Historical Outcome

Phase 05 started from the hypothesis that the model resolves conflicts by
identifying the conflict type and applying a family-specific resolution policy.
The main conclusion was narrower:

- family identity on this dataset was too confounded with surface wording to
  support a mechanistic claim
- conflict detection still showed real depth-dependent signal after proper
  lexical holdout and residualization

That narrower conflict-detection result fed directly into the later redesign
work retained in Phase 06 and Phase 09.

## Repo Status

Most Phase 05 one-off analysis scripts and local outputs were archived out of
`xenon` during the v1/v2 split. What remains here is the workflow/spec surface
and the notes needed to understand how Phase 05 connects to the later active
phases.
