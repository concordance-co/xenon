# Skill Work Review Prompt

Use this prompt with Claude or another agent when reviewing whether Xenon's
skills capture the way real agent sessions in this repo should work.

---

You are reviewing Xenon's repo-local skills and methodology docs. Your job is
not to review the scientific results directly. Your job is to decide whether the
skills would have guided agents toward the right process during real work in
this repo.

## Inputs

Review these repo surfaces:

- `AGENTS.md`
- `.agents/skills/*/SKILL.md`
- `.agents/skills/constructing-llm-probes/PROBES.md`
- `.agents/skills/constructing-llm-probes/ANALYSIS.md`
- `.agents/skills/synthetic-data-generation/references/*.md`
- `methodology/PRINCIPLES.md`
- `methodology/CHECKS.md`
- `methodology/FLYWHEEL.md`
- any provided agent-session transcripts, scratch notes, reports, or diffs from
  real Xenon work

Treat `.agents/skills/frontend-design` as vendored unless the user explicitly
asks you to review frontend behavior.

## Review Goal

Answer:

- Do the skills encode the actual working process agents need in Xenon?
- Are the router descriptions specific enough that the right skill would fire?
- Are important gotchas present where an agent would see them before making the
  mistake?
- Is methodology duplicated, contradicted, or missing between
  `methodology/` and `.agents/skills/`?
- Are the skills concise enough to load, with details split into references when
  needed?

## Must-Check Probe Process Rules

Verify that these rules exist somewhere agents will reliably encounter them:

- Probing should use layer sweeps over a broad range of layers, not one
  convenient layer.
- AUROC and balanced accuracy are Xenon's two key binary-probe metrics.
- AUROC and balanced accuracy should generally improve toward higher layers.
- High AUROC or balanced accuracy in early layers is a warning signal for
  lexical leakage, role/template leakage, duplicated rows, trivial labels, or a
  prompt-side feature.
- Lexical confounds are split-bound. Within-dataset lexical leakage is a
  warning, not automatically a blocker. A lexical confound blocks a transfer
  claim only when the shortcut is available across the train-to-heldout split
  used by the probe.

For each rule, say whether it is:

- `present_and_well_placed`
- `present_but_misplaced`
- `present_but_ambiguous`
- `missing`
- `contradicted`

## Review Method

1. Build a map of the skill set: each skill, what it owns, and what it should
   hand off to.
2. For each provided agent session, identify which skills should have fired and
   whether their current content would have prevented the observed mistakes.
3. Check skill-to-methodology alignment. The skills may summarize methodology,
   but should not silently fork it.
4. Check progressive disclosure. Long craft guidance should live in one-hop
   reference files, not in every `SKILL.md`.
5. Check evidence discipline. Research-facing skills should require
   `evidence_rung` and `claim_boundary` where artifacts are produced.
6. Check host metadata only for obvious drift: generated `agents/openai.yaml`
   should summarize the corresponding skill, not add new behavior.

## Output Format

Start with findings, ordered by severity:

```text
P0 / P1 / P2 / P3: <title>
Files:
- path:line
Problem:
Why it matters:
Suggested fix:
```

Then include:

- `Skill routing gaps`
- `Methodology alignment gaps`
- `Missing or weak gotchas`
- `Probe-process rule coverage`
- `Good changes to keep`
- `Questions for the user`

Keep recommendations surgical. Prefer edits to existing skills or methodology
docs over adding new skills. Do not recommend gstack-style framework machinery
unless there is clear evidence of current pain.
