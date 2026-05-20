---
name: garden-docs
description: Walk the documentation tree, cross-check claims against current code, and produce a pruning punch list. Use when docs feel stale, after a methodology shift, or on a regular cadence to fight drift.
---

# Garden Docs

Force a focused pass over repo documentation to catch drift, bloat, and stale
claims. The point of this skill is to be a forcing function — produce a punch
list grounded in `find` / `grep` evidence, not a vibes-based rewrite.

## When to use

- User notices docs feel out of date.
- A methodology, platform, or operations shift has just landed and downstream
  docs probably haven't caught up.
- Periodic cadence (e.g. weekly), invoked manually or via `/schedule`.

Do NOT use this skill to:
- restructure the doc taxonomy (that's a separate task with the user)
- rewrite docs end-to-end
- consolidate across files in the same pass

## Scope

In scope:
- `AGENTS.md`, `README.md`
- `methodology/` — `FLYWHEEL.md`, `PRINCIPLES.md`, `CHECKS.md`, `ROSTER.md`,
  `HYPOTHESES.md`, `templates/`
- `operations/` — `LOCALITY.md`, `INDEXING.md`, `REPORTING.md`
- `platform/` — `API.md`, `WORKFLOW.md`, `SPECS.md`, `ARCH.md`,
  `patching_best_practices.md`, `README.md`
- Workspace-level `README.md`, `PHASE.md`, `REAL_DATA.md`,
  `benchmark_context.md`
- `references/*-llms.txt` (check vendoring date and whether scope claims still
  match repo usage)

Out of scope:
- `methodology/archive/`, anything under `*/archive/` — intentionally preserved
- External research workspaces that are not checked into this repo — only touch
  them if the user explicitly asks and provides their location
- `.venv/`, `__pycache__/`, generated artifacts

## Procedure

Do not skip steps. Each finding must cite a file path and line number; if you
can't cite evidence, do not flag.

### 1. Inventory

List every in-scope markdown file. `find . -name "*.md" -not -path "./.venv/*"
-not -path "*/archive/*" -not -path "*/__pycache__/*"` is a reasonable starting
filter, then narrow to the scope above.

### 2. Per-file checks

For each file, run these checks. Cite line numbers with `file:line` so the user
can navigate.

1. **Dead references.** Every relative link, file path, function name, module
   path, and CLI flag mentioned should resolve.
   - For paths: check the file exists.
   - For symbols: `grep -rn "<symbol>" pipelines_v2/ scripts/` — if
     it's a public-looking name that's not in code, flag.
   - For CLI flags: run the relevant `--help` and compare. Don't trust the
     example to still be valid syntax.
2. **Layout-claim drift.** When a doc embeds a directory tree or repo map
   (e.g. `AGENTS.md` "Repo map" section), compare against `find` output.
   Newly added top-level dirs that aren't reflected, or dirs that are listed
   but no longer exist, are findings.
3. **Cross-doc contradictions.** Same claim phrased two ways across docs
   (e.g. "workflow.json is canonical" in one place vs "workflow.py is
   canonical" in another). Flag both lines.
4. **Bloat / redundancy.** Sections covered better elsewhere; trailing
   "summary of what we just said" paragraphs; foil framing patterns ("X, not
   Y") that make the docs longer without adding technical content.
5. **Orphaned files.** Markdown files in scope that are not linked from any
   index doc (`AGENTS.md`, `README.md`, `methodology/FLYWHEEL.md`, etc.).
   Either link them or propose deletion.
6. **Reference freshness.** For each `references/*-llms.txt`, check the
   vendored date in the header against today's date. Re-vendor if older than
   ~6 months OR if the upstream package version in `pyproject.toml` has moved
   meaningfully. Also: scan recent code under `pipelines_v2/` / `scripts/` for
   uses of the package that the reference does not cover, and uses the
   reference flags as "we don't use this" that have crept in.

### 3. Produce the punch list

Output keyed by file. Example shape:

    AGENTS.md
      L88: link to `methodology/HYPOTHESES.md` — exists ✓
      L131: claim about workflow layout — compare against `platform/SPECS.md`
        and current example paths before treating it as canonical
      L142: references `xenon.toml` `[pipelines_v2.modal]` — verified present

    platform/API.md
      L42: example uses `--flag-foo` — not in current cli.py argparse, drop
      L88: orphan link to `platform/legacy.md` — file does not exist, remove

    references/modal-llms.txt
      Header: vendored 2026-04-29 — fresh
      L40: claim "xenon does not use `modal.Sandbox`" — `grep` finds no
        matches, still accurate

Group findings into severity:
- **Wrong** — actively misleading (broken link, contradicted claim, dead CLI
  flag). Fix every one.
- **Stale** — outdated but not actively wrong (old framing that newer docs
  improve on).
- **Bloat** — could be shorter without losing information.

### 4. Walk the user through fixes

Discuss before compiling. Walk the punch list section by section with the user.
Apply edits surgically.

After applied edits, run `git log -1 --stat` to verify the diff if a commit
was made. Verify commits actually contain the claimed diff.

## Anti-patterns

- Rewriting docs end-to-end. Surgical edits only.
- Flagging "this could be clearer" without concrete evidence the reader will
  be misled. Style preferences are not findings.
- Deleting archive material — it is intentionally preserved.
- Adding new top-level docs. Fold new content into existing
  methodology/operations/platform docs unless the user asks for a new document.
- Using "X, not Y" foil framing in any rewrites you propose.
- Touching external research workspaces unprompted.

## Output discipline

Produce the punch list FIRST. Do not start editing until the user has seen it
and indicated which findings to act on. The user explicitly prefers section-
by-section review over batched edits + recompile.
