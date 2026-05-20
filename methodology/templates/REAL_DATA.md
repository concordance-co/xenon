# Real Data Template

The living context doc for a workspace's source data. Versioned and dated. Updated across the workspace's lifetime as phases surface new things.

File: `workspaces/<workspace>/REAL_DATA.md`

---

## Metadata

- Version
- Last updated
- Freeze date (if applicable)

## Data snapshot

- Source (benchmark URL, production dataset, trace export, etc.)
- Scope: what's in, what's out
- Schema — top-level fields
- Size

## Why this matters

What makes this data tractable or worth studying. What a researcher loses if they don't start here.

## Native labels / signal surfaces

Labels that exist natively. Distinctions the data was designed to carry.

## Refined latent label view

Labels that could support mech interp work. Separate:

- **Prompt-side candidates** — what could be probed from the input
- **Response-side candidates** — what could be probed from the generation
- **Validation-only labels** — useful as downstream checks, not as direct probe targets

For each, note whether it's currently probeable, needs augmentation, or is a nuisance variable.

## Known confounds

Aliases between labels. Correlations between intended signal and nuisance. Places where a naive probe would learn the wrong thing.

## Data-specific gotchas

Things that only apply to this dataset. Loading quirks. Viewer / README mismatches with actual rows. Schema drift between splits.

## Behavioral sanity notes

What's been checked. What hasn't. Current gate judgment.

## Feature hypotheses

What latents we think live in this data. First-pass labels mapped to feature sketches.

## Methods that fit

Pointer to roster entries that match this data's label structure.

## Methods to be careful with

What won't work cleanly on this data without augmentation or repair.

## Data gap list

What's missing for the research threads currently running or planned.

## Open questions

What we don't know yet.

## Active Research Threads

Links to thread READMEs. One-line status each.
