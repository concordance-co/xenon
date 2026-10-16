# Phase 00 — local research plumbing

## Premise

Verify a prepared pretrained GPT-2 runs offline through the canonical workflow, capture, residual edits, paired donors, comparison and local report. Eight bounded prompts, block-5 residual output at the last prompt token, fixed Paris (6342) versus London (3576) log-probability contrast. Identity must match exactly; full-state donor replacement must match an independent HF forward hook. No localization or paper reproduction claim.

## What we ran

`wr_t08_matched_final` and fresh credential-stripped `wr_t08_matched_repeat`, workflow hash `e06659131d5dae3c0f622f97d2fd6c87a7a69adeec2bb6e301f99b94014859b1`. CPU FP32 eager, pinned revision in model_lock.json. Captures/baseline/identity/zero cover eight examples; all three donor arms cover the same four recipients.

## Primary result

Plumbing passed: no-op exact; independent capture, donor target-logprob and full-state replacement errors all 0.0. Final ordinary workflow 31.11s, sampled peak process-tree RSS 1.09GiB; separate explicit-session warm experiment 4.25s, cold load 1.66s. Model and activation-bank C01 complete; every artifact C03 verified. Summary/report C01 remains incomplete and is disclosed. Repeat checked against the same deterministic oracle. Scientific verdict: inconclusive.

## Qualitative inspection

Generated continuations and target contrasts are available in the bounded report; output formatting is not reliably city-first. This is a plumbing demonstration.

## Corrections

The initial exploratory run used different templates in the nominal matched arm. After observing it, recipient/donor roles were corrected so opposite-label pairs share exact templates. The eight texts, site and metrics did not change. The corrected run is not presented as preregistered before all data. Earlier evidence is retained separately. The activation-bank numerical provenance declaration was repaired by integration before the final run.

## Running hypothesis

No scientific hypothesis promoted.

## Claim boundary

Only correct offline execution and measured intervention semantics are established. No selective factual mechanism, generalization, statistical effect or paper finding is established.

## Artifacts

- workflow.py: executable source; pack.json: recipe/source locks and smoke limits.
- README.md: exact isolated runtime installation, explicit model preparation, run, oracle and discovery commands.
- evidence.json: run, report and integration evidence references.
- inspect_run.py: bounded reading view of verified ReportSpec inputs.

## Open threads

Whole-profile interrupted compute resume remains unsupported and its T08 acceptance gate is pending. No recovery campaign was added. Independent Fable review and root integration remain pending. Original checkpoint download time was not measured; cached preparation verification took 0.226s. One successful oracle child encountered a transient supervisor cleanup PermissionError; a fresh host-executed retry passed, with original diagnostics retained.
