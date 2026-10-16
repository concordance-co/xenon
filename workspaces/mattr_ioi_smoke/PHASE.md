# Official MAttr integration smoke

## Premise

Verify the released MAttr optimizer, exact MIB evaluator, and Xenon artifact plumbing on cached GPT-2/official IOI. This is an engineering phase; no claim about attribution quality is being tested.

## What we ran

The pinned official entrypoint ran offline for two fitting steps, one seed (42), and two held-out validation examples on CPU. Input-node coverage, iso patching and the full official ten-point grid were retained. A subsequent Xenon LocalRunner invocation produced a complete cataloged artifact from the same bounded settings.

## Primary result

The official command exited successfully in 11.16 seconds, with sampled peak aggregate RSS 2.90 GB. Xenon artifact `external_mattr_1_69065e13` contains 157 named learned scores, ten official evaluator points and preserved raw outputs. These are plumbing checks, not research findings.

## Qualitative inspection

The first validation example contrasts “Greg and Jeff … Greg gave a necklace to” with the official source prompt replacing the second Greg with Jeff. Names and clean/source orientation come directly from the pinned IOI dataset.

## Corrections

The Xenon smoke wrapper initially used a nonexistent `load_result()` accessor after the artifact had already finalized. Reading the artifact with the public `result()` accessor verified the saved output without rerunning model work.

## Running hypothesis

No research hypothesis tested.

## Claim boundary

No mechanistic, causal-quality or reproduction claim is supported. The two-step/two-example run verifies integration only. Scientific verdict: `not_tested`.

## Artifacts

- `workflow.py`, `request.json`, `pack.json`: reusable bounded workflow and exact identities.
- `platform/MATTR.md`: setup/run commands and integration limits.
- Execution receipt and official logs: task T10 evidence; raw output bytes are preserved in the result artifact.

## Open threads

Dual-interpreter Modal execution and independent review remain integration gates. T11 owns reference comparisons and controls; they are outside this plumbing phase.
