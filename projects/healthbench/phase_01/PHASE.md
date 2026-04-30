# HealthBench Phase 01

## Premise

This phase tests a refined candidate latent: the scope-vs-commit gate.

Lens: a direction in activations may determine whether the model scopes its
answer with branches, conditionals, and "it depends" structure, or commits to
the modal case and answers directly when context is missing.

Hypothesis: the scope-vs-commit decision is readable before or shortly into
generation on synthetic medical matched triples where only context-completeness
varies. The first smoke is behavioral only: generate responses on a small set,
inspect self-consistency, and select prompts where Qwen bifurcates between
scope and commit modes before any activation capture or direction construction.

Design constraints for the first smoke:

- Fully synthetic, HealthBench-inspired, no copied benchmark strings.
- Matched triples: under-context, partial-context, full-context.
- Same syntactic frame across each triple.
- Length parity is load-bearing: missing fields are explicit as "not provided"
  rather than omitted, so the contrast is not just under prompt short vs full
  prompt long.
- N=16 samples per row family for self-consistency variance.
- No probe training and no binary labels in the discovery loop.
- Initial readout is qualitative plus cheap response-shape proxies: response
  length, conditional markers, question marks, and direct recommendation
  markers.

Success for the smoke means finding at least a few on-the-fence prompt families
where Qwen sometimes scopes and sometimes commits. Failure means either the
prompts force one behavior too strongly or the synthetic contrast is still
dominated by surface form.

## What we ran

- Created `healthbench_scope_vs_commit_smoke_v1` in Neon.
- Shape: 6 synthetic matched triples x 3 context-completeness conditions x
  16 stochastic samples = 288 rows.
- Conditions: `under_context`, `partial_context`, `full_context`.
- Visible prompts remain duplicated across samples, but `prompt_sha256`
  includes `sample_index` so the generation runtime does not deduplicate N=16.
- Workflow run: `wr_8c04e7ce5f7f_4b17b040`.
- Steps:
  - `generate_responses`: Qwen3-30B-A3B, temperature 0.8, top_p 0.95,
    max_tokens 384.
  - `summarize_response_shape`: cheap response-shape proxies only; no judge
    labels and no probe training.
  - `report`: local summary artifact.
- Activation smoke run: `wr_75e662fede81_170e3e3d`.
- Activation row universe: every fourth smoke row by `sample_index % 4 = 0`,
  yielding 72 rows total: 24 `under_context`, 24 `partial_context`, 24
  `full_context`.
- Captured layers: `0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44`.
- Capture sites:
  - `residual_user_prompt`: rendered user-message span, pooled at last token
    for direction analysis.
  - `residual_first_generated_context`: generated-token span from
    `max_tokens=2`, pooled at first generated-token context for direction
    analysis.
- Direction construction:
  - positive: `under_context`
  - negative: `full_context`
  - no behavior labels or judge labels used.
- Steering smoke run: `wr_6b1a3f6bb4ed_5bc586f5`.
- Steering target rows: 24 `partial_context` examples from the activation
  subset.
- Steering variants:
  - baseline partial-context generation
  - user-prompt-end direction at strength `+2`
  - user-prompt-end direction at strength `-2`
  - first-generated-context direction at strength `+2`, written at the final
    rendered prompt token because the current patcher operates during prefill
  - first-generated-context direction at strength `-2`, same write site
- Patch write layers: `28, 32, 36, 40`.

## Primary result

- The smoke generated all 288 responses.
- Aggregate `scope_minus_commit_proxy` shifted in the expected direction:
  - `under_context`: mean `1.3854`
  - `partial_context`: mean `1.2708`
  - `full_context`: mean `-0.9688`
- This is only a proxy read. It supports moving to qualitative review, not a
  claim about the latent direction.
- Highest proxy-variance rows for review:
  - `chest_pain_urgent_care::under_context`
  - `cough_syrup_child::partial_context`
  - `chest_pain_urgent_care::partial_context`
  - `acetaminophen_fever::full_context`
  - `chest_pain_urgent_care::full_context`
- Activation smoke produced two direction artifacts:
  - user-prompt end: `direction_1_fa3d146c`
  - first generated-token context: `direction_1_019d5638`
- Both directions covered all 12 requested layers with 24 positive and 24
  negative examples.
- Steering smoke completed all 4 patched variants with `patched_count=24` and
  `skipped_count=0`.
- Proxy means from the steering smoke:
  - baseline: mean `scope_minus_commit_proxy = 1.375`
  - user-prompt-end `+2`: `1.2083`
  - user-prompt-end `-2`: `1.7083`
  - first-generated-context direction `+2`: `0.75`
  - first-generated-context direction `-2`: `0.2917`
- These proxy shifts are not the predicted monotone story. Treat them as a
  reason to inspect outputs and patch stats, not as validation.

## Qualitative inspection

- Local review page: `http://127.0.0.1:8766/`
- The page shows all generated responses with prompt, triple, condition,
  sample index, word count, scope proxy, commit proxy, and
  scope-minus-commit proxy.
- Inspect before scaling. The proxy is intentionally rough and can mistake
  generic safety caveats for true scope behavior.

## Corrections

- Phase 00 showed that identical `prompt_sha256` values collapse repeated
  samples. Phase 01 fixes this by hashing the sample index into the request
  id while keeping the visible prompt unchanged.
- Prompt length parity was checked before upload. Uploaded triple-level
  min/max prompt-char ratios range from 1.18 to 1.29.

## Running hypothesis

- The prompt family appears suitable enough for a behavioral inspection loop:
  under and partial contexts produce more conditional/scoping structure than
  full contexts in aggregate.
- Candidate prompt rows should be selected by qualitative bifurcation, not by
  accepting the proxy score as a label.

## Claim boundary

- Activation directions have now been computed from context-condition contrast
  only. This is not yet behavioral validation of a steerable gate.
- Initial steering has been run, but it is not yet positive evidence for a
  clean steerable gate.
- No judge labels have been produced.
- The behavior smoke only validates that the small synthetic substrate can
  produce self-consistency variance worth inspecting.

## Artifacts

- `projects/healthbench/phase_01/specs/workflow.py`
- `projects/healthbench/phase_01/scripts/upload_scope_vs_commit_smoke_to_neon.py`
- `projects/healthbench/shared/scope_vs_commit.py`
- Neon table: `healthbench_scope_vs_commit_smoke_v1`
- Workflow run: `wr_8c04e7ce5f7f_4b17b040`
- Generation artifact: `generation_run_1_e89d3cb49275`
- Proxy summary artifact: `transform_1_afa88d9a`
- Local report:
  `projects/healthbench/phase_01/reports/scope_vs_commit_smoke/report_aa7f903613aa_1a8144b2/report.md`
- Local review page:
  `/private/tmp/healthbench_actual_context_review/index.html`
- Activation smoke workflow run: `wr_75e662fede81_170e3e3d`
- Capture artifact: `capture_1_86b4eb37e534`
- User-prompt-end direction: `direction_1_fa3d146c`
- First-generated-token direction: `direction_1_019d5638`
- Activation smoke report:
  `projects/healthbench/phase_01/reports/scope_vs_commit_activation_smoke/report_fbfd3a3966a1_d60b007a/report.md`
- Steering workflow run: `wr_6b1a3f6bb4ed_5bc586f5`
- Steering baseline artifact: `generation_run_1_fdadd7b95afb`
- Steering patched artifacts:
  - `patched_generation_1_1bfd4d07`
  - `patched_generation_1_4b416248`
  - `patched_generation_1_a6952c53`
  - `patched_generation_1_856edb2f`
- Steering summary artifact: `transform_1_6a5ebd50`
- Steering report:
  `projects/healthbench/phase_01/reports/scope_vs_commit_activation_smoke/report_bf4bf0476229_7e421859/report.md`
- Steering review page:
  `/private/tmp/healthbench_actual_context_review/index.html`

## Open threads

- Decide whether the first activation claim targets pre-generation prompt-end
  residuals or shortly-into-generation residuals.
- Keep literal-question asking and conditional-structure scoping
  distinguishable in logs; they may be separate features.
- If the smoke finds bifurcating rows, build a direction from under minus full
  activation differences and validate primarily through steering.
