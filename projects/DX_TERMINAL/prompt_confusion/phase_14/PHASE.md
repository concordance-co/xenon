# Phase 14: Mid-Prompt Synthetic Geometry

## Premise

Phase 13 found the cleanest real-transfer read at `L32 settings_end`, not at a
generic prompt-final token. Before we ask whether synthetic directions transfer
to real DX Terminal prompts, we should first ask whether the synthetic benchmark
itself has good readouts at locations that map cleanly onto real prompt
sections.

Question:

> If we train the settled prompt-confusion probes at section-local mid-prompt
> loci, do we recover the same family structure found at prompt EOS?

## What we ran

Completed run:

- Run: `wr_d94ff7683283_3a777553`
- Capture: `capture_1_de8f637033f3`
- Report: `report_b3313df51b28_76ec21d2`
- Report path:
  `projects/DX_TERMINAL/prompt_confusion/phase_14/reports/mid_prompt_geometry/report_b3313df51b28_76ec21d2/report.md`

Workflow commands:

```bash
uv run python -m pipelines_v2.cli workflow plan --file projects/DX_TERMINAL/prompt_confusion/phase_14/specs/workflow.py
uv run python -m pipelines_v2.cli workflow run --file projects/DX_TERMINAL/prompt_confusion/phase_14/specs/workflow.py --logging INFO
```

The workflow reads the settled three-family row surface from Neon:

- Table: `prompt_confusion_three_family_settled_v1`
- Phase 09 `trade_size`: 384 rows, 192 conflict / 192 aligned
- Phase 10 `risk_preference`: 384 rows, 192 conflict / 192 aligned
- Phase 12 `diversification_preference`: 384 rows, 192 conflict / 192 aligned

The workflow derives Phase 14 direction-class labels and strict combined split
labels in SQL, then recaptures the same settled prompts at mid-prompt loci.

Planned loci:

- `strategies_end`
- `settings_end`
- `portfolio_end`
- `market_end`
- `prompt_eos`

Planned analyses:

- strict lexical-holdout text gates per family
- strict lexical-holdout conflict probes per family and site
- positive-minus-negative directions per family and site
- direction-cosine comparison between section-local directions and prompt-EOS
  directions
- family-to-family geometry comparison at each site

## Primary result

Mid-prompt probes work, but the best readout locations are family-specific.
`settings_end` is strong for all three families, while `strategies_end` is a
null site under the strict holdout.

Best strict-holdout balanced accuracy by family/site:

| Site | `trade_size` | `risk_preference` | `diversification_preference` |
| --- | ---: | ---: | ---: |
| `strategies_end` | 0.5000 | 0.5000 | 0.5000 |
| `settings_end` | 0.9479 @ L24 | 0.9375 @ L36 | 1.0000 @ L20 |
| `portfolio_end` | 1.0000 @ L36 | 0.9167 @ L36 | 1.0000 @ L24 |
| `market_end` | 0.8542 @ L40 | 0.8958 @ L32 | 0.9062 @ L36 |
| `prompt_eos` | 0.9375 @ L36 | 0.8750 @ L40 | 0.8958 @ L36 |

Text gates were at chance under the strict combined split for all families
(`0.5000` balanced accuracy), so the readouts are not coming from the raw text
baseline used here.

Geometry did not simply preserve the prompt-EOS direction basis at
`settings_end`. Best same-family `settings_end` vs prompt-EOS cosine was
`risk_preference`, `0.3562` at L36. The strongest settings-site family pair was
`trade_size` vs `risk_preference`, `0.6391` at L40. The shared-mean
`settings_end` vs prompt-EOS cosine peaked at `0.2600` at L36. The
`portfolio_end` shared mean was closer to prompt EOS, peaking at `0.3730` at
L36; `settings_end` vs `portfolio_end` shared means peaked at `0.2958` at L36.

The Phase 14 probes used the strict both-axis holdout, grouped by
`matched_group_id`, plus text gates and shuffled-label selectivity. They did
not rerun the full earlier per-family confound battery at every mid-prompt
site. Treat the near-ceiling `diversification_preference` result as a strong
localization result, not a newly revalidated family-level claim.

The useful success criterion is not only high AUROC. The sharper test is:

- a mid-prompt site has strong held-out probe performance
- its family direction is similar to the corresponding prompt-EOS direction
- shared family geometry is preserved, especially around `settings_end`

## Qualitative inspection

Initial read:

- `settings_end` is a viable real-transfer-compatible probe site.
- `portfolio_end` is also very strong, especially for `trade_size` and
  `diversification_preference`; this may reflect that the synthetic conflicts
  remain explicitly represented beyond the settings block.
- `strategies_end` is a useful negative control: before ACTIVE SETTINGS, the
  strict conflict label is not decodable.
- Direction geometry at `settings_end` is not an obvious copy of prompt-EOS
  geometry. Probe quality and direction similarity should be treated as
  separate measurements.

The first read should focus on `settings_end` and `prompt_eos`. Other sites are
there to explain failures and offsets, not to create a broad search story.

## Corrections

This phase corrects a measurement-locus assumption in the prior workflow:

- prompt-EOS directions were useful for synthetic characterization
- real-transfer evidence points toward section-local loci, especially
  `settings_end`
- therefore the next synthetic direction bank should be trained and compared at
  section-local sites before another real-transfer pass

Bring-up correction:

- `trade_size.strict_combined_split` is null in the Neon upload, so Phase 14
  derives `phase14_strict_combined_split` from the two lexical split axes.
- `DirectionSpec` should not group by `matched_group_id`, because those groups
  intentionally contain both conflict and aligned members.

## Running hypothesis

`settings_end` should recover a strong `trade_size` readout and may preserve
part of the shared family geometry seen at prompt EOS. If it does, that gives a
cleaner bridge into real-data probing than reusing prompt-EOS directions.

Diversification may show a stronger site-specific offset, especially around
`portfolio_end`, which would support the Phase 12 baseline-shift hypothesis.

## Claim boundary

Safe to claim right now:

- Phase 14 ran the settled three-family synthetic prompts from Neon on Modal.
- Strict-holdout probes are strong at `settings_end`, `portfolio_end`, and
  `market_end`, but not at `strategies_end`.
- `settings_end` gives a clean mid-prompt readout surface, but its directions
  are only weak-to-moderately aligned with prompt-EOS directions.
- `portfolio_end` preserves more of the shared prompt-EOS mean direction than
  `settings_end`, despite both sites yielding strong probes.

Not supported yet, avoid claiming:

- that `settings_end` preserves the prior shared geometry
- that any new direction transfers to real data
- that any direction has causal leverage
- that the Phase 14 mid-prompt results have passed every prior per-family
  confound check

Preferred phrasing for the current state:

- "Phase 14 shows that the settled prompt-confusion labels are strongly
  decodable at real-transfer-compatible synthetic loci, especially
  `settings_end`, but section-local directions are not just prompt-EOS
  directions moved earlier."

## Artifacts

- Workflow: `projects/DX_TERMINAL/prompt_confusion/phase_14/specs/workflow.py`
- Report:
  `projects/DX_TERMINAL/prompt_confusion/phase_14/reports/mid_prompt_geometry/report_b3313df51b28_76ec21d2/report.md`
- Geometry result:
  `projects/DX_TERMINAL/prompt_confusion/phase_14/reports/mid_prompt_geometry/report_b3313df51b28_76ec21d2/results/mid_prompt_direction_geometry_results.json`

## Open threads

- Inspect `settings_end` vs `portfolio_end` as candidate direction-bank loci
  before the next real-transfer pass.
- Decide whether to train real-transfer directions directly at `settings_end`
  instead of reusing prompt-EOS directions.
- Audit why `settings_end` direction cosines to prompt EOS are modest despite
  strong held-out probe performance.
