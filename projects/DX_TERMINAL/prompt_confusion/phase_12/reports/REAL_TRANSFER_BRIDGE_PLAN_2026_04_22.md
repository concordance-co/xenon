# Real Transfer Bridge Plan

## Goal

Build a cleaner bridge from synthetic `prompt_confusion` benchmarks to real DX Terminal production prompts.

The first real-data pass showed that directly projecting synthetic conflict directions onto full production prompts at coarse global sites does not separate:

- high-signal complaint cohort
- baseline-control cohort

So the next step is to separate three failure modes:

1. concept failure
2. format failure
3. site failure

## Stage 1a: Template Control

Question:

- If we take settled synthetic `trade_size` benchmark content and render it into a real DX Terminal prompt template, does the synthetic conflict direction still separate aligned vs conflict?

Purpose:

- isolate prompt-template effects from content effects

Success meaning:

- if synthetic conflict still reads correctly in real template, then full-prompt transfer failure was likely site or content-selection related, not pure format mismatch

Failure meaning:

- real template itself disrupts the synthetic direction, so later transfer attempts need stronger structural control

## Stage 1b: Content Adapter

Question:

- If we take real production `trade_size`-relevant content and render it into the settled synthetic benchmark shape, does it read as conflict in our system?

Purpose:

- test whether real conflict content survives when format is controlled

Dataset design:

- one row per real complaint trace
- use high-precision `trade_size` cohorting from:
  - `strategies_text`
  - `slider_ts`
- start with strict slider extremes:
  - `slider_ts == 1` -> small setting
  - `slider_ts == 5` -> large setting
- require explicit size preference in `strategies_text`
- first pass labels:
  - `aligned`
  - `conflict`

Adapter prompt design:

- synthetic system prompt from Phase 09
- synthetic section order:
  - `TASK`
  - `STRATEGY`
  - `ACTIVE SETTINGS`
  - `PORTFOLIO`
  - `MARKET`
- preserve real content where possible:
  - real `strategies_text`
  - real slider values
  - extracted real portfolio block
  - extracted real market block

Success meaning:

- adapted real conflict rows separate from adapted real aligned rows on the settled synthetic `trade_size` direction

Failure meaning:

- either the synthetic benchmark is narrower than hoped, or our real-content labeling / preservation is still not faithful enough

## Stage 2: Span Localization On Full Real Prompts

Only run if Stage 1b shows real separation.

Question:

- where in the original production prompt does the benchmark-relevant conflict computation become readable?

Candidate spans:

- end of strategy block
- end of active settings block
- end of portfolio block
- end of market block
- end of combined benchmark-relevant region

Purpose:

- distinguish format success from site success

## First Family

Start with `trade_size` only.

Rationale:

- settled anchor family
- strongest synthetic benchmark
- most interpretable real-world mapping through `slider_ts`

## Current Execution Order

1. Build Stage 1b strict trade-size adapter dataset.
2. Capture adapted prompts on the same model and prompt-final site used by synthetic benchmark.
3. Score adapted prompts with settled synthetic `trade_size` direction.
4. If separation is real, proceed to Stage 1a and Stage 2 span localization.

## Guardrails

- prefer high-precision rows over larger noisy cohorts
- keep one row per trace for the first pass to avoid overcounting repeated ticks
- do not use thresholded conflict rates from synthetic calibration as a headline metric
- use score separation first, then inspect examples
