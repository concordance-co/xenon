# Phase 02: Forced-Choice Polar Ethical Advantage

## Purpose

Phase 01 showed that freeform ethical-vs-self-advantage generations are behaviorally interesting but lexically confounded: the generated explanation often contains the same words used to label the response. Phase 02 pivots to a forced-choice design.

The core idea is to keep the scenario and option text identical across prompt regimes, force the model to emit only `A`, `B`, `C`, or `D`, and read residual states before the answer is generated.

## Primary Design

- Dataset: 40 Phase 01 self-welfare conflict dilemmas.
- Conditions: ethical, self-serving, exploit, neutral.
- Options: ethical action, self-advantage action, procedural-seek-guidance distractor, delay/gather-info distractor.
- Option order: 4 deterministic rotations per dilemma, so ethical/self-advantage choices are not tied to a fixed letter.
- Generation: deterministic, max 4 tokens, expected output is one letter.
- Capture loci: `scenario_end`, `options_end`, `prompt_end`.
- Layers: L16, L24, L32, L40.

## Interpretation Discipline

Prompt-regime decoding is expected and is not by itself a moral-orientation claim. The more important checks are whether pre-answer residuals predict actual forced choice after option-letter balancing, and whether any direction is stable across option-order variants rather than being a letter or prompt-template artifact.
