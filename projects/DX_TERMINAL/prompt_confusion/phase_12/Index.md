# Prompt Confusion Phase 12 -- Index

Phase 12 introduces a third prompt-local conflict family:

- `diversification_preference`

The design goal is to make diversification an actual portfolio-aware
asset-selection problem rather than a vague style preference.

This phase keeps the successful single-family structure from Phase 09 and
Phase 10:

- crossed relational strategy vs settings disagreement
- prompt-local resolution
- one binding target setting
- one expected output asset

The new ingredient is that `PORTFOLIO` is now operational:

- `ALPHA` adds to the same sleeve the book already owns
- `BETA` broadens exposure into a distinct sleeve

## References

| File | What it covers |
|---|---|
| [design.md](specs/design.md) | Phase 12 scope and diversification design |
| [build_phase_12_dataset.py](scripts/build_phase_12_dataset.py) | Diversification dataset generator |
| [notes.md](notes.md) | Running methodology notes for the phase |

## Artifacts

- Local dataset: `outputs/phase_12_dataset/phase_12_dataset.jsonl`

## Intended execution model

1. keep the portfolio-conditioned mapping behaviorally legible
2. check that text gates stay near chance
3. run behavior smoke before building a full `pipelines_v2` workflow

## Current status

Phase 12 is now scaffolded:

- diversification dataset builder exists
- portfolio-conditioned prompt structure is defined
- initial dataset has been generated

The next question is whether the model treats diversification settings as
binding selection constraints or as softer preferences.
