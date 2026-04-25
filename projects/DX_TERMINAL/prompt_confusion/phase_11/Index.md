# Prompt Confusion Phase 11 -- Index

Phase 11 is the first explicit multi-conflict benchmark scaffold.

The immediate goal is to place two already-established prompt-local
conflict families in the same prompt:

- `trade_size`
- `risk_preference`

The intended question is no longer only:
- does the model represent one conflict family at a time?

It is now also:
- can the model represent both conflicts simultaneously?
- are the two conflict labels independently decodable from the same
  forward pass?
- does a double-conflict row look additive or interactive?

Phase 11 is intended as the intermediate step before returning to
activation patching on the shared Phase 10 conflict-family structure.

## References

| File | What it covers |
|---|---|
| [design.md](specs/design.md) | Phase 11 multi-conflict scope and intended analyses |
| [build_phase_11_dataset.py](scripts/build_phase_11_dataset.py) | Joint size-plus-risk benchmark scaffold |

## Artifacts

- Local dataset: `outputs/phase_11_dataset/phase_11_dataset.jsonl`

## Current status

This phase is now running with first-pass data.

What is already complete:

- multi-conflict dataset builder exists
- combined prompt structure is defined
- primary labels for size conflict, risk conflict, and joint conflict are
  emitted
- local behavior smoke has been run on:
  - aligned rows
  - single-conflict rows
  - double-conflict rows
- initial `pipelines_v2` capture and probing has completed

Current read:

- the joint prompt does not collapse
- `size` remains behaviorally and representationally clean
- `risk` remains the noisy axis, including some aligned aggressive-risk
  rows
- both conflict labels are still strongly readable from the same forward
  pass
- the shared size-risk geometry survives composition into one prompt

See [notes.md](notes.md) for the settled methodology trail and current
interpretation.
