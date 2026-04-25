# Experiment 03 Process-Supervision Closeout

## Decision

Do not run F/B/C probes as confirmatory claims on this annotation set. The original reviewer was miscalibrated, and the recalibrated multi-reviewer audit is not stable enough to authorize F/B/C under the frozen gate discipline.

## Gate Results

- first/legacy active reviewer F kappa: `0.996978559860772`; B IoU share: `0.6333333333333333`; C kappa: `1.0`.
- first/legacy active reviewer gates: `{'B': False, 'C': True, 'F': True}`
- multi-reviewer all-gates-pass: `{'B': False, 'C': False, 'F': False}`
- multi-reviewer any-gates-pass: `{'B': False, 'C': True, 'F': True}`
- validation errors: `0`

## Reviewer Stability

- `review_sample_30_annotations_recalibrated.jsonl`: F kappa `0.996978559860772`, B IoU share `0.6333333333333333`, C kappa `1.0`, gates `{'B': False, 'C': True, 'F': True}`
- `review_sample_30_annotations_recalibrated_02.jsonl`: F kappa `0.29678709838729833`, B IoU share `0.4666666666666667`, C kappa `0.5833333333333333`, gates `{'B': False, 'C': False, 'F': False}`
- `review_sample_30_annotations_recalibrated_03.jsonl`: F kappa `0.43634679601238213`, B IoU share `0.5666666666666667`, C kappa `0.5833333333333333`, gates `{'B': False, 'C': False, 'F': False}`

## Interpretation

The process-supervision idea remains conceptually interesting, but the current Codex-subagent annotation protocol is not reliable enough for confirmatory mechanistic probing.
The failure is at the behavioral-labelability layer, not the activation layer.

Most likely causes:

- The first reviewer used a different effective task and selected only top-level families; that result is invalid as a gate.
- After recalibration, reviewer 1 nearly duplicated primary annotations, while reviewers 2/3 remained substantially lower, indicating unstable audit behavior.
- Criterion-family coverage remains subjective when each row contains ~20-47 criteria and annotators must map raw criteria into broad semantic families.
- Commitment span boundaries are genuinely ambiguous in long-form responses with headings, conditional advice, and multi-step decision paths.
- Early-collapse vs sustained-multi-consideration is close to usable; two independent calibrated reviewers landed just below threshold, while one reviewer matched primary exactly.

## What Survived

- The frozen criterion-family taxonomy exists and is reusable as a future annotation aid.
- The 500 primary annotations are schema-valid and may be useful for qualitative inspection, but not as probe labels.
- Existing capture `capture_1_f2a9e4531dec` was verified to contain per-token generated residuals, so future span-level probing does not require recapture if reliable labels are produced.

## Exit

Per the precommitment, F/B/C are not authorized as confirmatory probe tracks. A future salvage attempt should use a narrower label, explicit calibration examples, and at least two independent reviewers before probing.
