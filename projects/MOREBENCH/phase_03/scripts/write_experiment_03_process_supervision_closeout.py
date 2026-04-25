#!/usr/bin/env python3
"""Write a closeout report for the process-supervision labelability pass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


BASE_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_03_process_supervision")
SUMMARY_PATH = BASE_DIR / "labelability_summary.json"
CLOSEOUT_PATH = BASE_DIR / "process_supervision_closeout.md"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    agreement = _mapping(summary.get("agreement"))
    gates = _mapping(agreement.get("gates"))
    aggregate = _mapping(summary.get("aggregate_agreement"))
    agreements_by_reviewer = _mapping(summary.get("agreements_by_reviewer"))
    all_gates = _mapping(aggregate.get("all_gates_pass"))
    any_gates = _mapping(aggregate.get("any_gates_pass"))
    lines = [
        "# Experiment 03 Process-Supervision Closeout",
        "",
        "## Decision",
        "",
        "Do not run F/B/C probes as confirmatory claims on this annotation set. The original reviewer was miscalibrated, and the recalibrated multi-reviewer audit is not stable enough to authorize F/B/C under the frozen gate discipline.",
        "",
        "## Gate Results",
        "",
        f"- first/legacy active reviewer F kappa: `{agreement.get('f_criterion_family_kappa')}`; B IoU share: `{agreement.get('b_commitment_span_iou_share_ge_0_5')}`; C kappa: `{agreement.get('c_consideration_kappa')}`.",
        f"- first/legacy active reviewer gates: `{dict(gates)}`",
        f"- multi-reviewer all-gates-pass: `{dict(all_gates)}`",
        f"- multi-reviewer any-gates-pass: `{dict(any_gates)}`",
        f"- validation errors: `{summary.get('validation_error_count')}`",
        "",
        "## Reviewer Stability",
        "",
    ]
    for path, record in agreements_by_reviewer.items():
        lines.append(
            "- `{}`: F kappa `{}`, B IoU share `{}`, C kappa `{}`, gates `{}`".format(
                Path(path).name,
                record.get("f_criterion_family_kappa"),
                record.get("b_commitment_span_iou_share_ge_0_5"),
                record.get("c_consideration_kappa"),
                record.get("gates"),
            )
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The process-supervision idea remains conceptually interesting, but the current Codex-subagent annotation protocol is not reliable enough for confirmatory mechanistic probing.",
        "The failure is at the behavioral-labelability layer, not the activation layer.",
        "",
        "Most likely causes:",
        "",
        "- The first reviewer used a different effective task and selected only top-level families; that result is invalid as a gate.",
        "- After recalibration, reviewer 1 nearly duplicated primary annotations, while reviewers 2/3 remained substantially lower, indicating unstable audit behavior.",
        "- Criterion-family coverage remains subjective when each row contains ~20-47 criteria and annotators must map raw criteria into broad semantic families.",
        "- Commitment span boundaries are genuinely ambiguous in long-form responses with headings, conditional advice, and multi-step decision paths.",
        "- Early-collapse vs sustained-multi-consideration is close to usable; two independent calibrated reviewers landed just below threshold, while one reviewer matched primary exactly.",
        "",
        "## What Survived",
        "",
        "- The frozen criterion-family taxonomy exists and is reusable as a future annotation aid.",
        "- The 500 primary annotations are schema-valid and may be useful for qualitative inspection, but not as probe labels.",
        "- Existing capture `capture_1_f2a9e4531dec` was verified to contain per-token generated residuals, so future span-level probing does not require recapture if reliable labels are produced.",
        "",
        "## Exit",
        "",
        "Per the precommitment, F/B/C are not authorized as confirmatory probe tracks. A future salvage attempt should use a narrower label, explicit calibration examples, and at least two independent reviewers before probing.",
        "",
    ])
    CLOSEOUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(CLOSEOUT_PATH)


if __name__ == "__main__":
    main()
