from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from projects.MOREBENCH.phase_03.scripts.analyze_experiment_02_family_transfer_tail import run_analysis


REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_tail_fraction_sweep")


def _extract_summary(payload: dict[str, Any]) -> dict[str, Any]:
    def summarize_block(block: dict[str, Any]) -> dict[str, Any]:
        if "error" in block:
            return {"error": block["error"]}
        return {
            "best_text_balanced_accuracy": block["best_text_baseline_by_balanced_accuracy"]["balanced_accuracy"],
            "best_text_macro_ovr_auroc": block["best_text_baseline_by_macro_ovr_auroc"]["macro_ovr_auroc"],
            "best_probe_layer": block["best_probe_layer"]["layer"],
            "best_probe_balanced_accuracy": block["best_probe_layer"]["balanced_accuracy"],
            "best_probe_macro_ovr_auroc": block["best_probe_layer"]["macro_ovr_auroc"],
            "probe_minus_best_text_balanced_accuracy": block["best_probe_layer"]["probe_minus_best_text_balanced_accuracy"],
            "probe_minus_best_text_macro_ovr_auroc": block["best_probe_layer"]["probe_minus_best_text_macro_ovr_auroc"],
            "layer_8": next((item for item in block["probe_by_layer"] if item["layer"] == 8), None),
            "layer_44": next((item for item in block["probe_by_layer"] if item["layer"] == 44), None),
        }

    return {
        "mixed_family_group_holdout": summarize_block(payload["mixed_family_group_holdout"]),
        "leave_one_family_out": {
            key: summarize_block(block)
            for key, block in payload["leave_one_family_out"].items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--description-capture-dataset-id", required=True)
    parser.add_argument("--description-capture-id", required=True)
    parser.add_argument("--name-capture-dataset-id", default=None)
    parser.add_argument("--name-capture-id", required=True)
    parser.add_argument("--name-capture-dataset-json", default=None)
    parser.add_argument("--alias-capture-dataset-id", required=True)
    parser.add_argument("--alias-capture-id", required=True)
    parser.add_argument("--fractions", nargs="*", type=float, default=[0.10, 0.25, 0.33, 0.50, 0.75])
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "tail_fraction_sweep_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detailed: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    for fraction in args.fractions:
        payload = run_analysis(
            description_capture_dataset_id=args.description_capture_dataset_id,
            description_capture_id=args.description_capture_id,
            name_capture_dataset_id=args.name_capture_dataset_id,
            name_capture_id=args.name_capture_id,
            alias_capture_dataset_id=args.alias_capture_dataset_id,
            alias_capture_id=args.alias_capture_id,
            tail_fraction=float(fraction),
            name_capture_dataset_json=args.name_capture_dataset_json,
        )
        key = f"{float(fraction):.2f}"
        detailed[key] = payload
        summary[key] = _extract_summary(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "detailed": detailed}, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
