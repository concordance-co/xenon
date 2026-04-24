from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _theory_only_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    per_class = dict(metrics.get("per_class", {}))
    filtered = {
        name: payload
        for name, payload in per_class.items()
        if name != "generic_ethics_control" and int(payload.get("support") or 0) > 0
    }
    recalls = [float(payload["recall"]) for payload in filtered.values() if payload.get("recall") is not None]
    aucs = [float(payload["auroc"]) for payload in filtered.values() if payload.get("auroc") is not None]
    return {
        "classes": sorted(filtered),
        "macro_theory_recall": round(float(np.mean(recalls)), 4) if recalls else None,
        "macro_theory_auroc": round(float(np.mean(aucs)), 4) if aucs else None,
    }


def _summarize_block(block: dict[str, Any]) -> dict[str, Any]:
    if "error" in block:
        return {"error": block["error"]}
    best_text_ba = dict(block["best_text_baseline_by_balanced_accuracy"])
    best_text_auroc = dict(block["best_text_baseline_by_macro_ovr_auroc"])
    best_probe = dict(block["best_probe_layer"])
    layer8 = next((dict(item) for item in block["probe_by_layer"] if item["layer"] == 8), None)
    layer44 = next((dict(item) for item in block["probe_by_layer"] if item["layer"] == 44), None)
    return {
        "best_text_baseline_by_balanced_accuracy": {
            "name": best_text_ba["name"],
            "balanced_accuracy": best_text_ba["balanced_accuracy"],
            "macro_ovr_auroc": best_text_ba["macro_ovr_auroc"],
            "theory_only": _theory_only_summary(best_text_ba),
        },
        "best_text_baseline_by_macro_ovr_auroc": {
            "name": best_text_auroc["name"],
            "balanced_accuracy": best_text_auroc["balanced_accuracy"],
            "macro_ovr_auroc": best_text_auroc["macro_ovr_auroc"],
            "theory_only": _theory_only_summary(best_text_auroc),
        },
        "best_probe_layer": {
            "layer": best_probe["layer"],
            "balanced_accuracy": best_probe["balanced_accuracy"],
            "macro_ovr_auroc": best_probe["macro_ovr_auroc"],
            "probe_minus_best_text_balanced_accuracy": best_probe["probe_minus_best_text_balanced_accuracy"],
            "probe_minus_best_text_macro_ovr_auroc": best_probe["probe_minus_best_text_macro_ovr_auroc"],
            "theory_only": _theory_only_summary(best_probe),
        },
        "layer_8": None if layer8 is None else {
            "balanced_accuracy": layer8["balanced_accuracy"],
            "macro_ovr_auroc": layer8["macro_ovr_auroc"],
            "probe_minus_best_text_balanced_accuracy": layer8["probe_minus_best_text_balanced_accuracy"],
            "probe_minus_best_text_macro_ovr_auroc": layer8["probe_minus_best_text_macro_ovr_auroc"],
            "theory_only": _theory_only_summary(layer8),
        },
        "layer_44": None if layer44 is None else {
            "balanced_accuracy": layer44["balanced_accuracy"],
            "macro_ovr_auroc": layer44["macro_ovr_auroc"],
            "probe_minus_best_text_balanced_accuracy": layer44["probe_minus_best_text_balanced_accuracy"],
            "probe_minus_best_text_macro_ovr_auroc": layer44["probe_minus_best_text_macro_ovr_auroc"],
            "theory_only": _theory_only_summary(layer44),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = {
        "mixed_family_group_holdout": _summarize_block(payload["mixed_family_group_holdout"]),
        "leave_one_family_out": {
            key: _summarize_block(block)
            for key, block in payload["leave_one_family_out"].items()
        },
        "within_family_group_holdout": {
            key: _summarize_block(block)
            for key, block in payload["within_family_group_holdout"].items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
