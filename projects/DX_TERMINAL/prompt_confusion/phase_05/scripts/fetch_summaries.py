"""Fetch summary.json files produced by Phase 05 analyses and write them locally."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "xenon-prompt-confusion-phase5-fetch-summaries"
ANALYSES = (
    "family_identity_probe",
    "cross_family_transfer",
    "family_geometry",
    "confound_battery",
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("xenon-data", create_if_missing=True)

base_image = modal.Image.debian_slim(python_version="3.13")


@app.function(volumes={"/data": data_volume}, image=base_image, timeout=120, cpu=1)
def fetch(analysis_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in analysis_names:
        path = Path(f"/data/analysis_results/prompt_confusion/phase_05/{name}/summary.json")
        if not path.exists():
            out[name] = None
            continue
        out[name] = json.loads(path.read_text())
    return out


@app.local_entrypoint()
def main() -> None:
    summaries = fetch.remote(list(ANALYSES))
    output_root = Path("projects/DX_TERMINAL/prompt_confusion/phase_05/outputs")
    for name, data in summaries.items():
        if data is None:
            print(f"[miss] {name}")
            continue
        analysis_dir = output_root / name
        analysis_dir.mkdir(parents=True, exist_ok=True)
        target = analysis_dir / "summary.json"
        target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"[ok]   {name} -> {target}")
