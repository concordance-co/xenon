from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipelines.db import connect_neon
from research.research_rerun.core import save_prompt_payload
from research.research_rerun.postmarket import build_postmarket_geometry_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real DX post-market risk and affordance ladder prompts in Neon.")
    parser.add_argument("--experiment-id", default="real_postmarket_geometry_bridge_v1")
    parser.add_argument("--risk-top-rosters", type=int, default=6)
    parser.add_argument("--risk-per-roster", type=int, default=5)
    parser.add_argument("--affordance-top-rosters", type=int, default=6)
    parser.add_argument("--affordance-per-roster", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis_results/real_postmarket_geometry_bridge"),
    )
    args = parser.parse_args()

    payload, summary = build_postmarket_geometry_payload(
        experiment_id=args.experiment_id,
        risk_top_rosters=args.risk_top_rosters,
        risk_per_roster=args.risk_per_roster,
        affordance_top_rosters=args.affordance_top_rosters,
        affordance_per_roster=args.affordance_per_roster,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.experiment_id}_manifest.json").write_text(json.dumps(summary, indent=2, default=str))

    with connect_neon(autocommit=False) as conn:
        save_prompt_payload(conn, payload)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
