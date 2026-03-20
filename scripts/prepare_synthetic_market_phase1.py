from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.interp.synthetic_market import (  # noqa: E402
    SyntheticMarketConfig,
    build_synthetic_market_dataset,
)
from pipelines.interp.synthetic_market_db import (  # noqa: E402
    table_snapshot,
    upload_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and upload the phase-1 synthetic market dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interp_exports/synthetic_market_phase1"),
        help="Local directory for synthetic dataset artifacts.",
    )
    parser.add_argument(
        "--scalar-steps",
        type=int,
        default=17,
        help="Number of scalar sweep steps per family.",
    )
    parser.add_argument(
        "--pairwise-variants",
        type=int,
        default=7,
        help="Number of tradeoff variants per pairwise family.",
    )
    parser.add_argument(
        "--archetype-variants",
        type=int,
        default=8,
        help="Number of perturbations per archetype family.",
    )
    parser.add_argument(
        "--phase-name",
        default="phase1",
        help="Phase tag stored in Neon.",
    )
    parser.add_argument(
        "--market-only",
        action="store_true",
        help="Emit only market-only prompts and skip the settings ladder.",
    )
    parser.add_argument(
        "--keep-existing-phase",
        action="store_true",
        help="Append instead of replacing rows for this phase name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_result = build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=args.output_dir,
            scalar_steps=args.scalar_steps,
            pairwise_variants=args.pairwise_variants,
            archetype_variants=args.archetype_variants,
            include_settings_variants=not args.market_only,
        )
    )
    upload_counts = upload_dataset(
        args.output_dir,
        phase_name=args.phase_name,
        replace_phase=not args.keep_existing_phase,
    )
    snapshot = table_snapshot(args.phase_name)
    print(json.dumps({
        "build_summary": build_result["summary"],
        "upload_counts": upload_counts,
        "db_snapshot": snapshot,
    }, indent=2))


if __name__ == "__main__":
    main()
