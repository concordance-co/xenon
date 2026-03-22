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
    parser = argparse.ArgumentParser(description="Build and upload the phase-4 market-representation dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interp_exports/synthetic_market_phase4_representation"),
        help="Local directory for phase-4 market-representation artifacts.",
    )
    parser.add_argument(
        "--representation-steps",
        type=int,
        default=9,
        help="Number of sweep positions per near-tie pairwise scenario.",
    )
    parser.add_argument(
        "--representation-background-variants",
        type=int,
        default=3,
        help="Number of distinct background rosters for the hard pairwise and rank-context families.",
    )
    parser.add_argument(
        "--phase-name",
        default="phase4_market_representation",
        help="Phase tag stored in Neon.",
    )
    parser.add_argument(
        "--log-id-base",
        type=int,
        default=0,
        help="Override the log_id base. Useful for keeping smoke and full runs disjoint.",
    )
    parser.add_argument(
        "--keep-existing-phase",
        action="store_true",
        help="Append instead of replacing rows for this phase name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.log_id_base > 0:
        log_id_base = args.log_id_base
    elif "smoke" in args.phase_name.lower():
        log_id_base = 2_147_200_000
    else:
        log_id_base = 2_147_300_000
    build_result = build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=args.output_dir,
            dataset_preset="phase4_market_representation",
            representation_steps=args.representation_steps,
            representation_background_variants=args.representation_background_variants,
            log_id_base=log_id_base,
            include_settings_variants=False,
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
