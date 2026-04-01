from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.interp.synthetic.market import (  # noqa: E402
    SyntheticMarketConfig,
    build_synthetic_market_dataset,
)
from pipelines.interp.synthetic.db import (  # noqa: E402
    table_snapshot,
    upload_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and upload the phase-13 set-geometry portfolio-ladder dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interp_exports/synthetic_market_phase13_set_geometry_portfolio_ladder"),
        help="Local directory for phase-13 set-geometry portfolio-ladder artifacts.",
    )
    parser.add_argument(
        "--permutation-variants",
        type=int,
        default=4,
        help="Number of row permutations to emit per geometry scenario.",
    )
    parser.add_argument(
        "--profile-surface-variants",
        type=int,
        default=2,
        help="Number of surface styles to emit per geometry scenario.",
    )
    parser.add_argument(
        "--relation-scale-variants",
        type=int,
        default=3,
        help="Number of global magnitude scale variants to emit per geometry scenario.",
    )
    parser.add_argument(
        "--phase-name",
        default="phase13_set_geometry_portfolio_ladder_v1",
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
        log_id_base = 2_147_245_000
    else:
        log_id_base = 2_147_240_000

    build_result = build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=args.output_dir,
            dataset_preset="phase13_set_geometry_portfolio_ladder",
            permutation_variants=args.permutation_variants,
            profile_surface_variants=args.profile_surface_variants,
            relation_scale_variants=args.relation_scale_variants,
            include_settings_variants=True,
            log_id_base=log_id_base,
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
