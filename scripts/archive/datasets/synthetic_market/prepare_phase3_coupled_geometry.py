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
    parser = argparse.ArgumentParser(description="Build and upload the coupled-factor synthetic geometry phase.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interp_exports/synthetic_market_phase3_coupled_geometry"),
        help="Local directory for coupled synthetic geometry artifacts.",
    )
    parser.add_argument(
        "--grid-steps",
        type=int,
        default=11,
        help="Number of positions along each coupled axis.",
    )
    parser.add_argument(
        "--coupled-background-variants",
        type=int,
        default=2,
        help="Number of dense background rosters per coupled family.",
    )
    parser.add_argument(
        "--coupled-minimal-templates",
        type=int,
        default=1,
        help="Number of minimal distractor templates per coupled family.",
    )
    parser.add_argument(
        "--phase-name",
        default="phase3_coupled_geometry",
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
        log_id_base = 2_120_000_000
    else:
        log_id_base = 2_130_000_000
    build_result = build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=args.output_dir,
            dataset_preset="phase3_coupled_geometry",
            coupled_grid_steps=args.grid_steps,
            coupled_background_variants=args.coupled_background_variants,
            coupled_minimal_templates=args.coupled_minimal_templates,
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
