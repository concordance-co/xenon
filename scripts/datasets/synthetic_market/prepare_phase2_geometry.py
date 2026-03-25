from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())
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
    parser = argparse.ArgumentParser(description="Build and upload the dense synthetic geometry phase.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interp_exports/synthetic_market_phase2_geometry"),
        help="Local directory for synthetic geometry dataset artifacts.",
    )
    parser.add_argument(
        "--scalar-steps",
        type=int,
        default=41,
        help="Number of scalar sweep positions per family.",
    )
    parser.add_argument(
        "--scalar-background-variants",
        type=int,
        default=6,
        help="Number of repeated dense-background rosters per scalar family.",
    )
    parser.add_argument(
        "--minimal-scalar-templates",
        type=int,
        default=2,
        help="Number of ultra-minimal distractor templates per scalar family.",
    )
    parser.add_argument(
        "--phase-name",
        default="phase2_geometry",
        help="Phase tag stored in Neon.",
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
            dataset_preset="phase2_geometry",
            scalar_steps=args.scalar_steps,
            scalar_background_variants=args.scalar_background_variants,
            minimal_scalar_templates=args.minimal_scalar_templates,
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
