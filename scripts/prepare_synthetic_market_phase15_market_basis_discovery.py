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
    parser = argparse.ArgumentParser(description="Build and upload the phase-15 market-basis discovery dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interp_exports/synthetic_market_phase15_market_basis_discovery"),
        help="Local directory for phase-15 market-basis discovery artifacts.",
    )
    parser.add_argument("--discovery-scalar-steps", type=int, default=7)
    parser.add_argument("--discovery-background-variants", type=int, default=3)
    parser.add_argument("--discovery-grid-steps", type=int, default=5)
    parser.add_argument("--phase-name", default="phase15_market_basis_discovery_v1")
    parser.add_argument("--log-id-base", type=int, default=0)
    parser.add_argument("--keep-existing-phase", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.log_id_base > 0:
        log_id_base = args.log_id_base
    elif "smoke" in args.phase_name.lower():
        log_id_base = 2_147_265_000
    else:
        log_id_base = 2_147_260_000

    build_result = build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=args.output_dir,
            dataset_preset="phase15_market_basis_discovery",
            discovery_scalar_steps=args.discovery_scalar_steps,
            discovery_background_variants=args.discovery_background_variants,
            discovery_grid_steps=args.discovery_grid_steps,
            include_settings_variants=False,
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
