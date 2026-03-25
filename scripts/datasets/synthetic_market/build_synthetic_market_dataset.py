from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.interp.synthetic_market import (
    SyntheticMarketConfig,
    build_synthetic_market_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a synthetic market-manifold dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interp_exports/synthetic_market"),
        help="Directory for prompts and parquet exports.",
    )
    parser.add_argument(
        "--scalar-steps",
        type=int,
        default=9,
        help="Number of steps per scalar sweep family.",
    )
    parser.add_argument(
        "--pairwise-variants",
        type=int,
        default=5,
        help="Number of tradeoff variants per pairwise family.",
    )
    parser.add_argument(
        "--archetype-variants",
        type=int,
        default=4,
        help="Number of perturbed variants per archetype family.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for future stochastic expansions.",
    )
    parser.add_argument(
        "--no-settings-variants",
        action="store_true",
        help="Disable low-risk/high-risk context variants and emit market-only prompts only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_synthetic_market_dataset(
        SyntheticMarketConfig(
            seed=args.seed,
            scalar_steps=args.scalar_steps,
            pairwise_variants=args.pairwise_variants,
            archetype_variants=args.archetype_variants,
            include_settings_variants=not args.no_settings_variants,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
