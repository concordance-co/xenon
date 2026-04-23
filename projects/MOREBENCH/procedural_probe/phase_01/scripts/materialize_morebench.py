from __future__ import annotations

"""Materialize MoReBench criterion-level rows from Hugging Face datasets."""

import argparse
from pathlib import Path

from projects.MOREBENCH.shared.morebench_dataset import (
    DEFAULT_SPLIT,
    MOREBENCH_REPO,
    PUBLIC_CONFIG,
    materialize_criterion_records,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=MOREBENCH_REPO)
    parser.add_argument("--config", default=PUBLIC_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output",
        default="projects/MOREBENCH/procedural_probe/phase_01/outputs/morebench_criteria.jsonl",
    )
    args = parser.parse_args()

    records = materialize_criterion_records(
        config=args.config,
        split=args.split,
        repo=args.repo,
        revision=args.revision,
        token=args.token,
        limit=args.limit,
    )
    write_jsonl(records, Path(args.output))
    print(f"wrote {len(records)} criterion rows to {args.output}")


if __name__ == "__main__":
    main()
