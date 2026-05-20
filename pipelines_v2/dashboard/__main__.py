"""CLI entrypoint: `uv run -m pipelines_v2.dashboard serve [...]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from pipelines_v2.core.config import load_workspace_config
from pipelines_v2.core.env import load_dotenv_if_present


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipelines_v2.dashboard")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the dashboard API server.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument(
        "--local-catalog-root",
        default=None,
        help="Optional local catalog root; falls back to xenon.toml or ~/.xenon/pipelines_v2/catalog.",
    )
    serve.add_argument(
        "--catalog-postgres-env",
        default=None,
        help="Optional env var name for a Postgres-backed catalog. Falls back to xenon.toml when omitted.",
    )
    serve.add_argument(
        "--static-dir",
        default=None,
        help="Optional prebuilt frontend directory (e.g. dashboard/dist). Falls back to xenon.toml when omitted.",
    )
    serve.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload for development.",
    )
    serve.add_argument(
        "--cache-list-ttl",
        type=float,
        default=15.0,
        help="TTL (seconds) for list_workflow_runs cache entries.",
    )
    serve.add_argument(
        "--cache-hot-ttl",
        type=float,
        default=15.0,
        help="TTL (seconds) for in-flight (non-terminal) run/step entries.",
    )
    serve.add_argument(
        "--cache-cold-ttl",
        type=float,
        default=3600.0,
        help="TTL (seconds) for terminal runs + artifact manifests (write-once).",
    )
    serve.add_argument(
        "--cache-maxsize",
        type=int,
        default=2048,
        help="Maximum number of cache entries before LRU eviction.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv_if_present()
    ns = _build_parser().parse_args(list(argv) if argv is not None else None)
    if ns.command == "serve":
        return _serve(ns)
    return 1


def _serve(ns: argparse.Namespace) -> int:
    import uvicorn

    from pipelines_v2.dashboard.server import create_app

    config = load_workspace_config()
    app = create_app(
        local_root=Path(ns.local_catalog_root) if ns.local_catalog_root else None,
        catalog_postgres_env=ns.catalog_postgres_env,
        static_dir=Path(ns.static_dir) if ns.static_dir else config.dashboard_static_dir(),
        cache_list_ttl=ns.cache_list_ttl,
        cache_hot_ttl=ns.cache_hot_ttl,
        cache_cold_ttl=ns.cache_cold_ttl,
        cache_maxsize=ns.cache_maxsize,
    )
    uvicorn.run(app, host=ns.host, port=ns.port, reload=ns.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
