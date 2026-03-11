"""Local CLI for querying the Xenon backend API.

Usage:
    uv run -m pipelines.backend query "SELECT COUNT(*) FROM vaults"
    uv run -m pipelines.backend stats
    uv run -m pipelines.backend schema vaults
    uv run -m pipelines.backend tables
    uv run -m pipelines.backend sample inference_logs 5
    uv run -m pipelines.backend parquet-list
    uv run -m pipelines.backend parquet-info interp_examples_v0_high_quality.parquet
    uv run -m pipelines.backend parquet-sample interp_sample_trade_v0.parquet 10
    uv run -m pipelines.backend activations
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

ENV_VAR = "XENON_BACKEND_URL"
URL_FILE = os.path.expanduser("~/.xenon_backend_url")


def _get_base_url() -> str:
    url = os.environ.get(ENV_VAR)
    if url:
        return url.rstrip("/")
    if os.path.exists(URL_FILE):
        with open(URL_FILE) as f:
            url = f.read().strip()
        if url:
            return url.rstrip("/")
    print(
        f"Error: No backend URL configured.\n"
        f"Set {ENV_VAR} env var or write the URL to {URL_FILE}\n"
        f"Deploy first: uv run --extra modal modal deploy pipelines/backend/app.py",
        file=sys.stderr,
    )
    sys.exit(1)


def _request(method: str, path: str, body: dict | None = None) -> dict:
    base = _get_base_url()
    url = f"{base}{path}"
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode() if e.fp else str(e)
        try:
            detail = json.loads(detail).get("detail", detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        print(f"Error {e.code}: {detail}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        sys.exit(1)


def _print_table(columns: list[str], rows: list[dict], max_col_width: int = 60) -> None:
    if not rows:
        print("(no rows)")
        return

    # Compute column widths
    widths = {c: len(c) for c in columns}
    for row in rows:
        for c in columns:
            val = str(row.get(c, ""))
            if len(val) > max_col_width:
                val = val[:max_col_width - 3] + "..."
            widths[c] = max(widths[c], len(val))

    # Header
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-+-".join("-" * widths[c] for c in columns))

    # Rows
    for row in rows:
        vals = []
        for c in columns:
            val = str(row.get(c, ""))
            if len(val) > max_col_width:
                val = val[:max_col_width - 3] + "..."
            vals.append(val.ljust(widths[c]))
        print(" | ".join(vals))


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2))


# --- Subcommands ---

def cmd_query(args: argparse.Namespace) -> None:
    result = _request("POST", "/query", {"sql": args.sql, "limit": args.limit})
    if args.json:
        _print_json(result)
    else:
        print(f"({result['row_count']} rows)")
        _print_table(result["columns"], result["rows"])

def cmd_stats(args: argparse.Namespace) -> None:
    _print_json(_request("GET", "/stats"))

def cmd_schema(args: argparse.Namespace) -> None:
    path = f"/schema?table={args.table}" if args.table else "/schema"
    result = _request("GET", path)
    if "tables" in result:
        for t in result["tables"]:
            print(t)
    else:
        _print_table(
            ["name", "type", "notnull", "pk", "default"],
            result["columns"],
        )

def cmd_tables(args: argparse.Namespace) -> None:
    result = _request("GET", "/tables")
    _print_table(["name", "count"], result["tables"])

def cmd_sample(args: argparse.Namespace) -> None:
    result = _request("GET", f"/sample/{args.table}?n={args.n}")
    print(f"({result['row_count']} rows from {result['table']})")
    _print_table(result["columns"], result["rows"])

def cmd_parquet_list(args: argparse.Namespace) -> None:
    result = _request("GET", "/parquet/list")
    _print_table(["name", "dir", "size_mb"], result["files"])

def cmd_parquet_info(args: argparse.Namespace) -> None:
    result = _request("GET", f"/parquet/info/{args.name}")
    print(f"File: {result['name']}")
    print(f"Rows: {result['num_rows']:,}")
    print(f"Columns: {result['num_columns']}")
    print(f"Row groups: {result['num_row_groups']}")
    print(f"Size: {result['size_mb']} MB")
    print()
    _print_table(["name", "type"], result["schema"])

def cmd_parquet_sample(args: argparse.Namespace) -> None:
    params = f"n={args.n}"
    if args.columns:
        params += f"&columns={args.columns}"
    result = _request("GET", f"/parquet/sample/{args.name}?{params}")
    print(f"({result['row_count']} of {result['total_rows']:,} rows from {result['name']})")
    _print_table(result["columns"], result["rows"])

def cmd_activations(args: argparse.Namespace) -> None:
    result = _request("GET", f"/activations/meta?limit={args.limit}")
    print(f"({result['total']} total activations)")
    if result["rows"]:
        _print_table(result["columns"], result["rows"])

def cmd_health(args: argparse.Namespace) -> None:
    _print_json(_request("GET", "/health"))

def cmd_reload(args: argparse.Namespace) -> None:
    _print_json(_request("POST", "/reload"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pipelines.backend",
        description="Query the Xenon backend API",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # query
    p = sub.add_parser("query", aliases=["q"], help="Run SQL query")
    p.add_argument("sql", help="SQL query string")
    p.add_argument("--limit", type=int, default=100, help="Max rows (default: 100)")
    p.add_argument("--json", action="store_true", help="Output raw JSON")
    p.set_defaults(func=cmd_query)

    # stats
    p = sub.add_parser("stats", help="Dashboard stats")
    p.set_defaults(func=cmd_stats)

    # schema
    p = sub.add_parser("schema", help="Table schema")
    p.add_argument("table", nargs="?", default="", help="Table name (omit for list)")
    p.set_defaults(func=cmd_schema)

    # tables
    p = sub.add_parser("tables", help="Table row counts")
    p.set_defaults(func=cmd_tables)

    # sample
    p = sub.add_parser("sample", help="Sample rows from table")
    p.add_argument("table", help="Table name")
    p.add_argument("n", nargs="?", type=int, default=10, help="Number of rows")
    p.set_defaults(func=cmd_sample)

    # parquet-list
    p = sub.add_parser("parquet-list", aliases=["pql"], help="List parquet files")
    p.set_defaults(func=cmd_parquet_list)

    # parquet-info
    p = sub.add_parser("parquet-info", aliases=["pqi"], help="Parquet file metadata")
    p.add_argument("name", help="Parquet filename")
    p.set_defaults(func=cmd_parquet_info)

    # parquet-sample
    p = sub.add_parser("parquet-sample", aliases=["pqs"], help="Sample from parquet")
    p.add_argument("name", help="Parquet filename")
    p.add_argument("n", nargs="?", type=int, default=10, help="Number of rows")
    p.add_argument("--columns", default="", help="Comma-separated column names")
    p.set_defaults(func=cmd_parquet_sample)

    # activations
    p = sub.add_parser("activations", aliases=["act"], help="Activation metadata")
    p.add_argument("--limit", type=int, default=50, help="Max rows")
    p.set_defaults(func=cmd_activations)

    # health
    p = sub.add_parser("health", help="Health check")
    p.set_defaults(func=cmd_health)

    # reload
    p = sub.add_parser("reload", help="Force volume refresh")
    p.set_defaults(func=cmd_reload)

    args = parser.parse_args(argv)
    args.func(args)
