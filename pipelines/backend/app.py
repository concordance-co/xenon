"""Modal backend API for querying Xenon data in Neon Postgres.

Usage:
    uv run --extra modal modal serve pipelines/backend/app.py    # dev
    uv run --extra modal modal deploy pipelines/backend/app.py   # prod

Endpoints:
    GET  /health                    Health check
    POST /query                     Run read-only SQL against Neon
    GET  /schema                    List tables or get one table's schema
    GET  /tables                    Row counts for all tables
    GET  /stats                     Cached dashboard_stats.json
    GET  /stats/live                Live stats from Neon (slower)
    GET  /sample/{table}            Sample N rows from a table
    GET  /parquet/list              List parquet files on the volume
    GET  /parquet/info/{name}       Parquet file metadata
    GET  /parquet/sample/{name}     Sample N rows from a parquet file
    GET  /activations/meta          Activations metadata.parquet summary
    POST /profile/dataset           Read-only dataset profiling
    POST /label/preview             Label-method and split viability preview
    GET  /prep-targets              List shared prep target specs
    GET  /prep-targets/{id}         Get one shared prep target spec
    POST /prep-targets              Create or update a shared prep target spec
    DELETE /prep-targets/{id}       Delete a shared prep target spec
    POST /reload                    Force volume refresh
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None  # type: ignore[assignment]
    sql = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]

try:
    import modal
except Exception:  # pragma: no cover - local fallback for tests without modal installed
    class _ModalStubApp:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def function(self, *_: Any, **__: Any):
            def _decorator(fn):
                return fn

            return _decorator

    class _ModalStubVolume:
        @staticmethod
        def from_name(*_: Any, **__: Any):
            return _ModalStubVolume()

        def reload(self) -> None:
            return None

    class _ModalStubImage:
        @staticmethod
        def debian_slim(*_: Any, **__: Any):
            return _ModalStubImage()

        def pip_install(self, *_: Any, **__: Any):
            return self

        def add_local_python_source(self, *_: Any, **__: Any):
            return self

    class _ModalStubSecret:
        @staticmethod
        def from_name(*_: Any, **__: Any):
            return _ModalStubSecret()

    class _ModalStubModule:
        App = _ModalStubApp
        Volume = _ModalStubVolume
        Image = _ModalStubImage
        Secret = _ModalStubSecret

        @staticmethod
        def asgi_app():
            def _decorator(fn):
                return fn

            return _decorator

    modal = _ModalStubModule()  # type: ignore[assignment]

app = modal.App("xenon-backend")

volume = modal.Volume.from_name("xenon-data", create_if_missing=True)

neon_secret = modal.Secret.from_name("xenon-neon")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("fastapi[standard]", "pyarrow", "numpy", "psycopg[binary,pool]")
    .add_local_python_source("pipelines")
)

EXPORTS_DIR = "/data/interp_exports"
ACTIVATIONS_DIR = "/data/activations"
STATS_PATH = "/data/dashboard_stats.json"

_READ_ONLY_PREFIXES = ("select", "explain", "with")
_MAX_DEFAULT_LIMIT = 1000
_MAX_PROFILE_LIMIT = 5000
_MAX_LABEL_PREVIEW_LIMIT = 10000
_SQL_FRAGMENT_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|vacuum|reindex|truncate|grant|revoke)\b",
    re.IGNORECASE,
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_sql(sql_str: str) -> str:
    return sql_str.strip().rstrip(";")


def _validate_read_only_sql(sql_str: str) -> str:
    """Validate SQL is read-only and return normalized SQL."""
    normalized = _normalize_sql(sql_str)
    lowered = normalized.lower()
    if not lowered.startswith(_READ_ONLY_PREFIXES):
        raise ValueError(
            "Only read-only queries allowed (SELECT/EXPLAIN/WITH). "
            f"Got: {lowered[:30]}..."
        )
    if _SQL_FRAGMENT_FORBIDDEN.search(lowered):
        raise ValueError("Read-only SQL cannot contain mutating keywords")
    return normalized


def _validate_sql_fragment(fragment: str, name: str) -> str:
    """Validate SQL expression/where fragments embedded into read-only SELECTs."""
    cleaned = fragment.strip()
    if not cleaned:
        raise ValueError(f"{name} cannot be empty")
    if ";" in cleaned:
        raise ValueError(f"{name} cannot contain ';'")
    if _SQL_FRAGMENT_FORBIDDEN.search(cleaned):
        raise ValueError(f"{name} contains forbidden SQL keywords")
    return cleaned


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _infer_type(values: list[Any]) -> str:
    sample = [v for v in values if v is not None]
    if not sample:
        return "null"
    if all(isinstance(v, bool) for v in sample):
        return "boolean"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in sample):
        return "integer"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in sample):
        return "number"
    if all(isinstance(v, str) for v in sample):
        return "string"
    return "mixed"


def _column_profiles(
    rows: list[dict[str, Any]],
    columns: list[str],
    selected: set[str] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    total = len(rows)
    for col in columns:
        if selected is not None and col not in selected:
            continue
        vals = [r.get(col) for r in rows]
        null_count = sum(1 for v in vals if v is None)
        non_null = [v for v in vals if v is not None]
        distinct = []
        seen: set[str] = set()
        for value in non_null:
            as_text = str(value)
            if as_text in seen:
                continue
            seen.add(as_text)
            distinct.append(value)
            if len(distinct) == 5:
                break
        out.append(
            {
                "column": col,
                "null_count": null_count,
                "null_rate": round((null_count / total) if total else 0.0, 4),
                "distinct_count": len({str(v) for v in non_null}),
                "type": _infer_type(vals),
                "sample_values": distinct,
            }
        )
    return out


def _label_distribution(values: list[Any]) -> list[dict[str, Any]]:
    counts = Counter("NULL" if v is None else str(v) for v in values)
    total = sum(counts.values())
    rows = []
    for label, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        rows.append(
            {
                "label": label,
                "count": count,
                "pct": round((count / total) if total else 0.0, 4),
            }
        )
    return rows


def _split_viability(label_counts: dict[str, int], split: dict[str, Any]) -> dict[str, Any]:
    total = sum(label_counts.values())
    reasons: list[str] = []

    train_pct = float(split.get("train_pct", 70))
    val_pct = float(split.get("val_pct", 15))
    test_pct = float(split.get("test_pct", 15))
    pct_sum = train_pct + val_pct + test_pct

    if abs(pct_sum - 100.0) > 0.01:
        reasons.append(f"split percentages must sum to 100 (got {pct_sum:.2f})")

    train_n = int(round(total * train_pct / 100.0))
    val_n = int(round(total * val_pct / 100.0))
    test_n = max(0, total - train_n - val_n)

    if train_pct > 0 and train_n == 0:
        reasons.append("train split receives 0 rows")
    if val_pct > 0 and val_n == 0:
        reasons.append("validation split receives 0 rows")
    if test_pct > 0 and test_n == 0:
        reasons.append("test split receives 0 rows")

    for label, count in sorted(label_counts.items()):
        if train_pct > 0 and count * train_pct / 100.0 < 1:
            reasons.append(f"label '{label}' too rare for train split")
        if val_pct > 0 and count * val_pct / 100.0 < 1:
            reasons.append(f"label '{label}' too rare for validation split")
        if test_pct > 0 and count * test_pct / 100.0 < 1:
            reasons.append(f"label '{label}' too rare for test split")

    mode = split.get("mode", "random_stratified")
    if mode == "group_holdout" and not split.get("group_key"):
        reasons.append("group_holdout requires split.group_key")
    if mode == "time_based" and not split.get("time_key"):
        reasons.append("time_based requires split.time_key")

    return {
        "viable": len(reasons) == 0,
        "reasons": reasons,
        "counts": {
            "train": train_n,
            "val": val_n,
            "test": test_n,
            "total": total,
        },
    }


def _build_label_expression(label: dict[str, Any]) -> str:
    mode = label.get("mode")
    expression_sql = _validate_sql_fragment(str(label.get("expression_sql", "")), "label.expression_sql")

    if mode == "direct":
        return f"({expression_sql})"

    if mode == "binary_rule":
        classes = label.get("classes") or ["negative", "positive"]
        negative = str(classes[0]) if len(classes) >= 1 else "negative"
        positive = str(classes[1]) if len(classes) >= 2 else "positive"
        return (
            f"CASE WHEN ({expression_sql}) THEN {_sql_literal(positive)} "
            f"ELSE {_sql_literal(negative)} END"
        )

    if mode == "bucket":
        buckets = label.get("buckets") or []
        if not buckets:
            raise ValueError("label.buckets is required for bucket mode")
        branches: list[str] = []
        for bucket in buckets:
            name = str(bucket.get("name", "bucket"))
            lower = bucket.get("min")
            upper = bucket.get("max")
            conds = []
            if lower is not None:
                conds.append(f"({expression_sql}) >= {float(lower)}")
            if upper is not None:
                conds.append(f"({expression_sql}) < {float(upper)}")
            if not conds:
                raise ValueError(f"bucket '{name}' must have min and/or max")
            branches.append(f"WHEN {' AND '.join(conds)} THEN {_sql_literal(name)}")
        return "CASE " + " ".join(branches) + " ELSE NULL END"

    raise ValueError(f"Unknown label mode: {mode}")


def _coerce_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _activation_coverage(
    activations_dir: Path,
    rows: list[dict[str, Any]],
    label_key: str = "__label_value",
) -> dict[str, Any]:
    meta_path = activations_dir / "metadata.parquet"
    if not meta_path.exists():
        return {
            "available": False,
            "reason": "metadata.parquet missing",
            "eligible_labeled": 0,
            "matched": 0,
            "coverage": None,
        }

    labeled_ids: set[int] = set()
    for row in rows:
        if row.get(label_key) is None:
            continue
        lid = row.get("log_id")
        if lid is None:
            continue
        try:
            labeled_ids.add(int(lid))
        except Exception:
            continue

    if not labeled_ids:
        return {
            "available": True,
            "reason": "no labeled rows with log_id",
            "eligible_labeled": 0,
            "matched": 0,
            "coverage": 0.0,
        }

    import pyarrow.parquet as pq

    table = pq.read_table(meta_path, columns=["log_id"])
    activation_ids = {int(x) for x in table.column("log_id").to_pylist() if x is not None}
    matched = len(labeled_ids & activation_ids)
    return {
        "available": True,
        "eligible_labeled": len(labeled_ids),
        "matched": matched,
        "coverage": round(matched / len(labeled_ids), 4),
    }


def _compute_live_stats(conn: psycopg.Connection) -> dict[str, Any]:
    """Compute dashboard stats by querying Neon directly (exact counts)."""
    from pipelines.stats import compute_stats

    return compute_stats(conn, approximate_counts=False)


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=300,
    cpu=1,
    scaledown_window=900,
    secrets=[neon_secret],
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web_app():
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    import pyarrow.parquet as pq
    import psycopg_pool

    from pipelines.db import require_neon_dsn

    api = FastAPI(title="Xenon Backend", version="0.4.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    exports_dir = Path(EXPORTS_DIR)
    activations_dir = Path(ACTIVATIONS_DIR)
    stats_path = Path(STATS_PATH)

    # --- Connection pool ---
    dsn = require_neon_dsn()
    pool = psycopg_pool.ConnectionPool(
        dsn,
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )

    def _validate_table_name(conn: psycopg.Connection, table: str) -> None:
        row = conn.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = %s)",
            [table],
        ).fetchone()
        if not row or not row["exists"]:
            raise HTTPException(status_code=404, detail=f"Table '{table}' not found")

    def _resolve_parquet(name: str) -> Path:
        if ".." in name or "/" in name:
            raise HTTPException(status_code=400, detail="Invalid filename")
        for d in [exports_dir, activations_dir]:
            p = d / name
            if p.exists() and p.suffix == ".parquet":
                return p
        raise HTTPException(status_code=404, detail=f"Parquet file '{name}' not found")

    def _source_sql(conn: psycopg.Connection, source: dict[str, Any]) -> str:
        mode = source.get("mode", "table")
        if mode == "table":
            table = source.get("table")
            if not isinstance(table, str) or not table.strip():
                raise HTTPException(status_code=400, detail="source.table is required for table mode")
            _validate_table_name(conn, table)
            return sql.SQL("SELECT * FROM {}").format(sql.Identifier(table)).as_string(conn)
        if mode == "sql":
            raw_sql = source.get("sql")
            if not isinstance(raw_sql, str) or not raw_sql.strip():
                raise HTTPException(status_code=400, detail="source.sql is required for sql mode")
            try:
                return _validate_read_only_sql(raw_sql)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=400, detail=f"Unsupported source mode: {mode}")

    @api.get("/health")
    def health():
        info: dict[str, Any] = {"status": "ok", "db_type": "neon_postgres"}
        try:
            with pool.connection() as conn:
                conn.execute("SELECT 1")
            info["db_connected"] = True
        except Exception:
            info["db_connected"] = False
        return info

    @api.post("/query")
    def query(req: dict[str, Any]):
        if not isinstance(req, dict):
            raise HTTPException(status_code=400, detail="Request body must be an object")
        sql_in = req.get("sql")
        if not isinstance(sql_in, str) or not sql_in.strip():
            raise HTTPException(status_code=400, detail="sql is required")
        limit_in = _coerce_int(req.get("limit"), 100)
        try:
            validated_sql = _validate_read_only_sql(sql_in)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not re.search(r"\bLIMIT\b", validated_sql, re.IGNORECASE):
            limit = min(max(1, limit_in), _MAX_DEFAULT_LIMIT)
            validated_sql = f"{validated_sql} LIMIT {limit}"

        try:
            with pool.connection() as conn:
                cursor = conn.execute(validated_sql)
                columns = [d.name for d in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                return {
                    "columns": columns,
                    "rows": [dict(r) for r in rows],
                    "row_count": len(rows),
                    "sql": validated_sql,
                }
        except psycopg.Error as e:
            raise HTTPException(status_code=503, detail=str(e))

    @api.get("/schema")
    def schema(table: str = Query(default="")):
        try:
            with pool.connection() as conn:
                if not table:
                    rows = conn.execute(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
                        "ORDER BY table_name"
                    ).fetchall()
                    return {"tables": [r["table_name"] for r in rows]}

                _validate_table_name(conn, table)
                cols = conn.execute(
                    "SELECT column_name, data_type, is_nullable, column_default "
                    "FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = %s "
                    "ORDER BY ordinal_position",
                    [table],
                ).fetchall()

                # Get primary key columns
                pk_cols = conn.execute(
                    "SELECT a.attname "
                    "FROM pg_index i "
                    "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
                    "WHERE i.indrelid = %s::regclass AND i.indisprimary",
                    [table],
                ).fetchall()
                pk_set = {r["attname"] for r in pk_cols}

                return {
                    "table": table,
                    "columns": [
                        {
                            "name": c["column_name"],
                            "type": c["data_type"],
                            "notnull": c["is_nullable"] == "NO",
                            "pk": c["column_name"] in pk_set,
                            "default": c["column_default"],
                        }
                        for c in cols
                    ],
                }
        except psycopg.Error as e:
            raise HTTPException(status_code=503, detail=f"Database error: {e}")

    @api.get("/tables")
    def tables():
        try:
            with pool.connection() as conn:
                names = conn.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
                    "ORDER BY table_name"
                ).fetchall()
                result = []
                for r in names:
                    name = r["table_name"]
                    count = conn.execute(
                        sql.SQL("SELECT COUNT(*) AS n FROM {}").format(sql.Identifier(name))
                    ).fetchone()
                    result.append({"name": name, "count": count["n"]})
                return {"tables": result}
        except psycopg.Error as e:
            raise HTTPException(status_code=503, detail=f"Database error: {e}")

    @api.get("/stats")
    def stats():
        if not stats_path.exists():
            raise HTTPException(status_code=404, detail="No dashboard_stats.json on volume")
        return json.loads(stats_path.read_text())

    @api.get("/stats/live")
    def stats_live():
        """Query Neon directly for live dashboard stats (slower than cached /stats)."""
        try:
            with pool.connection() as conn:
                return _compute_live_stats(conn)
        except psycopg.Error as e:
            raise HTTPException(status_code=503, detail=f"Database error: {e}")

    @api.get("/sample/{table}")
    def sample(table: str, n: int = Query(default=10, le=500)):
        try:
            with pool.connection() as conn:
                _validate_table_name(conn, table)
                rows = conn.execute(
                    sql.SQL("SELECT * FROM {} ORDER BY RANDOM() LIMIT %s").format(sql.Identifier(table)),
                    [n],
                ).fetchall()
                columns = list(rows[0].keys()) if rows else []
                return {
                    "table": table,
                    "columns": columns,
                    "rows": [dict(r) for r in rows],
                    "row_count": len(rows),
                }
        except psycopg.Error as e:
            raise HTTPException(status_code=503, detail=f"Database error: {e}")

    @api.get("/parquet/list")
    def parquet_list():
        files = []
        for d in [exports_dir, activations_dir]:
            if d.exists():
                for f in sorted(d.glob("*.parquet")):
                    size = f.stat().st_size
                    files.append(
                        {
                            "name": f.name,
                            "dir": str(d),
                            "size_mb": round(size / 1024 / 1024, 2),
                        }
                    )
        return {"files": files}

    @api.get("/parquet/info/{name}")
    def parquet_info(name: str):
        p = _resolve_parquet(name)
        meta = pq.read_metadata(p)
        pq_schema = pq.read_schema(p)
        return {
            "name": name,
            "num_rows": meta.num_rows,
            "num_columns": meta.num_columns,
            "num_row_groups": meta.num_row_groups,
            "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
            "schema": [
                {"name": pq_schema.field(i).name, "type": str(pq_schema.field(i).type)}
                for i in range(len(pq_schema))
            ],
        }

    @api.get("/parquet/sample/{name}")
    def parquet_sample(
        name: str,
        n: int = Query(default=10, le=100),
        columns: str = Query(default=""),
    ):
        p = _resolve_parquet(name)
        col_list = [c.strip() for c in columns.split(",") if c.strip()] or None
        try:
            t = pq.read_table(p, columns=col_list)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        total = t.num_rows
        t = t.slice(0, min(n, total))
        rows = t.to_pylist()
        return {
            "name": name,
            "total_rows": total,
            "columns": t.column_names,
            "rows": rows,
            "row_count": len(rows),
        }

    @api.get("/activations/meta")
    def activations_meta(limit: int = Query(default=50, le=500)):
        meta_path = activations_dir / "metadata.parquet"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="No activations metadata.parquet")
        t = pq.read_table(meta_path)
        rows = t.slice(0, min(limit, t.num_rows)).to_pylist()
        return {
            "total": t.num_rows,
            "columns": t.column_names,
            "rows": rows,
        }

    @api.post("/profile/dataset")
    def profile_dataset(req: dict[str, Any]):
        if not isinstance(req, dict):
            raise HTTPException(status_code=400, detail="Request body must be an object")
        try:
            with pool.connection() as conn:
                source_data = _coerce_dict(req.get("source"), "source")
                source = _source_sql(conn, source_data)
                limit = max(1, min(_coerce_int(req.get("limit"), 1000), _MAX_PROFILE_LIMIT))
                cursor_sql = f"SELECT * FROM ({source}) AS src LIMIT %s"
                cursor = conn.execute(cursor_sql, [limit])
                columns = [d.name for d in cursor.description] if cursor.description else []
                rows = [dict(r) for r in cursor.fetchall()]

                columns_req = req.get("columns")
                selected = (
                    {str(v) for v in columns_req if isinstance(v, str)}
                    if isinstance(columns_req, list)
                    else None
                )
                profile_rows = _column_profiles(rows, columns, selected)

                label_balance = []
                label_column = req.get("label_column")
                if isinstance(label_column, str) and label_column in columns:
                    label_balance = _label_distribution([r.get(label_column) for r in rows])

                stratified = []
                stratify_by = req.get("stratify_by")
                if (
                    isinstance(label_column, str)
                    and isinstance(stratify_by, str)
                    and label_column in columns
                    and stratify_by in columns
                ):
                    matrix: defaultdict[tuple[str, str], int] = defaultdict(int)
                    for row in rows:
                        strat = "NULL" if row.get(stratify_by) is None else str(row.get(stratify_by))
                        label = "NULL" if row.get(label_column) is None else str(row.get(label_column))
                        matrix[(strat, label)] += 1
                    stratified = [
                        {"stratum": s, "label": l, "count": c}
                        for (s, l), c in sorted(matrix.items(), key=lambda kv: (-kv[1], kv[0]))
                    ]

                return {
                    "source_sql": source,
                    "sample_limit": limit,
                    "row_count": len(rows),
                    "columns": columns,
                    "profiles": profile_rows,
                    "label_balance": label_balance,
                    "stratified": stratified,
                }
        except psycopg.Error as e:
            raise HTTPException(status_code=503, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @api.post("/label/preview")
    def label_preview(req: dict[str, Any]):
        if not isinstance(req, dict):
            raise HTTPException(status_code=400, detail="Request body must be an object")
        try:
            with pool.connection() as conn:
                spec = _coerce_dict(req.get("spec"), "spec")
                source = _source_sql(conn, _coerce_dict(spec.get("source"), "spec.source"))
                label_expr = _build_label_expression(_coerce_dict(spec.get("label"), "spec.label"))

                where_clause = ""
                filters = spec.get("filters") or {}
                if filters.get("sql_where"):
                    where_sql = _validate_sql_fragment(str(filters["sql_where"]), "filters.sql_where")
                    where_clause = f" WHERE ({where_sql})"

                limit = max(1, min(_coerce_int(req.get("limit"), 2000), _MAX_LABEL_PREVIEW_LIMIT))
                query_sql = (
                    "SELECT src.*, "
                    f"{label_expr} AS __label_value "
                    f"FROM ({source}) AS src{where_clause} "
                    "LIMIT %s"
                )
                cursor = conn.execute(query_sql, [limit])
                columns = [d.name for d in cursor.description] if cursor.description else []
                rows = [dict(r) for r in cursor.fetchall()]

                label_counts_raw = Counter(
                    str(r.get("__label_value")) for r in rows if r.get("__label_value") is not None
                )
                labeled_count = sum(label_counts_raw.values())
                label_dist = _label_distribution([r.get("__label_value") for r in rows])
                split_data = spec.get("split")
                split_eval = _split_viability(
                    label_counts_raw,
                    split_data if isinstance(split_data, dict) else {},
                )
                coverage = _activation_coverage(activations_dir, rows)

                min_class = min(label_counts_raw.values()) if label_counts_raw else 0
                max_class = max(label_counts_raw.values()) if label_counts_raw else 0
                class_ratio = round((max_class / min_class), 4) if min_class > 0 else None
                can_probe = len(label_counts_raw) >= 2 and min_class >= 4

                readiness_reasons = []
                if len(label_counts_raw) < 2:
                    readiness_reasons.append("need at least 2 non-null classes")
                if min_class < 4:
                    readiness_reasons.append("smallest class has fewer than 4 examples")
                if not split_eval["viable"]:
                    readiness_reasons.append("split viability checks failed")

                return {
                    "sample_limit": limit,
                    "row_count": len(rows),
                    "labeled_count": labeled_count,
                    "columns": columns,
                    "label_distribution": label_dist,
                    "missing_labels": {
                        "count": len(rows) - labeled_count,
                        "rate": round(((len(rows) - labeled_count) / len(rows)) if rows else 0.0, 4),
                    },
                    "split_viability": split_eval,
                    "activation_coverage": coverage,
                    "probe_readiness": {
                        "can_probe": can_probe,
                        "class_count": len(label_counts_raw),
                        "min_class_count": min_class,
                        "imbalance_ratio": class_ratio,
                        "recommended_n_folds": max(2, min(5, min_class)) if min_class else 2,
                        "reasons": readiness_reasons,
                    },
                    "generated_sql": query_sql,
                }
        except psycopg.Error as e:
            raise HTTPException(status_code=503, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @api.get("/prep-targets")
    def prep_targets_list():
        try:
            with pool.connection() as conn:
                rows = conn.execute(
                    "SELECT id, name, description, spec_json, created_at, updated_at "
                    "FROM prep_target_specs ORDER BY updated_at DESC"
                ).fetchall()
                specs = []
                for r in rows:
                    spec = dict(r.get("spec_json") or {})
                    spec["id"] = r["id"]
                    spec["name"] = r["name"]
                    spec["description"] = r.get("description")
                    spec["created_at"] = r["created_at"]
                    spec["updated_at"] = r["updated_at"]
                    specs.append(spec)
                return {"specs": specs}
        except psycopg.Error as e:
            raise HTTPException(status_code=503, detail=f"Database error: {e}")

    @api.get("/prep-targets/{spec_id}")
    def prep_targets_get(spec_id: str):
        try:
            with pool.connection() as conn:
                row = conn.execute(
                    "SELECT id, name, description, spec_json, created_at, updated_at "
                    "FROM prep_target_specs WHERE id = %s",
                    [spec_id],
                ).fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail=f"Spec not found: {spec_id}")
                spec = dict(row.get("spec_json") or {})
                spec["id"] = row["id"]
                spec["name"] = row["name"]
                spec["description"] = row.get("description")
                spec["created_at"] = row["created_at"]
                spec["updated_at"] = row["updated_at"]
                return {"spec": spec}
        except psycopg.Error as e:
            raise HTTPException(status_code=503, detail=f"Database error: {e}")

    @api.post("/prep-targets")
    def prep_targets_upsert(spec: dict[str, Any]):
        if not isinstance(spec, dict):
            raise HTTPException(status_code=400, detail="Request body must be an object")
        if not isinstance(spec.get("name"), str) or not str(spec.get("name")).strip():
            raise HTTPException(status_code=400, detail="spec.name is required")
        if not isinstance(spec.get("source"), dict):
            raise HTTPException(status_code=400, detail="spec.source is required")
        if not isinstance(spec.get("label"), dict):
            raise HTTPException(status_code=400, detail="spec.label is required")
        if not isinstance(spec.get("split"), dict):
            raise HTTPException(status_code=400, detail="spec.split is required")

        spec_id = spec.get("id") or uuid.uuid4().hex[:12]
        now = _now_iso()
        name = spec["name"]
        description = spec.get("description")

        # Build spec_json: everything except the top-level columns
        top_keys = {"id", "name", "description", "created_at", "updated_at"}
        spec_json = {k: v for k, v in spec.items() if k not in top_keys}

        try:
            with pool.connection() as conn:
                # Check if exists to preserve created_at
                existing = conn.execute(
                    "SELECT created_at FROM prep_target_specs WHERE id = %s",
                    [spec_id],
                ).fetchone()
                created_at = existing["created_at"] if existing else (spec.get("created_at") or now)

                conn.execute(
                    "INSERT INTO prep_target_specs (id, name, description, spec_json, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s::jsonb, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "name = EXCLUDED.name, description = EXCLUDED.description, "
                    "spec_json = EXCLUDED.spec_json, updated_at = EXCLUDED.updated_at",
                    [spec_id, name, description, json.dumps(spec_json), created_at, now],
                )

                # Read back the full spec
                prepared = dict(spec_json)
                prepared["id"] = spec_id
                prepared["name"] = name
                prepared["description"] = description
                prepared["created_at"] = created_at
                prepared["updated_at"] = now

                count_row = conn.execute("SELECT COUNT(*) AS n FROM prep_target_specs").fetchone()
                total = count_row["n"] if count_row else 0

                return {"spec": prepared, "count": total}
        except psycopg.Error as e:
            raise HTTPException(status_code=503, detail=f"Database error: {e}")

    @api.delete("/prep-targets/{spec_id}")
    def prep_targets_delete(spec_id: str):
        try:
            with pool.connection() as conn:
                result = conn.execute(
                    "DELETE FROM prep_target_specs WHERE id = %s RETURNING id",
                    [spec_id],
                ).fetchone()
                if not result:
                    raise HTTPException(status_code=404, detail=f"Spec not found: {spec_id}")
                count_row = conn.execute("SELECT COUNT(*) AS n FROM prep_target_specs").fetchone()
                total = count_row["n"] if count_row else 0
                return {"status": "deleted", "id": spec_id, "count": total}
        except psycopg.Error as e:
            raise HTTPException(status_code=503, detail=f"Database error: {e}")

    @api.post("/reload")
    def reload():
        volume.reload()
        return {"status": "reloaded"}

    return api
