"""Modal backend API for querying Xenon data on the volume.

Usage:
    uv run --extra modal modal serve pipelines/backend/app.py    # dev
    uv run --extra modal modal deploy pipelines/backend/app.py   # prod

Endpoints:
    GET  /health                    Health check + volume info
    POST /query                     Run read-only SQL against the DB
    GET  /schema                    List tables or get one table's schema
    GET  /tables                    Row counts for all tables
    GET  /stats                     Cached dashboard_stats.json
    GET  /sample/{table}            Sample N rows from a SQLite table
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
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

    class _ModalStubModule:
        App = _ModalStubApp
        Volume = _ModalStubVolume
        Image = _ModalStubImage

        @staticmethod
        def asgi_app():
            def _decorator(fn):
                return fn

            return _decorator

    modal = _ModalStubModule()  # type: ignore[assignment]

app = modal.App("xenon-backend")

volume = modal.Volume.from_name("xenon-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("fastapi[standard]", "pyarrow", "numpy")
)

DB_PATH = "/data/ingest/terminal_ingest.db"
EXPORTS_DIR = "/data/interp_exports"
ACTIVATIONS_DIR = "/data/activations"
STATS_PATH = "/data/dashboard_stats.json"
PREP_TARGETS_PATH = "/data/explorer/prep_target_specs.json"

_READ_ONLY_PREFIXES = ("select", "pragma", "explain", "with")
_MAX_DEFAULT_LIMIT = 1000
_MAX_PROFILE_LIMIT = 5000
_MAX_LABEL_PREVIEW_LIMIT = 10000
_SQL_FRAGMENT_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|vacuum|reindex|analyze)\b",
    re.IGNORECASE,
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_sql(sql: str) -> str:
    return sql.strip().rstrip(";")


def _validate_read_only_sql(sql: str) -> str:
    """Validate SQL is read-only and return normalized SQL."""
    normalized = _normalize_sql(sql)
    lowered = normalized.lower()
    if not lowered.startswith(_READ_ONLY_PREFIXES):
        raise ValueError(
            "Only read-only queries allowed (SELECT/PRAGMA/EXPLAIN/WITH). "
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


def _load_prep_targets(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, dict)]


def _save_prep_targets_atomic(path: Path, specs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(specs, ensure_ascii=True, separators=(",", ":"), sort_keys=False)
    tmp.write_text(payload)
    tmp.replace(path)


def _upsert_prep_target(
    specs: list[dict[str, Any]],
    spec: dict[str, Any],
    now_iso: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = dict(spec)
    if not prepared.get("id"):
        prepared["id"] = uuid.uuid4().hex[:12]

    existing_idx = next((i for i, s in enumerate(specs) if s.get("id") == prepared["id"]), None)
    if existing_idx is None:
        prepared["created_at"] = prepared.get("created_at") or now_iso
        prepared["updated_at"] = now_iso
        specs.append(prepared)
    else:
        existing = specs[existing_idx]
        prepared["created_at"] = existing.get("created_at") or now_iso
        prepared["updated_at"] = now_iso
        specs[existing_idx] = prepared

    specs.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return specs, prepared


def _coerce_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
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


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=120,
    cpu=1,
    scaledown_window=3600,
)
@modal.asgi_app()
def web_app():
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    import pyarrow.parquet as pq

    api = FastAPI(title="Xenon Backend", version="0.2.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    db_path = Path(DB_PATH)
    exports_dir = Path(EXPORTS_DIR)
    activations_dir = Path(ACTIVATIONS_DIR)
    stats_path = Path(STATS_PATH)
    prep_targets_path = Path(PREP_TARGETS_PATH)

    def _connect() -> sqlite3.Connection:
        if not db_path.exists():
            raise HTTPException(status_code=503, detail="Database not found on volume")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _validate_table_name(conn: sqlite3.Connection, table: str) -> None:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", [table]
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Table '{table}' not found")

    def _resolve_parquet(name: str) -> Path:
        if ".." in name or "/" in name:
            raise HTTPException(status_code=400, detail="Invalid filename")
        for d in [exports_dir, activations_dir]:
            p = d / name
            if p.exists() and p.suffix == ".parquet":
                return p
        raise HTTPException(status_code=404, detail=f"Parquet file '{name}' not found")

    def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
        return [dict(r) for r in rows]

    def _source_sql(conn: sqlite3.Connection, source: dict[str, Any]) -> str:
        mode = source.get("mode", "table")
        if mode == "table":
            table = source.get("table")
            if not isinstance(table, str) or not table.strip():
                raise HTTPException(status_code=400, detail="source.table is required for table mode")
            _validate_table_name(conn, table)
            return f"SELECT * FROM [{table}]"
        if mode == "sql":
            sql = source.get("sql")
            if not isinstance(sql, str) or not sql.strip():
                raise HTTPException(status_code=400, detail="source.sql is required for sql mode")
            try:
                return _validate_read_only_sql(sql)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=400, detail=f"Unsupported source mode: {mode}")

    def _execute_with_timeout(
        conn: sqlite3.Connection,
        sql: str,
        params: list[Any] | tuple[Any, ...] | None = None,
        timeout_s: float = 2.0,
    ) -> sqlite3.Cursor:
        started = time.monotonic()

        def _progress_handler() -> int:
            return 1 if (time.monotonic() - started) > timeout_s else 0

        conn.set_progress_handler(_progress_handler, 10_000)
        try:
            if params is None:
                return conn.execute(sql)
            return conn.execute(sql, params)
        finally:
            conn.set_progress_handler(None, 0)

    @api.get("/health")
    def health():
        info = {"status": "ok", "db_exists": db_path.exists()}
        if db_path.exists():
            info["db_size_mb"] = round(db_path.stat().st_size / 1024 / 1024, 1)
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
            sql = _validate_read_only_sql(sql_in)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
            limit = min(max(1, limit_in), _MAX_DEFAULT_LIMIT)
            sql = f"{sql} LIMIT {limit}"

        conn = _connect()
        try:
            cursor = _execute_with_timeout(conn, sql, timeout_s=2.5)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return {
                "columns": columns,
                "rows": _rows_to_dicts(rows),
                "row_count": len(rows),
                "sql": sql,
            }
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            conn.close()

    @api.get("/schema")
    def schema(table: str = Query(default="")):
        conn = _connect()
        try:
            if not table:
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
                return {"tables": [r["name"] for r in rows]}

            _validate_table_name(conn, table)
            cols = conn.execute(f"PRAGMA table_info([{table}])").fetchall()
            return {
                "table": table,
                "columns": [
                    {
                        "name": c["name"],
                        "type": c["type"],
                        "notnull": bool(c["notnull"]),
                        "pk": bool(c["pk"]),
                        "default": c["dflt_value"],
                    }
                    for c in cols
                ],
            }
        except sqlite3.DatabaseError as e:
            raise HTTPException(status_code=503, detail=f"Database error: {e}")
        finally:
            conn.close()

    @api.get("/tables")
    def tables():
        conn = _connect()
        try:
            names = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            result = []
            for r in names:
                name = r["name"]
                count = conn.execute(f"SELECT COUNT(*) AS n FROM [{name}]").fetchone()
                result.append({"name": name, "count": count["n"]})
            return {"tables": result}
        except sqlite3.DatabaseError as e:
            raise HTTPException(status_code=503, detail=f"Database error: {e}")
        finally:
            conn.close()

    @api.get("/stats")
    def stats():
        if not stats_path.exists():
            raise HTTPException(status_code=404, detail="No dashboard_stats.json on volume")
        return json.loads(stats_path.read_text())

    @api.get("/sample/{table}")
    def sample(table: str, n: int = Query(default=10, le=500)):
        conn = _connect()
        try:
            _validate_table_name(conn, table)
            rows = conn.execute(
                f"SELECT * FROM [{table}] ORDER BY RANDOM() LIMIT ?", [n]
            ).fetchall()
            columns = list(rows[0].keys()) if rows else []
            return {
                "table": table,
                "columns": columns,
                "rows": _rows_to_dicts(rows),
                "row_count": len(rows),
            }
        except sqlite3.DatabaseError as e:
            raise HTTPException(status_code=503, detail=f"Database error: {e}")
        finally:
            conn.close()

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
        schema = pq.read_schema(p)
        return {
            "name": name,
            "num_rows": meta.num_rows,
            "num_columns": meta.num_columns,
            "num_row_groups": meta.num_row_groups,
            "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
            "schema": [
                {"name": schema.field(i).name, "type": str(schema.field(i).type)}
                for i in range(len(schema))
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
        conn = _connect()
        try:
            source_data = _coerce_dict(req.get("source"), "source")
            source = _source_sql(conn, source_data)
            limit = max(1, min(_coerce_int(req.get("limit"), 1000), _MAX_PROFILE_LIMIT))
            sql = f"SELECT * FROM ({source}) AS src LIMIT ?"
            cursor = _execute_with_timeout(conn, sql, [limit], timeout_s=2.5)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = _rows_to_dicts(cursor.fetchall())

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
        except (sqlite3.OperationalError, sqlite3.DatabaseError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            conn.close()

    @api.post("/label/preview")
    def label_preview(req: dict[str, Any]):
        if not isinstance(req, dict):
            raise HTTPException(status_code=400, detail="Request body must be an object")
        conn = _connect()
        try:
            spec = _coerce_dict(req.get("spec"), "spec")
            source = _source_sql(conn, _coerce_dict(spec.get("source"), "spec.source"))
            label_expr = _build_label_expression(_coerce_dict(spec.get("label"), "spec.label"))

            where_clause = ""
            filters = spec.get("filters") or {}
            if filters.get("sql_where"):
                where_sql = _validate_sql_fragment(str(filters["sql_where"]), "filters.sql_where")
                where_clause = f" WHERE ({where_sql})"

            limit = max(1, min(_coerce_int(req.get("limit"), 2000), _MAX_LABEL_PREVIEW_LIMIT))
            query = (
                "SELECT src.*, "
                f"{label_expr} AS __label_value "
                f"FROM ({source}) AS src{where_clause} "
                "LIMIT ?"
            )
            cursor = _execute_with_timeout(conn, query, [limit], timeout_s=2.5)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = _rows_to_dicts(cursor.fetchall())

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
                "generated_sql": query,
            }
        except (sqlite3.OperationalError, sqlite3.DatabaseError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            conn.close()

    @api.get("/prep-targets")
    def prep_targets_list():
        specs = _load_prep_targets(prep_targets_path)
        specs.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return {"specs": specs}

    @api.get("/prep-targets/{spec_id}")
    def prep_targets_get(spec_id: str):
        specs = _load_prep_targets(prep_targets_path)
        for spec in specs:
            if spec.get("id") == spec_id:
                return {"spec": spec}
        raise HTTPException(status_code=404, detail=f"Spec not found: {spec_id}")

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

        specs = _load_prep_targets(prep_targets_path)
        now = _now_iso()
        updated_list, prepared = _upsert_prep_target(specs, spec, now)
        _save_prep_targets_atomic(prep_targets_path, updated_list)
        if hasattr(volume, "commit"):
            volume.commit()
        return {"spec": prepared, "count": len(updated_list)}

    @api.delete("/prep-targets/{spec_id}")
    def prep_targets_delete(spec_id: str):
        specs = _load_prep_targets(prep_targets_path)
        kept = [s for s in specs if s.get("id") != spec_id]
        if len(kept) == len(specs):
            raise HTTPException(status_code=404, detail=f"Spec not found: {spec_id}")
        _save_prep_targets_atomic(prep_targets_path, kept)
        if hasattr(volume, "commit"):
            volume.commit()
        return {"status": "deleted", "id": spec_id, "count": len(kept)}

    @api.post("/reload")
    def reload():
        volume.reload()
        return {"status": "reloaded"}

    return api
