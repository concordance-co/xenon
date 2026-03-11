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
    POST /reload                    Force volume refresh
"""

import modal

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

_READ_ONLY_PREFIXES = ("select", "pragma", "explain", "with")
_MAX_DEFAULT_LIMIT = 1000


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=120,
    cpu=1,
    scaledown_window=3600,
)
@modal.asgi_app()
def web_app():
    import json
    import re
    import sqlite3
    from pathlib import Path

    import pyarrow.parquet as pq
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    api = FastAPI(title="Xenon Backend", version="0.1.0")
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

    def _connect() -> sqlite3.Connection:
        if not db_path.exists():
            raise HTTPException(status_code=503, detail="Database not found on volume")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _validate_read_only(sql: str) -> None:
        stripped = sql.strip().lower()
        if not stripped.startswith(_READ_ONLY_PREFIXES):
            raise HTTPException(
                status_code=400,
                detail=f"Only read-only queries allowed (SELECT/PRAGMA/EXPLAIN/WITH). Got: {stripped[:30]}...",
            )

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

    def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
        return [dict(r) for r in rows]

    # --- Health ---

    @api.get("/health")
    def health():
        info = {"status": "ok", "db_exists": db_path.exists()}
        if db_path.exists():
            info["db_size_mb"] = round(db_path.stat().st_size / 1024 / 1024, 1)
        return info

    # --- SQL Query ---

    class QueryRequest(BaseModel):
        sql: str
        limit: int = 100

    @api.post("/query")
    def query(req: QueryRequest):
        _validate_read_only(req.sql)

        sql = req.sql.strip().rstrip(";")
        # Inject LIMIT if not present
        if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
            limit = min(req.limit, _MAX_DEFAULT_LIMIT)
            sql = f"{sql} LIMIT {limit}"

        conn = _connect()
        try:
            cursor = conn.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return {
                "columns": columns,
                "rows": [dict(r) for r in rows],
                "row_count": len(rows),
                "sql": sql,
            }
        except sqlite3.OperationalError as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            conn.close()

    # --- Schema ---

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
                    {"name": c["name"], "type": c["type"], "notnull": bool(c["notnull"]),
                     "pk": bool(c["pk"]), "default": c["dflt_value"]}
                    for c in cols
                ],
            }
        finally:
            conn.close()

    # --- Tables ---

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
        finally:
            conn.close()

    # --- Stats ---

    @api.get("/stats")
    def stats():
        if not stats_path.exists():
            raise HTTPException(status_code=404, detail="No dashboard_stats.json on volume")
        return json.loads(stats_path.read_text())

    # --- Sample ---

    @api.get("/sample/{table}")
    def sample(table: str, n: int = Query(default=10, le=500)):
        conn = _connect()
        try:
            _validate_table_name(conn, table)
            rows = conn.execute(
                f"SELECT * FROM [{table}] ORDER BY RANDOM() LIMIT ?", [n]
            ).fetchall()
            columns = [d[0] for d in rows[0].keys()] if rows else []
            return {"table": table, "columns": columns, "rows": _rows_to_dicts(rows), "row_count": len(rows)}
        finally:
            conn.close()

    # --- Parquet List ---

    @api.get("/parquet/list")
    def parquet_list():
        files = []
        for d in [exports_dir, activations_dir]:
            if d.exists():
                for f in sorted(d.glob("*.parquet")):
                    size = f.stat().st_size
                    files.append({
                        "name": f.name,
                        "dir": str(d),
                        "size_mb": round(size / 1024 / 1024, 2),
                    })
        return {"files": files}

    # --- Parquet Info ---

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

    # --- Parquet Sample ---

    @api.get("/parquet/sample/{name}")
    def parquet_sample(name: str, n: int = Query(default=10, le=100),
                       columns: str = Query(default="")):
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

    # --- Activations Metadata ---

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

    # --- Reload ---

    @api.post("/reload")
    def reload():
        volume.reload()
        return {"status": "reloaded"}

    return api
