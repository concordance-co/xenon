from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from psycopg import sql


_RELATION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SQL_FRAGMENT_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|vacuum|reindex|truncate|grant|revoke)\b",
    re.IGNORECASE,
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def validate_relation_name(name: str | None) -> str | None:
    if name is None:
        return None
    text = str(name).strip()
    if not text:
        return None
    if not _RELATION_NAME_RE.fullmatch(text):
        raise ValueError(f"Invalid relation name: {name!r}")
    return text


def validate_identifier(text: str, *, label: str) -> str:
    cleaned = validate_relation_name(text)
    if cleaned is None:
        raise ValueError(f"{label} is required")
    return cleaned


def _normalize_sql(sql_str: str) -> str:
    return sql_str.strip().rstrip(";")


def validate_read_only_sql(sql_str: str) -> str:
    normalized = _normalize_sql(sql_str)
    lowered = normalized.lower()
    if not lowered.startswith(("select", "with")):
        raise ValueError("Only read-only SELECT/WITH SQL is allowed")
    if _SQL_FRAGMENT_FORBIDDEN.search(lowered):
        raise ValueError("Read-only SQL cannot contain mutating keywords")
    return normalized


def validate_sql_fragment(fragment: str, *, label: str) -> str:
    cleaned = fragment.strip()
    if not cleaned:
        raise ValueError(f"{label} cannot be empty")
    if ";" in cleaned:
        raise ValueError(f"{label} cannot contain ';'")
    if _SQL_FRAGMENT_FORBIDDEN.search(cleaned):
        raise ValueError(f"{label} contains forbidden SQL keywords")
    return cleaned


def _slugify(text: str) -> str:
    lowered = re.sub(r"[^a-zA-Z0-9_]+", "_", str(text).strip().lower())
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "workflow"


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _build_row_key_sql(dataset: dict[str, Any]) -> str:
    identity = dict(dataset.get("identity") or {})
    identity_column = identity.get("column")
    if isinstance(identity_column, str) and identity_column.strip():
        column = validate_identifier(identity_column.strip(), label="dataset.identity.column")
    else:
        column = "log_id"
    return f"src.{column}::text AS workflow_row_key"


def _build_prompt_hash_sql() -> str:
    return "md5(COALESCE(to_jsonb(src)->>'prompt_messages_json', '')) AS workflow_prompt_hash"


def normalize_workflow_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("Workflow spec must be an object")

    normalized = dict(spec)
    if "dataset" not in normalized:
        if not isinstance(normalized.get("source"), dict):
            raise ValueError("Legacy prep-target spec must include source")
        if not isinstance(normalized.get("label"), dict):
            raise ValueError("Legacy prep-target spec must include label")
        if not isinstance(normalized.get("split"), dict):
            raise ValueError("Legacy prep-target spec must include split")
        dataset = {
            "source": dict(normalized["source"]),
            "filters": dict(normalized.get("filters") or {}),
            "label": dict(normalized["label"]),
            "split": dict(normalized["split"]),
            "probe_defaults": dict(normalized.get("probe_defaults") or {}),
            "identity": dict(normalized.get("identity") or {}),
            "publish_target": normalized.get("publish_target"),
            "publish_mode": normalized.get("publish_mode") or "view",
        }
        normalized = {
            "id": normalized.get("id"),
            "name": normalized.get("name"),
            "description": normalized.get("description"),
            "version": normalized.get("version") or 1,
            "dataset": dataset,
            "capture": dict(normalized.get("capture") or {}),
            "analysis": dict(normalized.get("analysis") or {}),
            "report": dict(normalized.get("report") or {}),
            "legacy_prep_target": True,
        }

    if not isinstance(normalized.get("name"), str) or not str(normalized.get("name")).strip():
        raise ValueError("spec.name is required")

    dataset = normalized.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("spec.dataset is required")
    if not isinstance(dataset.get("source"), dict):
        raise ValueError("spec.dataset.source is required")
    if not isinstance(dataset.get("label"), dict):
        raise ValueError("spec.dataset.label is required")

    source = dict(dataset["source"])
    mode = str(source.get("mode") or "table")
    if mode not in {"table", "sql"}:
        raise ValueError(f"Unsupported dataset source mode: {mode}")
    if mode == "table":
        source["table"] = validate_identifier(str(source.get("table") or ""), label="dataset.source.table")
    else:
        sql_text = source.get("sql")
        if not isinstance(sql_text, str):
            raise ValueError("dataset.source.sql is required for sql mode")
        source["sql"] = validate_read_only_sql(sql_text)

    label = dict(dataset["label"])
    label_mode = str(label.get("mode") or "")
    if label_mode not in {"direct", "binary_rule", "bucket"}:
        raise ValueError(f"Unsupported label mode: {label_mode}")
    label["expression_sql"] = validate_sql_fragment(str(label.get("expression_sql") or ""), label="label.expression_sql")

    if label_mode == "binary_rule":
        classes = label.get("classes") or ["negative", "positive"]
        if not isinstance(classes, list) or len(classes) != 2:
            raise ValueError("binary_rule labels require exactly 2 classes")
        label["classes"] = [str(classes[0]), str(classes[1])]
    if label_mode == "bucket":
        buckets = label.get("buckets")
        if not isinstance(buckets, list) or not buckets:
            raise ValueError("bucket labels require non-empty buckets")
        cleaned_buckets: list[dict[str, Any]] = []
        for bucket in buckets:
            if not isinstance(bucket, dict) or not str(bucket.get("name") or "").strip():
                raise ValueError("bucket labels require named buckets")
            cleaned_buckets.append(
                {
                    "name": str(bucket["name"]).strip(),
                    "min": bucket.get("min"),
                    "max": bucket.get("max"),
                }
            )
        label["buckets"] = cleaned_buckets

    normalized["dataset"] = dict(dataset)
    normalized["dataset"]["source"] = source
    normalized["dataset"]["label"] = label
    normalized["dataset"]["filters"] = dict(dataset.get("filters") or {})
    normalized["dataset"]["split"] = dict(dataset.get("split") or {})
    identity = dict(dataset.get("identity") or {})
    identity_column = identity.get("column")
    if identity_column is not None:
        identity["column"] = validate_identifier(str(identity_column).strip(), label="dataset.identity.column")
    normalized["dataset"]["identity"] = identity
    normalized["dataset"]["publish_mode"] = "materialized" if str(dataset.get("publish_mode") or "").lower() == "materialized" else "view"
    normalized["version"] = int(normalized.get("version") or 1)
    normalized["id"] = str(normalized.get("id") or uuid.uuid4().hex[:12])
    normalized["capture"] = dict(normalized.get("capture") or {})
    normalized["analysis"] = dict(normalized.get("analysis") or {})
    normalized["report"] = dict(normalized.get("report") or {})
    return normalized


def serialize_legacy_prep_target(spec: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_workflow_spec(spec)
    dataset = normalized["dataset"]
    legacy = {
        "id": normalized["id"],
        "name": normalized["name"],
        "description": normalized.get("description"),
        "source": dataset["source"],
        "filters": dataset.get("filters") or {},
        "label": dataset["label"],
        "split": dataset.get("split") or {},
        "probe_defaults": dataset.get("probe_defaults") or {},
        "identity": dataset.get("identity") or {},
    }
    if normalized.get("created_at"):
        legacy["created_at"] = normalized["created_at"]
    if normalized.get("updated_at"):
        legacy["updated_at"] = normalized["updated_at"]
    return legacy


def list_workflow_specs(conn: Any) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, name, description, version, spec_json, created_at, updated_at "
        "FROM workflow_specs ORDER BY updated_at DESC"
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        spec = dict(row.get("spec_json") or {})
        spec["id"] = row["id"]
        spec["name"] = row["name"]
        spec["description"] = row.get("description")
        spec["version"] = row["version"]
        spec["created_at"] = row["created_at"]
        spec["updated_at"] = row["updated_at"]
        out.append(spec)
    return out


def get_workflow_spec(conn: Any, spec_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT id, name, description, version, spec_json, created_at, updated_at "
        "FROM workflow_specs WHERE id = %s",
        [spec_id],
    ).fetchone()
    if not row:
        return None
    spec = dict(row.get("spec_json") or {})
    spec["id"] = row["id"]
    spec["name"] = row["name"]
    spec["description"] = row.get("description")
    spec["version"] = row["version"]
    spec["created_at"] = row["created_at"]
    spec["updated_at"] = row["updated_at"]
    return spec


def upsert_workflow_spec(conn: Any, spec: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_workflow_spec(spec)
    now = _now_iso()
    existing = conn.execute(
        "SELECT created_at FROM workflow_specs WHERE id = %s",
        [normalized["id"]],
    ).fetchone()
    created_at = existing["created_at"] if existing else (normalized.get("created_at") or now)

    spec_json = {
        k: v
        for k, v in normalized.items()
        if k not in {"id", "name", "description", "created_at", "updated_at"}
    }
    conn.execute(
        "INSERT INTO workflow_specs (id, name, description, version, spec_json, created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s) "
        "ON CONFLICT (id) DO UPDATE SET "
        "name = EXCLUDED.name, description = EXCLUDED.description, version = EXCLUDED.version, "
        "spec_json = EXCLUDED.spec_json, updated_at = EXCLUDED.updated_at",
        [
            normalized["id"],
            normalized["name"],
            normalized.get("description"),
            int(normalized["version"]),
            json.dumps(spec_json),
            created_at,
            now,
        ],
    )
    saved = dict(normalized)
    saved["created_at"] = created_at
    saved["updated_at"] = now
    return saved


def delete_workflow_spec(conn: Any, spec_id: str) -> bool:
    row = conn.execute(
        "DELETE FROM workflow_specs WHERE id = %s RETURNING id",
        [spec_id],
    ).fetchone()
    return bool(row)


def start_workflow_run(
    conn: Any,
    *,
    spec: dict[str, Any],
    run_type: str,
    source: str,
    resolved_config: dict[str, Any],
) -> dict[str, Any]:
    now = _now_iso()
    run_id = uuid.uuid4().hex[:12]
    conn.execute(
        "INSERT INTO workflow_runs "
        "(id, spec_id, spec_version, run_type, status, source, spec_snapshot_json, config_json, result_json, error_text, created_at, updated_at, completed_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s)",
        [
            run_id,
            spec["id"],
            int(spec.get("version") or 1),
            run_type,
            "running",
            source,
            json.dumps(spec),
            json.dumps(resolved_config),
            json.dumps({}),
            None,
            now,
            now,
            None,
        ],
    )
    return {
        "id": run_id,
        "spec_id": spec["id"],
        "spec_version": int(spec.get("version") or 1),
        "run_type": run_type,
        "status": "running",
        "source": source,
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
    }


def finish_workflow_run(
    conn: Any,
    *,
    run_id: str,
    status: str,
    result: dict[str, Any] | None = None,
    error_text: str | None = None,
) -> None:
    now = _now_iso()
    conn.execute(
        "UPDATE workflow_runs SET status = %s, result_json = %s::jsonb, error_text = %s, "
        "updated_at = %s, completed_at = %s WHERE id = %s",
        [status, json.dumps(result or {}), error_text, now, now, run_id],
    )


def list_workflow_runs(conn: Any) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, spec_id, spec_version, run_type, status, source, spec_snapshot_json, config_json, result_json, error_text, created_at, updated_at, completed_at "
        "FROM workflow_runs ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_workflow_run(conn: Any, run_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT id, spec_id, spec_version, run_type, status, source, spec_snapshot_json, config_json, result_json, error_text, created_at, updated_at, completed_at "
        "FROM workflow_runs WHERE id = %s",
        [run_id],
    ).fetchone()
    return dict(row) if row else None


def build_publication_name(spec: dict[str, Any]) -> str:
    dataset = dict(spec.get("dataset") or {})
    publish_target = dataset.get("publish_target")
    if isinstance(publish_target, str) and publish_target.strip():
        return validate_identifier(publish_target.strip(), label="dataset.publish_target")
    return f"workflow_dataset_{_slugify(spec['id'])}_v{int(spec.get('version') or 1)}"


def _build_source_sql(spec: dict[str, Any]) -> str:
    dataset = dict(spec["dataset"])
    source = dict(dataset["source"])
    mode = source["mode"]
    if mode == "table":
        table = validate_identifier(source["table"], label="dataset.source.table")
        base_sql = f"SELECT * FROM {table}"
    else:
        base_sql = validate_read_only_sql(str(source["sql"]))

    filters = dict(dataset.get("filters") or {})
    sql_where = filters.get("sql_where")
    if isinstance(sql_where, str) and sql_where.strip():
        where_sql = validate_sql_fragment(sql_where, label="dataset.filters.sql_where")
        return f"SELECT * FROM ({base_sql}) src WHERE {where_sql}"
    return base_sql


def _build_label_sql(label: dict[str, Any]) -> str:
    expr = validate_sql_fragment(str(label.get("expression_sql") or ""), label="dataset.label.expression_sql")
    mode = label["mode"]
    if mode == "direct":
        return f"({expr}) AS workflow_label"
    if mode == "binary_rule":
        false_label, true_label = label.get("classes") or ["negative", "positive"]
        return (
            "CASE WHEN ({expr}) THEN {true_value} ELSE {false_value} END AS workflow_label".format(
                expr=expr,
                true_value=_sql_literal(str(true_label)),
                false_value=_sql_literal(str(false_label)),
            )
        )
    if mode == "bucket":
        clauses: list[str] = []
        for bucket in label.get("buckets") or []:
            parts: list[str] = []
            if bucket.get("min") is not None:
                parts.append(f"({expr}) >= {float(bucket['min'])}")
            if bucket.get("max") is not None:
                parts.append(f"({expr}) < {float(bucket['max'])}")
            if not parts:
                parts.append("TRUE")
            bucket_name = str(bucket["name"]).replace("'", "''")
            clauses.append(f"WHEN {' AND '.join(parts)} THEN '{bucket_name}'")
        return "CASE " + " ".join(clauses) + " ELSE NULL END AS workflow_label"
    raise ValueError(f"Unsupported label mode: {mode}")


def publish_dataset(conn: Any, spec: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    normalized = normalize_workflow_spec(spec)
    relation_name = build_publication_name(normalized)
    publish_mode = normalized["dataset"].get("publish_mode") or "view"
    source_sql = _build_source_sql(normalized)
    label_sql = _build_label_sql(normalized["dataset"]["label"])
    row_key_sql = _build_row_key_sql(normalized["dataset"])
    prompt_hash_sql = _build_prompt_hash_sql()
    spec_id = normalized["id"].replace("'", "''")
    version = int(normalized.get("version") or 1)

    select_sql = (
        f"SELECT src.*, {label_sql}, {row_key_sql}, {prompt_hash_sql}, "
        f"'{spec_id}' AS workflow_spec_id, {version} AS workflow_spec_version "
        f"FROM ({source_sql}) src"
    )

    if publish_mode == "materialized":
        conn.execute(sql.SQL("DROP MATERIALIZED VIEW IF EXISTS {}").format(sql.Identifier(relation_name)))
        conn.execute(sql.SQL("CREATE MATERIALIZED VIEW {} AS ").format(sql.Identifier(relation_name)).as_string(conn) + select_sql)
    else:
        conn.execute(sql.SQL("DROP VIEW IF EXISTS {}").format(sql.Identifier(relation_name)))
        conn.execute(sql.SQL("CREATE VIEW {} AS ").format(sql.Identifier(relation_name)).as_string(conn) + select_sql)

    row = conn.execute(
        sql.SQL("SELECT COUNT(*) AS n FROM {}").format(sql.Identifier(relation_name)).as_string(conn)
    ).fetchone()
    row_count = int(row["n"]) if row else 0

    now = _now_iso()
    publication_id = uuid.uuid4().hex[:12]
    publication_json = {
        "source_sql": source_sql,
        "label_mode": normalized["dataset"]["label"]["mode"],
        "label_expression_sql": normalized["dataset"]["label"]["expression_sql"],
    }
    conn.execute(
        "INSERT INTO dataset_publications "
        "(id, spec_id, spec_version, run_id, relation_name, publish_mode, row_count, publication_json, created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s) "
        "ON CONFLICT (relation_name) DO UPDATE SET "
        "id = EXCLUDED.id, spec_id = EXCLUDED.spec_id, spec_version = EXCLUDED.spec_version, "
        "run_id = EXCLUDED.run_id, publish_mode = EXCLUDED.publish_mode, row_count = EXCLUDED.row_count, "
        "publication_json = EXCLUDED.publication_json, created_at = EXCLUDED.created_at, updated_at = EXCLUDED.updated_at",
        [
            publication_id,
            normalized["id"],
            version,
            run_id,
            relation_name,
            publish_mode,
            row_count,
            json.dumps(publication_json),
            now,
            now,
        ],
    )
    return {
        "id": publication_id,
        "spec_id": normalized["id"],
        "spec_version": version,
        "run_id": run_id,
        "relation_name": relation_name,
        "publish_mode": publish_mode,
        "row_count": row_count,
        "created_at": now,
        "updated_at": now,
    }


def get_latest_publication_for_spec(conn: Any, spec_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT id, spec_id, spec_version, run_id, relation_name, publish_mode, row_count, publication_json, created_at, updated_at "
        "FROM dataset_publications WHERE spec_id = %s ORDER BY created_at DESC LIMIT 1",
        [spec_id],
    ).fetchone()
    return dict(row) if row else None


def list_publications(conn: Any, *, spec_id: str | None = None) -> list[dict[str, Any]]:
    if spec_id:
        rows = conn.execute(
            "SELECT id, spec_id, spec_version, run_id, relation_name, publish_mode, row_count, publication_json, created_at, updated_at "
            "FROM dataset_publications WHERE spec_id = %s ORDER BY created_at DESC",
            [spec_id],
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, spec_id, spec_version, run_id, relation_name, publish_mode, row_count, publication_json, created_at, updated_at "
            "FROM dataset_publications ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def relation_has_column(conn: Any, relation_name: str, column_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s AND column_name = %s",
        [relation_name, column_name],
    ).fetchone()
    return bool(row)


def export_publication_labels(
    conn: Any,
    *,
    relation_name: str,
    output_path: Path,
    label_column: str = "workflow_label",
    group_column: str | None = None,
) -> Path:
    relation = validate_identifier(relation_name, label="publication relation_name")
    label_col = validate_identifier(label_column, label="label column")
    select_bits = ["log_id", f"{label_col} AS workflow_label"]
    group_col = validate_identifier(group_column, label="group column") if group_column else None
    if group_col and relation_has_column(conn, relation, group_col):
        select_bits.append(group_col)
    if relation_has_column(conn, relation, "workflow_row_key"):
        select_bits.append("workflow_row_key")
    else:
        select_bits.append("log_id::text AS workflow_row_key")
    if relation_has_column(conn, relation, "workflow_prompt_hash"):
        select_bits.append("workflow_prompt_hash")
    elif relation_has_column(conn, relation, "prompt_messages_json"):
        select_bits.append("md5(COALESCE(prompt_messages_json, '')) AS workflow_prompt_hash")
    if relation_has_column(conn, relation, "label_quality"):
        select_bits.append("label_quality")
    else:
        select_bits.append("'high'::text AS label_quality")
    if relation_has_column(conn, relation, "decision_type"):
        select_bits.append("decision_type")
    if relation_has_column(conn, relation, "trade_side"):
        select_bits.append("trade_side")
    if relation_has_column(conn, relation, "asset"):
        select_bits.append("asset")
    if relation_has_column(conn, relation, "was_profitable_1h"):
        select_bits.append("was_profitable_1h")
    if relation_has_column(conn, relation, "vault_risk_preference"):
        select_bits.append("vault_risk_preference")
    if relation_has_column(conn, relation, "strategy_family"):
        select_bits.append("strategy_family")

    query = "SELECT " + ", ".join(select_bits) + " FROM " + relation
    rows = conn.execute(query).fetchall()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([dict(r) for r in rows]), output_path)
    return output_path
