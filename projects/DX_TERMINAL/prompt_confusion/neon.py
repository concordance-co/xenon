from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psycopg
from psycopg.rows import dict_row


DEFAULT_DB_ENV_VAR = "XENON_NEON_DATABASE_URL"


def _env_file_value(env_var: str) -> str | None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return None
    prefix = f"{env_var}="
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip().strip('"').strip("'")
    return None


def validate_table_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe table name: {value}")
    return value


@contextmanager
def connect_neon(*, autocommit: bool = False, env_var: str = DEFAULT_DB_ENV_VAR) -> Iterator[psycopg.Connection]:
    dsn = os.environ.get(env_var) or _env_file_value(env_var)
    if not dsn:
        raise RuntimeError(f"{env_var} is not set")
    conn = psycopg.connect(dsn, autocommit=autocommit, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


def ensure_schema(conn: psycopg.Connection, schema: str = "public") -> None:
    if not schema or schema == "public":
        return
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
