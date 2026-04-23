from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row


DEFAULT_DB_ENV_VAR = "XENON_NEON_DATABASE_URL"


@contextmanager
def connect_neon(*, autocommit: bool = False, env_var: str = DEFAULT_DB_ENV_VAR) -> Iterator[psycopg.Connection]:
    dsn = os.environ.get(env_var)
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
