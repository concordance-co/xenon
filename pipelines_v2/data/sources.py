"""Source adapters for reading examples, labels, cases, and metadata."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

from pipelines_v2.core.types import RuntimeSecret
from pipelines_v2.data.datasets import Dataset

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Source(Protocol):
    """Connection and fetch interface for dataset sources."""

    kind: str
    defer_to_runtime: bool

    def identity(self) -> dict[str, Any]:
        ...

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        ...

    def fetch_dataset(self, **kwargs: Any) -> Dataset:
        ...


@dataclass(frozen=True, slots=True)
class InMemorySource:
    """Source backed by in-process Python records."""
    records: tuple[Mapping[str, Any], ...]
    kind: str = "memory"
    defer_to_runtime: bool = False

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> "InMemorySource":
        """Normalize arbitrary record iterables into an in-memory source."""
        return cls(records=tuple(dict(record) for record in records))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InMemorySource":
        return cls.from_records(payload.get("records", ()))

    def identity(self) -> dict[str, Any]:
        """Serialize the source so it can be embedded into a dataset ref."""
        return {
            "kind": self.kind,
            "records": [dict(record) for record in self.records],
        }

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return ()

    def fetch_dataset(self, **kwargs: Any) -> Dataset:
        """Materialize a dataset from the stored records."""
        return Dataset.from_records(self.records, **kwargs)


@dataclass(frozen=True, slots=True)
class PostgresSource:
    """Postgres-backed source for examples, labels, cases, and metadata."""

    url: str | None = None
    url_env_var: str | None = None
    kind: str = "postgres"
    defer_to_runtime: bool = True

    def __post_init__(self) -> None:
        has_url = bool(self.url)
        has_env = bool(self.url_env_var)
        if has_url == has_env:
            raise ValueError("PostgresSource requires exactly one of url or url_env_var")

    @classmethod
    def from_env(cls, env_var: str) -> "PostgresSource":
        """Create a runtime-resolved Postgres source from an env var name."""
        return cls(url_env_var=env_var)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PostgresSource":
        if payload.get("url_env_var") is not None:
            return cls(url_env_var=str(payload["url_env_var"]))
        if payload.get("url") is not None:
            return cls(url=str(payload["url"]))
        raise ValueError("PostgresSource payload requires url_env_var or url")

    def identity(self) -> dict[str, Any]:
        """Serialize the source for deferred runtime execution."""
        if self.url_env_var is None:
            raise ValueError(
                "Deferred PostgresSource specs must use an environment variable reference, not a raw URL"
            )
        return {
            "kind": self.kind,
            "url_env_var": self.url_env_var,
        }

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        """Declare the env var needed to connect at runtime."""
        if self.url_env_var is None:
            return ()
        return (RuntimeSecret(env_var=self.url_env_var),)

    def fetch_dataset(
        self,
        *,
        table: str | None = None,
        sql: str | None = None,
        prompt_column: str,
        example_key_column: str,
        prompt_hash_column: str | None = None,
        label_columns: Sequence[str] = (),
        case_columns: Sequence[str] = (),
        case_key_column: str | None = None,
        metadata_columns: Sequence[str] = (),
        limit: int | None = None,
        name: str | None = None,
        id: str | None = None,
    ) -> Dataset:
        """Fetch a dataset from either a relation or a SQL query.

        Exactly one of ``table`` or ``sql`` must be provided. ``sql`` is wrapped
        as a subquery so the dataset column projection and optional outer limit
        remain under library control.
        """
        import psycopg
        from psycopg.rows import dict_row

        if bool(table) == bool(sql):
            raise ValueError("PostgresSource.fetch_dataset requires exactly one of table or sql")

        columns = [
            example_key_column,
            prompt_column,
            *([prompt_hash_column] if prompt_hash_column else []),
            *label_columns,
            *case_columns,
            *([case_key_column] if case_key_column else []),
            *metadata_columns,
        ]
        unique_columns = list(dict.fromkeys(columns))
        select_sql = ", ".join(_quote_identifier(column) for column in unique_columns)
        if table is not None:
            query = f"SELECT {select_sql} FROM {_quote_relation(table)}"
        else:
            assert sql is not None
            base_sql = _normalize_sql_query(sql)
            query = f"SELECT {select_sql} FROM ({base_sql}) AS src"
        params: list[Any] = []
        if limit is not None:
            query += " LIMIT %s"
            params.append(limit)

        with psycopg.connect(self._resolved_url(), row_factory=dict_row) as conn:
            records = conn.execute(query, params).fetchall()

        return Dataset.from_records(
            records,
            prompt_column=prompt_column,
            example_key_column=example_key_column,
            prompt_hash_column=prompt_hash_column,
            label_columns=label_columns,
            case_columns=case_columns,
            case_key_column=case_key_column,
            metadata_columns=metadata_columns,
            id=id,
            name=name or (table if table is not None else "postgres_query"),
        )

    def connection_url(self) -> str:
        """Resolve the concrete connection string for local runtime use."""
        return self._resolved_url()

    def _resolved_url(self) -> str:
        if self.url is not None:
            return self.url
        if self.url_env_var is None:
            raise RuntimeError("PostgresSource has no connection reference")
        value = os.environ.get(self.url_env_var)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {self.url_env_var}")
        return value


def _quote_identifier(value: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid SQL identifier: {value!r}")
    return f'"{value}"'


def _quote_relation(value: str) -> str:
    parts = value.split(".")
    if not parts:
        raise ValueError("Relation name cannot be empty")
    return ".".join(_quote_identifier(part) for part in parts)


def _normalize_sql_query(value: str) -> str:
    normalized = str(value).strip()
    if normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    if not normalized:
        raise ValueError("SQL query cannot be empty")
    return normalized


def source_from_dict(payload: Mapping[str, Any]) -> Source:
    kind = str(payload["kind"])
    if kind == "memory":
        return InMemorySource.from_dict(payload)
    if kind == "postgres":
        return PostgresSource.from_dict(payload)
    raise ValueError(f"Unsupported source kind: {kind}")
