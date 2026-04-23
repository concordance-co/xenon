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

    def runtime_pip_packages(self) -> tuple[str, ...]:
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

    def runtime_pip_packages(self) -> tuple[str, ...]:
        return ()

    def fetch_dataset(self, **kwargs: Any) -> Dataset:
        """Materialize a dataset from the stored records."""
        return Dataset.from_records(self.records, **kwargs)


@dataclass(frozen=True, slots=True)
class ArtifactDatasetSource:
    """Deferred source that materializes a Dataset stored in an artifact result."""

    kind: str = "artifact_dataset"
    defer_to_runtime: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactDatasetSource":
        del payload
        return cls()

    def identity(self) -> dict[str, Any]:
        return {"kind": self.kind}

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return ()

    def runtime_pip_packages(self) -> tuple[str, ...]:
        return ()

    def fetch_dataset(
        self,
        *,
        artifact: Any,
        result_key: str = "dataset",
        provides_token_sections: bool = False,
        id: str | None = None,
        name: str | None = None,
    ) -> Dataset:
        """Load a serialized ``Dataset`` payload from an operation artifact result."""
        artifact_obj = _artifact_from_value(artifact)
        if not hasattr(artifact_obj, "result"):
            raise TypeError("ArtifactDatasetSource.fetch_dataset requires an artifact with result()")
        result = artifact_obj.result()
        if not isinstance(result, Mapping):
            raise TypeError("ArtifactDatasetSource artifact result must be a mapping")
        dataset_payload = _path_get(result, result_key)
        if not isinstance(dataset_payload, Mapping):
            raise TypeError(
                f"ArtifactDatasetSource result key {result_key!r} must resolve to a serialized Dataset mapping"
            )
        dataset = Dataset.from_dict(dataset_payload)
        resolved = dataset.resolve() if dataset.is_deferred else dataset
        if provides_token_sections and not all(_example_has_token_sections(example) for example in resolved.examples):
            raise ValueError(
                "ArtifactDatasetSource was declared with provides_token_sections=True, "
                "but not every materialized example has metadata['token_sections']."
            )
        if id is None and name is None:
            return resolved
        return Dataset.from_examples(
            resolved.examples,
            id=id or resolved.id,
            name=name or resolved.name,
        )


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

    def runtime_pip_packages(self) -> tuple[str, ...]:
        """Declare Python dependencies needed to fetch this source at runtime."""
        return ("psycopg[binary]",)

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


@dataclass(frozen=True, slots=True)
class HuggingFaceSource:
    """Deferred source that delegates loading to ``datasets.load_dataset``."""

    path: str
    name: str | None = None
    revision: str | None = None
    data_files: Any | None = None
    token_env_var: str | None = None
    trust_remote_code: bool = False
    kind: str = "huggingface"
    defer_to_runtime: bool = True

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ValueError("HuggingFaceSource path cannot be empty")
        object.__setattr__(self, "path", str(self.path))
        if self.name is not None:
            object.__setattr__(self, "name", str(self.name))
        if self.revision is not None:
            object.__setattr__(self, "revision", str(self.revision))
        if self.token_env_var is not None:
            object.__setattr__(self, "token_env_var", str(self.token_env_var))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HuggingFaceSource":
        return cls(
            path=str(payload["path"]),
            name=str(payload["name"]) if payload.get("name") is not None else None,
            revision=str(payload["revision"]) if payload.get("revision") is not None else None,
            data_files=payload.get("data_files"),
            token_env_var=str(payload["token_env_var"]) if payload.get("token_env_var") is not None else None,
            trust_remote_code=bool(payload.get("trust_remote_code", False)),
        )

    def identity(self) -> dict[str, Any]:
        """Serialize the source without exposing resolved token values."""
        payload: dict[str, Any] = {
            "kind": self.kind,
            "path": self.path,
        }
        if self.name is not None:
            payload["name"] = self.name
        if self.revision is not None:
            payload["revision"] = self.revision
        if self.data_files is not None:
            payload["data_files"] = self.data_files
        if self.token_env_var is not None:
            payload["token_env_var"] = self.token_env_var
        if self.trust_remote_code:
            payload["trust_remote_code"] = True
        return payload

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        if self.token_env_var is None:
            return ()
        return (RuntimeSecret(env_var=self.token_env_var),)

    def runtime_pip_packages(self) -> tuple[str, ...]:
        return ("datasets",)

    def fetch_dataset(
        self,
        *,
        split: str | None = None,
        prompt_column: str,
        prompt_template: Any | None = None,
        example_key_column: str,
        prompt_hash_column: str | None = None,
        label_columns: Sequence[str] = (),
        case_columns: Sequence[str] = (),
        case_key_column: str | None = None,
        metadata_columns: Sequence[str] = (),
        index_column: str | None = None,
        index_prefix: str | None = None,
        hash_columns: Mapping[str, str | Sequence[str]] | None = None,
        nested_record_column: str | None = None,
        nested_record_index_column: str | None = None,
        nested_record_field_paths: Mapping[str, str | Sequence[str]] | None = None,
        limit: int | None = None,
        name: str | None = None,
        id: str | None = None,
    ) -> Dataset:
        """Load with Hugging Face ``datasets`` and map the result into v2 examples."""
        from datasets import load_dataset

        kwargs: dict[str, Any] = {}
        if split is not None:
            kwargs["split"] = split
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.data_files is not None:
            kwargs["data_files"] = self.data_files
        if self.trust_remote_code:
            kwargs["trust_remote_code"] = True
        if self.token_env_var is not None:
            token = os.environ.get(self.token_env_var)
            if not token:
                raise RuntimeError(f"Missing required environment variable: {self.token_env_var}")
            kwargs["token"] = token

        if self.name is None:
            loaded = load_dataset(self.path, **kwargs)
        else:
            loaded = load_dataset(self.path, self.name, **kwargs)
        if isinstance(loaded, Mapping):
            available = ", ".join(sorted(str(key) for key in loaded.keys()))
            raise ValueError(
                "HuggingFaceSource.fetch_dataset requires a split when load_dataset returns "
                f"a DatasetDict; available splits: {available}"
            )

        if limit is not None:
            count = int(limit)
            if hasattr(loaded, "__len__"):
                count = min(count, len(loaded))
            loaded = loaded.select(range(count))
        return Dataset.from_hf_dataset(
            loaded,
            prompt_column=prompt_column,
            prompt_template=prompt_template,
            example_key_column=example_key_column,
            prompt_hash_column=prompt_hash_column,
            label_columns=label_columns,
            case_columns=case_columns,
            case_key_column=case_key_column,
            metadata_columns=metadata_columns,
            index_column=index_column,
            index_prefix=index_prefix or str(split or self.name or self.path).replace("/", "_"),
            hash_columns=hash_columns,
            nested_record_column=nested_record_column,
            nested_record_index_column=nested_record_index_column,
            nested_record_field_paths=nested_record_field_paths,
            id=id,
            name=name or self.name or self.path,
        )


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
    if kind == "artifact_dataset":
        return ArtifactDatasetSource.from_dict(payload)
    if kind == "postgres":
        return PostgresSource.from_dict(payload)
    if kind == "huggingface":
        return HuggingFaceSource.from_dict(payload)
    raise ValueError(f"Unsupported source kind: {kind}")


def _artifact_from_value(value: Any) -> Any:
    if hasattr(value, "result"):
        return value
    if isinstance(value, Mapping):
        from pipelines_v2.storage.artifacts import artifact_from_dict

        return artifact_from_dict(value)
    raise TypeError(f"Unsupported artifact value: {type(value).__name__}")


def _path_get(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in str(path).split("."):
        if not part:
            continue
        if not isinstance(current, Mapping):
            raise TypeError(f"Cannot resolve path {path!r}; {part!r} parent is not a mapping")
        current = current.get(part)
    return current


def _example_has_token_sections(example: Any) -> bool:
    metadata = getattr(example, "metadata", None)
    token_sections = metadata.get("token_sections") if isinstance(metadata, Mapping) else None
    return isinstance(token_sections, Mapping) and bool(token_sections)
