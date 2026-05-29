"""Source adapters for reading examples, labels, cases, and metadata."""

from __future__ import annotations

import os
import re
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence
from urllib.request import urlopen

from pipelines_v2.core.types import RuntimeSecret
from pipelines_v2.data.datasets import Dataset

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
logger = logging.getLogger(__name__)


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

    def fetch_label_values(
        self,
        *,
        label_name: str,
        example_key_column: str = "example_id",
        **_: Any,
    ) -> Mapping[str, Any]:
        return {
            str(record[example_key_column]): record[label_name]
            for record in self.records
            if label_name in record and example_key_column in record
        }


@dataclass(frozen=True, slots=True)
class UrlJsonSource:
    """Deferred source for small public JSON prompt packs."""

    files: tuple[Mapping[str, Any], ...]
    kind: str = "url_json"
    defer_to_runtime: bool = True

    def __post_init__(self) -> None:
        normalized: list[dict[str, Any]] = []
        for index, raw_file in enumerate(self.files):
            file_spec = dict(raw_file)
            url = str(file_spec.get("url") or "").strip()
            if not url:
                raise ValueError(f"UrlJsonSource file {index} requires a non-empty url")
            normalized.append(
                {
                    "url": url,
                    "labels": {str(key): value for key, value in dict(file_spec.get("labels", {})).items()},
                    "metadata": {str(key): value for key, value in dict(file_spec.get("metadata", {})).items()},
                    "cases": {str(key): value for key, value in dict(file_spec.get("cases", {})).items()},
                    "source_name": str(file_spec.get("source_name") or f"file_{index}"),
                }
            )
        object.__setattr__(self, "files", tuple(normalized))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UrlJsonSource":
        return cls(files=tuple(dict(item) for item in payload.get("files", ())))

    def identity(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "files": [dict(file_spec) for file_spec in self.files],
        }

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return ()

    def runtime_pip_packages(self) -> tuple[str, ...]:
        return ()

    def fetch_dataset(
        self,
        *,
        prompt_column: str = "prompt",
        prompt_template: Any | None = None,
        example_key_column: str = "example_id",
        prompt_hash_column: str | None = None,
        label_columns: Sequence[str] = (),
        case_columns: Sequence[str] = (),
        case_key_column: str | None = None,
        metadata_columns: Sequence[str] = (),
        limit: int | None = None,
        limit_per_file: int | None = None,
        id: str | None = None,
        name: str | None = None,
    ) -> Dataset:
        """Fetch JSON records, inject file labels, and map them into examples."""

        records: list[dict[str, Any]] = []
        for file_index, file_spec in enumerate(self.files):
            with urlopen(str(file_spec["url"])) as response:  # noqa: S310 - explicit public data source
                payload = json.loads(response.read().decode("utf-8"))
            raw_records = payload.get("examples") if isinstance(payload, Mapping) and "examples" in payload else payload
            if not isinstance(raw_records, Sequence) or isinstance(raw_records, str | bytes | bytearray):
                raise TypeError("UrlJsonSource JSON payload must be a list or object with an examples list")
            file_count = 0
            for row_index, raw_record in enumerate(raw_records):
                if not isinstance(raw_record, Mapping):
                    raise TypeError("UrlJsonSource JSON records must be mappings")
                record = dict(raw_record)
                source_name = str(file_spec.get("source_name") or f"file_{file_index}")
                record.setdefault(example_key_column, f"{source_name}_{row_index:06d}")
                record.setdefault("source_name", source_name)
                record.setdefault("source_url", str(file_spec["url"]))
                record.update(dict(file_spec.get("labels", {})))
                record.update(dict(file_spec.get("metadata", {})))
                record.update(dict(file_spec.get("cases", {})))
                records.append(record)
                file_count += 1
                if limit_per_file is not None and file_count >= int(limit_per_file):
                    break
                if limit is not None and len(records) >= int(limit):
                    break
            if limit is not None and len(records) >= int(limit):
                break
        return Dataset.from_records(
            records,
            prompt_column=prompt_column,
            prompt_template=prompt_template,
            example_key_column=example_key_column,
            prompt_hash_column=prompt_hash_column,
            label_columns=label_columns,
            case_columns=case_columns,
            case_key_column=case_key_column,
            metadata_columns=metadata_columns,
            id=id,
            name=name,
        )


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
class _PostgresDatasetQuery:
    sql: str
    params: list[Any]
    default_name: str


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
        execution_shard: Mapping[str, int] | None = None,
    ) -> Dataset:
        """Fetch a dataset from either a relation or a SQL query.

        Exactly one of ``table`` or ``sql`` must be provided. ``sql`` is wrapped
        as a subquery so the dataset column projection and optional outer limit
        remain under library control.
        """
        import psycopg
        from psycopg.rows import dict_row

        dataset_query = _postgres_dataset_query(
            table=table,
            sql=sql,
            prompt_column=prompt_column,
            example_key_column=example_key_column,
            prompt_hash_column=prompt_hash_column,
            label_columns=label_columns,
            case_columns=case_columns,
            case_key_column=case_key_column,
            metadata_columns=metadata_columns,
            limit=limit,
            execution_shard=execution_shard,
        )

        start = time.monotonic()
        logger.info(
            "Fetching Postgres dataset name=%s table=%s sql=%s labels=%s limit=%s shard=%s",
            name or dataset_query.default_name,
            table,
            bool(sql),
            tuple(label_columns),
            limit,
            dict(execution_shard) if execution_shard is not None else None,
        )
        with psycopg.connect(self._resolved_url(), row_factory=dict_row) as conn:
            records = conn.execute(dataset_query.sql, dataset_query.params).fetchall()
        logger.info(
            "Fetched Postgres dataset rows=%s elapsed_s=%.2f",
            len(records),
            time.monotonic() - start,
        )

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
            name=name or dataset_query.default_name,
        )

    def fetch_label_values(
        self,
        *,
        label_name: str,
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
        execution_shard: Mapping[str, int] | None = None,
    ) -> Mapping[str, Any]:
        """Fetch one label column keyed by example id without materializing prompts."""
        import psycopg
        from psycopg.rows import dict_row

        del prompt_column, label_columns, case_columns, case_key_column, metadata_columns, id
        dataset_query = _postgres_label_query(
            table=table,
            sql=sql,
            example_key_column=example_key_column,
            label_name=label_name,
            prompt_hash_column=prompt_hash_column,
            limit=limit,
            execution_shard=execution_shard,
        )
        start = time.monotonic()
        logger.info(
            "Fetching Postgres label values name=%s dataset=%s table=%s sql=%s limit=%s shard=%s",
            label_name,
            name or dataset_query.default_name,
            table,
            bool(sql),
            limit,
            dict(execution_shard) if execution_shard is not None else None,
        )
        with psycopg.connect(self._resolved_url(), row_factory=dict_row) as conn:
            records = conn.execute(dataset_query.sql, dataset_query.params).fetchall()
        logger.info(
            "Fetched Postgres label values name=%s rows=%s elapsed_s=%.2f",
            label_name,
            len(records),
            time.monotonic() - start,
        )
        return {str(record[example_key_column]): record[label_name] for record in records}

    def iter_dataset_batches(
        self,
        *,
        batch_size: int,
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
        execution_shard: Mapping[str, int] | None = None,
    ) -> Iterable[Dataset]:
        """Stream a Postgres dataset as materialized ``Dataset`` batches."""
        import psycopg
        from psycopg.rows import dict_row

        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("PostgresSource.iter_dataset_batches requires batch_size > 0")
        dataset_query = _postgres_dataset_query(
            table=table,
            sql=sql,
            prompt_column=prompt_column,
            example_key_column=example_key_column,
            prompt_hash_column=prompt_hash_column,
            label_columns=label_columns,
            case_columns=case_columns,
            case_key_column=case_key_column,
            metadata_columns=metadata_columns,
            limit=limit,
            execution_shard=execution_shard,
        )

        page_index = 0
        offset = 0
        page_order_columns = list(dict.fromkeys([*([prompt_hash_column] if prompt_hash_column else []), example_key_column]))
        page_order_sql = ", ".join(
            f"pipelines_v2_dataset_page.{_quote_identifier(column)}" for column in page_order_columns
        )
        paged_sql = (
            f"SELECT * FROM ({dataset_query.sql}) AS pipelines_v2_dataset_page "
            f"ORDER BY {page_order_sql} LIMIT %s OFFSET %s"
        )
        while True:
            with psycopg.connect(self._resolved_url(), row_factory=dict_row) as conn:
                records = conn.execute(
                    paged_sql,
                    [*dataset_query.params, batch_size, offset],
                ).fetchall()
            if not records:
                break
            logger.info(
                "Fetched Postgres dataset batch name=%s rows=%s batch_index=%s offset=%s",
                name or dataset_query.default_name,
                len(records),
                page_index,
                offset,
            )
            yield Dataset.from_records(
                records,
                prompt_column=prompt_column,
                example_key_column=example_key_column,
                prompt_hash_column=prompt_hash_column,
                label_columns=label_columns,
                case_columns=case_columns,
                case_key_column=case_key_column,
                metadata_columns=metadata_columns,
                id=id,
                name=name or dataset_query.default_name,
            )
            if len(records) < batch_size:
                break
            page_index += 1
            offset += batch_size

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


@dataclass(frozen=True, slots=True)
class HuggingFaceListContrastSource:
    """Deferred HF source that expands positive/negative list columns into rows."""

    path: str
    name: str | None = None
    revision: str | None = None
    token_env_var: str | None = None
    trust_remote_code: bool = False
    kind: str = "huggingface_list_contrast"
    defer_to_runtime: bool = True

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ValueError("HuggingFaceListContrastSource path cannot be empty")
        object.__setattr__(self, "path", str(self.path))
        if self.name is not None:
            object.__setattr__(self, "name", str(self.name))
        if self.revision is not None:
            object.__setattr__(self, "revision", str(self.revision))
        if self.token_env_var is not None:
            object.__setattr__(self, "token_env_var", str(self.token_env_var))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HuggingFaceListContrastSource":
        return cls(
            path=str(payload["path"]),
            name=str(payload["name"]) if payload.get("name") is not None else None,
            revision=str(payload["revision"]) if payload.get("revision") is not None else None,
            token_env_var=str(payload["token_env_var"]) if payload.get("token_env_var") is not None else None,
            trust_remote_code=bool(payload.get("trust_remote_code", False)),
        )

    def identity(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind, "path": self.path}
        if self.name is not None:
            payload["name"] = self.name
        if self.revision is not None:
            payload["revision"] = self.revision
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
        split: str,
        prompt_column: str,
        prompt_template: Any | None = None,
        example_key_column: str = "example_id",
        label_name: str = "contrast_label",
        positive_column: str,
        negative_column: str,
        positive_label: str = "positive",
        negative_label: str = "negative",
        answer_column: str = "answer",
        metadata_columns: Sequence[str] = (),
        limit: int | None = None,
        id: str | None = None,
        name: str | None = None,
    ) -> Dataset:
        from datasets import load_dataset

        kwargs: dict[str, Any] = {"split": split}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.trust_remote_code:
            kwargs["trust_remote_code"] = True
        if self.token_env_var is not None:
            token = os.environ.get(self.token_env_var)
            if not token:
                raise RuntimeError(f"Missing required environment variable: {self.token_env_var}")
            kwargs["token"] = token
        loaded = load_dataset(self.path, self.name, **kwargs) if self.name is not None else load_dataset(self.path, **kwargs)
        rows = loaded.to_list() if hasattr(loaded, "to_list") else [dict(row) for row in loaded]
        records: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            base = dict(row)
            for label, column in ((positive_label, positive_column), (negative_label, negative_column)):
                for answer_index, answer in enumerate(_coerce_answer_list(base.get(column))):
                    record = dict(base)
                    record[answer_column] = str(answer)
                    record[label_name] = str(label)
                    record[example_key_column] = f"{row_index:06d}_{label}_{answer_index:03d}"
                    records.append(record)
                    if limit is not None and len(records) >= int(limit):
                        break
                if limit is not None and len(records) >= int(limit):
                    break
            if limit is not None and len(records) >= int(limit):
                break
        return Dataset.from_records(
            records,
            prompt_column=prompt_column,
            prompt_template=prompt_template,
            example_key_column=example_key_column,
            label_columns=(label_name,),
            case_columns=(prompt_column,),
            case_key_column=prompt_column,
            metadata_columns=tuple(dict.fromkeys([answer_column, *metadata_columns])),
            id=id,
            name=name or self.name or self.path,
        )


def _coerce_answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _quote_identifier(value: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid SQL identifier: {value!r}")
    return f'"{value}"'


def _postgres_prompt_hash_shard_predicate(
    *,
    prompt_hash_column: str | None,
    execution_shard: Mapping[str, int] | None,
) -> str | None:
    if execution_shard is None:
        return None
    if prompt_hash_column is None:
        return None
    count = int(execution_shard.get("count") or 1)
    index = int(execution_shard.get("index") or 0)
    if count <= 1:
        return None
    if index < 0 or index >= count:
        raise ValueError(f"Invalid execution shard index {index} for shard count {count}")
    column = f"src.{_quote_identifier(prompt_hash_column)}"
    # Match runtime _prompt_hash_shard(): int(hash[:16], 16) % count, without
    # overflowing signed int64 for hashes whose top bit is set.
    value_expr = (
        f"((('x' || substr({column}, 1, 8))::bit(32)::bigint::numeric * 4294967296) + "
        f"(('x' || substr({column}, 9, 8))::bit(32)::bigint::numeric))"
    )
    return f"MOD({value_expr}, %s) = %s"


def _postgres_dataset_query(
    *,
    table: str | None,
    sql: str | None,
    prompt_column: str,
    example_key_column: str,
    prompt_hash_column: str | None,
    label_columns: Sequence[str],
    case_columns: Sequence[str],
    case_key_column: str | None,
    metadata_columns: Sequence[str],
    limit: int | None,
    execution_shard: Mapping[str, int] | None,
) -> _PostgresDatasetQuery:
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
        source_sql = _quote_relation(table)
        default_name = str(table)
    else:
        assert sql is not None
        base_sql = _normalize_sql_query(sql)
        source_sql = f"({base_sql})"
        default_name = "postgres_query"
    query = f"SELECT {select_sql} FROM {source_sql} AS src"
    params: list[Any] = []
    shard_predicate = _postgres_prompt_hash_shard_predicate(
        prompt_hash_column=prompt_hash_column,
        execution_shard=execution_shard,
    )
    if shard_predicate is not None:
        query += f" WHERE {shard_predicate}"
        assert execution_shard is not None
        params.extend([int(execution_shard["count"]), int(execution_shard["index"])])
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)
    return _PostgresDatasetQuery(sql=query, params=params, default_name=default_name)


def _postgres_label_query(
    *,
    table: str | None,
    sql: str | None,
    example_key_column: str,
    label_name: str,
    prompt_hash_column: str | None,
    limit: int | None,
    execution_shard: Mapping[str, int] | None,
) -> _PostgresDatasetQuery:
    if bool(table) == bool(sql):
        raise ValueError("PostgresSource.fetch_label_values requires exactly one of table or sql")

    unique_columns = list(dict.fromkeys([example_key_column, label_name, *([prompt_hash_column] if prompt_hash_column else [])]))
    select_sql = ", ".join(_quote_identifier(column) for column in unique_columns)
    if table is not None:
        source_sql = _quote_relation(table)
        default_name = str(table)
    else:
        assert sql is not None
        base_sql = _normalize_sql_query(sql)
        source_sql = f"({base_sql})"
        default_name = "postgres_query"
    query = f"SELECT {select_sql} FROM {source_sql} AS src"
    params: list[Any] = []
    shard_predicate = _postgres_prompt_hash_shard_predicate(
        prompt_hash_column=prompt_hash_column,
        execution_shard=execution_shard,
    )
    if shard_predicate is not None:
        query += f" WHERE {shard_predicate}"
        assert execution_shard is not None
        params.extend([int(execution_shard["count"]), int(execution_shard["index"])])
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)
    return _PostgresDatasetQuery(sql=query, params=params, default_name=default_name)


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
    if kind == "url_json":
        return UrlJsonSource.from_dict(payload)
    if kind == "artifact_dataset":
        return ArtifactDatasetSource.from_dict(payload)
    if kind == "postgres":
        return PostgresSource.from_dict(payload)
    if kind == "huggingface":
        return HuggingFaceSource.from_dict(payload)
    if kind == "huggingface_list_contrast":
        return HuggingFaceListContrastSource.from_dict(payload)
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
