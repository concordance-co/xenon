"""Dataset, example, label, and case abstractions for the v2 API."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pipelines_v2.core.types import RuntimeSecret, stable_hash

Prompt = str | Sequence[Mapping[str, Any]]


def prompt_hash(prompt: Prompt) -> str:
    if isinstance(prompt, str):
        payload: Any = prompt
    else:
        payload = list(prompt)
    return stable_hash(payload)


@dataclass(frozen=True, slots=True)
class Example:
    """One model input plus its labels, metadata, and case memberships."""

    key: str
    prompt: Prompt
    prompt_hash: str | None = None
    labels: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cases: Mapping[str, Any] = field(default_factory=dict)
    case_key: str | None = None

    def __post_init__(self) -> None:
        if not str(self.key).strip():
            raise ValueError("Example key cannot be empty")
        if self.prompt_hash is None:
            object.__setattr__(self, "prompt_hash", prompt_hash(self.prompt))
        object.__setattr__(self, "key", str(self.key))
        normalized_cases = {str(name): value for name, value in dict(self.cases).items() if value is not None}
        if self.case_key is not None:
            normalized_case_key = str(self.case_key)
            existing = normalized_cases.get("case_key")
            if existing is not None and str(existing) != normalized_case_key:
                raise ValueError("Example case_key conflicts with cases['case_key']")
            normalized_cases["case_key"] = normalized_case_key
            object.__setattr__(self, "case_key", normalized_case_key)
        elif "case_key" in normalized_cases:
            object.__setattr__(self, "case_key", str(normalized_cases["case_key"]))
        object.__setattr__(self, "cases", normalized_cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "prompt": self.prompt,
            "prompt_hash": self.prompt_hash,
            "labels": dict(self.labels),
            "metadata": dict(self.metadata),
            "cases": dict(self.cases),
            "case_key": self.case_key,
        }

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "prompt_hash": self.prompt_hash,
            "labels": dict(self.labels),
            "metadata": dict(self.metadata),
            "cases": dict(self.cases),
            "case_key": self.case_key,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Example":
        return cls(
            key=str(payload["key"]),
            prompt=_coerce_prompt(payload["prompt"]),
            prompt_hash=str(payload["prompt_hash"]) if payload.get("prompt_hash") is not None else None,
            labels=dict(payload.get("labels", {})),
            metadata=dict(payload.get("metadata", {})),
            cases=dict(payload.get("cases", {})),
            case_key=str(payload["case_key"]) if payload.get("case_key") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class LabelPredicate:
    """Filter over a label source, currently supporting equality tests only."""
    label_set: Any
    op: str
    value: Any

    kind: ClassVar[str] = "label_predicate"

    def matches(self, example_key: str, values: Mapping[str, Any] | None = None) -> bool:
        """Return whether the referenced example satisfies this predicate."""
        resolved = values if values is not None else self.label_set.resolve_values()
        actual = resolved[example_key]
        if self.op == "equals":
            return actual == self.value
        raise ValueError(f"Unsupported label predicate op: {self.op}")

    def resolve_example_keys(self) -> list[str]:
        """Materialize the matching example keys in the current runtime."""
        values = self.label_set.resolve_values()
        return [key for key in values if self.matches(key, values)]

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return self.label_set.runtime_secrets()

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label_set": self.label_set.semantic_dict() if hasattr(self.label_set, "semantic_dict") else self.label_set,
            "op": self.op,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LabelPredicate":
        from pipelines_v2.storage.artifacts import ArtifactLabelRef
        from pipelines_v2.workflow import StepLabelRef

        label_set_payload = payload["label_set"]
        if not isinstance(label_set_payload, Mapping):
            raise TypeError("LabelPredicate label_set payload must be a mapping")
        kind = label_set_payload.get("kind")
        if kind == ArtifactLabelRef.kind:
            label_ref: Any = ArtifactLabelRef.from_dict(label_set_payload)
        elif kind == StepLabelRef.kind:
            label_ref = StepLabelRef.from_dict(label_set_payload)
        else:
            label_ref = LabelSet.from_dict(label_set_payload)
        return cls(
            label_set=label_ref,
            op=str(payload["op"]),
            value=payload.get("value"),
        )


@dataclass(frozen=True, slots=True)
class LabelSet:
    """Reference to one named label column over a dataset."""

    name: str
    dataset: "Dataset"

    kind: ClassVar[str] = "label_set"

    def equals(self, value: Any) -> LabelPredicate:
        """Build a predicate selecting examples whose label equals ``value``."""
        return LabelPredicate(label_set=self, op="equals", value=value)

    def for_examples(self, example_keys: Iterable[str]) -> list[Any]:
        """Return label values in the same order as ``example_keys``."""
        values = self.resolve_values()
        return [values[key] for key in example_keys]

    @property
    def values(self) -> Mapping[str, Any]:
        """Return materialized example_key -> label mappings for local datasets."""
        if self.dataset.is_deferred:
            raise RuntimeError("Deferred label refs must be resolved in a runtime before reading values")
        return _label_values_from_dataset(self.dataset, self.name)

    def resolve_values(self) -> Mapping[str, Any]:
        """Resolve values even when the underlying dataset is deferred."""
        dataset = self.dataset.resolve() if self.dataset.is_deferred else self.dataset
        return _label_values_from_dataset(dataset, self.name)

    def resolve_example_keys(self) -> list[str]:
        """Return the example keys covered by this label ref."""
        return sorted(str(key) for key in self.resolve_values())

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return self.dataset.runtime_secrets()

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "dataset": self.dataset.semantic_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LabelSet":
        return cls(
            name=str(payload["name"]),
            dataset=Dataset.from_dict(payload["dataset"]),
        )


@dataclass(frozen=True, slots=True)
class CaseSet:
    """Reference to one named case/grouping column over a dataset."""

    name: str
    dataset: "Dataset"

    kind: ClassVar[str] = "case_set"

    def for_examples(self, example_keys: Iterable[str]) -> list[str]:
        """Return case ids in the same order as ``example_keys``."""
        values = self.resolve_values()
        return [values[key] for key in example_keys]

    @property
    def values(self) -> Mapping[str, str]:
        """Return materialized example_key -> case_id mappings for local datasets."""
        if self.dataset.is_deferred:
            raise RuntimeError("Deferred case refs must be resolved in a runtime before reading values")
        return _case_values_from_dataset(self.dataset, self.name)

    def resolve_values(self) -> Mapping[str, str]:
        """Resolve case ids even when the underlying dataset is deferred."""
        dataset = self.dataset.resolve() if self.dataset.is_deferred else self.dataset
        return _case_values_from_dataset(dataset, self.name)

    def resolve_example_keys(self) -> list[str]:
        """Return the example keys covered by this case ref."""
        return sorted(str(key) for key in self.resolve_values())

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return self.dataset.runtime_secrets()

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "dataset": self.dataset.semantic_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CaseSet":
        return cls(
            name=str(payload["name"]),
            dataset=Dataset.from_dict(payload["dataset"]),
        )


@dataclass(frozen=True, slots=True)
class Dataset:
    """A collection of examples, either materialized locally or deferred to a source."""

    examples: tuple[Example, ...] = ()
    id: str | None = None
    name: str | None = None
    source: Mapping[str, Any] | None = None
    fetch: Mapping[str, Any] | None = None
    selection: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source is not None and self.fetch is None:
            raise ValueError("Deferred datasets require fetch parameters")
        keys = [example.key for example in self.examples]
        duplicates = {key for key in keys if keys.count(key) > 1}
        if duplicates:
            raise ValueError(f"Duplicate example keys: {sorted(duplicates)}")
        if self.source is not None:
            object.__setattr__(self, "source", dict(self.source))
            object.__setattr__(self, "fetch", dict(self.fetch or {}))
            object.__setattr__(self, "selection", dict(self.selection))
        if self.id is None:
            if self.is_deferred:
                object.__setattr__(
                    self,
                    "id",
                    stable_hash(
                        {
                            "source": dict(self.source or {}),
                            "fetch": dict(self.fetch or {}),
                            "selection": dict(self.selection),
                        }
                    )[:24],
                )
            else:
                object.__setattr__(self, "id", stable_hash([self._example_ref(e) for e in self.examples])[:24])

    @classmethod
    def from_examples(
        cls,
        examples: Iterable[Example],
        *,
        id: str | None = None,
        name: str | None = None,
    ) -> "Dataset":
        """Build a dataset from already materialized ``Example`` objects."""
        return cls(examples=tuple(examples), id=id, name=name)

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        prompt_column: str = "prompt",
        example_key_column: str = "example_id",
        prompt_hash_column: str | None = None,
        label_columns: Sequence[str] = (),
        case_columns: Sequence[str] = (),
        case_key_column: str | None = None,
        metadata_columns: Sequence[str] = (),
        id: str | None = None,
        name: str | None = None,
    ) -> "Dataset":
        """Build a dataset by mapping record columns into examples, labels, and cases."""
        examples: list[Example] = []
        for record in records:
            prompt = _coerce_prompt(record[prompt_column])
            labels = {column: record.get(column) for column in label_columns}
            metadata = {column: record.get(column) for column in metadata_columns}
            case_names = list(dict.fromkeys([*case_columns, *([case_key_column] if case_key_column else [])]))
            cases = {column: record.get(column) for column in case_names if record.get(column) is not None}
            examples.append(
                Example(
                    key=str(record[example_key_column]),
                    prompt=prompt,
                    prompt_hash=(
                        str(record[prompt_hash_column])
                        if prompt_hash_column and record.get(prompt_hash_column) is not None
                        else None
                    ),
                    labels=labels,
                    metadata=metadata,
                    cases=cases,
                    case_key=(
                        str(record[case_key_column])
                        if case_key_column and record.get(case_key_column) is not None
                        else None
                    ),
                )
            )
        return cls.from_examples(examples, id=id, name=name)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        prompt_column: str = "prompt",
        example_key_column: str = "example_id",
        prompt_hash_column: str | None = None,
        label_columns: Sequence[str] = (),
        case_columns: Sequence[str] = (),
        case_key_column: str | None = None,
        metadata_columns: Sequence[str] = (),
        id: str | None = None,
        name: str | None = None,
    ) -> "Dataset":
        """Load records from JSON and map them into examples."""
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        records = payload["examples"] if isinstance(payload, dict) and "examples" in payload else payload
        if not isinstance(records, list):
            raise ValueError("Dataset JSON must be a list or an object with an 'examples' list")
        return cls.from_records(
            records,
            prompt_column=prompt_column,
            example_key_column=example_key_column,
            prompt_hash_column=prompt_hash_column,
            label_columns=label_columns,
            case_columns=case_columns,
            case_key_column=case_key_column,
            metadata_columns=metadata_columns,
            id=id,
            name=name,
        )

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        prompt_column: str = "prompt",
        example_key_column: str = "example_id",
        prompt_hash_column: str | None = None,
        label_columns: Sequence[str] = (),
        case_columns: Sequence[str] = (),
        case_key_column: str | None = None,
        metadata_columns: Sequence[str] = (),
        id: str | None = None,
        name: str | None = None,
    ) -> "Dataset":
        """Load records from parquet and map them into examples."""
        import pyarrow.parquet as pq

        records = pq.read_table(path).to_pylist()
        return cls.from_records(
            records,
            prompt_column=prompt_column,
            example_key_column=example_key_column,
            prompt_hash_column=prompt_hash_column,
            label_columns=label_columns,
            case_columns=case_columns,
            case_key_column=case_key_column,
            metadata_columns=metadata_columns,
            id=id,
            name=name,
        )

    @classmethod
    def from_source(cls, *, source: Any, **kwargs: Any) -> "Dataset":
        """Build a dataset from a source, optionally deferring fetch to the runtime."""
        if not hasattr(source, "fetch_dataset"):
            raise TypeError("source must provide fetch_dataset(...)")
        defer = kwargs.pop("defer", None)
        if defer is None:
            defer = bool(getattr(source, "defer_to_runtime", False))
        if not defer:
            return source.fetch_dataset(**kwargs)
        if not hasattr(source, "identity"):
            raise TypeError("deferred sources must provide identity()")
        return cls(
            id=str(kwargs["id"]) if kwargs.get("id") is not None else None,
            name=str(kwargs["name"]) if kwargs.get("name") is not None else None,
            source=source.identity(),
            fetch=dict(kwargs),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Dataset":
        """Deserialize either a materialized dataset or a deferred dataset ref."""
        if payload.get("source") is not None:
            return cls(
                examples=(),
                id=str(payload["id"]) if payload.get("id") is not None else None,
                name=str(payload["name"]) if payload.get("name") is not None else None,
                source=dict(payload["source"]),
                fetch=dict(payload.get("fetch", {})),
                selection=dict(payload.get("selection", {})),
            )
        examples_payload = payload.get("examples", ())
        if not isinstance(examples_payload, Sequence):
            raise TypeError("Dataset payload examples must be a sequence")
        return cls.from_examples(
            [Example.from_dict(example) for example in examples_payload],
            id=str(payload["id"]) if payload.get("id") is not None else None,
            name=str(payload["name"]) if payload.get("name") is not None else None,
        )

    @classmethod
    def from_postgres(cls, *, source: Any, **kwargs: Any) -> "Dataset":
        """Build a deferred Postgres dataset from a table or SQL query."""
        if getattr(source, "kind", None) != "postgres":
            raise TypeError("Dataset.from_postgres requires a PostgresSource")
        if bool(kwargs.get("table")) == bool(kwargs.get("sql")):
            raise ValueError("Dataset.from_postgres requires exactly one of table or sql")
        return cls.from_source(source=source, defer=True, **kwargs)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> "Dataset":
        """Load a dataset from a supported local file format."""
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            return cls.from_json(path, **kwargs)
        if suffix == ".parquet":
            return cls.from_parquet(path, **kwargs)
        raise ValueError(f"Unsupported dataset file suffix: {suffix}")

    def select(self, *, limit: int | None = None, keys: Iterable[str] | None = None) -> "Dataset":
        """Return a narrowed dataset while preserving deferred execution when applicable."""
        if self.is_deferred:
            selection = dict(self.selection)
            if keys is not None:
                new_keys = [str(key) for key in keys]
                existing_keys = selection.get("keys")
                if existing_keys is not None:
                    allowed = set(new_keys)
                    new_keys = [key for key in existing_keys if key in allowed]
                selection["keys"] = new_keys
            if limit is not None:
                current_limit = selection.get("limit")
                selection["limit"] = min(int(current_limit), int(limit)) if current_limit is not None else int(limit)
            return Dataset(
                examples=(),
                id=stable_hash([self.id, selection])[:24],
                name=self.name,
                source=self.source,
                fetch=self.fetch,
                selection=selection,
            )
        selected = self.examples
        if keys is not None:
            allowed = set(keys)
            selected = tuple(example for example in selected if example.key in allowed)
        if limit is not None:
            selected = selected[:limit]
        return Dataset(examples=selected, id=stable_hash([self.id, [e.key for e in selected]])[:24], name=self.name)

    def labels(self, name: str) -> LabelSet:
        """Return a named label reference over this dataset."""
        if not self.is_deferred:
            _label_values_from_dataset(self, name)
        return LabelSet(name=name, dataset=self)

    def cases(self, name: str = "case_key") -> CaseSet:
        """Return a named case/group reference over this dataset."""
        if not self.is_deferred:
            _case_values_from_dataset(self, name)
        return CaseSet(name=name, dataset=self)

    def example_keys(self) -> list[str]:
        """Return known example keys for materialized datasets or explicit selections."""
        if self.is_deferred:
            keys = self.selection.get("keys")
            return [str(key) for key in keys] if isinstance(keys, list) else []
        return [example.key for example in self.examples]

    def resolve_example_keys(self) -> list[str]:
        """Return the example keys covered by this dataset in the current runtime."""
        if self.is_deferred:
            known = self.example_keys()
            if known:
                return known
            return self.resolve().example_keys()
        return self.example_keys()

    def coverage(self) -> dict[str, Any]:
        """Return manifest-friendly coverage information for this dataset."""
        if self.is_deferred:
            return {
                "dataset_id": self.id,
                "dataset_name": self.name,
                "materialized": False,
                "source": dict(self.source or {}),
                "selection": dict(self.selection),
                "example_count": None,
                "example_keys": self.example_keys(),
                "prompt_hashes": {},
            }
        return {
            "dataset_id": self.id,
            "dataset_name": self.name,
            "materialized": True,
            "example_count": len(self.examples),
            "example_keys": self.example_keys(),
            "prompt_hashes": {example.key: example.prompt_hash for example in self.examples},
        }

    def semantic_dict(self) -> dict[str, Any]:
        if self.is_deferred:
            fetch = dict(self.fetch or {})
            fetch.pop("id", None)
            fetch.pop("name", None)
            return {
                "kind": "dataset",
                "mode": "deferred",
                "source": dict(self.source or {}),
                "fetch": fetch,
                "selection": dict(self.selection),
            }
        return {
            "kind": "dataset",
            "mode": "materialized",
            "examples": [example.semantic_dict() for example in self.examples],
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataset into a JSON-safe payload."""
        if self.is_deferred:
            return {
                "id": self.id,
                "name": self.name,
                "source": dict(self.source or {}),
                "fetch": dict(self.fetch or {}),
                "selection": dict(self.selection),
            }
        return {
            "id": self.id,
            "name": self.name,
            "example_count": len(self.examples),
            "examples": [example.to_dict() for example in self.examples],
        }

    @property
    def is_deferred(self) -> bool:
        """Whether this dataset will be resolved later inside a runtime."""
        return self.source is not None and not self.examples

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        """Return runtime secrets required to resolve a deferred dataset."""
        if not self.is_deferred:
            return ()
        from pipelines_v2.data.sources import source_from_dict

        source = source_from_dict(dict(self.source or {}))
        return source.runtime_secrets()

    def runtime_pip_packages(self) -> tuple[str, ...]:
        """Return Python packages required to resolve a deferred dataset."""
        if not self.is_deferred:
            return ()
        from pipelines_v2.data.sources import source_from_dict

        source = source_from_dict(dict(self.source or {}))
        if hasattr(source, "runtime_pip_packages"):
            return tuple(source.runtime_pip_packages())
        return ()

    def resolve(self) -> "Dataset":
        """Materialize a deferred dataset in the current runtime."""
        if not self.is_deferred:
            return self
        from pipelines_v2.data.sources import source_from_dict

        source = source_from_dict(dict(self.source or {}))
        dataset = source.fetch_dataset(**dict(self.fetch or {}))
        selection_keys = self.selection.get("keys")
        selection_limit = self.selection.get("limit")
        if selection_keys is not None or selection_limit is not None:
            dataset = dataset.select(
                keys=selection_keys,
                limit=int(selection_limit) if selection_limit is not None else None,
            )
        if self.id is not None or self.name is not None:
            dataset = Dataset.from_examples(
                dataset.examples,
                id=self.id,
                name=self.name or dataset.name,
            )
        return dataset

    @staticmethod
    def _example_ref(example: Example) -> dict[str, Any]:
        return {
            "key": example.key,
            "prompt_hash": example.prompt_hash,
            "labels": dict(example.labels),
            "cases": dict(example.cases),
            "case_key": example.case_key,
        }


def _coerce_prompt(value: Any) -> Prompt:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return value
            if isinstance(parsed, list):
                return parsed
        return value
    if isinstance(value, list):
        return value
    raise TypeError(f"Unsupported prompt value type: {type(value).__name__}")


def _label_values_from_dataset(dataset: Dataset, name: str) -> Mapping[str, Any]:
    missing = [example.key for example in dataset.examples if name not in example.labels]
    if missing:
        raise KeyError(f"Label {name!r} missing for examples: {missing[:5]}")
    return {example.key: example.labels[name] for example in dataset.examples}


def _case_values_from_dataset(dataset: Dataset, name: str) -> Mapping[str, str]:
    missing = [example.key for example in dataset.examples if example.cases.get(name) is None]
    if missing:
        raise KeyError(f"Case {name!r} missing for examples: {missing[:5]}")
    return {example.key: str(example.cases[name]) for example in dataset.examples}
