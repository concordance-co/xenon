"""Shared internal helpers for operation specs and execution."""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines_v2.core.paths import find_workspace_root, resolve_workspace_path
from pipelines_v2.core.types import RuntimeSecret, SpecValidationError
from pipelines_v2.data.datasets import CaseSet, Dataset, LabelPredicate, LabelSet

from .tokens import TokenSelector


def analysis_runtime_spec() -> Any:
    from pipelines_v2.engine.base import PythonRuntimeSpec

    return PythonRuntimeSpec(
        pip_packages=(
            "matplotlib",
            "numpy",
            "scikit-learn",
            "safetensors",
            "pyarrow",
            "psycopg[binary]",
        ),
        local_python_sources=("pipelines_v2",),
    )


def callable_import_ref(
    function: Any,
    *,
    local_python_sources: Sequence[str] | None,
    label: str,
) -> tuple[str, tuple[str, ...]]:
    if not callable(function):
        raise TypeError(f"{label}.from_function(...) expects a callable")
    qualname = str(getattr(function, "__qualname__", getattr(function, "__name__", "")))
    if "<locals>" in qualname or "." in qualname:
        raise SpecValidationError(f"{label} must be top-level named functions")
    source_file = inspect.getsourcefile(function)
    if source_file is None:
        raise SpecValidationError(f"Could not determine source file for {label[:-1].lower()}")
    source_path = Path(source_file).resolve()
    workspace_root = find_workspace_root(source_path)

    if local_python_sources is None:
        try:
            relative_path = source_path.relative_to(workspace_root)
        except ValueError as exc:
            raise SpecValidationError(f"{label} source file must live under the current workspace") from exc
        sources = (".",)
    else:
        sources = tuple(str(source) for source in local_python_sources if str(source).strip())
        if not sources:
            raise SpecValidationError(f"{label} local_python_sources cannot be empty when provided")
        relative_path = None
        for source in sources:
            source_root = resolve_workspace_path(source, workspace_root=workspace_root)
            try:
                relative_path = source_path.relative_to(source_root)
                break
            except ValueError:
                continue
        if relative_path is None:
            raise SpecValidationError(
                f"{label} source file {source_path} is not under any declared local_python_sources: {list(sources)}"
            )

    if relative_path.suffix != ".py":
        raise SpecValidationError(f"{label} source file must be a Python module")
    module_name = ".".join(relative_path.with_suffix("").parts)
    return f"{module_name}:{getattr(function, '__name__', '')}", sources


def load_importable_function(
    import_path: str,
    *,
    label: str,
    local_python_sources: Sequence[str] = (),
) -> Any:
    module_name, _, function_name = import_path.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = str(getattr(exc, "name", "") or "")
        if missing_name and missing_name != module_name and not module_name.startswith(f"{missing_name}."):
            raise
        workspace_root = find_workspace_root()
        source_paths = [
            str(resolve_workspace_path(source, workspace_root=workspace_root))
            for source in local_python_sources
        ]
        added_paths: list[str] = []
        for source_path in source_paths:
            if source_path not in sys.path:
                sys.path.insert(0, source_path)
                added_paths.append(source_path)
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as retry_exc:
            raise SpecValidationError(
                f"{label} module {module_name!r} could not be imported. "
                f"Tried local_python_sources={list(local_python_sources)}"
            ) from retry_exc
        finally:
            for source_path in reversed(added_paths):
                try:
                    sys.path.remove(source_path)
                except ValueError:
                    pass
    try:
        return getattr(module, function_name)
    except AttributeError as exc:
        raise SpecValidationError(
            f"{label} function {function_name!r} is not defined in module {module_name!r}"
        ) from exc


def runtime_secrets_from_refs(*values: Any) -> tuple[RuntimeSecret, ...]:
    secrets: dict[str, RuntimeSecret] = {}
    for value in values:
        for secret in iter_runtime_secrets(value):
            secrets.setdefault(secret.env_var, secret)
    return tuple(secrets[key] for key in sorted(secrets))


def iter_runtime_secrets(value: Any) -> tuple[RuntimeSecret, ...]:
    if value is None:
        return ()
    if hasattr(value, "runtime_secrets"):
        return tuple(value.runtime_secrets())
    if isinstance(value, tuple | list):
        secrets: list[RuntimeSecret] = []
        for item in value:
            secrets.extend(iter_runtime_secrets(item))
        return tuple(secrets)
    if isinstance(value, dict):
        secrets = []
        for item in value.values():
            secrets.extend(iter_runtime_secrets(item))
        return tuple(secrets)
    return ()


def contains_section_token_selector(value: Any) -> bool:
    if isinstance(value, TokenSelector):
        return value.kind == "section"
    if value is None:
        return False
    if hasattr(value, "__dataclass_fields__"):
        return any(contains_section_token_selector(getattr(value, field_name)) for field_name in value.__dataclass_fields__)
    if isinstance(value, tuple | list):
        return any(contains_section_token_selector(item) for item in value)
    if isinstance(value, dict):
        return any(contains_section_token_selector(item) for item in value.values())
    return False


def spec_uses_section_token_selector(value: Any) -> bool:
    return contains_section_token_selector(value)


def example_has_explicit_token_sections(example: Any) -> bool:
    metadata = getattr(example, "metadata", None)
    if not isinstance(metadata, Mapping):
        return False
    token_sections = metadata.get("token_sections")
    return isinstance(token_sections, Mapping) and bool(token_sections)


def merge_string_tuples(*values: Sequence[str]) -> tuple[str, ...]:
    merged: list[str] = []
    for value in values:
        for item in value:
            normalized = str(item)
            if normalized not in merged:
                merged.append(normalized)
    return tuple(merged)


def spec_value_from_dict(value: Any) -> Any:
    if isinstance(value, list):
        return [spec_value_from_dict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(spec_value_from_dict(item) for item in value)
    if not isinstance(value, dict):
        return value

    kind = value.get("kind")
    if kind == LabelSet.kind:
        return LabelSet.from_dict(value)
    if kind == CaseSet.kind:
        return CaseSet.from_dict(value)
    if kind == LabelPredicate.kind:
        return LabelPredicate.from_dict(value)

    from pipelines_v2.storage.artifacts import (
        ArtifactLabelRef,
        CaptureArtifact,
        FeatureLayerRef,
        FeatureRef,
        OperationArtifact,
    )
    from pipelines_v2.workflow import StepFeatureRef, StepLabelRef, StepRef

    if kind == FeatureRef.kind:
        return FeatureRef.from_dict(value)
    if kind == FeatureLayerRef.kind:
        return FeatureLayerRef.from_dict(value)
    if kind == ArtifactLabelRef.kind:
        return ArtifactLabelRef.from_dict(value)
    if kind == CaptureArtifact.kind:
        return CaptureArtifact.from_dict(value)
    if kind == OperationArtifact.kind:
        return OperationArtifact.from_dict(value)
    if kind == StepRef.kind:
        return StepRef.from_dict(value)
    if kind == StepFeatureRef.kind:
        return StepFeatureRef.from_dict(value)
    if kind == StepLabelRef.kind:
        return StepLabelRef.from_dict(value)
    return {str(key): spec_value_from_dict(item) for key, item in value.items()}


def row_selector_from_dict(value: Any) -> Any:
    if isinstance(value, list):
        return [row_selector_from_dict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(row_selector_from_dict(item) for item in value)
    if not isinstance(value, dict):
        return value
    if "examples" in value or "source" in value:
        return Dataset.from_dict(value)
    return spec_value_from_dict(value)
