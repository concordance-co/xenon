"""Serializable function-backed builder specs."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from pipelines_v2.core.types import SpecValidationError

from ._shared import callable_import_ref, load_importable_function


@dataclass(frozen=True, slots=True)
class PromptMetadataBuilder:
    """Serializable ref to a user-defined function that derives prompt metadata."""

    import_path: str
    local_python_sources: tuple[str, ...] = field(default_factory=tuple)
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        module_name, _, function_name = str(self.import_path).partition(":")
        if not module_name or not function_name:
            raise SpecValidationError(
                "PromptMetadataBuilder import_path must have the form 'module.path:function_name'"
            )
        object.__setattr__(self, "import_path", f"{module_name}:{function_name}")
        object.__setattr__(
            self,
            "local_python_sources",
            tuple(str(source) for source in self.local_python_sources if str(source).strip()),
        )
        object.__setattr__(self, "config", {str(key): value for key, value in dict(self.config).items()})

    @classmethod
    def from_function(
        cls,
        function: Any,
        *,
        local_python_sources: Sequence[str] | None = None,
    ) -> "PromptMetadataBuilder":
        import_path, sources = callable_import_ref(
            function,
            local_python_sources=local_python_sources,
            label="Prompt metadata builders",
        )
        return cls(import_path=import_path, local_python_sources=sources)

    @classmethod
    def chat_turns(
        cls,
        *,
        name_template: str = "{role}_turn_{index:03d}",
        include_section_records: bool = True,
    ) -> "PromptMetadataBuilder":
        return cls(
            import_path="pipelines_v2.engine.prompt_metadata:build_chat_turn_metadata",
            config={
                "name_template": str(name_template),
                "include_section_records": bool(include_section_records),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "import_path": self.import_path,
            "local_python_sources": list(self.local_python_sources),
            "config": dict(self.config),
        }

    def semantic_dict(self) -> dict[str, Any]:
        return {"import_path": self.import_path, "config": dict(self.config)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromptMetadataBuilder":
        return cls(
            import_path=str(payload["import_path"]),
            local_python_sources=tuple(str(source) for source in payload.get("local_python_sources", ())),
            config={str(key): value for key, value in dict(payload.get("config", {})).items()},
        )

    def build(self, rendered_prompt: str, **context: Any) -> dict[str, Any]:
        function = load_importable_function(
            self.import_path,
            label="Prompt metadata builder",
            local_python_sources=self.local_python_sources,
        )
        kwargs = {"rendered_prompt": rendered_prompt, **dict(self.config), **context}
        signature = inspect.signature(function)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_var_kwargs:
            payload = function(**kwargs)
        else:
            payload = function(
                **{
                    name: value
                    for name, value in kwargs.items()
                    if name in signature.parameters
                }
            )
        if not isinstance(payload, Mapping):
            raise TypeError(
                "Prompt metadata builder must return a mapping, "
                f"got {type(payload).__name__}"
            )
        return dict(payload)


@dataclass(frozen=True, slots=True)
class TransformResult:
    """Structured return value for ``TransformSpec`` builders."""

    payload: Mapping[str, Any] = field(default_factory=dict)
    labels: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    example_keys: Sequence[str] | None = None


@dataclass(frozen=True, slots=True)
class TransformBuilder:
    """Serializable ref to a user-defined transform function."""

    import_path: str
    local_python_sources: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        module_name, _, function_name = str(self.import_path).partition(":")
        if not module_name or not function_name:
            raise SpecValidationError(
                "TransformBuilder import_path must have the form 'module.path:function_name'"
            )
        object.__setattr__(self, "import_path", f"{module_name}:{function_name}")
        object.__setattr__(
            self,
            "local_python_sources",
            tuple(str(source) for source in self.local_python_sources if str(source).strip()),
        )

    @classmethod
    def from_function(
        cls,
        function: Any,
        *,
        local_python_sources: Sequence[str] | None = None,
    ) -> "TransformBuilder":
        import_path, sources = callable_import_ref(
            function,
            local_python_sources=local_python_sources,
            label="Transform builders",
        )
        return cls(import_path=import_path, local_python_sources=sources)

    def to_dict(self) -> dict[str, Any]:
        return {
            "import_path": self.import_path,
            "local_python_sources": list(self.local_python_sources),
        }

    def semantic_dict(self) -> dict[str, Any]:
        return {"import_path": self.import_path}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransformBuilder":
        return cls(
            import_path=str(payload["import_path"]),
            local_python_sources=tuple(str(source) for source in payload.get("local_python_sources", ())),
        )

    def build(self, inputs: Mapping[str, Any]) -> TransformResult | Mapping[str, Any]:
        function = load_importable_function(
            self.import_path,
            label="Transform builder",
            local_python_sources=self.local_python_sources,
        )
        return function(**dict(inputs))
