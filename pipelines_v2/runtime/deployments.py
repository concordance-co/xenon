"""Long-lived deployment specs and runtime context."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from pipelines_v2.core.types import RuntimeSecret, SpecValidationError, stable_hash, to_primitive
from pipelines_v2.engine.base import PythonRuntimeSpec
from pipelines_v2.operations.common._shared import callable_import_ref, load_importable_function
from pipelines_v2.runtime.modal import ModalResources

_LOG = logging.getLogger("pipelines_v2.deployments")


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    """User-defined deployment config passed to the service factory."""

    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", {str(key): value for key, value in dict(self.values).items()})

    def to_dict(self) -> dict[str, Any]:
        return {"values": to_primitive(dict(self.values))}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "DeploymentConfig":
        if payload is None:
            return cls()
        return cls(values=dict(payload.get("values", {})))


@dataclass(frozen=True, slots=True)
class ServiceFactory:
    """Serializable ref to a top-level function that builds a long-lived service."""

    import_path: str
    local_python_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        module_name, _, function_name = str(self.import_path).partition(":")
        if not module_name or not function_name:
            raise SpecValidationError("ServiceFactory import_path must have the form 'module.path:function_name'")
        sources = tuple(str(source).strip() for source in self.local_python_sources if str(source).strip())
        if not sources:
            raise SpecValidationError("ServiceFactory requires explicit local_python_sources")
        object.__setattr__(self, "import_path", f"{module_name}:{function_name}")
        object.__setattr__(self, "local_python_sources", sources)

    @classmethod
    def from_function(
        cls,
        function: Any,
        *,
        local_python_sources: Sequence[str],
    ) -> "ServiceFactory":
        """Create a service factory from an importable top-level function."""

        import_path, sources = callable_import_ref(
            function,
            local_python_sources=local_python_sources,
            label="Service factories",
        )
        return cls(import_path=import_path, local_python_sources=sources)

    def to_dict(self) -> dict[str, Any]:
        return {
            "import_path": self.import_path,
            "local_python_sources": list(self.local_python_sources),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServiceFactory":
        return cls(
            import_path=str(payload["import_path"]),
            local_python_sources=tuple(str(source) for source in payload.get("local_python_sources", ())),
        )

    def build(self, context: "DeploymentContext") -> Any:
        function = load_importable_function(
            self.import_path,
            label="Service factory",
            local_python_sources=self.local_python_sources,
        )
        signature = inspect.signature(function)
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            return function(ctx=context)
        if "ctx" in signature.parameters:
            return function(ctx=context)
        if "context" in signature.parameters:
            return function(context=context)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        required = [parameter for parameter in positional if parameter.default is inspect._empty]
        if len(required) == 0:
            return function()
        if len(required) == 1:
            return function(context)
        raise TypeError(
            f"Service factory {self.import_path!r} must accept zero or one required positional argument, "
            "or a named ctx/context argument"
        )


class EndpointSpec(Protocol):
    """Protocol for backend-neutral deployment endpoint declarations."""

    kind: str

    def to_dict(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class HTTPEndpoint:
    """HTTP endpoint declaration. V1 supports function-style endpoints."""

    label: str = "api"
    handler: str = ""
    method: str = "POST"
    mode: str = "function"
    kind: str = "http"

    def __post_init__(self) -> None:
        label = str(self.label).strip()
        if not label:
            raise SpecValidationError("HTTPEndpoint label cannot be empty")
        mode = str(self.mode).strip().lower()
        if mode not in {"function", "asgi"}:
            raise SpecValidationError("HTTPEndpoint mode must be one of {'function', 'asgi'}")
        handler = str(self.handler).strip()
        if mode == "function" and not handler:
            raise SpecValidationError("HTTPEndpoint function mode requires a handler")
        method = str(self.method).strip().upper()
        if method not in {"GET", "POST"}:
            raise SpecValidationError("HTTPEndpoint v1 only supports GET and POST")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "handler", handler)
        object.__setattr__(self, "method", method)

    @classmethod
    def asgi(cls, *, label: str = "api") -> "HTTPEndpoint":
        """Deprecated: ASGI endpoints are not used by the Modal deployment adapter."""

        return cls(label=label, mode="asgi")

    @classmethod
    def function(cls, *, label: str, handler: str, method: str = "POST") -> "HTTPEndpoint":
        return cls(label=label, handler=handler, method=method, mode="function")

    @classmethod
    def get(cls, *, label: str, handler: str) -> "HTTPEndpoint":
        return cls.function(label=label, handler=handler, method="GET")

    @classmethod
    def post(cls, *, label: str, handler: str) -> "HTTPEndpoint":
        return cls.function(label=label, handler=handler, method="POST")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "mode": self.mode,
            "label": self.label,
            "handler": self.handler,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HTTPEndpoint":
        kind = str(payload.get("kind") or "")
        if kind != "http":
            raise SpecValidationError(f"Unsupported endpoint kind: {kind!r}")
        return cls(
            label=str(payload.get("label", "api")),
            handler=str(payload.get("handler", "")),
            method=str(payload.get("method", "POST")),
            mode=str(payload.get("mode", "function")),
        )


def endpoint_spec_from_dict(payload: Mapping[str, Any]) -> EndpointSpec:
    kind = str(payload.get("kind") or "")
    if kind == "http":
        return HTTPEndpoint.from_dict(payload)
    raise SpecValidationError(f"Unsupported endpoint kind: {kind!r}")


@dataclass(frozen=True, slots=True)
class DeploymentSpec:
    """Backend-neutral description of one long-lived service deployment."""

    name: str
    service: ServiceFactory
    runtime: PythonRuntimeSpec
    endpoints: tuple[EndpointSpec, ...]
    config: DeploymentConfig = field(default_factory=DeploymentConfig)
    schema_version: int = 1

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise SpecValidationError("DeploymentSpec name cannot be empty")
        endpoints = tuple(self.endpoints)
        if not endpoints:
            raise SpecValidationError("DeploymentSpec requires at least one endpoint")
        if not isinstance(self.runtime, PythonRuntimeSpec):
            raise SpecValidationError("DeploymentSpec runtime must be a PythonRuntimeSpec")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "endpoints", endpoints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "deployment",
            "schema_version": int(self.schema_version),
            "name": self.name,
            "service": self.service.to_dict(),
            "runtime": python_runtime_spec_to_dict(self.runtime),
            "endpoints": [endpoint.to_dict() for endpoint in self.endpoints],
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeploymentSpec":
        kind = str(payload.get("kind") or "deployment")
        if kind != "deployment":
            raise SpecValidationError(f"DeploymentSpec expected kind 'deployment', got {kind!r}")
        return cls(
            name=str(payload["name"]),
            service=ServiceFactory.from_dict(dict(payload["service"])),
            runtime=python_runtime_spec_from_dict(dict(payload["runtime"])),
            endpoints=tuple(endpoint_spec_from_dict(dict(item)) for item in payload.get("endpoints", ())),
            config=DeploymentConfig.from_dict(payload.get("config")),
            schema_version=int(payload.get("schema_version", 1)),
        )

    def semantic_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class DeploymentTargetSpec:
    """Base class for deployment target profiles."""

    kind: str = "deployment_target"

    @classmethod
    def modal(
        cls,
        *,
        app_name: str,
        resources: ModalResources,
        min_containers: int | None = None,
        buffer_containers: int | None = None,
        startup_timeout_seconds: int | None = None,
        scaledown_window_seconds: int | None = None,
    ) -> "ModalDeploymentTargetSpec":
        return ModalDeploymentTargetSpec(
            app_name=app_name,
            resources=resources,
            min_containers=min_containers,
            buffer_containers=buffer_containers,
            startup_timeout_seconds=startup_timeout_seconds,
            scaledown_window_seconds=scaledown_window_seconds,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeploymentTargetSpec":
        kind = str(payload.get("kind") or "")
        if kind == "modal":
            return ModalDeploymentTargetSpec.from_dict(payload)
        raise SpecValidationError(f"Unsupported deployment target kind: {kind!r}")


@dataclass(frozen=True, slots=True)
class ModalDeploymentTargetSpec(DeploymentTargetSpec):
    """Modal target adapter configuration for one deployment."""

    app_name: str = ""
    resources: ModalResources = field(default_factory=ModalResources)
    min_containers: int | None = None
    buffer_containers: int | None = None
    startup_timeout_seconds: int | None = None
    scaledown_window_seconds: int | None = None
    kind: str = "modal"

    def __post_init__(self) -> None:
        app_name = str(self.app_name).strip()
        if not app_name:
            raise SpecValidationError("Modal deployment target requires app_name")
        object.__setattr__(self, "app_name", app_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "app_name": self.app_name,
            "resources": self.resources.to_dict(),
            "min_containers": self.min_containers,
            "buffer_containers": self.buffer_containers,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "scaledown_window_seconds": self.scaledown_window_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModalDeploymentTargetSpec":
        return cls(
            app_name=str(payload["app_name"]),
            resources=ModalResources.from_dict(dict(payload.get("resources", {}))),
            min_containers=_optional_int(payload.get("min_containers")),
            buffer_containers=_optional_int(payload.get("buffer_containers")),
            startup_timeout_seconds=_optional_int(payload.get("startup_timeout_seconds")),
            scaledown_window_seconds=_optional_int(payload.get("scaledown_window_seconds")),
        )


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    """Preflight result for one deployment target."""

    deployment_name: str
    target_name: str
    target_kind: str
    app_name: str | None
    endpoints: tuple[dict[str, Any], ...]
    runtime: dict[str, Any]
    resources: dict[str, Any]
    source_mounts: tuple[str, ...]
    checks: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors

    def validate(self) -> "DeploymentPlan":
        if self.errors:
            raise SpecValidationError("; ".join(self.errors))
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment": self.deployment_name,
            "target": self.target_name,
            "target_kind": self.target_kind,
            "app_name": self.app_name,
            "valid": self.valid,
            "endpoints": [dict(endpoint) for endpoint in self.endpoints],
            "runtime": dict(self.runtime),
            "resources": dict(self.resources),
            "source_mounts": list(self.source_mounts),
            "checks": list(self.checks),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class DeploymentHandle:
    """Structured result from a deployment controller action."""

    deployment_name: str
    target_name: str
    target_kind: str
    status: str
    app_name: str | None = None
    app_id: str | None = None
    url: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment": self.deployment_name,
            "target": self.target_name,
            "target_kind": self.target_kind,
            "status": self.status,
            "app_name": self.app_name,
            "app_id": self.app_id,
            "url": self.url,
            "metadata": dict(self.metadata),
        }


class DeploymentController(Protocol):
    """Controller contract for deployment backends."""

    def plan(self, deployment: DeploymentSpec, target: DeploymentTargetSpec, *, target_name: str) -> DeploymentPlan:
        ...

    def serve(self, deployment: DeploymentSpec, target: DeploymentTargetSpec, *, target_name: str) -> DeploymentHandle:
        ...

    def deploy(self, deployment: DeploymentSpec, target: DeploymentTargetSpec, *, target_name: str) -> DeploymentHandle:
        ...

    def status(self, deployment: DeploymentSpec, target: DeploymentTargetSpec, *, target_name: str) -> DeploymentHandle:
        ...

    def stop(self, deployment: DeploymentSpec, target: DeploymentTargetSpec, *, target_name: str) -> DeploymentHandle:
        ...


class DeploymentContext:
    """Container-local context passed into a service factory."""

    def __init__(
        self,
        *,
        name: str,
        target_name: str,
        config: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.name = str(name)
        self.target_name = str(target_name)
        self.config = dict(config or {})
        self.env = {str(key): str(value) for key, value in dict(env or {}).items()}
        self._singletons: dict[str, Any] = {}
        self._closers: list[Any] = []

    def singleton(self, key: str, factory: Any) -> Any:
        key = str(key)
        if key not in self._singletons:
            value = factory()
            self._singletons[key] = value
            self._register_closer(value)
        return self._singletons[key]

    def vllm_session(self, key: str, *, engine: Any, specs: Sequence[Any]) -> Any:
        def build() -> Any:
            from pipelines_v2.engine.vllm.session import build_vllm_session_runtime

            return build_vllm_session_runtime(engine=engine, specs=tuple(specs))

        return self.singleton(f"vllm_session:{key}", build)

    def vllm_intervention_runtime(self, key: str, *, engine: Any, spec: Any) -> Any:
        def build() -> Any:
            from pipelines_v2.engine.vllm.intervene import build_vllm_intervention_runtime

            return build_vllm_intervention_runtime(engine=engine, spec=spec)

        return self.singleton(f"vllm_intervention:{key}", build)

    def close(self) -> None:
        for value in reversed(self._closers):
            close = getattr(value, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception:
                _LOG.exception("deployment context close failed for %r", value)
        self._closers.clear()
        self._singletons.clear()

    def _register_closer(self, value: Any) -> None:
        if callable(getattr(value, "close", None)):
            self._closers.append(value)


def deployment_target_from_dict(payload: Mapping[str, Any]) -> DeploymentTargetSpec:
    return DeploymentTargetSpec.from_dict(payload)


def controller_for_target(target: DeploymentTargetSpec) -> DeploymentController:
    if isinstance(target, ModalDeploymentTargetSpec) or getattr(target, "kind", None) == "modal":
        from pipelines_v2.runtime.modal_deployments import ModalDeploymentController

        return ModalDeploymentController()
    raise SpecValidationError(f"Unsupported deployment target kind: {getattr(target, 'kind', None)!r}")


def python_runtime_spec_to_dict(runtime: PythonRuntimeSpec) -> dict[str, Any]:
    return {
        "kind": runtime.kind,
        "python_version": runtime.python_version,
        "pip_packages": list(runtime.pip_packages),
        "env": dict(runtime.env),
        "secrets": [secret.to_dict() for secret in runtime.secrets],
        "local_python_sources": list(runtime.local_python_sources),
    }


def python_runtime_spec_from_dict(payload: Mapping[str, Any]) -> PythonRuntimeSpec:
    kind = str(payload.get("kind") or "python")
    if kind != "python":
        raise SpecValidationError(f"Unsupported deployment runtime kind: {kind!r}")
    return PythonRuntimeSpec(
        python_version=str(payload.get("python_version") or "3.13"),
        pip_packages=tuple(str(item) for item in payload.get("pip_packages", ())),
        env={str(key): str(value) for key, value in dict(payload.get("env", {})).items()},
        secrets=tuple(RuntimeSecret.from_dict(dict(item)) for item in payload.get("secrets", ())),
        local_python_sources=tuple(str(item) for item in payload.get("local_python_sources", ())),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "DeploymentConfig",
    "DeploymentContext",
    "DeploymentController",
    "DeploymentHandle",
    "DeploymentPlan",
    "DeploymentSpec",
    "DeploymentTargetSpec",
    "EndpointSpec",
    "HTTPEndpoint",
    "ModalDeploymentTargetSpec",
    "ServiceFactory",
    "controller_for_target",
    "deployment_target_from_dict",
    "endpoint_spec_from_dict",
    "python_runtime_spec_from_dict",
    "python_runtime_spec_to_dict",
]
