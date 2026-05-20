"""Modal adapter for backend-neutral deployment specs."""

from __future__ import annotations

import atexit
import json
import logging
import os
from typing import Any

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.runtime.deployments import (
    DeploymentHandle,
    DeploymentPlan,
    DeploymentSpec,
    DeploymentTargetSpec,
    ModalDeploymentTargetSpec,
    python_runtime_spec_to_dict,
)
from pipelines_v2.runtime.env import merged_runtime_env
from pipelines_v2.runtime.modal_worker import _resolved_local_python_sources

_LOG = logging.getLogger("pipelines_v2.modal_deployments")
_MODAL_DEPLOYMENT_STATE: dict[str, dict[str, Any]] = {}


def _modal_deployment_get_endpoint() -> Any:
    return _dispatch_modal_deployment_endpoint(None)


def _modal_deployment_post_endpoint(item: dict[str, Any]) -> Any:
    return _dispatch_modal_deployment_endpoint(item)


def _modal_deployment_asgi_app() -> Any:
    return _modal_deployment_service()


def _dispatch_modal_deployment_endpoint(payload: dict[str, Any] | None) -> Any:
    service = _modal_deployment_service()
    handler_name = os.environ["PIPELINES_V2_DEPLOYMENT_ENDPOINT_HANDLER"]
    target_handler = getattr(service, handler_name, None)
    if not callable(target_handler):
        raise AttributeError(f"Deployment service has no callable handler {handler_name!r}")
    if payload is None:
        return target_handler()
    return target_handler(payload)


def _modal_deployment_service() -> Any:
    from pipelines_v2.runtime.deployments import DeploymentContext, DeploymentSpec

    deployment_payload = json.loads(os.environ["PIPELINES_V2_DEPLOYMENT_SPEC_JSON"])
    runtime_env = json.loads(os.environ.get("PIPELINES_V2_DEPLOYMENT_RUNTIME_ENV_JSON", "{}"))
    target_name = os.environ["PIPELINES_V2_DEPLOYMENT_TARGET_NAME"]
    state_key = f"{deployment_payload['name']}:{target_name}"

    if state_key not in _MODAL_DEPLOYMENT_STATE:
        deployment = DeploymentSpec.from_dict(deployment_payload)
        context = DeploymentContext(
            name=deployment.name,
            target_name=target_name,
            config=deployment.config.values,
            env=runtime_env,
        )
        service = deployment.service.build(context)
        _MODAL_DEPLOYMENT_STATE[state_key] = {"context": context, "service": service}
        atexit.register(context.close)

    return _MODAL_DEPLOYMENT_STATE[state_key]["service"]


class ModalDeploymentController:
    """Deployment controller that maps runtime deployment specs onto Modal apps."""

    kind = "modal"

    def plan(
        self,
        deployment: DeploymentSpec,
        target: DeploymentTargetSpec,
        *,
        target_name: str,
    ) -> DeploymentPlan:
        modal_target = _require_modal_target(target)
        errors = _plan_errors(deployment=deployment, target=modal_target)
        runtime_env = merged_runtime_env(deployment.runtime.env, modal_target.resources.env)
        sources = _deployment_sources(deployment)
        return DeploymentPlan(
            deployment_name=deployment.name,
            target_name=target_name,
            target_kind=modal_target.kind,
            app_name=modal_target.app_name,
            endpoints=tuple(endpoint.to_dict() for endpoint in deployment.endpoints),
            runtime={
                **python_runtime_spec_to_dict(deployment.runtime),
                "env": runtime_env,
            },
            resources={
                **modal_target.resources.to_dict(),
                "min_containers": modal_target.min_containers,
                "buffer_containers": modal_target.buffer_containers,
                "startup_timeout_seconds": modal_target.startup_timeout_seconds,
                "scaledown_window_seconds": modal_target.scaledown_window_seconds,
            },
            source_mounts=sources,
            checks=("runtime", "service_factory", "endpoints", "modal_resources"),
            errors=tuple(errors),
        )

    def serve(
        self,
        deployment: DeploymentSpec,
        target: DeploymentTargetSpec,
        *,
        target_name: str,
    ) -> DeploymentHandle:
        modal_target = _require_modal_target(target)
        self.plan(deployment, modal_target, target_name=target_name).validate()
        app = build_modal_app(deployment, modal_target, target_name=target_name)
        _LOG.info("serving Modal deployment app=%s target=%s", modal_target.app_name, target_name)
        if hasattr(app, "serve"):
            result = app.serve()
            return _handle_from_result(
                deployment=deployment,
                target=modal_target,
                target_name=target_name,
                status="served",
                result=result,
            )
        import time

        try:
            with app.run() as running_app:
                _LOG.info("Modal deployment is serving; press Ctrl-C to stop local serve")
                while True:
                    time.sleep(3600)
        except KeyboardInterrupt:
            return _handle_from_result(
                deployment=deployment,
                target=modal_target,
                target_name=target_name,
                status="stopped",
                result=app,
            )

    def deploy(
        self,
        deployment: DeploymentSpec,
        target: DeploymentTargetSpec,
        *,
        target_name: str,
    ) -> DeploymentHandle:
        modal_target = _require_modal_target(target)
        self.plan(deployment, modal_target, target_name=target_name).validate()
        app = build_modal_app(deployment, modal_target, target_name=target_name)
        if not hasattr(app, "deploy"):
            raise RuntimeError("Installed Modal SDK does not expose app.deploy(); use `modal deploy` as a fallback")
        _LOG.info("deploying Modal deployment app=%s target=%s", modal_target.app_name, target_name)
        result = app.deploy()
        return _handle_from_result(
            deployment=deployment,
            target=modal_target,
            target_name=target_name,
            status="deployed",
            result=result,
        )

    def status(
        self,
        deployment: DeploymentSpec,
        target: DeploymentTargetSpec,
        *,
        target_name: str,
    ) -> DeploymentHandle:
        modal_target = _require_modal_target(target)
        try:
            import modal

            lookup = getattr(modal.App, "lookup", None)
            if callable(lookup):
                app = lookup(modal_target.app_name)
                return _handle_from_result(
                    deployment=deployment,
                    target=modal_target,
                    target_name=target_name,
                    status="found",
                    result=app,
                )
        except Exception as exc:
            return DeploymentHandle(
                deployment_name=deployment.name,
                target_name=target_name,
                target_kind=modal_target.kind,
                status="unknown",
                app_name=modal_target.app_name,
                metadata={"error": str(exc), "fallback": f"modal app list | grep {modal_target.app_name}"},
            )
        return DeploymentHandle(
            deployment_name=deployment.name,
            target_name=target_name,
            target_kind=modal_target.kind,
            status="unknown",
            app_name=modal_target.app_name,
            metadata={"fallback": f"modal app list | grep {modal_target.app_name}"},
        )

    def stop(
        self,
        deployment: DeploymentSpec,
        target: DeploymentTargetSpec,
        *,
        target_name: str,
    ) -> DeploymentHandle:
        modal_target = _require_modal_target(target)
        import modal

        lookup = getattr(modal.App, "lookup", None)
        if not callable(lookup):
            raise RuntimeError("Installed Modal SDK does not expose App.lookup(); use `modal app stop` as a fallback")
        app = lookup(modal_target.app_name)
        stop = getattr(app, "stop", None)
        if callable(stop):
            result = stop()
        else:
            import subprocess

            completed = subprocess.run(
                ["modal", "app", "stop", modal_target.app_name],
                check=True,
                capture_output=True,
                text=True,
            )
            del completed
            result = app
        return _handle_from_result(
            deployment=deployment,
            target=modal_target,
            target_name=target_name,
            status="stopped",
            result=result,
        )


def build_modal_app(
    deployment: DeploymentSpec,
    target: ModalDeploymentTargetSpec,
    *,
    target_name: str,
) -> Any:
    """Build a Modal app from a backend-neutral deployment spec."""

    import modal

    runtime_env = merged_runtime_env(deployment.runtime.env, target.resources.env)
    app = modal.App(target.app_name)
    image = modal.Image.debian_slim(python_version=deployment.runtime.python_version)
    if deployment.runtime.pip_packages:
        image = image.pip_install(*deployment.runtime.pip_packages)
    source_mounts, pythonpath_entries = _resolved_local_python_sources(_deployment_sources(deployment))
    if pythonpath_entries:
        existing_pythonpath = runtime_env.get("PYTHONPATH", "")
        combined = [entry for entry in pythonpath_entries if entry]
        if existing_pythonpath:
            combined.append(existing_pythonpath)
        runtime_env["PYTHONPATH"] = ":".join(combined)
    if runtime_env:
        image = image.env(runtime_env)

    resources = target.resources
    function_kwargs: dict[str, Any] = {
        "image": image,
        "secrets": [modal.Secret.from_name(str(secret.name)) for secret in resources.secrets],
        "volumes": {
            volume.mount_path: modal.Volume.from_name(volume.name, create_if_missing=volume.create_if_missing)
            for volume in resources.volumes
        },
        "timeout": int(resources.timeout_seconds or 3600),
    }
    _set_if_not_none(function_kwargs, "gpu", resources.gpu)
    _set_if_not_none(function_kwargs, "cpu", resources.cpu)
    _set_if_not_none(function_kwargs, "memory", resources.memory_mb)
    _set_if_not_none(function_kwargs, "max_containers", resources.max_containers)
    _set_if_not_none(function_kwargs, "min_containers", target.min_containers)
    _set_if_not_none(function_kwargs, "buffer_containers", target.buffer_containers)
    _set_if_not_none(function_kwargs, "startup_timeout", target.startup_timeout_seconds)
    _set_if_not_none(function_kwargs, "scaledown_window", target.scaledown_window_seconds)

    asgi_endpoint = _asgi_endpoint(deployment)
    if asgi_endpoint is not None:
        _register_asgi_endpoint(
            app=app,
            modal=modal,
            function_kwargs=function_kwargs,
            image=image,
            source_mounts=source_mounts,
            deployment=deployment,
            runtime_env=runtime_env,
            target_name=target_name,
            endpoint=asgi_endpoint,
        )
    else:
        for endpoint in _function_endpoints(deployment):
            _register_function_endpoint(
                app=app,
                modal=modal,
                function_kwargs=function_kwargs,
                image=image,
                source_mounts=source_mounts,
                deployment=deployment,
                runtime_env=runtime_env,
                target_name=target_name,
                endpoint=endpoint,
            )

    return app


def _plan_errors(*, deployment: DeploymentSpec, target: ModalDeploymentTargetSpec) -> list[str]:
    errors: list[str] = []
    runtime_secrets = {secret.env_var for secret in deployment.runtime.secrets}
    provided = {env_var for secret in target.resources.secrets for env_var in secret.env_vars}
    missing = sorted(runtime_secrets - provided)
    if missing:
        errors.append(f"Modal deployment target is missing runtime secret bindings for env vars: {missing}")
    http_endpoints = [endpoint for endpoint in deployment.endpoints if getattr(endpoint, "kind", None) == "http"]
    if len(http_endpoints) != len(deployment.endpoints):
        errors.append("Modal deployment v1 only supports HTTP endpoints")
    asgi_endpoints = [endpoint for endpoint in http_endpoints if getattr(endpoint, "mode", None) == "asgi"]
    function_endpoints = [endpoint for endpoint in http_endpoints if getattr(endpoint, "mode", None) == "function"]
    unsupported = [
        endpoint for endpoint in http_endpoints if getattr(endpoint, "mode", None) not in {"asgi", "function"}
    ]
    if unsupported:
        errors.append("Modal deployment v1 only supports ASGI or function HTTP endpoints")
    if asgi_endpoints and function_endpoints:
        errors.append("Modal deployment cannot mix ASGI and function HTTP endpoints")
    if len(asgi_endpoints) > 1:
        errors.append("Modal deployment supports at most one ASGI HTTP endpoint")
    if not http_endpoints:
        errors.append("Modal deployment requires at least one HTTP endpoint")
    return errors


def _require_modal_target(target: DeploymentTargetSpec) -> ModalDeploymentTargetSpec:
    if isinstance(target, ModalDeploymentTargetSpec):
        return target
    if getattr(target, "kind", None) == "modal":
        return ModalDeploymentTargetSpec.from_dict(target.to_dict())  # type: ignore[attr-defined]
    raise SpecValidationError(f"Expected Modal deployment target, got {getattr(target, 'kind', None)!r}")


def _deployment_sources(deployment: DeploymentSpec) -> tuple[str, ...]:
    sources: list[str] = []
    for source in deployment.runtime.local_python_sources:
        if source not in sources:
            sources.append(source)
    for source in deployment.service.local_python_sources:
        if source not in sources:
            sources.append(source)
    return tuple(sources)


def _function_endpoints(deployment: DeploymentSpec) -> tuple[Any, ...]:
    return tuple(
        endpoint
        for endpoint in deployment.endpoints
        if getattr(endpoint, "kind", None) == "http" and getattr(endpoint, "mode", None) == "function"
    )


def _asgi_endpoint(deployment: DeploymentSpec) -> Any | None:
    endpoints = tuple(
        endpoint
        for endpoint in deployment.endpoints
        if getattr(endpoint, "kind", None) == "http" and getattr(endpoint, "mode", None) == "asgi"
    )
    if not endpoints:
        return None
    return endpoints[0]


def _register_asgi_endpoint(
    *,
    app: Any,
    modal: Any,
    function_kwargs: dict[str, Any],
    image: Any,
    source_mounts: list[tuple[Any, str]],
    deployment: DeploymentSpec,
    runtime_env: dict[str, str],
    target_name: str,
    endpoint: Any,
) -> None:
    endpoint_env = {
        "PIPELINES_V2_DEPLOYMENT_SPEC_JSON": json.dumps(deployment.to_dict(), sort_keys=True),
        "PIPELINES_V2_DEPLOYMENT_RUNTIME_ENV_JSON": json.dumps(runtime_env, sort_keys=True),
        "PIPELINES_V2_DEPLOYMENT_TARGET_NAME": target_name,
    }
    endpoint_kwargs = dict(function_kwargs)
    endpoint_image = image.env(endpoint_env)
    for local_path, remote_path in source_mounts:
        endpoint_image = endpoint_image.add_local_dir(str(local_path), remote_path=remote_path)
    endpoint_kwargs["image"] = endpoint_image
    endpoint_kwargs["name"] = f"deployment_{str(getattr(endpoint, 'label')).replace('-', '_')}"
    decorated = modal.asgi_app(label=str(getattr(endpoint, "label")))(_modal_deployment_asgi_app)
    app.function(**endpoint_kwargs)(decorated)


def _register_function_endpoint(
    *,
    app: Any,
    modal: Any,
    function_kwargs: dict[str, Any],
    image: Any,
    source_mounts: list[tuple[Any, str]],
    deployment: DeploymentSpec,
    runtime_env: dict[str, str],
    target_name: str,
    endpoint: Any,
) -> None:
    label = str(getattr(endpoint, "label"))
    handler_name = str(getattr(endpoint, "handler"))
    method = str(getattr(endpoint, "method", "POST")).upper()
    endpoint_env = {
        "PIPELINES_V2_DEPLOYMENT_SPEC_JSON": json.dumps(deployment.to_dict(), sort_keys=True),
        "PIPELINES_V2_DEPLOYMENT_RUNTIME_ENV_JSON": json.dumps(runtime_env, sort_keys=True),
        "PIPELINES_V2_DEPLOYMENT_TARGET_NAME": target_name,
        "PIPELINES_V2_DEPLOYMENT_ENDPOINT_HANDLER": handler_name,
    }
    endpoint_kwargs = dict(function_kwargs)
    endpoint_image = image.env(endpoint_env)
    for local_path, remote_path in source_mounts:
        endpoint_image = endpoint_image.add_local_dir(str(local_path), remote_path=remote_path)
    endpoint_kwargs["image"] = endpoint_image
    endpoint_kwargs["name"] = f"deployment_{label.replace('-', '_')}"

    if method == "GET":
        raw_handler = _modal_deployment_get_endpoint
    else:
        from fastapi import Body

        raw_handler = _modal_deployment_post_endpoint
        raw_handler.__signature__ = _post_endpoint_signature(Body)
    decorated = modal.fastapi_endpoint(method=method, label=label)(raw_handler)
    app.function(**endpoint_kwargs)(decorated)


def _post_endpoint_signature(body_factory: Any) -> Any:
    import inspect

    return inspect.Signature(
        parameters=[
            inspect.Parameter(
                "item",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=dict[str, Any],
                default=body_factory(...),
            )
        ],
        return_annotation=Any,
    )


def _set_if_not_none(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value


def _handle_from_result(
    *,
    deployment: DeploymentSpec,
    target: ModalDeploymentTargetSpec,
    target_name: str,
    status: str,
    result: Any,
) -> DeploymentHandle:
    app_id = getattr(result, "app_id", None)
    url = getattr(result, "url", None) or getattr(result, "web_url", None)
    metadata: dict[str, Any] = {}
    for attr in ("object_id", "app_page_url", "web_url"):
        value = getattr(result, attr, None)
        if value is not None:
            metadata[attr] = value
    return DeploymentHandle(
        deployment_name=deployment.name,
        target_name=target_name,
        target_kind=target.kind,
        status=status,
        app_name=target.app_name,
        app_id=str(app_id) if app_id is not None else None,
        url=str(url) if url is not None else None,
        metadata=metadata,
    )


__all__ = ["ModalDeploymentController", "build_modal_app"]
