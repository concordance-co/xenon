from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from pipelines_v2.api import (
    DeploymentConfig,
    DeploymentContext,
    DeploymentSpec,
    DeploymentTargetSpec,
    HTTPEndpoint,
    ModalResources,
    ModalSecret,
    ModalVolumeMount,
    PythonRuntimeSpec,
    ServiceFactory,
    SpecValidationError,
)
from pipelines_v2.cli import main as pipelines_v2_cli_main
from pipelines_v2.runtime.deployments import DeploymentTargetSpec as RuntimeDeploymentTargetSpec
from pipelines_v2.runtime.modal_deployments import ModalDeploymentController, build_modal_app


def _test_create_app(ctx: DeploymentContext) -> dict[str, str]:
    return {"deployment": ctx.name}


class _TestService:
    def __init__(self, ctx: DeploymentContext) -> None:
        self.ctx = ctx

    def health(self) -> dict[str, str]:
        return {"deployment": self.ctx.name}

    def echo(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"deployment": self.ctx.name, "payload": payload}


def _test_create_service(ctx: DeploymentContext) -> _TestService:
    return _TestService(ctx)


@pytest.mark.unit
def test_deployment_spec_roundtrips() -> None:
    spec = DeploymentSpec(
        name="test-service",
        service=ServiceFactory.from_function(_test_create_app, local_python_sources=("tests",)),
        runtime=PythonRuntimeSpec(
            python_version="3.13",
            pip_packages=("fastapi",),
            env={"A": "B"},
            local_python_sources=("pipelines_v2",),
        ),
        endpoints=(HTTPEndpoint.post(label="echo", handler="echo"),),
        config=DeploymentConfig(values={"x": 1}),
    )

    restored = DeploymentSpec.from_dict(spec.to_dict())

    assert restored.to_dict() == spec.to_dict()
    assert restored.semantic_hash() == spec.semantic_hash()


@pytest.mark.unit
def test_service_factory_requires_explicit_sources() -> None:
    with pytest.raises(TypeError):
        ServiceFactory.from_function(_test_create_app)  # type: ignore[call-arg]
    with pytest.raises(SpecValidationError, match="explicit local_python_sources"):
        ServiceFactory(import_path="tests.example:create_app", local_python_sources=())


@pytest.mark.unit
def test_modal_deployment_target_roundtrips() -> None:
    target = DeploymentTargetSpec.modal(
        app_name="test-app",
        resources=ModalResources(
            gpu="L4",
            cpu=2,
            memory_mb=4096,
            timeout_seconds=60,
            max_containers=2,
            secrets=(ModalSecret.from_env_var("HF_TOKEN", secret_name="huggingface"),),
            volumes=(ModalVolumeMount(name="models", mount_path="/models", create_if_missing=True),),
        ),
        min_containers=1,
        startup_timeout_seconds=120,
        scaledown_window_seconds=900,
    )

    restored = RuntimeDeploymentTargetSpec.from_dict(target.to_dict())

    assert restored.to_dict() == target.to_dict()


@pytest.mark.unit
def test_deployment_context_singleton_and_close() -> None:
    closed: list[str] = []

    class Resource:
        def close(self) -> None:
            closed.append("closed")

    ctx = DeploymentContext(name="svc", target_name="dev", config={"a": 1}, env={"B": "C"})

    first = ctx.singleton("resource", Resource)
    second = ctx.singleton("resource", Resource)
    ctx.close()

    assert first is second
    assert closed == ["closed"]
    assert ctx.config == {"a": 1}
    assert ctx.env == {"B": "C"}


@pytest.mark.unit
def test_modal_controller_plan_reports_sources_resources_and_lifecycle() -> None:
    spec = _deployment_spec()
    target = _modal_target()

    plan = ModalDeploymentController().plan(spec, target, target_name="prod")

    assert plan.valid
    payload = plan.to_dict()
    assert payload["deployment"] == "test-service"
    assert payload["target"] == "prod"
    assert payload["app_name"] == "test-service-prod"
    assert payload["endpoints"] == [
        {"kind": "http", "label": "echo", "mode": "function", "handler": "echo", "method": "POST"}
    ]
    assert payload["resources"]["gpu"] == "L4"
    assert payload["resources"]["min_containers"] == 1
    assert payload["resources"]["scaledown_window_seconds"] == 900
    assert payload["source_mounts"] == ["pipelines_v2", "tests"]


@pytest.mark.unit
def test_modal_build_app_threads_runtime_and_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_modal = _install_fake_modal(monkeypatch)
    spec = _deployment_spec()
    target = _modal_target()

    app = build_modal_app(spec, target, target_name="prod")

    assert app.name == "test-service-prod"
    assert app.function_kwargs["gpu"] == "L4"
    assert app.function_kwargs["min_containers"] == 1
    assert app.function_kwargs["scaledown_window"] == 900
    assert app.function_kwargs["volumes"] == {"/models": ("models", True)}
    assert app.function_kwargs["secrets"] == ["huggingface"]
    assert "serialized" not in app.function_kwargs
    assert app.function_kwargs["name"] == "deployment_echo"
    assert fake_modal.images[-1].env_payload["A"] == "target"
    assert fake_modal.images[-1].env_payload["PIPELINES_V2_DEPLOYMENT_ENDPOINT_HANDLER"] == "echo"
    assert fake_modal.images[-1].local_dirs
    assert getattr(app.deployed_function, "_modal_fastapi_label") == "echo"
    assert getattr(app.deployed_function, "_modal_fastapi_method") == "POST"


@pytest.mark.unit
def test_cli_deployment_plan_and_stop_confirmation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    deployment_file = tmp_path / "deployment.py"
    deployment_file.write_text(
        "\n".join(
            [
                "from pipelines_v2.api import DeploymentSpec, DeploymentTargetSpec, HTTPEndpoint, ModalResources, PythonRuntimeSpec, ServiceFactory",
                "",
                "def build_deployment():",
                "    return DeploymentSpec(",
                "        name='cli-service',",
                "        service=ServiceFactory(import_path='tests.pipelines_v2.runtime.test_deployments:_test_create_service', local_python_sources=('tests',)),",
                "        runtime=PythonRuntimeSpec(local_python_sources=('pipelines_v2',)),",
                "        endpoints=(HTTPEndpoint.post(label='echo', handler='echo'),),",
                "    )",
                "",
                "def build_deployment_targets():",
                "    return {'prod': DeploymentTargetSpec.modal(app_name='cli-service-prod', resources=ModalResources(cpu=1))}",
            ]
        ),
        encoding="utf-8",
    )

    assert pipelines_v2_cli_main(["deployment", "plan", "--file", str(deployment_file), "--target", "prod"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deployment"] == "cli-service"
    assert payload["target"] == "prod"

    with pytest.raises(RuntimeError, match="requires --yes"):
        pipelines_v2_cli_main(["deployment", "stop", "--file", str(deployment_file), "--target", "prod"])


@pytest.mark.unit
def test_assistant_axis_deployment_plans_without_modal_boilerplate() -> None:
    from papers.voice.assistant_axis.deployment import build_deployment, build_deployment_targets

    deployment = build_deployment()
    target = build_deployment_targets()["prod"]
    plan = ModalDeploymentController().plan(deployment, target, target_name="prod")

    assert plan.valid
    assert deployment.service.import_path == "papers.voice.assistant_axis.service_runtime:create_app"
    assert "papers/voice/assistant_axis" in deployment.service.local_python_sources
    assert deployment.runtime.env["ASSISTANT_AXIS_SERVICE_GPU_MEMORY_UTILIZATION"] == "0.99"
    assert deployment.runtime.env["XENON_ACTIVATION_PATCH_MAX_TOKENS"] == "128"
    assert [(endpoint.label, endpoint.mode) for endpoint in deployment.endpoints] == [("api", "asgi")]


def _deployment_spec() -> DeploymentSpec:
    return DeploymentSpec(
        name="test-service",
        service=ServiceFactory.from_function(_test_create_service, local_python_sources=("tests",)),
        runtime=PythonRuntimeSpec(
            python_version="3.13",
            pip_packages=("fastapi",),
            env={"A": "runtime"},
            local_python_sources=("pipelines_v2",),
        ),
        endpoints=(HTTPEndpoint.post(label="echo", handler="echo"),),
    )


def _modal_target() -> Any:
    return DeploymentTargetSpec.modal(
        app_name="test-service-prod",
        resources=ModalResources(
            gpu="L4",
            env={"A": "target"},
            secrets=(ModalSecret.from_env_var("HF_TOKEN", secret_name="huggingface"),),
            volumes=(ModalVolumeMount(name="models", mount_path="/models", create_if_missing=True),),
        ),
        min_containers=1,
        scaledown_window_seconds=900,
    )


def _install_fake_modal(monkeypatch: pytest.MonkeyPatch) -> Any:
    fake = types.SimpleNamespace(images=[])

    class FakeImage:
        def __init__(self, python_version: str) -> None:
            self.python_version = python_version
            self.packages: tuple[str, ...] = ()
            self.env_payload: dict[str, str] = {}
            self.local_dirs: list[tuple[str, str]] = []

        @classmethod
        def debian_slim(cls, *, python_version: str) -> "FakeImage":
            image = cls(python_version)
            fake.images.append(image)
            return image

        def pip_install(self, *packages: str) -> "FakeImage":
            self.packages = tuple(packages)
            return self

        def env(self, payload: dict[str, str]) -> "FakeImage":
            self.env_payload.update(payload)
            return self

        def add_local_dir(self, local_path: str, *, remote_path: str) -> "FakeImage":
            self.local_dirs.append((local_path, remote_path))
            return self

    class FakeSecret:
        @staticmethod
        def from_name(name: str) -> str:
            return name

    class FakeVolume:
        @staticmethod
        def from_name(name: str, *, create_if_missing: bool = False) -> tuple[str, bool]:
            return name, create_if_missing

    class FakeApp:
        def __init__(self, name: str) -> None:
            self.name = name
            self.function_kwargs: dict[str, Any] = {}
            self.deployed_function: Any | None = None

        def function(self, **kwargs: Any) -> Any:
            self.function_kwargs = kwargs

            def decorator(value: Any) -> Any:
                self.deployed_function = value
                return value

            return decorator

        def deploy(self) -> Any:
            return types.SimpleNamespace(app_id="ap-test")

    def lifecycle_decorator() -> Any:
        def decorator(value: Any) -> Any:
            return value

        return decorator

    def fastapi_endpoint(*, method: str, label: str) -> Any:
        def decorator(value: Any) -> Any:
            setattr(value, "_modal_fastapi_label", label)
            setattr(value, "_modal_fastapi_method", method)
            return value

        return decorator

    def asgi_app(*, label: str) -> Any:
        def decorator(value: Any) -> Any:
            setattr(value, "_modal_asgi_label", label)
            return value

        return decorator

    fake.App = FakeApp
    fake.Image = FakeImage
    fake.Secret = FakeSecret
    fake.Volume = FakeVolume
    fake.enter = lifecycle_decorator
    fake.exit = lifecycle_decorator
    fake.fastapi_endpoint = fastapi_endpoint
    fake.asgi_app = asgi_app
    monkeypatch.setitem(sys.modules, "modal", fake)
    return fake
