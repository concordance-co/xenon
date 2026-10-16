"""Target-specific immutable Python environments; optional stacks stay separate."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RuntimeLock:
    name: str
    target: str
    python_version: str
    requirements: str
    imports: tuple[str, ...]
    base_image: str
    cuda: str | None = None
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported runtime lock schema_version")
        if not self.name or not self.target or not self.base_image or not re.fullmatch(r"3\.\d+(?:\.\d+)?", self.python_version):
            raise ValueError("Runtime lock needs name, target, Python version and base image")
        if not self.packages:
            raise ValueError("Runtime lock needs exact transitive package pins")

    @property
    def packages(self) -> dict[str, str]:
        packages = {}
        for line in self.requirements.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
            if not match:
                raise ValueError("Runtime lock requires exact name==version pins; no editable, URL or loose requirements")
            name = re.sub(r"[-_.]+", "-", match[1]).lower()
            if name in packages:
                raise ValueError(f"Duplicate locked distribution: {name}")
            packages[name] = match[2]
        return packages

    @property
    def requirements_digest(self) -> str:
        return hashlib.sha256(self.requirements.encode()).hexdigest()

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "name": self.name, "target": self.target,
                "python_version": self.python_version, "requirements": self.requirements,
                "imports": list(self.imports), "base_image": self.base_image, "cuda": self.cuda}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeLock":
        allowed = {"schema_version", "name", "target", "python_version", "requirements", "imports", "base_image", "cuda"}
        if set(value) - allowed:
            raise ValueError("Unknown runtime lock fields")
        return cls(**{**value, "imports": tuple(value["imports"])})

    def verify_target(self, *, system: str | None = None, machine: str | None = None,
                      python_version: str | None = None) -> tuple[str, ...]:
        target = f"{(system or platform.system()).lower()}-{(machine or platform.machine()).lower()}"
        expected = self.target.removesuffix("-cuda").removesuffix("-cpu")
        errors = []
        if target != expected:
            errors.append(f"Runtime lock target {self.target} does not match {target}")
        version = python_version or platform.python_version()
        if not (version == self.python_version or version.startswith(self.python_version + ".")):
            errors.append(f"Runtime lock Python {self.python_version} does not match {version}")
        return tuple(errors)


def load_runtime_lock(name: str) -> RuntimeLock:
    if not re.fullmatch(r"[a-z0-9_-]+", name):
        raise ValueError("Runtime lock name must identify a bundled lock")
    path = Path(__file__).with_name("locks") / f"{name}.json"
    return RuntimeLock.from_dict(json.loads(path.read_text()))


def merge_runtime_specs(specs):
    """Preserve lock/build facts and reject incompatible batching before submit."""
    from dataclasses import replace
    specs = tuple(specs)
    if not specs:
        raise ValueError("Cannot merge an empty runtime set")
    if len({s.python_version for s in specs}) != 1:
        raise ValueError("Cannot merge runtimes with different Python versions")
    locks = {s.lock.identity if s.lock is not None else None for s in specs}
    if len(locks) != 1:
        raise ValueError("Cannot merge incompatible runtime locks or locked and legacy runtimes")
    builds = {json.dumps(s.runtime_identity, sort_keys=True) for s in specs}
    if len(builds) != 1:
        raise ValueError("Cannot merge incompatible runtime build identities")
    env = {}
    for spec in specs:
        for key, value in spec.env.items():
            if key in env and env[key] != value:
                raise ValueError(f"Conflicting runtime environment setting: {key}")
            env[key] = value
    def unique(values):
        return tuple(dict.fromkeys(values))
    return replace(specs[0], env=env,
        pip_packages=unique(item for spec in specs for item in spec.pip_packages),
        secrets=unique(item for spec in specs for item in spec.secrets),
        local_python_sources=unique(item for spec in specs for item in spec.local_python_sources))


def core_runtime_lock() -> RuntimeLock:
    return load_runtime_lock('core-local' if platform.system() == 'Darwin' and platform.machine() == 'arm64' else 'core-linux')


def runtime_for_target(runtime, *, target: str):
    """Select bundled alternatives explicitly; never reinterpret an external lock."""
    from dataclasses import replace
    if getattr(runtime, "lock", None) is not None and runtime.lock.name in {'core-local', 'core-linux'}:
        bundled = load_runtime_lock(runtime.lock.name)
        if runtime.lock.identity == bundled.identity:
            return replace(runtime, lock=load_runtime_lock('core-linux' if target == 'modal' else core_runtime_lock().name))
    return runtime


def build_locked_image(modal, runtime):
    """Construct from exact lock bytes. Loose legacy inputs stay visibly legacy."""
    import tempfile
    if getattr(runtime, "kind", None) == "circuit_tracer_tool":
        from pipelines_v2.runtime.circuit_tracer import build_circuit_tracer_image
        return build_circuit_tracer_image(modal, runtime)
    if getattr(runtime, "kind", None) == "external_tool":
        from pipelines_v2.runtime.external_tool import build_external_image
        return build_external_image(modal, runtime)
    if runtime.lock is None:
        image = modal.Image.debian_slim(python_version=runtime.python_version)
        return image.pip_install(*runtime.pip_packages) if runtime.pip_packages else image
    lock = runtime.lock
    if not lock.base_image.startswith('python@sha256:'):
        raise ValueError('Modal runtime requires an immutable Linux base image')
    image = modal.Image.from_registry(lock.base_image)
    # Modal defers reading context files until image hydration. Retain a small,
    # content-addressed recipe file instead of deleting a temporary directory.
    root = Path(tempfile.gettempdir()) / "xenon-runtime-locks"
    root.mkdir(parents=True, exist_ok=True)
    path = root / (lock.requirements_digest + ".txt")
    if not path.exists() or path.read_text() != lock.requirements:
        with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as stream:
            stream.write(lock.requirements)
            temporary = Path(stream.name)
        temporary.replace(path)
    return image.pip_install_from_requirements(str(path), extra_options="--no-deps")
