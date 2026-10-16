"""C02 staged admission. Static declarations never masquerade as measurements."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Mapping

from pipelines_v2.core.provenance import content_hash

_PHASES = {"static", "readiness", "qualification"}
_STATUSES = {"passed", "failed", "unresolved", "not_applicable"}


@dataclass(frozen=True, slots=True)
class AdmissionCheck:
    check_id: str
    phase: str
    status: str
    field_path: str
    message: str
    remediation: str | None = None
    evidence_ref: str | None = None
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or self.phase not in _PHASES or self.status not in _STATUSES:
            raise ValueError("Invalid or unsupported admission check version/phase/status")
        if not self.check_id or not self.field_path or not self.message:
            raise ValueError("Admission check requires id, field path and message")
        if self.phase == "qualification" and self.status == "passed" and not self.evidence_ref:
            raise ValueError("Passed qualification requires measured evidence")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True, slots=True)
class WorkloadEnvelope:
    model_identity: Any
    runtime_lock: str | None
    hardware_topology: Mapping[str, Any]
    dtype: str | None
    attention_impl: str | None
    operation_kind: str
    hook_set: tuple[str, ...]
    max_input_shape: Any
    max_output_shape: Any
    batch_size: int | None
    capture_shape: Any
    backward_or_optimizer: bool
    compile_config: Any
    host_memory_limit: int | None
    device_memory_limit: int | None
    timeout: int | None
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported workload envelope schema")

    def to_dict(self):
        return asdict(self)

    @property
    def identity(self):
        return content_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "hook_set": tuple(value["hook_set"])})


def redact(value: Any) -> Any:
    """Only configuration values, never resolved secret values, reach diagnostics."""
    if isinstance(value, Mapping):
        return {str(key): "[redacted]" if any(part in str(key).lower() for part in ("token", "password", "credential", "secret", "api_key", "authorization"))
                and str(key) not in {"max_tokens", "max_model_len", "max_num_batched_tokens", "tokenizer"}
                else redact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [redact(item) for item in value]
    if isinstance(value, str) and "://" in value:
        from urllib.parse import urlsplit, urlunsplit
        parsed = urlsplit(value)
        return urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, "", ""))
    return value


_NUMERICAL_ENV = frozenset({"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "PYTORCH_ENABLE_MPS_FALLBACK", "VLLM_USE_FLASHINFER_SAMPLER"})


def effective_configuration(spec, runner_identity: Mapping[str, Any]):
    from pipelines_v2.runtime.locks import runtime_for_target
    runtime = runtime_for_target(spec.runtime_spec(), target="modal" if runner_identity.get("kind") == "modal" else "local")
    resource_env = dict(dict(runner_identity.get("resources") or {}).get("env") or {})
    runtime_env = dict(getattr(runtime, "env", {}) or {})
    engine = spec.bound_engine()
    safe_env = {key: value if key in _NUMERICAL_ENV else "[redacted]" for key, value in {**runtime_env, **resource_env}.items()}
    safe_resources = {**dict(runner_identity.get("resources") or {}), "env": {key: safe_env[key] for key in resource_env}}
    return {"engine": {"value": redact(engine.identity()) if engine else None, "origin": "operation.engine"},
            "resources": {"value": redact(safe_resources), "origin": "runner.resources"},
            "environment": {key: {"value": safe_env[key],
                "origin": "runner.resources.env" if key in resource_env else "operation.runtime.env"}
                for key, value in {**runtime_env, **resource_env}.items()},
            "runtime_lock": {"value": runtime.lock.identity if getattr(runtime, "lock", None) else None,
                             "origin": "operation.runtime.lock" if getattr(runtime, "lock", None) else "legacy_compatibility"}}


def workload_envelope(spec, runner_identity: Mapping[str, Any]) -> WorkloadEnvelope:
    if spec.kind in {"diffusion_generate", "diffusion_capture", "diffusion_intervene"}:
        from pipelines_v2.engine.diffusers.readiness import diffusion_envelope
        return diffusion_envelope(spec, runner_identity)
    resources = dict(runner_identity.get("resources") or {})
    engine = spec.bound_engine()
    descriptor = engine.identity() if engine else {}
    extra = dict(descriptor.get("extra") or {})
    generation = getattr(spec, "generation", None)
    from pipelines_v2.runtime.locks import runtime_for_target
    runtime = runtime_for_target(spec.runtime_spec(), target="modal" if runner_identity.get("kind") == "modal" else "local")
    return WorkloadEnvelope(model_identity=descriptor.get("model_id"),
        runtime_lock=runtime.lock.identity if getattr(runtime, "lock", None) else None,
        hardware_topology={"gpu": resources.get("gpu"), "tensor_parallel": descriptor.get("tensor_parallel_size", 1),
                           "pipeline_parallel": descriptor.get("pipeline_parallel_size", 1), "independent_shards": resources.get("shard_count") or 1},
        dtype=descriptor.get("dtype") or extra.get("dtype"), attention_impl=extra.get("attention_backend"),
        operation_kind=spec.kind, hook_set=tuple(sorted(str(getattr(site, "site", type(site).__name__)) for site in getattr(spec, "sites", ()))),
        max_input_shape=descriptor.get("max_model_len"), max_output_shape=getattr(generation, "max_tokens", None),
        batch_size=descriptor.get("max_num_seqs"), capture_shape=None, backward_or_optimizer=False,
        compile_config={"engine_policy": {key: descriptor.get(key) for key in ("enforce_eager", "enable_prefix_caching", "enable_chunked_prefill", "async_scheduling", "max_num_batched_tokens", "gpu_memory_utilization")},
            "numerical_environment": {key: value for key, value in {**dict(getattr(runtime, "env", {}) or {}), **dict(resources.get("env") or {})}.items() if key in _NUMERICAL_ENV}}, host_memory_limit=resources.get("memory_mb"),
        device_memory_limit=None, timeout=resources.get("timeout_seconds"))


def static_admission(spec, runner_identity: Mapping[str, Any], *, workspace_root: Path | None = None,
                     artifact_store: Any = None, catalog: Any = None) -> tuple[AdmissionCheck, ...]:
    """No imports of model libraries, network, deferred data reads or GPU allocation."""
    from pipelines_v2.core.paths import find_workspace_root
    checks = []
    if spec.kind in {"diffusion_generate", "diffusion_capture", "diffusion_intervene"}:
        from pipelines_v2.engine.diffusers.readiness import static_diffusion_checks
        checks.extend(static_diffusion_checks(spec, runner_identity))
    def add(name, status, path, message, remediation=None):
        checks.append(AdmissionCheck(name, "static", status, path, message, remediation))
    resources = dict(runner_identity.get("resources") or {})
    for name in ("cpu", "memory_mb", "timeout_seconds", "max_containers", "shard_count"):
        value = resources.get(name)
        if value is not None:
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0 and (name == "cpu" or type(value) is int)
            add(f"resources.{name}", "passed" if valid else "failed", f"resources.{name}",
                "Finite positive resource value" if valid else f"{name} must be a finite positive {'number' if name == 'cpu' else 'integer'}",
                None if valid else "Set a finite positive value before submission")
    if runner_identity.get('kind')=='modal' and spec.kind in {'external_mattr','circuit_trace'}:
        for field,outer,scale in [('memory_bytes','memory_mb',1024**2),('timeout_seconds','timeout_seconds',1)]:
            requested=getattr(spec,field)
            allocated=resources.get(outer)
            valid=type(allocated) is int and requested<=allocated*scale
            add('resources.operation_'+field,'passed' if valid else 'failed',field,
                'Operation budget fits runner allocation' if valid else 'Operation budget exceeds or lacks explicit runner allocation')
        if spec.kind=='circuit_trace' and not resources.get('gpu'):
            add('resources.tracer_gpu','failed','resources.gpu','Circuit tracing requires an explicit GPU')
    if runner_identity.get('kind')=='modal' and spec.kind.startswith('crosscoder_'):
        from pipelines_v2.representation.crosscoder.settings import Limits
        limits=Limits(**spec.limits)
        device=spec.execution['device']
        okay=bool(resources.get('gpu'))==(device=='cuda')
        add('resources.crosscoder_device','passed' if okay else 'failed','resources.gpu','Crosscoder device must agree with explicit GPU allocation')
        for name,value,allocated in [('memory',limits.max_process_rss_bytes,(resources.get('memory_mb') or 0)*1024**2),('timeout',limits.timeout_seconds,resources.get('timeout_seconds') or 0)]:
            add('resources.crosscoder_'+name,'passed' if value<=allocated else 'failed','resources','Crosscoder operation budget must fit runner '+name)
        from pathlib import PurePosixPath
        from pipelines_v2.runtime.modal_worker import _mounted_volumes
        try:
            mounts=[PurePosixPath(m.mount_path) for m in _mounted_volumes(store_config=artifact_store.identity(),resources=resources)] if artifact_store is not None else [PurePosixPath(m['mount_path']) for m in resources.get('volumes',[])]
        except ValueError as exc:
            mounts=[]
            add('resources.crosscoder_mount','failed','resources.volumes',str(exc))
        single=(resources.get('shard_count') or 1)==1 and str(resources.get('gpu') or '').partition(':')[2] in {'','1'}
        add('resources.crosscoder_topology','passed' if single else 'failed','resources','Crosscoder requires one shard and at most one GPU')
        for path in (*getattr(spec,'model_paths',()),getattr(spec,'reference_path','')):
            if path and not any(PurePosixPath(path).is_relative_to(m) for m in mounts):
                add('resources.crosscoder_mount','failed','resources.volumes','Prepared model/reference path lacks a Modal mount')
    if runner_identity.get('kind')=='modal' and spec.kind=='sae_dictionary_import':
        from pathlib import PurePosixPath
        from pipelines_v2.runtime.modal_worker import _mounted_volumes
        try:
            mounts=[PurePosixPath(m.mount_path) for m in _mounted_volumes(store_config=artifact_store.identity(),resources=resources)] if artifact_store is not None else [PurePosixPath(m['mount_path']) for m in resources.get('volumes',[])]
        except ValueError as exc:
            mounts=[]
            add('resources.sae_mount','failed','resources.volumes',str(exc))
        for field in ('model_path','cache_root'):
            path=PurePosixPath(getattr(spec,field))
            valid=path.is_absolute() and '..' not in path.parts and any(path.is_relative_to(m) for m in mounts)
            add('resources.sae_mount','passed' if valid else 'failed',field,'Prepared SAE/model path must lie within a Modal mount')
    gpu = resources.get("gpu")
    if gpu is not None:
        # Modal's documented public device strings; measured fit remains T06.
        supported = {"T4", "L4", "A10", "L40S", "A100", "A100-40GB", "A100-80GB", "RTX-PRO-6000", "H100", "H100!", "H200", "B200", "B200+", "B300"}
        parts = str(gpu).upper().split(":")
        okay = len(parts) <= 2 and parts[0] in supported and (len(parts) == 1 or parts[1].isdigit() and 0 < int(parts[1]) <= (4 if parts[0] == "A10" else 8))
        add("resources.gpu", "passed" if okay else "failed", "resources.gpu", "Recognized GPU type/count" if okay else "Unsupported GPU type or device count", "Select a documented Modal GPU type/count" if not okay else None)
        engine = spec.bound_engine()
        dtype = getattr(engine, "dtype", None)
        if dtype == "bfloat16" and parts[0] == "T4":
            add("resources.dtype", "failed", "engine.dtype", "T4 does not support native bfloat16", "Choose float16 or an Ampere/newer device")
    for name in ("seed", "random_state"):
        value = getattr(spec, name, None)
        if value is not None and (type(value) is not int or not 0 <= value < 2**32):
            add("numerics." + name, "failed", name, "RNG seed must be an integer in [0, 2**32)")
    solver = getattr(spec, "svd_solver", None)
    if solver is not None and solver not in {"auto", "full", "covariance_eigh", "arpack", "randomized"}:
        add("numerics.svd_solver", "failed", "svd_solver", "Unsupported PCA solver")
    from pipelines_v2.runtime.locks import runtime_for_target
    runtime = runtime_for_target(spec.runtime_spec(), target="modal" if runner_identity.get("kind") == "modal" else "local")
    if runner_identity.get('kind') == 'modal' and getattr(runtime, 'kind', None) == 'external_tool':
        from pipelines_v2.runtime.external_tool import tool_readiness_profile
        profile = tool_readiness_profile(runtime)
        enough_memory = (resources.get('memory_mb') or 0) >= profile['container_memory_mb']
        enough_time = (resources.get('readiness_timeout_seconds') or 600) >= profile['container_timeout_minimum']
        add('runtime.tool_readiness_memory', 'passed' if enough_memory else 'failed', 'resources.memory_mb',
            f"Tool readiness requires {profile['memory_bytes']} child RSS bytes plus 2048 MiB control headroom",
            f"Set resources.memory_mb >= {profile['container_memory_mb']}; CPU readiness receives the same headroom")
        add('runtime.tool_readiness_timeout', 'passed' if enough_time else 'failed', 'resources.readiness_timeout_seconds',
            f"Tool import deadline is {profile['timeout_seconds']} seconds plus control setup",
            f"Set readiness_timeout_seconds >= {profile['container_timeout_minimum']}")
    lock = getattr(runtime, "lock", None)
    add("runtime.lock", "passed" if lock else "unresolved", "runtime.lock",
        "Exact target lock declared; installation is checked at readiness" if lock else "Legacy pip_packages mode has no verified transitive lock",
        None if lock else "Select a target-specific RuntimeLock for qualified execution")
    if lock is not None:
        declared_python = getattr(runtime, "python_version", "")
        if not (lock.python_version == declared_python or lock.python_version.startswith(declared_python + ".")):
            add("runtime.python", "failed", "runtime.python_version", "Declared Python differs from target lock", "Select a lock matching the requested interpreter")
        if gpu is not None and str(gpu).split(":")[0].upper() in {"B300", "B200+"} and lock.cuda and tuple(int(part) for part in lock.cuda.split(".")[:2]) < (13, 1):
            add("runtime.cuda_gpu", "failed", "resources.gpu", "This GPU profile may require CUDA 13.1; selected lock has an older CUDA runtime", "Select a compatible target lock and qualify its exact profile")
        try:
            locked = lock.packages
            extras = getattr(runtime, "pip_packages", ())
            # Legacy wrapper requirements may coexist only if already represented by the lock.
            from packaging.requirements import Requirement
            missing = []
            for value in extras:
                requirement = Requirement(value)
                name = re.sub(r"[-_.]+", "-", requirement.name).lower()
                if name not in locked or requirement.url or not requirement.specifier.contains(locked[name], prereleases=True):
                    missing.append(requirement.name)
            add("runtime.lock_coverage", "failed" if missing else "passed", "runtime.pip_packages",
                "Lock omits required distributions: " + ", ".join(missing) if missing else "Runtime requirements are covered by the lock",
                "Resolve a complete lock including operation dependencies" if missing else None)
        except ValueError as exc:
            add("runtime.lock_valid", "failed", "runtime.lock", str(exc), "Regenerate the lock")
    root = workspace_root or find_workspace_root()
    for mount in getattr(runtime, "local_python_sources", ()):
        if runner_identity.get("kind") in {"local", "test"} and str(mount) == ".":
            add(f"source.mount.{mount}", "not_applicable", "runtime.local_python_sources", "Local process executes existing imports without packaging a workspace mount")
            continue
        try:
            from pipelines_v2.runtime.modal_worker import _resolve_local_python_source
            roots = tuple(dict.fromkeys((Path(root), find_workspace_root(Path(__file__)))))
            path, _ = _resolve_local_python_source(str(mount), source_roots=roots)
            okay = str(mount) not in {".", "", "/"} and not any((candidate / mount).is_symlink() for candidate in roots)
        except (FileNotFoundError, ValueError):
            okay = False
        add(f"source.mount.{mount}", "passed" if okay else "failed", "runtime.local_python_sources",
            "Source mount exists" if okay else "Missing, symlinked or excessively broad source mount", "Declare narrow existing source roots" if not okay else None)
    engine = spec.bound_engine()
    if engine is not None:
        kind = engine.identity().get("kind")
        patch = getattr(spec, "patch", None)
        if patch is not None:
            supported = getattr(engine, "supported_patch_operators", lambda: frozenset())()
            okay = patch.operator in supported
            add("engine.patch_operator", "passed" if okay else "failed", "patch.operator",
                f"Operator {patch.operator} {'supported' if okay else 'not supported'} by {kind}", "Choose an operator declared by this engine" if not okay else None)
        if kind != "toy":
            add("model.immutable_identity", "unresolved", "engine.model_id", "Immutable checkpoint identity requires readiness resolution", "Resolve revision and verify checkpoint completeness before weight load")
        generation = getattr(spec, "generation", None)
        maximum = getattr(engine, "max_model_len", None)
        output = getattr(generation, "max_tokens", None)
        if maximum is not None and output is not None and type(maximum) is int and type(output) is int:
            add("model.context_output", "failed" if output >= maximum else "passed", "generation.max_tokens",
                "Output budget must leave room for input tokens" if output >= maximum else "Output budget fits declared context; tokenized input checked at readiness",
                "Reduce output budget or increase supported context" if output >= maximum else None)
    dataset = getattr(spec, "dataset", None)
    if dataset is not None:
        if not dataset.is_deferred and any(getattr(row, "images", ()) for row in dataset.examples):
            supported = getattr(engine, "supported_input_modalities", lambda: frozenset({"text"}))() if engine else frozenset()
            add("engine.image_input", "passed" if "image" in supported else "failed", "dataset.images",
                "Engine supports image inputs" if "image" in supported else "Bound engine does not support image inputs", "Select an image-capable engine")
        add("data.schema", "unresolved" if dataset.is_deferred else "passed", "dataset",
            "Deferred schema requires remote readiness" if dataset.is_deferred else "Materialized dataset schema available",
            "Validate data where it lives before model initialization" if dataset.is_deferred else None)
    for secret in tuple(getattr(runtime, "secrets", ())) + tuple(spec.runtime_secrets()):
        okay = bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", secret.env_var))
        add(f"secret.binding.{secret.env_var}", "passed" if okay else "failed", "runtime.secrets",
            "Secret binding name valid; access unverified" if okay else "Invalid secret binding environment name")
    add("store.intent", "passed" if artifact_store is not None else "unresolved", "artifacts", "Artifact store configured" if artifact_store else "Artifact store not supplied")
    add("catalog.intent", "unresolved" if catalog is None or getattr(catalog, "kind", None) in {"none", "null"} else "passed", "catalog",
        "Catalog indexing intent is explicit" if catalog else "Catalog indexing intent not supplied")
    checks.append(AdmissionCheck("runtime.qualification", "qualification", "unresolved", "qualification",
        "No measured profile established by static planning", "Run and persist the exact-envelope canary through T06"))
    return tuple(checks)


def read_capability_matrix(engines: Mapping[str, Any]) -> dict[str, Any]:
    return {name: {"capabilities": sorted(item.value for item in engine.capabilities()),
                   "patch_operators": sorted(getattr(engine, "supported_patch_operators", lambda: ())()),
                   "device_qualification": "unresolved"} for name, engine in engines.items()}
