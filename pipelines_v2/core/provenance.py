"""Versioned scientific identity, independent of operation hashes and payload checksums.

Resolution is deliberately offline. Missing facts permit execution, but never reuse.
Admission may supply immutable model/runtime facts through an injected resolver.
"""
from __future__ import annotations

import dataclasses
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable
from enum import Enum
from functools import lru_cache

KEY = "provenance_v1"


def canonical_json(value: Any) -> str:
    """Strict JSON: no repr-based identities, NaNs, or implicit object coercion."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class ProvenanceV1:
    """Immutable JSON-backed C01 record; accessors return independent copies."""
    _json: str

    def __init__(self, *, source: Mapping[str, Any], runtime: Mapping[str, Any],
                 model_components: Any, data: Mapping[str, Any], numerics: Mapping[str, Any],
                 methods: Mapping[str, Any], seeds: Mapping[str, Any],
                 unresolved: Any = (), worker_compatibility: Mapping[str, Any] | None = None):
        body = dict(schema_version=1, source=dict(source), runtime=dict(runtime),
                    model_components=list(model_components), data=dict(data), numerics=dict(numerics),
                    methods=dict(methods), seeds=dict(seeds), unresolved=sorted(set(unresolved)))
        _validate_body(body)
        body["scientific_identity"] = content_hash(_scientific_body(body))
        worker = dict(worker_compatibility or {
            "schema_version": 1, "runtime": body["runtime"], "model_components": body["model_components"],
            "numerics": body["numerics"], "hooks": body["methods"].get("hook_version"),
            "source_digest": body["source"].get("mounted_source_digest"),
            "reset_policy": "not_qualified",
        })
        worker.pop("identity", None)
        body["worker_compatibility"] = {**worker, "identity": content_hash(worker)}
        object.__setattr__(self, "_json", canonical_json(body))

    @property
    def scientific_identity(self) -> str:
        return self.to_dict()["scientific_identity"]

    @property
    def identity_complete(self) -> bool:
        """Identity completeness only; this is NOT measured runtime qualification."""
        return not self.to_dict()["unresolved"]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._json)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProvenanceV1":
        if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
            raise ValueError("Unsupported provenance schema_version; install a reader supporting this version")
        fields = ("source", "runtime", "model_components", "data", "numerics", "methods", "seeds", "unresolved", "worker_compatibility")
        extra = set(payload) - set(fields) - {"schema_version", "scientific_identity"}
        if extra:
            raise ValueError(f"Unsupported provenance fields: {sorted(extra)}")
        missing = set(fields) - payload.keys()
        if missing:
            raise ValueError(f"Provenance v1 missing required fields: {sorted(missing)}")
        result = cls(**{key: payload[key] for key in fields})
        if payload.get("scientific_identity") != result.scientific_identity:
            raise ValueError("Provenance scientific_identity does not match its resolved inputs")
        if payload["worker_compatibility"] != result.to_dict()["worker_compatibility"]:
            raise ValueError("Provenance worker compatibility identity does not match descriptor")
        return result


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_body(body: dict[str, Any]) -> None:
    required = {
        "source": ("repository_id", "head", "dirty_digest", "mounted_source_digest"),
        "runtime": ("family", "lock_digest", "python", "platform", "installed_package_digest", "image_identity", "image_reason"),
        "data": ("content_digest", "ordered_example_key_digest", "split_digest", "preprocessing"),
        "numerics": ("dtype", "attention_implementation", "device", "determinism", "compile", "matmul"),
        "methods": ("operation_version", "hook_version", "evaluator"),
    }
    for section, keys in required.items():
        if not isinstance(body[section], Mapping):
            raise ValueError(f"Provenance {section} must be an object")
        missing = set(keys) - body[section].keys()
        if missing:
            raise ValueError(f"Provenance {section} missing fields: {sorted(missing)}")
    if not isinstance(body["seeds"], Mapping):
        raise ValueError("Provenance seeds must be an object")
    if any(not isinstance(item, str) or not item for item in body["unresolved"]):
        raise ValueError("Provenance unresolved reasons must be nonempty strings")
    for component in body["model_components"]:
        required_keys = {"role", "repository_id", "requested_revision", "resolved_revision", "file_set_digest"}
        if not isinstance(component, Mapping) or required_keys - component.keys():
            raise ValueError("Model component requires role, repository_id, requested_revision, resolved_revision and file_set_digest")
        if any(not isinstance(component[key], str) or not component[key] for key in ("role", "repository_id")):
            raise ValueError("Model component role and repository_id must be nonempty strings")
        revision = component["resolved_revision"]
        immutable = isinstance(revision, str) and bool(re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64}|sha256:[0-9a-f]{64}|builtin:[A-Za-z0-9._-]+)", revision))
        if not immutable or not _is_digest(component["file_set_digest"]):
            reason = f"model_components.{component['role']}.immutable_identity"
            if reason not in body["unresolved"]:
                body["unresolved"].append(reason)
    # Missing required observations cannot be promoted by an empty unresolved list.
    for section, keys in {
        "source": ("mounted_source_digest",),
        "runtime": ("lock_digest", "python", "platform", "installed_package_digest"),
        "data": ("content_digest", "ordered_example_key_digest", "split_digest", "preprocessing"),
    }.items():
        for key in keys:
            if body[section][key] is None or body[section][key] == "" or (key.endswith("digest") and not _is_digest(body[section][key])):
                body["unresolved"].append(f"{section}.{key}")
    if body["runtime"]["image_identity"] is None and (body["runtime"]["family"] != "local" or not body["runtime"]["image_reason"]):
        body["unresolved"].append("runtime.image_identity")
    for section in ("numerics", "methods"):
        for key, value in body[section].items():
            if value is None or value == "":
                body["unresolved"].append(f"{section}.{key}")
    body["unresolved"] = sorted(set(body["unresolved"]))
    _reject_secrets(body)
    canonical_json(body)


def _scientific_body(body: Mapping[str, Any]) -> dict[str, Any]:
    # HEAD/dirty status and requested aliases are diagnostic. Bytes/resolved refs
    # carry identity, so moving a checkout or committing identical bytes is safe.
    components = [{key: value for key, value in item.items() if key != "requested_revision"}
                  for item in body["model_components"]]
    return {"schema_version": 1, "source": {"repository_id": body["source"]["repository_id"],
            "mounted_source_digest": body["source"]["mounted_source_digest"]},
            "runtime": body["runtime"], "model_components": components,
            "data": body["data"], "numerics": body["numerics"], "methods": body["methods"], "seeds": body["seeds"]}


@dataclasses.dataclass(frozen=True, slots=True)
class ReuseDecision:
    reusable: bool
    reasons: tuple[str, ...]
    expected_identity: str
    actual_identity: str | None


def compare_provenance(expected: ProvenanceV1, actual: ProvenanceV1 | None) -> ReuseDecision:
    reasons: list[str] = []
    if not expected.identity_complete:
        reasons.append("current identity unresolved: " + ", ".join(expected.to_dict()["unresolved"]))
    if actual is None:
        reasons.append("legacy artifact has no scientific provenance")
    elif not actual.identity_complete:
        reasons.append("artifact identity unresolved: " + ", ".join(actual.to_dict()["unresolved"]))
    if actual is not None and expected.scientific_identity != actual.scientific_identity:
        old, new = _scientific_body(actual.to_dict()), _scientific_body(expected.to_dict())
        reasons.extend(f"{key} changed" for key in new if old[key] != new[key])
    return ReuseDecision(not reasons, tuple(reasons), expected.scientific_identity,
                         actual.scientific_identity if actual else None)


def provenance_from_metadata(metadata: Mapping[str, Any]) -> ProvenanceV1 | None:
    payload = metadata.get(KEY)
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("provenance_v1 metadata must be an object")
    return ProvenanceV1.from_dict(payload)


def with_provenance(metadata: Mapping[str, Any], provenance: ProvenanceV1) -> dict[str, Any]:
    return {**metadata, KEY: provenance.to_dict()}


_EXCLUDED = {".git", ".planning", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}
_SECRET_NAMES = {".env", "credentials", "credentials.json", "secrets.json", "secrets.toml", "id_rsa", "id_ed25519"}


def source_snapshot(root: Path, sources: tuple[str, ...] = ("pipelines_v2",), *, repository_id: str = "xenon") -> dict[str, Any]:
    """Hash explicitly mounted content; never copy bytes, git URLs or env values."""
    root = root.resolve()
    entries: list[dict[str, str]] = []
    unresolved: list[str] = []
    for mount in sorted(set(sources)):
        raw = Path(mount)
        original = root / raw if not raw.is_absolute() else raw
        if any(candidate.is_symlink() for candidate in (original, *original.parents)):
            unresolved.append("symlinked mount requires an explicit resolved source manifest")
            continue
        target = original.resolve()
        if target in {Path("/"), Path.home(), root} or not target.exists():
            unresolved.append("source mount missing or too broad")
            continue
        # Stable logical mount names, never absolute physical paths.
        logical = raw.as_posix() if not raw.is_absolute() else target.name
        paths = [target] if target.is_file() else sorted(target.rglob("*"))
        for path in paths:
            relative = path.relative_to(target) if path != target else Path(path.name)
            if any(part in _EXCLUDED or part in {"docs", "documentation", ".aws", ".ssh"} for part in relative.parts) or path.suffix in {".pyc", ".pyo"} or path.name in {"README.md", "README.rst", "AGENTS.md", "CHANGELOG.md"}:
                continue
            if path.name in _SECRET_NAMES or path.name.startswith(".env."):
                unresolved.append("excluded secret-bearing source file")
                continue
            if path.is_symlink():
                unresolved.append("symlinked source requires an explicit resolved source manifest")
                continue
            if not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append({"path": f"{logical}/{relative.as_posix()}", "sha256": digest})
    entries.sort(key=lambda item: item["path"])
    digest = content_hash(entries) if entries else None
    head = None
    dirty_digest = None
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        # Hash tracked dirty bytes, not the diff text (which could disclose secrets).
        changed = subprocess.run(["git", "-C", str(root), "diff", "--name-only", "HEAD"], capture_output=True, text=True, check=True).stdout.splitlines()
        changed.extend(subprocess.run(["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True, check=True).stdout.splitlines())
        by_path = {item["path"]: item for item in entries}
        dirty_digest = content_hash([by_path[name] for name in sorted(set(changed)) if name in by_path])
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"repository_id": repository_id, "head": head, "dirty_digest": dirty_digest,
            "mounted_source_digest": digest, "unresolved": sorted(set(unresolved))}


_PACKAGE_CACHE: tuple[Any, tuple[tuple[str, str], ...]] | None = None


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


@lru_cache(maxsize=1024)
def _owned_distribution_metadata(record_path: str, signature: tuple[int, int] | None) -> tuple[str, ...]:
    """Read wheel ownership once per RECORD change, retaining only metadata paths."""
    if signature is None:
        return ()
    record = Path(record_path)
    try:
        with record.open(newline="", encoding="utf-8") as stream:
            return tuple(str((record.parent.parent / row[0]).resolve())
                         for row in csv.reader(stream)
                         if row and row[0].endswith(".dist-info/METADATA"))
    except (OSError, UnicodeError, csv.Error):
        return ()


def _installed_packages() -> tuple[tuple[str, str], ...]:
    global _PACKAGE_CACHE
    # Some imports (notably setuptools via vLLM) expose already-installed vendor
    # dist-info through sys.path. Include owned distributions before that import,
    # so warm-worker history cannot change the installed environment identity.
    providers = {}
    pending = list(importlib.metadata.distributions())
    while pending:
        distribution = pending.pop()
        metadata_path = getattr(distribution, "_path", None)
        if metadata_path is None:
            metadata_path = Path(distribution.locate_file(""))
        path = Path(metadata_path).resolve()
        if str(path) in providers:
            continue
        providers[str(path)] = distribution
        record = path / "RECORD"
        for owned in _owned_distribution_metadata(str(record), _file_signature(record)):
            owned_path = Path(owned)
            if owned_path.is_file() and str(owned_path.parent) not in providers:
                pending.append(importlib.metadata.Distribution.at(owned_path.parent))
    # Distinct providers remain distinct, even when names/versions coincide.
    # A duplicated search path resolves to the same provider and is counted once.
    key = tuple(sorted((path, _file_signature(Path(path) / "METADATA")) for path in providers))
    if _PACKAGE_CACHE is None or _PACKAGE_CACHE[0] != key:
        values = tuple(sorted((str(item.metadata.get("Name", "unknown")).lower(), item.version)
                              for item in providers.values()))
        _PACKAGE_CACHE = (key, values)
    return _PACKAGE_CACHE[1]

def observed_runtime(*, family: str = "local") -> dict[str, Any]:
    packages = content_hash(_installed_packages())
    return {"family": family, "lock_digest": content_hash({"observed_packages": packages, "python": platform.python_version()}),
            "lock_kind": "observed_environment", "package_inventory_policy": "owned-distributions-v1",
            "python": platform.python_version(),
            "platform": platform.platform(), "installed_package_digest": packages,
            "image_identity": None, "image_reason": "local process has no container image" if family == "local" else "image identity unresolved"}


def _walk(value: Any, seen: set[int] | None = None, *, skipped_fields: Mapping[int, set[str]] | None = None):
    seen = seen if seen is not None else set()
    if id(value) in seen:
        return
    seen.add(id(value))
    yield value
    # Artifacts contain stores/caches; never traverse those internals.
    if callable(getattr(value, "manifest", None)):
        return
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            if field.name not in (skipped_fields or {}).get(id(value), set()):
                yield from _walk(getattr(value, field.name), seen, skipped_fields=skipped_fields)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item, seen, skipped_fields=skipped_fields)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk(item, seen, skipped_fields=skipped_fields)


def resolve_provenance(spec: Any, runner_identity: Mapping[str, Any] | None = None, *,
                       workspace_root: Path | None = None, source: Mapping[str, Any] | None = None,
                       runtime: Mapping[str, Any] | None = None, model_components: Any = None,
                       numerics: Mapping[str, Any] | None = None, methods: Mapping[str, Any] | None = None,
                       seeds: Mapping[str, Any] | None = None) -> ProvenanceV1:
    """Observe available facts without loading models, querying data or contacting Hub.

    Tests/admission inject source/runtime/component facts. Deferred datasets and
    unknown engines remain executable, explicitly unresolved and non-reusable.
    """
    if spec.kind in {"diffusion_generate", "diffusion_capture", "diffusion_intervene"}:
        from pipelines_v2.engine.diffusers.readiness import resolve_diffusion_provenance
        return resolve_diffusion_provenance(spec, runner_identity, workspace_root=workspace_root, source=source,
            runtime=runtime, model_components=model_components, numerics=numerics, methods=methods, seeds=seeds)
    from pipelines_v2.core.paths import find_workspace_root
    from pipelines_v2.data.datasets import Dataset

    runner_identity = dict(runner_identity or {"kind": "local"})
    supplied_numerics = numerics is not None
    family = str(runner_identity.get("kind", "local"))
    engine = spec.bound_engine()
    runtime_spec = spec.runtime_spec()
    from .provenance_observation import observe_operation
    facts = observe_operation(spec, runner_identity)
    field_identities = dict(facts.field_identities) if facts is not None else {}
    declared_fields = {field.name for field in dataclasses.fields(spec)} if dataclasses.is_dataclass(spec) else set()
    if field_identities.keys() - declared_fields:
        raise ValueError('Observed field identities must name declared operation fields')
    for name, descriptor in field_identities.items():
        if descriptor is None or descriptor == {} or descriptor == '' or descriptor == []:
            raise ValueError('Observed operation fields require nonempty content descriptors')
        _reject_secrets(descriptor)
        canonical_json(descriptor)
        if any(isinstance(item, Dataset) or callable(getattr(item, 'manifest', None))
               for item in _walk(getattr(spec, name))):
            raise ValueError('Operation observers cannot replace dataset or artifact identities')
    if facts is not None:
        if model_components is None and facts.model_components is not None:
            model_components = list(facts.model_components)
        if numerics is None and facts.numerics is not None:
            numerics = dict(facts.numerics)
    mounts = tuple(getattr(runtime_spec, "local_python_sources", ()) or ("pipelines_v2",))
    source = dict(source) if source is not None else source_snapshot(workspace_root or find_workspace_root(), mounts)
    unresolved = list(source.pop("unresolved", ()))
    if facts is not None:
        unresolved.extend(facts.unresolved)
    if runtime is None:
        runtime = observed_runtime(family="local" if family in {"local", "test"} else family)
        if getattr(runtime_spec, "lock", None) is not None:
            runtime["requested_lock_digest"] = runtime_spec.lock.identity
        if family not in {"local", "test"}:
            unresolved.append("runtime.target_image_and_hardware")
    runtime = dict(runtime)
    if facts is not None and facts.runtime is not None:
        operation_runtime = dict(facts.runtime)
        if (operation_runtime.get('family') == 'modal'
                and operation_runtime.get('image_reason') == 'shared_with_control_runtime'
                and runtime.get('family') == 'modal' and runtime.get('image_identity')):
            operation_runtime['image_identity'] = runtime['image_identity']
        for key in ('family', 'lock_digest', 'python', 'platform', 'installed_package_digest', 'image_reason'):
            if operation_runtime.get(key) in (None, ''):
                unresolved.append(f'runtime.operation_runtime.{key}')
        for key in ('lock_digest', 'installed_package_digest'):
            if not _is_digest(operation_runtime.get(key)):
                unresolved.append(f'runtime.operation_runtime.{key}')
        if operation_runtime.get('image_identity') is None and operation_runtime.get('family') != 'local':
            unresolved.append('runtime.operation_runtime.image_identity')
        runtime['operation_runtime'] = operation_runtime
    engine_kind = engine.identity().get("kind") if engine is not None else None
    if engine_kind == 'vit_recognition' and family in {'local', 'test'}:
        if model_components is None:
            try:
                model_components = engine.observed_components()
            except (ValueError, OSError):
                unresolved.append('model.recognition_assets_unverified')
        if numerics is None:
            from pipelines_v2.runtime.numerics import model_numerical_policy
            numerics, _ = model_numerical_policy(spec)
    if engine_kind in ('torch', 'torch_native') and family in {'local', 'test'}:
        if model_components is None:
            from pipelines_v2.runtime.readiness import local_checkpoint_components
            try:
                model_components, _ = local_checkpoint_components(Path(engine.resolved_model_path()), requested_revision=engine.revision)
            except (ValueError, OSError, KeyError):
                unresolved.append('model.local_checkpoint_unavailable')
        if numerics is None:
            from pipelines_v2.runtime.numerics import model_numerical_policy
            numerics, _ = model_numerical_policy(spec)
    if model_components is None:
        model_components = []
        if engine is not None:
            if engine_kind == "toy":
                model_components = [{"role": role, "repository_id": f"builtin:toy-{role}", "requested_revision": None,
                    "resolved_revision": "builtin:toy-v1", "file_set_digest": source["mounted_source_digest"]}
                    for role in ("model", "tokenizer")]
            else:
                descriptor = engine.identity()
                model_components = [{"role": role, "repository_id": str(descriptor.get("model_id", engine_kind)),
                    "requested_revision": descriptor.get("revision"), "resolved_revision": None, "file_set_digest": None}
                    for role in ("model", "tokenizer")]
    data_items: list[Any] = []
    ordered_keys: list[Any] = []
    split_items: list[Any] = []
    upstream: list[Any] = []
    upstream_content: list[Any] = []
    seed_values: dict[str, Any] = {}
    if engine is not None and getattr(engine, "model_seed", None) is not None:
        seed_values["model.initialization"] = engine.model_seed
    seen_datasets: set[str] = set()
    for value in _walk(spec, skipped_fields={id(spec): set(field_identities)}):
        if not isinstance(value, (str, int, float, bool, type(None), Enum, Path, Mapping, tuple, list, set, frozenset)) and not dataclasses.is_dataclass(value) and not callable(getattr(value, "manifest", None)) and not callable(getattr(value, "capabilities", None)):
            unresolved.append("methods.unobserved_configuration")
        if isinstance(value, Dataset):
            if value.is_deferred:
                unresolved.append("data.deferred_content")
                continue
            row_hashes = [example_content_hash(row) for row in value.examples]
            dataset_digest = content_hash(row_hashes)
            if dataset_digest in seen_datasets:
                continue
            seen_datasets.add(dataset_digest)
            data_items.append(row_hashes)
            ordered_keys.append([row.key for row in value.examples])
            split_items.append([{"key": row.key, "cases": dict(row.cases)} for row in value.examples])
        manifest_fn = getattr(value, "manifest", None)
        if callable(manifest_fn):
            manifest = manifest_fn()
            parent = provenance_from_metadata(manifest.metadata)
            if parent is None or not parent.identity_complete:
                unresolved.append("data.upstream_unqualified")
            upstream.append(parent.scientific_identity if parent else "legacy")
            completion = manifest.metadata.get('completion_v1')
            if completion is not None:
                from .provenance_summaries import payload_identity
                try:
                    upstream_content.append(payload_identity(manifest))
                except ValueError:
                    unresolved.append('data.upstream_content_unverified')
            elif facts is not None:
                unresolved.append('data.upstream_content_unverified')
        if isinstance(value, Path):
            unresolved.append("data.unobserved_file_input")
        if dataclasses.is_dataclass(value):
            from pipelines_v2.operations.common.native import NativeSite
            for field in dataclasses.fields(value):
                if value is spec and field.name in field_identities:
                    continue
                item = getattr(value, field.name)
                if field.name != "output_dir" and not (isinstance(value, NativeSite) and field.name == "module_path") and not (value is engine and field.name == "model_path_root" and all(component.get("file_set_digest") for component in model_components)) and item is not None and any(token in field.name for token in ("path", "file", "checkpoint")) and isinstance(item, (str, Path)):
                    unresolved.append("data.unobserved_file_input")
                if engine is None and field.name in {"repo_id", "model_id", "repository_id"} and item:
                    unresolved.append("data.unresolved_external_asset")
                if "seed" in field.name:
                    seed_values[f"{type(value).__name__}.{field.name}"] = getattr(value, field.name)
    if numerics is not None and engine is None:
        from pipelines_v2.runtime.numerics import cpu_analysis_policy
        _, policy_seeds = cpu_analysis_policy(spec)
        seed_values.update(policy_seeds)
    if numerics is None:
        numerics = {"dtype": "toy-exact" if engine_kind == "toy" else "operation-defined" if engine is None else None,
                    "attention_implementation": "not_applicable" if engine_kind in {None, "toy"} else None,
                    "device": dict(runner_identity.get("resources") or {}).get("device") or ("cpu" if family in {"local", "test"} else None),
                    "determinism": "operation-defined", "compile": "operation-defined", "matmul": "operation-defined"}
        if engine_kind not in {None, "toy"}:
            unresolved.append("numerics.effective_model_configuration")
        if engine is None and spec.kind not in {"label_map", "label_fields"}:
            from pipelines_v2.runtime.numerics import cpu_analysis_policy
            policy, policy_seeds = cpu_analysis_policy(spec)
            if policy is None:
                unresolved.append("numerics.operation_policy")
            else:
                numerics = policy
                seed_values.update(policy_seeds)
        numerical_env = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                         "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "PYTORCH_ENABLE_MPS_FALLBACK")
        effective_env = {**os.environ, **dict(getattr(runtime_spec, "env", {}) or {}), **dict(dict(runner_identity.get("resources") or {}).get("env") or {})}
        numerics["environment"] = {key: effective_env[key] for key in numerical_env if key in effective_env}
    # Requested numerical environment must invalidate coordinator-side reuse even
    # before a runner applies it around execution. Never include arbitrary env.
    numerics = dict(numerics)
    numerical_env = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                     "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "PYTORCH_ENABLE_MPS_FALLBACK")
    base_env = numerics.get('requested_environment', numerics.get('environment', {})) if supplied_numerics else os.environ
    effective_env = {**base_env, **dict(getattr(runtime_spec, "env", {}) or {}), **dict(dict(runner_identity.get("resources") or {}).get("env") or {})}
    numerics['requested_environment'] = {key: effective_env[key] for key in numerical_env if key in effective_env}
    if facts is not None:
        seed_values.update(facts.seeds)
    semantic = _scientific_spec(spec, model_components, field_identities=field_identities)
    # Only its digest is persisted: specs may contain credential references or
    # private prompts. The digest complements resolved content; it is not data identity.
    methods = dict(methods or {"operation_version": f"{spec.kind}:v{spec.schema_version}",
                               "hook_version": "builtin-v1" if engine_kind == "toy" else "xenon-vllm-v1" if engine_kind == "vllm" else "gpt2-residual-v1" if engine_kind == "torch" else engine.adapter_version if engine_kind in {"torch_native", "vit_recognition"} else "not_applicable" if engine is None else None,
                               "evaluator": {"operation_configuration_digest": content_hash(semantic)}})
    if facts is not None and facts.methods:
        methods['evaluator'] = {'base': methods['evaluator'], 'observed_operation': dict(facts.methods)}
    data_contents = {'datasets': data_items, 'upstream': upstream}
    if upstream_content:
        data_contents['upstream_content'] = upstream_content
    if field_identities:
        data_contents['observed_operation_fields'] = field_identities
    return ProvenanceV1(source=source, runtime=runtime, model_components=model_components,
        data={"content_digest": content_hash(data_contents) if "data.deferred_content" not in unresolved else None,
              "ordered_example_key_digest": content_hash(ordered_keys), "split_digest": content_hash(split_items),
              "preprocessing": "raw-spec-and-component-config-v1", "upstream_identities": upstream,
              **({'upstream_content': upstream_content} if upstream_content else {}),
              **({'observed_operation_fields': field_identities} if field_identities else {})},
        numerics=numerics, methods=methods, seeds=seeds or seed_values, unresolved=unresolved)


def _scientific_spec(value: Any, model_components: Any, *, field_identities: Mapping[str, Any] | None = None) -> Any:
    from enum import Enum
    from pipelines_v2.data.datasets import Dataset
    if isinstance(value, Dataset):
        if value.is_deferred:
            # It is unqualified; only non-secret source type/query shape is diagnostic.
            return {"dataset": "unresolved", "request_digest": content_hash(_without_credentials(value.to_dict()))}
        return {"examples": [example_content_projection(row) for row in value.examples]}
    manifest = getattr(value, "manifest", None)
    if callable(manifest):
        record = provenance_from_metadata(manifest().metadata)
        return {"artifact_kind": manifest().artifact_kind,
                "scientific_identity": record.scientific_identity if record else "legacy"}
    if callable(getattr(value, "semantic_identity", None)) and callable(getattr(value, "capabilities", None)):
        identity = _without_credentials(dict(value.semantic_identity()))
        identity.pop("model_path_root", None)
        identity.pop("model_id", None)
        identity.pop("revision", None)
        return {"engine": identity, "components": [{k: v for k, v in component.items() if k != "requested_revision"} for component in model_components]}
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {"kind": getattr(value, "kind", type(value).__name__), **{
            field.name: field_identities[field.name] if field_identities and field.name in field_identities else _scientific_spec(getattr(value, field.name), model_components)
            for field in dataclasses.fields(value) if field.name not in {"output_dir"}
            and not (getattr(value, "kind", None) == "recognition_microscope" and field.name == "image_roots")}}
    if isinstance(value, Mapping):
        return {str(key): _scientific_spec(item, model_components) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scientific_spec(item, model_components) for item in value]
    if isinstance(value, Path):
        # Unclassified file paths cannot establish content identity.
        return {"unresolved_file_input": True}
    if isinstance(value, (set, frozenset)):
        return sorted((_scientific_spec(item, model_components) for item in value), key=canonical_json)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return {"unobserved_configuration_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def resolve_for_runner(spec: Any, runner: Any) -> ProvenanceV1:
    resolver: Callable[..., ProvenanceV1] | None = getattr(runner, "provenance_resolver", None)
    if resolver is not None:
        result = resolver(spec)
        if not isinstance(result, ProvenanceV1):
            raise TypeError("provenance_resolver must return ProvenanceV1")
        return result
    identity = runner.identity() if callable(getattr(runner, "identity", None)) else {"kind": "local"}
    admitted = getattr(runner, "admission_records", {}).get(spec.semantic_hash())
    if admitted is not None:
        from pipelines_v2.runtime.readiness import AdmissionRecord, admitted_provenance_for_request, requires_target_refresh
        admitted = admitted if isinstance(admitted, AdmissionRecord) else AdmissionRecord.from_dict(admitted)
        target_readiness = getattr(runner, 'admission_readiness_records', {}).get(spec.semantic_hash())
        if target_readiness is not None:
            return admitted_provenance_for_request(spec, identity, admitted,
                workspace_root=getattr(runner, 'workspace_root', None), target_readiness=target_readiness)
        refresh = getattr(runner, "refresh_admission_for_reuse", None)
        if callable(refresh) and requires_target_refresh(spec, admitted) and not refresh(spec, admitted):
            runner.admission_records.pop(spec.semantic_hash(), None)
            return resolve_provenance(spec, identity, workspace_root=getattr(runner, "workspace_root", None))
        return admitted_provenance_for_request(spec, identity, admitted, workspace_root=getattr(runner, "workspace_root", None))
    return resolve_provenance(spec, identity, workspace_root=getattr(runner, "workspace_root", None))


def context_provenance(context: Any) -> ProvenanceV1 | None:
    if context is None:
        return None
    payload = context if isinstance(context, Mapping) else context.to_manifest_dict()
    return provenance_from_metadata(payload)


def execution_context(spec: Any, runner_identity: Mapping[str, Any], context: Mapping[str, Any] | None, *, observed: ProvenanceV1 | None = None) -> dict[str, Any]:
    """Observe worker facts; submitting-host runtime facts never stand in for them."""
    import uuid
    result = dict(context or {})
    actual = observed or resolve_provenance(spec, runner_identity)
    origin = provenance_from_metadata(result)
    if origin is not None:
        body = actual.to_dict()
        origin_source = origin.to_dict()["source"]
        if body["source"]["mounted_source_digest"] == origin_source["mounted_source_digest"]:
            body["source"] = origin_source
            body.pop("scientific_identity")
            body.pop("schema_version")
            actual = ProvenanceV1(**body)
        else:
            result["submitting_source_digest"] = origin_source["mounted_source_digest"]
    result[KEY] = actual.to_dict()
    # No unresolved workload may resume somebody else's partials. Every call is
    # executable, but has its own directory until admission resolves its facts.
    if not actual.identity_complete:
        result["provenance_attempt"] = uuid.uuid4().hex
    return result


def example_content_projection(example: Any) -> dict[str, Any]:
    result = {"key": example.key, "prompt": example.prompt, "labels": dict(example.labels),
              "cases": dict(example.cases), "metadata": dict(example.metadata)}
    images = getattr(example, "images", ())
    if images:
        result["images"] = [image.semantic_dict() for image in images]
    return result


def example_content_hash(example: Any) -> str:
    return content_hash(example_content_projection(example))


def bind_dataset_provenance(provenance: ProvenanceV1, *, row_hashes: list[str], keys: list[str],
                            splits: list[Any], parent_identity: str | None = None,
                            partition: Mapping[str, Any] | None = None) -> ProvenanceV1:
    """Freeze observed input bytes without retaining prompts or changing legacy hashes."""
    body = provenance.to_dict()
    body.pop("schema_version")
    body.pop("scientific_identity")
    data = dict(body["data"])
    content = {"datasets": [row_hashes], "upstream": data.get("upstream_identities", [])}
    content.update({key: data[key] for key in ('upstream_content', 'observed_operation_fields') if key in data})
    data.update(content_digest=content_hash(content),
                ordered_example_key_digest=content_hash([keys]), split_digest=content_hash([splits]))
    if parent_identity is not None:
        data["parent_workload_identity"] = parent_identity
        data["partition"] = dict(partition or {})
    body["data"] = data
    body["unresolved"] = [reason for reason in body["unresolved"] if reason not in {"data.deferred_content", "data.content_digest"}]
    return ProvenanceV1(**body)


def bind_materialized_dataset(provenance: ProvenanceV1, dataset: Any, *, parent_identity: str | None = None,
                              partition: Mapping[str, Any] | None = None) -> ProvenanceV1:
    return bind_dataset_provenance(provenance, row_hashes=[example_content_hash(row) for row in dataset.examples],
        keys=[row.key for row in dataset.examples], splits=[{"key": row.key, "cases": dict(row.cases)} for row in dataset.examples],
        parent_identity=parent_identity, partition=partition)


def _reject_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _CREDENTIAL_KEYS:
                raise ValueError("Secret-bearing fields are forbidden in scientific provenance")
            _reject_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_secrets(item)
    elif isinstance(value, str) and value.startswith(("http://", "https://")):
        from urllib.parse import parse_qsl, urlsplit
        parsed = urlsplit(value)
        if parsed.username is not None or any(key.lower() in _CREDENTIAL_KEYS for key, _ in parse_qsl(parsed.query)):
            raise ValueError("Credential-bearing URLs are forbidden in scientific provenance")


_CREDENTIAL_KEYS = {"password", "token", "hf_token", "huggingface_token", "use_auth_token", "access_token", "api_key", "secret", "authorization", "token_env_var"}


def _without_credentials(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _without_credentials(item) for key, item in value.items() if str(key).lower() not in _CREDENTIAL_KEYS}
    if isinstance(value, (tuple, list)):
        return [_without_credentials(item) for item in value]
    return value
