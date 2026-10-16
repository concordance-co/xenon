"""Bounded pre-weight readiness and independently verified admission observations."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from pipelines_v2.core.provenance import ProvenanceV1, content_hash, observed_runtime, resolve_provenance, source_snapshot
from pipelines_v2.runtime.admission import AdmissionCheck, WorkloadEnvelope, workload_envelope


def _check(name, status, field, message, remediation=None):
    return AdmissionCheck(name, "readiness", status, field, message, remediation)


_FILE_DIGESTS: dict[tuple[Any, ...], str] = {}


def _file_digest(path: Path) -> str:
    # Immutable cache files can be tens of GB. Rehash on any inode/size/mtime/
    # ctime change; moving a mount never enters scientific identity.
    stat = path.stat()
    key = (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key in _FILE_DIGESTS:
        return _FILE_DIGESTS[key]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    result = digest.hexdigest()
    after = path.stat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != key[1:]:
        raise ValueError("Checkpoint changed during hashing")
    if len(_FILE_DIGESTS) > 4096:
        _FILE_DIGESTS.clear()
    _FILE_DIGESTS[key] = result
    return result


def local_checkpoint_components(path: Path, *, requested_revision: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify local checkpoint coverage and hash actual bytes, never path aliases."""
    path = Path(path)
    if not path.is_dir():
        raise ValueError("Checkpoint directory is unavailable")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError("Checkpoint is missing config.json")
    config = json.loads(config_path.read_text())
    indexes = list(path.glob("*.index.json"))
    required = set()
    for index in indexes:
        payload = json.loads(index.read_text())
        for filename in dict(payload.get("weight_map") or {}).values():
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Checkpoint index contains an unsafe shard path")
            required.add(relative)
    weights = set(path.glob("*.safetensors")) | set(path.glob("pytorch_model*.bin")) | {path / item for item in required if item.suffix in {".safetensors", ".bin"}}
    if not weights or any(not (path / item).is_file() or (path / item).stat().st_size == 0 for item in required):
        raise ValueError("Checkpoint cache is incomplete: missing or empty weight shards")
    for weight in weights:
        if weight.stat().st_size == 0:
            raise ValueError("Checkpoint contains an empty weight file")
        if weight.suffix == ".safetensors":
            with weight.open("rb") as stream:
                header_size = int.from_bytes(stream.read(8), "little")
                if not 2 <= header_size <= min(weight.stat().st_size - 8, 100_000_000):
                    raise ValueError("Invalid or truncated safetensors checkpoint header")
                header = json.loads(stream.read(header_size))
                end = max((item["data_offsets"][1] for key, item in header.items() if key != "__metadata__"), default=0)
                if 8 + header_size + end != weight.stat().st_size:
                    raise ValueError("Truncated or oversized safetensors checkpoint payload")
    tokenizer_names = {"tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt", "merges.txt",
                       "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"}
    tokenizer_files = [item for item in path.iterdir() if item.is_file() and item.name in tokenizer_names]
    if not any(item.name in {"tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt"} for item in tokenizer_files):
        raise ValueError("Checkpoint cache is incomplete: tokenizer assets missing")
    model_files = sorted(weights | set(indexes) | {config_path} | set(path.rglob("*.py")) | set(path.glob("generation_config.json")))
    inventories = {"model": [(item.relative_to(path).as_posix(), _file_digest(item)) for item in model_files],
                   "tokenizer": [(item.relative_to(path).as_posix(), _file_digest(item)) for item in sorted(tokenizer_files)]}
    components = []
    for role, files in inventories.items():
        digest = content_hash(files)
        components.append({"role": role, "repository_id": "local-content:" + digest,
            "requested_revision": requested_revision, "resolved_revision": "sha256:" + digest, "file_set_digest": digest})
    return components, config


def hub_components(repository: str, *, revision: str | None = None, token: str | None = None, attempts: int = 1):
    """Explicit metadata access only. Never downloads weights or declares cache readiness."""
    if not 1 <= attempts <= 3:
        raise ValueError("Hub metadata transport attempts must be between one and three")
    from huggingface_hub import HfApi
    import urllib.error
    for attempt in range(attempts):
        try:
            info = HfApi(token=token).model_info(repository, revision=revision, files_metadata=True)
            break
        except (TimeoutError, ConnectionError, urllib.error.URLError):
            if attempt + 1 == attempts:
                raise
    if not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
        raise ValueError("Hub did not resolve an immutable model commit")
    files = []
    for item in info.siblings:
        lfs = getattr(item, "lfs", None)
        sha = lfs.get("sha256") if isinstance(lfs, Mapping) else getattr(lfs, "sha256", None)
        files.append((item.rfilename, sha or getattr(item, "blob_id", None)))
    files.sort()
    if any(identity is None for _, identity in files):
        raise ValueError("Hub file identity metadata is incomplete")
    return [{"role": role, "repository_id": repository, "requested_revision": revision,
             "resolved_revision": info.sha, "file_set_digest": content_hash(files)} for role in ("model", "tokenizer")]


def runtime_readiness(runtime_spec, *, image_identity: str | None = None, family: str = "local", import_modules: bool = True):
    checks = []
    runtime = observed_runtime(family=family)
    lock = getattr(runtime_spec, "lock", None)
    if lock is None:
        checks.append(_check("runtime.lock", "unresolved", "runtime.lock", "Legacy runtime lacks a transitive lock", "Supply a target lock"))
        return runtime, tuple(checks)
    errors = list(lock.verify_target())
    for name, expected in lock.packages.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != expected:
            errors.append(f"{name}: expected {expected}, observed {actual or 'missing'}")
    checks.append(_check("runtime.lock_inventory", "failed" if errors else "passed", "runtime.lock", "; ".join(errors) if errors else "Target Python and every pinned distribution match lock"))
    for name in lock.imports:
        if not import_modules:
            checks.append(_check("runtime.import." + name, "unresolved", "runtime.imports", "Import has not been executed"))
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:
            checks.append(_check("runtime.import." + name, "failed", "runtime.imports", f"{name} import failed ({type(exc).__name__})", "Repair the isolated lock before loading weights"))
        else:
            checks.append(_check("runtime.import." + name, "passed", "runtime.imports", f"{name} import succeeded; device compatibility remains unqualified"))
    runtime = {**observed_runtime(family=family), "lock_digest": lock.identity,
               "lock_kind": "verified_target_lock" if not errors else "mismatched_target_lock",
               "requirements_digest": lock.requirements_digest,
               "image_identity": image_identity,
               "image_reason": "local process has no container image" if family == "local" else None}
    if family != "local" and not image_identity:
        checks.append(_check("runtime.image", "unresolved", "runtime.image_identity", "Actual image identity not supplied", "Bind the built immutable image ID"))
    return runtime, tuple(checks)


@dataclass(frozen=True, slots=True)
class AdmissionRecord:
    checks: tuple[AdmissionCheck, ...]
    provenance: ProvenanceV1
    envelope: Mapping[str, Any]
    schema_version: int = 1
    scope: str = "execution"

    def __post_init__(self):
        if self.schema_version != 1 or self.scope not in {"cpu_preflight", "execution"}:
            raise ValueError("Unsupported admission record version or scope")
        WorkloadEnvelope.from_dict(self.envelope)

    @property
    def ready(self):
        return bool(self.checks) and all(item.status in {"passed", "not_applicable"} for item in self.checks)

    def to_dict(self):
        return {"schema_version": self.schema_version, "checks": [item.to_dict() for item in self.checks],
                "provenance_v1": self.provenance.to_dict(), "envelope": dict(self.envelope), "scope": self.scope}

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported admission record schema")
        if set(payload) != {"schema_version", "checks", "provenance_v1", "envelope", "scope"}:
            raise ValueError("Unknown or missing admission record fields")
        return cls(tuple(AdmissionCheck.from_dict(item) for item in payload["checks"]), ProvenanceV1.from_dict(payload["provenance_v1"]), payload["envelope"], scope=payload["scope"])


def writer_registry_readiness() -> AdmissionCheck:
    """Explicit online readiness; lazy from_name alone proves no access."""
    try:
        import modal
        registry = modal.Dict.from_name("xenon-artifact-writers-v1", create_if_missing=False)
        registry.get("__xenon_readiness_probe__", None)
    except Exception as exc:
        return _check("writers.access", "failed", "modal.writer_registry",
            f"Writer registry is unavailable ({type(exc).__name__})",
            "Provision explicitly with: modal dict create xenon-artifact-writers-v1; verify this environment has access")
    return _check("writers.access", "passed", "modal.writer_registry", "Existing writer registry read succeeded")


def store_readiness(store) -> AdmissionCheck:
    import tempfile
    if store is None:
        return _check("store.access", "unresolved", "artifact_store", "No store supplied for execution-host access verification")
    try:
        root = Path(store.root)
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=root) as stream:
            stream.write(b"xenon-readiness")
            stream.flush()
            stream.seek(0)
            if stream.read() != b"xenon-readiness":
                raise OSError("Store readback mismatch")
    except (OSError, AttributeError) as exc:
        return _check("store.access", "failed", "artifact_store", f"Store write/read failed ({type(exc).__name__})", "Repair the mounted store permissions before loading weights")
    return _check("store.access", "passed", "artifact_store", "Execution-host store write/read succeeded")


def admit_execution(spec, runner_identity, *, workspace_root=None, image_identity=None,
                    numerics=None, model_components=None, import_modules=True, artifact_store=None,
                    check_writer_registry=False, scope="execution", runtime_spec=None) -> AdmissionRecord:
    """Run on the execution host before weight initialization; no remote-data pull.

    Injected facts are an explicit test/extension seam. Worker integration never
    forwards caller model/numerical assertions through these arguments.
    """
    if spec.kind in {"diffusion_generate", "diffusion_capture", "diffusion_intervene"}:
        from pipelines_v2.engine.diffusers.readiness import admit_diffusion
        return admit_diffusion(spec, runner_identity, workspace_root=workspace_root, image_identity=image_identity,
            import_modules=import_modules, artifact_store=artifact_store,
            check_writer_registry=check_writer_registry, scope=scope)
    from pipelines_v2.runtime.locks import runtime_for_target
    family = "local" if runner_identity.get("kind") in {"local", "test"} else "modal"
    runtime_spec = runtime_spec or runtime_for_target(spec.runtime_spec(), target=family)
    runtime, checks = runtime_readiness(runtime_spec, image_identity=image_identity,
        family=family, import_modules=import_modules)
    checks = list(checks)
    if getattr(runtime_spec, 'kind', None) == 'circuit_tracer_tool':
        try:
            from pipelines_v2.runtime.circuit_tracer import check_circuit_tracer_runtime
            check_circuit_tracer_runtime(runtime_spec)
            spec.verify_inputs()
        except Exception as exc:
            checks.append(_check('circuit_tracer.ready', 'failed', 'runtime.tool_lock',
                f'Pinned circuit-tracer runtime/assets readiness failed ({type(exc).__name__})',
                'Run explicit circuit-tracer prepare on CPU; verify its separate tool image and exact model/PLT files'))
        else:
            checks.append(_check('circuit_tracer.ready', 'passed', 'runtime.tool_lock',
                'Separate pinned tool imports; prepared model/tokenizer and all PLT bytes match'))
    if getattr(runtime_spec, 'kind', None) == 'external_tool':
        try:
            from pipelines_v2.runtime.external_tool import check_tool_runtime
            check_tool_runtime(runtime_spec)
            spec.verify_inputs()
        except Exception as exc:
            from pipelines_v2.runtime.external_tool import ToolReadinessError
            detail = str(exc) if isinstance(exc, ToolReadinessError) else type(exc).__name__
            checks.append(_check('external_tool.ready', 'failed', 'runtime.tool_lock',
                f'Isolated tool/assets readiness failed ({detail})',
                'Provision pinned checkout, submodules, separate interpreter and locked input assets before GPU allocation'))
        else:
            checks.append(_check('external_tool.ready', 'passed', 'runtime.tool_lock',
                'Pinned tool and evaluator import separately; model/data file hashes match'))
    checks.append(store_readiness(artifact_store))
    if check_writer_registry:
        checks.append(writer_registry_readiness())
    for secret in tuple(getattr(runtime_spec, "secrets", ())) + tuple(spec.runtime_secrets()):
        present = bool(os.environ.get(secret.env_var))
        checks.append(_check("secret.access." + secret.env_var, "passed" if present else "failed", "runtime.secrets", "Secret environment value is available" if present else "Required secret value is missing", "Bind and verify the secret on this execution host" if not present else None))
    engine = spec.bound_engine()
    if engine is not None and engine.identity().get("kind") != "toy" and model_components is None:
        path = getattr(engine, "resolved_model_path", lambda: getattr(engine, "model_id", ""))()
        try:
            model_components, config = local_checkpoint_components(Path(path), requested_revision=getattr(engine, "revision", None))
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
            checks.append(_check("model.cache", "failed", "engine.model_id", f"Checkpoint verification failed ({type(exc).__name__}): missing, truncated or invalid required assets", "Complete and validate checkpoint assets on the mounted model cache"))
        else:
            checks.append(_check("model.cache", "passed", "engine.model_id", "Model and tokenizer cache coverage and bytes verified"))
    dataset = getattr(spec, "dataset", None)
    if dataset is not None:
        valid = not dataset.is_deferred and all(isinstance(row.prompt, str) and isinstance(row.key, str) for row in dataset.examples)
        checks.append(_check("data.schema", "unresolved" if dataset.is_deferred else "passed" if valid else "failed", "dataset", "Deferred data schema requires authorized materialization" if dataset.is_deferred else "Materialized data schema checked"))
        if valid and engine is not None and engine.identity().get("kind") == "vllm" and model_components:
            try:
                from transformers import AutoTokenizer
                from pipelines_v2.engine.vllm.capture import _preflight_spec_prompt_lengths
                tokenizer = AutoTokenizer.from_pretrained(engine.resolved_model_path(), local_files_only=True, trust_remote_code=True)
                _preflight_spec_prompt_lengths(engine=engine, spec=spec, tokenizer=tokenizer)
            except Exception as exc:
                checks.append(_check("data.token_budget", "failed", "dataset", f"Tokenizer/context preflight failed ({type(exc).__name__})", "Check local tokenizer assets, context length and output budget"))
            else:
                checks.append(_check("data.token_budget", "passed", "dataset", "Actual tokenizer prompt/output bounds checked before weights"))
    if engine is not None and engine.identity().get('kind') in {'torch','torch_native'} and model_components:
        try:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(engine.resolved_model_path(),local_files_only=True)
            prompts=[row.prompt for row in dataset.examples] if dataset is not None and not dataset.is_deferred else list(getattr(spec,'prompts',()))
            for pair in getattr(spec,'pairs',()): prompts.extend(getattr(pair,k) for k in ('prompt','opposite','same_label','unmatched'))
            limit=config.get('n_positions',config.get('max_position_embeddings'))
            declared=getattr(spec,'max_prompt_tokens',limit)
            limit=min(limit,declared) if limit and declared else limit or declared
            generation=getattr(spec,'generation',None)
            extra=int(generation.max_tokens or 0) if generation is not None and generation.enabled else 0
            if any(not isinstance(prompt,str) or not (0<len(tokenizer.encode(prompt,add_special_tokens=False))<=limit-extra) for prompt in prompts):
                raise ValueError('Prompt plus generation exceeds declared context')
        except Exception as exc:
            checks.append(_check('data.token_budget','failed','dataset',f'Torch tokenizer/context preflight failed ({type(exc).__name__})','Check prepared tokenizer and prompt/output budgets'))
        else:
            checks.append(_check('data.token_budget','passed','dataset','Actual Torch prompt/output bounds checked before weights'))
    policy_seeds = {}
    if numerics is None and engine is not None and scope == "execution":
        from pipelines_v2.runtime.numerics import model_numerical_policy
        numerics, policy_seeds = model_numerical_policy(spec)
    if numerics is not None:
        numerical_env = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "PYTORCH_ENABLE_MPS_FALLBACK", "VLLM_USE_FLASHINFER_SAMPLER")
        numerics = {**numerics, "environment": {key: os.environ[key] for key in numerical_env if key in os.environ}}
    record = resolve_provenance(spec, runner_identity, workspace_root=workspace_root, runtime=runtime,
                                model_components=model_components, numerics=numerics)
    if policy_seeds:
        body = record.to_dict()
        body["seeds"] = {**body["seeds"], **policy_seeds}
        body.pop("scientific_identity"); body.pop("schema_version")
        record = ProvenanceV1(**body)
    bad_source = [reason for reason in record.to_dict()["unresolved"] if reason.startswith("source.") or "source" in reason or "mount" in reason or "secret-bearing" in reason]
    checks.append(_check("source.packaging", "failed" if bad_source else "passed", "source",
        "Mounted source is incomplete or unsafe" if bad_source else "Mounted source bytes recorded"))
    return AdmissionRecord(tuple(checks), record, workload_envelope(spec, runner_identity).to_dict(), scope=scope)


def admitted_provenance_for_request(spec, runner_identity, admitted: AdmissionRecord, *, workspace_root=None, target_readiness: AdmissionRecord | None = None) -> ProvenanceV1:
    """Recompute request/source/data against a persisted execution-host observation.

    This is a reuse binding, not permission to skip worker verification or T06
    qualification. CPU readiness cannot certify a future GPU execution.
    """
    if spec.kind in {"diffusion_generate", "diffusion_capture", "diffusion_intervene"}:
        from pipelines_v2.engine.diffusers.readiness import bind_diffusion_admission
        return bind_diffusion_admission(spec, runner_identity, admitted, workspace_root=workspace_root)
    if not isinstance(admitted, AdmissionRecord):
        admitted = AdmissionRecord.from_dict(admitted)
    if admitted.scope != "execution" or not admitted.ready or not admitted.provenance.identity_complete:
        raise ValueError("Admission record is not complete execution evidence; refresh readiness")
    if workload_envelope(spec, runner_identity).identity != WorkloadEnvelope.from_dict(admitted.envelope).identity:
        raise ValueError("Admission workload envelope changed; refresh readiness")
    body = admitted.provenance.to_dict()
    if target_readiness is not None:
        from pipelines_v2.core.provenance_observation import _OBSERVERS
        if spec.kind not in _OBSERVERS:
            raise ValueError("Target observer binding is unavailable for this operation")
        if (target_readiness.scope != 'cpu_preflight'
                or workload_envelope(spec, runner_identity).identity != WorkloadEnvelope.from_dict(target_readiness.envelope).identity
                or not readiness_matches_execution(target_readiness, admitted)):
            raise ValueError("Fresh target readiness differs from persisted execution facts")
        # Only numerical device observations may be absent on the CPU target.
        # This record was obtained for the exact current serialized request by
        # ModalRunner.readiness; it contains actual remote asset observations.
        unresolved = target_readiness.provenance.to_dict()['unresolved']
        if any(not reason.startswith('numerics.') for reason in unresolved):
            raise ValueError("Fresh target operation identity is incomplete")
        runtime_spec = spec.runtime_spec()
        mounts = tuple(getattr(runtime_spec, 'local_python_sources', ()) or ('pipelines_v2',))
        from pipelines_v2.core.paths import find_workspace_root
        source = source_snapshot(workspace_root or find_workspace_root(), mounts)
        if source.get('unresolved') or source['mounted_source_digest'] != body['source']['mounted_source_digest']:
            raise ValueError("Submitting source differs from observed target source")
        return admitted.provenance
    engine = spec.bound_engine()
    if engine is not None and engine.identity().get("kind") != "toy":
        path = Path(getattr(engine, "resolved_model_path", lambda: getattr(engine, "model_id", ""))())
        if path.is_dir():
            current_components, _ = local_checkpoint_components(path, requested_revision=getattr(engine, "revision", None))
            if current_components != body["model_components"]:
                raise ValueError("Admission model/tokenizer bytes changed; refresh readiness")
    expected = resolve_provenance(spec, runner_identity, workspace_root=workspace_root,
        runtime=body["runtime"], model_components=body["model_components"], numerics=body["numerics"])
    # Explicit policy seeds are derived on the worker in addition to spec seeds.
    derived = {key: value for key, value in body["seeds"].items() if key == "model.initialization"}
    if derived:
        value = expected.to_dict(); value["seeds"].update(derived)
        value.pop("scientific_identity"); value.pop("schema_version")
        expected = ProvenanceV1(**value)
    if expected.scientific_identity != admitted.provenance.scientific_identity:
        raise ValueError("Admission source, request, data or upstream identity changed; refresh readiness")
    return expected


def verify_admitted_execution(spec, runner_identity, admitted: AdmissionRecord, **kwargs) -> ProvenanceV1:
    """Never promote a submitting host's package or model assertions unchecked."""
    admitted_provenance_for_request(spec, runner_identity, admitted, workspace_root=kwargs.get("workspace_root"))
    actual = admit_execution(spec, runner_identity, **kwargs)
    if not actual.ready:
        raise ValueError("Worker readiness failed: " + "; ".join(item.message for item in actual.checks if item.status not in {"passed", "not_applicable"}))
    old, new = admitted.provenance.to_dict(), actual.provenance.to_dict()
    sections = [section for section in ("source", "runtime", "model_components", "data", "numerics", "methods", "seeds")
                if (old[section].get("mounted_source_digest") != new[section].get("mounted_source_digest") if section == "source" else old[section] != new[section])]
    if sections or not actual.provenance.identity_complete:
        raise ValueError("Admitted execution facts drifted: " + ", ".join(sections or new["unresolved"]))
    return actual.provenance


def requires_target_refresh(spec, admitted: AdmissionRecord) -> bool:
    """Mutable remote paths need fresh byte observation before caller reuse.

    An explicit immutable requested revision pins the admitted content as intent;
    the worker still checks bytes before any new execution.
    """
    engine = spec.bound_engine()
    if engine is None or engine.identity().get("kind") == "toy":
        return False
    revision = getattr(engine, "revision", None)
    components = admitted.provenance.to_dict()["model_components"]
    if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision):
        return not components or any(item.get("requested_revision") != revision for item in components)
    if isinstance(revision, str) and revision.startswith("sha256:"):
        model = next((item for item in components if item.get("role") == "model"), {})
        return model.get("resolved_revision") != revision
    return True


def readiness_matches_execution(readiness: AdmissionRecord, execution: AdmissionRecord) -> bool:
    """Compare fresh CPU-visible facts, without certifying GPU observations."""
    if not readiness.ready:
        return False
    current, prior = readiness.provenance.to_dict(), execution.provenance.to_dict()
    if any(current[key] != prior[key] for key in ("model_components", "data", "methods", "seeds")):
        return False
    if current["source"]["mounted_source_digest"] != prior["source"]["mounted_source_digest"]:
        return False
    keys = ("lock_digest", "python", "platform", "installed_package_digest", "image_identity", "requirements_digest", "operation_runtime")
    return all(current["runtime"].get(key) == prior["runtime"].get(key) for key in keys)
