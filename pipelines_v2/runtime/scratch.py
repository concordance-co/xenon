"""Opt-in offline local workflow profile. Shared workflow defaults are untouched."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any, Mapping

from pipelines_v2.core.paths import pipelines_v2_catalog_root, pipelines_v2_state_root
from pipelines_v2.core.types import RuntimeSecret, utc_now_iso, stable_hash
from pipelines_v2.data.datasets import Dataset
from pipelines_v2.engine.toy import ToyEngine
from pipelines_v2.storage.artifacts import ArtifactManifest, CaptureArtifact, OperationArtifact, artifact_from_manifest
from pipelines_v2.storage.local import FileCatalog, LocalArtifactStore
from pipelines_v2.workflow.records import WorkflowRunRecord
from pipelines_v2.workflow.specs import WorkflowSpec
from .scratch_supervision import ScratchLimits, ScratchLimitError, ScratchPreparationError, supervise, tree_bytes


def _walk(value):
    yield value
    if isinstance(value, (CaptureArtifact, OperationArtifact)):
        return
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _walk(child)
    elif is_dataclass(value):
        for descriptor in fields(value):
            yield from _walk(getattr(value, descriptor.name))


def validate_scratch_workflow(workflow: WorkflowSpec, limits: ScratchLimits, *, model_cache: Path | None = None,
                              runner_specs: Mapping[str, Any] | None = None, check_tokens: bool = False):
    """Structural preflight never resolves remote datasets or initializes a client."""
    if runner_specs:
        for name, runner in runner_specs.items():
            if getattr(runner, 'kind', None) not in ('local', 'local_scratch'):
                raise ValueError(f'local-scratch rejects cloud runner {name!r}; declare a local scratch workflow explicitly')
    observed_examples = set()
    known_keys = set()
    artifact_keys = set()
    for step in workflow.steps:
        engine = step.spec.bound_engine()
        if engine is not None and not isinstance(engine, ToyEngine):
            if getattr(engine, 'kind', None) not in ('torch', 'torch_native', 'vit_recognition'):
                raise ValueError('local-scratch supports Toy and explicitly prepared CPU Torch adapters; cloud/GPU engines are rejected')
            require_prepared_model_cache(model_cache)
            if getattr(engine, 'device', 'cpu') != 'cpu':
                raise ValueError('Strict local-scratch currently enforces CPU RSS only; device-memory limits are not enforceable')
            configured = Path(str(getattr(engine, 'model_id', ''))).expanduser().resolve()
            if not configured.is_relative_to(model_cache.resolve()):
                raise ValueError('Scratch model_id must point inside the explicitly prepared offline model cache')
        for value in _walk(step.spec):
            if isinstance(value, RuntimeSecret):
                raise ValueError('local-scratch forbids secret-backed API/cloud providers')
            if isinstance(value, Dataset):
                if value.is_deferred:
                    raise ValueError('local-scratch rejects remote/deferred datasets; use a bounded local fixture')
                observed_examples.update(stable_hash(example.to_dict()) for example in value.examples)
                known_keys.update(example.key for example in value.examples)
                if len(observed_examples) > limits.max_examples:
                    raise ScratchLimitError('Scratch example-count limit exceeded before execution')
                if isinstance(engine, ToyEngine) and engine.sequence_length > limits.max_input_tokens:
                    raise ScratchLimitError('Scratch input-token limit exceeded (actual Toy synthetic sequence length)')
                if engine is not None and not isinstance(engine, ToyEngine) and check_tokens:
                    counts = getattr(engine, 'scratch_token_counts', None)
                    if not callable(counts):
                        raise ValueError('Prepared model adapter needs exact offline scratch_token_counts support before local execution')
                    actual = list(counts(value.examples))
                    if len(actual) != len(value.examples) or any(type(n) is not int or n < 0 for n in actual):
                        raise ValueError('Offline tokenizer returned invalid token counts')
                    if any(n > limits.max_input_tokens for n in actual):
                        raise ScratchLimitError('Scratch input-token limit exceeded before model execution')
            if isinstance(value, (CaptureArtifact, OperationArtifact)):
                if getattr(value.store, 'kind', None) != 'local':
                    raise ValueError('local-scratch rejects remote artifact refs, even when cached')
                coverage = value.manifest().example_coverage
                count = coverage.get('example_count')
                keys = coverage.get('example_keys')
                if type(count) is not int or count > limits.max_examples or not isinstance(keys, list) or len(keys) != count:
                    raise ScratchLimitError('Scratch artifact input requires bounded known example coverage')
                artifact_keys.update(keys)
        generation = getattr(step.spec, 'generation', None)
        if generation is not None and getattr(generation, 'enabled', True):
            maximum = generation.max_tokens
            if type(maximum) is not int or maximum > limits.max_generated_tokens or maximum < 0:
                raise ScratchLimitError('Scratch generated-token limit exceeded before execution')

    if len(observed_examples) + len(artifact_keys - known_keys) > limits.max_examples:
        raise ScratchLimitError('Scratch aggregate example-count limit exceeded before execution')

def require_prepared_model_cache(path: Path | None) -> Path:
    if path is None or not Path(path).is_dir() or not any(Path(path).iterdir()):
        raise ScratchPreparationError('Offline model cache is missing. Prepare separately: hf download <model> --revision <immutable-commit> --local-dir <cache>; then pass that prepared cache explicitly. Scratch never downloads models.')
    return Path(path).resolve()


@dataclass(frozen=True, slots=True)
class LocalScratchProfile:
    limits: ScratchLimits = field(default_factory=ScratchLimits)
    artifact_root: Path | None = None
    catalog_root: Path | None = None
    model_cache: Path | None = None
    device: str = 'cpu'
    kind: str = 'local_scratch'

    def __post_init__(self):
        if self.kind != 'local_scratch':
            raise ValueError('LocalScratchProfile kind must be local_scratch')
        if self.device != 'cpu':
            raise ValueError('Strict scratch device-memory enforcement is unsupported; choose CPU explicitly (no MPS fallback)')
        for name in ('artifact_root', 'catalog_root', 'model_cache'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())

    def to_dict(self):
        return {'schema_version':1,'kind':self.kind,'limits':self.limits.to_dict(),
                'artifact_root':str(self.artifact_root or pipelines_v2_state_root() / 'artifacts'),
                'catalog_root':str(self.catalog_root or pipelines_v2_catalog_root()),
                'model_cache':str(self.model_cache) if self.model_cache else None,'device':self.device}

    @classmethod
    def from_dict(cls, payload):
        data = dict(payload)
        version = data.pop('schema_version', None)
        if type(version) is not int or version != 1 or data.pop('kind', None) != 'local_scratch':
            raise ValueError('Unsupported local scratch profile')
        data['limits'] = ScratchLimits.from_dict(data['limits'])
        return cls(**data)

    def run(self, workflow: WorkflowSpec, *, runner_specs: Mapping[str, Any] | None = None, run_id: str | None = None):
        validate_scratch_workflow(workflow, self.limits, model_cache=self.model_cache, runner_specs=runner_specs)
        if self.model_cache is not None:
            require_prepared_model_cache(self.model_cache)
        return self._run_request({'workflow':workflow.to_dict()}, workflow=workflow, run_id=run_id)

    def run_file(self, workflow_file: str | Path, *, run_id: str | None = None):
        """Import/build a workflow only inside the supervised offline child."""
        if self.model_cache is not None:
            require_prepared_model_cache(self.model_cache)
        path = Path(workflow_file).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'Missing scratch workflow source: {path}')
        return self._run_request({'workflow_file':str(path)}, workflow=None, run_id=run_id)

    def recover(self, run_id: str):
        """Reindex an accepted, atomically promoted run without rerunning compute."""
        from pipelines_v2.storage.finalization import validate_completion
        from pipelines_v2.workflow.orchestrator import WorkflowResult
        config = self.to_dict()
        if not run_id or Path(run_id).name != run_id or run_id in ('.', '..'):
            raise ValueError('Scratch run_id must be one safe path component')
        root = Path(config['artifact_root'])
        attempt = root.parent / 'scratch' / run_id
        saved = json.loads((attempt / 'request.json').read_text())
        if saved['profile'] != config:
            raise ValueError('Recovery requires the original saved scratch profile')
        diagnostic = json.loads((attempt / 'diagnostics.json').read_text())
        if diagnostic.get('status') != 'passed':
            raise ValueError('Cannot recover a scratch run that failed supervision')
        staged_catalog = FileCatalog(attempt / 'catalog')
        record = staged_catalog.load_workflow_run(run_id)
        if record is None or record.status != 'completed':
            raise ValueError('Cannot recover an incomplete scratch workflow')
        store = LocalArtifactStore(root / run_id)
        produced = json.loads((attempt / 'result.json').read_text())
        manifests = {name: ArtifactManifest.from_dict(json.loads((Path(store.root) / payload['artifact_id'] / 'manifest.json').read_text())) for name, payload in produced.items()}
        for manifest in manifests.values():
            validate_completion(manifest, store=store)
        catalog = FileCatalog(config['catalog_root'])
        for manifest in manifests.values():
            catalog.record_artifact(manifest)
        for step in staged_catalog.list_workflow_steps(run_id):
            catalog.record_workflow_step(step)
        catalog.record_workflow_run(replace(record, status='completed', finished_at=utc_now_iso(),
                                            workflow_payload={**record.workflow_payload, 'scratch_profile':config}))
        return WorkflowResult(run_id=run_id, workflow_hash=record.workflow_hash,
                              step_results={name:artifact_from_manifest(manifest,store=store) for name,manifest in manifests.items()})

    def _run_request(self, request, *, workflow, run_id):
        from pipelines_v2.workflow.orchestrator import WorkflowResult
        config = self.to_dict()
        artifact_root = Path(config['artifact_root'])
        catalog = FileCatalog(config['catalog_root'])  # Never consult shared defaults.
        artifact_root.mkdir(parents=True, exist_ok=True)
        # Generated roots carry their own ignore rule, including custom in-repo roots.
        for output_root in (artifact_root, Path(config['catalog_root']), artifact_root.parent / 'scratch'):
            output_root.mkdir(parents=True, exist_ok=True)
            ignore = output_root / '.gitignore'
            if not ignore.exists():
                ignore.write_text('*\n')
        run_id = run_id or 'wr_scratch_' + uuid.uuid4().hex
        if not run_id or Path(run_id).name != run_id or run_id in ('.', '..'):
            raise ValueError('Scratch run_id must be one safe path component')
        if catalog.load_workflow_run(run_id) is not None:
            raise ValueError('Scratch run_id already exists; use a new run ID')
        attempt = artifact_root.parent / 'scratch' / run_id
        attempt.mkdir(parents=True, exist_ok=False)
        staged = attempt / 'artifacts'
        staged.mkdir()
        request = {**request, 'profile':config, 'run_id':run_id}
        (attempt / 'request.json').write_text(json.dumps(request, indent=2))
        initial = workflow or WorkflowSpec(steps=(),name='scratch-source-pending')
        record = WorkflowRunRecord(run_id=run_id,workflow_name=initial.name,workflow_hash=initial.semantic_hash(),
                                   workflow_spec_hash=initial.spec_hash(),workflow_payload={**initial.to_dict(),'scratch_profile':config},
                                   status='running',started_at=utc_now_iso())
        catalog.record_workflow_run(record)
        try:
            supervise([sys.executable, '-m', 'pipelines_v2.runtime.scratch_worker', str(attempt / 'request.json')],
                      attempt=attempt, output_root=staged, limits=self.limits)
            staged_catalog = FileCatalog(attempt / 'catalog')
            produced_record = staged_catalog.load_workflow_run(run_id)
            if produced_record is None or produced_record.status != 'completed':
                raise RuntimeError('Scratch worker did not complete its workflow')
            record = replace(produced_record,workflow_payload={**produced_record.workflow_payload,'scratch_profile':config})
            artifacts = _publish_attempt(attempt, artifact_root, catalog, self.limits)
            for step in staged_catalog.list_workflow_steps(run_id):
                catalog.record_workflow_step(step)
            catalog.record_workflow_run(replace(record, status='completed', finished_at=utc_now_iso()))
            return WorkflowResult(run_id=run_id,workflow_hash=record.workflow_hash,step_results=artifacts)
        except BaseException as exc:
            catalog.record_workflow_run(replace(record,status='failed',finished_at=utc_now_iso(),error=str(exc)))
            raise


def _remap(value, before: str, after: str):
    if isinstance(value, dict):
        return {key:_remap(child,before,after) for key,child in value.items()}
    if isinstance(value, list):
        return [_remap(child,before,after) for child in value]
    if isinstance(value, str) and (value == before or value.startswith(before + '/')):
        return after + value[len(before):]
    return value


def _publish_attempt(attempt: Path, root: Path, catalog: FileCatalog, limits: ScratchLimits):
    """Atomically promote a validated run directory; never copy tensor payloads.

    C03 v1 relocation changes only paths and the descriptor integrity hash.
    Payload checksums are re-derived before relocation if presentation JSON
    changed. Canonical identities remain untouched. No budget check is deferred
    until after the directory becomes visible.
    """
    from pipelines_v2.core.provenance import content_hash
    from pipelines_v2.storage.finalization import finalize_manifest, validate_completion, payload_paths
    from .scratch_store import BudgetedLocalArtifactStore
    staged = attempt / 'artifacts'
    final_root = root / attempt.name
    if final_root.exists():
        raise ValueError('Scratch canonical run directory already exists')
    store = BudgetedLocalArtifactStore(staged,max_new_bytes=limits.max_artifact_bytes)
    produced = json.loads((attempt / 'result.json').read_text())
    manifests = {name:ArtifactManifest.from_dict(payload) for name,payload in produced.items()}
    ids = [manifest.artifact_id for manifest in manifests.values()]
    if len(ids) != len(set(ids)):
        raise ValueError('Scratch relocation requires unique artifact IDs')
    for manifest in manifests.values():
        validate_completion(manifest,store=store)
        owned_root = (staged / manifest.artifact_id).resolve()
        for raw in payload_paths(manifest.storage_refs):
            if not Path(raw).resolve().is_relative_to(owned_root):
                raise ValueError('Scratch relocation refuses out-of-artifact payload refs')
    relocated = {}
    for name, manifest in manifests.items():
        artifact_dir = staged / manifest.artifact_id
        # Only private staging finals can be discarded for presentation rewrites.
        (artifact_dir / 'manifest.json').unlink()
        body = manifest.to_dict()
        body['storage_refs'].pop('manifest',None)
        body['metadata'].pop('completion_v1',None)
        body['metadata'].pop('checkpoint_journal',None)
        if manifest.artifact_kind == 'report':
            for path in artifact_dir.rglob('*.json'):
                value = json.loads(path.read_text())
                store.write_json(manifest.artifact_id,str(path.relative_to(artifact_dir)),
                                 _remap(value,str(staged),str(final_root)))
        def refresh_refs(value):
            if isinstance(value, dict):
                if isinstance(value.get('path'), str):
                    from pipelines_v2.storage.finalization import sha256_file
                    path = Path(value['path'])
                    value['bytes'] = path.stat().st_size
                    if 'sha256' in value:
                        value['sha256'] = sha256_file(path)
                for child in value.values():
                    refresh_refs(child)
            elif isinstance(value, list):
                for child in value:
                    refresh_refs(child)
        refresh_refs(body['storage_refs'])
        # Refresh completion for actual bytes at their current private paths.
        complete = finalize_manifest(store,ArtifactManifest.from_dict(body))
        validate_completion(complete,store=store)
        transformed = _remap(complete.to_dict(),str(staged),str(final_root))
        transformed['metadata']['provenance_v1'] = complete.metadata['provenance_v1']
        transformed['storage_refs'].pop('manifest',None)
        completion = transformed['metadata']['completion_v1']
        completion['integrity_hash'] = content_hash({key:value for key,value in completion.items() if key != 'integrity_hash'})
        (artifact_dir / 'manifest.json').unlink()
        store.write_json(manifest.artifact_id,'manifest.json',transformed)
        relocated[name] = ArtifactManifest.from_dict(transformed)
    if tree_bytes(staged) > limits.max_artifact_bytes:
        raise ScratchLimitError('Scratch final artifact-byte limit exceeded before publication')
    # Same-filesystem atomic directory rename publishes all complete artifacts at once.
    # No copying fallback: a cross-device destination fails explicitly.
    staged.rename(final_root)
    directory_fd = os.open(root,os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    final_store = LocalArtifactStore(final_root)
    for manifest in relocated.values():
        validate_completion(manifest,store=final_store)
    completed = {}
    for name,manifest in relocated.items():
        catalog.record_artifact(manifest)
        completed[name] = artifact_from_manifest(manifest,store=final_store)
    return completed
