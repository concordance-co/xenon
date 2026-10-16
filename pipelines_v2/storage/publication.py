"""Explicit bounded artifact publication. Never schedules model work or changes source files."""
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any
from pipelines_v2.core.provenance import content_hash
from pipelines_v2.storage.artifacts import ArtifactManifest
from pipelines_v2.storage.finalization import validate_completion, atomic_replace

_ALLOWED = frozenset({'recognition_microscope', 'direction', 'centroid', 'geometry', 'basis', 'subspace', 'probe', 'generation_run'})
_SUMMARIES = frozenset({'attribution_graph', 'external_attribution', 'transform', 'report',
    'sae_feature_splice', 'crosscoder_feature_splice'})
_DICTIONARIES = {'sae_dictionary_import': 'dictionary', 'crosscoder_import': 'dictionary', 'crosscoder_fit': 'checkpoint'}
_DERIVED = frozenset({'sae_encode', 'sae_decode', 'crosscoder_encode', 'crosscoder_decode'})
_ALLOWED = _ALLOWED | _SUMMARIES | _DICTIONARIES.keys() | _DERIVED
_BULK = frozenset({'capture', 'activation_bank', 'crosscoder_pair_capture', 'diffusion_capture'})


def publication_source_lineage(manifest):
    """Read at most 16 preserved publication hops; callers verify result bytes first."""
    provenance = manifest.provenance()
    if provenance is None or not provenance.identity_complete:
        raise ValueError('Publication lineage requires complete execution identity')
    publication = manifest.metadata.get('publication_v1')
    kind = manifest.artifact_kind
    for _ in range(16):
        if not isinstance(publication, dict):
            raise ValueError('Missing publication lineage')
        if (publication.get('execution_identity_preserved') is not True or
                publication.get('source_scientific_identity') != provenance.scientific_identity):
            raise ValueError('Publication execution identity is not preserved')
        if publication.get('published_artifact_kind', kind) != kind:
            raise ValueError('Publication kind declaration mismatch')
        kind = publication.get('source_artifact_kind', kind)
        context = publication.get('source_workflow', {})
        if not isinstance(context, dict):
            raise ValueError('Invalid publication source workflow')
        if context:
            return {k:v for k,v in context.items() if k in
                    {'run_id','step_name','workflow_step_key','workflow_hash'}}, publication.get('source_workflow_artifact_kind', kind)
        publication = publication.get('prior_publication')
        if publication is None:
            return {}, kind
    raise ValueError('Publication lineage exceeds 16 hops')


def _positive_budget(value, label):
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError(f'{label} requires an explicit positive finite integer ceiling')


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()



def _read_source_json(item, path, descriptor, limit):
    if descriptor['bytes'] > limit: raise ValueError('JSON payload ceiling exceeded')
    if item.store.kind == 'local':
        with Path(path).open('rb') as stream: data = stream.read(limit+1)
    else:
        import modal
        from pipelines_v2.storage.modal import modal_volume_mount_path
        relative = PurePosixPath(path).relative_to(PurePosixPath(modal_volume_mount_path(item.store.root)))
        chunks, count = [], 0
        for chunk in modal.Volume.from_name(item.store.name).read_file(str(relative)):
            count += len(chunk)
            if count > limit: raise ValueError('Actual remote JSON exceeds ceiling')
            chunks.append(chunk)
        data = b''.join(chunks)
    if len(data) > limit or len(data) != descriptor['bytes'] or _sha(data) != descriptor['sha256']:
        raise ValueError('Source JSON changed after integrity admission')
    return data

def _check_identity_paths(value):
    # C01 can name ORIGINAL artifact IDs. Relocation must never rewrite them.
    if isinstance(value, dict):
        for item in value.values(): _check_identity_paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value: _check_identity_paths(item)
    elif isinstance(value, str) and value.startswith(('/', 'file:', '~/')):
        raise ValueError('Immutable scientific identity contains an unportable absolute path')


def _portable_payload(kind, payload, *, result_only, graph_source=None):
    """Validate known result schemas and explicitly project presentation/raw diagnostics."""
    if not isinstance(payload, dict): raise ValueError('Publication result must be a JSON object')
    result = json.loads(json.dumps(payload, allow_nan=False))
    omitted = []
    if kind == 'attribution_graph' or (kind == 'transform' and 'graph' in result):
        from pipelines_v2.operations.graphs import AttributionGraph
        AttributionGraph.from_dict(result['graph'])
    elif kind == 'external_attribution' or result.get('kind') == 'external_attribution':
        from pipelines_v2.operations.attribution.contracts import validate_result
        validate_result(result)
        raw = result.get('raw_outputs', {})
        if any('data' in item for item in raw.values()):
            if not result_only: raise ValueError('Official attribution publication requires result_only=True to omit raw producer blobs')
            result['raw_outputs'] = {name:{k:v for k,v in entry.items() if k != 'data'} for name,entry in raw.items()}
            omitted.append('raw_outputs.*.data')
        compute = result.get('effective_compute', {})
        if 'argv' in compute:
            if not result_only: raise ValueError('Use result_only=True for portable official attribution diagnostics')
            compute.pop('argv'); omitted.append('effective_compute.argv')
    elif kind == 'report':
        if not result_only or result.get('kind') != 'report_result' or result.get('template') != 'summary':
            raise ValueError('Reports require explicit result_only=True and the built-in summary template')
        if not isinstance(result.get('inputs'), list) or not isinstance(result.get('summary'), dict):
            raise ValueError('Invalid built-in report summary')
        result.pop('output_dir', None); omitted.append('output_dir/rendered presentation files')
        if 'evidence' in result:
            from pipelines_v2.experiments import ExperimentPack, EvidenceRecord
            bundle = result['evidence']; pack = ExperimentPack.from_dict(bundle['pack'])
            for record in bundle['records']: EvidenceRecord.from_dict(record, pack=pack)
        # Engine/cache/mount descriptors are presentation diagnostics. C01 stays
        # unchanged on the source and published manifests, including all parents.
        allowed = {'artifact_id', 'artifact_kind', 'created_at', 'workflow', 'run_id',
                   'scientific_identity', 'provenance_status', 'unresolved', 'name', 'example_coverage'}
        result['inputs'] = [{k:v for k,v in row.items() if k in allowed} for row in result['inputs']]
        omitted.append('inputs runtime/engine/storage presentation descriptors')
    elif kind == 'transform':
        tag = result.get('kind')
        if tag in {'native_route_report', 'circuit_trace_report'}:
            if not result.get('source_artifact_id') and graph_source:
                result['source_artifact_id'] = graph_source
            if not result_only or not result.get('source_artifact_id') or not result.get('graph_id'):
                raise ValueError('Graph report views require explicit result_only and linked source graph')
            for field in ('graph_html', 'graph_json'):
                if field in result: result.pop(field); omitted.append(field)
        elif tag == 'neuron_dossier_result':
            if result.get('schema_version') != 1 or not isinstance(result.get('dossiers'), list) or not isinstance(result.get('summary'), dict):
                raise ValueError('Invalid neuron dossier summary')
            if any(not row.get('neuron_id') or not isinstance(row.get('observations'), dict) for row in result['dossiers']):
                raise ValueError('Invalid neuron dossier coordinates')
        else: raise ValueError('Transform output has no supported typed publication summary')
    elif kind in _DICTIONARIES:
        if kind == 'sae_dictionary_import' and (result.get('type') != 'sae' or result.get('schema_version') != 1):
            raise ValueError('Invalid SAE dictionary metadata')
        if kind == 'crosscoder_fit' and result.get('complete') is not True:
            raise ValueError('Only complete crosscoder fit checkpoints may be published')
        if kind == 'crosscoder_import' and result.get('fit_complete') is not True:
            raise ValueError('Only completed crosscoder dictionaries may be published')
    elif kind in _DERIVED:
        if not result_only: raise ValueError('Per-example codes/reconstructions are activation data; use result_only=True for derived summaries')
        if not (result.get('dictionary_id') or result.get('dictionary_identity')):
            raise ValueError('Derived summary lacks dictionary identity')
    elif kind in {'sae_feature_splice', 'crosscoder_feature_splice'}:
        if not isinstance(result.get('rows'), list) or not (result.get('dictionary_id') or result.get('dictionary_identity')):
            raise ValueError('Feature splice summary requires rows and dictionary identity')
    return result, omitted



def _sha(data): return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class PublicationTarget:
    name: str
    root: str
    schema_version: int = 1

    def __post_init__(self):
        p=PurePosixPath(self.root)
        if self.schema_version!=1 or not self.name or '/' in self.name or not p.is_absolute() or '..' in p.parts or len(p.parts)<3 or str(p)!=self.root:
            raise ValueError('Publication target needs an existing volume name and normalized /mount/subdirectory root')
    def to_dict(self): return {'schema_version':1,'name':self.name,'root':self.root}


@dataclass
class PublicationPlan:
    request: dict
    blobs: dict[str, bytes | Path] = field(repr=False)
    source_audit: list[dict] = field(default_factory=list)
    def to_dict(self):
        return {**self.request,'source_audit':self.source_audit,'no_recomputation':True,'hardware_requalification':False}


def _rewrite(value, paths, ids, target, *, parent_key=None):
    if isinstance(value, dict):
        if value.get('schema_version') == 1 and 'scientific_identity' in value and 'worker_compatibility' in value:
            _check_identity_paths(value)
            return value
        if 'store' in value and 'path' in value and value['store'] not in {'local', 'local_path', 'modal_volume'}:
            raise ValueError('Only explicit local dependency closure is supported')
        result={k:_rewrite(v,paths,ids,target,parent_key=k) for k,v in value.items()}
        if value.get('store') == 'modal_volume' and (value.get('name') != target.name or value['path'] not in paths):
            raise ValueError('Remote ref is outside the verified shared dependency closure')
        if 'store' in value and 'path' in value:
            result.update(store='modal_volume',name=target.name)
        return result
    if isinstance(value, list): return [_rewrite(v,paths,ids,target,parent_key=parent_key) for v in value]
    if isinstance(value, tuple): return [_rewrite(v,paths,ids,target,parent_key=parent_key) for v in value]
    if isinstance(value,str):
        if value in paths: return paths[value]
        if value in ids: return ids[value]
        if parent_key == 'pointer' and value.startswith('/'):
            return value  # RFC6901 selector; validated by C05, not a filesystem path.
        if value.startswith('/') or value.startswith('file:') or value.startswith('~/'):
            raise ValueError('Undeclared or ambiguous local path in payload/manifest; include the dependency or recompute')
        if parent_key in {'artifact_id','source_artifact_id','capture_id','dictionary_id'} and value and value not in ids.values():
            raise ValueError('Payload artifact reference is outside the supplied dependency closure')
    return value


def _resize_refs(value, sizes):
    if isinstance(value,dict):
        result={k:_resize_refs(v,sizes) for k,v in value.items()}
        if 'path' in result and result['path'] in sizes:
            if result.get('tensor_path'):
                result['metadata_bytes']=sizes[result.get('metadata_path',result['path'])]
                result['tensor_bytes']=sizes[result['tensor_path']]
                result['bytes']=result['metadata_bytes']+result['tensor_bytes']
            else: result['bytes']=sizes[result['path']]
        return result
    if isinstance(value,list): return [_resize_refs(v,sizes) for v in value]
    return value


def plan_publication(artifact, *, target: PublicationTarget, max_payload_bytes: int, dependencies=(),
                     result_only: bool = False, max_json_bytes: int = 16 * 1024**2,
                     max_artifacts: int = 128, max_metadata_bytes: int = 1024**2):
    """Prepare an explicit closed DAG; large tensor blobs stay file-backed.

    Already-shared dependencies are verified on their volume and referenced in
    place. Local captures cannot enter a publication closure, even as summaries.
    result_only explicitly projects supported presentation/activation-derived
    results; the published kind and lineage record that it is a partial view.
    """
    for value, label in ((max_payload_bytes, 'Publication payload'), (max_json_bytes, 'JSON payload'),
                         (max_artifacts, 'Dependency count'), (max_metadata_bytes, 'Request metadata')):
        _positive_budget(value, label)
    if type(result_only) is not bool: raise ValueError('result_only must be explicit boolean')
    if getattr(artifact.store, 'kind', None) not in {'local','modal_volume'}: raise ValueError('Publication source must be local or Modal')
    all_artifacts = {}
    for item in (artifact, *dependencies):
        if item.id in all_artifacts: raise ValueError('Duplicate dependency ID')
        if getattr(item.store, 'kind', None) not in {'local', 'modal_volume'}:
            raise ValueError('Dependencies must be local or already shared on Modal')
        all_artifacts[item.id] = item
    if len(all_artifacts) > max_artifacts: raise ValueError('Publication dependency count ceiling exceeded')
    ordered, active, seen = [], set(), set()
    def visit(key):
        if key in active: raise ValueError('Publication dependency cycle')
        if key in seen: return
        if key not in all_artifacts: raise ValueError(f'Missing explicit dependency: {key}')
        active.add(key)
        for parent in all_artifacts[key].manifest().input_artifact_refs: visit(parent)
        active.remove(key); seen.add(key); ordered.append(all_artifacts[key])
    visit(artifact.id)
    if seen != set(all_artifacts): raise ValueError('Unused supplied dependency; do not upload unrelated artifacts')
    roots, manifests, shared, selected_refs, views = {}, {}, {}, {}, {}
    audit, payloads, total = [], {}, 0
    normalized_json, original_json, remote_sources = {}, {}, {}
    from pipelines_v2.storage.finalization import payload_paths
    for item in ordered:
        m = item.manifest(); p = m.provenance()
        if p is None or not p.identity_complete:
            raise ValueError('Publication requires complete semantic provenance; unresolved artifacts need honest producer identity resolution')
        if getattr(item.store, 'kind', None) == 'modal_volume' and item.id != artifact.id:
            if item.store.name != target.name or not set(m.input_artifact_refs) <= set(shared):
                raise ValueError('Shared dependencies must be a closed existing DAG on the target volume')
            # ModalVolumeStore delegates checksum verification to its CPU worker;
            # this never calls load_feature/localize/read_safetensors on the client.
            validate_completion(m, store=item.store)
            shared[item.id] = m
            continue
        if m.artifact_kind not in _ALLOWED or m.artifact_kind in _BULK:
            raise ValueError('Artifact data class is not eligible; local bulk activations must remain outside publication')
        if item.store.kind == 'modal_volume' and item.store.name != target.name:
            raise ValueError('Mounted publication copy currently requires the same Modal volume; cross-volume names are unsupported')
        validate_completion(m, store=item.store)
        root = Path(item.store.root)/item.id
        if item.store.kind == 'local': root = root.resolve()
        else: remote_sources[item.id] = m
        if Path(item.id).name != item.id or (item.store.kind == 'local' and not root.is_dir()):
            raise ValueError('Invalid artifact root')
        roots[item.id] = root; manifests[item.id] = m
        refs = {k:v for k,v in m.storage_refs.items() if k != 'manifest'}
        omitted = []; output_kind = m.artifact_kind
        is_new = m.artifact_kind in (_SUMMARIES | _DICTIONARIES.keys() | _DERIVED)
        if is_new:
            ref = refs.get('result', {})
            if ref.get('format') != 'json' or type(ref.get('bytes')) is not int or not 0 <= ref['bytes'] <= max_json_bytes:
                raise ValueError('Typed publication requires a bounded declared JSON result')
            result_path = Path(ref['path'])
            if not result_path.is_relative_to(root) or '..' in result_path.parts or (item.store.kind == 'local' and result_path.resolve() != result_path):
                raise ValueError('Declared result escapes its artifact root')
            descriptor = next(d for d in m.metadata['completion_v1']['payloads'] if d['path'] == str(result_path))
            raw = _read_source_json(item, str(result_path), descriptor, max_json_bytes)
            original_json[str(result_path)] = raw
            if sum(map(len,original_json.values())) > max_json_bytes: raise ValueError('Aggregate JSON read ceiling exceeded')
            graph_sources=[key for key in m.input_artifact_refs if all_artifacts[key].manifest().artifact_kind=='attribution_graph']
            normalized, omitted = _portable_payload(m.artifact_kind, json.loads(raw), result_only=result_only,
                graph_source=graph_sources[0] if len(graph_sources)==1 else None)
            normalized_json[str(result_path)] = normalized
            if m.artifact_kind in _DICTIONARIES:
                feature = _DICTIONARIES[m.artifact_kind]
                if set(refs) != {'result', 'features'} or set(refs['features']) != {feature}:
                    raise ValueError('Dictionary publication requires exactly its typed dictionary/checkpoint feature')
                feature_ref = refs['features'][feature]
                expected_format = 'sae_tensors_v1' if m.artifact_kind.startswith('sae_') else 'crosscoder_safetensors_v1'
                if feature_ref.get('format') != expected_format:
                    raise ValueError('Dictionary feature codec differs from the typed publication contract')
            elif m.artifact_kind in _DERIVED or m.artifact_kind == 'report':
                refs = {'result': refs['result']}
                output_kind = m.artifact_kind + '_summary'
                omitted += ['non-result storage payloads']
            elif set(refs) != {'result'}:
                raise ValueError('Typed summary publication may not smuggle feature/capture payloads')
            if m.artifact_kind == 'external_attribution' and omitted:
                output_kind = 'external_attribution_summary'
        selected_refs[item.id] = refs
        selected = set(payload_paths(refs))
        descriptors = m.metadata['completion_v1']['payloads']
        views[item.id] = {'source_artifact_kind':m.artifact_kind, 'published_artifact_kind':output_kind,
            'view':'result_summary' if omitted or len(selected) != len(descriptors) else 'complete_artifact',
            'omitted_fields':omitted,
            'omitted_payloads':[{'name':Path(d['path']).name,'sha256':d['sha256'],'bytes':d['bytes']} for d in descriptors if d['path'] not in selected]}
        audit.append({'artifact_id':item.id,'local_root':str(root),
            'source_manifest_digest':content_hash({**m.to_dict(),'storage_refs':{k:v for k,v in m.storage_refs.items() if k!='manifest'}}),
            'scientific_identity':p.scientific_identity})
        for descriptor in descriptors:
            if descriptor['path'] not in selected: continue
            path = Path(descriptor['path'])
            if not path.is_absolute() or '..' in path.parts or not path.is_relative_to(root) or (item.store.kind == 'local' and (path.resolve()!=path or path.is_symlink())):
                raise ValueError('Declared payload escapes its artifact root or follows a symlink')
            if path.suffix not in ('.json','.safetensors'): raise ValueError('Unsupported opaque payload; cannot prove reference closure')
            if path.suffix == '.json' and descriptor['bytes'] > max_json_bytes: raise ValueError('JSON payload ceiling exceeded')
            if str(path) not in payloads:
                # JSON projections can shrink large official output, but reading it
                # still obeys max_json_bytes; upload admission uses projected bytes.
                size = len(json.dumps(normalized_json[str(path)], allow_nan=False).encode()) if str(path) in normalized_json else descriptor['bytes']
                total += size
                if total > max_payload_bytes: raise ValueError('Publication payload ceiling exceeded')
                payloads[str(path)] = (item.id, descriptor)
    # C05 report publication rechecks actual linked measurements, not only its
    # author-declared record shape. This reader performs bounded one-hop JSON
    # reads; it never requests tensor features or recursively follows reports.
    for key,m in manifests.items():
        if m.artifact_kind == 'report' and 'evidence' in normalized_json.get(m.storage_refs['result']['path'], {}):
            from types import SimpleNamespace
            from pipelines_v2.evidence import EvidenceReader
            catalog = SimpleNamespace(load_artifact=lambda artifact_id: all_artifacts[artifact_id].manifest() if artifact_id in all_artifacts else None)
            checked = EvidenceReader(catalog).artifact(key, include_result=True)['evidence_verification']
            if checked['status'] != 'verified': raise ValueError('Unverified report evidence: '+checked.get('reason','unknown'))
    ids = {key:'published_'+content_hash({'source_id':key,'completion':m.metadata['completion_v1']['integrity_hash'],
        'target':target.to_dict(), 'publication_policy':3, **({'view':views[key]} if views[key]['view'] != 'complete_artifact' else {})})[:28] for key,m in manifests.items()}
    ids.update({key:key for key in shared})
    paths = {raw:str(PurePosixPath(target.root)/ids[key]/Path(raw).relative_to(roots[key]).as_posix()) for raw,(key,_) in payloads.items()}
    for m in shared.values():
        for path in payload_paths(m.storage_refs): paths[path] = path
        ref = m.storage_refs.get('manifest', {})
        if ref.get('path'): paths[ref['path']] = ref['path']
    blobs, transfers, sizes, originals = {}, [], {}, {}
    for raw,(key,descriptor) in payloads.items():
        path = Path(raw)
        item = all_artifacts[key]
        if item.store.kind == 'local' and (path.stat().st_size != descriptor['bytes'] or _file_sha(path) != descriptor['sha256']):
            raise ValueError('Source changed after integrity admission')
        if raw.endswith('.json'):
            original = original_json.get(raw)
            if original is None:
                original = _read_source_json(item, raw, descriptor, max_json_bytes)
                original_json[raw] = original
                if sum(map(len,original_json.values())) > max_json_bytes: raise ValueError('Aggregate JSON read ceiling exceeded')
            originals[raw] = original
            decoded = json.loads(original)
            rewritten = _rewrite(normalized_json.get(raw, decoded), paths, ids, target)
            published = original if rewritten == decoded else json.dumps(rewritten,sort_keys=True,indent=2,allow_nan=False).encode()
            digest = _sha(published); size = len(published)
        else:
            # Never load dictionary weights into a Python byte buffer.
            if item.store.kind == 'local':
                from safetensors import safe_open
                with safe_open(str(path), framework='numpy') as header: header.keys()
                published = path
            else: published = None  # Mounted copy, never a client tensor download.
            digest = descriptor['sha256']; size = descriptor['bytes']
        if published is not None: blobs[digest] = published
        sizes[paths[raw]] = size
        transfers.append({'destination':paths[raw],'sha256':digest,'bytes':size,'source_sha256':descriptor['sha256'],
            'source_artifact_id':key,'source_relative_path':Path(raw).relative_to(roots[key]).as_posix(),
            **({'remote_source':{'name':target.name,'path':raw}} if published is None else {})})
    for _ in range(64):
        changed = False
        for raw,original in originals.items():
            decoded = json.loads(original)
            updated = _resize_refs(_rewrite(normalized_json.get(raw, decoded),paths,ids,target),sizes)
            data = original if updated == decoded else json.dumps(updated,sort_keys=True,indent=2,allow_nan=False).encode()
            if len(data) > max_json_bytes: raise ValueError('Rewritten JSON exceeds JSON payload ceiling')
            transfer = next(row for row in transfers if row['destination']==paths[raw])
            digest = _sha(data)
            if digest != transfer['sha256']:
                changed = True; transfer.update(sha256=digest,bytes=len(data)); blobs[digest]=data; sizes[paths[raw]]=len(data)
        if not changed: break
    else: raise ValueError('Payload reference sizes do not converge; unsupported dependency structure')
    for row in transfers:
        if row['source_artifact_id'] in remote_sources and row['sha256'] == row['source_sha256']:
            row['remote_source'] = {'name':target.name, 'path':str(roots[row['source_artifact_id']]/row['source_relative_path'])}
    blobs = {row['sha256']:blobs[row['sha256']] for row in transfers if 'remote_source' not in row}
    if sum(row['bytes'] for row in transfers)>max_payload_bytes: raise ValueError('Rewritten publication exceeds payload ceiling')
    if sum(row['bytes'] for row in transfers if row['destination'].endswith('.json')) > max_json_bytes:
        raise ValueError('Aggregate rewritten JSON ceiling exceeded')
    operation_id='pub_'+content_hash({'ids':ids,'payloads':transfers,'target':target.to_dict()})[:28]
    prepared=[]
    for key,m in manifests.items():
        metadata={k:v for k,v in m.metadata.items() if k not in {'completion_v1','checkpoint_journal','published_report','publication_v1'}}
        provenance=metadata.pop('provenance_v1'); _check_identity_paths(provenance)
        metadata=_rewrite(metadata,paths,ids,target); metadata['provenance_v1']=provenance
        source_workflow = {k:v for k,v in m.workflow_context.items() if k in {'run_id','step_name','workflow_step_key','workflow_hash'}}
        lineage_kind = m.artifact_kind
        if not source_workflow and 'publication_v1' in m.metadata:
            source_workflow, lineage_kind = publication_source_lineage(m)
        metadata['publication_v1']={'schema_version':1,'operation_id':operation_id,'source_artifact_id':key,
            'source_manifest_digest':next(a['source_manifest_digest'] for a in audit if a['artifact_id']==key),
            'source_completion_digest':m.metadata['completion_v1']['integrity_hash'],
            'source_scientific_identity':m.provenance().scientific_identity,
            'source_workflow':source_workflow, 'source_workflow_artifact_kind':lineage_kind,
            'payload_hashes':[row for row in transfers if row['source_artifact_id']==key],
            'hardware_requalification':False,'execution_identity_preserved':True,'publication_policy':3, **views[key],
            **({'prior_publication':m.metadata['publication_v1']} if 'publication_v1' in m.metadata else {})}
        refs=_resize_refs(_rewrite(selected_refs[key],paths,ids,target),sizes)
        remote=replace(m,artifact_id=ids[key],artifact_kind=views[key]['published_artifact_kind'],storage_refs=refs,metadata=metadata,
            input_artifact_refs=tuple(ids[parent] for parent in m.input_artifact_refs),
            runner={'kind':'publication','source_execution_kind':m.runner.get('kind'),'target':target.to_dict()},engine={},workflow_context={})
        prepared.append(remote.to_dict())
    request={'schema_version':1,'operation_id':operation_id,'target':target.to_dict(),'artifact_id':ids[artifact.id],
        'max_payload_bytes':max_payload_bytes,'max_json_bytes':max_json_bytes,'max_artifacts':max_artifacts,'max_metadata_bytes':max_metadata_bytes,
        'payload_bytes':sum(row['bytes'] for row in transfers),'payloads':transfers,'manifests':prepared,
        'shared_dependencies':[m.to_dict() for m in shared.values()],
        'remote_sources':[m.to_dict() for m in remote_sources.values()]}
    if len(json.dumps(request).encode())>max_metadata_bytes: raise ValueError('Publication metadata ceiling exceeded')
    return PublicationPlan(request,blobs,audit)

def publish_artifact(plan: PublicationPlan, *, transport, catalog, receipt_path):
    """Upload -> mounted finalization -> catalog/mirror. Retry this exact plan after failure."""
    path=Path(receipt_path)
    receipt={'schema_version':1,'operation_id':plan.request['operation_id'],'artifact_id':plan.request['artifact_id'],
        'source_audit':plan.source_audit,'target':plan.request['target'],'payload_bytes':plan.request['payload_bytes'],
        'scientific_identities':{m['artifact_id']:m['metadata']['provenance_v1']['scientific_identity'] for m in plan.request['manifests']},
        'hardware_requalification':False,'no_recomputation':True,'status':'planned'}
    def save():
        path.parent.mkdir(parents=True,exist_ok=True);atomic_replace(path,lambda temp:temp.write_text(json.dumps(receipt,indent=2)+'\n'))
    save()
    try:
        for raw in plan.request.get('shared_dependencies', ()):
            shared = ArtifactManifest.from_dict(raw)
            registered = catalog.load_artifact(shared.artifact_id)
            if registered is None or registered.metadata.get('provenance_v1') != shared.metadata.get('provenance_v1') or registered.metadata.get('completion_v1') != shared.metadata.get('completion_v1'):
                raise ValueError('Shared dependency is not registered with the verified identity/completion')
        transport.stage(plan)
        manifests=[ArtifactManifest.from_dict(m) for m in transport.finalize(plan.request)]
        if [m.artifact_id for m in manifests]!=[m['artifact_id'] for m in plan.request['manifests']]:raise ValueError('Finalizer returned unexpected artifact coverage')
        for actual,expected in zip(manifests,plan.request['manifests']):
            if actual.metadata['provenance_v1']!=expected['metadata']['provenance_v1']:raise ValueError('Finalizer changed scientific identity')
    except BaseException as exc:
        receipt.update(status='transfer_or_finalization_failed',error_type=type(exc).__name__);save();raise
    receipt['status']='storage_complete_catalog_pending';save()
    for manifest in manifests:
        prior=catalog.load_artifact(manifest.artifact_id)
        if prior is not None and prior.to_dict()!=manifest.to_dict():raise ValueError('Catalog already contains conflicting publication identity')
        catalog.record_artifact(manifest)
    receipt.update(status='published',published_artifact_ids=[m.artifact_id for m in manifests]);save()
    return receipt


def _load_local(path):
    from pipelines_v2.storage.artifacts import artifact_from_manifest
    from pipelines_v2.storage.local import LocalArtifactStore
    p=Path(path).resolve();manifest=ArtifactManifest.from_dict(json.loads(p.read_text()))
    from pipelines_v2.storage.inference import artifact_store_from_manifest, first_storage_ref
    if first_storage_ref(manifest.storage_refs).get('store') == 'modal_volume':
        return artifact_from_manifest(manifest,store=artifact_store_from_manifest(manifest))
    if p.parent.name!=manifest.artifact_id:raise ValueError('Manifest must reside in its artifact-ID directory')
    return artifact_from_manifest(manifest,store=LocalArtifactStore(p.parent.parent))


def main():
    import argparse
    from pipelines_v2.core.env import load_dotenv_if_present
    from pipelines_v2.core.paths import find_workspace_root
    load_dotenv_if_present(search_roots=(find_workspace_root(),))
    from pipelines_v2.storage.publication_modal import ModalPublicationTransport
    from pipelines_v2.storage.composite import CompositeCatalog
    from pipelines_v2.storage.postgres import PostgresCatalog
    from pipelines_v2.storage.local import FileCatalog
    from pipelines_v2.data.sources import PostgresSource
    from pipelines_v2.core.paths import pipelines_v2_catalog_root
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest');source.add_argument('--artifact-id',help='Existing shared catalog artifact; mounted copy, no tensor download')
    p.add_argument('--dependency-manifest',action='append',default=[])
    p.add_argument('--dependency-artifact-id',action='append',default=[])
    p.add_argument('--volume',required=True);p.add_argument('--root',required=True);p.add_argument('--max-bytes',type=int,required=True)
    p.add_argument('--plan-output',required=True);p.add_argument('--receipt');p.add_argument('--publish',action='store_true')
    p.add_argument('--catalog-env',default='XENON_NEON_DATABASE_URL');p.add_argument('--mirror-root')
    p.add_argument('--result-only',action='store_true',help='Explicit typed summary projection; omit rendered reports/raw diagnostics/per-example tensors')
    p.add_argument('--max-json-bytes',type=int,default=16*1024**2)
    p.add_argument('--max-artifacts',type=int,default=128)
    p.add_argument('--max-metadata-bytes',type=int,default=1024**2)
    a=p.parse_args()
    catalog=CompositeCatalog((PostgresCatalog(PostgresSource.from_env(a.catalog_env)),FileCatalog(a.mirror_root or pipelines_v2_catalog_root()))) if a.publish or a.artifact_id or a.dependency_artifact_id else None
    def from_catalog(key):
        from pipelines_v2.storage.artifacts import artifact_from_manifest
        from pipelines_v2.storage.inference import artifact_store_from_manifest
        manifest=catalog.load_artifact(key)
        if manifest is None:raise ValueError('Missing shared artifact: '+key)
        return artifact_from_manifest(manifest,store=artifact_store_from_manifest(manifest))
    plan=plan_publication(from_catalog(a.artifact_id) if a.artifact_id else _load_local(a.manifest),
        dependencies=(*(_load_local(v) for v in a.dependency_manifest),*(from_catalog(v) for v in a.dependency_artifact_id)),
        target=PublicationTarget(a.volume,a.root),max_payload_bytes=a.max_bytes,result_only=a.result_only,
        max_json_bytes=a.max_json_bytes,max_artifacts=a.max_artifacts,max_metadata_bytes=a.max_metadata_bytes)
    Path(a.plan_output).write_text(json.dumps(plan.to_dict(),indent=2)+'\n')
    if not a.publish:print(json.dumps({'operation_id':plan.request['operation_id'],'payload_bytes':plan.request['payload_bytes'],'status':'admitted_no_upload'}));return
    if not a.receipt:p.error('--publish requires --receipt')
    print(json.dumps(publish_artifact(plan,transport=ModalPublicationTransport(),catalog=catalog,receipt_path=a.receipt),indent=2))


if __name__=='__main__':main()
