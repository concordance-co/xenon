"""Client-side Modal uploads followed by CPU-only mounted C03 finalization."""
import hashlib
import io
import json
from pathlib import Path
import shutil
from pipelines_v2.core.provenance import content_hash
from pipelines_v2.storage.artifacts import ArtifactManifest
from pipelines_v2.storage.finalization import atomic_replace, finalize_manifest, validate_completion, writer_lock


def _digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()



def _verify_mounted_manifest(manifest, store, target):
    mount = Path(target.root).parts[1]
    def check_refs(value):
        if isinstance(value, dict):
            if 'store' in value and 'path' in value:
                path=Path(value['path'])
                if not path.is_absolute() or '..' in path.parts or len(path.parts)<3 or path.parts[1]!=mount or path.resolve()!=path:
                    raise ValueError('Remote source path escapes its mounted volume')
                if value['store'] != 'modal_volume' or value.get('name', target.name) != target.name:
                    raise ValueError('Remote source must remain on the mounted target volume')
            for child in value.values(): check_refs(child)
        elif isinstance(value, list):
            for child in value: check_refs(child)
    check_refs(manifest.storage_refs)
    if manifest.provenance() is None or not manifest.provenance().identity_complete:
        raise ValueError('Unresolved remote source identity')
    validate_completion(manifest, store=store)

def finalize_publication_request(request, store):
    """Runs only with the destination volume mounted (or explicit fake in tests)."""
    from .publication import PublicationTarget, _positive_budget
    target=PublicationTarget(**request['target']);root=Path(target.root)
    if request.get('schema_version')!=1 or str(store.root)!=target.root or store.kind!='modal_volume':
        raise ValueError('Publication finalizer target/schema mismatch')
    op=request['operation_id']
    if not op.startswith('pub_') or Path(op).name!=op:raise ValueError('Unsafe publication operation ID')
    for name, default in (('max_payload_bytes', None), ('max_json_bytes', 16*1024**2), ('max_artifacts',32), ('max_metadata_bytes',1024**2)):
        _positive_budget(request.get(name, default), name)
    if len(json.dumps(request).encode()) > request.get('max_metadata_bytes',1024**2):
        raise ValueError('Publication request metadata ceiling exceeded')
    if any(type(p.get('bytes')) is not int or p['bytes'] < 0 for p in request['payloads']):
        raise ValueError('Invalid publication byte descriptor')
    if sum(p['bytes'] for p in request['payloads'])>request['max_payload_bytes']:
        raise ValueError('Publication exceeds mounted-worker byte ceiling')
    manifests=[ArtifactManifest.from_dict(p) for p in request['manifests']]
    shared=[ArtifactManifest.from_dict(p) for p in request.get('shared_dependencies',())]
    ids={m.artifact_id for m in manifests};known=set();destinations=set()
    if len(ids)!=len(manifests) or not 1<=len(ids) or len(ids)+len(shared)>request.get('max_artifacts',32):
        raise ValueError('Invalid publication manifest coverage')
    from pipelines_v2.storage.finalization import payload_paths
    for dependency in shared:
        if dependency.artifact_id in known or dependency.artifact_id in ids or not set(dependency.input_artifact_refs)<=known:
            raise ValueError('Invalid shared dependency closure')
        _verify_mounted_manifest(dependency,store,target)
        known.add(dependency.artifact_id)
    remote_sources={}
    for raw in request.get('remote_sources',()):
        source=ArtifactManifest.from_dict(raw)
        if source.artifact_id in remote_sources: raise ValueError('Duplicate remote source')
        _verify_mounted_manifest(source,store,target)
        remote_sources[source.artifact_id]=source
    for m in manifests:
        if not m.artifact_id.startswith('published_') or Path(m.artifact_id).name!=m.artifact_id or not set(m.input_artifact_refs)<=known:
            raise ValueError('Publication dependencies must be topologically closed')
        if m.provenance() is None or not m.provenance().identity_complete or m.metadata['publication_v1']['operation_id']!=op:
            raise ValueError('Publication manifest identity mismatch')
        known.add(m.artifact_id)
    for payload in request['payloads']:
        destination=Path(payload['destination'])
        if not destination.is_relative_to(root) or '..' in destination.parts or destination.relative_to(root).parts[0] not in ids or destination.name=='manifest.json':
            raise ValueError('Payload destination escapes declared publication roots')
        if str(destination) in destinations or type(payload['bytes'])is not int or payload['bytes']<0:
            raise ValueError('Duplicate or invalid payload')
        destinations.add(str(destination))
        digest=payload['sha256']
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid staged digest')
    from pipelines_v2.storage.finalization import payload_paths
    if {p for m in manifests for p in payload_paths(m.storage_refs)}!=destinations:
        raise ValueError('Manifest refs do not exactly cover uploaded payloads')
    if sum(p['bytes'] for p in request['payloads'] if p['destination'].endswith('.json')) > request.get('max_json_bytes',16*1024**2):
        raise ValueError('Aggregate publication JSON ceiling exceeded')
    if sum(p['bytes'] for p in request['payloads']) != request['payload_bytes']:
        raise ValueError('Publication byte total differs from descriptor inventory')
    def staged_path(payload):
        remote=payload.get('remote_source')
        if remote is None: return root/'.publication_stage'/op/payload['sha256']
        source=remote_sources.get(payload['source_artifact_id'])
        if source is None or remote.get('name') != target.name:
            raise ValueError('Unverified remote copy source')
        matching=[item for item in source.metadata['completion_v1']['payloads'] if item['path']==remote.get('path')]
        if len(matching)!=1 or any(matching[0][key]!=payload[key] for key in ('bytes','sha256')):
            raise ValueError('Remote copy differs from source completion inventory')
        return Path(remote['path'])
    with writer_lock(store,'publication:'+op):
        # Validate every staged byte before creating any final manifest.
        for payload in request['payloads']:
            staged=staged_path(payload)
            if not staged.is_file() or staged.stat().st_size!=payload['bytes'] or _digest(staged)!=payload['sha256']:
                raise ValueError('Staged publication checksum mismatch or missing upload')
        for payload in request['payloads']:
            if payload['destination'].endswith('.json') and payload['bytes'] > request.get('max_json_bytes',16*1024**2):
                raise ValueError('Publication JSON payload ceiling exceeded')
        for payload in request['payloads']:
            destination=Path(payload['destination']);destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                if destination.stat().st_size!=payload['bytes'] or _digest(destination)!=payload['sha256']:raise ValueError('Conflicting immutable destination payload')
            else:
                source=staged_path(payload)
                atomic_replace(destination,lambda temporary:shutil.copyfile(source,temporary))
        completed=[]
        for manifest in manifests:
            existing=root/manifest.artifact_id/'manifest.json'
            if existing.exists():
                prior=ArtifactManifest.from_dict(json.loads(existing.read_text()));validate_completion(prior,store=store)
                expected=manifest.to_dict();observed=prior.to_dict()
                observed['metadata'].pop('completion_v1',None);observed['storage_refs'].pop('manifest',None)
                if observed!=expected:raise ValueError('Conflicting immutable publication manifest')
            final=finalize_manifest(store,manifest)
            final.storage_refs['manifest']['name']=target.name
            validate_completion(final,store=store);completed.append(final.to_dict())
        return completed


class ModalPublicationTransport:
    def stage(self, plan):
        import modal
        target=plan.request['target'];volume=modal.Volume.from_name(target['name'])
        from pipelines_v2.storage.modal import modal_volume_mount_path
        mount=Path(modal_volume_mount_path(target['root']))
        relative=Path(target['root']).relative_to(mount)/'.publication_stage'/plan.request['operation_id']
        try:existing={Path(entry.path).name for entry in volume.listdir('/'+relative.as_posix())}
        except (FileNotFoundError,modal.exception.NotFoundError):existing=set()
        missing={digest:data for digest,data in plan.blobs.items() if digest not in existing}
        if missing:
            with volume.batch_upload(force=True) as batch:
                for digest,data in missing.items():
                    if isinstance(data,Path):
                        if _digest(data)!=digest: raise ValueError('Source tensor changed after publication plan')
                        source=data
                    else:
                        if hashlib.sha256(data).hexdigest()!=digest: raise ValueError('Planned JSON bytes changed')
                        source=io.BytesIO(data)
                    batch.put_file(source,'/'+(relative/digest).as_posix())

    def finalize(self, request):
        import modal
        from pipelines_v2.storage.modal import modal_volume_mount_path
        from pipelines_v2.runtime.deadlines import bounded_remote_call
        target=request['target'];volume=modal.Volume.from_name(target['name']);mount=modal_volume_mount_path(target['root'])
        image=modal.Image.debian_slim(python_version='3.13').pip_install('numpy==2.4.2','safetensors==0.7.0').add_local_python_source('pipelines_v2')
        app=modal.App('xenon-artifact-publication')
        @app.function(image=image,volumes={mount:volume},cpu=1,memory=512,timeout=600,max_containers=1,serialized=True)
        def finish(payload):
            from pipelines_v2.storage.modal import ModalVolumeStore
            from pipelines_v2.storage.publication_modal import finalize_publication_request
            volume.reload()
            return finalize_publication_request(payload,ModalVolumeStore(name=target['name'],root=target['root']))
        with app.run():return bounded_remote_call(finish,request,timeout_seconds=660)
