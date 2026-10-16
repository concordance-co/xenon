from dataclasses import dataclass, replace
from pathlib import Path
import json
import numpy as np
import pytest
from pipelines_v2.api import ArtifactManifest, LocalArtifactStore, FileCatalog
from pipelines_v2.storage.artifacts import artifact_from_manifest
from pipelines_v2.storage.finalization import finalize_manifest, validate_completion
from pipelines_v2.storage.publication import PublicationTarget, plan_publication, publish_artifact
from pipelines_v2.storage.publication_modal import finalize_publication_request
from tests.pipelines_v2.storage.test_finalization import identity


@dataclass(frozen=True)
class Mounted(LocalArtifactStore):
    kind: str='modal_volume'
    def write_json(self,*a,**kw):return {**super().write_json(*a,**kw),'name':'fake'}


class Transport:
    def __init__(self,target):self.target=target;self.interrupt=False;self.corrupt=False
    def stage(self,plan):
        directory=Path(self.target.root)/'.publication_stage'/plan.request['operation_id'];directory.mkdir(parents=True,exist_ok=True)
        for digest,data in plan.blobs.items():
            (directory/digest).write_bytes(b'bad' if self.corrupt else data.read_bytes() if isinstance(data,Path) else data)
            if self.interrupt:raise OSError('interrupted upload')
    def finalize(self,request):return finalize_publication_request(request,Mounted(self.target.root))


def fixture(root,key='local',parents=(),payload=None):
    store=LocalArtifactStore(root)
    tensor=store.write_safetensors(key,'features/a.safetensors',{'x':np.arange(4,dtype='float32')})
    metadata=store.write_json(key,'features/a.json',{'kind':'fixture','tensor_ref':tensor})
    result=store.write_json(key,'result.json',payload or {'kind':'generation_result','rows':[]})
    label=store.write_json(key,'labels/label.json',{'values':{'a':1}})
    p=identity();assert p.identity_complete
    manifest=ArtifactManifest(key,'generation_run',1,'spec','semantic','2026-10-16T00:00:00Z',{}, {'kind':'local'},tuple(parents),
        {'example_count':1,'example_keys':['a']}, {'result':result,'nested':{'features':{'a':dict(metadata,tensor_path=tensor['path'],metadata_path=metadata['path'],tensor_bytes=tensor['bytes'],metadata_bytes=metadata['bytes'],bytes=tensor['bytes']+metadata['bytes'])}},'labels':{'x':label}},
        {'provenance_v1':p.to_dict()})
    return artifact_from_manifest(finalize_manifest(store,manifest),store=store)


def target(tmp_path):return PublicationTarget('fake',str(tmp_path/'remote'/'artifacts'))


def test_nested_refs_identity_and_source_are_preserved(tmp_path):
    source=fixture(tmp_path/'local');before={str(p):p.read_bytes() for p in (tmp_path/'local').rglob('*') if p.is_file()}
    t=target(tmp_path);plan=plan_publication(source,target=t,max_payload_bytes=16384)
    catalog=FileCatalog(tmp_path/'catalog');transport=Transport(t)
    receipt=publish_artifact(plan,transport=transport,catalog=catalog,receipt_path=tmp_path/'receipt.json')
    remote=catalog.load_artifact(receipt['artifact_id']);validate_completion(remote,store=Mounted(t.root))
    assert remote.metadata['provenance_v1']==source.manifest().metadata['provenance_v1']
    assert remote.metadata['publication_v1']['source_artifact_id']==source.id
    assert str(tmp_path/'local') not in json.dumps(remote.to_dict())
    feature=remote.storage_refs['nested']['features']['a']
    nested=json.loads(Path(feature['path']).read_text())['tensor_ref']
    assert nested['store']=='modal_volume' and nested['path'].startswith(t.root)
    assert Path(feature['tensor_path']).read_bytes()==Path(source.manifest().storage_refs['nested']['features']['a']['tensor_path']).read_bytes()
    assert publish_artifact(plan,transport=transport,catalog=catalog,receipt_path=tmp_path/'receipt.json')==receipt
    assert before=={str(p):p.read_bytes() for p in (tmp_path/'local').rglob('*') if p.is_file()}


def test_interruption_checksum_and_catalog_failure_are_recoverable(tmp_path):
    source=fixture(tmp_path/'local');t=target(tmp_path);plan=plan_publication(source,target=t,max_payload_bytes=16384)
    transport=Transport(t);catalog=FileCatalog(tmp_path/'catalog');receipt=tmp_path/'receipt.json'
    transport.interrupt=True
    with pytest.raises(OSError):publish_artifact(plan,transport=transport,catalog=catalog,receipt_path=receipt)
    assert not list(Path(t.root).glob('published_*/manifest.json'))
    transport.interrupt=False;transport.corrupt=True
    with pytest.raises(ValueError,match='checksum'):publish_artifact(plan,transport=transport,catalog=catalog,receipt_path=receipt)
    assert not list(Path(t.root).glob('published_*/manifest.json'))
    transport.corrupt=False
    class Fails:
        def load_artifact(self,key):return None
        def record_artifact(self,m):raise OSError('catalog unavailable')
    with pytest.raises(OSError):publish_artifact(plan,transport=transport,catalog=Fails(),receipt_path=receipt)
    assert json.loads(receipt.read_text())['status']=='storage_complete_catalog_pending'
    assert len(list(Path(t.root).glob('published_*/manifest.json')))==1
    assert publish_artifact(plan,transport=transport,catalog=catalog,receipt_path=receipt)['status']=='published'


def test_dependency_closure_and_rewritten_ids(tmp_path):
    parent=fixture(tmp_path/'local','parent');child=fixture(tmp_path/'local','child',parents=('parent',),payload={'artifact_id':'parent'})
    with pytest.raises(ValueError,match='Missing explicit dependency'):plan_publication(child,target=target(tmp_path),max_payload_bytes=16384)
    plan=plan_publication(child,dependencies=(parent,),target=target(tmp_path),max_payload_bytes=16384)
    assert len(plan.request['manifests'])==2
    assert plan.request['manifests'][1]['input_artifact_refs']==[plan.request['manifests'][0]['artifact_id']]
    catalog=FileCatalog(tmp_path/'catalog')
    receipt=publish_artifact(plan,transport=Transport(target(tmp_path)),catalog=catalog,receipt_path=tmp_path/'receipt.json')
    remote=catalog.load_artifact(receipt['artifact_id'])
    assert json.loads(Path(remote.storage_refs['result']['path']).read_text())['artifact_id']==remote.input_artifact_refs[0]


def test_admission_rejects_oversize_unresolved_classes_and_ambiguous_paths(tmp_path):
    source=fixture(tmp_path/'local');t=target(tmp_path)
    with pytest.raises(ValueError,match='ceiling'):plan_publication(source,target=t,max_payload_bytes=10)
    unknown=artifact_from_manifest(replace(source.manifest(),artifact_kind='capture'),store=source.store)
    with pytest.raises(ValueError,match='data class'):plan_publication(unknown,target=t,max_payload_bytes=16384)
    unresolved=artifact_from_manifest(replace(source.manifest(),metadata={}),store=source.store)
    with pytest.raises(ValueError,match='provenance'):plan_publication(unresolved,target=t,max_payload_bytes=16384)
    dangling=fixture(tmp_path/'other',payload={'path':'/outside/not-declared.json'})
    with pytest.raises(ValueError,match='Undeclared'):plan_publication(dangling,target=t,max_payload_bytes=16384)


def test_tampered_finalizer_request_rejected(tmp_path):
    source=fixture(tmp_path/'local');t=target(tmp_path);plan=plan_publication(source,target=t,max_payload_bytes=16384)
    transport=Transport(t);transport.stage(plan)
    plan.request['payloads'][0]['destination']='/outside/result.json'
    with pytest.raises(ValueError,match='escapes'):transport.finalize(plan.request)


def test_source_escape_and_payload_reference_to_missing_artifact_fail(tmp_path):
    source=fixture(tmp_path/'local');outside=LocalArtifactStore(tmp_path/'outside').write_json('other','foreign.json',{})
    bad=replace(source.manifest(),artifact_id='escape',storage_refs={'result':outside},metadata={'provenance_v1':source.manifest().metadata['provenance_v1']})
    completed=finalize_manifest(source.store,bad)
    with pytest.raises(ValueError,match='escapes'):plan_publication(artifact_from_manifest(completed,store=source.store),target=target(tmp_path),max_payload_bytes=16384)
    missing=fixture(tmp_path/'missing',payload={'artifact_id':'not-in-closure'})
    with pytest.raises(ValueError,match='outside the supplied'):plan_publication(missing,target=target(tmp_path),max_payload_bytes=16384)


def test_disk_reloaded_source_has_same_publication_identity(tmp_path):
    from pipelines_v2.storage.publication import _load_local
    source=fixture(tmp_path/'local');loaded=_load_local(tmp_path/'local'/source.id/'manifest.json')
    a=plan_publication(source,target=target(tmp_path),max_payload_bytes=16384)
    b=plan_publication(loaded,target=target(tmp_path),max_payload_bytes=16384)
    assert a.request==b.request


def typed_fixture(root, kind, payload, *, key='typed', parents=(), features=None):
    from pipelines_v2.storage.features import write_capture_features
    from pipelines_v2.representation.sae.codec import register_codec
    register_codec()
    store = LocalArtifactStore(root)
    refs = {'result':store.write_json(key, 'result.json', payload)}
    if features: refs['features'] = write_capture_features(store,key,features)
    m = ArtifactManifest(key,kind,1,'spec','semantic','2026-10-16T00:00:00Z',{}, {'kind':'local'},tuple(parents),
        {'example_count':0},refs,{'provenance_v1':identity().to_dict()})
    return artifact_from_manifest(finalize_manifest(store,m),store=store)


def graph_payload():
    from pipelines_v2.operations.graphs import AttributionGraph, GraphNode
    graph = AttributionGraph(source_run_ids=('fixture',), provenance={'scope':'engineering'},target={'token_id':1},
        model_identity='fixture-model', algorithm={'name':'fixture'}, semantics='replacement_flow',
        nodes=(GraphNode.create('output', {'token_id':1}, 'target'),), edges=(), evidence=(), approximations=(),
        assumptions={'scope':'plumbing'},pruning={'none':True},omitted_contributions={'none':True})
    return {'graph':graph.to_dict(), 'scientific_verdict':'not_tested'}


def test_typed_graph_publication_preserves_graph_and_c01(tmp_path):
    payload = graph_payload()
    source = typed_fixture(tmp_path/'local','attribution_graph',payload)
    plan = plan_publication(source,target=target(tmp_path),max_payload_bytes=16384)
    catalog = FileCatalog(tmp_path/'catalog')
    receipt = publish_artifact(plan,transport=Transport(target(tmp_path)),catalog=catalog,receipt_path=tmp_path/'receipt.json')
    published = artifact_from_manifest(catalog.load_artifact(receipt['artifact_id']),store=Mounted(target(tmp_path).root))
    assert published.result() == payload
    assert published.manifest().provenance().to_dict() == source.manifest().provenance().to_dict()
    assert published.manifest().metadata['publication_v1']['view'] == 'complete_artifact'
    malformed = json.loads(json.dumps(payload)); malformed['graph']['graph_id']='0'*64
    bad = typed_fixture(tmp_path/'bad','attribution_graph',malformed)
    with pytest.raises(ValueError,match='Graph identity'):plan_publication(bad,target=target(tmp_path),max_payload_bytes=16384)


def test_dictionary_larger_than_old_limit_streams_and_roundtrips(tmp_path):
    from pipelines_v2.representation.sae.dictionary import SAEDictionary
    width = 1500000  # 18MB FP32 weights: explicit budget, no activation capture.
    metadata = {'schema_version':1,'type':'sae','architecture':'standard','activation':'relu','normalization':'none'}
    weights = {'W_enc':np.zeros((1,width),np.float32),'W_dec':np.zeros((width,1),np.float32),
               'b_enc':np.zeros(width,np.float32),'b_dec':np.zeros(1,np.float32)}
    source = typed_fixture(tmp_path/'local','sae_dictionary_import',metadata,
        features={'dictionary':{'kind':'sae_tensors','weights':weights,'metadata':metadata}})
    with pytest.raises(ValueError,match='ceiling'):
        plan_publication(source,target=target(tmp_path),max_payload_bytes=16*1024**2)
    plan = plan_publication(source,target=target(tmp_path),max_payload_bytes=24*1024**2)
    assert plan.request['payload_bytes'] > 16*1024**2
    assert any(isinstance(blob,Path) for blob in plan.blobs.values())
    assert sum(len(blob) for blob in plan.blobs.values() if isinstance(blob,bytes)) < 16384
    catalog = FileCatalog(tmp_path/'catalog')
    receipt = publish_artifact(plan,transport=Transport(target(tmp_path)),catalog=catalog,receipt_path=tmp_path/'receipt.json')
    remote = artifact_from_manifest(catalog.load_artifact(receipt['artifact_id']),store=Mounted(target(tmp_path).root))
    feature = remote.feature('dictionary').load()
    dictionary = SAEDictionary(feature['weights'],feature['metadata'])
    assert dictionary.d_sae == width
    assert np.array_equal(dictionary.encode([[1.]]), np.zeros((1,width),np.float32))


def test_report_view_is_explicit_partial_and_omits_local_presentation(tmp_path):
    source = typed_fixture(tmp_path/'local','report',{'kind':'report_result','template':'summary',
        'output_dir':'/private/report/path','inputs':[], 'summary':{'input_count':0}})
    with pytest.raises(ValueError,match='result_only'):
        plan_publication(source,target=target(tmp_path),max_payload_bytes=16384)
    plan = plan_publication(source,target=target(tmp_path),max_payload_bytes=16384,result_only=True)
    remote = plan.request['manifests'][0]
    assert remote['artifact_kind'] == 'report_summary'
    assert remote['metadata']['publication_v1']['view'] == 'result_summary'
    payload = json.loads(plan.blobs[plan.request['payloads'][0]['sha256']])
    assert 'output_dir' not in payload
    assert remote['metadata']['provenance_v1'] == source.manifest().metadata['provenance_v1']
    assert remote['metadata']['publication_v1']['source_completion_digest'] == source.manifest().metadata['completion_v1']['integrity_hash']


def test_arbitrary_transform_and_local_capture_dependencies_stay_rejected(tmp_path):
    arbitrary = typed_fixture(tmp_path/'local','transform',{'blob':'not a declared result schema'})
    with pytest.raises(ValueError,match='typed publication'):
        plan_publication(arbitrary,target=target(tmp_path),max_payload_bytes=16384)
    capture = typed_fixture(tmp_path/'local','crosscoder_pair_capture',{},key='capture')
    derived = typed_fixture(tmp_path/'local','sae_encode',{'dictionary_id':'capture'},key='derived',parents=('capture',))
    with pytest.raises(ValueError,match='bulk'):
        plan_publication(derived,dependencies=(capture,),target=target(tmp_path),max_payload_bytes=16384,result_only=True)


def test_json_and_manifest_budgets_are_enforced_on_both_sides(tmp_path):
    source=fixture(tmp_path/'local');t=target(tmp_path)
    with pytest.raises(ValueError,match='JSON'):
        plan_publication(source,target=t,max_payload_bytes=16384,max_json_bytes=10)
    with pytest.raises(ValueError,match='metadata'):
        plan_publication(source,target=t,max_payload_bytes=16384,max_metadata_bytes=10)
    plan=plan_publication(source,target=t,max_payload_bytes=16384)
    transport=Transport(t);transport.stage(plan)
    plan.request['max_json_bytes']=10
    with pytest.raises(ValueError,match='JSON'):
        transport.finalize(plan.request)
    assert not list(Path(t.root).glob('published_*/manifest.json'))


def test_shared_capture_dependency_is_referenced_not_uploaded(tmp_path):
    from types import SimpleNamespace
    from pipelines_v2.storage.publication import _read_source_json
    t=target(tmp_path)
    # A preexisting shared capture is a closed upstream dependency, not eligible
    # for upload from local storage. Fixture bytes stand in for its mounted store.
    shared_store=Mounted(tmp_path/'shared')
    ref=shared_store.write_json('capture','result.json',{'scope':'shared capture metadata'})
    m=ArtifactManifest('capture','capture',1,'s','s','now',{}, {},(),{}, {'result':ref}, {'provenance_v1':identity().to_dict()})
    m=finalize_manifest(shared_store,m)
    shared=artifact_from_manifest(m,store=SimpleNamespace(kind='modal_volume',name='fake',root=shared_store.root))
    source=typed_fixture(tmp_path/'local','transform',{'kind':'neuron_dossier_result','schema_version':1,'summary':{},'dossiers':[], 'source_artifact_id':'capture'},parents=('capture',))
    plan=plan_publication(source,dependencies=(shared,),target=t,max_payload_bytes=16384)
    assert len(plan.request['shared_dependencies'])==1 and len(plan.request['manifests'])==1
    assert plan.request['manifests'][0]['input_artifact_refs']==['capture']
    assert all(row['source_artifact_id']==source.id for row in plan.request['payloads'])
    catalog=FileCatalog(tmp_path/'catalog');catalog.record_artifact(m)
    receipt=publish_artifact(plan,transport=Transport(t),catalog=catalog,receipt_path=tmp_path/'receipt.json')
    assert receipt['status']=='published'
    assert catalog.load_artifact('capture').to_dict()==m.to_dict()


def test_modal_primary_uses_mounted_tensor_copy_without_client_download(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import pipelines_v2.storage.publication as publication
    metadata={'schema_version':1,'type':'sae','architecture':'standard','activation':'relu','normalization':'none'}
    weights={'W_enc':np.zeros((2,3),np.float32),'W_dec':np.zeros((3,2),np.float32),'b_enc':np.zeros(3,np.float32),'b_dec':np.zeros(2,np.float32)}
    original=typed_fixture(tmp_path/'local','sae_dictionary_import',metadata,features={'dictionary':{'kind':'sae_tensors','weights':weights,'metadata':metadata}})
    original = artifact_from_manifest(replace(original.manifest(), workflow_context={'run_id':'source-run', 'workflow_hash':'a'*64, 'step_name':'dictionary'}), store=original.store)
    first_target=target(tmp_path)
    first=plan_publication(original,target=first_target,max_payload_bytes=16384)
    catalog=FileCatalog(tmp_path/'catalog')
    receipt=publish_artifact(first,transport=Transport(first_target),catalog=catalog,receipt_path=tmp_path/'one.json')
    m=catalog.load_artifact(receipt['artifact_id'])
    source=artifact_from_manifest(m,store=SimpleNamespace(kind='modal_volume',name='fake',root=first_target.root))
    reads=[]
    def read_json(item,path,descriptor,limit):
        assert path.endswith('.json') and descriptor['bytes']<=limit
        reads.append(path)
        return Path(path).read_bytes()
    monkeypatch.setattr(publication,'_read_source_json',read_json)
    second_target=PublicationTarget('fake',str(tmp_path/'another'/'publication'))
    plan=plan_publication(source,target=second_target,max_payload_bytes=16384)
    copied=[row for row in plan.request['payloads'] if 'remote_source' in row]
    assert any(row['destination'].endswith('.safetensors') for row in copied)
    assert all(not isinstance(blob,Path) for blob in plan.blobs.values())
    second=publish_artifact(plan,transport=Transport(second_target),catalog=catalog,receipt_path=tmp_path/'two.json')
    remote=catalog.load_artifact(second['artifact_id'])
    assert remote.metadata['provenance_v1']==m.metadata['provenance_v1']
    validate_completion(remote,store=Mounted(second_target.root))
    assert remote.metadata['publication_v1']['prior_publication']==m.metadata['publication_v1']
    assert remote.metadata['publication_v1']['source_workflow']==original.manifest().workflow_context
    assert reads and all(name.endswith('.json') for name in reads)
    plan.request['payloads'][0]['remote_source']={'name':'other-volume','path':'/forged'}
    with pytest.raises(ValueError,match='source'):
        Transport(second_target).finalize(plan.request)


def test_official_attribution_is_an_explicit_compact_view(tmp_path):
    from tests.pipelines_v2.operations.test_external_mattr import result_fixture
    payload=result_fixture()
    payload['effective_compute']['argv']=['/private/tool/python','/private/request.json']
    payload['raw_outputs']={'scores.pt':{'encoding':'base64','bytes':3,'sha256':'0'*64,'data':'YWJj'}}
    source=typed_fixture(tmp_path/'local','external_attribution',payload)
    with pytest.raises(ValueError,match='result_only'):
        plan_publication(source,target=target(tmp_path),max_payload_bytes=65536)
    plan=plan_publication(source,target=target(tmp_path),max_payload_bytes=65536,result_only=True)
    m=plan.request['manifests'][0]
    value=json.loads(plan.blobs[plan.request['payloads'][0]['sha256']])
    assert m['artifact_kind']=='external_attribution_summary'
    assert 'argv' not in value['effective_compute'] and 'data' not in value['raw_outputs']['scores.pt']
    assert value['request_identity']==payload['request_identity']
    assert value['nodes']==payload['nodes'] and value['points']==payload['points']
    assert m['metadata']['publication_v1']['omitted_fields']==['raw_outputs.*.data','effective_compute.argv']


def test_derived_codes_publish_summary_without_per_example_tensors(tmp_path):
    metadata={'schema_version':1,'type':'sae','architecture':'standard','activation':'relu','normalization':'none'}
    weights={'W_enc':np.zeros((2,3),np.float32),'W_dec':np.zeros((3,2),np.float32),'b_enc':np.zeros(3,np.float32),'b_dec':np.zeros(2,np.float32)}
    dictionary=typed_fixture(tmp_path/'local','sae_dictionary_import',metadata,key='dictionary',
        features={'dictionary':{'kind':'sae_tensors','weights':weights,'metadata':metadata}})
    source=typed_fixture(tmp_path/'local','sae_encode',{'dictionary_id':'dictionary','fidelity':{'mse':0.1}},parents=('dictionary',),
        features={'codes':{'kind':'sae_tensors','records':{'example':{'values':np.zeros((2,3),np.float32)}}}})
    with pytest.raises(ValueError,match='activation data'):
        plan_publication(source,dependencies=(dictionary,),target=target(tmp_path),max_payload_bytes=65536)
    plan=plan_publication(source,dependencies=(dictionary,),target=target(tmp_path),max_payload_bytes=65536,result_only=True)
    m=plan.request['manifests'][-1]
    assert m['artifact_kind']=='sae_encode_summary' and set(m['storage_refs'])=={'result'}
    assert m['metadata']['publication_v1']['omitted_payloads']
    assert all(not row['destination'].endswith('.safetensors') for row in plan.request['payloads'] if row['source_artifact_id']==source.id)
