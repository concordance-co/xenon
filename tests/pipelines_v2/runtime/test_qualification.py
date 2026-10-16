from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json

import pytest

from pipelines_v2.api import LocalArtifactStore, ToyEngine, GenerationRunSpec, GenerationSpec, Dataset, Example
from pipelines_v2.core.provenance import resolve_provenance
from pipelines_v2.core.types import utc_now_iso
from pipelines_v2.engine.diffusers.session import ImageRequest
from pipelines_v2.runtime.admission import workload_envelope
from pipelines_v2.runtime.qualification import CanaryGate, QualificationFailed, assess_measurements, profile_identity, session_key
from pipelines_v2.storage.artifacts import ArtifactManifest
from pipelines_v2.storage.finalization import finalize_manifest, validate_completion


def measurements():
    return {'loading': {'peak_rss_bytes': 100, 'peak_reserved_vram_bytes': 100, 'seconds': 1,
            'total_vram_bytes': 1000, 'driver_version': 'fixture-driver', 'device': 'fixture-gpu'},
        'peak_rss_bytes': 200, 'peak_reserved_vram_bytes': 200, 'seconds': 2}


@pytest.fixture
def fixture(tmp_path):
    toy = GenerationRunSpec(engine=ToyEngine(), dataset=Dataset.from_examples([Example(key='a',prompt='a')]), generation=GenerationSpec(enabled=True,max_tokens=1))
    provenance = resolve_provenance(toy)
    spec = SimpleNamespace(images=(ImageRequest('a','a',width=64,height=64,steps=2),), engine=toy.engine,
                           spec_hash=toy.spec_hash, semantic_hash=toy.semantic_hash)
    context = {'provenance_v1': provenance.to_dict(), 'step_name': 'fixture',
               'admission_v1': {'provenance_v1': provenance.to_dict(), 'envelope': workload_envelope(toy,{'kind':'local'}).to_dict()}}
    store = LocalArtifactStore(tmp_path)
    manifest = ArtifactManifest(artifact_id='useful-image',artifact_kind='diffusion_image',schema_version=1,
        operation_spec_hash=toy.spec_hash(),operation_semantic_hash=toy.semantic_hash(),created_at=utc_now_iso(),
        engine=toy.engine.identity(),runner={'kind':'fixture'},input_artifact_refs=(),
        example_coverage={'example_count':1,'example_keys':['a']},
        storage_refs={'result':store.write_json('useful-image','result.json',{'fixture':True})},
        metadata={**measurements(),'provenance_v1':provenance.to_dict()})
    manifest=finalize_manifest(store,manifest)
    return dict(store=store,spec=spec,runner_config={'kind':'fixture','resources':{'memory_mb':1}},context=context),manifest


def test_useful_image_is_canary_and_persisted_pass_gates_continuation(fixture):
    kwargs, image = fixture
    sequence=[]
    with CanaryGate(**kwargs,progress_callback=lambda event: sequence.append(event['stage'])).owner() as gate:
        sequence.append('useful-image')
        receipt=gate.observe(image)
        assert validate_completion(ArtifactManifest.from_dict(receipt),store=kwargs['store'])
        sequence.append('next-image')
    assert sequence==['qualification_pending','useful-image','qualification_passed','next-image']
    with CanaryGate(**kwargs).owner() as gate:
        assert gate.receipt['artifact_id']==receipt['artifact_id']
        assert gate.observe(image)['artifact_id']==receipt['artifact_id']  # no second canary or duplicate artifact
    assert len(list(kwargs['store'].root.glob('qualification_*')))==1


def test_failure_persists_diagnostic_but_never_releases_next_image(fixture):
    kwargs,image=fixture
    image=replace(image,metadata={**image.metadata,'peak_reserved_vram_bytes':950})
    submitted=[]
    with pytest.raises(QualificationFailed,match='no remaining images'):
        with CanaryGate(**kwargs).owner() as gate:
            submitted.append('first')
            gate.observe(image)
            submitted.append('second')
    assert submitted==['first']
    assert not (kwargs['store'].root/'_qualification_profiles').exists()
    receipt=json.loads(next(kwargs['store'].root.glob('qualification_*/result.json')).read_text())
    assert receipt['measurements']['status']=='failed'


@pytest.mark.parametrize('field',['peak_rss_bytes','peak_reserved_vram_bytes','seconds'])
def test_missing_or_estimated_measurements_never_qualify(field):
    values=measurements();values.pop(field)
    assert assess_measurements(values,host_limit_mb=1)['status']=='unresolved'
    values[field]=float('nan')
    assert assess_measurements(values,host_limit_mb=1)['status']=='unresolved'


def test_exact_profile_changes_for_every_envelope_dimension(fixture):
    kwargs,_=fixture; admission=kwargs['context']['admission_v1']; expected=profile_identity(admission)
    for key in admission['envelope']:
        changed=deepcopy(admission);changed['envelope'][key]={'changed':True}
        assert profile_identity(changed)!=expected,key
    for key in ('source','runtime','model_components','numerics'):
        changed=deepcopy(admission);changed['provenance_v1'][key]={'changed':True}
        assert profile_identity(changed)!=expected,key
    from pipelines_v2.core.provenance import ProvenanceV1
    assert session_key(ProvenanceV1.from_dict(admission['provenance_v1']),{})


def test_canary_rejects_unmeasured_heterogeneous_requests(fixture):
    kwargs,_=fixture
    kwargs['spec'].images+=(ImageRequest('b','b',width=128,height=64,steps=2),)
    with pytest.raises(ValueError,match='homogeneous'):CanaryGate(**kwargs)


def test_corrupt_receipt_never_releases_fanout(fixture):
    kwargs,image=fixture
    with CanaryGate(**kwargs).owner() as gate:gate.observe(image)
    path=next(kwargs['store'].root.glob('qualification_*/result.json'))
    path.write_text('{}')
    with pytest.raises(Exception,match='integrity|checksum|payload|size|mismatch'):
        with CanaryGate(**kwargs).owner():pytest.fail('released corrupt receipt')


def test_generation_only_measurement_cannot_qualify_requested_hook(fixture):
    from pipelines_v2.engine.diffusers.hooks import DenoisingHook
    kwargs,image=fixture
    kwargs['spec'].hooks=(DenoisingHook('mid_block.resnets.1',(1,),(1280,32,32),operator='ablate'),)
    with pytest.raises(QualificationFailed,match='hook/denoising coverage'):
        with CanaryGate(**kwargs).owner() as gate:gate.observe(image)


def test_session_key_ignores_input_but_invalidates_numerics_resources_and_source(fixture):
    from pipelines_v2.core.provenance import ProvenanceV1
    kwargs,_=fixture
    body=deepcopy(kwargs['context']['provenance_v1'])
    provenance=ProvenanceV1.from_dict(body)
    identity=session_key(provenance,kwargs['runner_config'])
    for key in ('scientific_identity','worker_compatibility','schema_version'):body.pop(key,None)
    changed=deepcopy(body);changed['seeds']={'changed':123}
    assert session_key(ProvenanceV1(**changed),kwargs['runner_config'])==identity
    for field in ('source','numerics','runtime'):
        changed=deepcopy(body);changed[field]['fixture_changed']=True
        assert session_key(ProvenanceV1(**changed),kwargs['runner_config'])!=identity
    assert session_key(provenance,{'resources':{'memory_mb':2}})!=identity


def test_failed_real_path_records_failure_without_qualifying_or_repeating(fixture):
    kwargs,_=fixture
    with pytest.raises(RuntimeError,match='fixture OOM'):
        with CanaryGate(**kwargs).owner():raise RuntimeError('fixture OOM')
    result=json.loads(next(kwargs['store'].root.glob('qualification_*/result.json')).read_text())
    assert result['measurements']=={'status':'failed','reason':'canary execution failed','error_type':'RuntimeError'}
    assert result['sample_image'] is None
    assert not (kwargs['store'].root/'_qualification_profiles').exists()
