"""Generic remote artifacts retain independently observed worker admission."""
from pipelines_v2.api import Dataset,Example,LabelMapSpec,LocalArtifactStore
from pipelines_v2.core.provenance import resolve_provenance
from pipelines_v2.runtime.admission import AdmissionCheck,workload_envelope
from pipelines_v2.runtime.readiness import AdmissionRecord
from pipelines_v2.runtime.remote_executor import execute_remote
from pipelines_v2.storage.artifacts import ArtifactManifest
from pipelines_v2.storage.finalization import validate_completion


def test_generic_remote_publisher_preserves_actual_worker_admission(tmp_path,monkeypatch):
    dataset=Dataset.from_examples([Example(key='a',prompt='fixture',labels={'label':'yes'})])
    spec=LabelMapSpec(source=dataset.labels('label'),mapping={'yes':'positive'},output_name='mapped')
    identity={'kind':'local','admission_required':True}
    observed=resolve_provenance(spec,identity)
    assert observed.identity_complete
    worker_record=AdmissionRecord((AdmissionCheck('worker.fixture','readiness','passed','fixture','Observed worker fixture'),),observed,workload_envelope(spec,identity).to_dict())
    calls=[]
    def admit(*args,**kwargs):
        calls.append(args[0]);return worker_record
    monkeypatch.setattr('pipelines_v2.runtime.readiness.admit_execution',admit)
    store=LocalArtifactStore(tmp_path/'artifacts')
    payload=execute_remote(runner_config=identity,store_config=store.identity(),spec_payload=spec.to_dict())
    manifest=ArtifactManifest.from_dict(payload)
    assert len(calls)==1
    assert manifest.metadata['admission_v1']==worker_record.to_dict()
    assert manifest.workflow_context['admission_v1']==worker_record.to_dict()
    assert manifest.metadata['parent_provenance_v1']['scientific_identity']==observed.scientific_identity
    assert manifest.provenance().scientific_identity==observed.scientific_identity
    assert validate_completion(manifest,store=store)
