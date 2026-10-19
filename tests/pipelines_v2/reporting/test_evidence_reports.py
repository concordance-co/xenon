from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from pipelines_v2.api import Dataset, Example, LabelMapSpec, ReportSpec
from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.experiments import EvidenceRecord, ExperimentPack
from pipelines_v2.reporting.evidence import evidence_report_payload, render_evidence_markdown, MAX_RESULT_BYTES
from pipelines_v2.runtime.local import LocalRunner
from pipelines_v2.storage import FileCatalog, LocalArtifactStore
from pipelines_v2.storage.artifacts import OperationArtifact
from pipelines_v2.workflow import WorkflowSpec, WorkflowStep, WorkflowOrchestrator

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def observed(tmp_path):
    runner = LocalRunner(artifacts=LocalArtifactStore(tmp_path / 'artifacts'), catalog=FileCatalog(tmp_path / 'catalog'))
    import importlib.util
    module_spec = importlib.util.spec_from_file_location('observed_toy', ROOT / 'examples/toy_pack/workflow.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    workflow = module.unchecked_workflow()
    workflow = replace(workflow, steps=tuple(replace(step, spec=replace(step.spec, output_dir=str(tmp_path / 'reports'))) if step.name == 'report' else step for step in workflow.steps))
    run = WorkflowOrchestrator(runners={'fixture_local':runner, 'report_local':runner}).run(workflow)
    artifact = run.step_results['labels']
    pack = ExperimentPack.load(ROOT / 'examples/toy_pack/pack.json')
    link = {'artifact_id':artifact.id, 'run_id':artifact.manifest().workflow_context['run_id'],
            'scientific_identity':artifact.manifest().provenance().scientific_identity,
            'selector':{'domain':'result', 'pointer':'/mapped_count', 'expected':2}}
    record = dict(schema_version=1,evidence_id='negative-count-fixture',pack_identity=pack.identity,preset='smoke',question_id='Q-count',hypothesis_id=None,
                  engineering_status='tested',scientific_verdict='refuted',metric='mapped_count',evaluator_digest=pack.to_dict()['metrics']['mapped_count']['evaluator_digest'],
                  split='fixture',split_link=None,control_arms={},effect={'value':2,'link':0},uncertainty={'value':None,'link':None,'reason':'Exact fixture count; no population inference'},
                  evidence_level='engineering',limitations=['Only tests the false fixture assertion that three rows mapped; no model claim.'],links=[link])
    return runner, artifact, pack, record


def test_negative_result_is_completed_engineering_report(observed, tmp_path):
    runner, artifact, pack, record = observed
    validated = EvidenceRecord.from_dict(record, pack=pack)
    assert EvidenceRecord.from_dict(validated.to_dict(), pack=pack).identity == validated.identity
    spec = ReportSpec(inputs=(artifact,), evidence={'pack':pack.to_dict(),'records':[record]}, output_dir=str(tmp_path / 'reports'))
    report = runner.run(spec)
    assert report.result()['evidence']['records'][0]['scientific_verdict'] == 'refuted'
    markdown = Path(report.manifest().storage_refs['report']['path']).read_text()
    assert '**tested**' in markdown and '**refuted**' in markdown
    assert artifact.id in markdown and '/mapped_count' in markdown
    assert not (Path(report.manifest().storage_refs['report']['path']).parent / 'results').exists()
    assert ReportSpec.from_dict(spec.to_dict()).evidence == spec.evidence
    assert 'evidence' not in ReportSpec().to_dict()


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(scientific_verdict='supported'),
    lambda r: r.update(evidence_level='causal'),
    lambda r: r.update(evaluator_digest='0'*64),
    lambda r: r.update(pack_identity='0'*64),
    lambda r: r.update(effect={'value':3,'link':0}),
    lambda r: r.update(links=[]),
    lambda r: r.update(uncertainty={'value':float('nan'),'link':None,'reason':'bad'}),
    lambda r: r.update(limitations=['x'*200000]),
    lambda r: r['links'][0].update(artifact_id='latest'),
])
def test_invalid_record_rejected(observed, mutation):
    _, _, pack, record = observed
    mutation(record)
    with pytest.raises(SpecValidationError):
        EvidenceRecord.from_dict(record, pack=pack)


@pytest.mark.parametrize('field,value', [('artifact_id','missing'), ('run_id','wrong'), ('scientific_identity','0'*64)])
def test_missing_or_wrong_links_rejected(observed, field, value):
    _, artifact, pack, record = observed
    record['links'][0][field] = value
    with pytest.raises(SpecValidationError):
        evidence_report_payload({'pack':pack.to_dict(),'records':[record]}, (artifact,))


def test_selector_and_payload_caps_before_read(observed, monkeypatch):
    _, artifact, pack, record = observed
    record['links'][0]['selector']['pointer'] = '/missing'
    with pytest.raises(ValueError, match='does not resolve'):
        evidence_report_payload({'pack':pack.to_dict(),'records':[record]}, (artifact,))
    record['links'][0]['selector']['pointer'] = '/mapped_count'
    bad_ref = {**artifact.manifest().storage_refs['result'], 'bytes': MAX_RESULT_BYTES+1}
    manifest = replace(artifact.manifest(), storage_refs={**artifact.manifest().storage_refs,'result':bad_ref})
    bounded = OperationArtifact(manifest, artifact.store)
    monkeypatch.setattr(OperationArtifact, 'result', lambda self: pytest.fail('must reject before downloading'))
    with pytest.raises(SpecValidationError, match='cap before reading'):
        evidence_report_payload({'pack':pack.to_dict(),'records':[record]}, (bounded,))


def test_no_capture_or_full_generation_download(observed, monkeypatch):
    _, artifact, pack, record = observed
    for name in ('localize', 'load_feature', 'load_label'):
        monkeypatch.setattr(OperationArtifact, name, lambda *args: pytest.fail('bulk payload access'))
    payload = evidence_report_payload({'pack':pack.to_dict(),'records':[record]}, (artifact,))
    assert payload['diagnostics']['linked_results'] == 1
    assert payload['inputs'][0]['scientific_identity'] == artifact.manifest().provenance().scientific_identity


def test_supported_claim_needs_observed_heldout_split(observed):
    _, artifact, pack, record = observed
    plan = pack.to_dict()
    plan['split_policy']['held_out'] = ['fixture']
    plan['controls'] = {'baseline': {'purpose':'comparison','required':True}}
    plan['presets']['reference']['enabled'] = True
    pack = ExperimentPack.from_dict(plan)
    record.update(pack_identity=pack.identity, preset='reference', scientific_verdict='supported',
                  evidence_level='behavioral', control_arms={'baseline':0})
    with pytest.raises(SpecValidationError, match='linked held-out'):
        EvidenceRecord.from_dict(record, pack=pack)


def test_workflow_mismatch_cannot_be_relabelled_as_pack_evidence(observed):
    _, artifact, pack, record = observed
    other = OperationArtifact(replace(artifact.manifest(), workflow_context={**artifact.manifest().workflow_context, 'workflow_hash':'0'*64}), artifact.store)
    with pytest.raises(SpecValidationError, match='different workflow'):
        evidence_report_payload({'pack':pack.to_dict(),'records':[record]}, (other,))


def test_scientific_refutation_needs_controls_and_heldout_evidence(observed):
    _, _, pack, record = observed
    record.update(preset='reference', scientific_verdict='refuted', evidence_level='causal')
    with pytest.raises(SpecValidationError, match='planned controls'):
        EvidenceRecord.from_dict(record, pack=pack)
    plan = pack.to_dict()
    plan['controls'] = {'baseline': {'purpose':'comparison','required':True}}
    pack = ExperimentPack.from_dict(plan)
    record.update(pack_identity=pack.identity, control_arms={'baseline':0})
    with pytest.raises(SpecValidationError, match='held-out'):
        EvidenceRecord.from_dict(record, pack=pack)


def read_evidence_fixture(observed, tmp_path, *, mutate_bundle=None, mutate_manifest=None):
    """Real toy producer, full C05 bundle, immutable manifests and C03 JSON bytes."""
    from pipelines_v2.evidence import EvidenceReader
    from pipelines_v2.storage.finalization import finalize_manifest
    from types import SimpleNamespace
    runner, artifact, pack, record = observed
    bundle = evidence_report_payload({'pack':pack.to_dict(), 'records':[record]}, (artifact,))
    if mutate_bundle:
        mutate_bundle(bundle)
    store = LocalArtifactStore(tmp_path/'reader-artifacts')
    result_ref = store.write_json('evidence_fixture', 'result.json', {'evidence':bundle})
    # The producer manifest supplies genuine resolved ToyEngine C01; the wrapper
    # is a stored fixture, not a claim that generic ReportSpec numerics resolve.
    wrapper = replace(artifact.manifest(), artifact_id='evidence_fixture', artifact_kind='report',
                      metadata={k:v for k,v in artifact.manifest().metadata.items() if k!='completion_v1'},
                      storage_refs={'result':result_ref})
    wrapper = finalize_manifest(store, wrapper)
    source = artifact.manifest()
    if mutate_manifest:
        source = mutate_manifest(source)
    catalog = SimpleNamespace(load_artifact=lambda key:{wrapper.artifact_id:wrapper, source.artifact_id:source}.get(key))
    reader = EvidenceReader(catalog)
    return reader, reader.artifact(wrapper.artifact_id, include_result=True)


def test_full_c05_bundle_validates_actual_selectors_and_compares(observed, tmp_path):
    from copy import deepcopy
    from pipelines_v2.evidence import compare_evidence
    reader, artifact = read_evidence_fixture(observed, tmp_path)
    assert artifact['evidence_verification']['status'] == 'verified', artifact['evidence_verification']
    assert artifact['result']['evidence']['pack'] == observed[2].to_dict()
    other = deepcopy(artifact)
    other['artifact_id'] = 'second_stored_fixture'
    comparison = compare_evidence([artifact, other], metric='mapped_count')
    assert comparison['comparable'], comparison
    assert comparison['rows'][0]['value'] == 2
    assert comparison['aggregation'] == 'none'
    # A mutated payload cannot reuse the earlier verification result.
    other['result']['evidence']['records'][0]['effect']['value'] = 9
    assert not compare_evidence([artifact, other], metric='mapped_count')['comparable']


@pytest.mark.parametrize('mutate', [
    lambda b:b.pop('pack'),
    lambda b:b['records'][0].pop('schema_version'),
    lambda b:b['records'][0].update(evaluator_digest='not-a-digest'),
    lambda b:b['records'][0].update(pack_identity='0'*64),
    lambda b:b['records'][0]['links'][0].update(run_id='other-run'),
    lambda b:b['records'][0]['links'][0].update(scientific_identity='0'*64),
    lambda b:b['records'][0]['links'][0]['selector'].update(pointer='/not-measured'),
    lambda b:(b['records'][0]['effect'].update(value=9), b['records'][0]['links'][0]['selector'].update(expected=9)),
])
def test_reader_keeps_forged_and_legacy_evidence_visible_unverified(observed, tmp_path, mutate):
    from pipelines_v2.evidence import compare_evidence
    _, artifact = read_evidence_fixture(observed, tmp_path, mutate_bundle=mutate)
    assert artifact['result']['evidence']
    assert artifact['evidence_verification']['status'] == 'unverified'
    assert not compare_evidence([artifact, artifact], metric='mapped_count')['comparable']


@pytest.mark.parametrize('mutate', [
    lambda m:replace(m, workflow_context={**m.workflow_context, 'workflow_hash':'0'*64}),
    lambda m:replace(m, workflow_context={**m.workflow_context, 'step_name':'unknown'}),
    lambda m:replace(m, artifact_kind='capture'),
    lambda m:replace(m, metadata={k:v for k,v in m.metadata.items() if k!='provenance_v1'}),
    lambda m:replace(m, storage_refs={'result':{**m.storage_refs['result'], 'bytes':256*1024+1}}),
])
def test_reader_rejects_wrong_linked_workflow_kind_identity_and_size(observed, tmp_path, mutate):
    _, artifact = read_evidence_fixture(observed, tmp_path, mutate_manifest=mutate)
    assert artifact['evidence_verification']['status'] == 'unverified'


def test_actual_reportspec_nesting_preserves_pack_and_verifies_links(observed, tmp_path):
    from pipelines_v2.evidence import EvidenceReader, compare_evidence
    runner, source, pack, record = observed
    report = runner.run(ReportSpec(inputs=(source,), evidence={'pack':pack.to_dict(),'records':[record]}, output_dir=str(tmp_path/'actual-report')))
    view = EvidenceReader(runner.catalog).artifact(report.id, include_result=True)
    assert view['result']['evidence']['pack'] == pack.to_dict()
    assert view['evidence_verification']['status'] == 'verified', view['evidence_verification']
    if not view['identity_complete']:
        assert not compare_evidence([view,view], metric='mapped_count')['comparable']


def published_fixture_manifest(manifest, *, hops=1):
    publication = {'execution_identity_preserved':True,
        'source_scientific_identity':manifest.provenance().scientific_identity,
        'source_workflow':manifest.workflow_context, 'source_artifact_kind':manifest.artifact_kind,
        'published_artifact_kind':manifest.artifact_kind}
    for _ in range(hops-1):
        publication = {**publication, 'source_workflow':{}, 'prior_publication':publication}
    return replace(manifest, workflow_context={}, metadata={**manifest.metadata, 'publication_v1':publication})


@pytest.mark.parametrize('hops', [1, 2, 16])
def test_reader_verifies_original_workflow_after_publication(observed, tmp_path, hops):
    _, view = read_evidence_fixture(observed, tmp_path,
        mutate_manifest=lambda m:published_fixture_manifest(m, hops=hops))
    assert view['evidence_verification']['status'] == 'verified', view['evidence_verification']
    assert view['evidence_verification']['linked_result_bytes'] > 0


@pytest.mark.parametrize('mutation', ['identity','flag','checksum','kind','depth'])
def test_reader_rejects_unproven_publication_lineage(observed, tmp_path, mutation):
    def change(manifest):
        m = published_fixture_manifest(manifest, hops=17 if mutation == 'depth' else 1)
        publication = m.metadata['publication_v1']
        if mutation == 'identity': publication['source_scientific_identity'] = '0'*64
        if mutation == 'flag': publication['execution_identity_preserved'] = False
        if mutation == 'kind': publication['published_artifact_kind'] = 'report_summary'
        if mutation == 'checksum':
            from copy import deepcopy
            metadata = deepcopy(m.metadata)
            for payload in metadata['completion_v1']['payloads']:
                if payload['path'] == m.storage_refs['result']['path']:
                    payload['sha256'] = '0'*64
            m = replace(m, metadata=metadata)
        return m
    _, view = read_evidence_fixture(observed, tmp_path, mutate_manifest=change)
    assert view['evidence_verification']['status'] == 'unverified'


def test_published_partial_kind_stays_explicit_with_original_step_lineage(observed, tmp_path):
    def change(manifest):
        m = published_fixture_manifest(manifest)
        m.metadata['publication_v1'].update(published_artifact_kind='report_summary',
            view='result_summary', omitted_fields=['rendered presentation'])
        return replace(m, artifact_kind='report_summary')
    _, view = read_evidence_fixture(observed, tmp_path, mutate_manifest=change)
    assert view['evidence_verification']['status'] == 'verified', view['evidence_verification']
