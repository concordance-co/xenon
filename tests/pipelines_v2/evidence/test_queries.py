import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pipelines_v2.evidence import EvidenceReader, ViewLimits


def fixture_reader(payload, *, kind='transform', limits=None, checksum=True):
    data = json.dumps(payload).encode()
    ref = {'store':'modal_volume', 'name':'must-not-open', 'path':'/data/result.json', 'format':'json', 'bytes':len(data)}
    completion = {'state':'complete', 'payloads':[{'path':ref['path'], 'sha256':hashlib.sha256(data).hexdigest()}]} if checksum else {}
    manifest = SimpleNamespace(provenance=lambda:None, metadata={'completion_v1':completion}, artifact_kind=kind, workflow_context={}, input_artifact_refs=(), storage_refs={'result':ref, 'features':{'path':'forbidden-activation-bank'}})
    calls = []
    def read(ref, cap):
        assert ref['path'] == '/data/result.json'
        calls.append((ref, cap))
        return data
    catalog = SimpleNamespace(load_artifact=lambda key:manifest)
    return EvidenceReader(catalog, limits=limits, read_json_bytes=read), calls, manifest


def test_metadata_never_reads_payload_and_summary_only_reads_declared_result():
    reader, calls, _ = fixture_reader({'summary':'ready'})
    assert not reader.artifact('a')['identity_complete']
    assert calls == []
    assert reader.artifact('a', include_result=True)['summary_checksum_verified']
    assert len(calls) == 1


@pytest.mark.parametrize('kind', ['capture', 'generation_run', 'patched_generation', 'diffusion_capture'])
def test_raw_model_artifacts_are_not_summaries(kind):
    reader, calls, _ = fixture_reader({}, kind=kind)
    with pytest.raises(ValueError, match='Raw captures'):
        reader.artifact('a', include_result=True)
    assert calls == []


def test_actual_and_cumulative_bytes_and_checksum():
    reader, calls, manifest = fixture_reader({'value':1}, limits=ViewLimits(result_bytes=20))
    reader.artifact('a', include_result=True)
    with pytest.raises(ValueError, match='remaining'):
        reader.artifact('b', include_result=True)
    assert len(calls) == 1
    reader, _, manifest = fixture_reader({})
    reader.read_json_bytes = lambda ref, cap:b'x'*100
    with pytest.raises(ValueError, match='actual byte budget'):
        reader.artifact('a', include_result=True)
    reader, _, manifest = fixture_reader({})
    manifest.metadata['completion_v1']['payloads'][0]['sha256'] = '0'*64
    with pytest.raises(ValueError, match='checksum'):
        reader.artifact('a', include_result=True)


@pytest.mark.parametrize('payload', [{'evidence':{'records':[{},{}]}}, {'graph':{'nodes':[{},{}]}}, {'cells':[{},{}]}, {'examples':[1,2]}, {'dossiers':[{'observations':{'top':[{'context_id':'a'}, {'context_id':'b'}]}}]}])
def test_panel_caps(payload):
    reader, _, _ = fixture_reader(payload, limits=ViewLimits(graph_nodes=1, images=1, examples=1))
    with pytest.raises(ValueError, match='cap'):
        reader.artifact('a', include_result=True)


def test_readonly_catalog_does_not_create_missing_paths(tmp_path):
    from pipelines_v2.evidence.catalog import local_catalog
    catalog = local_catalog(tmp_path/'absent')
    assert EvidenceReader(catalog).runs()['runs'] == []
    with pytest.raises(KeyError):
        EvidenceReader(catalog).run('missing')
    assert not (tmp_path/'absent').exists()


def test_cli_roundtrip_and_import_light(tmp_path):
    code = "import sys; from pipelines_v2.evidence import EvidenceReader; from pipelines_v2.evidence.cli import main; assert not any(k in sys.modules for k in ('torch','diffusers','transformers','modal')); raise SystemExit(main(sys.argv[1:]))"
    result = subprocess.run([sys.executable, '-c', code, '--catalog-root', str(tmp_path/'absent'), 'runs'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['runs'] == []
    assert not (tmp_path/'absent').exists()


def test_local_discovery_bounds_actual_deserializations_even_with_filters(tmp_path, monkeypatch):
    from pipelines_v2.evidence.catalog import local_catalog
    from pipelines_v2.workflow.records import WorkflowRunRecord
    root = tmp_path/'catalog'/'workflow_runs'
    root.mkdir(parents=True)
    for index in range(5):
        record = WorkflowRunRecord(str(index), 'bounded', 'hash', 'spec', {}, 'complete', str(index))
        (root/f'{index}.json').write_text(json.dumps(record.to_dict()))
    actual = WorkflowRunRecord.from_dict
    calls = []
    def counted(value):
        calls.append(value['run_id'])
        return actual(value)
    monkeypatch.setattr(WorkflowRunRecord, 'from_dict', counted)
    catalog = local_catalog(root.parent)
    assert catalog.list_workflow_runs(status='missing', limit=2) == []
    assert len(calls) == 2 and catalog.scan_files == 2 and catalog.scan_truncated
    assert catalog.scan_bytes <= 16*1024**2


def test_local_discovery_oversized_metadata_is_explicitly_truncated(tmp_path):
    from pipelines_v2.evidence.catalog import local_catalog
    root = tmp_path/'workflow_runs'
    root.mkdir()
    (root/'large.json').write_bytes(b' '*(1024**2+10))
    result = EvidenceReader(local_catalog(tmp_path)).runs()
    assert result['runs'] == [] and result['scan_truncated']
    assert result['scan_bytes'] == 1024**2+1 and result['scan_files'] == 1
