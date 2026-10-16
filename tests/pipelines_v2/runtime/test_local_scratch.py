from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from pipelines_v2.api import CaptureSpec, Dataset, Example, GenerationSpec, ResidualSite, ToyEngine
from pipelines_v2.runtime.scratch import LocalScratchProfile, require_prepared_model_cache, validate_scratch_workflow
from pipelines_v2.runtime.scratch_store import BudgetedLocalArtifactStore
from pipelines_v2.runtime.scratch_supervision import ScratchLimits, ScratchLimitError, ScratchPreparationError, supervise, tree_bytes
from pipelines_v2.workflow import WorkflowSpec, WorkflowStep


def workflow(*, count=2, tokens=8, generated=0):
    dataset=Dataset.from_examples(tuple(Example(key=f'key-{i}',prompt=f'prompt {i}') for i in range(count)))
    return WorkflowSpec(name='offline-scratch-fixture',steps=(WorkflowStep('capture','scratch',CaptureSpec(
        dataset=dataset,engine=ToyEngine(sequence_length=tokens),sites=(ResidualSite(name='resid',site='resid_post',layers=(0,)),),
        generation=GenerationSpec(enabled=generated>0,max_tokens=generated))),))


@pytest.mark.parametrize('changed', [{'max_examples':0},{'max_input_tokens':True},{'timeout_seconds':float('inf')},{'max_artifact_bytes':-1}])
def test_invalid_limits(changed):
    with pytest.raises(ValueError):
        ScratchLimits(**changed)


def test_profile_serializes_explicit_increases_and_defaults(monkeypatch,tmp_path):
    monkeypatch.setenv('XENON_HOME',str(tmp_path))
    profile=LocalScratchProfile(limits=ScratchLimits(max_examples=128))
    assert profile.to_dict()['limits']['max_examples']==128
    assert LocalScratchProfile.from_dict(profile.to_dict()).to_dict()==profile.to_dict()
    assert profile.to_dict()['artifact_root']==str(tmp_path/'pipelines_v2/artifacts')
    assert profile.to_dict()['catalog_root']==str(tmp_path/'pipelines_v2/catalog')
    with pytest.raises(ValueError,match='device-memory'):
        LocalScratchProfile(device='mps')


@pytest.mark.parametrize('kwargs,match',[({'count':65},'example-count'),({'tokens':257},'input-token'),({'generated':33},'generated-token')])
def test_input_limits_before_execution(kwargs,match):
    with pytest.raises(ScratchLimitError,match=match):
        validate_scratch_workflow(workflow(**kwargs),ScratchLimits())


def test_missing_model_cache_has_explicit_preparation(tmp_path):
    with pytest.raises(ScratchPreparationError,match='hf download'):
        require_prepared_model_cache(tmp_path/'absent')
    (tmp_path/'model').mkdir()
    with pytest.raises(ScratchPreparationError,match='never downloads'):
        require_prepared_model_cache(tmp_path/'model')


def test_rejects_cloud_runner_without_constructing_it():
    class Cloud:
        kind='modal'
        def to_runner(self):
            pytest.fail('Cloud client must never initialize')
    with pytest.raises(ValueError,match='cloud runner'):
        validate_scratch_workflow(workflow(),ScratchLimits(),runner_specs={'scratch':Cloud()})


def test_budgeted_json_write_removes_partial_temp_and_has_no_final(tmp_path):
    store=BudgetedLocalArtifactStore(tmp_path,max_new_bytes=64)
    store.make_artifact_dir('artifact')
    with pytest.raises(ScratchLimitError,match='while writing'):
        store.write_json('artifact','result.json',{'oversized':'x'*1024})
    assert tree_bytes(tmp_path)==0
    assert not (tmp_path/'artifact/manifest.json').exists()


def test_tensor_payload_guard_before_serialization(tmp_path):
    store=BudgetedLocalArtifactStore(tmp_path,max_new_bytes=64)
    store.make_artifact_dir('artifact')
    with pytest.raises(ScratchLimitError,match='before serialization'):
        store.write_safetensors('artifact','feature.safetensors',{'a':np.zeros(128,dtype=np.float32)})
    assert tree_bytes(tmp_path)==0


def run_code(tmp_path, code, **limits):
    attempt=tmp_path/'attempt'; attempt.mkdir()
    output=attempt/'artifacts';output.mkdir()
    result=supervise([sys.executable,'-c',code],attempt=attempt,output_root=output,
                     limits=ScratchLimits(timeout_seconds=10,**limits))
    return result,attempt


def test_supervisor_runs_inherited_python_network_sentinel(tmp_path):
    result,attempt=run_code(tmp_path,"import subprocess,sys; p=subprocess.run([sys.executable,'-c',\"import socket; socket.create_connection(('127.0.0.1', 9))\"],capture_output=True,text=True); assert p.returncode != 0; assert 'local-scratch forbids network' in p.stderr")
    assert result['status']=='passed'
    assert result['observed_peak_tree_rss_bytes']>0


def test_os_sandbox_blocks_native_child_network(tmp_path):
    # Native curl bypasses the Python audit sentinel; the OS boundary must still block it.
    import shutil
    curl=shutil.which('curl')
    if curl is None:
        pytest.skip('native curl sentinel unavailable')
    server=socket.socket();server.bind(('127.0.0.1',0));server.listen();server.settimeout(1)
    port=server.getsockname()[1]
    code=f"import subprocess; p=subprocess.run([{curl!r},'--noproxy','*','--connect-timeout','1','http://127.0.0.1:{port}'],capture_output=True); assert p.returncode != 0"
    try:
        result,_=run_code(tmp_path,code)
        with pytest.raises(socket.timeout):
            server.accept()
    finally:
        server.close()
    assert result['status']=='passed'


def test_wall_time_kills_child_tree(tmp_path):
    attempt=tmp_path/'attempt';attempt.mkdir();output=attempt/'artifacts';output.mkdir()
    pid_path=attempt/'child.pid'
    code=f"import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); pathlib.Path({str(pid_path)!r}).write_text(str(p.pid)); time.sleep(20)"
    with pytest.raises(ScratchLimitError,match='wall-time'):
        supervise([sys.executable,'-c',code],attempt=attempt,output_root=output,limits=ScratchLimits(timeout_seconds=.6))
    import psutil
    assert not psutil.pid_exists(int(pid_path.read_text()))
    assert json.loads((attempt/'diagnostics.json').read_text())['status']=='failed'


def test_rss_limit_counts_child_allocation(tmp_path):
    attempt=tmp_path/'attempt';attempt.mkdir();output=attempt/'artifacts';output.mkdir()
    pid_path=attempt/'child.pid'
    code=f"import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; a=bytearray(128*1024*1024); time.sleep(20)']); pathlib.Path({str(pid_path)!r}).write_text(str(p.pid)); time.sleep(20)"
    with pytest.raises(ScratchLimitError,match='process-tree RSS'):
        supervise([sys.executable,'-c',code],attempt=attempt,output_root=output,limits=ScratchLimits(timeout_seconds=10,max_process_tree_rss_bytes=80*1024*1024))
    import psutil
    assert not psutil.pid_exists(int(pid_path.read_text()))
    assert json.loads((attempt/'diagnostics.json').read_text())['observed_peak_tree_rss_bytes']>80*1024*1024


def test_output_polling_catches_non_store_writer(tmp_path):
    attempt=tmp_path/'attempt';attempt.mkdir();output=attempt/'artifacts';output.mkdir()
    code=f"import pathlib,time; pathlib.Path({str(output/'payload')!r}).write_bytes(b'x'*4096); time.sleep(2)"
    with pytest.raises(ScratchLimitError,match='artifact-byte'):
        supervise([sys.executable,'-c',code],attempt=attempt,output_root=output,limits=ScratchLimits(max_artifact_bytes=1024))
    assert not list(output.rglob('manifest.json'))


def test_rejects_remote_dataset_without_fetch(monkeypatch):
    deferred=Dataset(source={'kind':'postgres','env_var':'DO_NOT_READ'},fetch={'table':'fixture'})
    base=workflow()
    remote=replace(base,steps=(replace(base.steps[0],spec=replace(base.steps[0].spec,dataset=deferred)),))
    with pytest.raises(ValueError,match='remote/deferred'):
        validate_scratch_workflow(remote,ScratchLimits())


def test_python_cloud_import_sentinels_are_inherited(tmp_path):
    code="import subprocess,sys; p=subprocess.run([sys.executable,'-c','import modal'],capture_output=True,text=True); assert p.returncode != 0; assert 'forbids cloud' in p.stderr"
    result,_=run_code(tmp_path,code)
    assert result['status']=='passed'


def test_os_sandbox_forbids_writes_outside_attempt(tmp_path):
    outside=tmp_path/'outside'
    code=f"from pathlib import Path; Path({str(outside)!r}).write_text('forbidden')"
    with pytest.raises(RuntimeError,match='worker failed'):
        run_code(tmp_path,code)
    assert not outside.exists()


def test_example_budget_is_shared_across_workflow_steps():
    first=workflow(count=40)
    second=workflow(count=40)
    dataset=Dataset.from_examples(tuple(replace(example,key='other-'+example.key) for example in second.steps[0].spec.dataset.examples))
    both=WorkflowSpec(steps=(first.steps[0],replace(second.steps[0],name='other',spec=replace(second.steps[0].spec,dataset=dataset))))
    with pytest.raises(ScratchLimitError,match='example-count'):
        validate_scratch_workflow(both,ScratchLimits())


def test_supervised_workflow_catalog_and_durable_identity(tmp_path, monkeypatch):
    from pipelines_v2.storage.local import FileCatalog
    from pipelines_v2.storage.finalization import validate_completion
    monkeypatch.setenv('XENON_HOME', str(tmp_path))
    # Shared defaults exist in the real checkout; a scratch run bypasses them.
    monkeypatch.setenv('XENON_CATALOG_POSTGRES_DSN', 'postgresql://must-never-connect.invalid/db')
    profile = LocalScratchProfile()
    result = profile.run(workflow())
    catalog = FileCatalog(profile.to_dict()['catalog_root'])
    assert catalog.load_workflow_run(result.run_id).status == 'completed'
    manifest = result.step_results['capture'].manifest()
    assert validate_completion(manifest, store=result.step_results['capture'].store)
    assert manifest.provenance().scientific_identity
    attempt = Path(profile.to_dict()['artifact_root']).parent / 'scratch' / result.run_id
    original = json.loads((attempt / 'result.json').read_text())['capture']
    assert original['metadata']['provenance_v1'] == manifest.metadata['provenance_v1']
    assert [item['sha256'] for item in original['metadata']['completion_v1']['payloads']] == [item['sha256'] for item in manifest.metadata['completion_v1']['payloads']]
    assert profile.recover(result.run_id).step_results['capture'].id == manifest.artifact_id
    assert not (attempt / 'artifacts').exists()


def test_artifact_budget_failure_never_publishes_run(tmp_path, monkeypatch):
    from pipelines_v2.storage.local import FileCatalog
    monkeypatch.setenv('XENON_HOME', str(tmp_path))
    profile = LocalScratchProfile(limits=ScratchLimits(max_artifact_bytes=1024))
    with pytest.raises((RuntimeError, ScratchLimitError), match='byte'):
        profile.run(workflow(), run_id='wr_over_bytes')
    assert not (Path(profile.to_dict()['artifact_root']) / 'wr_over_bytes').exists()
    assert FileCatalog(profile.to_dict()['catalog_root']).load_workflow_run('wr_over_bytes').status == 'failed'
    with pytest.raises(ValueError, match='failed supervision'):
        profile.recover('wr_over_bytes')


def test_recover_after_catalog_failure_without_compute(tmp_path, monkeypatch):
    from pipelines_v2.storage.local import FileCatalog
    monkeypatch.setenv('XENON_HOME', str(tmp_path))
    profile = LocalScratchProfile()
    original = FileCatalog.record_artifact
    def fail_catalog(self, manifest):
        if str(self.root) == profile.to_dict()['catalog_root']:
            raise OSError('injected catalog publication failure')
        return original(self, manifest)
    monkeypatch.setattr(FileCatalog, 'record_artifact', fail_catalog)
    with pytest.raises(OSError, match='publication failure'):
        profile.run(workflow(), run_id='wr_catalog_failure')
    monkeypatch.setattr(FileCatalog, 'record_artifact', original)
    result = profile.recover('wr_catalog_failure')
    assert result.step_results['capture'].manifest().provenance()
    assert FileCatalog(profile.to_dict()['catalog_root']).load_workflow_run(result.run_id).status == 'completed'


def source_file(tmp_path, prefix=''):
    source = tmp_path / 'workflow.py'
    source.write_text(prefix + '''
from pipelines_v2.api import Dataset, Example, LabelMapSpec, ReportSpec, StepRef, WorkflowSpec, WorkflowStep

def build_dataset():
    return Dataset.from_examples([Example(key='a',prompt='alpha',labels={'class':'yes'}), Example(key='b',prompt='beta',labels={'class':'no'})])

def build_workflow(dataset):
    return WorkflowSpec(name='scratch-cli-fixture', steps=(
        WorkflowStep('labels','scratch',LabelMapSpec(source=dataset.labels('class'),mapping={'yes':1,'no':0},output_name='mapped')),
        WorkflowStep('report','scratch',ReportSpec(inputs=(StepRef('labels'),))),
    ))
''')
    return source


def test_public_cli_imports_inside_sandbox_and_builds_report(tmp_path):
    from pipelines_v2.storage.local import FileCatalog
    from pipelines_v2.storage.finalization import validate_completion
    from pipelines_v2.storage.artifacts import ArtifactManifest
    path = source_file(tmp_path)
    env = dict(os.environ, XENON_HOME=str(tmp_path / 'state'))
    command = [sys.executable, '-m', 'pipelines_v2.cli', 'workflow', 'run', '--profile', 'local-scratch', '--file', str(path)]
    process = subprocess.run(command, env=env, text=True, capture_output=True, timeout=45)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    root = Path(result['execution_profile']['artifact_root']) / result['run_id']
    report_id = result['steps']['report']['artifact_id']
    manifest = ArtifactManifest.from_dict(json.loads((root / report_id / 'manifest.json').read_text()))
    from pipelines_v2.storage.local import LocalArtifactStore
    assert validate_completion(manifest, store=LocalArtifactStore(root))
    assert FileCatalog(result['execution_profile']['catalog_root']).load_workflow_run(result['run_id']).status == 'completed'
    for path in (root / report_id).rglob('*.json'):
        if path.name != 'manifest.json':
            assert '/scratch/wr_' not in path.read_text()
    assert (root / report_id / 'report.md').is_file()


def test_workflow_import_network_is_blocked_before_builder(tmp_path):
    path = source_file(tmp_path, "import socket\nsocket.create_connection(('127.0.0.1', 9))\n")
    profile = LocalScratchProfile(artifact_root=tmp_path/'artifacts', catalog_root=tmp_path/'catalog')
    with pytest.raises(RuntimeError, match='forbids network'):
        profile.run_file(path, run_id='wr_import_network')
    assert not (tmp_path/'artifacts/wr_import_network').exists()
