from dataclasses import replace
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import pytest

from pipelines_v2.operations.attribution import MAttrSpec,official_tool_lock,validate_result
from pipelines_v2.operations.attribution.contracts import EVALUATOR,GRID,expected_nodes,normalize_official_outputs,digest
from pipelines_v2.runtime.external_tool import ExternalToolRuntimeSpec,run_bounded
from pipelines_v2.runtime.locks import core_runtime_lock


def test_tool_environment_blocks_python_shadowing_and_preserves_gpu_settings(tmp_path):
    import subprocess
    from pipelines_v2.runtime.external_tool import external_tool_environment

    poison = tmp_path/'poison'
    poison.mkdir()
    (poison/'fractions.py').write_text("MARKER = 'poisoned dependency'\n")
    inherited = {**os.environ, 'PYTHONPATH':str(poison), 'CUDA_VISIBLE_DEVICES':'2,3'}
    # Establish that this payload really shadows an import without the boundary.
    unsafe = subprocess.run([sys.executable, '-c', 'import fractions; print(fractions.MARKER)'],
                            env=inherited, cwd=poison, capture_output=True, text=True, check=True, timeout=10)
    assert unsafe.stdout.strip() == 'poisoned dependency'
    inherited.update(PYTHONHOME=str(poison), PYTHONUSERBASE=str(poison),
                     PYTHONSTARTUP=str(poison/'startup.py'), PYTHONINSPECT='1',
                     PYTHONBREAKPOINT='poison.hook', PYTHONNOUSERSITE='0', PYTHONSAFEPATH='0')
    env = external_tool_environment(inherited)
    code = "import fractions,json,os,site; print(json.dumps([str(fractions.Fraction(1,2)),os.environ['CUDA_VISIBLE_DEVICES'],site.ENABLE_USER_SITE]))"
    safe = subprocess.run([sys.executable, '-c', code], env=env, cwd=poison,
                          capture_output=True, text=True, check=True, timeout=10)
    assert json.loads(safe.stdout) == ['1/2', '2,3', False]
    assert env['PATH'] == inherited['PATH']
    assert not {'PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','PYTHONSTARTUP','PYTHONINSPECT','PYTHONBREAKPOINT'} & env.keys()


def spec_fixture():
    import pipelines_v2.operations.attribution.specs as module
    assets=json.loads(Path(module.__file__).with_name('gpt2_ioi_assets.json').read_text())
    runtime=ExternalToolRuntimeSpec(lock=core_runtime_lock(),local_python_sources=('pipelines_v2',),tool_lock=official_tool_lock())
    request={'schema_version':1,'task':'ioi','model':{'name':'gpt2',**assets['model']},'data':{'repository':'mib-bench/ioi',**assets['data']},
        'evaluator':EVALUATOR,'tool_lock':runtime.tool_identity,'basis':'attention_head_mlp_input',
        'orientation':'iso_base_clean_source_s2_io_flip','include_input':True,'seed':42,'dtype':'float32',
        'fitting':{'steps':2,'lr':.05,'optimizer':'adam','adam_eps':1e-8,'T':.5,'n_iters':30,'masking':'topk','k_schedule':'uniform','train_batch_size':1,'k_avg':1,'cf_cache_gb':0},
        'evaluation':{'split':'validation','train_split':'train','examples':2,'batch_size':1,'grid':'official_ten_point'}}
    return MAttrSpec(request=request,runtime=runtime,model_path='/model',dataset_path='/data')


def official_fixture():
    importance={'cfg':{'n_layers':12,'n_heads':12,'d_model':768},'nodes':{n:{'score':float(i)} for i,n in enumerate(expected_nodes())}}
    importance['nodes']['logits']={'in_graph':True}
    # CPR=fraction: integral on [0.001,1] is (1-.001**2)/2, exactly hand-computable.
    metrics={'faithfulnesses':list(GRID),'accuracies':[.5]*10,'weighted_edge_counts':[float(i) for i in range(10)],
             'area_under':(1-.001**2)/2,'area_from_1':.999-(1-.001**2)/2,'average':sum(GRID)/10,'acc_auc':.5}
    return importance,metrics


def result_fixture():
    imp,metrics=official_fixture()
    return normalize_official_outputs(imp,metrics,request=spec_fixture().request,compute={},raw_outputs={})


def test_roundtrip_pinned_request_result_and_hand_computed_metrics():
    from pipelines_v2.operations.registry import operation_spec_from_dict
    from pipelines_v2.engine.base import PythonRuntimeSpec
    spec=spec_fixture();decoded=operation_spec_from_dict(json.loads(json.dumps(spec.to_dict())))
    assert decoded==spec
    assert isinstance(PythonRuntimeSpec.from_dict(spec.runtime.to_dict()),ExternalToolRuntimeSpec)
    assert replace(spec,model_path='/relocated',dataset_path='/other').semantic_hash()==spec.semantic_hash()
    result=validate_result(json.loads(json.dumps(result_fixture())))
    assert len(result['nodes'])==157
    assert result['aggregates']['area_under']==pytest.approx(.4999995)
    assert result['aggregates']['acc_auc']==.5
    assert result['ranking'][0]=='m11'


def test_qwen_uses_its_own_pins_node_basis_dtype_and_output_names(tmp_path):
    from pipelines_v2.operations.attribution.contracts import model_profile
    from pipelines_v2.operations.registry import operation_spec_from_dict
    spec=MAttrSpec.qwen25_ioi(model_path='/model',dataset_path='/data',runtime=spec_fixture().runtime)
    assert operation_spec_from_dict(spec.to_dict()) == spec
    assert spec.request['dtype']=='bfloat16'
    assert model_profile(spec.request)['path']=='Qwen/Qwen2.5-0.5B'
    assert spec.request['model']['revision']=='060db6499f32faf8b98477b0a26969ef7d8b9987'
    _,metrics=official_fixture()
    nodes=expected_nodes(24,14)
    imp={'cfg':{'n_layers':24,'n_heads':14,'d_model':896},'nodes':{n:{'score':float(i)} for i,n in enumerate(nodes)}}
    imp['nodes']['logits']={}
    result=normalize_official_outputs(imp,metrics,request=spec.request,compute={},raw_outputs={})
    assert len(result['nodes'])==361 and result['ranking'][0]=='m23'
    bad=copy.deepcopy(result);bad['request']['dtype']='float32'
    with pytest.raises(ValueError,match='numerical'):validate_result(bad)
    imp['cfg']['n_heads']=12
    with pytest.raises(ValueError,match='dimensions'):
        normalize_official_outputs(imp,metrics,request=spec.request,compute={},raw_outputs={})


@pytest.mark.parametrize('mutation',[
    lambda v:v['nodes'].append(v['nodes'][0]),
    lambda v:v['nodes'][0].update(score=float('nan')),
    lambda v:v['points'][2].update(requested_k=999),
    lambda v:v['points'][0].update(accuracy=1.1),
    lambda v:v['points'][0].update(weighted_edge_count=-1),
    lambda v:v['request']['model'].update(revision='a'*40),
    lambda v:v.update(evaluator='not-official'),
    lambda v:v.update(reproduction_label='paper_reproduction'),
])
def test_reject_malformed_or_relabelled_results(mutation):
    value=result_fixture();mutation(value)
    with pytest.raises(ValueError):validate_result(value)


def test_undefined_metric_retains_reason_and_ordinary_scores_reject_nan():
    imp,metrics=official_fixture();metrics['faithfulnesses'][0]={'value':None,'undefined_reason':'zero reference denominator'}
    value=normalize_official_outputs(imp,metrics,request=spec_fixture().request,compute={},raw_outputs={})
    assert value['points'][0]['cpr']['value'] is None
    imp['nodes']['input']['score']=float('nan')
    with pytest.raises(ValueError):normalize_official_outputs(imp,metrics,request=spec_fixture().request,compute={},raw_outputs={})


def test_importer_checks_producer_seed_model_binding_and_exact_bytes(tmp_path):
    from pipelines_v2.operations.attribution.execution import import_official
    spec=spec_fixture();imp,metrics=official_fixture();r=spec.request
    args={'model':'gpt2','task':'ioi','steps':2,'seed':42,'mode':'iso','masking':'topk','k_schedule':'uniform','lr':.05,'include_input':True,'split':'validation','train_split':'train','eval_examples':2,'ablation':'patching','optimizer':'adam','adam_eps':1e-8,'T':.5,'n_iters':30,'train_batch_size':1,'batch_size':1,'k_avg':1,'cf_cache_gb':0}
    (tmp_path/'ioi_gpt2_importances.json').write_text(json.dumps(imp))
    (tmp_path/'official_metrics.json').write_text(json.dumps({'args':args,'metrics':metrics,'installed_packages':{},'train_time_s':0,'train_time_seconds':0}))
    for name in ('ioi_gpt2_scores.pt','ioi_gpt2_validation.pkl','ioi_gpt2_trainlog.csv'):(tmp_path/name).write_text('opaque fixture')
    def bind():
        binding={'scientific_request':r,'observed_assets':{role:r[role]['files'] for role in ('model','data')},'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir() if p.name!='producer_binding.json'}}
        (tmp_path/'producer_binding.json').write_text(json.dumps(binding))
    bind();result=import_official(spec,tmp_path,compute={});assert result['scientific_verdict']=='not_tested'
    request=copy.deepcopy(r);request['seed']=43
    with pytest.raises(ValueError,match='Producer'):import_official(replace(spec,request=request),tmp_path,compute={})
    (tmp_path/'ioi_gpt2_trainlog.csv').write_text('changed')
    with pytest.raises(ValueError,match='bytes changed'):import_official(spec,tmp_path,compute={})
    bind();payload=json.loads((tmp_path/'official_metrics.json').read_text());payload['args']['seed']=41
    (tmp_path/'official_metrics.json').write_text(json.dumps(payload));bind()
    with pytest.raises(ValueError,match='arguments'):import_official(spec,tmp_path,compute={})


def invoke(tmp_path,code,**limits):
    return run_bounded([sys.executable,'-c',code],cwd=tmp_path,env=os.environ.copy(),output_dir=tmp_path/'out',
        timeout_seconds=limits.get('timeout',3),memory_bytes=1024**3,output_bytes=limits.get('output',1024**2))


def test_bounded_process_preserves_argv_and_failure(tmp_path):
    result=invoke(tmp_path,"print('literal $() ; not a shell')")
    assert result['exit_code']==0
    assert (tmp_path/'out/stdout.log').read_text().strip()=='literal $() ; not a shell'
    with pytest.raises(RuntimeError,match='exited 7'):invoke(tmp_path,'raise SystemExit(7)')


def test_timeout_and_output_limits(tmp_path):
    with pytest.raises(RuntimeError,match='wall deadline'):invoke(tmp_path,'import time;time.sleep(10)',timeout=.1)
    with pytest.raises(RuntimeError,match='output disk'):invoke(tmp_path,"print('x'*100000)",output=1000)


def test_local_and_remote_artifact_dispatch_no_manifest_on_failure(tmp_path,monkeypatch):
    from pipelines_v2.runtime.local import LocalRunner
    from pipelines_v2.runtime.remote_executor import execute_remote
    from pipelines_v2.storage.local import LocalArtifactStore
    from pipelines_v2.operations.execution.common import OperationExecutionResult
    import pipelines_v2.operations.attribution.execution as execution
    spec=spec_fixture();store=LocalArtifactStore(tmp_path/'store');runner=LocalRunner(artifacts=store)
    monkeypatch.setattr(execution,'run_mattr',lambda spec:OperationExecutionResult(payload=result_fixture()))
    artifact=runner.run(spec);assert artifact.result()['kind']=='external_attribution'
    payload=execute_remote(runner_config={'kind':'test'},store_config=store.identity(),spec_payload=spec.to_dict())
    assert payload['artifact_kind']=='external_attribution'
    before=set((tmp_path/'store').rglob('manifest.json'))
    def fail(spec):raise RuntimeError('upstream failed')
    monkeypatch.setattr(execution,'run_mattr',fail)
    with pytest.raises(RuntimeError,match='upstream failed'):runner.run(spec)
    assert set((tmp_path/'store').rglob('manifest.json'))==before


def test_modal_external_runtime_does_not_batch_with_native():
    from pipelines_v2.runtime.modal import ModalRunner,ModalResources
    from pipelines_v2.storage.modal import ModalVolumeStore
    runner=ModalRunner(resources=ModalResources(enable_workflow_batching=True),artifacts=ModalVolumeStore(name='fixture',root='/artifacts'))
    assert runner.workflow_batch_key(spec_fixture()) is None


def test_tool_readiness_budget_roundtrip_and_static_host_headroom():
    from pipelines_v2.runtime.external_tool import tool_readiness_profile
    from pipelines_v2.runtime.admission import static_admission
    spec=spec_fixture();profile=tool_readiness_profile(spec.runtime)
    assert profile['memory_bytes']==8*1024**3 and profile['container_memory_mb']==10240
    assert 'readiness_memory_bytes' not in spec.runtime.to_dict()
    runtime=replace(spec.runtime,readiness_memory_bytes=10*1024**3,readiness_timeout_seconds=180)
    assert ExternalToolRuntimeSpec.from_dict(runtime.to_dict())==runtime
    for value in (0,-1,True):
        with pytest.raises(ValueError):replace(runtime,readiness_memory_bytes=value)
    checks=static_admission(spec,{'kind':'modal','resources':{'memory_mb':4096,'readiness_timeout_seconds':120}})
    assert {c.check_id for c in checks if c.status=='failed'} >= {'runtime.tool_readiness_memory','runtime.tool_readiness_timeout'}
    checks=static_admission(spec,{'kind':'modal','resources':{'memory_mb':16384,'readiness_timeout_seconds':300}})
    assert all(c.status=='passed' for c in checks if c.check_id.startswith('runtime.tool_readiness_'))


def test_tool_readiness_retains_sanitized_actionable_failure_and_cleans_success(tmp_path,monkeypatch):
    import pipelines_v2.runtime.external_tool as module
    root=tmp_path/'diagnostics';root.mkdir()
    monkeypatch.setattr(module,'verify_tool_source',lambda runtime:None)
    monkeypatch.setattr(module.tempfile,'mkdtemp',lambda **kwargs:str(root))
    monkeypatch.setenv('HF_TOKEN','dummy-secret-never-serialize')
    def fail(*args,**kwargs):
        assert kwargs['memory_bytes']==8*1024**3 and kwargs['timeout_seconds']==120
        assert kwargs['env']['OMP_NUM_THREADS']=='2'
        logs=Path(kwargs['output_dir']);logs.mkdir(exist_ok=True)
        (logs/'stderr.log').write_text('ImportError: missing module; token=dummy-secret-never-serialize\n')
        raise RuntimeError('External tool exited 7')
    monkeypatch.setattr(module,'run_bounded',fail)
    with pytest.raises(module.ToolReadinessError,match='exited 7') as exc:
        module.check_tool_runtime(spec_fixture().runtime)
    assert 'dummy-secret-never-serialize' not in str(exc.value)
    assert 'ImportError: missing module' in str(exc.value)
    assert (root/'logs/stderr.log').exists()
    monkeypatch.setattr(module,'run_bounded',lambda *a,**kw:{'exit_code':0})
    assert module.check_tool_runtime(spec_fixture().runtime)=={'exit_code':0}
    assert not root.exists()
