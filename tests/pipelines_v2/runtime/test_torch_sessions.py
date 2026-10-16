from contextlib import contextmanager
from types import SimpleNamespace
import threading

import pytest

from pipelines_v2.runtime.sdxl_scope import WorkerState, scoped_workflow, SDXLWorkflowScope
from pipelines_v2.runtime.modal import ModalResources, ModalRunner
from pipelines_v2.storage.modal import ModalVolumeStore


def test_worker_retains_one_session_across_dependent_operations_and_closes():
    closed=[];state=WorkerState(idle_seconds=.02)
    with state.operation():state._diffusion_sessions['model']=SimpleNamespace(close=lambda:closed.append(True))
    with state.operation():assert 'model' in state._diffusion_sessions
    # Synchronize on timer, not an arbitrary long sleep.
    state._timer.join(1)
    assert closed==[True] and state._diffusion_sessions=={}
    state.close()


def test_failure_closes_loaded_state_before_another_operation():
    closed=[];state=WorkerState()
    with pytest.raises(RuntimeError,match='failure'):
        with state.operation():
            state._diffusion_sessions['model']=SimpleNamespace(close=lambda:closed.append(True))
            raise RuntimeError('failure')
    assert closed==[True] and state._diffusion_sessions=={}
    state.close()


def test_scope_wraps_existing_dag_once_and_closes_on_failure():
    events=[]
    class Runner:
        @contextmanager
        def workflow_scope(self):
            events.append('enter')
            try:yield
            finally:events.append('close')
    runner=Runner()
    class Orchestrator:
        runners={'one':runner,'same':runner}
        @scoped_workflow
        def run(self,workflow):
            events.extend(['baseline','dependent-hook'])
            raise ValueError('dag failure')
    with pytest.raises(ValueError):Orchestrator().run(None)
    assert events==['enter','baseline','dependent-hook','close']


def test_opt_in_serialization_does_not_change_legacy_resource_payload():
    assert 'enable_workflow_sessions' not in ModalResources().to_dict()
    enabled=ModalResources(enable_workflow_sessions=True)
    assert ModalResources.from_dict(enabled.to_dict())==enabled


def test_runner_observes_before_reuse_and_restores_prior_records(monkeypatch):
    events=[]
    class FakeScope:
        def __init__(self,runner):self.events=events
        def supports(self,spec):return True
        def observe(self,spec):events.append('fresh GPU identity');return {'fixture':True}
        def close(self):events.append('close')
    monkeypatch.setattr('pipelines_v2.runtime.sdxl_scope.SDXLWorkflowScope',FakeScope)
    monkeypatch.setattr(ModalRunner,'plan',lambda *args:SimpleNamespace(validate=lambda:None))
    runner=ModalRunner(ModalResources(enable_workflow_sessions=True),ModalVolumeStore('test','/test'))
    spec=SimpleNamespace(semantic_hash=lambda:'key')
    runner.admission_records['existing']={'preserved':True}
    with runner.workflow_scope():
        runner.prepare_execution_identity(spec)
        assert runner.admission_records['key']=={'fixture':True}
    assert runner.admission_records=={'existing':{'preserved':True}}
    assert events==['fresh GPU identity','close']


def test_failed_scope_rejects_remaining_images_without_transport():
    scope=SDXLWorkflowScope(SimpleNamespace())
    scope._failed=True
    with pytest.raises(RuntimeError,match='fanout is halted'):scope._call('execute',None)


def test_idle_cap_cannot_be_extended():
    with pytest.raises(ValueError):WorkerState(idle_seconds=301)


def test_scoped_transport_uses_one_app_cpu_first_fresh_gpu_observation_and_close(monkeypatch,tmp_path):
    import sys
    from pipelines_v2.api import Dataset, Example, GenerationRunSpec, GenerationSpec, ToyEngine
    from pipelines_v2.engine.diffusers.specs import DiffusionGenerationSpec
    from pipelines_v2.engine.diffusers.session import ImageRequest
    from pipelines_v2.runtime.modal import ModalVolumeMount
    from pipelines_v2.runtime.readiness import AdmissionRecord
    from pipelines_v2.runtime.admission import AdmissionCheck,workload_envelope
    from pipelines_v2.core.provenance import resolve_provenance
    import pipelines_v2.runtime.modal_worker as mw
    calls=[];definitions={};apps=[]
    class Image:
        def env(self,value):return self
        def add_local_dir(self,*args,**kwargs):return self
    class Function:
        def __init__(self,fn):self.fn=fn
        def spawn(self,*args):
            calls.append(self.fn.__name__)
            return SimpleNamespace(get=lambda timeout:self.fn(*args),cancel=lambda **kwargs:None)
        def remote_gen(self,*args):
            calls.append((self.fn.__name__,args[1]))
            yield from self.fn(*args)
    class App:
        def __init__(self,*args):apps.append(self)
        def function(self,**kwargs):
            def decorate(fn):definitions[fn.__name__]=kwargs;return Function(fn)
            return decorate
        def run(self):return self
        def __enter__(self):calls.append('app-enter');return self
        def __exit__(self,*unused):calls.append('app-close')
    modal=SimpleNamespace(App=App,Volume=SimpleNamespace(from_name=lambda *args,**kwargs:SimpleNamespace(commit=lambda:None)),
                          Secret=SimpleNamespace(from_name=lambda *args,**kwargs:None))
    monkeypatch.setitem(sys.modules,'modal',modal)
    monkeypatch.setattr('pipelines_v2.runtime.locks.build_locked_image',lambda *args:Image())
    monkeypatch.setattr(mw,'_resolved_local_python_sources',lambda *args,**kwargs:((),()))
    toy=GenerationRunSpec(engine=ToyEngine(),dataset=Dataset.from_examples([Example(key='a',prompt='a')]),generation=GenerationSpec(enabled=True,max_tokens=1))
    record=AdmissionRecord((AdmissionCheck('fixture','readiness','passed','fixture','fixture'),),resolve_provenance(toy),workload_envelope(toy,{}).to_dict(),scope='cpu_preflight')
    monkeypatch.setattr('pipelines_v2.runtime.readiness.admit_execution',lambda *args,**kwargs:record)
    loads=[0]
    def dispatch(scope_id,mode,config,store,spec,context,report):
        if mode=='observe':return {'admission':record.to_dict(),'worker_id':'one-worker','model_load_count':loads[0],'gpu_allocated':True}
        if mode=='close':return {'closed':True,'worker_id':'one-worker','model_load_count':loads[0]}
        loads[0]=1
        report({'step_name':context['step_name'],'stage':'diffusion_image_complete'})
        return {'artifact_id':context['step_name'],'metadata':{'session_v1':{'model_load_count':loads[0]}}}
    monkeypatch.setattr('pipelines_v2.runtime.sdxl_scope.dispatch_worker',dispatch)
    runner=ModalRunner(ModalResources(gpu='A100-80GB',memory_mb=65536,timeout_seconds=1200,enable_workflow_sessions=True,
        volumes=(ModalVolumeMount('models','/models'),)),ModalVolumeStore('artifacts','/data/results'))
    spec=DiffusionGenerationSpec(images=(ImageRequest('a','a'),))
    scope=SDXLWorkflowScope(runner);events=[]
    for name in ('baseline','dependent-edit'):
        scope.observe(spec)
        scope.execute(spec,{'step_name':name},events.append)
    scope.close()
    assert len(apps)==1 and loads==[1]
    assert calls==['app-enter','preflight',('stream','observe'),('stream','execute'),('stream','observe'),('stream','execute'),('stream','close'),'app-close']
    assert [event['step_name'] for event in events]==['baseline','dependent-edit']
    assert 'gpu' not in definitions['preflight']
    assert definitions['stream']['scaledown_window']==300 and definitions['stream']['max_containers']==1
