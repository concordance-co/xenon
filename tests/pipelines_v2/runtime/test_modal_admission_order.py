"""CPU readiness runs before GPU invocation; merge never inherits GPU resources."""
from types import SimpleNamespace
import sys
import pytest
from pipelines_v2.api import Dataset, Example, GenerationRunSpec, GenerationSpec, ToyEngine
from pipelines_v2.core.provenance import resolve_provenance
from pipelines_v2.runtime.admission import AdmissionCheck, workload_envelope
from pipelines_v2.runtime.readiness import AdmissionRecord
from pipelines_v2.runtime.modal_worker import run_on_modal


@pytest.mark.parametrize('status',['passed','failed'])
@pytest.mark.parametrize('external_tool',[False,True])
def test_cpu_readiness_precedes_gpu_and_failure_never_invokes_gpu(monkeypatch,tmp_path,status,external_tool):
    calls=[]
    resources={}
    class Image:
        def pip_install(self,*args):return self
        def env(self,*args):return self
        def add_local_dir(self,*args,**kwargs):return self
    class Function:
        def __init__(self,fn):self.fn=fn
        def spawn(self,*args):
            calls.append(self.fn.__name__)
            return SimpleNamespace(get=lambda timeout:self.fn(*args),cancel=lambda **kwargs:None)
        def remote_gen(self,*args):
            calls.append(self.fn.__name__)
            yield from self.fn(*args)
    class App:
        app_id='ap-fixture'
        def __init__(self,*args):pass
        def function(self,**kwargs):
            def decorate(fn):
                resources[fn.__name__]=kwargs
                return Function(fn)
            return decorate
        def run(self):return self
        def __enter__(self):return self
        def __exit__(self,*args):return False
    modal=SimpleNamespace(App=App,Image=SimpleNamespace(debian_slim=lambda **kwargs:Image()),
        Volume=SimpleNamespace(from_name=lambda *args,**kwargs:SimpleNamespace(commit=lambda:None)),
        Secret=SimpleNamespace(from_name=lambda *args,**kwargs:None))
    monkeypatch.setitem(sys.modules,'modal',modal)
    if external_tool:
        from pipelines_v2.runtime.external_tool import ExternalToolRuntimeSpec
        from pipelines_v2.operations.attribution import official_tool_lock
        monkeypatch.setattr(ToyEngine,'runtime_spec',lambda self:ExternalToolRuntimeSpec(tool_lock=official_tool_lock()))
        monkeypatch.setattr('pipelines_v2.runtime.modal_worker.build_locked_image',lambda *args:Image())
    operation=GenerationRunSpec(engine=ToyEngine(),dataset=Dataset.from_examples([Example(key='a',prompt='a')]),generation=GenerationSpec(enabled=True,max_tokens=1))
    identity={'kind':'modal','resources':{'gpu':'A100-80GB','timeout_seconds':3,'client_timeout_seconds':5}}
    record=AdmissionRecord((AdmissionCheck('cache','readiness',status,'model','fixture cache check'),),resolve_provenance(operation),workload_envelope(operation,identity).to_dict(),scope='cpu_preflight')
    monkeypatch.setattr('pipelines_v2.runtime.readiness.admit_execution',lambda *args,**kwargs:record)
    monkeypatch.setattr('pipelines_v2.runtime.remote_executor.execute_remote',lambda **kwargs:{'runner':{}})
    args=dict(runner_config=identity,store_config={'kind':'modal_volume','name':'fixture','root':str(tmp_path)},spec_payload=operation.to_dict())
    if status=='failed':
        with pytest.raises(ValueError,match='before GPU'):
            run_on_modal(**args)
        assert calls==['_remote_readiness']
    else:
        run_on_modal(**args)
        assert calls==['_remote_readiness','_remote_execute_stream']
    assert resources['_remote_readiness']['memory']==(10240 if external_tool else 4096)
    assert 'gpu' not in resources['_remote_readiness']
    assert 'gpu' not in resources['_remote_merge_shards']
    assert resources['_remote_execute_stream']['gpu']=='A100-80GB'
    assert resources['_remote_merge_shards']['memory']==4096
