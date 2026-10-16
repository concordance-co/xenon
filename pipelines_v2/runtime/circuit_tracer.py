"""Separate circuit-tracer interpreter, source pin and complete dependency lock."""
from dataclasses import dataclass,field,fields
import hashlib
import json
from pathlib import Path
from pipelines_v2.engine.base import PythonRuntimeSpec


def tool_directory():return Path(__file__).parents[1]/'operations/circuit_tracer'
def official_lock():return json.loads((tool_directory()/'tool_lock.json').read_text())


@dataclass(frozen=True,slots=True)
class CircuitTracerRuntimeSpec(PythonRuntimeSpec):
    kind: str = 'circuit_tracer_tool'
    tool_lock: dict = field(default_factory=official_lock)
    tool_python: str = '/opt/xenon-tracer/bin/python'

    def __post_init__(self):
        if self.kind!='circuit_tracer_tool' or self.tool_lock!=official_lock():raise ValueError('Unsupported circuit-tracer source/dependency lock')
        object.__setattr__(self,'runtime_identity',{'circuit_tracer_lock':self.tool_identity})

    @property
    def tool_identity(self):
        return hashlib.sha256(json.dumps(self.tool_lock,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def control_runtime(self):return PythonRuntimeSpec(**{f.name:getattr(self,f.name) for f in fields(PythonRuntimeSpec) if f.name!='kind'})

    @classmethod
    def from_dict(cls,payload):
        control=PythonRuntimeSpec.from_dict({k:v for k,v in payload.items() if k not in {'tool_lock','tool_python','kind'}})
        return cls(**{f.name:getattr(control,f.name) for f in fields(PythonRuntimeSpec) if f.name!='kind'},
                   **{k:payload[k] for k in ('tool_lock','tool_python') if k in payload})


def _install_tool(requirements, tool_python):
    import subprocess,tempfile,pathlib
    with tempfile.TemporaryDirectory() as root:
        path=pathlib.Path(root)/'requirements.lock';path.write_text(requirements)
        subprocess.run(['uv','venv','--python','3.12',str(pathlib.Path(tool_python).parent.parent)],check=True,timeout=180)
        subprocess.run(['uv','pip','install','--python',tool_python,'--no-deps','-r',str(path)],check=True,timeout=900)
        subprocess.run([tool_python,'-c','import torch,transformer_lens,circuit_tracer;from circuit_tracer import ReplacementModel,attribute'],check=True,timeout=120)


def build_circuit_tracer_image(modal,runtime):
    from pipelines_v2.runtime.locks import build_locked_image
    requirements=(tool_directory()/'requirements.lock').read_text()
    if hashlib.sha256(requirements.encode()).hexdigest()!=runtime.tool_lock['requirements_sha256']:raise ValueError('Circuit-tracer requirements changed')
    image=build_locked_image(modal,runtime.control_runtime()).apt_install('git').pip_install('uv==0.8.11')
    return image.run_function(_install_tool,kwargs={'requirements':requirements,'tool_python':runtime.tool_python},timeout=1200)


def check_circuit_tracer_runtime(runtime):
    import tempfile
    from pipelines_v2.runtime.external_tool import run_bounded,external_tool_environment
    with tempfile.TemporaryDirectory(prefix='xenon-tracer-readiness-') as root:
        root=Path(root);request=root/'request.json'
        request.write_text(json.dumps({'tool_lock':runtime.tool_lock,'requirements':(tool_directory()/'requirements.lock').read_text()}))
        env={**external_tool_environment(),'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
        return run_bounded([runtime.tool_python,str(tool_directory()/'bridge.py'),'check',str(request)],cwd=root,env=env,
            output_dir=root/'logs',timeout_seconds=120,memory_bytes=4*1024**3,output_bytes=1024**2)
