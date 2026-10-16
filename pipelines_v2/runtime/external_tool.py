"""Isolated, file-based upstream tool execution. No tool imports in control Python."""
from __future__ import annotations
from dataclasses import dataclass, field, fields
import hashlib
import json
import os
import logging
import re
import shutil
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

from pipelines_v2.engine.base import PythonRuntimeSpec


def external_tool_environment(inherited=None):
    """Keep OS/GPU settings while preventing control-Python import shadowing.

    Callers add their explicit offline and thread policy after this boundary.
    The tool interpreter and bridge's pinned source paths select Python code;
    the control process's import paths and interactive startup settings do not.
    """
    excluded = {
        'PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'PYTHONSTARTUP',
        'PYTHONINSPECT', 'PYTHONBREAKPOINT', 'PYTHONEXECUTABLE',
        'PYTHONPLATLIBDIR', 'PYTHONCASEOK', 'PYTHONPYCACHEPREFIX',
        '__PYVENV_LAUNCHER__',
    }
    source = os.environ if inherited is None else inherited
    env = {key: value for key, value in source.items() if key not in excluded}
    env.update(PYTHONNOUSERSITE='1', PYTHONSAFEPATH='1')
    return env


@dataclass(frozen=True, slots=True)
class ExternalToolRuntimeSpec(PythonRuntimeSpec):
    kind: str = 'external_tool'
    tool_lock: dict[str, Any] = field(default_factory=dict)
    tool_root: str = '/opt/xenon-tools/mattr'
    mib_root: str = '/opt/xenon-tools/mib'
    tool_python: str = '/opt/xenon-tools/venv/bin/python'
    readiness_memory_bytes: int = field(default=8*1024**3, metadata={'omit_if_default': True})
    readiness_timeout_seconds: int = field(default=120, metadata={'omit_if_default': True})

    def __post_init__(self):
        if self.kind != 'external_tool' or self.tool_lock.get('schema_version') != 1:
            raise ValueError('Unsupported external tool descriptor')
        if any(type(value) is not int or value <= 0 for value in (self.readiness_memory_bytes,self.readiness_timeout_seconds)):
            raise ValueError('External tool readiness memory and timeout must be positive integers')
        from pipelines_v2.operations.attribution.contracts import MATTR_COMMIT, MIB_COMMIT, EAP_COMMIT
        lock=self.tool_lock
        if (lock.get('revision'),lock.get('mib_revision'),lock.get('submodules',{}).get('EAP-IG',{}).get('revision')) != (MATTR_COMMIT,MIB_COMMIT,EAP_COMMIT):
            raise ValueError('Unsupported external tool source pins')
        object.__setattr__(self,'runtime_identity',{'external_tool_lock': self.tool_identity})

    @property
    def tool_identity(self):
        from pipelines_v2.operations.attribution.contracts import digest
        return digest(self.tool_lock)

    def control_runtime(self):
        return PythonRuntimeSpec(**{f.name:getattr(self,f.name) for f in fields(PythonRuntimeSpec) if f.name!='kind'})

    @classmethod
    def from_dict(cls,payload):
        control=PythonRuntimeSpec.from_dict({k:v for k,v in payload.items() if k not in {'tool_lock','tool_root','mib_root','tool_python','readiness_memory_bytes','readiness_timeout_seconds','kind'}})
        return cls(**{f.name:getattr(control,f.name) for f in fields(PythonRuntimeSpec) if f.name!='kind'},
                   **{k:payload[k] for k in ('tool_lock','tool_root','mib_root','tool_python','readiness_memory_bytes','readiness_timeout_seconds') if k in payload})


def _git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True,timeout=30,stderr=subprocess.DEVNULL).strip()


def verify_tool_source(runtime):
    root=Path(runtime.tool_root);mib=Path(runtime.mib_root);lock=runtime.tool_lock
    for path,revision in [(root,lock['revision']),(mib,lock['mib_revision']),(mib/'EAP-IG',lock['submodules']['EAP-IG']['revision'])]:
        if _git(path,'rev-parse','HEAD')!=revision or _git(path,'status','--porcelain','--untracked-files=no'):
            raise ValueError('External source checkout differs from locked revision')
    for path,key in [(root/'uv.lock','uv_lock_sha256'),(root/lock['entrypoint'],'entrypoint_sha256'),(mib/'MIB_circuit_track/evaluation.py','evaluator_source_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=lock[key]:
            raise ValueError('External entrypoint, dependency lock or evaluator bytes changed')
    if not Path(runtime.tool_python).is_file():
        raise ValueError('Missing isolated tool interpreter; provision the locked tool environment')


def run_bounded(argv, *, cwd, env, output_dir, timeout_seconds, memory_bytes, output_bytes):
    """One attempt; bound aggregate process RSS, output and wall time; kill group on failure."""
    import psutil
    if not argv or timeout_seconds<=0 or memory_bytes<=0 or output_bytes<=0:
        raise ValueError('External tool requires finite positive budgets and argument vector')
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();peak=0;reason=None
    with (output/'stdout.log').open('wb') as stdout,(output/'stderr.log').open('wb') as stderr:
        process=subprocess.Popen(list(argv),cwd=cwd,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        try:
            while True:
                code=process.poll()
                try:
                    parent=psutil.Process(process.pid);rss=sum(p.memory_info().rss for p in [parent,*parent.children(recursive=True)])
                    peak=max(peak,rss)
                except psutil.Error:
                    pass
                size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file() and not p.is_symlink())
                if time.monotonic()-start>timeout_seconds:reason='wall deadline'
                if peak>memory_bytes:reason='process-tree RSS budget'
                if size>output_bytes:reason='output disk budget'
                if reason:raise RuntimeError(f'External tool exceeded {reason}; peak_rss_bytes={peak}; memory_limit_bytes={memory_bytes}; elapsed_seconds={time.monotonic()-start:.3f}; diagnostics: {output}')
                if code is not None:
                    if code:raise RuntimeError(f'External tool exited {code}; diagnostics: {output}')
                    break
                time.sleep(.05)
        finally:
            # A child can outlive a successful leader; the entire owned process group ends here.
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
    return {'elapsed_seconds':time.monotonic()-start,'peak_process_tree_rss_bytes':peak,
            'exit_code':code,'argv':list(argv)}


def tool_readiness_profile(runtime):
    """Explicit child-process ceiling plus control-interpreter host headroom."""
    return {'memory_bytes':runtime.readiness_memory_bytes,
            'timeout_seconds':runtime.readiness_timeout_seconds,
            'container_memory_mb':(runtime.readiness_memory_bytes+1024**2-1)//1024**2+2048,
            'container_timeout_minimum':runtime.readiness_timeout_seconds+30}


class ToolReadinessError(RuntimeError):
    """Sanitized, bounded failure detail safe for readiness records/logs."""


def _redact_readiness_detail(text, environment):
    for key,value in environment.items():
        if value and any(part in key.upper() for part in ('TOKEN','SECRET','PASSWORD','CREDENTIAL','API_KEY','DATABASE_URL')):
            text=text.replace(value,'[redacted]')
    text=re.sub(r'(https?://)[^/@\s]+:[^/@\s]+@',r'\1[redacted]@',text)
    text=re.sub(r'(?i)(bearer\s+|(?:token|password|api_key)=)[^\s&]+',r'\1[redacted]',text)
    return text[-4000:]


def check_tool_runtime(runtime):
    verify_tool_source(runtime)
    request={'tool_root':runtime.tool_root,'mib_root':runtime.mib_root}
    bridge=Path(__file__).parents[1]/'operations/attribution/bridge.py'
    root=Path(tempfile.mkdtemp(prefix='xenon-tool-check-'))
    p=root/'request.json';p.write_text(json.dumps(request))
    env={**external_tool_environment(),'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','WANDB_MODE':'disabled',
         'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    profile=tool_readiness_profile(runtime)
    try:
        result=run_bounded([runtime.tool_python,str(bridge),'check',str(p)],cwd=runtime.tool_root,env=env,
                    output_dir=root/'logs',timeout_seconds=profile['timeout_seconds'],memory_bytes=profile['memory_bytes'],output_bytes=1024**2)
    except Exception as exc:
        stderr=root/'logs'/'stderr.log'
        tail=''
        if stderr.is_file():
            with stderr.open('rb') as stream:
                stream.seek(max(0,stderr.stat().st_size-16384))
                tail=stream.read(16384).decode('utf-8',errors='replace')
        detail=_redact_readiness_detail(f'{type(exc).__name__}: {exc}; diagnostics retained at {root}; stderr: {tail}',env)
        logging.getLogger(__name__).error('External tool readiness failed: %s',detail)
        raise ToolReadinessError(detail) from exc
    else:
        shutil.rmtree(root)
        return result


def _provision_mattr(root,mib,venv,lock):
    """Image-build function: explicit argv, pinned sources, upstream's untouched uv lock."""
    import subprocess,os,pathlib
    for url,dest,commit in [(lock['repository'],root,lock['revision']),(lock['mib_repository'],mib,lock['mib_revision'])]:
        subprocess.run(['git','clone',url,dest],check=True,timeout=120)
        subprocess.run(['git','-C',dest,'checkout','--detach',commit],check=True,timeout=30)
    subprocess.run(['git','-C',mib,'submodule','update','--init','--recursive'],check=True,timeout=120)
    env={**os.environ,'UV_PROJECT_ENVIRONMENT':venv,'UV_PYTHON_INSTALL_DIR':'/opt/xenon-tools/python'}
    subprocess.run(['uv','sync','--frozen','--no-dev','--python',lock['python_version'],'--project',root],env=env,check=True,timeout=900)
    subprocess.run([str(pathlib.Path(venv)/'bin/python'),'-c',
        "import sys;sys.path[:0]=[sys.argv[1],sys.argv[1]+'/EAP-IG/src'];import matryoshka_attribution,transformer_lens;from MIB_circuit_track.evaluation import evaluate_area_under_curve",mib],check=True,timeout=120)


def build_external_image(modal,runtime):
    from pipelines_v2.runtime.locks import build_locked_image
    image=build_locked_image(modal,runtime.control_runtime())
    image=image.apt_install('git').pip_install('uv==0.8.11')
    return image.run_function(_provision_mattr,kwargs={'root':runtime.tool_root,'mib':runtime.mib_root,
        'venv':str(Path(runtime.tool_python).parent.parent),'lock':runtime.tool_lock},timeout=1200)
