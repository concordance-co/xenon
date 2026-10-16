"""Load only unmodified, hash-pinned upstream dictionary/trainer modules."""
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import types


def lock():return json.loads(Path(__file__).with_name('reference_lock.json').read_text())


def reference_modules(path):
    # This CPU-only integration has no multiprocessing progress writers.
    # Avoid tqdm's semaphore tracker outliving the supervised worker.
    import threading
    from tqdm import tqdm
    tqdm.set_lock(threading.RLock())
    root=Path(path).resolve();manifest=lock()
    for relative,digest in manifest['files'].items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError('Crosscoder reference source mismatch: '+relative)
    name='_xenon_crosscoder_'+manifest['revision']
    # Avoid upstream __init__ importing the unused NNsight dataset collector.
    # The actual dictionary, utilities and trainer source execute unchanged.
    if name not in sys.modules:
        package=types.ModuleType(name);package.__path__=[str(root/'dictionary_learning')];sys.modules[name]=package
    elif list(sys.modules[name].__path__)!=[str(root/'dictionary_learning')]:
        raise ValueError('Pinned reference already loaded from a different source root')
    return importlib.import_module(name+'.dictionary'),importlib.import_module(name+'.trainers.crosscoder')


def runtime_spec(profile="darwin-arm64"):
    from pipelines_v2.engine.base import PythonRuntimeSpec
    from pipelines_v2.runtime.locks import RuntimeLock
    requirements=Path(__file__).with_name('requirements-'+profile+'.lock').read_text()
    if profile == 'darwin-arm64':
        runtime=RuntimeLock('crosscoder-cpu',profile,'3.13',requirements,('torch','transformers','datasets','zstandard','einops'), 'host-cpu-python-3.13')
    elif profile == 'linux-x86_64-cuda':
        base=json.loads(Path(__file__).with_name('linux_profile.json').read_text())
        runtime=RuntimeLock('crosscoder-cuda',profile,base['python_version'],requirements,('torch','transformers','datasets','zstandard','einops','psutil'),base['base_image'],cuda=base['cuda'])
    else:
        raise ValueError('Unsupported crosscoder runtime profile')
    return PythonRuntimeSpec(python_version='3.13',lock=runtime,local_python_sources=('pipelines_v2',),env={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})


def verify_runtime(profile="darwin-arm64"):
    runtime=runtime_spec(profile).lock;errors=list(runtime.verify_target())
    for name,version in runtime.packages.items():
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual=None
        if actual!=version:errors.append(f'{name}: expected {version}, observed {actual}')
    if errors:raise ValueError('Crosscoder isolated runtime differs: '+'; '.join(errors))
    return runtime.identity
