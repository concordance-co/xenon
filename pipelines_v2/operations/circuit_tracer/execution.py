import base64
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from pipelines_v2.runtime.external_tool import run_bounded,external_tool_environment
from pipelines_v2.operations.execution.common import OperationExecutionResult
from pipelines_v2.operations.graphs.contracts import digest
from .importer import import_graph


def request_for(spec,output):
    root=Path(__file__).parent
    requirements=(root/'requirements.lock').read_text()
    if hashlib.sha256(requirements.encode()).hexdigest()!=spec.runtime.tool_lock['requirements_sha256']:raise ValueError('Tool requirements content differs from locked identity')
    return {'scientific':spec.request,'model_path':spec.model_path,'dictionary_path':spec.dictionary_path,'output_dir':str(output),
            'tool_lock':spec.runtime.tool_lock,'requirements':requirements}


def import_result(spec,output,compute):
    output=Path(output)
    binding=json.loads((output/'producer_binding.json').read_text())
    if binding['request_identity']!=digest(spec.request) or set(binding['outputs'])!={'upstream_graph.pt','upstream_graph.json'}:raise ValueError('Trace producer binding mismatch')
    for name,sha in binding['outputs'].items():
        if hashlib.sha256((output/name).read_bytes()).hexdigest()!=sha:raise ValueError('Trace output bytes changed')
    raw=json.loads((output/'upstream_graph.json').read_text());graph=import_graph(raw,expected_request=spec.request)
    blobs={};total=0
    for name in ('upstream_graph.pt','upstream_graph.json','producer_binding.json','stdout.log','stderr.log'):
        path=output/name
        if not path.is_file():continue
        data=path.read_bytes();total+=len(data)
        if total>spec.output_bytes:raise ValueError('Trace persistence budget exceeded')
        blobs[name]={'encoding':'base64','sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'data':base64.b64encode(data).decode()}
    return {'kind':'circuit_trace','schema_version':1,'graph':graph.to_dict(),'raw_outputs':blobs,'fidelity':raw['fidelity'],
        'original_model_validation':raw['original_model_validation'],'compute':{**compute,**raw['compute']},'tool_packages':raw['tool_packages'],
        'dictionary_type':raw['dictionary_type'],'scientific_verdict':'not_tested','engineering_status':'passed'}


def run_circuit_trace(spec):
    root=Path(tempfile.mkdtemp(prefix='xenon-circuit-trace-'));output=root/'output';output.mkdir()
    request=root/'request.json';request.write_text(json.dumps(request_for(spec,output)))
    env={**external_tool_environment(),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','WANDB_MODE':'disabled',
         'PYTHONDONTWRITEBYTECODE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    compute=run_bounded([spec.runtime.tool_python,str(Path(__file__).with_name('bridge.py')),'run',str(request)],cwd=root,env=env,
        output_dir=output,timeout_seconds=spec.timeout_seconds,memory_bytes=spec.memory_bytes,output_bytes=spec.output_bytes)
    result=import_result(spec,output,compute)
    shutil.rmtree(root)
    return OperationExecutionResult(payload=result,metadata={'circuit_tracer_v1':{'request_identity':digest(spec.request),
        'tool_identity':spec.runtime.tool_identity,'dictionary_type':'PLT','scientific_verdict':'not_tested'}},example_coverage={'prompts':1})
