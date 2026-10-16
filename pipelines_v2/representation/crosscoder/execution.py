"""Artifact operations for the bounded paired-model capability."""
import numpy as np
from pipelines_v2.core.provenance import content_hash
from pipelines_v2.operations.execution.common import OperationExecutionResult
from pipelines_v2.storage.finalization import validate_completion
from .alignment import align_sides,validate_pairs
from .fitting import FitConfig,fit,validate_checkpoint,CrosscoderDictionary
from .qwen import QwenSession
from .reference import verify_runtime
from .settings import Execution,Limits,check_deadline


def coverage(keys):return {'materialized':True,'example_count':len(keys),'example_keys':list(keys)}


def checked_feature(artifact,name):
    validate_completion(artifact.manifest(),store=artifact.store)
    return artifact.feature(name).load()


def capture_examples(spec):
    """Bounded preparation; Postgres is iterated before any model allocation."""
    from pipelines_v2.data.datasets import Example
    from .settings import Limits
    limits=Limits(**spec.limits)
    if spec.dataset is None:
        rows=[Example(key=f'prompt_{i}',prompt=prompt) for i,prompt in enumerate(spec.prompts)]
    elif not spec.dataset.is_deferred:
        rows=list(spec.dataset.examples)
    else:
        from pipelines_v2.data.sources import source_from_dict
        source=source_from_dict(dict(spec.dataset.source))
        if not hasattr(source,'iter_dataset_batches'):
            raise ValueError('Prepare this dataset as bounded examples, or use streaming Postgres capture')
        if spec.dataset.selection:
            raise ValueError('Apply the explicit selection in the Postgres fetch before crosscoder capture')
        rows=[]
        for batch in source.iter_dataset_batches(**dict(spec.dataset.fetch),batch_size=min(limits.max_examples+1,256)):
            rows.extend(batch.examples)
            if len(rows)>limits.max_examples:raise ValueError('Example preparation budget exceeded')
    if len(rows)<2 or len(rows)>limits.max_examples or len({row.key for row in rows})!=len(rows):
        raise ValueError('Capture requires distinct document keys within the explicit example budget')
    if any(not isinstance(row.prompt,str) or not row.prompt.strip() or row.images for row in rows):
        raise ValueError('Aligned crosscoder capture requires nonempty raw text, without images/chat templates')
    return rows


def _run_capture(spec):
    import torch
    from .settings import Execution,Limits,check_deadline
    execution=Execution(**spec.execution);limits=Limits(**spec.limits);deadline=limits.deadline()
    examples=capture_examples(spec)
    train_keys=list(spec.train_keys) or [row.key for row in examples[:spec.train_count]]
    if not train_keys or not set(train_keys)<{row.key for row in examples}:
        raise ValueError('Train keys must be a nonempty proper subset of capture keys')
    # Tokenize both pinned models and reject incompatible input coordinates before weights.
    from transformers import AutoTokenizer
    from .qwen import model_identity
    tokenized=[];models=[]
    for side,path in enumerate(spec.model_paths):
        check_deadline(deadline);models.append(model_identity(path,side))
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        tokenized.append([tokenizer.encode(row.prompt,add_special_tokens=False) for row in examples])
    if tokenized[0]!=tokenized[1]:raise ValueError('Tokenization differs across the pinned model pair')
    total=sum(map(len,tokenized[0]))
    if any(not ids or len(ids)>min(limits.max_tokens_per_example,32768) for ids in tokenized[0]) or total>limits.max_total_tokens:
        raise ValueError('Capture token preparation budget exceeded')
    if total*2*896*4>limits.max_activation_bytes:raise ValueError('Paired activation preparation budget exceeded')
    sides=[];oracles=[]
    for side,path in enumerate(spec.model_paths):
        session=QwenSession(path,side=side,execution=execution,limits=limits)
        try:
            rows=[];oracles.append(session.parity(tokenized[side][0],layer=spec.layer))
            for example,ids in zip(examples,tokenized[side],strict=True):
                check_deadline(deadline);_,values=session.forward(ids,layer=spec.layer)
                rows.append({'key':example.key,'token_ids':ids,'positions':list(range(len(ids))),
                    'prompt_digest':content_hash(example.prompt),'values':values[0].detach().cpu().numpy().copy()})
            sides.append(rows)
        finally:session.close()
    pairs=align_sides(sides,models,layer=spec.layer,train_keys=train_keys,execution=spec.execution)
    return OperationExecutionResult(payload={'scope':'aligned paired-model capture','alignment':pairs['alignment'],
        'parity':oracles,'prompts':{row.key:row.prompt for row in examples}},features={'pairs':pairs},
        metadata={'paired_alignment_v1':pairs['alignment']},example_coverage=coverage(pairs['records']))


def fit_in_subprocess(pairs,reference_path,config,checkpoint,stop_after,execution,limits):
    import json,sys,tempfile
    from pathlib import Path
    from pipelines_v2.storage.local import LocalArtifactStore
    from pipelines_v2.storage.features import write_capture_features,load_feature_payload
    from pipelines_v2.runtime.external_tool import run_bounded,external_tool_environment
    from .settings import Limits
    budget=Limits(**limits)
    with tempfile.TemporaryDirectory(prefix='crosscoder-fit-') as directory:
        root=Path(directory);store=LocalArtifactStore(root)
        refs=write_capture_features(store,'input',{'pairs':pairs,**({'checkpoint':checkpoint} if checkpoint is not None else {})})
        def serial(ref):return ref.to_dict() if hasattr(ref,'to_dict') else ref
        request={'pairs':serial(refs['pairs']),'checkpoint':serial(refs['checkpoint']) if checkpoint is not None else None,
                 'reference_path':reference_path,'config':config,'stop_after':stop_after,'execution':execution,'limits':limits}
        (root/'request.json').write_text(json.dumps(request))
        env=external_tool_environment()
        # Explicit mounted/installed Xenon root, never inherited arbitrary Python paths.
        env['PYTHONPATH']=str(Path(__file__).resolve().parents[3])
        run_bounded([sys.executable,'-m','pipelines_v2.representation.crosscoder.fit_worker',str(root)],
                    cwd=root,env=env,output_dir=root,timeout_seconds=budget.timeout_seconds,
                    memory_bytes=budget.max_process_rss_bytes,output_bytes=budget.max_output_bytes)
        state=load_feature_payload(store,json.loads((root/'response.json').read_text()))
        validate_checkpoint(state)
        return state


def _run_fit(spec):
    pairs=checked_feature(spec.pairs,'pairs');validate_pairs(pairs)
    prior=checked_feature(spec.resume_from,'checkpoint') if spec.resume_from is not None else None
    state=fit_in_subprocess(pairs,spec.reference_path,spec.config,prior,spec.stop_after,spec.execution,spec.limits)
    return OperationExecutionResult(payload={'complete':state['complete'],'step':state['step'],'fit_identity':state['fit_identity'],
        'checkpoint_digest':state['checkpoint_digest'],'losses':state['losses'],
        'checkpoint_boundary':'explicit stop_after; ordinary C03 operation artifact',
        'scope':'configured reference fit; feature quality not evaluated'},features={'checkpoint':state},
        metadata={'crosscoder_fit_v1':{'identity':state['fit_identity'],'step':state['step'],'complete':state['complete']}},
        example_coverage=coverage([row['key'] for row in state['alignment']['rows'] if row['split']=='train']))


def _run_import(spec):
    state=checked_feature(spec.checkpoint,'checkpoint');dictionary=CrosscoderDictionary(state,spec.reference_path,Execution(**spec.execution))
    return OperationExecutionResult(payload={'dictionary_identity':dictionary.identity,'alignment':state['alignment'],
        'reference':state['reference'],'fit_complete':True,'raw_code_convention':'CrossCoder.encode; not norm-scaled forward output_features'},
        features={'dictionary':state},example_coverage=coverage([]))


def dictionary_from(artifact,reference_path,execution):return CrosscoderDictionary(checked_feature(artifact,'dictionary'),reference_path,Execution(**execution))


def _run_encode(spec):
    dictionary=dictionary_from(spec.dictionary,spec.reference_path,spec.execution);pairs=checked_feature(spec.feature.artifact,spec.feature.name);alignment=validate_pairs(pairs)
    if alignment['identity']!=dictionary.checkpoint['alignment']['identity']:raise ValueError('Dictionary/capture alignment or model-side order differs')
    records={};errors={}
    for row in alignment['rows']:
        key=row['key'];record=pairs['records'][key];values=np.asarray(record['values'])
        codes=dictionary.encode(values);reconstruction=dictionary.decode(codes)
        records[key]={'token_ids':record['token_ids'],'positions':record['positions'],'values':codes}
        errors[key]={'split':row['split'],'mse_by_model':np.mean((values-reconstruction)**2,axis=(0,2)).tolist(),
                     'mean_active_features':float((codes>0).sum(-1).mean())}
    return OperationExecutionResult(payload={'dictionary_identity':dictionary.identity,'reconstruction':errors},
        features={'codes':{'kind':'crosscoder_tensors','dictionary_identity':dictionary.identity,'alignment':alignment,'records':records}},example_coverage=coverage(records))


def _run_decode(spec):
    dictionary=dictionary_from(spec.dictionary,spec.reference_path,spec.execution);codes=checked_feature(spec.feature.artifact,spec.feature.name)
    if codes['dictionary_identity']!=dictionary.identity:raise ValueError('Codes belong to another dictionary')
    records={key:{**record,'values':dictionary.decode(record['values'])} for key,record in codes['records'].items()}
    return OperationExecutionResult(payload={'dictionary_identity':dictionary.identity,'meaning':'paired reconstruction, not a no-op'},
        features={'reconstruction':{'kind':'crosscoder_tensors','alignment':codes['alignment'],'records':records}},example_coverage=coverage(records))


def _run_splice(spec):
    import torch
    execution=Execution(**spec.execution);limits=Limits(**spec.limits);deadline=limits.deadline()
    dictionary=dictionary_from(spec.dictionary,spec.reference_path,spec.execution)
    pairs=checked_feature(spec.pairs,'pairs');alignment=validate_pairs(pairs)
    if alignment['identity']!=dictionary.checkpoint['alignment']['identity']:raise ValueError('Dictionary and paired capture differ')
    if alignment.get('execution',spec.execution)!=spec.execution:raise ValueError('Splice numerical execution must match capture')
    heldout=[row for row in alignment['rows'] if row['split']=='heldout']
    if not heldout or len(heldout)>limits.max_examples:raise ValueError('Heldout rows exceed explicit example budget')
    deltas=dictionary.feature_delta(spec.feature_index,spec.amount)
    model=dictionary.model
    if model.has_activation_normalizer:raise ValueError('Unsupported normalization for independent splice oracle')
    results=[]
    for side,path in enumerate(spec.model_paths):
        session=QwenSession(path,side=side,execution=execution,limits=limits)
        try:
            if session.identity!=alignment['model_sides'][side]:raise ValueError('Original model side changed since capture')
            for row in heldout:
                check_deadline(deadline)
                values=pairs['records'][row['key']]['values']
                reconstruction=dictionary.decode(dictionary.encode(values))
                baseline,original=session.forward(row['token_ids'],layer=alignment['layer']);outputs={}
                torch.testing.assert_close(original[0].cpu(),torch.from_numpy(values[:,side]),rtol=1e-4,atol=1e-5)
                delta=torch.from_numpy(deltas[side]).to(execution.device)
                replacement=torch.from_numpy(reconstruction[-1,side]).to(execution.device)
                for mode in ('noop','delta','reconstruct'):
                    def edit(value,mode=mode):
                        if mode=='noop':return value
                        changed=value.clone();changed[:,-1]=value[:,-1]+delta if mode=='delta' else replacement
                        return changed
                    outputs[mode]=session.forward(row['token_ids'],layer=alignment['layer'],edit=edit)[0]
                if not torch.equal(outputs['noop'],baseline):raise ValueError('Feature no-op changed logits')
                bias=model.decoder.bias[side].detach()
                independent_delta=(spec.amount*model.decoder.weight[side,spec.feature_index].detach()+bias)-bias
                torch.testing.assert_close(independent_delta,delta,rtol=1e-4,atol=1e-5)
                def direct_hook(module,args):
                    changed=args[0].clone();changed[:,-1]+=independent_delta;return (changed,*args[1:])
                handle=session.model.model.layers[alignment['layer']].register_forward_pre_hook(direct_hook)
                try:
                    ids=torch.tensor([row['token_ids']],device=execution.device)
                    with torch.inference_mode():oracle=session.model(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(ids.shape[1],device=execution.device)[None],use_cache=False).logits
                finally:handle.remove()
                torch.testing.assert_close(oracle,outputs['delta'],rtol=1e-4,atol=1e-5)
                results.append({'model':session.identity,'key':row['key'],'token_count':len(row['token_ids']),
                    'arms':{mode:{'max_abs_logit_delta':float((output[0,-1]-baseline[0,-1]).abs().max())} for mode,output in outputs.items()},
                    'direct_delta_logit_error':float((oracle-outputs['delta']).abs().max())})
        finally:session.close()
    return OperationExecutionResult(payload={'scope':'configured original-model feature edits','scientific_verdict':'not_tested',
        'feature_index':spec.feature_index,'amount':spec.amount,'coordinate':'original resid_pre last prompt token',
        'rows':results,'dictionary_identity':dictionary.identity},example_coverage=coverage([row['key'] for row in heldout]))


def _execute(spec, callback):
    from .settings import numerical_scope
    execution=Execution(**spec.execution);limits=Limits(**spec.limits)
    verify_runtime(execution.runtime_profile)
    with numerical_scope(execution,limits):
        result=callback(spec)
    # Bound materialized payload/tensors before the generic runner publishes them.
    from .fitting import tree_digest
    import json
    def array_bytes(value):
        if isinstance(value,np.ndarray):return value.nbytes
        if isinstance(value,dict):return sum(array_bytes(item) for item in value.values())
        if isinstance(value,(tuple,list)):return sum(array_bytes(item) for item in value)
        return 0
    size=array_bytes(result.features)+len(json.dumps(tree_digest({'payload':result.payload,'features':result.features})).encode())
    if size>limits.max_output_bytes:raise ValueError('Crosscoder output exceeds explicit byte budget')
    return result


def run_capture(spec):return _execute(spec,_run_capture)
def run_fit(spec):return _execute(spec,_run_fit)
def run_import(spec):return _execute(spec,_run_import)
def run_encode(spec):return _execute(spec,_run_encode)
def run_decode(spec):return _execute(spec,_run_decode)
def run_splice(spec):return _execute(spec,_run_splice)
