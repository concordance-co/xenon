"""Artifact executors and explicit opt-in registration for SAE operations."""
import numpy as np
from pipelines_v2.operations.execution.common import OperationExecutionResult
from .dictionary import SAEDictionary, centered, validate_coordinate_evidence, verify_original_model
from .splice import torch_splice


def load_dictionary(artifact):
    payload = artifact.feature('dictionary').load()
    return SAEDictionary(payload['weights'], payload['metadata'])


def run_import(spec):
    from pipelines_v2.engine.torch.engine import checkpoint_identity
    dictionary = SAEDictionary.import_pinned(spec.cache_root,
        model_checkpoint_digest=verify_original_model(spec.model_path),coordinate_evidence=spec.coordinate_evidence)
    return OperationExecutionResult(payload=dictionary.metadata,
        features={'dictionary': {'kind':'sae_tensors','weights':dictionary.weights,'metadata':dictionary.metadata}},
        metadata={'dictionary_v1':dictionary.metadata},
        example_coverage={'materialized':True,'example_count':0,'example_keys':[]})


def run_encode(spec):
    dictionary = load_dictionary(spec.dictionary)
    payload = spec.feature.load()
    expected = dictionary.metadata['input_site']
    if payload.get('kind')!='residual' or payload.get('site')!='resid_pre' or payload.get('model_identity')!=expected['checkpoint_digest']:
        raise ValueError('Input capture is not the qualified original GPT2 residual coordinate system')
    if '6' not in payload.get('layers',{}): raise ValueError('SAE input requires original GPT2 layer6')
    records = {}; fidelity = {}
    for key, row in payload['layers']['6'].items():
        values = np.asarray(row['values'],dtype=np.float32)
        records[key] = {'tokens':row['tokens'],'values':dictionary.encode_original(values)}
        fidelity[key] = dictionary.fidelity(values)
    return OperationExecutionResult(payload={'dictionary_id':spec.dictionary.id,'fidelity':fidelity},
        features={'codes':{'kind':'sae_tensors','dictionary_id':spec.dictionary.id,'records':records}},
        example_coverage={'materialized':True,'example_count':len(records),'example_keys':list(records)})


def run_decode(spec):
    dictionary = load_dictionary(spec.dictionary); payload=spec.feature.load()
    if payload.get('kind')!='sae_tensors' or payload.get('dictionary_id')!=spec.dictionary.id:
        raise ValueError('Feature codes belong to a different dictionary')
    records={key:{'tokens':row['tokens'],'values':dictionary.decode(row['values'])} for key,row in payload['records'].items()}
    return OperationExecutionResult(payload={'dictionary_id':spec.dictionary.id,'coordinate_system':'centered TransformerLens residual; not an original activation/no-op'},
        features={'reconstruction':{'kind':'sae_tensors','dictionary_id':spec.dictionary.id,'records':records}},
        example_coverage={'materialized':True,'example_count':len(records),'example_keys':list(records)})


def run_splice(spec):
    import torch
    from pipelines_v2.engine.torch.engine import checkpoint_identity
    dictionary=load_dictionary(spec.dictionary)
    validate_coordinate_evidence(dictionary.metadata['coordinate_evidence'],verify_original_model(spec.engine.model_id))
    rows=[]; edits=dict(spec.feature_edits)
    with spec.engine.open_session() as session:
        for example in spec.examples():
            prompt=example.prompt
            ids,mask=session.encode([prompt])
            if ids.shape[1]>spec.max_prompt_tokens:raise ValueError('SAE prompt exceeds max_prompt_tokens')
            baseline,capture=session.forward(ids,mask,capture=(('resid_pre',6),))
            # Session forward returns logits and capture payloads; all edits target only the last prompt token.
            original=capture['resid_pre',6].cpu().numpy()
            entry={'example_key':example.key,'prompt':prompt,'token_count':int(ids.shape[1]),'fidelity':dictionary.fidelity(original[0]),'arms':{}}
            outputs={}
            for mode in ('noop','reconstruct','delta'):
                callback=torch_splice(dictionary,mode=mode,feature_edits=edits if mode=='delta' else {},token_positions=(-1,))
                logits,_=session.forward(ids,mask,edits={('resid_pre',6):callback})
                outputs[mode]=logits
                difference=logits[0,-1]-baseline[0,-1]
                top=torch.topk(difference.abs(),5).indices.tolist()
                entry['arms'][mode]={'max_abs_logit_delta':float(difference.abs().max()),'logit_delta_rms':float(difference.square().mean().sqrt()),
                    'top_token_changes':[{'token_id':i,'text':session.tokenizer.decode([i]),'logit_delta':float(difference[i])} for i in top]}
            if not torch.equal(outputs['noop'],baseline): raise ValueError('Explicit no-op changed original-model logits')
            if spec.verify_oracle:
                # Independent torch decoded-feature delta; does not call the splice helper.
                delta=sum(float(amount)*torch.from_numpy(dictionary.weights['W_dec'][index]) for index,amount in edits.items())
                delta=delta-delta.mean()
                def oracle(value):
                    changed=value.clone();changed[:,-1]+=delta.to(value);return changed
                expected,_=session.forward(ids,mask,edits={('resid_pre',6):oracle})
                oracle_error=float((expected-outputs['delta']).abs().max())
                if oracle_error>1e-4:raise ValueError('Decoded-delta splice disagrees with independent forward')
                entry['independent_delta_logit_max_abs']=oracle_error
            rows.append(entry)
    return OperationExecutionResult(payload={'scope':'real pretrained dictionary plumbing; no feature quality claim','dictionary_id':spec.dictionary.id,
        'feature_edits':[list(v) for v in spec.feature_edits],'site':'resid_pre','layer':6,'token_selection':'last prompt token','rows':rows},
        metadata={'dictionary_v1':dictionary.metadata},
        example_coverage={'materialized':True,'example_count':len(rows),'example_keys':[row['example_key'] for row in rows]})


def register():
    from pipelines_v2.operations.registry import register_operation,operation_registration
    from .codec import register_codec
    from pipelines_v2.operations.representation.sae import SAEDictionaryImportSpec,SAEEncodeSpec,SAEDecodeSpec,SAEFeatureSpliceSpec
    register_codec()
    for cls,executor in [(SAEDictionaryImportSpec,run_import),(SAEEncodeSpec,run_encode),(SAEDecodeSpec,run_decode),(SAEFeatureSpliceSpec,run_splice)]:
        if operation_registration(cls.kind) is None:register_operation(cls.kind,loader=cls.from_dict,executor=executor)
