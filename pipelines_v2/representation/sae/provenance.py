"""Observed dictionary/model bytes and explicit inference policies."""
from importlib.metadata import version
from pipelines_v2.core.provenance_observation import OperationFacts, file_set_identity


def provenance_facts(spec, runner_identity):
    fields={};models=None
    if spec.kind=='sae_dictionary_import':
        from .dictionary import WEIGHTS_SHA256,CONFIG_SHA256,REVISION,REPOSITORY,MODEL_REVISION,verify_original_model
        verify_original_model(spec.model_path)
        fields['cache_root']=file_set_identity(spec.cache_root,('cfg.json','sae_weights.safetensors'),expected={'cfg.json':CONFIG_SHA256,'sae_weights.safetensors':WEIGHTS_SHA256})
        from pathlib import Path
        names=tuple(p.name for p in Path(spec.model_path).iterdir() if p.name in {'config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json'})
        fields['model_path']=file_set_identity(spec.model_path,names)
        from pipelines_v2.runtime.readiness import local_checkpoint_components
        components,_=local_checkpoint_components(Path(spec.model_path),requested_revision=MODEL_REVISION)
        models=tuple(components)+({'role':'sae_dictionary','repository_id':REPOSITORY,
            'requested_revision':REVISION,'resolved_revision':REVISION,
            'file_set_digest':fields['cache_root']['content_digest']},)
    if spec.kind=='sae_feature_splice':
        from pipelines_v2.runtime.numerics import model_numerical_policy
        numerics,seeds=model_numerical_policy(spec)
        if numerics is not None:
            numerics={**numerics,'dictionary_policy':'fp32-centered-relu-torch-v2','fidelity':{'numpy':version('numpy'),'dtype':'float32'}}
    else:
        numerics={'dtype':'float32','device':'cpu','attention_implementation':'not_applicable','compile':False,
                  'determinism':'fixed-weight-inference-no-rng','matmul':{'numpy':version('numpy')},
                  'policy_version':'sae-centered-relu-numpy-v1'};seeds={}
    return OperationFacts(field_identities=fields,model_components=models,numerics=numerics,seeds=seeds,
                          methods={'dictionary_coordinate_policy':'original-gpt2-centered-layer6-v1'})
