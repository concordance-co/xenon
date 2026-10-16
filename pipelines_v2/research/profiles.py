"""Resource declarations, never automatic memory-fit or hardware qualifications."""

def llm_modal_profile(name='small', *, model_volume='xenon-models', **overrides):
    from pipelines_v2.runtime.modal import ModalResources,ModalVolumeMount
    profiles={'small':{'gpu':'L4','cpu':4,'memory_mb':16384,'timeout_seconds':900},
              'large':{'gpu':'A100-80GB','cpu':8,'memory_mb':65536,'timeout_seconds':1200}}
    if name not in profiles:raise ValueError('Use small or large profile')
    values={**profiles[name],'max_containers':1,'shard_count':1,'readiness_timeout_seconds':300,
            'volumes':(ModalVolumeMount(model_volume,'/models'),),**overrides}
    return ModalResources(**values)
