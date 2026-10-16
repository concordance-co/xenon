"""SAE arrays use the shared safetensors codec; no pickle or JSON tensor lists."""

def register_codec():
    from pipelines_v2.storage.features import (register_feature_codec,
        encode_tensor_feature_payload,decode_feature_payload,_FEATURE_CODECS)
    if 'sae_tensors' not in _FEATURE_CODECS:
        register_feature_codec('sae_tensors',storage_format='sae_tensors_v1',
            encode=encode_tensor_feature_payload,decode=decode_feature_payload)
