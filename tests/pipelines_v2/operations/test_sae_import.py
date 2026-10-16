import numpy as np
import pytest
from pipelines_v2.representation.sae.dictionary import SAEDictionary


def tiny_dictionary():
    return SAEDictionary({'W_enc':np.eye(3,dtype=np.float32),'W_dec':np.eye(3,dtype=np.float32),
        'b_enc':np.zeros(3,dtype=np.float32),'b_dec':np.zeros(3,dtype=np.float32)},
        {'architecture':'standard','activation':'relu','normalization':'none'})


def test_encode_decode_and_analytical_error():
    dictionary=tiny_dictionary();x=np.array([[-1,0,1],[1,0,-1]],dtype=np.float32)
    assert np.array_equal(dictionary.encode(x),np.maximum(x,0))
    assert np.array_equal(dictionary.decode(dictionary.encode(x)),np.maximum(x,0))
    metrics=dictionary.fidelity(x)
    assert metrics['mse']==pytest.approx(1/3)
    assert metrics['normalized_mse']==pytest.approx(.5)
    assert metrics['mean_l0']==1
    assert metrics['explained_variance']==pytest.approx(.5)


def test_dictionary_rejects_unsupported_conventions_and_nonfinite():
    dictionary=tiny_dictionary()
    with pytest.raises(ValueError,match='standard ReLU'):
        SAEDictionary(dictionary.weights,{**dictionary.metadata,'normalization':'layer_norm'})
    broken={k:v.copy() for k,v in dictionary.weights.items()};broken['b_enc'][0]=np.nan
    with pytest.raises(ValueError,match='finite FP32'):SAEDictionary(broken,dictionary.metadata)
    with pytest.raises(ValueError,match='dimensions'):SAEDictionary({**dictionary.weights,'b_enc':np.zeros(4,dtype=np.float32)},dictionary.metadata)


def test_sae_codec_roundtrip_and_namespaced_tensors():
    from pipelines_v2.representation.sae.execution import register
    from pipelines_v2.storage.features import encode_feature_payload,decode_feature_payload
    register();dictionary=tiny_dictionary()
    payload={'kind':'sae_tensors','weights':dictionary.weights,'metadata':dictionary.metadata}
    metadata,tensors=encode_feature_payload(payload,tensor_key_prefix='dictionary_')
    assert all(key.startswith('dictionary_') for key in tensors)
    decoded=decode_feature_payload(metadata,tensors)
    for name,value in dictionary.weights.items():assert np.array_equal(decoded['weights'][name],value)


def test_import_refuses_unknown_revision_and_changed_bytes(tmp_path):
    from pipelines_v2.operations.representation.sae import SAEDictionaryImportSpec
    with pytest.raises(ValueError,match='checkpoint'):
        SAEDictionaryImportSpec(cache_root=str(tmp_path),model_path=str(tmp_path),revision='mutable')
    (tmp_path/'cfg.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        SAEDictionary.import_pinned(tmp_path,model_checkpoint_digest='a'*64,coordinate_evidence={})


def test_coordinate_evidence_binds_original_model_and_site():
    from pipelines_v2.representation.sae.dictionary import validate_coordinate_evidence,ADAPTER_VERSION,MODEL_REVISION,WEIGHTS_SHA256,SITE
    evidence={'adapter_version':ADAPTER_VERSION,'model_checkpoint_digest':'a'*64,'model_revision':MODEL_REVISION,
        'dictionary_weights_sha256':WEIGHTS_SHA256,'site':SITE,'coordinate_map':'per_token_center','passed':True,
        'prompt_count':2,'reference_versions':{'transformer-lens':'3.2.1'},'tensor_max_abs':0.0,'encode_max_abs':0.0,'decode_max_abs':0.0}
    validate_coordinate_evidence(evidence,'a'*64)
    for bad in ({**evidence,'site':'blocks.5.hook_resid_pre'},{**evidence,'coordinate_map':'identity'},{**evidence,'tensor_max_abs':1}):
        with pytest.raises(ValueError):validate_coordinate_evidence(bad,'a'*64)
    with pytest.raises(ValueError):validate_coordinate_evidence(evidence,'b'*64)


def test_encode_selects_layer6_from_multilayer_capture(monkeypatch):
    from types import SimpleNamespace
    from pipelines_v2.representation.sae import execution
    dictionary = tiny_dictionary()
    dictionary.metadata['input_site'] = {'checkpoint_digest':'qualified-model'}
    monkeypatch.setattr(execution, 'load_dictionary', lambda artifact:dictionary)
    row = {'tokens':[3, 4], 'values':np.array([[-1,0,1],[1,0,-1]], dtype=np.float32)}
    payload = {'kind':'residual', 'site':'resid_pre', 'model_identity':'qualified-model',
               'layers':{'6':{'example':row}}}
    spec = SimpleNamespace(dictionary=SimpleNamespace(id='dictionary'),
                           feature=SimpleNamespace(load=lambda:payload))
    single = execution.run_encode(spec)
    # Invalid data at layer 4 proves the unrelated layer is never read/encoded.
    payload['layers']['4'] = {'other-example':{'values':'not a tensor'}}
    multiple = execution.run_encode(spec)
    assert multiple.example_coverage == single.example_coverage
    assert multiple.payload == single.payload
    assert list(multiple.features['codes']['records']) == ['example']
    assert np.array_equal(multiple.features['codes']['records']['example']['values'],
                          single.features['codes']['records']['example']['values'])
    for field, value in [('model_identity','wrong-model'), ('site','resid_post'), ('kind','attention')]:
        original = payload[field]
        payload[field] = value
        with pytest.raises(ValueError, match='coordinate system'):
            execution.run_encode(spec)
        payload[field] = original
    del payload['layers']['6']
    with pytest.raises(ValueError, match='layer6'):
        execution.run_encode(spec)
