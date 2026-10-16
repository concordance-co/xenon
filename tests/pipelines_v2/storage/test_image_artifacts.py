import json

import numpy as np
import pytest

from pipelines_v2.storage.features import (encode_feature_payload, load_feature_payload,
    merge_feature_payloads, write_capture_features)
from pipelines_v2.storage.image_features import SpatialTensor, spatial_summary
from pipelines_v2.storage.local import LocalArtifactStore


def payload(kind='image', key='one'):
    values = np.arange(12,dtype=np.float32).reshape(3,2,2)
    descriptor = SpatialTensor.from_array(values, kind=kind,axes=('channel','y','x'),
        preprocessing_digest='a'*64,origin_ref={'content_digest':'b'*64},
        **({'color_space':'RGB','value_range':(0,255)} if kind=='image' else {'component_digest':'c'*64}))
    return {'kind':kind,'schema_version':1,'records':{key:{'descriptor':descriptor.to_dict(),'values':values}}}


@pytest.mark.parametrize('kind',['image','latent'])
def test_spatial_codec_round_trip_and_safe_metadata(tmp_path,kind):
    value = payload(kind)
    metadata, tensors = encode_feature_payload(value,tensor_key_prefix='unit_')
    assert all(k.startswith('unit_') for k in tensors)
    assert '[[[' not in json.dumps(metadata)
    store = LocalArtifactStore(root=tmp_path)
    refs = write_capture_features(store,'artifact',{'a':value})
    actual = load_feature_payload(store,refs['a'])
    assert actual['records']['one']['descriptor'] == value['records']['one']['descriptor']
    np.testing.assert_array_equal(actual['records']['one']['values'],value['records']['one']['values'])
    assert actual['records']['one']['values'].dtype == np.float32
    summary = spatial_summary(actual)
    assert 'values' not in json.dumps(summary)
    with pytest.raises(ValueError,match='budget'):
        spatial_summary(actual,max_bytes=1)


@pytest.mark.parametrize('mutation', ['hash','shape','dtype','range','kind','nan','axes'])
def test_malformed_spatial_tensors_rejected(mutation):
    value=payload(); record=value['records']['one'];desc=record['descriptor']
    if mutation=='hash': desc['content_digest']='f'*64
    elif mutation=='shape': desc['shape']=[3,4,4]
    elif mutation=='dtype': desc['dtype']='float16'
    elif mutation=='range': desc['value_range']=[0,1]
    elif mutation=='kind': desc['kind']='latent'
    elif mutation=='nan': record['values'][0,0,0]=np.nan
    elif mutation=='axes': desc['axes']=['x','y','channel']
    with pytest.raises(ValueError): encode_feature_payload(value)


def test_merge_empty_and_reject_duplicate_keys():
    value=payload()
    empty={'kind':'image','schema_version':1,'records':{}}
    merged=merge_feature_payloads(empty,value)
    assert set(merged['records'])=={'one'}
    assert empty['records']=={}
    with pytest.raises(ValueError,match='duplicate'):
        merge_feature_payloads(value,value)
    merged=merge_feature_payloads(value,payload(key='two'))
    assert set(merged['records'])=={'one','two'}
    metadata,tensors=encode_feature_payload(empty)
    assert metadata==empty and tensors=={}


def test_spatial_reader_preserves_remote_transfer_policy(tmp_path, monkeypatch):
    from pipelines_v2.core.types import TransferPolicy, TransferPolicyError
    from pipelines_v2.storage.modal import ModalVolumeStore
    value = payload('latent')
    metadata, _ = encode_feature_payload(value)
    metadata_path = tmp_path/'metadata.json'
    metadata_path.write_text(json.dumps(metadata))
    store = ModalVolumeStore(name='fixture', root='/data/artifacts/t25-policy',
        local_cache_root=tmp_path/'cache', transfer_policy=TransferPolicy(max_download_bytes=32))
    monkeypatch.setattr(ModalVolumeStore, '_run_modal_volume_get',
                        lambda *a,**k: pytest.fail('must reject before download'))
    ref = {'store':'modal_volume','name':'fixture','format':'latent_safetensors_v1',
           'metadata_path':str(metadata_path),'metadata_bytes':metadata_path.stat().st_size,
           'tensor_path':'/data/artifacts/t25-policy/missing/features.safetensors','tensor_bytes':1024}
    with pytest.raises(TransferPolicyError):
        load_feature_payload(store,ref)


@pytest.mark.parametrize('kind',['image','latent'])
def test_empty_spatial_payload_persists_without_tensor_file(tmp_path,kind):
    value={'kind':kind,'schema_version':1,'records':{}}
    store=LocalArtifactStore(root=tmp_path)
    ref=write_capture_features(store,'empty',{'features':value})['features']
    assert ref['tensor_path'] is None and ref['tensor_bytes']==0
    assert load_feature_payload(store,ref)==value
