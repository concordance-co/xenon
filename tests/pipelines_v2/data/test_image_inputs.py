from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipelines_v2.core.provenance import example_content_hash
from pipelines_v2.core.types import to_primitive
from pipelines_v2.data.datasets import Dataset, Example, prompt_hash
from pipelines_v2.data.images import (
    DiffusionCoordinates, ImageInput, SpatialMapping, SpatialSelector,
    execution_input_identity, execution_shard_index, prepare_image,
)


def prepared(tmp_path, name='a.png', value=0, preprocessing=None):
    path = tmp_path / name
    Image.fromarray(np.full((3, 4, 3), value, dtype=np.uint8)).save(path)
    return prepare_image(path, cache_root=tmp_path/'cache', preprocessing=preprocessing or {'op':'identity'})


def test_images_identity_is_content_not_path_or_display_name(tmp_path):
    image = prepared(tmp_path)
    same = prepared(tmp_path, 'copy.png')
    changed = prepared(tmp_path, 'b.png', 255)
    different_recipe = replace(image, preprocessing_digest='b'*64)
    def ex(images):
        return Example('one', 'look', images=images)
    a, b = ex((image,)), ex((replace(same, image_id='another'),))
    assert execution_input_identity(a) == execution_input_identity(b)
    assert example_content_hash(a) == example_content_hash(b)
    assert Dataset.from_examples([a]).id == Dataset.from_examples([b]).id
    assert a.semantic_dict() == b.semantic_dict()
    for other in (ex((changed,)), ex((different_recipe,)), ex((image, changed)), ex((changed, image))):
        assert execution_input_identity(a) != execution_input_identity(other)
        assert example_content_hash(a) != example_content_hash(other)
    assert execution_input_identity(ex((image,changed))) != execution_input_identity(ex((changed,image)))
    assert execution_input_identity(replace(a,prompt='new text')) != execution_input_identity(a)
    assert [execution_shard_index(a,n) for n in (1,2,11,101)] == [execution_shard_index(b,n) for n in (1,2,11,101)]
    assert len({execution_shard_index(ex((prepared(tmp_path, f'{n}.png', n),)),97) for n in range(16)}) > 1


def test_prepared_file_verified_locally_and_on_fake_mount(tmp_path, monkeypatch):
    import socket
    monkeypatch.setattr(socket, 'create_connection', lambda *a,**k: pytest.fail('network forbidden'))
    image = prepared(tmp_path)
    raw = image.verify_at_execution(roots=(tmp_path,))
    mounted = replace(image, store_ref={**image.store_ref, 'store':'modal_volume','name':'images'})
    assert mounted.verify_at_execution(roots=(tmp_path,)) == raw
    with pytest.raises(ValueError, match='outside'):
        image.verify_at_execution(roots=(tmp_path/'other',))
    Path(image.store_ref['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='content changed'):
        image.verify_at_execution(roots=(tmp_path,))


def test_preparation_bounds_and_no_implicit_network(tmp_path):
    image = prepared(tmp_path)
    path = image.store_ref['path']
    with pytest.raises(ValueError, match='pixel budget'):
        prepare_image(path, cache_root=tmp_path/'c', preprocessing={'op':'identity'}, max_pixels=1)
    with pytest.raises(ValueError, match='byte budget'):
        prepare_image(path, cache_root=tmp_path/'c', preprocessing={'op':'identity'}, max_bytes=1)
    with pytest.raises(ValueError, match='URLs'):
        prepare_image('https://example.com/image.png', cache_root=tmp_path/'c', preprocessing={'op':'identity'})
    with pytest.raises(ValueError, match='preprocessing'):
        prepare_image(path, cache_root=tmp_path/'c', preprocessing={})
    with pytest.raises(TypeError):
        image.store_ref['path'] = 'changed'


def test_wire_roundtrip_and_legacy_hashes(tmp_path):
    text = Example('one', 'hello')
    assert text.to_dict() == {'key':'one','prompt':'hello','prompt_hash':prompt_hash('hello'),
                            'labels':{},'metadata':{},'cases':{},'case_key':None}
    assert 'images' not in to_primitive(text)
    for count in (1,2,7):
        assert execution_shard_index(text,count) == int(text.prompt_hash[:16],16)%count
    image = prepared(tmp_path)
    example = Example('one','look', images=(image,))
    dataset = Dataset.from_examples([example])
    for payload in (dataset.to_dict(), to_primitive(dataset)):
        restored = Dataset.from_dict(json.loads(json.dumps(payload)))
        assert restored.examples[0].images[0].semantic_dict() == image.semantic_dict()
        assert execution_input_identity(restored.examples[0]) == execution_input_identity(example)
    assert dataset.coverage()['execution_input_identities']['one'] == execution_input_identity(example)
    for version in (True, 2, '1', None):
        with pytest.raises(ValueError, match='schema_version'):
            ImageInput.from_dict({**image.to_dict(), 'schema_version':version})


def test_spatial_mapping_pixel_centers_resize_crop_and_mask():
    source = SpatialSelector('input',(2,3))
    target = SpatialSelector('preprocessed',(4,4))
    # Resize x2 then crop one column on each side.
    mapping = SpatialMapping(source,target,(2,2),(0,-1))
    np.testing.assert_array_equal(mapping.map_points([[0,0],[1,2]]),[[.5,-.5],[2.5,3.5]])
    mask = np.array([[0,1,0],[0,0,1]],dtype=float)
    np.testing.assert_array_equal(mapping.map_mask(mask),
        [[0,1,1,0],[0,1,1,0],[0,0,0,1],[0,0,0,1]])
    assert SpatialMapping.from_dict(mapping.to_dict()) == mapping
    assert SpatialMapping.from_dict(to_primitive(mapping)) == mapping
    with pytest.raises(ValueError,match='shape'):
        source.validate_mask(np.ones((3,2)))
    with pytest.raises(ValueError,match='finite'):
        source.validate_mask(np.full((2,3),np.nan))
    with pytest.raises(ValueError,match='coordinate'):
        SpatialSelector('token',(2,3))
    bilinear = SpatialMapping(source,replace(target,interpolation='bilinear'),(2,2),(0,-1))
    assert np.any((bilinear.map_mask(mask)>0)&(bilinear.map_mask(mask)<1))


def test_diffusion_step_is_not_scheduler_timestep():
    args = dict(prompt='a',negative_prompt='',initial_noise_digest='a'*64,
        initial_noise_ref={'store':'local','path':'/noise.safetensors','bytes':100},
        generator_state_ref={'store':'local','path':'/rng.safetensors','bytes':100},
        scheduler_name='fixture',scheduler_config_digest='b'*64,step_index=2,
        timestep_value=900.5,denoising_window=(1,5),cfg_enabled=True,branch='both',guidance_scale=7.5)
    value = DiffusionCoordinates(**args)
    assert DiffusionCoordinates.from_dict(value.to_dict()) == value
    assert DiffusionCoordinates.from_dict(to_primitive(value)) == value
    assert value.step_index == 2 and value.timestep_value == 900.5
    with pytest.raises(ValueError,match='CFG-disabled'):
        DiffusionCoordinates(**{**args,'cfg_enabled':False})
    with pytest.raises(ValueError,match='window'):
        DiffusionCoordinates(**{**args,'step_index':5})


def test_deferred_image_records_and_postgres_projection(tmp_path, monkeypatch):
    import sys
    import types
    from pipelines_v2.data.sources import InMemorySource, PostgresSource
    image = prepared(tmp_path)
    records = [{'example_id':'one','prompt':'look','images':[image.to_dict()]}]
    deferred = Dataset.from_source(source=InMemorySource(records=tuple(records)), defer=True,
                                  images_column='images')
    assert deferred.resolve().examples[0].images[0].content_digest == image.content_digest
    queries = []
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, query, params): queries.append(query); return self
        def fetchall(self): return records
    fake = types.ModuleType('psycopg'); fake.connect = lambda *a, **k: Connection()
    rows = types.ModuleType('psycopg.rows'); rows.dict_row = object()
    monkeypatch.setitem(sys.modules, 'psycopg', fake)
    monkeypatch.setitem(sys.modules, 'psycopg.rows', rows)
    source = PostgresSource(url='postgresql://fixture/db')
    kwargs = dict(table='images',prompt_column='prompt',example_key_column='example_id',images_column='images')
    actual = source.fetch_dataset(**kwargs)
    assert actual.examples[0].images[0].content_digest == image.content_digest
    assert '"images"' in queries[0].split('FROM')[0]
    assert actual.examples[0].metadata == {}
    batches = list(source.iter_dataset_batches(batch_size=2,**kwargs))
    assert batches[0].examples[0].images[0].content_digest == image.content_digest
    with pytest.raises(ValueError,match='resolved execution identity'):
        source.fetch_dataset(**kwargs,prompt_hash_column='prompt_hash',execution_shard={'count':2,'index':0})


def test_image_column_null_and_actionable_errors():
    assert Dataset.from_records([{'example_id':'row-null','prompt':'text','images':None}], images_column='images').examples[0].images == ()
    for row in ({'example_id':'missing','prompt':'text'},
                {'example_id':'bad-string','prompt':'text','images':'[]'},
                {'example_id':'bad-record','prompt':'text','images':[None]}):
        with pytest.raises(ValueError,match=row['example_id']):
            Dataset.from_records([row],images_column='images')


def test_pillow_missing_is_actionable(monkeypatch):
    import builtins
    from pipelines_v2.data.images import _decode_description
    original = builtins.__import__
    def no_pillow(name,*args,**kwargs):
        if name == 'PIL': raise ImportError('fixture missing optional dependency')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',no_pillow)
    with pytest.raises(RuntimeError,match='uv sync --extra vision'):
        _decode_description(b'not decoded',100)


def test_cfg_scale_changes_coordinates_and_references_are_explicit():
    args = dict(prompt='a',negative_prompt='',initial_noise_digest='a'*64,
        initial_noise_ref={'store':'local','path':'/noise.safetensors','bytes':100},
        generator_state_ref={'store':'local','path':'/rng.safetensors','bytes':100},
        scheduler_name='fixture',scheduler_config_digest='b'*64,step_index=0,
        timestep_value=999,denoising_window=(0,20),cfg_enabled=True,branch='both',guidance_scale=7.5)
    first=DiffusionCoordinates(**args)
    assert first.to_dict()!=replace(first,guidance_scale=5).to_dict()
    for invalid in (True,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='guidance_scale'):
            replace(first,guidance_scale=invalid)
    with pytest.raises(ValueError,match='absolute'):
        replace(first,initial_noise_ref={'store':'local','path':'relative','bytes':100})
    with pytest.raises(ValueError,match='bytes'):
        replace(first,generator_state_ref={'store':'local','path':'/rng'})


def test_remote_two_shard_image_capture_routing_and_text_parity(tmp_path):
    from pipelines_v2.api import CaptureSpec, ToyEngine, ResidualSite
    from pipelines_v2.runtime.remote_executor import _apply_execution_shard, _prompt_hash_shard
    # A fake dispatcher exercises the exact capture partitioner before a model runs.
    images = [prepared(tmp_path, f'route-{i}.png', i) for i in range(16)]
    examples = [Example(str(i), 'identical prompt', images=(image,)) for i,image in enumerate(images)]
    spec = CaptureSpec(engine=ToyEngine(), dataset=Dataset.from_examples(examples), sites=(ResidualSite(name='resid',site='resid_post',layers=(0,)),))
    received = []
    for index in range(2):
        routed = _apply_execution_shard(spec=spec, workflow_context={'execution_shard':{'index':index,'count':2}})
        received.append([example.key for example in routed.dataset.examples])
        assert all(execution_shard_index(example,2) == index for example in routed.dataset.examples)
    assert all(received)  # Identical prompts no longer force every image onto one worker.
    assert sorted(received[0]+received[1]) == sorted(example.key for example in examples)
    assert not set(received[0]) & set(received[1])
    text = [Example(str(i), str(i)) for i in range(20)] + [Example('legacy', 'x', prompt_hash='legacy-nonhex')]
    spec = replace(spec, dataset=Dataset.from_examples(text))
    for count in (1,2,7):
        for index in range(count):
            routed = _apply_execution_shard(spec=spec, workflow_context={'execution_shard':{'index':index,'count':count}})
            assert list(routed.dataset.examples) == [ex for ex in text if _prompt_hash_shard(ex.prompt_hash,count)==index]


@pytest.mark.parametrize('images_column', [None, 'images'])
def test_remote_deferred_sql_pushdown_respects_images(tmp_path, monkeypatch, images_column):
    from pipelines_v2.api import CaptureSpec, ToyEngine, ResidualSite
    from pipelines_v2.data.sources import PostgresSource
    from pipelines_v2.runtime.remote_executor import _resolve_dataset_for_execution, _iter_streamed_dataset_batches
    import pipelines_v2.data.sources as sources
    image = prepared(tmp_path)
    resolved = Dataset.from_examples([Example('one','look',images=(image,) if images_column else ())])
    calls = []
    class FakeSource:
        def fetch_dataset(self, **kwargs):
            calls.append(kwargs)
            return resolved
        def iter_dataset_batches(self, **kwargs):
            calls.append(kwargs)
            yield resolved
    monkeypatch.setattr(sources, 'source_from_dict', lambda payload: FakeSource())
    kwargs = dict(table='fixture',prompt_column='prompt',example_key_column='example_id',prompt_hash_column='prompt_hash')
    if images_column:
        kwargs['images_column'] = images_column
    dataset = Dataset.from_postgres(source=PostgresSource.from_env('XENON_FIXTURE_DATABASE_URL'), **kwargs)
    spec = CaptureSpec(engine=ToyEngine(),dataset=dataset, sites=(ResidualSite(name='resid',site='resid_post',layers=(0,)),))
    context = {'execution_shard':{'index':1,'count':2}}
    assert _resolve_dataset_for_execution(spec=spec,workflow_context=context).dataset.examples == resolved.examples
    assert list(_iter_streamed_dataset_batches(dataset=dataset,workflow_context=context,batch_size=2))[0].examples == resolved.examples
    assert len(calls) == 2
    for call in calls:
        assert ('execution_shard' in call) is (images_column is None)
        if images_column:
            assert call['images_column'] == images_column
