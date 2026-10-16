import copy
import random
import pytest
from pipelines_v2.operations.neurons import QUANTITIES, SummaryLimits, summarize_observations


def observation(i, value=None):
    value = float(i - 5) if value is None else value
    return {'neuron_id': 'model:adapter:0:2', 'context_id': str(i), 'text': f'original {i}',
        'split': 'discovery' if i % 2 == 0 else 'heldout', 'source': {'document_id': str(i)},
        'token_window': {'start': 0, 'stop': 3, 'selected_position': 2},
        'values': dict(zip(QUANTITIES, (value, -value, value * 2, value)))}


def test_stream_agrees_with_exhaustive_signed_values():
    rows = [observation(i) for i in range(11)]
    result = summarize_observations(iter(rows), SummaryLimits(k=2))
    n = result['neurons'][0]
    assert [r['context_id'] for r in n['top']] == ['10', '9']
    assert [r['context_id'] for r in n['bottom']] == ['0', '1']
    assert n['bottom'][0]['values']['mlp_product'] == -5
    assert n['distribution']['mlp_product']['mean'] == 0
    assert n['distribution']['gate_activation']['negative'] == 5
    assert n['distribution']['mlp_product']['quantiles']['0.5'] == 0
    assert sum(n['distribution']['mlp_product']['histogram']) == 11
    assert [r['context_id'] for r in n['weak']] == ['5']
    assert n['bottom'][0]['product_percentile'] == 1 / 11


def test_retention_is_bounded_and_batch_order_invariant():
    rows = [observation(i) for i in range(100)]
    limits = SummaryLimits(k=3, reservoir_size=5)
    a = summarize_observations(iter(rows), limits)['neurons'][0]
    random.Random(4).shuffle(rows)
    b = summarize_observations(iter(rows), limits)['neurons'][0]
    for name in ('top', 'bottom', 'random', 'weak', 'discovery'):
        assert a[name] == b[name]
        assert len(a[name]) <= (3 if name in ('top', 'bottom') else 5)
    assert a['count'] == 100
    assert a['distribution'] == b['distribution']
    assert a['distribution']['mlp_product']['quantile_sample_count'] == 5


@pytest.mark.parametrize('change', [lambda r: r.update(text='x' * 401), lambda r: r.update(split='unknown'),
    lambda r: r['values'].update(mlp_product=float('nan')), lambda r: r.pop('source')])
def test_invalid_observations_fail(change):
    row = observation(0)
    change(row)
    with pytest.raises(ValueError):
        summarize_observations([row])


def test_budgets_and_duplicates_fail():
    with pytest.raises(ValueError, match='duplicate'):
        summarize_observations([observation(0), observation(0)])
    with pytest.raises(ValueError, match='budget'):
        summarize_observations([observation(0), observation(1)], SummaryLimits(max_observations=1))
    with pytest.raises(ValueError, match='neuron budget'):
        summarize_observations([observation(0), dict(observation(1), neuron_id='second')], SummaryLimits(max_neurons=1))


def fake_capture():
    from types import SimpleNamespace
    import numpy as np
    from pipelines_v2.api import NativeSite, NativeSelection
    from pipelines_v2.operations.common.native import QWEN_VERSION
    payloads = {}
    for q in QUANTITIES:
        site = NativeSite(name=q, model_identity='fixture-model', adapter_version=QWEN_VERSION,
            module_path='model.layers.0.mlp', layer=0, quantity=q, selection=NativeSelection(neurons=(0,)),
            axes=('batch', 'token', 'neuron'))
        p = site.empty_payload()
        p['layers']['0'] = {key: {'tokens': [1], 'token_count': 2, 'units': [0], 'values': np.array([[value]], dtype=np.float32)} for key, value in (('a', -2.), ('b', 3.))}
        payloads[q] = p
    refs = {q: {'bytes': 128} for q in QUANTITIES}
    return SimpleNamespace(id='capture-fixture', manifest=lambda: SimpleNamespace(storage_refs={'features': refs}),
        load_feature=lambda q: payloads[q]), payloads, refs


def test_capture_transform_envelope_alignment_and_bounds():
    from pipelines_v2.operations.neurons.transforms import summarize_capture
    from pipelines_v2.operations.common.builders import TransformResult
    capture, payloads, refs = fake_capture()
    contexts = {'a': {'text': 'alpha beta', 'split': 'discovery', 'document_id': 'doc-a'},
        'b': {'text': 'gamma delta', 'split': 'heldout', 'document_id': 'doc-b'}}
    result = summarize_capture(capture, contexts)
    assert isinstance(result, TransformResult)
    assert result.payload['summary']['observation_count'] == 2
    assert result.payload['neurons'][0]['bottom'][0]['source']['capture_id'] == capture.id
    assert result.example_keys == ('a', 'b')
    with pytest.raises(ValueError, match='budget'):
        summarize_capture(capture, contexts, max_feature_bytes=500)
    refs[QUANTITIES[0]]['bytes'] = None
    with pytest.raises(ValueError, match='size missing'):
        summarize_capture(capture, contexts)
    refs[QUANTITIES[0]]['bytes'] = 128
    contexts['b']['document_id'] = 'doc-a'
    with pytest.raises(ValueError, match='both discovery and heldout'):
        summarize_capture(capture, contexts)
    contexts['b']['document_id'] = 'doc-b'
    payloads['mlp_product']['coordinate']['model_identity'] = 'another-model'
    with pytest.raises(ValueError, match='identities'):
        summarize_capture(capture, contexts)
