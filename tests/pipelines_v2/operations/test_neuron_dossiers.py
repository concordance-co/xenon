import pytest
from pipelines_v2.operations.neurons import (summarize_observations, ExplanationCandidate,
    StoredExplanationProvider, build_dossiers)
from tests.pipelines_v2.operations.test_neuron_summaries import observation


def summary():
    return summarize_observations([observation(i) for i in range(4)])


def test_provider_never_sees_heldout_and_unexplained_stays_in_denominator():
    class Spy:
        def explain(self, neuron_id, contexts):
            assert contexts and all(c['split'] == 'discovery' for c in contexts)
            assert all(set(c['values']) == {'mlp_product'} for c in contexts)
            return None
    result = build_dossiers(summary(), Spy(), view='product_only')
    assert result['summary']['neuron_count'] == result['summary']['unexplained_count'] == 1
    assert result['dossiers'][0]['scientific_verdict'] == 'not_tested'


def test_manual_candidate_links_only_visible_discovery_contexts():
    s = summary()
    n = s['neurons'][0]
    candidate = ExplanationCandidate(n['neuron_id'], 'A fixture, not a scientific explanation.', 'manual', 'v1',
        tuple(row['context_id'] for row in n['discovery']))
    result = build_dossiers(s, StoredExplanationProvider([candidate]))
    assert result['summary']['unexplained_count'] == 0
    assert result['dossiers'][0]['explanation_evaluation'] == 'not_tested'


@pytest.mark.parametrize('ids,view', [(('1',), 'signed'), (('not-seen',), 'signed'), ((), 'signed'), (('0',), 'product_only')])
def test_heldout_unseen_or_wrong_view_candidate_rejected(ids, view):
    s = summary()
    c = ExplanationCandidate(s['neurons'][0]['neuron_id'], 'fixture', 'manual', 'v1', ids, view=view)
    with pytest.raises(ValueError):
        build_dossiers(s, StoredExplanationProvider([c]))


def test_dossier_transform_envelope_and_single_edited_neuron():
    from types import SimpleNamespace
    from pipelines_v2.operations.neurons.transforms import summarize_capture, dossier_transform
    from pipelines_v2.operations.common.builders import TransformResult
    from tests.pipelines_v2.operations.test_neuron_summaries import fake_capture
    capture, _, _ = fake_capture()
    summarized = summarize_capture(capture, {'a': {'text': 'a', 'split': 'discovery', 'document_id': 'a'},
        'b': {'text': 'b', 'split': 'heldout', 'document_id': 'b'}})
    def artifact(name, payload):
        return SimpleNamespace(id=name, result=lambda: payload)
    before = {'example_key': 'a', 'target_logprobs': {'10': -2.}, 'generated_token_ids': [10], 'generated_text': 'a'}
    after = dict(before, status='ok', target_logprobs={'10': -3.})
    args = dict(summary=artifact('summary', summarized.payload), baseline=artifact('baseline', {'rows': [before]}),
        identity=artifact('identity', {'rows': [dict(before, status='ok')]}), edit=artifact('edit', {'rows': [after]}),
        manual_candidates={'0': 'unverified fixture'})
    result = dossier_transform(**args)
    assert isinstance(result, TransformResult)
    assert result.payload['kind'] == 'neuron_dossier_result'
    assert result.payload['dossiers'][0]['intervention']['target_logprob_deltas'] == {'10': -1.}
    args['identity'] = artifact('identity', {'rows': [after]})
    with pytest.raises(ValueError, match='identity control'):
        dossier_transform(**args)
