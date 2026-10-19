"""Bounded pack contracts. The separately recorded pretrained gate is mandatory."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.local_research.phase_00.fixtures import PAIRS, PROMPTS, TARGETS, DONORS, TOKENS, SEED, examples
from examples.local_research.phase_00.workflow import build_workflow
from examples.local_research.phase_00.analysis import summarize


def fixture_results():
    rows = []
    for example in examples():
        correct = example.labels['country'] == 'France'
        rows.append({'example_key': example.key, 'example': example.to_dict(),
                     'target_logprobs': {str(TOKENS[0]): -1. if correct else -3., str(TOKENS[1]): -3. if correct else -1.},
                     'generated_token_ids': [TOKENS[0] if correct else TOKENS[1]],
                     'generated_text': ' Paris' if correct else ' London'})
    arms = {arm: deepcopy(rows) for arm in ('baseline', 'identity', 'zero')}
    for arm, mapping in PAIRS.items():
        arms[arm] = [{**deepcopy(row), 'donor_example_key': mapping[row['example_key']]} for row in rows if row['example_key'] in TARGETS]
    return arms


def artifacts(arms):
    return {arm: SimpleNamespace(result=lambda rows=rows: {'rows': rows}) for arm, rows in arms.items()}


def test_pack_locks_eight_prompts_and_explicit_pairing():
    workflow = build_workflow()
    assert len(PROMPTS) == len({p[0] for p in PROMPTS}) == len({p[1] for p in PROMPTS}) == 8
    assert {step.name for step in workflow.steps} >= {'capture', 'bank', 'baseline', 'identity', 'zero', *PAIRS, 'summary', 'report'}
    for arm, mapping in PAIRS.items():
        assert set(mapping) == set(TARGETS)
        assert set(mapping.values()) == set(DONORS)
        spec = next(s.spec for s in workflow.steps if s.name == arm)
        from pipelines_v2.operations.interventions.runtime import resolve_patched_generation_cases
        cases, skipped = resolve_patched_generation_cases(spec)
        assert not skipped
        assert {case['target'].key: case['donor'].key for case in cases} == mapping
    prompts = {e.key:e.prompt.replace('France','COUNTRY').replace('England','COUNTRY') for e in examples()}
    assert all(prompts[t] == prompts[d] for t,d in PAIRS['matched'].items())
    labels = {e.key:e.labels['country'] for e in examples()}
    assert all(labels[t] != labels[d] for t,d in PAIRS['matched'].items())
    assert all(labels[t] == labels[d] for t,d in PAIRS['same_label'].items())
    import random
    expected=list(DONORS);random.Random(SEED).shuffle(expected)
    assert list(PAIRS['random_donor'].values()) == expected


def test_analysis_counts_denominators_and_keeps_null_honest():
    payload = summarize(**artifacts(fixture_results())).payload
    assert payload['summary']['identity_exact']
    assert payload['summary']['baseline_correct_token_contrast'] == 8
    assert payload['summary']['scientific_verdict'] == 'inconclusive'
    assert all(a['n'] == 4 and a['mean_opposite_label_shift'] == 0 for a in payload['summary']['arms'].values())
    assert len(payload['rows']) == 36


def test_identity_failure_is_engineering_failure():
    arms = fixture_results();arms['identity'][0]['generated_token_ids'] = [9]
    with pytest.raises(ValueError,match='Identity control'):
        summarize(**artifacts(arms))


def test_wrong_donor_and_incomplete_control_cannot_silently_pass():
    arms=fixture_results();arms['matched'][0]['donor_example_key']='wrong'
    with pytest.raises(ValueError,match='Resolved donor'):
        summarize(**artifacts(arms))
    arms=fixture_results();arms['same_label'].pop()
    with pytest.raises(ValueError,match='same four recipients'):
        summarize(**artifacts(arms))


def test_directional_flip_and_format_diagnostics():
    arms=fixture_results()
    row=arms['matched'][0]
    row['target_logprobs']={str(TOKENS[0]):-4.,str(TOKENS[1]):-1.}
    row['generated_text']=' the city is'
    summary=summarize(**artifacts(arms)).payload['summary']['arms']['matched']
    assert summary['mean_opposite_label_shift']==1.25
    assert summary['opposite_label_flips']==1
    assert summary['non_city_first_outputs']==1


def test_preparation_requires_pinned_bytes(tmp_path):
    from examples.local_research.phase_00.prepare import prepare
    (tmp_path/'README.md').write_text('not the pinned checkpoint')
    with pytest.raises(ValueError,match='Incomplete prepared snapshot'):
        prepare(tmp_path)


def test_offline_fixture_token_bounds_when_explicitly_prepared():
    import os
    cache=os.environ.get('XENON_TEST_GPT2_CACHE')
    if not cache:
        pytest.skip('Set XENON_TEST_GPT2_CACHE for the separately prepared pretrained fixture')
    from pipelines_v2.engine.torch import TorchEngine
    from examples.local_research.phase_00.fixtures import REVISION
    engine=TorchEngine(model_id=str(Path(cache)),revision=REVISION)
    assert max(engine.scratch_token_counts(examples())) <= 128
