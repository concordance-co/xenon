"""Bounded study-specific analysis; no model calls and no scientific promotion."""
from pipelines_v2.operations.common.builders import TransformResult
from examples.local_research.phase_00.fixtures import PAIRS, TARGETS, TOKENS


def contrast(row):
    return row['target_logprobs'][str(TOKENS[0])] - row['target_logprobs'][str(TOKENS[1])]


def summarize(**inputs):
    payloads = {name: artifact.result() for name, artifact in inputs.items()}
    rows = {name: {row['example_key']: row for row in payload['rows']} for name, payload in payloads.items()}
    baseline = rows['baseline']
    if len(baseline) != 8 or set(rows['identity']) != set(baseline) or set(rows['zero']) != set(baseline):
        raise ValueError('Expected eight aligned baseline/identity/zero rows')
    records, noop = [], True
    for key, row in baseline.items():
        country = row['example']['labels']['country']
        sign = 1 if country == 'France' else -1
        before = contrast(row)
        same = row['target_logprobs'] == rows['identity'][key]['target_logprobs'] and row['generated_token_ids'] == rows['identity'][key]['generated_token_ids']
        noop &= same
        for arm in ('baseline', 'identity', 'zero', *PAIRS):
            if key not in rows[arm]:
                continue
            after = rows[arm][key]
            donor = PAIRS.get(arm, {}).get(key)
            if donor is not None and after['donor_example_key'] != donor:
                raise ValueError('Resolved donor differs from frozen pair mapping')
            text = after['generated_text'].strip()
            records.append({'example_key': key, 'arm': arm, 'donor': donor,
                'paris_minus_london': contrast(after), 'delta': contrast(after) - before,
                'recipient_correct_margin': sign * contrast(after),
                'toward_opposite_label': -sign * (contrast(after) - before),
                'opposite_label_flip': sign * before > 0 and sign * contrast(after) < 0,
                'reverse_flip': sign * before < 0 and sign * contrast(after) > 0,
                'output_not_city_first': not (text.startswith('Paris') or text.startswith('London')),
                'generated_text': after['generated_text'], 'baseline_correct': sign * before > 0})
    for arm in PAIRS:
        if set(rows[arm]) != set(TARGETS):
            raise ValueError('Donor arms must cover exactly the same four recipients')
    if not noop:
        raise ValueError('Identity control differs from baseline')
    aggregates = {}
    for arm in rows:
        selected = [r for r in records if r['arm'] == arm and r['example_key'] in TARGETS]
        aggregates[arm] = {'n': len(selected), 'mean_opposite_label_shift': sum(r['toward_opposite_label'] for r in selected) / len(selected),
            'opposite_label_flips': sum(r['opposite_label_flip'] for r in selected),
            'reverse_flips': sum(r['reverse_flip'] for r in selected),
            'non_city_first_outputs': sum(r['output_not_city_first'] for r in selected)}
    return TransformResult(payload={'kind': 'local_research_summary', 'summary': {
        'example_count': 8, 'recipient_count': 4, 'identity_exact': noop,
        'baseline_correct_token_contrast': sum(r['baseline_correct'] for r in records if r['arm'] == 'baseline'),
        'scientific_verdict': 'inconclusive', 'evidence_level': 'engineering', 'arms': aggregates},
        'limitations': ['Eight prompt texts use four templates and two country labels; these are correlated examples, not independent statistical replicates.',
            'No held-out prompts, localization sweep, statistical uncertainty or paper reproduction.',
            'Block 5 last-prompt residual is an engineering fixture, not an identified factual mechanism.',
            'Country names and answer tokens are lexical confounds; full-state replacement changes multiple attributes.',
            'Seeded random donor may share a label or template; exact pairs are reported.',
            'Non-city-first continuation is a task-format diagnostic, not a general grammar evaluator.'],
        'pairs': PAIRS, 'rows': records}, example_keys=tuple(baseline))
