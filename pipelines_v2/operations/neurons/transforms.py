"""Existing TransformSpec adapters: bounded native capture -> summary -> dossiers."""
from pipelines_v2.core.provenance import content_hash
from pipelines_v2.operations.common.builders import TransformResult
from .summaries import QUANTITIES, SummaryLimits, summarize_observations
from .dossiers import ExplanationCandidate, StoredExplanationProvider, build_dossiers


def summarize_capture(capture, contexts, limits=None, max_feature_bytes=33554432):
    """Load only explicitly bounded selected features; not a full-capture streamer.

    Contexts supply original, bounded prompt windows. Token indices refer to the
    captured prompt, not inferred character offsets. The streaming summary core
    is also usable directly with larger readers that yield observations.
    """
    from pipelines_v2.storage.native_features import validate_native_payload
    limits = SummaryLimits(**(limits or {}))
    manifest = capture.manifest()
    refs = manifest.storage_refs['features']
    sizes = [refs[q].get('bytes') for q in QUANTITIES]
    if type(max_feature_bytes) is not int or max_feature_bytes <= 0:
        raise ValueError('feature load bound must be a positive byte budget')
    if any(type(size) is not int or size < 0 for size in sizes) or sum(sizes) > max_feature_bytes:
        raise ValueError('selected feature load budget exceeded or size missing')
    payloads = {q: capture.load_feature(q) for q in QUANTITIES}
    for payload in payloads.values():
        validate_native_payload(payload)
    first = payloads[QUANTITIES[0]]
    coord = first['coordinate']
    normalized = lambda c: {k: v for k, v in c.items() if k not in ('name', 'quantity')}
    for q, payload in payloads.items():
        if payload['coordinate']['quantity'] != q or normalized(payload['coordinate']) != normalized(coord):
            raise ValueError('native feature identities/selections differ')
        if set(payload['layers']) != set(first['layers']):
            raise ValueError('native feature layer mismatch')
    layer = str(coord['layer'])
    rows = first['layers'][layer]
    if set(contexts) != set(rows):
        raise ValueError('original contexts must match capture examples exactly')
    for payload in payloads.values():
        if set(payload['layers'][layer]) != set(rows):
            raise ValueError('native example coverage mismatch')
        for key, row in payload['layers'][layer].items():
            if any(row[field] != rows[key][field] for field in ('tokens', 'token_count', 'units')):
                raise ValueError('native token/unit coordinates differ')
    document_splits = {}
    for context in contexts.values():
        document = context['document_id']
        if not isinstance(document, str) or not document:
            raise ValueError('original document reference required')
        if document in document_splits and document_splits[document] != context['split']:
            raise ValueError('document appears in both discovery and heldout')
        document_splits[document] = context['split']
    corpus_hash = content_hash(contexts)
    def observations():
        for key in sorted(rows):
            row, context = rows[key], contexts[key]
            for token_index, token in enumerate(row['tokens']):
                for unit_index, unit in enumerate(row['units']):
                    yield {'neuron_id': f'{coord["model_identity"]}:{coord["adapter_version"]}:{layer}:{unit}',
                        'context_id': f'{corpus_hash}:{key}:{token}', 'text': context['text'], 'split': context['split'],
                        'source': {'capture_id': capture.id, 'corpus_hash': corpus_hash, 'document_id': context['document_id'],
                            'example_key': key, 'layer': int(layer), 'neuron': unit, 'model_identity': coord['model_identity'],
                            'adapter_version': coord['adapter_version']},
                        'token_window': {'start': 0, 'stop': row['token_count'], 'selected_position': token},
                        'values': {q: float(p['layers'][layer][key]['values'][token_index, unit_index]) for q, p in payloads.items()}}
    result = summarize_observations(observations(), limits)
    result['source_capture_id'] = capture.id
    result['corpus_hash'] = corpus_hash
    result['selected_features'] = list(QUANTITIES)
    return TransformResult(payload=result, example_keys=tuple(sorted(rows)))


def dossier_transform(summary, baseline, identity, edit, manual_candidates, edited_neuron=0):
    import math
    payload = summary.result()
    arms = {name: artifact.result()['rows'] for name, artifact in (('baseline', baseline), ('identity', identity), ('edit', edit))}
    if any(len(rows) != 1 or rows[0].get('status', 'ok') != 'ok' for rows in arms.values()):
        raise ValueError('bounded dossier intervention requires one successful row per arm')
    before, same, after = (arms[name][0] for name in ('baseline', 'identity', 'edit'))
    if len({row['example_key'] for row in (before, same, after)}) != 1:
        raise ValueError('intervention arms must align')
    if before['target_logprobs'] != same['target_logprobs'] or before['generated_token_ids'] != same['generated_token_ids']:
        raise ValueError('identity control failed')
    if set(before['target_logprobs']) != set(after['target_logprobs']):
        raise ValueError('target-logprob coordinates differ')
    deltas = {token: after['target_logprobs'][token] - value for token, value in before['target_logprobs'].items()}
    if not deltas or any(not math.isfinite(value) for value in deltas.values()):
        raise ValueError('finite target-logprob deltas required')
    candidates, interventions = [], {}
    for neuron in payload['neurons']:
        unit = neuron['random'][0]['source']['neuron']
        description = manual_candidates.get(str(unit))
        if description:
            candidates.append(ExplanationCandidate(neuron['neuron_id'], description, 'stored-manual-fixture', '1',
                tuple(row['context_id'] for row in neuron['discovery'])))
        if unit == edited_neuron:
            interventions[neuron['neuron_id']] = {'status': 'engineering_checked', 'example_key': before['example_key'],
                'operator': 'scale gate_preactivation by zero at last prompt token', 'identity_exact': True,
                'target_logprob_deltas': deltas, 'baseline_text': before['generated_text'], 'edited_text': after['generated_text'],
                'evidence': {name: artifact.id for name, artifact in (('baseline', baseline), ('identity', identity), ('edit', edit))},
                'interpretation': 'plumbing check only; no neuron function or task benefit inferred'}
    if len(interventions) != 1:
        raise ValueError('edited neuron must appear exactly once')
    result = build_dossiers(payload, StoredExplanationProvider(candidates), interventions=interventions)
    result['summary'].update(identity_exact=True, intervention_count=1)
    result['evidence'] = {'summary_id': summary.id, 'capture_id': payload['source_capture_id'], 'corpus_hash': payload['corpus_hash']}
    result['limitations'] = ['Four authored prompts, three fixed layer-0 neurons; no representative sampling.',
        'Manual candidate is a stored fixture, not a validated explanation; no judge or human evaluation.',
        'No predictive, shuffled-explanation, matched-control, bootstrap, 32B or paper-replication study.']
    return TransformResult(payload=result, example_keys=(before['example_key'],))
