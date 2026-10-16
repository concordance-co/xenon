"""C06: strict, file-based attribution results; rankings are not edge contributions."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
GRID = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1.)
MATTR_COMMIT = 'af82b2eef145dd3e66c793a413352d6931a152d7'
MIB_COMMIT = '582fdbc8a2b0f7063a265c8f94851fd2a3aa5480'
EAP_COMMIT = '3fb341eafdaba63632954aecbbb96b4b08e13d02'
EVALUATOR = f'aryamanarora/MIB-circuit-track@{MIB_COMMIT}:MIB_circuit_track.evaluation.evaluate_area_under_curve'

MODEL_PROFILES = {
    'gpt2': {'assets':'gpt2_ioi_assets.json', 'path':'gpt2', 'dtype':'float32', 'layers':12, 'heads':12, 'width':768},
    'qwen2.5': {'assets':'qwen25_ioi_assets.json', 'path':'Qwen/Qwen2.5-0.5B', 'dtype':'bfloat16', 'layers':24, 'heads':14, 'width':896},
}


def model_profile(request):
    name = request.get('model', {}).get('name')
    if name not in MODEL_PROFILES:
        raise ValueError('Unsupported task/model identity')
    return MODEL_PROFILES[name]


def locked_assets(model='gpt2'):
    profile = model_profile({'model':{'name':model}})
    return json.loads(Path(__file__).with_name(profile['assets']).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'{label} must be finite')
    return float(value)


def metric(value, label):
    if isinstance(value, dict) and set(value) == {'value', 'undefined_reason'}:
        if value['value'] is None and isinstance(value['undefined_reason'], str) and value['undefined_reason'].strip():
            return value
        raise ValueError(f'{label}: malformed undefined metric')
    return finite(value, label)


def expected_nodes(layers=12, heads=12):
    return ['input'] + [f'a{layer}.h{head}' for layer in range(layers) for head in range(heads)] + [f'm{layer}' for layer in range(layers)]


def validate_request_identity(request):
    profile=model_profile(request)
    assets=locked_assets(request['model']['name'])
    if request.get('schema_version')!=1 or request.get('task')!='ioi':
        raise ValueError('Unsupported task/model identity')
    if any(request.get('model',{}).get(k)!=assets['model'][k] for k in ('revision','files')):
        raise ValueError('Request differs from pinned model assets')
    data=request.get('data',{})
    if set(data.get('files',{}))!={'train.json','dev.json'}:
        raise ValueError('IOI data must identify train.json and dev.json')
    import re
    if not all(isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for v in data['files'].values()):
        raise ValueError('IOI data requires content hashes')
    if data.get('repository')=='mib-bench/ioi' and any(data.get(k)!=assets['data'][k] for k in ('revision','files')):
        raise ValueError('Official IOI label requires exact pinned data; use a content-addressed custom dataset identity')
    if request.get('dtype')!=profile['dtype'] or request.get('orientation')!='iso_base_clean_source_s2_io_flip' or request.get('basis')!='attention_head_mlp_input' or request.get('include_input') is not True:
        raise ValueError('Unsupported numerical/basis/orientation coverage')
    if request.get('evaluator')!=EVALUATOR:
        raise ValueError('Wrong evaluator identity')


def normalize_official_outputs(importances, official_metrics, *, request, compute, raw_outputs):
    """Preserve official values; never load pickle in the Xenon control interpreter."""
    validate_request_identity(request)
    profile=model_profile(request)
    if request['evaluator'] != EVALUATOR:
        raise ValueError('Wrong official evaluator identity')
    cfg = importances['cfg']
    if (cfg['n_layers'], cfg['n_heads'], cfg['d_model']) != (profile['layers'], profile['heads'], profile['width']):
        raise ValueError('Expected the locked model dimensions')
    nodes = importances['nodes']
    expected = expected_nodes(profile['layers'], profile['heads'])
    if set(nodes) != set(expected) | {'logits'}:
        raise ValueError('Missing or unexpected node coverage')
    scored = [{'node_id': n, 'score': finite(nodes[n].get('score'), f'node {n}')} for n in expected]
    ranking = sorted(scored, key=lambda x: (-x['score'], expected.index(x['node_id'])))
    vectors = ['faithfulnesses', 'accuracies', 'weighted_edge_counts']
    if any(len(official_metrics[name]) != len(GRID) for name in vectors):
        raise ValueError('Official curve must cover the complete ten-point grid')
    points = []
    for i, fraction in enumerate(GRID):
        accuracy = metric(official_metrics['accuracies'][i], 'accuracy')
        if isinstance(accuracy, float) and not 0 <= accuracy <= 1:
            raise ValueError('Accuracy must be in [0, 1]')
        count = finite(official_metrics['weighted_edge_counts'][i], 'weighted edge count')
        if count < 0:
            raise ValueError('Negative weighted edge count')
        points.append({'fraction': fraction, 'requested_k': int(fraction * len(expected)),
                       'weighted_edge_count': count, 'cpr': metric(official_metrics['faithfulnesses'][i], 'CPR'),
                       'accuracy': accuracy})
    aggregates = {name: metric(official_metrics[name], name) for name in ('area_under', 'area_from_1', 'average', 'acc_auc')}
    result = {'schema_version': 1, 'kind': 'external_attribution', 'request_identity': digest(request),
              'request': request, 'nodes': scored, 'ranking': [x['node_id'] for x in ranking],
              'score_units': 'unbounded learned mask logits; descending, signed scores',
              'basis': ('GPT-2' if request['model']['name']=='gpt2' else 'Qwen2.5-0.5B')+' attention-head outputs before output projection and MLP outputs; input embedding included',
              'masking_convention': 'iso: top-k clean/base, complement counterfactual/source; patching',
              'points': points, 'aggregates': aggregates,
              'metric_definitions': {'cpr': '(ablated mean logit difference - corrupted reference)/(clean reference - corrupted reference)',
                 'accuracy': 'fraction of examples with positive base-answer minus source-answer logit difference',
                 'area_under': 'official linear-sparsity trapezoidal CPR integral on [0.001,1]',
                 'area_from_1': 'official linear-sparsity trapezoidal absolute(1-CPR) integral',
                 'acc_auc': 'official normalized log-sparsity trapezoidal accuracy integral',
                 'weighted_edge_count': 'retained graph edge count after node pruning; not attribution'},
              'evaluator': request['evaluator'], 'raw_outputs': raw_outputs, 'effective_compute': compute,
              'uncertainty': {'scope': 'single method/seed', 'seed': request['seed'], 'interval': None},
              'resume_semantics': 'restart_unfinished_method_seed; no optimizer/RNG continuation',
              'reproduction_label': 'integration_smoke', 'scientific_verdict': 'not_tested',
              'limitations': ['Reduced fitting and validation subset; no reference reproduction claim',
                             'No random/same-label controls or held-out confirmatory campaign',
                             'Node ranking is not an edge contribution; raw upstream edge fields are not promoted']}
    validate_result(result)
    return result


def validate_result(value: dict[str, Any]):
    if value.get('schema_version') != 1 or value.get('kind') != 'external_attribution':
        raise ValueError('Unsupported C06 result version/kind')
    request = value['request']
    validate_request_identity(request)
    profile=model_profile(request)
    if value['request_identity'] != digest(request) or value['evaluator'] != EVALUATOR or request['evaluator'] != EVALUATOR:
        raise ValueError('C06 request/evaluator identity mismatch')
    ids = [x['node_id'] for x in value['nodes']]
    if len(set(ids)) != len(ids) or set(ids) != set(expected_nodes(profile['layers'], profile['heads'])):
        raise ValueError('C06 duplicate/missing node identities')
    for x in value['nodes']:
        finite(x['score'], 'node score')
    ranking = value['ranking']
    scores = {x['node_id']: x['score'] for x in value['nodes']}
    if len(ranking) != len(ids) or set(ranking) != set(ids) or any(scores[a] < scores[b] for a,b in zip(ranking,ranking[1:])):
        raise ValueError('C06 invalid descending ranking')
    if len(value['points']) != len(GRID):
        raise ValueError('C06 incomplete sparsity grid')
    for p,f in zip(value['points'],GRID):
        if p['fraction'] != f or p['requested_k'] != int(f*len(ids)):
            raise ValueError('C06 k/coverage mismatch')
        metric(p['cpr'],'CPR')
        accuracy=metric(p['accuracy'],'accuracy')
        if isinstance(accuracy,float) and not 0<=accuracy<=1:raise ValueError('Accuracy outside [0,1]')
        if finite(p['weighted_edge_count'],'edge count')<0:raise ValueError('Negative edge count')
    for k in ('area_under','area_from_1','average','acc_auc'):
        metric(value['aggregates'][k],k)
    if value['reproduction_label'] != 'integration_smoke':
        raise ValueError('This adapter only certifies an integration smoke')
    json.dumps(value, allow_nan=False)
    return value
