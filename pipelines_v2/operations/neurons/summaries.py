"""Bounded, signed native-neuron summaries over an observation iterator.

Memory is O(max_neurons * (k + reservoir_size) + max_observations): the latter
is an explicitly bounded duplicate-ID set. Reservoirs use seeded min-hash
selection, making retained IDs independent of input/batch ordering. Floating
point aggregate sums can differ at roundoff precision with reordered input.
"""
from bisect import bisect_right
from dataclasses import dataclass, asdict
import hashlib
import math

QUANTITIES = ('gate_preactivation', 'gate_activation', 'up_projection', 'mlp_product')


@dataclass(frozen=True)
class SummaryLimits:
    k: int = 3
    reservoir_size: int = 64
    max_neurons: int = 32
    max_observations: int = 65536
    max_context_chars: int = 400
    seed: int = 0
    weak_threshold: float = 0.1
    histogram_edges: tuple[float, ...] = (-1., -0.1, 0., 0.1, 1.)

    def __post_init__(self):
        for name in ('k', 'reservoir_size', 'max_neurons', 'max_observations', 'max_context_chars'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(self.weak_threshold) or self.weak_threshold < 0:
            raise ValueError('weak_threshold must be finite and nonnegative')
        edges = self.histogram_edges
        if not edges or any(not math.isfinite(x) for x in edges) or any(a >= b for a, b in zip(edges, edges[1:])):
            raise ValueError('histogram edges must be finite and strictly increasing')


def _priority(context, seed):
    return hashlib.sha256(f'{seed}:{context["context_id"]}'.encode()).hexdigest(), context['context_id']


def _retain(rows, row, count, key):
    rows.append(row)
    rows.sort(key=key)
    del rows[count:]


def summarize_observations(observations, limits=SummaryLimits()):
    """Consume explicit neuron/token observations without retaining the stream."""
    states, seen = {}, set()
    for item in observations:
        row = dict(item)
        neuron, context = row['neuron_id'], row['context_id']
        if not isinstance(neuron, str) or not isinstance(context, str) or not neuron or not context:
            raise ValueError('nonempty stable neuron/context IDs required')
        identity = (neuron, context)
        if identity in seen:
            raise ValueError('duplicate neuron/context observation')
        if len(seen) >= limits.max_observations:
            raise ValueError('observation budget exceeded')
        seen.add(identity)
        if row.get('split') not in ('discovery', 'heldout'):
            raise ValueError('explicit discovery/heldout split required')
        if not isinstance(row.get('text'), str) or len(row['text']) > limits.max_context_chars:
            raise ValueError('original text window exceeds declared bound')
        if not row.get('source') or not row.get('token_window'):
            raise ValueError('source and original token-window references required')
        values = {q: float(row['values'][q]) for q in QUANTITIES}
        if any(not math.isfinite(v) for v in values.values()):
            raise ValueError('native values must be finite')
        row['values'] = values
        if neuron not in states:
            if len(states) >= limits.max_neurons:
                raise ValueError('neuron budget exceeded')
            states[neuron] = {'neuron_id': neuron, 'count': 0, 'split_counts': {'discovery': 0, 'heldout': 0},
                'top': [], 'bottom': [], 'random': [], 'weak': [], 'discovery': [],
                'distribution': {q: {'sum': 0., 'min': math.inf, 'max': -math.inf,
                    'negative': 0, 'zero': 0, 'positive': 0,
                    'histogram': [0] * (len(limits.histogram_edges) + 1)} for q in QUANTITIES}}
        state = states[neuron]
        state['count'] += 1
        state['split_counts'][row['split']] += 1
        for q, v in values.items():
            d = state['distribution'][q]
            d['sum'] += v
            d['min'], d['max'] = min(d['min'], v), max(d['max'], v)
            d['negative' if v < 0 else 'positive' if v > 0 else 'zero'] += 1
            d['histogram'][bisect_right(limits.histogram_edges, v)] += 1
        _retain(state['top'], row, limits.k, lambda r: (-r['values']['mlp_product'], r['context_id']))
        _retain(state['bottom'], row, limits.k, lambda r: (r['values']['mlp_product'], r['context_id']))
        _retain(state['random'], row, limits.reservoir_size, lambda r: _priority(r, limits.seed))
        if abs(values['mlp_product']) <= limits.weak_threshold:
            _retain(state['weak'], row, limits.reservoir_size, lambda r: _priority(r, limits.seed))
        if row['split'] == 'discovery':
            _retain(state['discovery'], row, limits.reservoir_size, lambda r: _priority(r, limits.seed))
    for state in states.values():
        sample = state['random']
        for q, d in state['distribution'].items():
            ordered = sorted(r['values'][q] for r in sample)
            d['mean'] = d.pop('sum') / state['count']
            d['quantiles'] = {str(p): ordered[round(p * (len(ordered) - 1))] for p in (0., .25, .5, .75, 1.)}
            d['quantile_method'] = 'exact nearest rank (rounded index)' if len(sample) == state['count'] else 'seeded min-hash reservoir nearest rank (rounded index)'
            d['quantile_sample_count'] = len(sample)
        for name in ('top', 'bottom', 'random', 'weak', 'discovery'):
            state[name] = [dict(r, product_percentile=sum(x['values']['mlp_product'] <= r['values']['mlp_product'] for x in sample) / len(sample),
                percentile_basis='empirical <= CDF of random reservoir') for r in state[name]]
    return {'schema_version': 1, 'kind': 'neuron_summary_result', 'limits': asdict(limits),
        'histogram_convention': '[-inf,e0), [e0,e1), ..., [elast,+inf)',
        'summary': {'neuron_count': len(states), 'observation_count': len(seen), 'evidence_level': 'engineering'},
        'neurons': [states[key] for key in sorted(states)]}
