"""Stored/manual candidate explanations; no network, judge, or scientific promotion."""
from dataclasses import dataclass, asdict
from typing import Protocol


@dataclass(frozen=True)
class ExplanationCandidate:
    neuron_id: str
    text: str
    provider_id: str
    provider_version: str
    created_from_context_ids: tuple[str, ...]
    view: str = 'signed'
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or self.view not in ('signed', 'product_only'):
            raise ValueError('unsupported explanation schema/view')
        if not all(isinstance(x, str) and x.strip() for x in (self.neuron_id, self.text, self.provider_id, self.provider_version)):
            raise ValueError('candidate identity, text and provider version required')
        if len(self.text) > 2000:
            raise ValueError('candidate exceeds 2000-character bound')


class ExplanationProvider(Protocol):
    def explain(self, neuron_id: str, contexts: tuple[dict, ...]) -> ExplanationCandidate | None: ...


class StoredExplanationProvider:
    """Versioned user-authored records; missing records remain unexplained."""
    def __init__(self, candidates):
        self.candidates = {}
        for candidate in candidates:
            if candidate.neuron_id in self.candidates:
                raise ValueError('duplicate explanation candidate')
            self.candidates[candidate.neuron_id] = candidate

    def explain(self, neuron_id, contexts):
        return self.candidates.get(neuron_id)


ManualExplanationProvider = StoredExplanationProvider


def build_dossiers(summary, provider: ExplanationProvider, *, view='signed', interventions=None):
    if view not in ('signed', 'product_only'):
        raise ValueError('unsupported explanation view')
    dossiers = []
    for neuron in summary['neurons']:
        # Only this separate discovery reservoir crosses the provider boundary.
        contexts = []
        for original in neuron['discovery']:
            if original['split'] != 'discovery':
                raise ValueError('heldout leakage at explanation-provider boundary')
            context = dict(original, values=dict(original['values']))
            if view == 'product_only':
                context['values'] = {'mlp_product': context['values']['mlp_product']}
            contexts.append(context)
        candidate = provider.explain(neuron['neuron_id'], tuple(contexts))
        if candidate is not None:
            allowed = {row['context_id'] for row in contexts}
            if candidate.neuron_id != neuron['neuron_id'] or candidate.view != view or not candidate.created_from_context_ids or not set(candidate.created_from_context_ids) <= allowed:
                raise ValueError('candidate references unseen, heldout, or wrong-view evidence')
        dossiers.append({'neuron_id': neuron['neuron_id'], 'observations': neuron,
            'explanation': asdict(candidate) if candidate else None,
            'explanation_status': 'unverified_manual_candidate' if candidate else 'unexplained',
            'explanation_evaluation': 'not_tested', 'human_validation': 'not_available',
            'intervention': (interventions or {}).get(neuron['neuron_id'], {'status': 'not_tested'}),
            'scientific_verdict': 'not_tested'})
    return {'schema_version': 1, 'kind': 'neuron_dossier_result',
        'summary': {'neuron_count': len(dossiers), 'unexplained_count': sum(d['explanation'] is None for d in dossiers),
            'evidence_level': 'engineering', 'scientific_verdict': 'not_tested', 'explanation_view': view},
        'dossiers': dossiers}
