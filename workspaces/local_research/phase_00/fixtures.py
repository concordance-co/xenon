"""Frozen before execution: eight prompts, four recipients, three donor maps."""
import random

REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
MODEL = 'workspaces/local_research/.cache/gpt2'
TOKENS = (6342, 3576)  # exactly ' Paris', ' London' under the pinned tokenizer
LAYER = 5
SEED = 1729
# Two matched templates per role. All fixtures are exploratory; none held out.
PROMPTS = (
    ('t-fr-0', 'The capital of France is', 'France', 'target'),
    ('t-en-0', 'The capital city of England is', 'England', 'target'),
    ('t-fr-1', 'Q: What is the capital of France? A:', 'France', 'target'),
    ('t-en-1', 'England has its capital in', 'England', 'target'),
    ('d-fr-0', 'The capital city of France is', 'France', 'donor'),
    ('d-en-0', 'The capital of England is', 'England', 'donor'),
    ('d-fr-1', 'France has its capital in', 'France', 'donor'),
    ('d-en-1', 'Q: What is the capital of England? A:', 'England', 'donor'),
)
TARGETS = tuple(row[0] for row in PROMPTS if row[3] == 'target')
DONORS = tuple(row[0] for row in PROMPTS if row[3] == 'donor')
_random = list(DONORS)
random.Random(SEED).shuffle(_random)
PAIRS = {
    'matched': dict(zip(TARGETS, ('d-en-0', 'd-fr-0', 'd-en-1', 'd-fr-1'))),
    # Same label, different template: tests nonspecific prompt/context disruption.
    'same_label': dict(zip(TARGETS, ('d-fr-1', 'd-en-1', 'd-fr-0', 'd-en-0'))),
    'random_donor': dict(zip(TARGETS, _random)),
}


def examples():
    from pipelines_v2.api import Example
    result = []
    for key, prompt, country, role in PROMPTS:
        labels = {'country': country, 'role': role}
        for arm, mapping in PAIRS.items():
            labels[arm] = key if role == 'target' else next(t for t, d in mapping.items() if d == key)
        result.append(Example(key=key, prompt=prompt, labels=labels))
    return tuple(result)
