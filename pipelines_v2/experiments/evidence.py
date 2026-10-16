"""C05 evidence records. Verdicts are declarations checked against linked evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .packs import ExperimentPack, canonical, fields, names, require, sha, text

LEVELS = {'engineering': 0, 'behavioral': 1, 'representational': 2,
          'localized_representational': 3, 'causal': 4, 'mechanistic': 5}


@dataclass(frozen=True)
class EvidenceRecord:
    _json: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, pack: ExperimentPack) -> 'EvidenceRecord':
        data = json.loads(canonical(payload))
        fields(data, 'schema_version evidence_id pack_identity preset question_id hypothesis_id engineering_status scientific_verdict metric evaluator_digest split split_link control_arms effect uncertainty evidence_level limitations links', 'evidence')
        require(type(data['schema_version']) is int and data['schema_version'] == 1, 'Unsupported evidence schema_version')
        text(data['evidence_id'], 'evidence_id')
        require(data['pack_identity'] == pack.identity, 'Evidence pack identity mismatch')
        plan = pack.to_dict()
        require(data['preset'] in plan['presets'], 'Unknown evidence preset')
        require(data['question_id'] in plan['question_ids'], 'Unknown question')
        require(data['hypothesis_id'] is None or data['hypothesis_id'] in plan['hypothesis_ids'], 'Unknown hypothesis')
        require(data['engineering_status'] in ('tested', 'qualified', 'pending', 'failed'), 'Invalid engineering status')
        require(data['scientific_verdict'] in ('supported', 'refuted', 'inconclusive', 'not_tested'), 'Invalid scientific verdict')
        require(data['metric'] in plan['metrics'], 'Unknown metric')
        require(data['evaluator_digest'] == plan['metrics'][data['metric']]['evaluator_digest'], 'Evaluator identity mismatch')
        require(data['split'] in plan['split_policy']['partitions'], 'Unknown split')
        names(data['limitations'], 'limitations')
        require(data['evidence_level'] in LEVELS, 'Unknown methodology evidence level')
        require(isinstance(data['links'], list) and bool(data['links']), 'Evidence requires direct artifact links')
        for link in data['links']:
            fields(link, 'artifact_id run_id scientific_identity selector', 'evidence link')
            for key in ('artifact_id', 'run_id'):
                text(link[key], key)
                require(link[key].lower() not in ('latest', 'current'), 'Evidence must use immutable IDs')
            sha(link['scientific_identity'], 'scientific identity')
            selector = link['selector']
            fields(selector, 'domain pointer expected', 'selector')
            require(selector['domain'] in ('manifest', 'result'), 'Selector domain must be manifest or bounded result')
            require(isinstance(selector['pointer'], str) and selector['pointer'].startswith('/') and selector['pointer'] != '/', 'Exact JSON pointer required')
        fields(data['effect'], 'value link', 'effect')
        index = data['effect']['link']
        require(type(index) is int and 0 <= index < len(data['links']), 'Effect requires a valid link index')
        require(data['effect']['value'] == data['links'][index]['selector']['expected'], 'Effect must match the linked measurement')
        require(type(data['effect']['value']) in (int, float), 'Effect must be a finite numeric measurement')
        require(data['links'][index]['selector']['domain'] == 'result', 'Effect must reference a result measurement')
        fields(data['uncertainty'], 'value reason link', 'uncertainty')
        text(data['uncertainty']['reason'], 'uncertainty method or unavailable reason')
        require(data['uncertainty']['value'] is None or type(data['uncertainty']['value']) in (int, float), 'Uncertainty must be numeric or explicitly unavailable')
        uncertainty_index = data['uncertainty']['link']
        if data['uncertainty']['value'] is None:
            require(uncertainty_index is None, 'Unavailable uncertainty cannot claim a measurement link')
        else:
            require(type(uncertainty_index) is int and 0 <= uncertainty_index < len(data['links']), 'Measured uncertainty requires a link')
            require(data['uncertainty']['value'] == data['links'][uncertainty_index]['selector']['expected'], 'Uncertainty must match linked measurement')
        split_index = data['split_link']
        if split_index is not None:
            require(type(split_index) is int and 0 <= split_index < len(data['links']), 'Split requires a valid link')
            require(data['links'][split_index]['selector']['expected'] == data['split'], 'Split link must identify the observed split')
        require(isinstance(data['control_arms'], dict) and set(data['control_arms']) <= set(plan['controls']), 'Unknown control arms')
        for index in data['control_arms'].values():
            require(type(index) is int and 0 <= index < len(data['links']), 'Control requires a valid evidence link')
        if data['scientific_verdict'] == 'supported':
            require(LEVELS[data['evidence_level']] > 0, 'Code success alone supports no scientific claim')
        if data['scientific_verdict'] in ('supported', 'refuted') and LEVELS[data['evidence_level']] > 0:
            required = {name for name, control in plan['controls'].items() if control['required']}
            require(bool(required) and required <= set(data['control_arms']), 'Scientific conclusions require planned controls')
            require(data['split'] in plan['split_policy']['held_out'] and split_index is not None, 'Scientific conclusions require linked held-out evidence')
            measurements = [data['links'][index] for index in [data['effect']['link'], *data['control_arms'].values()]]
            targets = {(link['artifact_id'], link['selector']['domain'], link['selector']['pointer']) for link in measurements}
            require(len(targets) == len(measurements), 'Controls need distinct selectors from the target effect')
            require(data['engineering_status'] in ('tested', 'qualified'), 'Scientific conclusions require completed engineering')
            require(data['preset'] not in ('setup', 'smoke'), 'Smoke fixtures are not scientific support')
        if data['preset'] in ('setup', 'smoke'):
            require(data['evidence_level'] == 'engineering', 'Fixture evidence must stay at engineering level')
        return cls(canonical(data))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._json)

    @property
    def identity(self) -> str:
        return hashlib.sha256(self._json.encode()).hexdigest()


def pointer_value(payload: Any, pointer: str) -> Any:
    """Resolve RFC6901 pointers; no expressions, globbing or remote lookup."""
    value = payload
    try:
        for token in pointer.split('/')[1:]:
            # Reject invalid escapes instead of silently selecting another key.
            require('~' not in token.replace('~0', '').replace('~1', ''), 'Invalid JSON pointer escape')
            token = token.replace('~1', '/').replace('~0', '~')
            if isinstance(value, list):
                require(token.isdigit() and (token == '0' or not token.startswith('0')), 'Invalid array index')
                value = value[int(token)]
            else:
                value = value[token]
        return value
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f'Evidence selector does not resolve: {pointer}') from exc
