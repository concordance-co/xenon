"""C07: typed graph evidence. Scores of different semantics are never interchangeable."""
from dataclasses import dataclass, field, asdict
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def finite(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError('Graph measurements must be finite numbers')
    return value


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    kind: str
    coordinate: dict
    label: str

    def __post_init__(self):
        if self.kind not in ('native_site', 'output', 'embedding_token', 'error', 'transcoder_feature', 'token'):
            raise ValueError('Unknown graph node kind')
        if not self.coordinate or not self.label or self.node_id != digest({'kind': self.kind, 'coordinate': self.coordinate}):
            raise ValueError('Node identity must bind its typed coordinate')
        if self.kind == 'native_site':
            from pipelines_v2.operations.common.native import NativeSite
            NativeSite.from_dict(self.coordinate)

    @classmethod
    def create(cls, kind, coordinate, label):
        return cls(digest({'kind':kind, 'coordinate':coordinate}), kind, coordinate, label)


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    semantics: str
    metric: str
    units: str
    score: float
    baseline: str
    intervention: str
    evidence_ids: tuple[str, ...]
    sign_convention: str = 'positive increases target minus foil'

    def __post_init__(self):
        if self.semantics not in ('native_intervention_effect', 'replacement_flow'):
            raise ValueError('Rankings are annotations, not graph edges')
        if not all((self.source, self.target, self.metric, self.units, self.baseline, self.intervention, self.evidence_ids, self.sign_convention)):
            raise ValueError('Edge lacks measurement semantics or evidence')
        finite(self.score)
        object.__setattr__(self, 'evidence_ids', tuple(self.evidence_ids))


@dataclass(frozen=True)
class RouteEvidence:
    evidence_id: str
    node_id: str
    example_key: str
    split: str
    template: str
    arm: str
    read_site: dict
    write_site: dict
    donor: dict
    intervention: dict
    target: dict
    baseline: float
    observed: float
    collateral: dict
    engineering_status: str
    scientific_verdict: str = 'not_tested'

    def __post_init__(self):
        if self.split != 'heldout' or not self.template or not self.example_key:
            raise ValueError('Route measurements require explicit held-out coverage')
        if self.arm not in ('identity', 'zero', 'opposite', 'same_label', 'unmatched'):
            raise ValueError('Unknown route control')
        if self.scientific_verdict != 'not_tested' or self.engineering_status not in ('passed', 'rejected'):
            raise ValueError('Plumbing evidence cannot certify a scientific route')
        from pipelines_v2.operations.common.native import NativeSite
        read, write = NativeSite.from_dict(self.read_site), NativeSite.from_dict(self.write_site)
        from dataclasses import replace
        if read.read_or_write != 'read' or write != replace(read, read_or_write='write'):
            raise ValueError('Read/write coordinates must differ only in access mode')
        if self.node_id != GraphNode.create('native_site', self.read_site, read.name).node_id:
            raise ValueError('Evidence node does not match its read coordinate')
        finite(self.baseline); finite(self.observed)
        for value in self.collateral.values(): finite(value)
        expected = digest({k:v for k,v in asdict(self).items() if k != 'evidence_id'})
        if self.evidence_id != expected: raise ValueError('Evidence content identity mismatch')

    @classmethod
    def create(cls, **values):
        values.setdefault('scientific_verdict', 'not_tested')
        return cls(evidence_id=digest(values), **values)


@dataclass(frozen=True)
class AttributionGraph:
    source_run_ids: tuple[str, ...]
    provenance: dict
    target: dict
    model_identity: str
    algorithm: dict
    semantics: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    evidence: tuple[RouteEvidence, ...]
    approximations: tuple[str, ...]
    assumptions: dict
    pruning: dict
    omitted_contributions: dict
    annotations: tuple[dict, ...] = ()
    candidates: tuple[dict, ...] = ()
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or self.semantics not in ('native_intervention_effect', 'replacement_flow'):
            raise ValueError('Unsupported C07 version/semantics')
        if not all((self.source_run_ids, self.provenance, self.target, self.model_identity, self.algorithm, self.assumptions, self.pruning, self.omitted_contributions)):
            raise ValueError('C07 requires provenance, target, algorithm and approximation accounting')
        if len(self.nodes)>100 or len(self.edges)>300 or len(self.evidence)>160:
            raise ValueError('Graph exceeds bounded v1 report envelope')
        nodes={node.node_id:node for node in self.nodes}
        evidence={item.evidence_id:item for item in self.evidence}
        if len(nodes)!=len(self.nodes) or len(evidence)!=len(self.evidence): raise ValueError('Duplicate graph identities')
        for node in self.nodes:
            if node.kind=='native_site' and node.coordinate['model_identity'] != self.model_identity:
                raise ValueError('Mixed model identities')
        for item in self.evidence:
            if item.node_id not in nodes or item.target != self.target:
                raise ValueError('Evidence does not bind graph node/target')
        for edge in self.edges:
            if edge.source not in nodes or edge.target not in nodes or edge.semantics != self.semantics:
                raise ValueError('Edge endpoint/semantics mismatch')
            if self.semantics=='native_intervention_effect':
                if nodes[edge.target].kind!='output' or edge.units!='logit' or edge.metric!='mean_target_margin_delta':
                    raise ValueError('Native direct effects must terminate at their measured output')
                rows=[evidence.get(key) for key in edge.evidence_ids]
                if any(row is None or row.node_id!=edge.source or row.arm!=edge.intervention for row in rows):
                    raise ValueError('Edge lacks corresponding intervention evidence')
                expected=sum(row.observed-row.baseline for row in rows)/len(rows)
                if not math.isclose(edge.score, expected, rel_tol=1e-9, abs_tol=1e-9): raise ValueError('Edge disagrees with recorded effects')
        if any(a.get('semantics')!='ranking_annotation' for a in self.annotations):
            raise ValueError('Annotations must stay rankings')
        # Flow conservation is not defined for independent original-model interventions.
        if self.semantics=='native_intervention_effect' and self.omitted_contributions.get('conservation_defined') is not False:
            raise ValueError('Native direct effects are nonadditive; no conservation claim')
        json.dumps(self.to_dict(), allow_nan=False)

    @property
    def graph_id(self): return digest(asdict(self))

    def to_dict(self): return json.loads(json.dumps({'graph_id':self.graph_id, **asdict(self)}, allow_nan=False))

    @classmethod
    def from_dict(cls, value):
        data=dict(value); expected=data.pop('graph_id')
        data['nodes']=tuple(GraphNode(**v) for v in data['nodes'])
        data['edges']=tuple(GraphEdge(**v) for v in data['edges'])
        data['evidence']=tuple(RouteEvidence(**v) for v in data['evidence'])
        for key in ('source_run_ids','approximations','annotations','candidates'):data[key]=tuple(data[key])
        result=cls(**data)
        if result.graph_id!=expected:raise ValueError('Graph identity mismatch')
        return result
