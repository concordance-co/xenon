from dataclasses import replace
import json
from types import SimpleNamespace
import pytest
from tests.pipelines_v2.engine.test_native_sites import session_for
from tests.pipelines_v2.operations.test_graph_interventions import fixture_spec
from tests.pipelines_v2.operations.test_external_mattr import result_fixture
from pipelines_v2.operations.graphs import AttributionGraph,GraphEdge
from pipelines_v2.operations.graphs.execution import evaluate_routes
from pipelines_v2.operations.graphs.annotations import annotate_c06
from pipelines_v2.operations.graphs.report import render_graph,graph_report_transform


@pytest.fixture(scope='module')
def graph_payload():
    with session_for('gpt2') as session:return evaluate_routes(fixture_spec(session),session)


def test_graph_roundtrip_and_report_transform_envelope(graph_payload,tmp_path):
    graph=AttributionGraph.from_dict(json.loads(json.dumps(graph_payload['graph'])))
    assert graph.to_dict()==graph_payload['graph']
    html=render_graph(graph)
    assert 'not tested' in html and 'Held-out controls' in html
    output=graph_report_transform(route=SimpleNamespace(result=lambda:graph_payload),output_dir=str(tmp_path))
    from pipelines_v2.operations.execution.common import coerce_transform_result
    result=coerce_transform_result(output)
    assert result.payload['evidence_count']==5
    assert (tmp_path/'graph.html').read_text()==render_graph(graph,examples=graph_payload['examples'])


@pytest.mark.parametrize('change',[
    {'provenance':{}}, {'semantics':'ranking'}, {'omitted_contributions':{'conservation_defined':True}},
    {'target':{'metric':'changed'}}])
def test_graph_rejects_missing_or_inconsistent_contract(graph_payload,change):
    graph=AttributionGraph.from_dict(graph_payload['graph'])
    with pytest.raises(ValueError):replace(graph,**change)


def test_ranking_cannot_be_edge_and_effect_must_match_evidence(graph_payload):
    graph=AttributionGraph.from_dict(graph_payload['graph'])
    with pytest.raises(ValueError,match='Rankings'):replace(graph.edges[0],semantics='ranking_annotation')
    with pytest.raises(ValueError,match='disagrees'):replace(graph,edges=(replace(graph.edges[0],score=999.),))
    with pytest.raises(ValueError,match='endpoint'):replace(graph,edges=(replace(graph.edges[0],source='missing'),))


def test_annotation_namespace_coverage_and_explicit_tensor_join(graph_payload):
    graph=AttributionGraph.from_dict(graph_payload['graph']);result=result_fixture()
    annotation=annotate_c06(graph,result)
    assert annotation.provenance['c06']['matched']==0
    assert len(annotation.provenance['c06']['unmatched_node_ids'])==157
    assert all(a['native_node_id'] is None for a in annotation.annotations)
    assert annotation.edges==graph.edges
    source='a0.h1';target=graph.nodes[0].node_id
    next(n for n in result['nodes'] if n['node_id']==source)['score']=1000.
    result['ranking'].remove(source);result['ranking'].insert(0,source)
    with pytest.raises(ValueError,match='observed'):annotate_c06(graph,result,mapping={source:target})
    evidence={'model_identity':graph.model_identity,'model_revision':result['request']['model']['revision'],
        'source_loading_transforms':['explicit fixture identity transform'],'source_site_semantics':'fixture input',
        'native_site_semantics':graph.nodes[0].coordinate,'example_keys':['heldout-0'],'token_positions':[2],'evidence_refs':['fixture://tensor-oracle']}
    joined=annotate_c06(graph,result,mapping={source:target},upstream_tensors={source:[[1.,2.]]},native_tensors={target:[[1.,2.]]},basis_evidence=evidence)
    assert joined.provenance['c06']['matched']==1
    assert joined.annotations[0]['native_node_id']==target
    with pytest.raises(ValueError,match='disagree'):
        annotate_c06(graph,result,mapping={source:target},upstream_tensors={source:[[1.,2.]]},native_tensors={target:[[1.,3.]]},basis_evidence=evidence)


def test_report_escapes_labels(graph_payload):
    graph=AttributionGraph.from_dict(graph_payload['graph'])
    graph=replace(graph,nodes=(replace(graph.nodes[0],label='<script>alert(1)</script>'),*graph.nodes[1:]))
    html=render_graph(graph)
    assert '<script>' not in html and '&lt;script&gt;' in html
