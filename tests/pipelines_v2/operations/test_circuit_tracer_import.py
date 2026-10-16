import copy
from dataclasses import replace
import json
import pytest
from pipelines_v2.operations.graphs import AttributionGraph
from pipelines_v2.operations.graphs.contracts import digest
from pipelines_v2.operations.circuit_tracer.importer import import_graph
from pipelines_v2.operations.circuit_tracer.report import render_trace
from pipelines_v2.operations.circuit_tracer.specs import CircuitTraceSpec
from pipelines_v2.runtime.circuit_tracer import CircuitTracerRuntimeSpec


def fixture(kind='PLT'):
    dictionary={'dictionary_type':kind,'layer_map':{'0':[0] if kind=='PLT' else [0,1],'1':[1]},'revision':'fixture-only'}
    pair={'model':{'revision':'fixture-only'},'dictionary':dictionary,'tool':{'revision':'255d989e93be1299d53d67b59e9fd76a0e274b14'}}
    request={'pair':pair,'max_feature_nodes':2}
    # 1 feature + 2*2 errors + 2 tokens + 1 logit. Matrix rows are targets.
    matrix=[[0.]*8 for _ in range(8)]
    matrix[0][5]=.25;matrix[7][0]=-2.;matrix[7][2]=.75;matrix[7][6]=-.1
    raw={'schema_version':1,'producer':'circuit-tracer-v0.5.3','request_identity':digest(request),'scientific_request':request,
        'model_identity':digest(pair['model']),'dictionary_identity':digest(dictionary),'dictionary_type':kind,'layer_map':dictionary['layer_map'],
        'input_string':'fixture','input_tokens':[0,4],'n_layers':2,'active_features':[[0,1,3],[1,1,8]],'activation_values':[2.,3.],
        'selected_features':[0],'adjacency_matrix':matrix,'matrix_orientation':'target_rows_source_columns',
        'logit_targets':[{'token_str':'yes','vocab_idx':5}],'logit_probabilities':[.3]}
    return raw,request


@pytest.mark.parametrize('kind',['PLT','CLT'])
def test_pinned_schema_preserves_errors_signed_edges_and_reversible_ids(kind):
    raw,request=fixture(kind);graph=import_graph(raw,expected_request=request)
    decoded=AttributionGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert decoded.to_dict()==graph.to_dict()
    assert len(graph.nodes)==8 and len(graph.edges)==4
    assert [e.score for e in graph.edges]==[.25,-2.,.75,-.1]
    assert len([n for n in graph.nodes if n.kind=='error'])==4
    mapping={entry['matrix_index']:entry['node_id'] for entry in graph.provenance['reversible_id_map']}
    assert next(e for e in graph.edges if e.score==-2.).source==mapping[0]
    assert next(e for e in graph.edges if e.score==-2.).target==mapping[7]
    assert graph.nodes[0].coordinate['destination_layers']==([0] if kind=='PLT' else [0,1])
    assert graph.provenance['dictionary_type']==kind
    assert graph.pruning['omitted_active_features']==1


@pytest.mark.parametrize('mutate',[
    lambda raw:raw.update(schema_version=2),lambda raw:raw.update(model_identity='wrong'),
    lambda raw:raw.update(dictionary_type='CLT'),lambda raw:raw.update(matrix_orientation='source_rows'),
    lambda raw:raw['adjacency_matrix'].pop(2),lambda raw:raw['adjacency_matrix'][7].__setitem__(0,float('nan')),
    lambda raw:raw.update(selected_features=[999]),lambda raw:raw['layer_map'].update({'0':[0,1]}),
])
def test_import_fails_on_unknown_or_mismatched_graph(mutate):
    raw,request=fixture();request=copy.deepcopy(request);mutate(raw)
    with pytest.raises(ValueError):import_graph(raw,expected_request=request)


def test_request_binding_cannot_relabel_prompt_or_pair():
    raw,request=fixture();request={**request,'prompt':'changed'}
    with pytest.raises(ValueError,match='binding'):import_graph(raw,expected_request=request)


def test_runtime_and_operation_codecs_are_separate_from_mattr():
    from pipelines_v2.engine.base import PythonRuntimeSpec
    from pipelines_v2.operations.registry import operation_spec_from_dict
    runtime=CircuitTracerRuntimeSpec(local_python_sources=('pipelines_v2',))
    spec=CircuitTraceSpec(runtime=runtime,model_path='/model',dictionary_path='/dictionary')
    assert isinstance(PythonRuntimeSpec.from_dict(runtime.to_dict()),CircuitTracerRuntimeSpec)
    assert operation_spec_from_dict(spec.to_dict())==spec
    assert replace(spec,model_path='/moved',dictionary_path='/elsewhere').semantic_hash()==spec.semantic_hash()
    with pytest.raises(ValueError):replace(runtime,tool_lock={**runtime.tool_lock,'revision':'wrong'})
    assert replace(spec,max_feature_nodes=100,max_tokens=128).max_feature_nodes==100
    with pytest.raises(ValueError):replace(spec,max_feature_nodes=0)


def test_trace_report_keeps_semantics_and_unvalidated_status_visible():
    raw,request=fixture();graph=import_graph(raw,expected_request=request)
    html=render_trace({'graph':graph.to_dict(),'fidelity':{'scope':'same prompt'},'original_model_validation':{'status':'unvalidated'},'compute':{}})
    assert 'PLT replacement-flow graph' in html
    assert 'not a live CLT' in html and 'unvalidated' in html
    assert 'All nodes, including errors' in html
    assert 'upstream_matrix:7:0' in html


def test_trace_transform_preserves_public_payload_envelope(tmp_path):
    from types import SimpleNamespace
    from pipelines_v2.operations.circuit_tracer.report import trace_report_transform
    from pipelines_v2.operations.execution.common import coerce_transform_result
    raw,request=fixture();graph=import_graph(raw,expected_request=request)
    payload={'graph':graph.to_dict(),'fidelity':{},'original_model_validation':{'status':'unvalidated'},'compute':{},'dictionary_type':'PLT'}
    result=coerce_transform_result(trace_report_transform(trace=SimpleNamespace(id='fixture-artifact',result=lambda:payload),output_dir=tmp_path))
    assert result.payload['edge_count']==4 and result.payload['source_artifact_id']=='fixture-artifact'
    assert (tmp_path/'graph.html').is_file() and (tmp_path/'graph.json').is_file()


def test_actual_qwen_trace_matches_pinned_upstream_exporter_coordinates_and_direction():
    """Oracle is the official exporter run on the retained real trace, not our mapper.

    v0.5.3 create_nodes uses divmod(index - selected_features, n_pos) for
    errors; create_used_nodes_and_edges reads matrix[dsts, srcs]. The fixture
    records those unchanged functions' actual Node/edge output and source hashes.
    No upstream package, weights, network or model execution is needed here.
    """
    from pathlib import Path

    fixture_path = Path(__file__).parents[1] / 'fixtures/circuit_tracer_qwen3_actual_v053.json'
    retained = json.loads(fixture_path.read_text())
    runtime_lock = CircuitTracerRuntimeSpec().tool_lock
    assert retained['origin']['tool_revision'] == runtime_lock['revision']
    for path, source in retained['origin']['upstream_sources'].items():
        assert source['sha256'] == runtime_lock['sources'][path]

    raw = copy.deepcopy(retained['raw'])
    # Project away unused active-feature metadata only. Selected features keep
    # their original matrix order/coordinates; all 90 actual edges are retained.
    records = retained['selected_feature_records']
    assert [record['index'] for record in records] == raw['selected_features']
    raw['active_features'] = [record['coordinate'] for record in records]
    raw['activation_values'] = [record['activation'] for record in records]
    raw['selected_features'] = list(range(len(records)))
    size = retained['matrix_size']
    raw['adjacency_matrix'] = [[0.0] * size for _ in range(size)]
    for target, source, value in retained['nonzero_entries']:
        raw['adjacency_matrix'][target][source] = value
    graph = import_graph(raw, expected_request=raw['scientific_request'])

    oracle = retained['oracle']
    by_matrix_index = {
        entry['matrix_index']: entry['node_id']
        for entry in graph.provenance['reversible_id_map']
    }
    nodes = {node.node_id: node for node in graph.nodes}
    error_id = by_matrix_index[oracle['error_matrix_index']]
    output_id = by_matrix_index[oracle['output_matrix_index']]
    error = nodes[error_id]
    expected_error = oracle['error_node']
    # Layer 7 / position 1 deliberately distinguishes layer-major from
    # position-major ordering; this is actual upstream label "7_-1_1".
    assert expected_error['node_id'] == '7_-1_1'
    assert error.kind == 'error'
    assert error.coordinate['layer'] == int(expected_error['layer'])
    assert error.coordinate['token_position'] == expected_error['ctx_idx']
    assert nodes[output_id].kind == 'output'
    assert nodes[output_id].coordinate['vocab_idx'] == oracle['output_node']['feature'] == 198
    expected_edge = oracle['error_to_output_edge']
    assert expected_edge['source'] == expected_error['node_id']
    assert expected_edge['target'] == oracle['output_node']['node_id']
    actual_edge = next(e for e in graph.edges if (e.source, e.target) == (error_id, output_id))
    assert actual_edge.score == expected_edge['weight'] == 0.19827695190906525
    assert actual_edge.evidence_ids == ('upstream_matrix:60:17',)
    assert not any(e.source == output_id and e.target == error_id for e in graph.edges)

    feature_id = by_matrix_index[oracle['feature_matrix_index']]
    feature_edge = next(e for e in graph.edges if (e.source, e.target) == (feature_id, output_id))
    assert feature_edge.score == oracle['feature_to_output_edge']['weight'] == -0.8701759576797485
    assert len(graph.nodes) == size == 61
    assert len(graph.edges) == oracle['nonzero_edge_count'] == 90
    assert sum(node.kind == 'error' for node in graph.nodes) == 56
