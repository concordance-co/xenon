"""Lossless bounded upstream graph import: rows are targets, columns are sources."""
from pipelines_v2.operations.graphs.contracts import AttributionGraph,GraphNode,GraphEdge,digest,finite


def import_graph(raw, *, expected_request):
    if raw.get('schema_version')!=1 or raw.get('producer')!='circuit-tracer-v0.5.3':raise ValueError('Unknown upstream graph schema')
    if raw.get('scientific_request')!=expected_request or raw.get('request_identity')!=digest(expected_request):raise ValueError('Trace request/pair binding mismatch')
    pair=expected_request['pair'];dictionary=pair['dictionary']
    if raw['dictionary_type'] not in ('PLT','CLT') or raw['dictionary_type']!=dictionary['dictionary_type'] or raw['layer_map']!=dictionary['layer_map']:
        raise ValueError('Dictionary type/layer mapping mismatch')
    if raw['model_identity']!=digest(pair['model']) or raw['dictionary_identity']!=digest(dictionary):raise ValueError('Model/dictionary identity mismatch')
    if raw['matrix_orientation']!='target_rows_source_columns':raise ValueError('Unknown adjacency orientation')
    layers=raw['n_layers'];tokens=raw['input_tokens'];features=raw['active_features'];selected=raw['selected_features'];targets=raw['logit_targets']
    if type(layers)is not int or not 1<=layers<=80 or not tokens or len(tokens)>2 or len(targets)!=1:raise ValueError('Trace exceeds supported bounded graph cell')
    if set(raw['layer_map'])!={str(i) for i in range(layers)}:raise ValueError('Incomplete source/destination layer mapping')
    for source,destinations in raw['layer_map'].items():
        if not destinations or len(set(destinations))!=len(destinations) or any(type(i)is not int or not int(source)<=i<layers for i in destinations):raise ValueError('Invalid transcoder destination layers')
        if raw['dictionary_type']=='PLT' and destinations!=[int(source)]:raise ValueError('PLT cannot write across layers')
    if len(selected)>expected_request['max_feature_nodes'] or len(set(selected))!=len(selected) or any(type(i)is not int or not 0<=i<len(features) for i in selected):raise ValueError('Invalid selected feature coverage')
    if len(features)!=len(raw['activation_values']):raise ValueError('Active feature/value alignment mismatch')
    for coordinate,value in zip(features,raw['activation_values']):
        if len(coordinate)!=3 or any(type(i)is not int or i<0 for i in coordinate) or coordinate[0]>=layers or coordinate[1]>=len(tokens):raise ValueError('Invalid upstream feature coordinate')
        finite(value)
    common={'model_identity':raw['model_identity'],'dictionary_identity':raw['dictionary_identity'],'dictionary_type':raw['dictionary_type']}
    nodes=[];id_map=[]
    def add(kind,coordinate,label,upstream):
        node=GraphNode.create(kind,{**common,**coordinate},label);nodes.append(node);id_map.append({'matrix_index':len(nodes)-1,'upstream':upstream,'node_id':node.node_id})
    for index in selected:
        layer,pos,feature=features[index]
        add('transcoder_feature',{'source_layer':layer,'destination_layers':raw['layer_map'][str(layer)],'token_position':pos,'feature_index':feature,'activation':raw['activation_values'][index]},f'L{layer} feature {feature} @ {pos}',{'kind':'feature','active_feature_index':index,'coordinate':features[index]})
    for layer in range(layers):
        for pos in range(len(tokens)):add('error',{'layer':layer,'token_position':pos,'definition':'original MLP output minus transcoder reconstruction'},f'L{layer} error @ {pos}',{'kind':'error','layer':layer,'position':pos})
    for pos,token in enumerate(tokens):
        if type(token)is not int or token<0:raise ValueError('Invalid token identity')
        add('token',{'token_position':pos,'token_id':token},f'Token {token} @ {pos}',{'kind':'token','position':pos,'token_id':token})
    for target in targets:
        if set(target)!={'token_str','vocab_idx'} or type(target['vocab_idx'])is not int or target['vocab_idx']<0:raise ValueError('Unsupported logit target')
        add('output',target,'Logit '+target['token_str'],{'kind':'logit',**target})
    if len(raw['logit_probabilities'])!=len(targets) or any(not 0<=finite(p)<=1 for p in raw['logit_probabilities']):raise ValueError('Invalid target probability coverage')
    matrix=raw['adjacency_matrix'];n=len(nodes)
    if len(matrix)!=n or any(len(row)!=n for row in matrix):raise ValueError('Adjacency does not preserve all selected features, errors, tokens and logits')
    edges=[]
    for target,row in enumerate(matrix):
        for source,score in enumerate(row):
            finite(score)
            if not score:continue
            if nodes[target].kind not in ('transcoder_feature','output'):raise ValueError('Unexpected incoming edge to fixed error/token node')
            edges.append(GraphEdge(nodes[source].node_id,nodes[target].node_id,'replacement_flow','local_replacement_direct_effect',
                'target preactivation units' if nodes[target].kind=='transcoder_feature' else 'logit',score,
                'zero source contribution in local linear replacement model','frozen-attention/normalization local attribution',
                (f'upstream_matrix:{target}:{source}',),'signed source contribution to target under upstream linearization'))
    raw_identity=digest(raw)
    return AttributionGraph(source_run_ids=('request:'+raw['request_identity'],),provenance={'upstream_graph_identity':raw_identity,'request_identity':raw['request_identity'],
        'tool_revision':pair['tool']['revision'],'dictionary_identity':raw['dictionary_identity'],'dictionary_type':raw['dictionary_type'],'reversible_id_map':id_map,
        'raw_artifacts':'enclosing result.raw_outputs; upstream_graph.json plus upstream_graph.pt'},
        target={'logit_targets':targets,'probabilities':raw['logit_probabilities'],'input_tokens':tokens,'input_string':raw['input_string']},
        model_identity=raw['model_identity'],algorithm={'name':'circuit-tracer','version':'0.5.3','backend':'transformerlens','max_feature_nodes':expected_request['max_feature_nodes']},
        semantics='replacement_flow',nodes=tuple(nodes),edges=tuple(edges),evidence=(),approximations=(
            'Local linear replacement model; attention patterns and normalization scales frozen for attribution',
            'Feature search capped; not a complete circuit','Replacement-flow weights are not original-model causal validation'),
        assumptions={'attention':'cached forward patterns; gradients stopped','normalization':'cached normalization scales; gradients stopped',
            'MLP':'pretrained transcoder features plus explicit reconstruction error','BOS':'upstream first position errors/features ignored','units':'signed gradient contraction with source activation'},
        pruning={'rule':'upstream max_feature_nodes selection; no additional importer pruning','total_active_features':len(features),'retained_features':len(selected),
            'omitted_active_features':len(features)-len(selected),'omitted_zero_entries':n*n-len(edges)},
        omitted_contributions={'conservation_defined':False,'error_nodes':[n.node_id for n in nodes if n.kind=='error'],
            'error_nodes_preserved':layers*len(tokens),'all_nonzero_upstream_edges_preserved':True,'reason':'No inferred missing mass; exact upstream error nodes retained'})
