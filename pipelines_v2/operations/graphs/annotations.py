"""C06 rankings stay in their own namespace unless observed tensors verify a join."""
from dataclasses import replace
from .contracts import digest
from pipelines_v2.operations.attribution.contracts import validate_result


def annotate_c06(graph, result, *, mapping=None, upstream_tensors=None, native_tensors=None, basis_evidence=None, limit=20):
    validate_result(result)
    if type(limit)is not int or not 1<=limit<=100:raise ValueError('Annotation display cap must be 1..100')
    mapping=mapping or {}
    nodes={n.node_id:n for n in graph.nodes}
    source_nodes={n['node_id']:n for n in result['nodes']}
    if len(set(mapping.values()))!=len(mapping):raise ValueError('Basis mapping must be one-to-one')
    proof=None
    if mapping:
        import numpy as np
        required={'model_identity','model_revision','source_loading_transforms','source_site_semantics','native_site_semantics','example_keys','token_positions','evidence_refs'}
        if not isinstance(basis_evidence,dict) or set(basis_evidence)!=required or any(not value for value in basis_evidence.values()):
            raise ValueError('A native join requires observed model, transforms, sites, tokens and evidence refs')
        if basis_evidence['model_identity']!=graph.model_identity or basis_evidence['model_revision']!=result['request']['model']['revision']:
            raise ValueError('Basis mapping model identity mismatch')
        if not isinstance(upstream_tensors,dict) or not isinstance(native_tensors,dict):raise ValueError('Native join requires matched observed tensors')
        measurements=[]
        for source,target in mapping.items():
            if source not in source_nodes or target not in nodes or nodes[target].kind!='native_site':raise ValueError('Unmatched basis coordinate')
            coordinate=nodes[target].coordinate
            import re
            head=re.fullmatch(r'a(\d+)\.h(\d+)',source)
            mlp=re.fullmatch(r'm(\d+)',source)
            if head:
                if coordinate['quantity']!='attention_head_output' or coordinate['layer']!=int(head[1]) or tuple(coordinate['selection']['heads'])!=(int(head[2]),):
                    raise ValueError('Upstream/native head scope mismatch')
            elif mlp:
                if coordinate['quantity']=='attention_head_output' or coordinate['layer']!=int(mlp[1]):
                    raise ValueError('Upstream/native MLP scope mismatch')
            else:raise ValueError('No native basis join for upstream input node')
            left=np.asarray(upstream_tensors[source]);right=np.asarray(native_tensors[target])
            if not left.size or left.shape!=right.shape or left.ndim<2 or left.shape[0]!=len(basis_evidence['example_keys']):raise ValueError('Basis tensor coverage/shape mismatch')
            if not np.isfinite(left).all() or not np.isfinite(right).all() or not np.allclose(left,right,rtol=1e-4,atol=1e-5):raise ValueError('Observed basis tensor values disagree')
            measurements.append({'upstream':source,'native':target,'shape':list(left.shape),'source_digest':digest(left.tolist()),'native_digest':digest(right.tolist()),'max_absolute_error':float(np.max(np.abs(left-right)))})
        proof={'evidence':basis_evidence,'measurements':measurements,'rtol':1e-4,'atol':1e-5,'scope':'observed aligned rows only; not universal coordinate equivalence'}
    selected=result['ranking'][:limit]
    if set(mapping)-set(selected):raise ValueError('Explicit mapped nodes must be inside displayed ranking cap')
    annotations=tuple({'semantics':'ranking_annotation','namespace':result['evaluator'],'upstream_node_id':key,
        'native_node_id':mapping.get(key),'score':source_nodes[key]['score'],'units':result['score_units'],
        'request_identity':result['request_identity'],'basis':result['request']['basis']} for key in selected)
    provenance={**graph.provenance,'c06':{'request_identity':result['request_identity'],'total':len(source_nodes),'displayed':len(selected),
        'matched':len(mapping),'unmatched_node_ids':[k for k in result['ranking'] if k not in mapping],'omitted_annotations':len(source_nodes)-len(selected),'basis_verification':proof}}
    return replace(graph,annotations=annotations,provenance=provenance)
