from html import escape
from pathlib import Path
import json
from pipelines_v2.operations.graphs import AttributionGraph


def render_trace(payload):
    graph=AttributionGraph.from_dict(payload['graph'])
    if graph.semantics!='replacement_flow':raise ValueError('Trace renderer requires replacement-flow semantics')
    nodes={n.node_id:n for n in graph.nodes}
    text=lambda value:escape(str(value),quote=True)
    rows=''.join(f'<tr><td>{text(nodes[e.source].label)}</td><td>{text(nodes[e.target].label)}</td><td>{e.score:.6g}</td><td>{text(e.units)}</td><td>{text(e.evidence_ids[0])}</td></tr>' for e in graph.edges)
    coordinates=''.join(f'<tr><td>{text(n.label)}</td><td>{text(n.kind)}</td><td><details><summary>Inspect</summary><pre>{text(n.coordinate)}</pre></details></td></tr>' for n in graph.nodes)
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Qwen3 PLT trace</title><style>
body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;background:#faf9f5;color:#28343b}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #dedbd2}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.notice{{border-left:4px solid #a66431;padding:14px;background:#fff3df}}.scroll{{max-height:600px;overflow:auto}}</style>
<h1>Qwen3 per-layer transcoder trace</h1><p class="notice">PLT replacement-flow graph. Scientific verdict: not tested. Original-model route remains unvalidated. This is not a live CLT demonstration.</p>
<p>{len(graph.nodes)} nodes, {len(graph.edges)} signed edges; every upstream error node and nonzero edge retained. Rows in the upstream matrix are targets and columns are sources.</p>
<h2>Input and target</h2><pre>{text(graph.target)}</pre>
<h2>Replacement diagnostics</h2><p>Same traced prompt only. Graph influence ratios and the no-error-correction forward diagnostic are different measurements; neither is held-out fidelity qualification.</p><pre>{text(payload['fidelity'])}</pre>
<h2>Original-model intervention status</h2><pre>{text(payload['original_model_validation'])}</pre>
<h2>Signed replacement flows</h2><p>Attention patterns and normalization scales are frozen for attribution. These weights are not native causal intervention effects.</p><div class="scroll"><table><tr><th>Source</th><th>Target</th><th>Weight</th><th>Units</th><th>Raw coordinate</th></tr>{rows}</table></div>
<h2>All nodes, including errors</h2><div class="scroll"><table><tr><th>Node</th><th>Type</th><th>Coordinate</th></tr>{coordinates}</table></div>
<h2>Coverage, identities and compute</h2><pre>{text({'graph_id':graph.graph_id,'provenance':graph.provenance,'pruning':graph.pruning,'omitted_contributions':graph.omitted_contributions,'compute':payload['compute']})}</pre></html>'''


def trace_report_transform(*,trace,output_dir):
    payload=trace.result();root=Path(output_dir).expanduser();root.mkdir(parents=True,exist_ok=True)
    graph=AttributionGraph.from_dict(payload['graph'])
    (root/'graph.json').write_text(json.dumps(graph.to_dict(),indent=2))
    (root/'graph.html').write_text(render_trace(payload))
    return {'payload':{'kind':'circuit_trace_report','graph_id':graph.graph_id,'source_artifact_id':trace.id,
        'graph_html':str((root/'graph.html').resolve()),'graph_json':str((root/'graph.json').resolve()),
        'node_count':len(graph.nodes),'edge_count':len(graph.edges),'dictionary_type':payload['dictionary_type'],
        'original_model_validation':payload['original_model_validation'],'scientific_verdict':'not_tested'}}
