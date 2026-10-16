"""Bounded standalone HTML, with all values escaped and no activation-bank reads."""
from html import escape
from pathlib import Path
from .contracts import AttributionGraph


def render_graph(graph, *, examples=()):
    if isinstance(graph,dict):graph=AttributionGraph.from_dict(graph)
    def text(value):return escape(str(value),quote=True)
    nodes={n.node_id:n for n in graph.nodes}
    position={node.node_id:(160 if node.kind!='output' else 600,65+65*i) for i,node in enumerate(graph.nodes)}
    height=max(220,80+65*len(nodes))
    lines=[]
    for edge in graph.edges:
        x,y=position[edge.source];xx,yy=position[edge.target]
        lines.append(f'<line x1="{x}" y1="{y}" x2="{xx}" y2="{yy}" stroke="#82969f"/>')
    for node in graph.nodes:
        x,y=position[node.node_id]
        lines.append(f'<circle cx="{x}" cy="{y}" r="8" fill="#14645c"/><text x="{x+15}" y="{y+5}">{text(node.label)}</text>')
    rows=[]
    for edge in graph.edges:
        rows.append(f'<tr><td>{text(nodes[edge.source].label)}</td><td>{text(nodes[edge.target].label)}</td><td>{text(edge.intervention)}</td><td>{edge.score:.6g}</td><td><a href="#e-{edge.evidence_ids[0]}">Measured controls</a></td></tr>')
    evidence=[]
    for item in graph.evidence:
        evidence.append(f'<tr id="e-{item.evidence_id}"><td>{text(item.example_key)}</td><td>{text(nodes[item.node_id].label)}</td><td>{text(item.arm)}</td><td>{item.baseline:.6g}</td><td>{item.observed:.6g}</td><td>{item.observed-item.baseline:.6g}</td><td>{item.collateral["next_token_kl"]:.6g}</td><td>{text(item.engineering_status)}</td></tr>')
    annotations=''.join(f'<tr><td>{text(a["upstream_node_id"])}</td><td>{a["score"]:.6g}</td><td>{text(a.get("native_node_id") or "unmatched; no native basis proof")}</td></tr>' for a in graph.annotations)
    details=''.join(f'<details><summary>{text(n.label)} coordinate</summary><pre>{text(n.coordinate)}</pre></details>' for n in graph.nodes)
    example_details=''.join(f'<details><summary>Example {text(e["key"])}</summary><pre>{text(e)}</pre></details>' for e in examples)
    candidates=''.join(f'<details><summary>Embedding candidates — {text(c["example_key"])}</summary><pre>{text(c)}</pre></details>' for c in graph.candidates)
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Native route plumbing</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#203038;background:#f7faf9}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ccd7d3}}svg{{background:white;width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.badge{{color:#80530b}}a{{color:#14645c}}</style>
<h1>Native route plumbing</h1><p class="badge">Scientific verdict: not tested. Measured direct interventions; no validated circuit claim.</p>
<p>Edges show mean held-out zero-replacement effects on next-token correct-minus-foil logit margin. They are nonadditive. Other arms are below. Layout does not imply head-to-head routing.</p>
<svg viewBox="0 0 900 {height}" role="img" aria-label="Selected native sites and measured output">{''.join(lines)}</svg>
<table><tr><th>Source</th><th>Target</th><th>Intervention</th><th>Logit delta</th><th>Evidence</th></tr>{''.join(rows)}</table>
<h2>Held-out controls</h2><table><tr><th>Example</th><th>Site</th><th>Arm</th><th>Before</th><th>After</th><th>Delta</th><th>Next-token KL</th><th>Engineering</th></tr>{''.join(evidence)}</table>
<h2>Examples</h2>{example_details}
<h2>Coordinates and candidate gradients</h2><p>Finite-grid IG residuals are reported; convergence is not established. Candidate scores are not causal route evidence.</p>{details}{candidates}
<h2>External ranking annotations</h2><p>Separate upstream namespace; scores are not causal edge weights.</p><table><tr><th>Upstream node</th><th>Ranking score</th><th>Native join</th></tr>{annotations}</table>
<h2>Scope and provenance</h2><pre>{text({'graph_id':graph.graph_id,'provenance':graph.provenance,'algorithm':graph.algorithm,'assumptions':graph.assumptions,'approximations':graph.approximations,'omitted_contributions':graph.omitted_contributions})}</pre></html>'''


def graph_report_transform(*, route, output_dir):
    """Public TransformBuilder adapter; retain the payload envelope explicitly."""
    payload=route.result()
    graph=AttributionGraph.from_dict(payload['graph'])
    root=Path(output_dir).expanduser();root.mkdir(parents=True,exist_ok=True)
    # Executor-level graphs have a request anchor; the rendered artifact adds an actual durable source reference.
    from dataclasses import replace
    if hasattr(route, 'manifest'):
        manifest=route.manifest()
        graph=replace(graph,provenance={**graph.provenance,'source_artifact_id':route.id,
            'source_scientific_identity':manifest.metadata.get('provenance_v1',{}).get('scientific_identity')})
    path=root/'graph.html';path.write_text(render_graph(graph,examples=payload.get('examples',())))
    return {'payload':{'kind':'native_route_report','graph_id':graph.graph_id,'graph_html':str(path.resolve()),
        'edge_count':len(graph.edges),'evidence_count':len(graph.evidence),'scientific_verdict':'not_tested','engineering_status':payload['engineering_status']}}
