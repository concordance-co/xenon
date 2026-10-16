"""Standalone, offline evidence panels. No scripts, external URLs or raw tensor reads."""
import base64
from html import escape
import json
from pathlib import Path


def _text(value):
    return escape(str(value), quote=True)


def _details(label, value):
    return '<details><summary>'+_text(label)+'</summary><pre>'+_text(json.dumps(value, indent=2, ensure_ascii=False))+'</pre></details>'


def _table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+_text(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+_text(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def _gallery(payload):
    cells = []
    for cell in payload['cells']:
        preview = cell.get('preview', '')
        valid = False
        if isinstance(preview, str) and preview.startswith('data:image/jpeg;base64,'):
            try:
                raw = base64.b64decode(preview.split(',', 1)[1], validate=True)
                valid = raw.startswith(b'\xff\xd8\xff')
            except ValueError:
                pass
        image = '<img loading="lazy" alt="'+_text(cell.get('arm', 'Image'))+'" src="'+_text(preview)+'">' if valid else '<p class="muted">Preview unavailable in this bounded summary.</p>'
        cells.append('<article><h3>'+_text(cell.get('arm', 'Image'))+'</h3>'+image+_details('Measurements and source', {k:v for k,v in cell.items() if k != 'preview'})+'</article>')
    return '<h2>Paired image gallery</h2><p>Inspect visual changes alongside the measurements. Image differences alone establish no semantic control.</p><div class="gallery">'+''.join(cells)+'</div>'+_details('Gallery checks and limits', {k:v for k,v in payload.items() if k != 'cells'})


def _dossiers(payload):
    body = '<h2>Neuron microscope</h2>'+_details('Summary', payload.get('summary', {}))
    for dossier in payload['dossiers']:
        observations = dossier.get('observations', {})
        body += '<article><h3>'+_text(dossier['neuron_id'].split(':')[-2:])+'</h3><p class="badge">'+_text(dossier.get('explanation_status', 'unexplained'))+'</p>'
        body += '<p>'+_text((dossier.get('explanation') or {}).get('text', 'No explanation proposed.'))+'</p>'
        for group in ('top', 'bottom', 'random'):
            rows = observations.get(group, ())
            if rows:
                body += '<h4>'+_text(group.title())+' contexts</h4>'+_table(('Text', 'Split', 'Gate pre', 'Gate', 'Up', 'Product'), [
                    (r.get('text'), r.get('split'), *(f'{r.get("values", {}).get(k):.5g}' if isinstance(r.get('values', {}).get(k), (int, float)) else 'unavailable' for k in ('gate_preactivation', 'gate_activation', 'up_projection', 'mlp_product'))) for r in rows])
        body += _details('Intervention evidence', dossier.get('intervention', {}))+_details('Exact coordinates and source contexts', observations)+'</article>'
    return body


def _graph(payload):
    graph = payload['graph']
    nodes = graph.get('nodes', ())
    labels = {n['node_id']: n.get('label', n['node_id']) for n in nodes}
    body = '<h2>Typed attribution graph</h2><p>Read the edge type and intervention definition before interpreting a score. Direct effects and replacement-model flow have different meanings.</p>'
    body += '<p class="badge">'+_text(graph.get('semantics', 'unspecified semantics'))+'</p>'
    body += _table(('Source', 'Target', 'Edge semantics', 'Intervention', 'Score / units', 'Evidence IDs'), [
        (labels.get(e.get('source'), e.get('source')), labels.get(e.get('target'), e.get('target')), e.get('semantics', e.get('kind', 'unspecified')), e.get('intervention', 'unvalidated'), str(e.get('score'))+' '+str(e.get('units', '')), ', '.join(e.get('evidence_ids', ()))) for e in graph.get('edges', ())])
    body += _details('Measured controls and examples', {'evidence': graph.get('evidence', ()), 'examples': payload.get('examples', ())})
    body += _details('Node coordinates', nodes)+_details('Graph assumptions and provenance', {k:v for k,v in graph.items() if k not in ('nodes', 'edges', 'evidence')})
    body += _details('Validation status, fidelity and measured resources', {key:payload[key] for key in
        ('engineering_status','scientific_verdict','fidelity','original_model_validation','compute') if key in payload})
    return body


def _attribution(payload):
    request = payload.get('request', {})
    scores = {node['node_id']:node['score'] for node in payload.get('nodes', ())}
    body = '<h2>Official attribution</h2><p>'+_text(request.get('model', {}).get('name', 'unknown model'))+' · '+_text(request.get('dtype', 'unknown precision'))+' · '+_text(payload.get('reproduction_label', 'unqualified'))+'</p>'
    body += '<p class="badge">Rankings are not causal edge contributions. Scientific verdict: '+_text(payload.get('scientific_verdict', 'not_tested'))+'</p>'
    body += '<h3>Top 20 ranked nodes</h3>'+_table(('Rank', 'Node', 'Score'), [(i+1,n,scores.get(n)) for i,n in enumerate(payload.get('ranking', ())[:20])])
    body += '<h3>Official sparsity curve</h3>'+_table(('Fraction', 'Requested nodes', 'CPR', 'Accuracy', 'Retained edge count'), [(p.get('fraction'),p.get('requested_k'),p.get('cpr'),p.get('accuracy'),p.get('weighted_edge_count')) for p in payload.get('points', ())])
    body += _details('Metric definitions, request and limits', {k:payload.get(k) for k in ('evaluator','metric_definitions','request','aggregates','limitations','uncertainty','resume_semantics')})
    body += _details('Raw producer output inventory', {name:{k:v for k,v in item.items() if k != 'data'} for name,item in payload.get('raw_outputs', {}).items()})
    return body


def _crosscoder(payload):
    body = '<h2>Paired-model feature edit</h2><p>'+_text(payload.get('scope', 'Plumbing check'))+'</p>'
    body += '<p>Feature '+_text(payload.get('feature_index'))+' · amount '+_text(payload.get('amount'))+' · '+_text(payload.get('coordinate'))+'</p>'
    body += _table(('Model', 'No-op max logit change', 'Feature edit max logit change', 'Reconstruction max logit change', 'Direct edit oracle error'), [
        (row.get('model', {}).get('repository'),
         *(row.get('arms', {}).get(arm, {}).get('max_abs_logit_delta') for arm in ('noop','delta','reconstruct')),
         row.get('direct_delta_logit_error')) for row in payload.get('rows', ())])
    body += '<p class="muted">Reconstruction and feature addition are separate interventions. These values verify execution, not feature meaning.</p>'
    return body+_details('Dictionary and model identities', payload)


def render_workbench(artifacts, *, run=None, comparison=None, title='Evidence workbench'):
    """Render only results already admitted by EvidenceReader's bounded reads."""
    panels = []
    for index, artifact in enumerate(artifacts):
        payload = artifact.get('result', {})
        if artifact.get('kind') == 'crosscoder_feature_splice':
            body = _crosscoder(payload)
        elif payload.get('kind') == 'external_attribution':
            body = _attribution(payload)
        elif 'dossiers' in payload:
            body = _dossiers(payload)
        elif 'cells' in payload:
            body = _gallery(payload)
        elif 'graph' in payload:
            body = _graph(payload)
        else:
            body = '<h2>Evidence summary</h2>'+_details('Result', payload)
        status = 'Identity resolved' if artifact.get('identity_complete') else 'Identity unresolved'
        body = '<section id="artifact-'+str(index)+'"><p class="eyebrow">'+_text(artifact['kind'])+'</p><p class="source">'+_text(artifact['artifact_id'])+'</p><p class="badge">'+status+' · '+_text(artifact.get('completion_declared', 'unknown'))+' · summary checksum '+('verified' if artifact.get('summary_checksum_verified') else 'unverified')+'</p>'+body
        body += _details('Provenance, qualification and source artifacts', {k:v for k,v in artifact.items() if k != 'result'})+'</section>'
        panels.append(body)
    nav = ''.join('<a href="#artifact-'+str(i)+'">'+_text(a['artifact_id'])+'</a>' for i,a in enumerate(artifacts))
    extra = ''
    if run:
        extra += '<section id="run"><h2>Run and replay</h2>'+_details('Workflow, step failures, exact replay config', run)+'</section>'
    if comparison:
        extra += '<section id="compare"><h2>Comparison: '+('compatible definitions' if comparison['comparable'] else 'not comparable')+'</h2>'+_table(('Artifact', comparison['metric'], 'Compatibility limits'), [(r['artifact_id'], r['value'], '; '.join(r['reasons']) or 'same declared semantics') for r in comparison['rows']])+_details('Definition and provenance differences', comparison)+'</section>'
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'"><title>'''+_text(title)+'''</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f5f3ed;color:#232d29;font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header,main{max-width:1320px;margin:auto;padding:32px}header{border-bottom:1px solid #bfc8be}h1{font:48px/1.08 Georgia,serif;letter-spacing:-1.5px;margin:12px 0}h2{font:30px/1.2 Georgia,serif}h3{margin:0 0 14px;font-size:18px}.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#32634c}.muted{color:#626c65}.badge{display:inline-block;background:#e8e5d4;padding:4px 10px;border-left:3px solid #a48635;font-size:13px}.source,pre{overflow-wrap:anywhere;white-space:pre-wrap;font:12px/1.6 ui-monospace,monospace}nav{display:flex;flex-wrap:wrap;gap:8px}a{color:#245e47}nav a{padding:5px 9px;border:1px solid #c7cec5;border-radius:3px;font-size:12px}section{padding:28px 0 40px;border-bottom:1px solid #bfc8be}.gallery{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px}article{background:#fffefb;border:1px solid #d5d9d0;padding:18px;margin:14px 0}img{display:block;width:100%;height:auto}details{margin:12px 0}summary{cursor:pointer;color:#245e47;font-weight:600}pre{background:#eceee7;padding:16px;max-height:480px;overflow:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;text-align:left;vertical-align:top;border-bottom:1px solid #d7dbd2}th{color:#52685c;font-weight:600}td:first-child{min-width:140px}@media(max-width:900px){.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:560px){header,main{padding:20px}h1{font-size:36px}.gallery{grid-template-columns:1fr}}
</style></head><body><header><p class="eyebrow">Xenon / research evidence</p><h1>'''+_text(title)+'''</h1><p class="muted">Stored observations, explicit provenance, inspectable limits. This offline view does not execute experiments.</p><nav>'''+nav+'''</nav></header><main>'''+extra+''.join(panels)+'</main></body></html>'


def write_workbench(path, artifacts, **kwargs):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_workbench(artifacts, **kwargs), encoding='utf-8')
    return path.resolve()
