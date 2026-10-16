from pipelines_v2.evidence.report import render_workbench


def test_panels_escape_untrusted_values_and_do_not_fetch_external_images():
    records = [
        {'artifact_id':'<script>bad()</script>', 'kind':'transform', 'result':{'cells':[{'arm':'<script>x</script>', 'preview':'https://tracker.invalid/image'}]}},
        {'artifact_id':'dossier', 'kind':'transform', 'result':{'dossiers':[{'neuron_id':'model:0:2','observations':{},'explanation':{'text':'<script>bad()</script>'}}]}},
        {'artifact_id':'graph', 'kind':'attribution_graph', 'result':{'graph':{'nodes':[{'node_id':'n','label':'<script>bad()</script>'}], 'edges':[{'source':'n','target':'n','score':1,'kind':'replacement_flow'}]}}},
    ]
    html = render_workbench(records)
    assert '<script>' not in html
    assert '<img' not in html
    assert 'https://tracker.invalid' not in html
    assert 'Paired image gallery' in html and 'Neuron microscope' in html and 'Typed attribution graph' in html
    assert 'replacement_flow' in html
    assert "default-src 'none'" in html
