"""Validate local recognition artifacts and render a bounded stored-image gallery."""
import argparse,base64,html,io,json
from pathlib import Path
from pipelines_v2.storage.artifacts import ArtifactManifest, artifact_from_manifest
from pipelines_v2.storage.finalization import validate_completion
from pipelines_v2.storage.local import LocalArtifactStore


def inspect(run_json, output):
    from PIL import Image
    run=json.loads(Path(run_json).read_text());root=Path(run['execution_profile']['artifact_root'])/run['run_id'];store=LocalArtifactStore(root)
    artifacts={}; provenance={}
    for name,record in run['steps'].items():
        manifest=ArtifactManifest.from_dict(json.loads((root/record['artifact_id']/'manifest.json').read_text()))
        validate_completion(manifest,store=store);artifacts[name]=artifact_from_manifest(manifest,store=store)
        provenance[name]={'artifact_id':manifest.artifact_id,'C03_verified':True,'C01_complete':manifest.provenance().identity_complete,
            'unresolved':manifest.provenance().to_dict()['unresolved']}
    if not provenance['recognition']['C01_complete']:raise ValueError('Recognition execution identity incomplete')
    artifact=artifacts['recognition'];payload=artifact.result()
    if payload['kind']!='recognition_result' or not 1<=len(payload['rows'])<=4:raise ValueError('Unexpected bounded recognition result')
    refs=artifact.manifest().storage_refs['features']['input_images']
    if refs['bytes']>16*1024*1024:raise ValueError('Gallery transfer budget exceeded')
    images=artifact.load_feature('input_images')['records']
    parts=['<!doctype html><meta charset="utf-8"><title>ViT recognition microscope</title>',
        '<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:20px;color:#17212b}article{border-top:1px solid #bbb;margin-top:24px}img{max-width:384px}pre{white-space:pre-wrap;overflow-wrap:anywhere}td,th{padding:6px;text-align:left}</style>',
        '<h1>ViT recognition microscope</h1><p>Engineering plumbing: two synthetic fixtures, no scientific evaluation or image generation.</p>',
        '<p>Class token 0 has no image box. Patch tokens 1–196 use row-major 14×14 locations. Boxes show patch origins; attention mixes global image context.</p>']
    for row in payload['rows']:
        array=images[row['example_key']]['values'];buffer=io.BytesIO();Image.fromarray(array).save(buffer,format='PNG')
        data=base64.b64encode(buffer.getvalue()).decode()
        parts += ['<article><h2>'+html.escape(row['example_key'])+'</h2>',f'<img alt="Synthetic input fixture" src="data:image/png;base64,{data}">',
            '<p>Image bytes SHA256: '+html.escape(row['input']['content_digest'])+'</p>',
            '<p>Baseline class label (not an accuracy judgment): '+html.escape(row['predicted_label'])+'. Identity is exact. Maximum logit change: '+str(row['max_logit_change'])+'</p>',
            '<h3>Selected coordinates</h3><pre>'+html.escape(json.dumps({'geometry':row['geometry'],'channels':payload['site']['channels'],
                'baseline':row['capture'],'edited':row['edited_capture']},indent=2))+'</pre></article>']
    parts += ['<h2>Evidence</h2><pre>'+html.escape(json.dumps({'run_id':run['run_id'],'provenance':provenance,'limits':payload['limitations']},indent=2))+'</pre>']
    Path(output).write_text('\n'.join(parts)+'\n')
    diagnostics=json.loads((root.parent.parent/'scratch'/run['run_id']/'diagnostics.json').read_text())
    return {'run_id':run['run_id'],'workflow_hash':run['workflow_hash'],'summary':payload['summary'],'provenance':provenance,
        'diagnostics':diagnostics,'rows':[{'example_key':r['example_key'],'identity_max_error':r['identity_max_error'],'max_logit_change':r['max_logit_change']} for r in payload['rows']]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-json',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(inspect(a.run_json,a.output),indent=2))
