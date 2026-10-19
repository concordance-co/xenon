"""Read and validate local artifacts, then render bounded neuron dossiers."""
import argparse
import json
from pathlib import Path
from pipelines_v2.storage.artifacts import ArtifactManifest, artifact_from_manifest
from pipelines_v2.storage.finalization import validate_completion
from pipelines_v2.storage.local import LocalArtifactStore


def inspect(run_json, output):
    run = json.loads(Path(run_json).read_text())
    root = Path(run['execution_profile']['artifact_root']) / run['run_id']
    store = LocalArtifactStore(root)
    artifacts, provenance = {}, {}
    for name, record in run['steps'].items():
        manifest = ArtifactManifest.from_dict(json.loads((root / record['artifact_id'] / 'manifest.json').read_text()))
        validate_completion(manifest, store=store)
        artifacts[name] = artifact_from_manifest(manifest, store=store)
        provenance[name] = {'artifact_id': manifest.artifact_id, 'C03_verified': True,
            'C01_complete': manifest.provenance().identity_complete,
            'unresolved': manifest.provenance().to_dict()['unresolved']}
    for name in ('capture', 'baseline', 'identity', 'edit'):
        if not provenance[name]['C01_complete']:
            raise ValueError(f'Missing execution identity for {name}')
    summary, dossiers = artifacts['summary'].result(), artifacts['dossiers'].result()
    if summary['kind'] != 'neuron_summary_result' or dossiers['kind'] != 'neuron_dossier_result':
        raise ValueError('Unexpected transform payload envelope')
    if summary['summary']['observation_count'] != 12 or dossiers['summary']['neuron_count'] != 3 or not dossiers['summary']['identity_exact']:
        raise ValueError('Unexpected fixture coverage/control result')
    diagnostics = json.loads((root.parent.parent / 'scratch' / run['run_id'] / 'diagnostics.json').read_text())
    lines = ['# Native neuron microscope', '', f"Run `{run['run_id']}`; ReportSpec `{artifacts['report'].id}`.", '',
        'Engineering plumbing only. Four authored prompts; Qwen3-0.6B layer 0, neurons 0–2. Plain text, no chat template or thinking wrapper.', '',
        'All four raw signed quantities are retained; top/bottom are ranked by signed MLP product. Manual candidates are unverified; scientific verdict: not tested.', '',
        f"Identity is exact. Runtime {diagnostics['elapsed_seconds']:.2f}s; sampled peak tree RSS {diagnostics['observed_peak_tree_rss_bytes']/1024**3:.2f} GiB."]
    for dossier in dossiers['dossiers']:
        obs = dossier['observations']
        lines += ['', f"## Neuron {obs['random'][0]['source']['neuron']}", '', f"Coordinate `{dossier['neuron_id']}`.", '',
            f"{dossier['explanation_status']}: " + (dossier['explanation']['text'] if dossier['explanation'] else 'No candidate; retained in denominator.'), '',
            '| Extreme | Original text | Token | Gate pre | SiLU gate | Up | Product |', '|---|---|---:|---:|---:|---:|---:|']
        for group in ('top', 'bottom'):
            for row in obs[group]:
                v = row['values']
                text = row['text'].replace('|', '&#124;').replace('\n', ' ')
                lines.append(f"| {group} | {text} | {row['token_window']['selected_position']} | {v['gate_preactivation']:.5f} | {v['gate_activation']:.5f} | {v['up_projection']:.5f} | {v['mlp_product']:.5f} |")
        lines += ['', 'Intervention: ' + json.dumps(dossier['intervention'], sort_keys=True)]
    lines += ['', '## Evidence and limits', '', f"Capture `{artifacts['capture'].id}`; summary `{artifacts['summary'].id}`; dossiers `{artifacts['dossiers'].id}`.", '']
    lines += ['- ' + limit for limit in dossiers['limitations']]
    lines += [f"- {name}: C03 verified; C01 complete={record['C01_complete']}, unresolved={record['unresolved']}." for name, record in provenance.items()]
    Path(output).write_text('\n'.join(lines) + '\n')
    return {'run_id': run['run_id'], 'workflow_hash': run['workflow_hash'], 'summary': dossiers['summary'],
        'diagnostics': diagnostics, 'provenance': provenance}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-json', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.run_json, args.output), indent=2))
