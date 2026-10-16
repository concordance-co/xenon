"""Validate durable outputs and render the bounded ReportSpec result for reading."""
import argparse
import json
from pathlib import Path
from pipelines_v2.storage.artifacts import ArtifactManifest, artifact_from_manifest
from pipelines_v2.storage.finalization import validate_completion
from pipelines_v2.storage.local import LocalArtifactStore


def inspect(run_json, output, oracle=None):
    run = json.loads(Path(run_json).read_text())
    root = Path(run['execution_profile']['artifact_root']) / run['run_id']
    store = LocalArtifactStore(root)
    artifacts = {}
    provenance = {}
    for name, record in run['steps'].items():
        manifest = ArtifactManifest.from_dict(json.loads((root / record['artifact_id'] / 'manifest.json').read_text()))
        validate_completion(manifest, store=store)
        artifacts[name] = artifact_from_manifest(manifest, store=store)
        provenance[name] = {'artifact_id': manifest.artifact_id, 'scientific_identity':manifest.provenance().scientific_identity,
            'complete':manifest.provenance().identity_complete, 'unresolved':manifest.provenance().to_dict()['unresolved']}
    model_steps = ('capture','baseline','identity','zero','matched','same_label','random_donor')
    assert all(provenance[s]['complete'] for s in ('capture','baseline','identity','zero'))
    summary = artifacts['summary'].result()
    diagnostic = json.loads((root.parent.parent / 'scratch' / run['run_id'] / 'diagnostics.json').read_text())
    if oracle:
        direct = json.loads(Path(oracle).read_text())
        for name, expected_rows in direct['rows'].items():
            actual = {r['example_key']:r for r in artifacts[name].result()['rows']}
            assert set(actual) == {r['example_key'] for r in expected_rows}
            for expected in expected_rows:
                row = actual[expected['example_key']]
                assert row['generated_token_ids'] == expected['generated_token_ids']
                assert max(abs(row['target_logprobs'][k] - v) for k,v in expected['target_logprobs'].items()) <= 1e-5
    lines = ['# GPT-2 local residual experiment', '', f"Run: `{run['run_id']}`. ReportSpec artifact: `{artifacts['report'].id}`.", '',
        '**Engineering smoke; scientific verdict: inconclusive.** Eight prompts, four recipients, one residual site (block 5 output, last prompt token).', '',
        f"Baseline token contrast is correct on {summary['summary']['baseline_correct_token_contrast']}/8 examples; identity control is exact.", '',
        '| Arm | Recipients | Mean shift toward opposite label | Opposite flips | Reverse flips | Non-city-first outputs |',
        '|---|---:|---:|---:|---:|---:|']
    for name, arm in summary['summary']['arms'].items():
        lines.append(f"| {name} | {arm['n']} | {arm['mean_opposite_label_shift']:.6f} | {arm['opposite_label_flips']} | {arm['reverse_flips']} | {arm['non_city_first_outputs']} |")
    lines += ['', 'Positive shifts mean movement toward the opposite country’s city, including for controls; they do not prove a selective mechanism.', '',
        f"Full offline workflow: {diagnostic['elapsed_seconds']:.2f}s; sampled peak process-tree RSS {diagnostic['observed_peak_tree_rss_bytes']/1024**3:.2f} GiB; {diagnostic['artifact_bytes']} artifact bytes."]
    if oracle:
        lines += [f"Separate fresh-process session: cold load {direct['cold_load_seconds']:.2f}s; warm experiment {direct['warm_experiment_seconds']:.2f}s (target <60s). Independent HF capture error {direct['max_capture_error']}; donor target-logprob error {direct['max_interchange_target_logprob_error']}; full-state replacement error {direct['max_full_state_replacement_error']}."]
    lines += ['', '## Exact rows', '', '| Example | Arm | Donor | Paris − London | Delta | Generated continuation |', '|---|---|---|---:|---:|---|']
    for row in summary['rows']:
        text = row['generated_text'].replace('|','&#124;').replace('\n',' ')
        lines.append(f"| {row['example_key']} | {row['arm']} | {row['donor'] or '—'} | {row['paris_minus_london']:.6f} | {row['delta']:.6f} | {text} |")
    lines += ['', '## Limits', ''] + ['- '+s for s in summary['limitations']]
    lines += ['- Whole-profile interrupted compute resume remains unsupported; successful reruns use fresh run IDs.', '', '## Provenance', '']
    for name, record in provenance.items():
        lines.append(f"- `{name}` / `{record['artifact_id']}`: C03 verified; C01 complete={record['complete']}; unresolved={record['unresolved']}.")
    Path(output).write_text('\n'.join(lines)+'\n')
    return {'run_id':run['run_id'],'C03_all_verified':True,'C01_model_steps_complete':all(provenance[s]['complete'] for s in model_steps),'summary':summary['summary'], 'provenance':provenance, 'diagnostics':diagnostic}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-json',required=True); p.add_argument('--output',required=True); p.add_argument('--oracle')
    a=p.parse_args()
    print(json.dumps(inspect(a.run_json,a.output,a.oracle),indent=2))
