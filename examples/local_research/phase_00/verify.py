"""Independent pretrained forward oracle, run inside the existing scratch supervisor."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time
from examples.local_research.phase_00.fixtures import LAYER, TOKENS, PAIRS


def oracle(output):
    import socket
    # Python, subprocess Python, native network, and cloud imports must be fenced.
    blocked = {}
    try:
        socket.create_connection(('1.1.1.1', 443), timeout=1)
    except OSError:
        blocked['python_network'] = True
    child = subprocess.run([sys.executable, '-c', "import socket;socket.create_connection(('1.1.1.1',443),timeout=1)"], capture_output=True)
    blocked['child_python_network'] = child.returncode != 0
    child = subprocess.run(['/usr/bin/curl', '--connect-timeout', '1', 'https://1.1.1.1'], capture_output=True)
    blocked['native_network'] = child.returncode != 0
    try:
        __import__('modal')
    except PermissionError:
        blocked['cloud_import'] = True
    if len(blocked) != 4 or not all(blocked.values()):
        raise AssertionError(f'Offline fences failed: {blocked}')
    import torch
    from examples.local_research.phase_00.workflow import build_workflow
    workflow = build_workflow()
    specs = {step.name: step.spec for step in workflow.steps}
    engine = specs['capture'].engine
    cold_start = time.monotonic()
    with engine.open_session() as session:
        cold_seconds = time.monotonic() - cold_start
        assert session.tokenizer.encode(' Paris', add_special_tokens=False) == [TOKENS[0]]
        assert session.tokenizer.encode(' London', add_special_tokens=False) == [TOKENS[1]]
        warm_start = time.monotonic()
        capture = engine.capture(specs['capture'], session=session)
        bank = capture.features['block5']
        all_rows = {'baseline': engine.generate(specs['baseline'], session=session).rows}
        for arm in ('identity', 'zero', *PAIRS):
            spec = specs[arm]
            if arm in PAIRS:
                spec = replace(spec, patch=replace(spec.patch, activation_bank=SimpleNamespace(result=lambda: {**bank, 'kind': 'activation_bank_result'})))
            all_rows[arm] = engine.intervene(spec, session=session).rows
        warm_seconds = time.monotonic() - warm_start
        examples = {e.key: e for e in specs['capture'].dataset.examples}
        errors, capture_errors, replacement_errors = [], [], []
        # Independent raw HF hooks: does not call TorchSession.forward or residual_edit.
        for key, example in examples.items():
            ids, mask = session.encode([example.prompt])
            recorded = []
            handle = session.model.transformer.h[LAYER].register_forward_hook(lambda m,a,o: recorded.append((o[0] if isinstance(o, tuple) else o).detach().clone()))
            with torch.no_grad():
                session.model(input_ids=ids, attention_mask=mask, use_cache=False)
            handle.remove()
            expected = torch.as_tensor(bank['layers'][str(LAYER)][key]['values'])
            capture_errors.append(float((recorded[0][0,-1] - expected[0]).abs().max()))
        for arm, mapping in PAIRS.items():
            indexed = {r['example_key']: r for r in all_rows[arm]}
            for target, donor in mapping.items():
                donor_values = torch.as_tensor(bank['layers'][str(LAYER)][donor]['values'][0])
                ids, mask = session.encode([examples[target].prompt])
                def patch(module, args, output):
                    value = output[0] if isinstance(output, tuple) else output
                    value = value.clone()
                    value[:, -1, :] = donor_values
                    replacement_errors.append(float((value[0,-1] - donor_values).abs().max()))
                    return (value, *output[1:]) if isinstance(output, tuple) else value
                handle = session.model.transformer.h[LAYER].register_forward_hook(patch)
                try:
                    with torch.no_grad():
                        logits = session.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                    direct = torch.log_softmax(logits[0,-1], dim=-1)
                finally:
                    handle.remove()
                errors.extend(abs(float(direct[t]) - indexed[target]['target_logprobs'][str(t)]) for t in TOKENS)
        assert max(errors + capture_errors + replacement_errors) <= 1e-5
        baseline = {r['example_key']: r for r in all_rows['baseline']}
        assert all(r['target_logprobs'] == baseline[r['example_key']]['target_logprobs'] and r['generated_token_ids'] == baseline[r['example_key']]['generated_token_ids'] for r in all_rows['identity'])
    Path(output).write_text(json.dumps({'cold_load_seconds': cold_seconds, 'warm_experiment_seconds': warm_seconds,
        'warm_target_seconds': 60, 'warm_target_met': warm_seconds < 60, 'max_capture_error': max(capture_errors),
        'max_interchange_target_logprob_error': max(errors), 'max_full_state_replacement_error': max(replacement_errors),
        'offline_sentinels': blocked, 'rows': all_rows}, indent=2))


def main():
    from pipelines_v2.runtime.scratch_supervision import supervise, ScratchLimits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.worker:
        oracle(output)
    else:
        if output.parent.exists():
            raise ValueError('Choose a fresh output directory for auditable verification')
        env = {k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL', 'API_KEY', 'DATABASE_URL', 'PGHOST'))}
        diagnostics = supervise([sys.executable, '-m', 'examples.local_research.phase_00.verify', '--worker', '--output', str(output)],
            attempt=output.parent, output_root=output.parent, limits=ScratchLimits(max_examples=8, max_input_tokens=128, max_generated_tokens=16), env=env)
        print(json.dumps({'output':str(output), 'diagnostics':diagnostics}, indent=2))


if __name__ == '__main__':
    main()
