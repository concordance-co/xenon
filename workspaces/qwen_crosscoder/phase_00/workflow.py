"""Canonical crosscoder workflow; XENON_CROSSCODER_CONFIG selects a JSON config."""
import json
import os
from pathlib import Path
from pipelines_v2.representation.crosscoder.workflow import dataset_from_config,workflow_from_config,runner_specs_from_config

PROMPTS=('The capital of France is','A blue bird flew over the','Two plus three equals','The opposite of hot is')


def configuration():
    path=os.environ.get('XENON_CROSSCODER_CONFIG')
    if path:return json.loads(Path(path).read_text())
    return {'model_paths':[os.environ['XENON_CROSSCODER_BASE'],os.environ['XENON_CROSSCODER_INSTRUCT']],
        'reference_path':os.environ['XENON_CROSSCODER_REFERENCE'],'prompts':list(PROMPTS),
        'output':os.environ.get('XENON_CROSSCODER_OUTPUT','/private/tmp/xenon-crosscoder-plumbing'),
        'checkpoint_step':10,'limits':{'max_examples':4,'max_tokens_per_example':128,'max_total_tokens':512,
            'max_activation_bytes':64*1024**2,'max_output_bytes':64*1024**2,'timeout_seconds':300}}


def build_dataset():return dataset_from_config(configuration())
def build_workflow(dataset=None):return workflow_from_config(configuration(),dataset)
def build_runner_specs():return runner_specs_from_config(configuration())
