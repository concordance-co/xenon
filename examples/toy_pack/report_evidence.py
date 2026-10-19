"""Record a negative engineering result from a completed toy workflow run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipelines_v2.api import LocalArtifactStore, ReportSpec
from pipelines_v2.experiments import ExperimentPack
from pipelines_v2.runtime.local import LocalRunner
from pipelines_v2.storage import FileCatalog
from pipelines_v2.storage.artifacts import OperationArtifact

ROOT = Path(__file__).resolve().parent


def build_negative_report(run_id: str):
    catalog = FileCatalog(ROOT / '.catalog')
    steps = catalog.list_workflow_steps(run_id)
    step = next(item for item in steps if item.step_name == 'labels' and item.artifact_id)
    manifest = catalog.load_artifact(step.artifact_id)
    artifact = OperationArtifact(manifest, LocalArtifactStore(ROOT / 'artifacts'))
    # Exact fixture-sized payload only. The report validates sizes and selectors again.
    size = manifest.storage_refs['result'].get('bytes')
    if type(size) is not int or not 0 <= size <= 4096:
        raise ValueError('Expected a tiny fixture result')
    count = artifact.result()['mapped_count']
    if count != 2:
        raise ValueError('Toy fixture changed; review its assertion before reporting')
    pack = ExperimentPack.load(ROOT / 'pack.json')
    link = {'artifact_id':artifact.id, 'run_id':run_id,
            'scientific_identity':manifest.provenance().scientific_identity,
            'selector':{'domain':'result', 'pointer':'/mapped_count', 'expected':count}}
    record = dict(schema_version=1, evidence_id='three-row-assertion', pack_identity=pack.identity,
                  preset='smoke', question_id='Q-count', hypothesis_id=None,
                  engineering_status='tested', scientific_verdict='refuted', metric='mapped_count',
                  evaluator_digest=pack.to_dict()['metrics']['mapped_count']['evaluator_digest'],
                  split='fixture', split_link=None, control_arms={}, effect={'value':count,'link':0},
                  uncertainty={'value':None,'link':None,'reason':'Exact fixture count; no population estimate.'},
                  evidence_level='engineering', limitations=['Refutes only the deliberately false assertion that this fixture maps three rows. No model or paper claim.'], links=[link])
    runner = LocalRunner(artifacts=LocalArtifactStore(ROOT / 'artifacts'), catalog=catalog)
    report = runner.run(ReportSpec(inputs=(artifact,), evidence={'pack':pack.to_dict(),'records':[record]},
                                   output_dir=str(ROOT / 'reports'), template='experiment_evidence'))
    return {'run_id':run_id,'source_artifact_id':artifact.id, 'source_provenance':link['scientific_identity'],
            'report_artifact_id':report.id, 'report_path':report.uri, 'verdict':'refuted', 'engineering_status':'tested'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    print(json.dumps(build_negative_report(args.run_id), indent=2))
