"""Import-light evidence CLI; also mounted under pipelines_v2.cli evidence."""
import argparse
import json


def configure(parser):
    parser.add_argument('--catalog-root', help='Existing local catalog mirror (defaults to ~/.xenon/pipelines_v2/catalog)')
    commands = parser.add_subparsers(dest='evidence_command', required=True)
    runs = commands.add_parser('runs', help='Discover stored runs; never launch jobs')
    for name in ('study', 'model', 'operation', 'status'):
        runs.add_argument('--'+name)
    runs.add_argument('--offset', type=int, default=0)
    runs.add_argument('--limit', type=int, default=25)
    show = commands.add_parser('show', help='Run steps, failures, source identities and replay configuration')
    show.add_argument('run_id')
    artifact = commands.add_parser('artifact', help='Artifact provenance and optional bounded summary')
    artifact.add_argument('artifact_id')
    artifact.add_argument('--result', action='store_true')
    compare = commands.add_parser('compare', help='Compare explicitly compatible C05 records; never average')
    compare.add_argument('artifact_ids', nargs='+')
    compare.add_argument('--metric', required=True)
    report = commands.add_parser('report', help='Build a standalone local evidence workbench')
    report.add_argument('artifact_ids', nargs='+')
    report.add_argument('--run-id')
    report.add_argument('--metric', help='Include explicit compatibility matrix')
    report.add_argument('--output', required=True)
    return parser


def execute(ns):
    from .catalog import local_catalog
    from .queries import EvidenceReader, compare_evidence
    reader = EvidenceReader(local_catalog(ns.catalog_root))
    try:
        if ns.evidence_command == 'runs':
            result = reader.runs(**{name:getattr(ns, name) for name in ('study','model','operation','status','offset','limit')})
        elif ns.evidence_command == 'show':
            result = reader.run(ns.run_id)
        elif ns.evidence_command == 'artifact':
            result = reader.artifact(ns.artifact_id, include_result=ns.result)
        else:
            if len(ns.artifact_ids) > 32:
                raise ValueError('At most 32 explicit artifacts per view')
            artifacts = [reader.artifact(key, include_result=True) for key in ns.artifact_ids]
            comparison = compare_evidence(artifacts, metric=ns.metric) if ns.metric else None
            if ns.evidence_command == 'compare':
                result = comparison
            else:
                from .report import write_workbench
                path = write_workbench(ns.output, artifacts, run=reader.run(ns.run_id) if ns.run_id else None, comparison=comparison)
                result = {'report': str(path), 'summary_bytes_read': reader.bytes_read}
    except (KeyError, ValueError, OSError) as exc:
        print(json.dumps({'status':'unavailable', 'reason':str(exc), 'action':'Check the catalog root, immutable artifact IDs, payload access and view limits.'}))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main(argv=None):
    return execute(configure(argparse.ArgumentParser(prog='python -m pipelines_v2.evidence')).parse_args(argv))
