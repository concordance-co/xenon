#!/usr/bin/env python3
"""Validate this planning package; does not execute any planned task."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re


def validate(root: Path, repo: Path) -> dict:
    graph = json.loads((root / 'graph.json').read_text())
    errors: list[str] = []
    nodes = {n['id']: n for n in graph['tasks']}
    if len(nodes) != len(graph['tasks']):
        errors.append('Duplicate task IDs')
    expected_order = graph['topological_order']
    if len(expected_order) != len(nodes) or set(expected_order) != set(nodes):
        errors.append('Topological order does not cover exactly the tasks')
    position = {v: i for i, v in enumerate(expected_order)}
    contract_text = (root / 'CONTRACTS.md').read_text()
    known_contracts = set(re.findall(r'^## (C\d+) ', contract_text, re.M))
    required = ['Dispatch contract', 'Read first', 'Required outputs', 'Implementation sequence',
                'Verification and acceptance', 'Resource and external dependency limits',
                'Outside this task', 'Handoff']
    edge_count = 0
    for task_id, node in nodes.items():
        plan = root / node['plan']
        if not plan.is_file():
            errors.append(f'{task_id}: missing {plan}')
            continue
        text = plan.read_text()
        if not text.startswith(f'# {task_id} {node["title"]}\n'):
            errors.append(f'{task_id}: title mismatch')
        for section in required:
            if f'## {section}\n' not in text:
                errors.append(f'{task_id}: missing section {section}')
        for field, label in [('start_after', 'Start after'), ('integration_requires', 'Acceptance also requires')]:
            match = re.search(rf'^- {label}: (.*)$', text, re.M)
            actual = re.findall(r'\[(T\d+)\]', match.group(1)) if match else None
            if actual != node[field]:
                errors.append(f'{task_id}: {field} differs from plan')
        deps = node['start_after'] + node['integration_requires']
        if len(deps) != len(set(deps)):
            errors.append(f'{task_id}: duplicate dependency')
        for dep in deps:
            edge_count += 1
            if dep not in nodes:
                errors.append(f'{task_id}: unknown predecessor {dep}')
            elif position.get(dep, 10**6) >= position.get(task_id, -1):
                errors.append(f'{task_id}: cyclic or incorrectly ordered dependency {dep}')
        for contract in node['contracts']:
            if contract not in known_contracts:
                errors.append(f'{task_id}: undefined contract {contract}')
        if node['status'] != 'planned':
            errors.append(f'{task_id}: planning package contains execution status')
    for milestone in graph['milestones']:
        for task_id in milestone['requires']:
            if task_id not in nodes:
                errors.append(f'{milestone["id"]}: unknown task {task_id}')
    # Resolve links as if staged at platform/research_hub_execution, even in a draft directory.
    virtual_root = repo / 'platform' / 'archive' / 'research_hub_execution'
    local_links = 0
    for path in root.rglob('*.md'):
        text = path.read_text()
        if any(line.rstrip() != line for line in text.splitlines()):
            errors.append(f'{path.name}: trailing whitespace')
        if text.count('```') % 2:
            errors.append(f'{path.name}: unmatched fenced block')
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' in target or target.startswith('#'):
                continue
            rel = target.split('#', 1)[0]
            virtual = (virtual_root / path.relative_to(root).parent / rel).resolve()
            if virtual.is_relative_to(virtual_root):
                resolved = root / virtual.relative_to(virtual_root)
            else:
                resolved = virtual
            local_links += 1
            if not resolved.exists():
                errors.append(f'{path.name}: broken link {target}')
    declared = {(d, n['id'], '-') for n in nodes.values() for d in n['start_after']}
    declared |= {(d, n['id'], '.') for n in nodes.values() for d in n['integration_requires']}
    diagram = (root / 'DAG.md').read_text()
    drawn = {(a,b,'-' if arrow == '-->' else '.') for a,arrow,b in
             re.findall(r'^  (T\d+) (-->|-\.->) (T\d+)$', diagram, re.M)}
    if drawn != declared:
        errors.append('DAG.md edges differ from graph.json')
    if errors:
        raise ValueError('\n'.join(errors))
    return {'tasks': len(nodes), 'dependency_edges': edge_count,
            'milestones': len(graph['milestones']), 'contracts': len(known_contracts),
            'local_links_checked': local_links, 'result': 'passed'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--repo', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = args.repo.resolve() if args.repo else root.parents[2]
    print(json.dumps(validate(root, repo), indent=2))
