"""C03 failure boundaries exercise actual files, process death, and competing writers."""
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import threading

import pytest

from pipelines_v2.api import ArtifactManifest, LocalArtifactStore
from pipelines_v2.storage.finalization import (CheckpointJournal, IntegrityError, finalize_manifest,
    local_writer_lock, validate_completion, describe_payloads)
from pipelines_v2.storage.leases import atomic_owner
from pipelines_v2.core.provenance import resolve_provenance
from pipelines_v2.api import Dataset, Example, GenerationRunSpec, GenerationSpec, ToyEngine


def identity():
    return resolve_provenance(GenerationRunSpec(engine=ToyEngine(), dataset=Dataset.from_examples([Example(key='a', prompt='a')]), generation=GenerationSpec(enabled=True,max_tokens=1)))


def manifest(store):
    ref = store.write_json('fixture', 'result.json', {'rows': []})
    return ArtifactManifest(artifact_id='fixture', artifact_kind='generation_run', schema_version=1,
        operation_spec_hash='s', operation_semantic_hash='s', created_at='fixture', engine={}, runner={},
        input_artifact_refs=(), example_coverage={'example_keys': [], 'explicit_empty': True},
        storage_refs={'result': ref}, metadata={})


def test_completion_is_explicit_idempotent_and_detects_corruption(tmp_path):
    store = LocalArtifactStore(tmp_path)
    raw = manifest(store)
    assert validate_completion(raw, store=store, legacy_policy='read_only') is False
    complete = finalize_manifest(store, raw)
    assert validate_completion(complete, store=store)
    assert finalize_manifest(store, raw) == complete
    Path(raw.storage_refs['result']['path']).write_text('{')
    with pytest.raises(IntegrityError, match='checksum'):
        validate_completion(complete, store=store)
    with pytest.raises(IntegrityError):
        finalize_manifest(store, raw)


def test_commit_failure_never_publishes_manifest(tmp_path):
    class Store:
        def __init__(self):
            self.delegate = LocalArtifactStore(tmp_path)
            self.root = tmp_path
        def __getattr__(self, key):
            return getattr(self.delegate, key)
        def commit_visibility(self):
            raise OSError('volume visibility unavailable')
    store = Store()
    raw = manifest(store)
    with pytest.raises(OSError, match='visibility'):
        finalize_manifest(store, raw)
    assert not (tmp_path/'fixture'/'manifest.json').exists()


def test_final_manifest_without_catalog_is_recoverable(tmp_path):
    store = LocalArtifactStore(tmp_path)
    complete = finalize_manifest(store, manifest(store))
    restored = ArtifactManifest.from_dict(json.loads((tmp_path/'fixture'/'manifest.json').read_text()))
    assert validate_completion(restored, store=store)
    assert restored.metadata == complete.metadata
    assert restored.artifact_id == complete.artifact_id


@pytest.mark.parametrize('stage', ['payload', 'journal', 'before_manifest', 'after_manifest'])
def test_process_death_preserves_only_verified_units(tmp_path, stage):
    provenance = identity()
    def child():
        store = LocalArtifactStore(tmp_path)
        with local_writer_lock(store, 'unit'):
            journal = CheckpointJournal(store=store, artifact_id='fixture', semantic_hash='s', provenance=provenance,
                expected={'unit_keys':['a'], 'explicit_empty':False})
            ref = store.write_json('fixture','checkpoints/unit.json', {'example_key':'a'})
            if stage == 'payload':
                os._exit(17)
            journal.append('a', refs={'result':ref}, coverage={'example_key':'a'})
            if stage in {'journal','before_manifest'}:
                os._exit(17)
            finalize_manifest(store, manifest(store))
            os._exit(17)
    process = multiprocessing.get_context('fork').Process(target=child)
    process.start(); process.join(10)
    assert not process.is_alive()
    assert process.exitcode == 17
    store = LocalArtifactStore(tmp_path)
    with local_writer_lock(store, 'unit'):  # killed process cannot retain local ownership
        journal = CheckpointJournal(store=store, artifact_id='fixture', semantic_hash='s', provenance=provenance,
            expected={'unit_keys':['a'], 'explicit_empty':False})
        assert len(journal.body['completed_units']) == (stage != 'payload')
        assert (tmp_path/'fixture'/'manifest.json').exists() == (stage == 'after_manifest')


def test_checkpoint_rejects_changed_contract_and_corrupt_bytes(tmp_path):
    store, provenance = LocalArtifactStore(tmp_path), identity()
    args = dict(store=store, artifact_id='fixture', semantic_hash='s', provenance=provenance,
                expected={'unit_keys':['a']}, shard={'index':0,'count':2})
    journal = CheckpointJournal(**args)
    ref = store.write_json('fixture','unit.json', {'a':1})
    journal.append('a', refs={'result':ref}, coverage={'key':'a'})
    for changed in ({'semantic_hash':'different'}, {'shard':{'index':0,'count':3}}, {'expected':{'unit_keys':['b']}}):
        with pytest.raises(IntegrityError, match='mismatch'):
            CheckpointJournal(**{**args, **changed})
    Path(ref['path']).write_text('broken')
    with pytest.raises(IntegrityError, match='checksum'):
        CheckpointJournal(**args)


class AtomicTable:
    def __init__(self):
        self.values = {}; self.mutex = threading.Lock()
    def put(self, key, value, *, skip_if_exists=False):
        with self.mutex:
            if key in self.values and skip_if_exists:
                return False
            self.values[key] = value
            return True
    def get(self, key):
        with self.mutex: return self.values.get(key)
    def pop(self, key, default=None):
        with self.mutex: return self.values.pop(key, default)


def test_atomic_ownership_excludes_live_writer_and_recovers_only_verified_stop():
    table = AtomicTable()
    old, new = {'call_id':'old'}, {'call_id':'new'}
    with atomic_owner(table, 'artifact', old, lambda _:False):
        with pytest.raises(IntegrityError, match='still active'):
            with atomic_owner(table, 'artifact', new, lambda _:False):
                pytest.fail('concurrent writer admitted')
        assert table.get('artifact:owner:0') == old
    table.put('artifact:owner:1', old)
    with atomic_owner(table, 'artifact', new, lambda owner: owner == old):
        assert table.get('artifact:owner:2') == new
    assert table.get('artifact:owner:2:released') == new


def test_atomic_acquire_race_admits_one_owner():
    table = AtomicTable()
    start = threading.Barrier(2)
    finish = threading.Barrier(2)
    acquired = []
    def run(index):
        start.wait()
        try:
            with atomic_owner(table, 'artifact', {'call_id':str(index)}, lambda _:False):
                acquired.append(index)
                finish.wait(timeout=5)
        except IntegrityError:
            finish.wait(timeout=5)
    workers = [threading.Thread(target=run,args=(i,)) for i in range(2)]
    for worker in workers: worker.start()
    for worker in workers: worker.join(10)
    assert len(acquired) == 1


def test_journal_roundtrip_rejects_schema_tampering_duplicate_and_missing_units(tmp_path):
    from pipelines_v2.core.provenance import content_hash
    store, provenance = LocalArtifactStore(tmp_path), identity()
    args = dict(store=store, artifact_id='fixture', semantic_hash='s', provenance=provenance,
                expected={'unit_keys':['a','b']})
    journal = CheckpointJournal(**args)
    ref = store.write_json('fixture','a.json', {'a':1})
    journal.append('a',refs={'result':ref},coverage={'key':'a'})
    with pytest.raises(IntegrityError, match='incomplete'):journal.verify_full()
    with pytest.raises(IntegrityError, match='Duplicate planned'):
        CheckpointJournal(**{**args,'expected':{'unit_keys':['a','a']}})
    body = json.loads(journal.path.read_text())
    body['completed_units'].append(body['completed_units'][0])
    body['journal_hash'] = content_hash({key:value for key,value in body.items() if key != 'journal_hash'})
    journal.path.write_text(json.dumps(body))
    with pytest.raises(IntegrityError, match='Duplicate completed'):CheckpointJournal(**args)


def test_progress_thread_preserves_worker_ownership_context():
    import contextvars
    from pipelines_v2.runtime.modal_worker import _execute_with_progress_stream
    owner = contextvars.ContextVar('test_owner')
    owner.set('modal-input-fixture')
    envelopes = list(_execute_with_progress_stream(lambda report: owner.get()))
    assert envelopes[-1]['result'] == 'modal-input-fixture'


def test_interrupted_recovery_owner_does_not_orphan_a_mutex():
    table = AtomicTable()
    old, crashed_recovery, current = ({'call_id':value} for value in ('original','crashed-recovery','current'))
    table.put('artifact:owner:0', old)
    table.put('artifact:owner:1', crashed_recovery)  # killed immediately after successful recovery acquisition
    stopped = lambda owner: owner['call_id'] in {'original','crashed-recovery'}
    with atomic_owner(table, 'artifact', current, stopped):
        assert table.get('artifact:owner:2') == current
    assert table.get('artifact:owner:2:released') == current


def test_installed_modal_ownership_api_contract():
    modal = pytest.importorskip('modal')
    from modal.call_graph import InputStatus
    assert callable(modal.current_function_call_id)
    assert callable(modal.current_input_id)
    assert callable(modal.FunctionCall.from_id)
    assert {status.name for status in InputStatus} >= {'SUCCESS','FAILURE','INIT_FAILURE','TERMINATED','TIMEOUT'}


@pytest.fixture
def modal_owned_call(monkeypatch):
    import modal
    from modal.call_graph import InputInfo, InputStatus
    from types import SimpleNamespace
    canonical = 'in-01M533PKV843N76WBKCWYCQDFP'
    table = AtomicTable()
    state = SimpleNamespace(table=table, canonical=canonical, cancel_error=None, graph_error=None,
        cancels=[], graphs=0, owner={'call_id':'fc-owned','input_id':canonical + ':1792179654528-0'})
    state.node = lambda status=InputStatus.FAILURE, input_id=canonical, call_id='fc-owned': InputInfo(
        input_id, call_id, '', status, 'fixture', '', [])
    state.nodes = [state.node()]
    class Call:
        def cancel(self, *, terminate_containers):
            state.cancels.append(terminate_containers)
            if state.cancel_error: raise state.cancel_error
        def get_call_graph(self):
            state.graphs += 1
            if state.graph_error: raise state.graph_error
            return state.nodes
    monkeypatch.setattr(modal.FunctionCall, 'from_id', lambda _: Call())
    monkeypatch.setattr(modal.Dict, 'from_name', lambda name, *, create_if_missing: table)
    return state


@pytest.mark.parametrize('status', ['SUCCESS','FAILURE','INIT_FAILURE','TERMINATED','TIMEOUT'])
def test_modal_attempt_suffix_requires_cancel_and_matching_terminal_input(modal_owned_call, status):
    from modal.call_graph import InputStatus
    from pipelines_v2.storage.leases import _modal_terminated, cancel_modal_writer_call
    state = modal_owned_call
    state.nodes = [state.node(getattr(InputStatus,status))]
    # In particular FAILURE alone can describe a previous attempt while a new
    # automatic retry is alive. No graph-only status releases ownership.
    assert _modal_terminated(state.owner) is False
    assert state.cancels == []
    certificate = cancel_modal_writer_call('fc-owned', expected_input_id=state.owner['input_id'])
    assert certificate['canonical_input_ids'] == [state.canonical]
    before = state.graphs
    assert _modal_terminated(state.owner) is True
    assert state.graphs == before + 1
    assert state.cancels == [True]
    assert cancel_modal_writer_call('fc-owned') == certificate
    assert _modal_terminated({**state.owner,'call_id':'fc-other'}) is False
    assert _modal_terminated({**state.owner,'input_id':'in-01M533PKV843N76WBKCWYCQDFQ'}) is False
    assert _modal_terminated({**state.owner,'canonical_input_id':'forged'}) is False
    assert _modal_terminated({**state.owner,'input_id':state.canonical + ':unknown'}) is False


def test_stale_failure_does_not_admit_live_retry_without_coordinator_cancel(modal_owned_call):
    from pipelines_v2.storage.leases import _modal_terminated
    state = modal_owned_call
    state.table.put('artifact:owner:0', state.owner)
    retry_owner = {**state.owner, 'input_id':state.canonical + ':1792179654529-1'}
    with pytest.raises(IntegrityError, match='termination unverified'):
        with atomic_owner(state.table, 'artifact', retry_owner, _modal_terminated):
            pytest.fail('Stale FAILURE cannot permit a second writer')
    assert state.cancels == []
    assert state.graphs == 0


@pytest.mark.parametrize('failure', ['cancel','graph','pending','empty','wrong_call','wrong_input','unknown_input','write'])
def test_cancel_proof_unavailable_never_publishes_certificate(modal_owned_call, failure):
    from modal.call_graph import InputStatus
    from pipelines_v2.storage.leases import cancel_modal_writer_call
    state = modal_owned_call
    if failure == 'cancel': state.cancel_error = RuntimeError('not acknowledged')
    if failure == 'graph': state.graph_error = RuntimeError('unavailable')
    if failure == 'pending': state.nodes.append(state.node(InputStatus.PENDING))
    if failure == 'empty': state.nodes = []
    if failure == 'wrong_call': state.nodes = [state.node(call_id='fc-other')]
    if failure == 'wrong_input': state.nodes = [state.node(input_id='in-01M533PKV843N76WBKCWYCQDFQ')]
    if failure == 'unknown_input': state.nodes = [state.node(input_id='unknown')]
    if failure == 'write': state.table.put = lambda *args, **kwargs: False
    with pytest.raises(IntegrityError):
        cancel_modal_writer_call('fc-owned',expected_input_id=state.owner['input_id'])
    assert state.table.get('stopped-call:fc-owned') is None


def test_certificate_requires_fresh_exact_input_set_and_terminal_graph(modal_owned_call):
    from modal.call_graph import InputStatus
    from pipelines_v2.storage.leases import _modal_terminated, cancel_modal_writer_call
    state = modal_owned_call
    cancel_modal_writer_call('fc-owned')
    for nodes in ([], [state.node(InputStatus.PENDING)],
                  [state.node(),state.node(InputStatus.PENDING)],
                  [state.node(),state.node(input_id='in-01M533PKV843N76WBKCWYCQDFQ')]):
        state.nodes = nodes
        assert _modal_terminated(state.owner) is False
    state.graph_error = RuntimeError('provider unavailable')
    assert _modal_terminated(state.owner) is False


@pytest.mark.parametrize('change', [
    {'call_id':'fc-other'}, {'canonical_input_ids':[]}, {'canonical_input_ids':['unknown']},
    {'canonical_input_ids':['in-01M533PKV843N76WBKCWYCQDFQ']},
    {'kind':'timestamp-expired'}, {'terminate_containers':False}, {'terminate_containers':1},
    {'created_at':'2000-01-01'}, {'canonical_input_ids':'in-01M533PKV843N76WBKCWYCQDFP'}])
def test_malformed_or_mismatched_certificate_never_releases_owner(modal_owned_call, change):
    from pipelines_v2.storage.leases import _modal_terminated, cancel_modal_writer_call
    state = modal_owned_call
    certificate = cancel_modal_writer_call('fc-owned')
    state.table.put('stopped-call:fc-owned', {**certificate, **change})
    assert _modal_terminated(state.owner) is False


def test_conflicting_certificate_is_immutable_and_invalid_expected_id_never_cancels(modal_owned_call):
    from pipelines_v2.storage.leases import cancel_modal_writer_call
    state = modal_owned_call
    state.table.put('stopped-call:fc-owned', {'kind':'conflicting'})
    with pytest.raises(IntegrityError, match='Conflicting'):
        cancel_modal_writer_call('fc-owned')
    assert state.table.get('stopped-call:fc-owned') == {'kind':'conflicting'}
    state.cancels.clear()
    with pytest.raises(IntegrityError, match='identity'):
        cancel_modal_writer_call('fc-owned',expected_input_id='invalid')
    assert state.cancels == []


@pytest.mark.parametrize('raises', [False,True])
def test_normal_release_needs_no_cancellation_certificate(raises):
    table = AtomicTable()
    owner = {'call_id':'owned'}
    try:
        with atomic_owner(table,'artifact',owner,lambda _:pytest.fail('No prior owner')):
            if raises: raise ValueError('ordinary worker exception')
    except ValueError: pass
    with atomic_owner(table,'artifact',{'call_id':'next'},lambda _:pytest.fail('Released owner needs no proof')):
        pass


def test_owner_conflict_without_observable_owner_fails_closed():
    from pipelines_v2.storage.leases import atomic_owner
    class UnobservableOwner:
        def get(self, key): return None
        def put(self, key, value, *, skip_if_exists): return False
    with pytest.raises(IntegrityError, match='could not be read'):
        with atomic_owner(UnobservableOwner(), 'key', {'call_id':'ours'}, lambda previous: pytest.fail('Missing owner cannot be checked')):
            pytest.fail('Unverified owner must not be bypassed')
