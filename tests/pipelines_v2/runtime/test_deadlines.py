from types import SimpleNamespace
import threading
import pytest

from pipelines_v2.runtime.deadlines import bounded_remote_call, bounded_stream


def test_result_deadline_cancels_once_without_resubmit():
    calls=[]
    class Call:
        def get(self, timeout):
            calls.append(('get',timeout))
            raise TimeoutError('queue deadline')
        def cancel(self, **kwargs):
            calls.append(('cancel',kwargs))
    def spawn(*args):
        calls.append(('spawn',args))
        return Call()
    with pytest.raises(TimeoutError,match='queue deadline'):
        bounded_remote_call(SimpleNamespace(spawn=spawn),'request',timeout_seconds=.1)
    assert [call[0] for call in calls] == ['spawn','get','cancel']
    assert calls[-1][1] == {'terminate_containers':True}


def test_stream_wall_deadline_includes_no_progress_wait():
    release=threading.Event()
    def stream():
        yield {'kind':'progress'}
        release.wait(1)
        yield {'kind':'result'}
    try:
        values=bounded_stream(stream(),timeout_seconds=.05)
        assert next(values)['kind']=='progress'
        with pytest.raises(TimeoutError,match='wall deadline'):
            next(values)
    finally:
        release.set()


def test_stream_preserves_errors_and_success():
    assert list(bounded_stream(iter([1,2]),timeout_seconds=1)) == [1,2]
    def broken():
        raise ValueError('failure')
        yield
    with pytest.raises(ValueError,match='failure'):
        list(bounded_stream(broken(),timeout_seconds=1))


def test_late_submission_handle_is_cancelled_without_retry():
    released=threading.Event()
    cancelled=threading.Event()
    calls=[]
    def spawn():
        calls.append('spawn')
        released.wait(1)
        return SimpleNamespace(cancel=lambda **kwargs:cancelled.set())
    try:
        with pytest.raises(TimeoutError,match='submission'):
            bounded_remote_call(SimpleNamespace(spawn=spawn),timeout_seconds=.03)
        released.set()
        assert cancelled.wait(1)
        assert calls==['spawn']
    finally:
        released.set()


def test_stream_failure_cancels_only_its_reported_call(monkeypatch):
    from pipelines_v2.runtime.modal_worker import _consume_progress_stream
    cancelled=[]
    monkeypatch.setattr('pipelines_v2.runtime.modal_worker.cancel_owned_call',lambda call_id,**kwargs:cancelled.append((call_id,kwargs)))
    def stream():
        yield {'kind':'call_identity','call_id':'fc-owned','input_id':'in-owned:1-0'}
        raise ValueError('worker failed')
    with pytest.raises(ValueError,match='worker failed'):
        _consume_progress_stream(stream(),None,1)
    assert cancelled==[('fc-owned',{'expected_input_id':'in-owned:1-0'})]
