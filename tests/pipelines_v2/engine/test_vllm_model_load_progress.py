from __future__ import annotations

import json

from pipelines_v2.engine.vllm.model_load_progress import (
    _aggregate_loader_progress,
    _tracked_tqdm,
    enable_model_load_progress,
)


class _Progress:
    def __init__(self, values, *, desc: str, total: int) -> None:
        self._values = list(values)
        self.desc = desc
        self.total = total

    def __iter__(self):
        return iter(self._values)


def test_model_load_progress_does_not_replace_vllm_worker_by_default(
    monkeypatch,
) -> None:
    monkeypatch.delenv("XENON_VLLM_SHARD_PROGRESS", raising=False)
    llm_kwargs = {}

    enable_model_load_progress(llm_kwargs, lambda _payload: None)

    assert "worker_cls" not in llm_kwargs


def test_model_load_progress_worker_requires_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("XENON_VLLM_SHARD_PROGRESS", "1")
    llm_kwargs = {}

    enable_model_load_progress(llm_kwargs, lambda _payload: None)

    assert llm_kwargs["worker_cls"].endswith(".XenonProgressGPUWorker")


def test_tracked_tqdm_emits_each_completed_checkpoint_shard(tmp_path) -> None:
    progress_path = tmp_path / "progress.jsonl"

    def fake_tqdm(iterable, *args, **kwargs):
        del args
        return _Progress(
            iterable,
            desc=str(kwargs.get("desc") or ""),
            total=len(iterable),
        )

    tracked = _tracked_tqdm(fake_tqdm, progress_path)
    values = list(
        tracked(
            ["shard-1", "shard-2"],
            desc="Loading safetensors checkpoint shards",
        )
    )

    assert values == ["shard-1", "shard-2"]
    events = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["current"] for event in events] == [0, 1, 2]
    assert all(event["total"] == 2 for event in events)
    assert len({event["loader_id"] for event in events}) == 1


def test_tracked_tqdm_leaves_unrelated_progress_untouched(tmp_path) -> None:
    progress_path = tmp_path / "progress.jsonl"

    def fake_tqdm(iterable, *args, **kwargs):
        del args
        return _Progress(
            iterable,
            desc=str(kwargs.get("desc") or ""),
            total=len(iterable),
        )

    tracked = _tracked_tqdm(fake_tqdm, progress_path)
    values = list(tracked([1, 2], desc="Compiling graphs"))

    assert values == [1, 2]
    assert not progress_path.exists()


def test_loader_progress_aggregates_concurrent_worker_sequences() -> None:
    payload = _aggregate_loader_progress(
        {
            "worker-1:1": (8, 16, "running"),
            "worker-2:1": (4, 16, "running"),
        }
    )

    assert payload == {
        "stage": "model_loading",
        "status": "running",
        "message": "Loading checkpoint shards",
        "current": 12,
        "total": 32,
        "unit": "shards",
    }
