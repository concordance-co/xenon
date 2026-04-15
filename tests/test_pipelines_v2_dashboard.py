"""Phase A dashboard backend tests.

Covers:
- composite catalog merge dedupe + newest-first ordering
- run detail normalization: stable DAG nodes/edges from persisted workflow payloads
- /api/runs and /api/runs/{run_id} endpoints via FastAPI TestClient
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pipelines_v2.storage.composite import CompositeCatalog
from pipelines_v2.storage.local import FileCatalog
from pipelines_v2.workflow.records import WorkflowRunRecord, WorkflowStepRecord

from pipelines_v2.dashboard.catalog import DashboardCatalog
from pipelines_v2.dashboard.normalize import build_run_detail, build_run_summary


# ---------------------------------------------------------------------------
# Helpers: fabricate a persisted workflow payload without booting any runners.
# The dashboard must render this cleanly from catalog state alone.
# ---------------------------------------------------------------------------


def _capture_step_payload(name: str, runner: str = "capture") -> dict[str, Any]:
    # Minimal valid-ish capture spec dict. The dashboard only parses `name`,
    # `runner`, `spec.kind`, and optionally follows StepRef-style back-edges
    # inside `spec`. A minimal kind is enough for the normalization layer.
    return {
        "name": name,
        "runner": runner,
        "spec": {"kind": "capture"},
        "depends_on": [],
    }


def _probe_step_payload(name: str, depends_on: list[str], runner: str = "analysis") -> dict[str, Any]:
    return {
        "name": name,
        "runner": runner,
        "spec": {"kind": "probe"},
        "depends_on": depends_on,
    }


def _report_step_payload(name: str, depends_on: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "runner": "report",
        "spec": {"kind": "report"},
        "depends_on": depends_on,
    }


def _workflow_payload(steps: list[dict[str, Any]], *, name: str = "demo") -> dict[str, Any]:
    return {
        "kind": "workflow",
        "schema_version": 1,
        "name": name,
        "steps": steps,
    }


def _make_run_record(
    *,
    run_id: str,
    workflow_payload: dict[str, Any],
    started_at: str,
    status: str = "completed",
    finished_at: str | None = None,
) -> WorkflowRunRecord:
    return WorkflowRunRecord(
        run_id=run_id,
        workflow_name=workflow_payload.get("name"),
        workflow_hash="wh_" + run_id,
        workflow_spec_hash="wsh_" + run_id,
        workflow_payload=workflow_payload,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
    )


def _make_step_record(
    *,
    run_id: str,
    step_index: int,
    step_name: str,
    runner: str,
    status: str = "completed",
    artifact_id: str | None = None,
    artifact_kind: str | None = None,
    reused_from_run_id: str | None = None,
) -> WorkflowStepRecord:
    return WorkflowStepRecord(
        run_id=run_id,
        workflow_hash="wh_" + run_id,
        workflow_step_key=f"wh_{run_id}.{step_name}",
        step_name=step_name,
        step_index=step_index,
        runner=runner,
        status=status,
        step_semantic_hash=f"ss_{step_name}",
        step_spec_hash=f"sp_{step_name}",
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        started_at="2026-04-15T00:00:00Z",
        finished_at="2026-04-15T00:01:00Z",
        reused_from_run_id=reused_from_run_id,
    )


# ---------------------------------------------------------------------------
# CompositeCatalog behavior — the dashboard relies on merge dedupe + ordering.
# ---------------------------------------------------------------------------


def test_composite_catalog_dedupes_by_run_id_and_orders_newest_first(tmp_path: Path) -> None:
    primary = FileCatalog(root=tmp_path / "primary")
    secondary = FileCatalog(root=tmp_path / "secondary")

    wf = _workflow_payload([_capture_step_payload("cap")])

    older = _make_run_record(run_id="r1", workflow_payload=wf, started_at="2026-04-10T00:00:00Z")
    middle = _make_run_record(run_id="r2", workflow_payload=wf, started_at="2026-04-12T00:00:00Z")
    newer = _make_run_record(run_id="r3", workflow_payload=wf, started_at="2026-04-14T00:00:00Z")

    # Write the same run to both catalogs to exercise dedupe.
    for record in (older, middle):
        primary.record_workflow_run(record)
    for record in (middle, newer):
        secondary.record_workflow_run(record)

    composite = CompositeCatalog(catalogs=(primary, secondary))
    runs = composite.list_workflow_runs()

    run_ids = [r.run_id for r in runs]
    assert run_ids == ["r3", "r2", "r1"], "newest-first order with dedupe"
    assert len(runs) == 3


# ---------------------------------------------------------------------------
# Normalization: workflow payload -> DAG nodes/edges + step summaries.
# ---------------------------------------------------------------------------


def test_build_run_detail_produces_stable_nodes_and_edges() -> None:
    wf = _workflow_payload(
        [
            _capture_step_payload("cap"),
            _probe_step_payload("probe", depends_on=["cap"]),
            _report_step_payload("report", depends_on=["probe"]),
        ]
    )
    run = _make_run_record(run_id="r1", workflow_payload=wf, started_at="2026-04-15T00:00:00Z")
    steps = [
        _make_step_record(run_id="r1", step_index=0, step_name="cap", runner="capture", artifact_id="a1"),
        _make_step_record(run_id="r1", step_index=1, step_name="probe", runner="analysis", artifact_id="a2"),
        _make_step_record(
            run_id="r1",
            step_index=2,
            step_name="report",
            runner="report",
            artifact_id="a3",
            artifact_kind="report",
        ),
    ]

    detail = build_run_detail(run, steps)

    node_ids = [n.id for n in detail.nodes]
    assert node_ids == ["cap", "probe", "report"], "nodes follow step_index ordering"

    families = {n.id: n.family for n in detail.nodes}
    assert families == {"cap": "capture", "probe": "readout", "report": "report"}

    edges = sorted((e.source, e.target, e.kind) for e in detail.edges)
    assert edges == [
        ("cap", "probe", "declared"),
        ("probe", "report", "declared"),
    ]

    assert detail.run.has_report is True
    assert detail.run.step_counts.total == 3
    assert detail.run.step_counts.completed == 3


def test_build_run_detail_includes_unexecuted_steps_as_pending() -> None:
    wf = _workflow_payload(
        [
            _capture_step_payload("cap"),
            _probe_step_payload("probe", depends_on=["cap"]),
        ]
    )
    run = _make_run_record(
        run_id="r1",
        workflow_payload=wf,
        started_at="2026-04-15T00:00:00Z",
        status="running",
        finished_at=None,
    )
    # Only the capture step has run.
    steps = [
        _make_step_record(run_id="r1", step_index=0, step_name="cap", runner="capture", artifact_id="a1"),
    ]

    detail = build_run_detail(run, steps)

    statuses = {s.step_name: s.status for s in detail.steps}
    assert statuses == {"cap": "completed", "probe": "pending"}
    assert [n.id for n in detail.nodes] == ["cap", "probe"]
    assert any(e.source == "cap" and e.target == "probe" for e in detail.edges)


def test_run_summary_flags_reused_and_has_report() -> None:
    wf = _workflow_payload([_capture_step_payload("cap"), _report_step_payload("report", ["cap"])])
    run = _make_run_record(run_id="r1", workflow_payload=wf, started_at="2026-04-15T00:00:00Z")
    steps = [
        _make_step_record(
            run_id="r1",
            step_index=0,
            step_name="cap",
            runner="capture",
            status="reused",
            reused_from_run_id="r_prior",
        ),
        _make_step_record(run_id="r1", step_index=1, step_name="report", runner="report"),
    ]
    summary = build_run_summary(run, steps)

    assert summary.has_report is True
    assert summary.step_counts.reused == 1
    assert summary.step_counts.completed == 1


# ---------------------------------------------------------------------------
# Endpoint smoke via FastAPI TestClient. Skipped if FastAPI isn't installed.
# ---------------------------------------------------------------------------


pytest.importorskip("fastapi")


def _seed_catalog(tmp_path: Path) -> DashboardCatalog:
    root = tmp_path / "catalog"
    local = FileCatalog(root=root)
    composite = CompositeCatalog(catalogs=(local,))

    wf = _workflow_payload(
        [
            _capture_step_payload("cap"),
            _probe_step_payload("probe", depends_on=["cap"]),
        ]
    )
    run = _make_run_record(run_id="run_abc", workflow_payload=wf, started_at="2026-04-15T00:00:00Z")
    local.record_workflow_run(run)
    local.record_workflow_step(_make_step_record(run_id="run_abc", step_index=0, step_name="cap", runner="capture"))
    local.record_workflow_step(_make_step_record(run_id="run_abc", step_index=1, step_name="probe", runner="analysis"))

    return DashboardCatalog(local=local, composite=composite, raw=composite, local_root=root, postgres_env=None)


def test_api_runs_and_run_detail_roundtrip(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from pipelines_v2.dashboard.server import create_app

    app = create_app(catalog=_seed_catalog(tmp_path))
    client = TestClient(app)

    resp = client.get("/api/runs")
    assert resp.status_code == 200
    payload = resp.json()
    assert [r["run_id"] for r in payload["runs"]] == ["run_abc"]
    assert payload["runs"][0]["step_counts"]["total"] == 2

    resp = client.get("/api/runs/run_abc")
    assert resp.status_code == 200
    detail = resp.json()
    assert [n["id"] for n in detail["nodes"]] == ["cap", "probe"]
    assert {(e["source"], e["target"]) for e in detail["edges"]} == {("cap", "probe")}
    assert detail["run"]["workflow_name"] == "demo"
    assert detail["workflow_payload"]["steps"][0]["name"] == "cap"


def test_cached_catalog_hits_inner_once_for_repeat_reads(tmp_path: Path) -> None:
    """Read-through TTL cache should coalesce repeat calls — critical for
    keeping Postgres traffic bounded on the dashboard."""
    from pipelines_v2.dashboard.caching import CachedCatalog

    class Counting:
        def __init__(self, inner) -> None:
            self.inner = inner
            self.runs_calls = 0
            self.load_calls = 0
            self.artifact_calls = 0

        def identity(self) -> dict[str, Any]:
            return {"kind": "counting"}

        def list_workflow_runs(self, **kwargs: Any):
            self.runs_calls += 1
            return self.inner.list_workflow_runs(**kwargs)

        def load_workflow_run(self, run_id: str):
            self.load_calls += 1
            return self.inner.load_workflow_run(run_id)

        def list_workflow_steps(self, run_id: str):
            return self.inner.list_workflow_steps(run_id)

        def load_artifact(self, artifact_id: str):
            self.artifact_calls += 1
            return self.inner.load_artifact(artifact_id)

        def find_artifact_for_workflow_step(self, **kwargs: Any):
            return self.inner.find_artifact_for_workflow_step(**kwargs)

    primary = FileCatalog(root=tmp_path / "p")
    wf = _workflow_payload([_capture_step_payload("cap")])
    primary.record_workflow_run(_make_run_record(run_id="rA", workflow_payload=wf, started_at="2026-04-15T00:00:00Z"))
    counting = Counting(primary)
    cached = CachedCatalog(counting, list_ttl=60.0, hot_ttl=60.0, cold_ttl=60.0)

    # Repeat list_workflow_runs → hits inner exactly once.
    cached.list_workflow_runs()
    cached.list_workflow_runs()
    cached.list_workflow_runs()
    assert counting.runs_calls == 1

    # list_workflow_runs also seeds load_workflow_run so follow-up is free.
    cached.load_workflow_run("rA")
    cached.load_workflow_run("rA")
    assert counting.load_calls == 0

    # Miss responses are cached for a short window (miss_ttl) — repeats still hit once.
    cached.load_artifact("nope")
    cached.load_artifact("nope")
    assert counting.artifact_calls == 1

    stats = cached.stats()
    assert stats["hits"] > 0
    assert stats["misses"] > 0


def test_runs_endpoint_uses_pg_counts_fast_path(tmp_path: Path) -> None:
    """When a dashboard pg pool is available, /api/runs should use the
    aggregated `step_status_counts` query instead of N × list_workflow_steps."""
    from fastapi.testclient import TestClient

    from pipelines_v2.dashboard.server import create_app
    from pipelines_v2.dashboard.catalog import DashboardCatalog
    from pipelines_v2.storage.composite import CompositeCatalog
    from pipelines_v2.storage.local import FileCatalog

    # Seed 3 runs with differing step status histograms.
    root = tmp_path / "catalog"
    local = FileCatalog(root=root)
    composite = CompositeCatalog(catalogs=(local,))
    wf = _workflow_payload([_capture_step_payload("cap"), _probe_step_payload("probe", ["cap"])])
    for i, (completed, failed) in enumerate([(2, 0), (1, 1), (0, 0)]):
        rid = f"run_pg_{i}"
        local.record_workflow_run(
            _make_run_record(run_id=rid, workflow_payload=wf, started_at=f"2026-04-1{i}T00:00:00Z")
        )
        for j in range(completed):
            local.record_workflow_step(
                _make_step_record(run_id=rid, step_index=j, step_name=f"s{j}", runner="x", status="completed")
            )
        for j in range(failed):
            local.record_workflow_step(
                _make_step_record(
                    run_id=rid,
                    step_index=completed + j,
                    step_name=f"f{j}",
                    runner="x",
                    status="failed",
                )
            )

    # Fake pg that mimics the fast-path methods without touching Postgres.
    class FakePg:
        def __init__(self, records) -> None:
            self.calls: list[list[str]] = []
            self.light_calls: int = 0
            self._by_id = {r.run_id: r for r in records}

        def list_workflow_runs_light(self, **_kwargs) -> list[dict[str, Any]]:
            self.light_calls += 1
            rows = []
            for rid in ("run_pg_0", "run_pg_1", "run_pg_2"):
                r = self._by_id[rid]
                rows.append(
                    {
                        "run_id": r.run_id,
                        "workflow_name": r.workflow_name,
                        "workflow_hash": r.workflow_hash,
                        "workflow_spec_hash": r.workflow_spec_hash,
                        "status": r.status,
                        "started_at": r.started_at,
                        "parent_run_id": r.parent_run_id,
                        "finished_at": r.finished_at,
                        "error": r.error,
                        "has_report": False,
                    }
                )
            return rows

        def step_status_counts(self, run_ids: list[str]) -> dict[str, dict[str, int]]:
            self.calls.append(list(run_ids))
            return {
                "run_pg_0": {"completed": 2},
                "run_pg_1": {"completed": 1, "failed": 1},
                # run_pg_2 intentionally absent → we still emit a summary with
                # zero-counts; the fallback local merge adds nothing new here.
            }

        def batch_load_artifacts(self, _: Any) -> dict[str, Any]:
            return {}

        def find_artifacts_for_run(self, _: Any) -> dict[str, Any]:
            return {}

        def stats(self) -> dict[str, Any]:
            return {"pool": {}}

        def close(self) -> None:
            pass

    all_records = local.list_workflow_runs()
    fake = FakePg(all_records)
    dash = DashboardCatalog(
        local=local,
        composite=composite,
        raw=composite,
        local_root=root,
        postgres_env=None,
        pg=fake,  # type: ignore[arg-type]
    )
    app = create_app(catalog=dash)
    client = TestClient(app)

    resp = client.get("/api/runs")
    assert resp.status_code == 200
    runs = {r["run_id"]: r for r in resp.json()["runs"]}
    assert runs["run_pg_0"]["step_counts"]["completed"] == 2
    assert runs["run_pg_0"]["step_counts"]["total"] == 2
    assert runs["run_pg_1"]["step_counts"]["failed"] == 1
    assert runs["run_pg_1"]["step_counts"]["completed"] == 1
    # run_pg_2 isn't returned by step_status_counts → empty counts.
    assert runs["run_pg_2"]["step_counts"]["total"] == 0
    # Lightweight list fired once, aggregated counts fired once with all ids.
    assert fake.light_calls == 1
    assert len(fake.calls) == 1
    assert sorted(fake.calls[0]) == ["run_pg_0", "run_pg_1", "run_pg_2"]


def test_bulk_steps_detail_endpoint(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from pipelines_v2.dashboard.server import create_app

    app = create_app(catalog=_seed_catalog(tmp_path))
    client = TestClient(app)

    resp = client.get("/api/runs/run_abc/steps-detail")
    assert resp.status_code == 200
    payload = resp.json()
    names = [d["step"]["step_name"] for d in payload["step_details"]]
    assert names == ["cap", "probe"]


def test_api_run_detail_404_for_unknown_run(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from pipelines_v2.dashboard.server import create_app

    app = create_app(catalog=_seed_catalog(tmp_path))
    client = TestClient(app)

    resp = client.get("/api/runs/does_not_exist")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Phase C — step detail
# ---------------------------------------------------------------------------


def test_step_detail_normalizes_spec_summary_and_deps(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from pipelines_v2.dashboard.server import create_app

    app = create_app(catalog=_seed_catalog(tmp_path))
    client = TestClient(app)

    resp = client.get("/api/runs/run_abc/steps/probe")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["step"]["step_name"] == "probe"
    assert payload["spec"] == {"kind": "probe"}
    # spec_summary always promotes `kind` first.
    assert payload["spec_summary"][0] == {"label": "kind", "value": "probe"}
    assert [u["step_name"] for u in payload["upstream"]] == ["cap"]
    assert payload["downstream"] == []


def test_step_detail_404_for_unknown_step(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from pipelines_v2.dashboard.server import create_app

    app = create_app(catalog=_seed_catalog(tmp_path))
    client = TestClient(app)

    resp = client.get("/api/runs/run_abc/steps/nope")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Phase D — dataset / label previews
# ---------------------------------------------------------------------------


def _seed_catalog_with_real_capture(tmp_path: Path):
    """Register a run whose persisted payload is a *real* capture workflow.

    This exercises the dataset-preview path that rehydrates a Dataset from the
    persisted workflow payload and samples rows.
    """
    from pipelines_v2.api import (
        CaptureSpec,
        Dataset,
        Example,
        GenerationSpec,
    )
    from pipelines_v2.engine.toy import ToyEngine
    from pipelines_v2.operations.capture.sites import ResidualSite
    from pipelines_v2.storage.local import FileCatalog
    from pipelines_v2.storage.composite import CompositeCatalog
    from pipelines_v2.dashboard.catalog import DashboardCatalog
    from pipelines_v2.workflow.specs import WorkflowSpec, WorkflowStep

    examples = [
        Example(key=f"ex{i}", prompt=f"Prompt number {i}", labels={"class": "pos" if i % 2 == 0 else "neg"})
        for i in range(8)
    ]
    dataset = Dataset.from_examples(examples, name="tiny")
    capture_spec = CaptureSpec(
        engine=ToyEngine(),
        dataset=dataset,
        sites=(ResidualSite(name="resid", site="post", layers=(0,)),),
        generation=GenerationSpec(max_tokens=4),
        prompt_metadata_builder=None,
    )
    capture = WorkflowStep(name="cap", runner="capture", spec=capture_spec)
    workflow = WorkflowSpec(name="real_capture_workflow", steps=(capture,))

    root = tmp_path / "catalog"
    local = FileCatalog(root=root)
    composite = CompositeCatalog(catalogs=(local,))

    run = _make_run_record(
        run_id="run_real",
        workflow_payload=workflow.to_dict(),
        started_at="2026-04-15T00:00:00Z",
    )
    local.record_workflow_run(run)
    local.record_workflow_step(_make_step_record(run_id="run_real", step_index=0, step_name="cap", runner="capture"))
    return DashboardCatalog(local=local, composite=composite, raw=composite, local_root=root, postgres_env=None)


def test_dataset_preview_samples_in_memory_dataset(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from pipelines_v2.dashboard.server import create_app

    dash = _seed_catalog_with_real_capture(tmp_path)
    app = create_app(catalog=dash)
    client = TestClient(app)

    resp = client.get("/api/runs/run_real/steps/cap/dataset-preview?sample_size=5")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["available"] is True
    assert len(payload["rows"]) == 5
    assert payload["total_rows"] == 8
    assert payload["source"]["kind"] == "memory"
    assert payload["rows"][0]["prompt_preview"].startswith("Prompt number")


def test_label_preview_bucket_distribution(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from pipelines_v2.dashboard.server import create_app

    dash = _seed_catalog_with_real_capture(tmp_path)
    app = create_app(catalog=dash)
    client = TestClient(app)

    # request the full dataset so the distribution matches deterministically.
    resp = client.get("/api/runs/run_real/steps/cap/label-preview?sample_size=8")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["available"] is True
    class_label = next(l for l in payload["labels"] if l["label_name"] == "class")
    values = {b["value"]: b["count"] for b in class_label["buckets"]}
    assert values == {"pos": 4, "neg": 4}


def test_dataset_preview_missing_postgres_env_returns_unavailable(tmp_path: Path, monkeypatch) -> None:
    """Deferred Postgres dataset without local secrets should degrade to an
    unavailable state, not crash."""
    from fastapi.testclient import TestClient

    from pipelines_v2.api import (
        CaptureSpec,
        Dataset,
        GenerationSpec,
        PostgresSource,
    )
    from pipelines_v2.engine.toy import ToyEngine
    from pipelines_v2.operations.capture.sites import ResidualSite
    from pipelines_v2.storage.local import FileCatalog
    from pipelines_v2.storage.composite import CompositeCatalog
    from pipelines_v2.dashboard.catalog import DashboardCatalog
    from pipelines_v2.dashboard.server import create_app
    from pipelines_v2.workflow.specs import WorkflowSpec, WorkflowStep

    monkeypatch.delenv("DASHBOARD_TEST_DB", raising=False)

    source = PostgresSource(url_env_var="DASHBOARD_TEST_DB")
    dataset = Dataset.from_source(
        source=source,
        defer=True,
        fetch={
            "table": "examples",
            "prompt_column": "prompt",
            "example_key_column": "key",
            "label_columns": ("class",),
        },
    )
    capture_spec = CaptureSpec(
        engine=ToyEngine(),
        dataset=dataset,
        sites=(ResidualSite(name="resid", site="post", layers=(0,)),),
        generation=GenerationSpec(max_tokens=4),
    )
    capture = WorkflowStep(name="cap", runner="capture", spec=capture_spec)
    workflow = WorkflowSpec(name="pg_workflow", steps=(capture,))

    root = tmp_path / "catalog"
    local = FileCatalog(root=root)
    composite = CompositeCatalog(catalogs=(local,))
    run = _make_run_record(run_id="run_pg", workflow_payload=workflow.to_dict(), started_at="2026-04-15T00:00:00Z")
    local.record_workflow_run(run)
    local.record_workflow_step(_make_step_record(run_id="run_pg", step_index=0, step_name="cap", runner="capture"))
    dash = DashboardCatalog(local=local, composite=composite, raw=composite, local_root=root, postgres_env=None)

    app = create_app(catalog=dash)
    client = TestClient(app)

    resp = client.get("/api/runs/run_pg/steps/cap/dataset-preview")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["available"] is False
    assert "DASHBOARD_TEST_DB" in payload["reason"]


# ---------------------------------------------------------------------------
# Phase E — prompt preview
# ---------------------------------------------------------------------------


def test_prompt_preview_without_builder_returns_prompt_text(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from pipelines_v2.dashboard.server import create_app

    dash = _seed_catalog_with_real_capture(tmp_path)
    app = create_app(catalog=dash)
    client = TestClient(app)

    resp = client.get("/api/runs/run_real/steps/cap/prompt-preview?max_examples=2")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["available"] is True
    assert len(payload["examples"]) == 2
    # No builder, no tokenizer required, no section selector: not degraded.
    assert payload["degraded"] is False
    assert payload["examples"][0]["sections"] == []
    assert payload["examples"][0]["text"].startswith("Prompt number")


def test_prompt_preview_section_selector_without_builder_is_hard_unresolved(tmp_path: Path) -> None:
    """A probe that uses TokenSelector.section(...) on a capture with no
    prompt_metadata_builder must return a hard unresolved state, not fake spans."""
    from fastapi.testclient import TestClient
    from pipelines_v2.api import (
        CaptureSpec,
        Dataset,
        Example,
        GenerationSpec,
        ProbeSpec,
        StepRef,
    )
    from pipelines_v2.engine.toy import ToyEngine
    from pipelines_v2.operations.capture.sites import ResidualSite
    from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector
    from pipelines_v2.storage.local import FileCatalog
    from pipelines_v2.storage.composite import CompositeCatalog
    from pipelines_v2.dashboard.catalog import DashboardCatalog
    from pipelines_v2.dashboard.server import create_app
    from pipelines_v2.workflow.specs import WorkflowSpec, WorkflowStep

    examples = [Example(key=f"k{i}", prompt=f"p{i}", labels={"cls": "a"}) for i in range(4)]
    dataset = Dataset.from_examples(examples)
    capture = WorkflowStep(
        name="cap",
        runner="capture",
        spec=CaptureSpec(
            engine=ToyEngine(),
            dataset=dataset,
            sites=(ResidualSite(name="r", site="post", layers=(0,)),),
            generation=GenerationSpec(max_tokens=2),
        ),
    )
    probe = WorkflowStep(
        name="probe",
        runner="analysis",
        spec=ProbeSpec(
            feature=StepRef(step="cap").feature("r"),
            labels=StepRef(step="cap").label("cls"),
            tokens=TokenSelector.section("STRATEGY"),
            pooling=TokenPooling.mean(),
        ),
    )
    workflow = WorkflowSpec(name="section_wf", steps=(capture, probe))

    root = tmp_path / "catalog"
    local = FileCatalog(root=root)
    composite = CompositeCatalog(catalogs=(local,))
    run = _make_run_record(run_id="run_sec", workflow_payload=workflow.to_dict(), started_at="2026-04-15T00:00:00Z")
    local.record_workflow_run(run)
    local.record_workflow_step(_make_step_record(run_id="run_sec", step_index=0, step_name="cap", runner="capture"))
    local.record_workflow_step(_make_step_record(run_id="run_sec", step_index=1, step_name="probe", runner="analysis"))
    dash = DashboardCatalog(local=local, composite=composite, raw=composite, local_root=root, postgres_env=None)

    app = create_app(catalog=dash)
    client = TestClient(app)

    resp = client.get("/api/runs/run_sec/steps/probe/prompt-preview")
    assert resp.status_code == 200
    payload = resp.json()
    # Should be a hard unresolved state — no fake spans invented.
    assert payload["available"] is False
    assert "STRATEGY" in (payload["reason"] or "") or "section" in (payload["reason"] or "").lower()


# ---------------------------------------------------------------------------
# Phase F — reports
# ---------------------------------------------------------------------------


def test_report_detail_reads_local_report_root(tmp_path: Path) -> None:
    """Fabricate an on-disk report directory and a matching manifest and verify
    the endpoint reads it without touching anything outside the root."""
    import json as _json

    from fastapi.testclient import TestClient
    from pipelines_v2.dashboard.server import create_app
    from pipelines_v2.storage.artifacts import ArtifactManifest
    from pipelines_v2.storage.local import FileCatalog
    from pipelines_v2.storage.composite import CompositeCatalog
    from pipelines_v2.dashboard.catalog import DashboardCatalog

    report_root = tmp_path / "reports" / "artifact_r1"
    (report_root / "assets").mkdir(parents=True)
    (report_root / "tables").mkdir(parents=True)
    (report_root / "results").mkdir(parents=True)

    # Write a PNG placeholder and a tables JSON + manifest.
    (report_root / "assets" / "probe.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (report_root / "tables" / "probe.json").write_text(_json.dumps(
        {"rows": [{"metric": "acc", "value": 0.8}, {"metric": "f1", "value": 0.75}],
         "step_name": "probe", "result_kind": "probe_result"}))
    (report_root / "results" / "probe_results.json").write_text(_json.dumps({"kind": "probe_result"}))
    (report_root / "report.json").write_text(_json.dumps(
        {"template": "summary", "step_summaries": {"probe": {"headline_metrics": {"acc": 0.8}}}}))
    (report_root / "summary.json").write_text(_json.dumps(
        {"step_summaries": {"probe": {"headline_metrics": {"acc": 0.8}}}}))
    (report_root / "assets" / "manifest.json").write_text(_json.dumps({
        "figures": {
            "probe/main": {
                "figure_id": "probe/main",
                "path": "assets/probe.png",
                "step_name": "probe",
                "result_kind": "probe_result",
                "chart_kind": "bar",
                "title": "Probe accuracy",
                "caption": "accuracy per class",
                "primary": True,
            }
        },
        "tables": {
            "probe": {
                "path": "tables/probe.json",
                "step_name": "probe",
                "result_kind": "probe_result",
            }
        },
        "unsupported_inputs": [],
    }))

    # Register a report-kind manifest pointing at that root.
    catalog_root = tmp_path / "catalog"
    local = FileCatalog(root=catalog_root)
    composite = CompositeCatalog(catalogs=(local,))
    manifest = ArtifactManifest.from_dict({
        "artifact_id": "artifact_r1",
        "artifact_kind": "report",
        "schema_version": 1,
        "operation_spec_hash": "sh",
        "operation_semantic_hash": "sh",
        "created_at": "2026-04-15T00:00:00Z",
        "engine": {},
        "runner": {},
        "input_artifact_refs": [],
        "example_coverage": {},
        "storage_refs": {},
        "metadata": {"published_report": {"output_dir": str(report_root)}},
        "workflow_context": {"run_id": "run_report"},
    })
    local.record_artifact(manifest)

    dash = DashboardCatalog(local=local, composite=composite, raw=composite, local_root=catalog_root, postgres_env=None)
    app = create_app(catalog=dash)
    client = TestClient(app)

    resp = client.get("/api/reports/artifact_r1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "run_report"
    assert [f["figure_id"] for f in data["figures"]] == ["probe/main"]
    assert data["tables"][0]["rows"] == 2
    assert data["tables"][0]["columns"] == ["metric", "value"]
    assert data["results"][0]["name"] == "probe_results.json"

    # Asset serving: valid path returns bytes.
    resp = client.get("/api/reports/artifact_r1/assets/assets/probe.png")
    assert resp.status_code == 200
    assert resp.content.startswith(b"\x89PNG")

    # Path traversal is rejected.
    resp = client.get("/api/reports/artifact_r1/assets/../../outside.txt")
    assert resp.status_code in (400, 404)


def test_report_detail_404_for_non_report_artifact(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from pipelines_v2.dashboard.server import create_app
    from pipelines_v2.storage.artifacts import ArtifactManifest
    from pipelines_v2.storage.local import FileCatalog
    from pipelines_v2.storage.composite import CompositeCatalog
    from pipelines_v2.dashboard.catalog import DashboardCatalog

    catalog_root = tmp_path / "catalog"
    local = FileCatalog(root=catalog_root)
    composite = CompositeCatalog(catalogs=(local,))
    # A probe artifact (wrong kind) should be 404, not 500.
    manifest = ArtifactManifest.from_dict({
        "artifact_id": "not_a_report",
        "artifact_kind": "probe",
        "schema_version": 1,
        "operation_spec_hash": "h", "operation_semantic_hash": "h",
        "created_at": "2026-04-15T00:00:00Z",
        "engine": {}, "runner": {}, "input_artifact_refs": [],
        "example_coverage": {}, "storage_refs": {}, "metadata": {}, "workflow_context": {},
    })
    local.record_artifact(manifest)

    dash = DashboardCatalog(local=local, composite=composite, raw=composite, local_root=catalog_root, postgres_env=None)
    app = create_app(catalog=dash)
    client = TestClient(app)

    resp = client.get("/api/reports/not_a_report")
    assert resp.status_code == 404
