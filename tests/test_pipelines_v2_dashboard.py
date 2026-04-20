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
from pipelines_v2.dashboard.pg import DashboardPg
from pipelines_v2.dashboard.result_preview import read_result_payload
from pipelines_v2.storage.artifacts import ArtifactManifest


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

    # Write all three to the primary (preferred) catalog so the preferred
    # fast-path returns the full set. CompositeCatalog now prefers the local
    # file catalog for workflow metadata when present.
    for record in (older, middle, newer):
        primary.record_workflow_run(record)
    # Write a duplicate into secondary to verify dedupe still works when
    # the preferred path isn't taken.
    secondary.record_workflow_run(middle)

    composite = CompositeCatalog(catalogs=(primary, secondary))
    runs = composite.list_workflow_runs()

    run_ids = [r.run_id for r in runs]
    assert run_ids == ["r3", "r2", "r1"], "newest-first order"
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


def test_run_bundle_cache_is_app_scoped(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from pipelines_v2.dashboard.server import create_app

    same_run_id = "shared_run"
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first_local = FileCatalog(root=first_root)
    first_composite = CompositeCatalog(catalogs=(first_local,))
    first_workflow = _workflow_payload([_capture_step_payload("cap")], name="wf_one")
    first_local.record_workflow_run(
        _make_run_record(
            run_id=same_run_id,
            workflow_payload=first_workflow,
            started_at="2026-04-15T00:00:00Z",
        )
    )
    first_local.record_workflow_step(
        _make_step_record(run_id=same_run_id, step_index=0, step_name="cap", runner="capture")
    )

    second_local = FileCatalog(root=second_root)
    second_composite = CompositeCatalog(catalogs=(second_local,))
    second_workflow = _workflow_payload([_capture_step_payload("cap")], name="wf_two")
    second_local.record_workflow_run(
        _make_run_record(
            run_id=same_run_id,
            workflow_payload=second_workflow,
            started_at="2026-04-16T00:00:00Z",
        )
    )
    second_local.record_workflow_step(
        _make_step_record(run_id=same_run_id, step_index=0, step_name="cap", runner="capture")
    )

    first_app = create_app(
        catalog=DashboardCatalog(
            local=first_local,
            composite=first_composite,
            raw=first_composite,
            local_root=first_root,
            postgres_env=None,
        )
    )
    second_app = create_app(
        catalog=DashboardCatalog(
            local=second_local,
            composite=second_composite,
            raw=second_composite,
            local_root=second_root,
            postgres_env=None,
        )
    )

    first_client = TestClient(first_app)
    second_client = TestClient(second_app)

    first_resp = first_client.get(f"/api/runs/{same_run_id}")
    second_resp = second_client.get(f"/api/runs/{same_run_id}")

    assert first_resp.status_code == 200
    assert second_resp.status_code == 200
    assert first_resp.json()["run"]["workflow_name"] == "wf_one"
    assert second_resp.json()["run"]["workflow_name"] == "wf_two"


def test_run_bundle_builder_state_cleans_up_after_failure(tmp_path: Path) -> None:
    from pipelines_v2.dashboard import server as dashboard_server
    from pipelines_v2.dashboard.server import create_app

    app = create_app(catalog=_seed_catalog(tmp_path))
    runtime_state = app.state.dashboard_runtime
    dash = app.state.dashboard_catalog

    original = dashboard_server.build_run_detail
    dashboard_server.build_run_detail = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        with pytest.raises(RuntimeError, match="boom"):
            dashboard_server._load_run_bundle(runtime_state, dash, "run_abc")
    finally:
        dashboard_server.build_run_detail = original

    assert runtime_state.run_bundle_building == {}
    bundle = dashboard_server._load_run_bundle(runtime_state, dash, "run_abc")
    assert bundle.run.run_id == "run_abc"


def test_run_detail_prefers_fresher_step_records_over_longer_local_list(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from pipelines_v2.dashboard.server import create_app

    root = tmp_path / "catalog"
    local = FileCatalog(root=root)
    composite = CompositeCatalog(catalogs=(local,))
    workflow = _workflow_payload(
        [
            _capture_step_payload("cap"),
            _probe_step_payload("probe", depends_on=["cap"]),
        ],
        name="merge_demo",
    )
    run = _make_run_record(
        run_id="run_merge",
        workflow_payload=workflow,
        started_at="2026-04-15T00:00:00Z",
        status="running",
    )
    local.record_workflow_run(run)
    stale_cap = _make_step_record(
        run_id="run_merge",
        step_index=0,
        step_name="cap",
        runner="capture",
        status="completed",
    )
    stale_probe = _make_step_record(
        run_id="run_merge",
        step_index=1,
        step_name="probe",
        runner="analysis",
        status="completed",
    )
    local.record_workflow_step(stale_cap)
    local.record_workflow_step(stale_probe)

    fresher_cap = WorkflowStepRecord(
        run_id="run_merge",
        workflow_hash="wh_run_merge",
        workflow_step_key="wh_run_merge.cap",
        step_name="cap",
        step_index=0,
        runner="capture",
        status="failed",
        step_semantic_hash="ss_cap",
        step_spec_hash="sp_cap",
        started_at="2026-04-15T00:00:00Z",
        finished_at="2026-04-15T00:02:00Z",
    )

    class FakePg:
        def list_workflow_steps(self, run_id: str) -> list[WorkflowStepRecord]:
            return [fresher_cap] if run_id == "run_merge" else []

        def load_workflow_run(self, _run_id: str) -> None:
            return None

        def load_artifact(self, _artifact_id: str) -> None:
            return None

        def find_artifact_for_workflow_step(self, **_kwargs: Any) -> None:
            return None

        def close(self) -> None:
            pass

        def stats(self) -> dict[str, Any]:
            return {"pool": {}}

    dash = DashboardCatalog(
        local=local,
        composite=composite,
        raw=composite,
        local_root=root,
        postgres_env=None,
        pg=FakePg(),  # type: ignore[arg-type]
    )
    app = create_app(catalog=dash)
    client = TestClient(app)

    resp = client.get("/api/runs/run_merge")
    assert resp.status_code == 200
    steps = {step["step_name"]: step for step in resp.json()["steps"]}
    assert steps["cap"]["status"] == "failed"
    assert steps["probe"]["status"] == "completed"


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


def test_file_catalog_light_run_summaries_backfill_and_reuse_sidecars(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    local = FileCatalog(root=root)
    workflow = _workflow_payload(
        [
            _capture_step_payload("cap"),
            _probe_step_payload("probe", depends_on=["cap"]),
        ],
        name="light_runs",
    )
    local.record_workflow_run(
        _make_run_record(
            run_id="run_light",
            workflow_payload=workflow,
            started_at="2026-04-15T00:00:00Z",
            status="running",
        )
    )
    local.record_workflow_step(
        _make_step_record(
            run_id="run_light",
            step_index=0,
            step_name="cap",
            runner="capture",
            status="completed",
        )
    )

    summary_path = root / "workflow_run_summaries" / "run_light.json"
    assert summary_path.is_file()

    rows = local.list_workflow_runs_light()
    assert rows[0]["run_id"] == "run_light"
    assert rows[0]["step_total"] == 2
    assert rows[0]["step_counts"]["completed"] == 1

    # The light listing should keep working from the sidecar even if the full
    # workflow payload file is unreadable.
    (root / "workflow_runs" / "run_light.json").write_text("{not valid json", encoding="utf-8")
    rows = local.list_workflow_runs_light()
    assert rows[0]["run_id"] == "run_light"


def test_read_result_payload_accepts_local_store_refs(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text('{"summary":{"ok":true}}', encoding="utf-8")
    manifest = ArtifactManifest(
        artifact_id="report_local_test",
        artifact_kind="report",
        schema_version=1,
        operation_spec_hash="spec",
        operation_semantic_hash="semantic",
        created_at="2026-04-15T00:00:00Z",
        engine={},
        runner={"kind": "local"},
        input_artifact_refs=(),
        example_coverage={},
        storage_refs={
            "result": {
                "store": "local",
                "path": str(result_path),
                "format": "json",
                "bytes": result_path.stat().st_size,
            }
        },
        metadata={},
        workflow_context={},
    )

    preview = read_result_payload(
        artifact_manifest=manifest,
        report_manifest=None,
        step_name="report",
    )

    assert preview.available is True
    assert preview.path == str(result_path)
    assert preview.payload["summary"]["ok"] is True


def test_read_result_payload_truncates_large_local_results(tmp_path: Path) -> None:
    import json as _json

    result_path = tmp_path / "large_result.json"
    result_path.write_text(
        _json.dumps({"rows": [{"value": "x" * 2_000_000}]}),
        encoding="utf-8",
    )
    manifest = ArtifactManifest(
        artifact_id="large_result_test",
        artifact_kind="probe",
        schema_version=1,
        operation_spec_hash="spec",
        operation_semantic_hash="semantic",
        created_at="2026-04-15T00:00:00Z",
        engine={},
        runner={"kind": "local"},
        input_artifact_refs=(),
        example_coverage={},
        storage_refs={
            "result": {
                "store": "local",
                "path": str(result_path),
                "format": "json",
                "bytes": result_path.stat().st_size,
            }
        },
        metadata={},
        workflow_context={},
    )

    preview = read_result_payload(
        artifact_manifest=manifest,
        report_manifest=None,
        step_name="probe",
    )

    assert preview.available is True
    assert preview.truncated is True
    assert preview.bytes == result_path.stat().st_size
    assert "__dashboard_notice__" in (preview.payload or {})
    assert preview.tables == []


def test_dashboard_pg_retries_once_on_stale_connection_error() -> None:
    class FakeOperationalError(Exception):
        __module__ = "psycopg"

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql: str, params: tuple[Any, ...]) -> None:
            self.sql = sql
            self.params = params

        def fetchone(self):
            return ("ok",)

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return FakeCursor()

    class FakePool:
        def __init__(self) -> None:
            self.calls = 0

        def connection(self):
            self.calls += 1
            if self.calls == 1:
                raise FakeOperationalError("discarding closed connection: <psycopg.Connection [BAD]>")
            return FakeConn()

    pg = DashboardPg.__new__(DashboardPg)
    pg._pool = FakePool()

    row = pg._fetchone("SELECT 1", ())

    assert row == ("ok",)
    assert pg._pool.calls == 2


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


def test_bulk_steps_detail_includes_pending_workflow_steps(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from pipelines_v2.dashboard.server import create_app
    from pipelines_v2.storage.composite import CompositeCatalog
    from pipelines_v2.storage.local import FileCatalog

    root = tmp_path / "catalog"
    local = FileCatalog(root=root)
    composite = CompositeCatalog(catalogs=(local,))
    wf = _workflow_payload(
        [
            _capture_step_payload("cap"),
            _probe_step_payload("probe", depends_on=["cap"]),
        ]
    )
    run = _make_run_record(
        run_id="run_pending",
        workflow_payload=wf,
        started_at="2026-04-15T00:00:00Z",
        status="running",
    )
    local.record_workflow_run(run)
    local.record_workflow_step(
        _make_step_record(run_id="run_pending", step_index=0, step_name="cap", runner="capture")
    )

    app = create_app(catalog=DashboardCatalog(local=local, composite=composite, raw=composite, local_root=root, postgres_env=None))
    client = TestClient(app)

    resp = client.get("/api/runs/run_pending/steps-detail")
    assert resp.status_code == 200
    payload = resp.json()
    assert [d["step"]["step_name"] for d in payload["step_details"]] == ["cap", "probe"]
    assert payload["step_details"][1]["step"]["status"] == "pending"


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


def test_prompt_preview_reuses_cached_tokenizer_loads(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types

    from fastapi.testclient import TestClient

    from pipelines_v2.api import CaptureSpec, Dataset, Example, GenerationSpec, VLLMEngine
    from pipelines_v2.dashboard.catalog import DashboardCatalog
    from pipelines_v2.dashboard.server import create_app
    from pipelines_v2.operations.capture.sites import ResidualSite
    from pipelines_v2.storage.composite import CompositeCatalog
    from pipelines_v2.storage.local import FileCatalog
    from pipelines_v2.workflow.specs import WorkflowSpec, WorkflowStep

    calls = {"count": 0}

    class _FakeTokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
            return {
                "offset_mapping": [(0, len(text))],
            }

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, trust_remote_code: bool = False):
            calls["count"] += 1
            return _FakeTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=_FakeAutoTokenizer))

    dataset = Dataset.from_examples([Example(key="a", prompt="hello", labels={"cls": "x"})])
    capture = WorkflowStep(
        name="cap",
        runner="capture",
        spec=CaptureSpec(
            engine=VLLMEngine(model_id="fake/model"),
            dataset=dataset,
            sites=(ResidualSite(name="resid", site="post", layers=(0,)),),
            generation=GenerationSpec(max_tokens=1),
        ),
    )
    workflow = WorkflowSpec(name="tokenizer_cache", steps=(capture,))

    root = tmp_path / "catalog"
    local = FileCatalog(root=root)
    composite = CompositeCatalog(catalogs=(local,))
    run = _make_run_record(run_id="run_tok", workflow_payload=workflow.to_dict(), started_at="2026-04-15T00:00:00Z")
    local.record_workflow_run(run)
    local.record_workflow_step(_make_step_record(run_id="run_tok", step_index=0, step_name="cap", runner="capture"))
    dash = DashboardCatalog(local=local, composite=composite, raw=composite, local_root=root, postgres_env=None)

    app = create_app(catalog=dash)
    client = TestClient(app)

    first = client.get("/api/runs/run_tok/steps/cap/prompt-preview")
    second = client.get("/api/runs/run_tok/steps/cap/prompt-preview")

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls["count"] == 1


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


def test_report_detail_uses_manifest_table_metadata_without_table_file(tmp_path: Path) -> None:
    import json as _json

    from fastapi.testclient import TestClient
    from pipelines_v2.dashboard.server import create_app
    from pipelines_v2.storage.local import FileCatalog
    from pipelines_v2.storage.composite import CompositeCatalog

    report_root = tmp_path / "reports" / "artifact_r2"
    (report_root / "assets").mkdir(parents=True)
    (report_root / "report.json").write_text(_json.dumps({"template": "summary"}))
    (report_root / "summary.json").write_text(_json.dumps({"template": "summary"}))
    (report_root / "assets" / "manifest.json").write_text(
        _json.dumps(
            {
                "figures": {},
                "tables": {
                    "probe": {
                        "path": "tables/probe.json",
                        "step_name": "probe",
                        "result_kind": "probe_result",
                        "rows": 2,
                        "columns": ["metric", "value"],
                    }
                },
                "unsupported_inputs": [],
            }
        )
    )

    catalog_root = tmp_path / "catalog_meta"
    local = FileCatalog(root=catalog_root)
    composite = CompositeCatalog(catalogs=(local,))
    manifest = ArtifactManifest.from_dict(
        {
            "artifact_id": "artifact_r2",
            "artifact_kind": "report",
            "schema_version": 1,
            "operation_spec_hash": "h",
            "operation_semantic_hash": "h",
            "created_at": "2026-04-15T00:00:00Z",
            "engine": {},
            "runner": {},
            "input_artifact_refs": [],
            "example_coverage": {},
            "storage_refs": {},
            "metadata": {"published_report": {"output_dir": str(report_root)}},
            "workflow_context": {"run_id": "run_report_meta"},
        }
    )
    local.record_artifact(manifest)

    app = create_app(
        catalog=DashboardCatalog(
            local=local,
            composite=composite,
            raw=composite,
            local_root=catalog_root,
            postgres_env=None,
        )
    )
    client = TestClient(app)

    resp = client.get("/api/reports/artifact_r2")
    assert resp.status_code == 200
    table = resp.json()["tables"][0]
    assert table["rows"] == 2
    assert table["columns"] == ["metric", "value"]


def test_generate_report_materializes_local_report_and_updates_run_detail(tmp_path: Path) -> None:
    import json as _json

    from fastapi.testclient import TestClient

    from pipelines_v2.dashboard.server import create_app
    from pipelines_v2.data.datasets import Dataset, Example
    from pipelines_v2.engine.toy import ToyEngine
    from pipelines_v2.operations.capture.sites import ResidualSite
    from pipelines_v2.operations.capture.specs import CaptureSpec, GenerationSpec
    from pipelines_v2.operations.reports import ReportSpec
    from pipelines_v2.workflow.specs import StepRef, WorkflowSpec, WorkflowStep

    catalog_root = tmp_path / "catalog"
    local = FileCatalog(root=catalog_root)
    composite = CompositeCatalog(catalogs=(local,))

    source_artifacts_root = tmp_path / "source_artifacts"
    source_artifact_id = "cap_seed"
    source_artifact_root = source_artifacts_root / source_artifact_id
    source_artifact_root.mkdir(parents=True)
    (source_artifact_root / "manifest.json").write_text(_json.dumps({"artifact_id": source_artifact_id}))

    report_output_root = tmp_path / "report_outputs"
    workflow_payload = WorkflowSpec(
        name="reportable_demo",
        steps=(
            WorkflowStep(
                name="cap",
                runner="capture",
                spec=CaptureSpec(
                    engine=ToyEngine(),
                    dataset=Dataset.from_examples(
                        (Example(key="ex1", prompt="demo prompt", labels={"cls": "a"}),)
                    ),
                    sites=(ResidualSite(name="resid", site="post", layers=(0,)),),
                    generation=GenerationSpec(max_tokens=0),
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report",
                spec=ReportSpec(
                    inputs=(StepRef(step="cap"),),
                    template="summary",
                    output_dir=str(report_output_root),
                ),
                depends_on=("cap",),
            ),
        ),
    )
    run = _make_run_record(
        run_id="run_reportable",
        workflow_payload=workflow_payload.to_dict(),
        started_at="2026-04-15T00:00:00Z",
        status="completed",
        finished_at="2026-04-15T00:05:00Z",
    )
    local.record_workflow_run(run)
    local.record_workflow_step(
        _make_step_record(
            run_id="run_reportable",
            step_index=0,
            step_name="cap",
            runner="capture",
            artifact_id=source_artifact_id,
            artifact_kind="capture",
        )
    )
    local.record_workflow_step(
        _make_step_record(
            run_id="run_reportable",
            step_index=1,
            step_name="report",
            runner="report",
            status="pending",
            artifact_id=None,
            artifact_kind=None,
        )
    )
    local.record_artifact(
        ArtifactManifest.from_dict(
            {
                "artifact_id": source_artifact_id,
                "artifact_kind": "capture",
                "schema_version": 1,
                "operation_spec_hash": "cap_hash",
                "operation_semantic_hash": "cap_hash",
                "created_at": "2026-04-15T00:01:00Z",
                "engine": {},
                "runner": {"kind": "local"},
                "input_artifact_refs": [],
                "example_coverage": {},
                "storage_refs": {
                    "manifest": {
                        "store": "local_path",
                        "path": str(source_artifact_root / "manifest.json"),
                        "format": "json",
                    }
                },
                "metadata": {},
                "workflow_context": {
                    "run_id": "run_reportable",
                    "workflow_name": "reportable_demo",
                    "workflow_hash": "wh_run_reportable",
                    "workflow_step_key": "wh_run_reportable.cap",
                    "step_name": "cap",
                    "step_index": 0,
                    "runner": "capture",
                    "step_semantic_hash": "ss_cap",
                    "step_spec_hash": "sp_cap",
                },
            }
        )
    )

    dash = DashboardCatalog(local=local, composite=composite, raw=composite, local_root=catalog_root, postgres_env=None)
    app = create_app(catalog=dash)
    client = TestClient(app)

    before = client.get("/api/runs/run_reportable")
    assert before.status_code == 200
    assert before.json()["report"]["has_report_step"] is True
    assert before.json()["report"]["local_available"] is False

    resp = client.post("/api/runs/run_reportable/report")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["run_id"] == "run_reportable"
    assert payload["step_name"] == "report"
    assert payload["report"]["run_id"] == "run_reportable"
    assert payload["report"]["report"]["template"] == "summary"

    generated_artifact_id = payload["artifact_id"]
    detail = client.get(f"/api/reports/{generated_artifact_id}")
    assert detail.status_code == 200
    assert detail.json()["artifact_id"] == generated_artifact_id

    after = client.get("/api/runs/run_reportable")
    assert after.status_code == 200
    assert after.json()["report"]["local_available"] is True
    assert after.json()["report"]["artifact_id"] == generated_artifact_id


def test_generate_report_falls_back_to_raw_catalog_when_cached_catalog_misses(tmp_path: Path) -> None:
    import json as _json

    from fastapi.testclient import TestClient

    from pipelines_v2.dashboard.server import create_app
    from pipelines_v2.data.datasets import Dataset, Example
    from pipelines_v2.engine.toy import ToyEngine
    from pipelines_v2.operations.capture.sites import ResidualSite
    from pipelines_v2.operations.capture.specs import CaptureSpec, GenerationSpec
    from pipelines_v2.operations.reports import ReportSpec
    from pipelines_v2.workflow.specs import StepRef, WorkflowSpec, WorkflowStep

    class MissCatalog:
        kind = "miss"

        def identity(self) -> dict[str, str]:
            return {"kind": self.kind}

        def load_workflow_run(self, _run_id: str) -> None:
            return None

        def list_workflow_steps(self, _run_id: str) -> list[Any]:
            return []

        def load_artifact(self, _artifact_id: str) -> None:
            return None

        def find_artifact_for_workflow_step(
            self,
            *,
            run_id: str,  # noqa: ARG002 - interface compatibility
            workflow_step_key: str,  # noqa: ARG002 - interface compatibility
        ) -> None:
            return None

    catalog_root = tmp_path / "catalog"
    local = FileCatalog(root=catalog_root)
    remote = FileCatalog(root=tmp_path / "remote_catalog")
    raw = CompositeCatalog(catalogs=(local, remote))

    source_artifacts_root = tmp_path / "remote_artifacts"
    source_artifact_id = "cap_remote_seed"
    source_artifact_root = source_artifacts_root / source_artifact_id
    source_artifact_root.mkdir(parents=True)
    (source_artifact_root / "manifest.json").write_text(_json.dumps({"artifact_id": source_artifact_id}))

    workflow_payload = WorkflowSpec(
        name="remote_reportable_demo",
        steps=(
            WorkflowStep(
                name="cap",
                runner="capture",
                spec=CaptureSpec(
                    engine=ToyEngine(),
                    dataset=Dataset.from_examples(
                        (Example(key="ex1", prompt="demo prompt", labels={"cls": "a"}),)
                    ),
                    sites=(ResidualSite(name="resid", site="post", layers=(0,)),),
                    generation=GenerationSpec(max_tokens=0),
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report",
                spec=ReportSpec(
                    inputs=(StepRef(step="cap"),),
                    template="summary",
                    output_dir=str(tmp_path / "report_outputs"),
                ),
                depends_on=("cap",),
            ),
        ),
    )
    run = _make_run_record(
        run_id="run_reportable_remote",
        workflow_payload=workflow_payload.to_dict(),
        started_at="2026-04-15T00:00:00Z",
        status="completed",
        finished_at="2026-04-15T00:05:00Z",
    )
    remote.record_workflow_run(run)
    remote.record_workflow_step(
        _make_step_record(
            run_id="run_reportable_remote",
            step_index=0,
            step_name="cap",
            runner="capture",
            artifact_id=source_artifact_id,
            artifact_kind="capture",
        )
    )
    remote.record_workflow_step(
        _make_step_record(
            run_id="run_reportable_remote",
            step_index=1,
            step_name="report",
            runner="report",
            status="pending",
            artifact_id=None,
            artifact_kind=None,
        )
    )
    remote.record_artifact(
        ArtifactManifest.from_dict(
            {
                "artifact_id": source_artifact_id,
                "artifact_kind": "capture",
                "schema_version": 1,
                "operation_spec_hash": "cap_hash_remote",
                "operation_semantic_hash": "cap_hash_remote",
                "created_at": "2026-04-15T00:01:00Z",
                "engine": {},
                "runner": {"kind": "local"},
                "input_artifact_refs": [],
                "example_coverage": {},
                "storage_refs": {
                    "manifest": {
                        "store": "local_path",
                        "path": str(source_artifact_root / "manifest.json"),
                        "format": "json",
                    }
                },
                "metadata": {},
                "workflow_context": {
                    "run_id": "run_reportable_remote",
                    "workflow_name": "remote_reportable_demo",
                    "workflow_hash": "wh_run_reportable_remote",
                    "workflow_step_key": "wh_run_reportable_remote.cap",
                    "step_name": "cap",
                    "step_index": 0,
                    "runner": "capture",
                    "step_semantic_hash": "ss_cap",
                    "step_spec_hash": "sp_cap",
                },
            }
        )
    )

    dash = DashboardCatalog(
        local=local,
        composite=MissCatalog(),
        raw=raw,
        local_root=catalog_root,
        postgres_env=None,
    )
    app = create_app(catalog=dash)
    client = TestClient(app)

    detail = client.get("/api/runs/run_reportable_remote")
    assert detail.status_code == 200
    assert detail.json()["run"]["run_id"] == "run_reportable_remote"

    resp = client.post("/api/runs/run_reportable_remote/report")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["run_id"] == "run_reportable_remote"

    after = client.get("/api/runs/run_reportable_remote")
    assert after.status_code == 200
    assert after.json()["report"]["local_available"] is True
    assert after.json()["report"]["artifact_id"] == payload["artifact_id"]
