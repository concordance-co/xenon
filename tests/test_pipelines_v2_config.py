from __future__ import annotations

import argparse
from pathlib import Path

from pipelines_v2.cli import _build_runners
from pipelines_v2.core.config import load_workspace_config
from pipelines_v2.dashboard.catalog import build_catalog
from pipelines_v2.runtime.specs import LocalRunnerSpec
from pipelines_v2.storage.local import FileCatalog, LocalArtifactStore


def _write_workspace(tmp_path: Path, *, config_text: str) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'tmp'\nversion = '0.0.0'\n", encoding="utf-8")
    (tmp_path / "xenon.toml").write_text(config_text, encoding="utf-8")


def test_load_workspace_config_reads_repo_defaults(tmp_path: Path) -> None:
    _write_workspace(
        tmp_path,
        config_text="""
[pipelines_v2]
catalog_postgres_env = "TEST_EXTERNAL_DB"
local_catalog_root = "state/catalog"

[pipelines_v2.dashboard]
use_workspace_catalog = true
static_dir = "dashboard/dist"
""".strip(),
    )

    config = load_workspace_config(tmp_path)

    assert config.config_path == (tmp_path / "xenon.toml")
    assert config.workflow_catalog_postgres_env() == "TEST_EXTERNAL_DB"
    assert config.workflow_local_catalog_root() == (tmp_path / "state" / "catalog").resolve()
    assert config.dashboard_catalog_postgres_env() == "TEST_EXTERNAL_DB"
    assert config.dashboard_static_dir() == (tmp_path / "dashboard" / "dist").resolve()


def test_dashboard_build_catalog_uses_workspace_config_defaults(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_workspace(
        tmp_path,
        config_text="""
[pipelines_v2]
catalog_postgres_env = "TEST_EXTERNAL_DB"
local_catalog_root = "state/catalog"
""".strip(),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_EXTERNAL_DB", "postgresql://example.test/xenon")

    dash = build_catalog()

    assert dash.postgres_env == "TEST_EXTERNAL_DB"
    assert dash.local_root == (tmp_path / "state" / "catalog").resolve()
    assert [getattr(catalog, "kind", None) for catalog in dash.raw.catalogs] == ["file", "postgres"]
    dash.close()


def test_cli_workspace_catalog_default_applies_when_runner_specs_leave_catalog_unset(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_workspace(
        tmp_path,
        config_text="""
[pipelines_v2]
catalog_postgres_env = "TEST_EXTERNAL_DB"
""".strip(),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_EXTERNAL_DB", "postgresql://example.test/xenon")

    ns = argparse.Namespace(
        file=None,
        catalog_postgres_env=None,
        local_catalog_root=None,
    )
    runners = _build_runners(
        ns,
        {
            "capture_gpu": LocalRunnerSpec(artifacts=LocalArtifactStore(tmp_path / "capture")),
            "analysis_cpu": LocalRunnerSpec(artifacts=LocalArtifactStore(tmp_path / "analysis")),
        },
    )

    capture_identity = runners["capture_gpu"].catalog.identity()
    assert capture_identity["kind"] == "composite"
    assert [catalog["kind"] for catalog in capture_identity["catalogs"]] == ["file", "postgres"]


def test_cli_workspace_catalog_default_does_not_override_explicit_workflow_catalog(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_workspace(
        tmp_path,
        config_text="""
[pipelines_v2]
catalog_postgres_env = "TEST_EXTERNAL_DB"
""".strip(),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_EXTERNAL_DB", "postgresql://example.test/xenon")

    explicit_catalog = FileCatalog(tmp_path / "workflow_catalog")
    ns = argparse.Namespace(
        file=None,
        catalog_postgres_env=None,
        local_catalog_root=None,
    )
    runners = _build_runners(
        ns,
        {
            "capture_gpu": LocalRunnerSpec(
                artifacts=LocalArtifactStore(tmp_path / "capture"),
                catalog=explicit_catalog,
            ),
            "analysis_cpu": LocalRunnerSpec(artifacts=LocalArtifactStore(tmp_path / "analysis")),
        },
    )

    capture_identity = runners["capture_gpu"].catalog.identity()
    assert capture_identity["kind"] == "composite"
    assert all(catalog["kind"] != "postgres" for catalog in capture_identity["catalogs"])
