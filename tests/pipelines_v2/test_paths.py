from __future__ import annotations

from pathlib import Path

import pytest

from pipelines_v2.core.paths import find_workspace_root, resolve_workspace_path


def test_workspace_root_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    override = tmp_path / "xenon-checkout"
    override.mkdir()
    monkeypatch.setenv("XENON_WORKSPACE_ROOT", str(override))
    assert find_workspace_root() == override.resolve()
    assert find_workspace_root(start=tmp_path) == override.resolve()


def test_workspace_root_env_override_expands_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XENON_WORKSPACE_ROOT", "~/xenon")
    assert find_workspace_root() == (tmp_path / "xenon").resolve()


def test_workspace_root_detection_without_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XENON_WORKSPACE_ROOT", raising=False)
    root = find_workspace_root()
    assert (root / "pyproject.toml").exists() or (root / ".git").exists()
    assert (root / "pipelines_v2").is_dir()


def test_resolve_workspace_path_uses_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XENON_WORKSPACE_ROOT", str(tmp_path))
    assert resolve_workspace_path("pipelines_v2") == (tmp_path / "pipelines_v2").resolve()
