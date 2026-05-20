"""Workspace-root config for CLI and dashboard defaults."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from pipelines_v2.core.paths import find_workspace_root

CONFIG_FILENAME = "xenon.toml"


@dataclass(frozen=True, slots=True)
class WorkflowDefaults:
    """Workspace defaults for `pipelines_v2 workflow ...` commands."""

    catalog_postgres_env: str | None = None
    local_catalog_root: Path | None = None


@dataclass(frozen=True, slots=True)
class DashboardDefaults:
    """Workspace defaults for `pipelines_v2.dashboard serve`."""

    use_workspace_catalog: bool = True
    catalog_postgres_env: str | None = None
    local_catalog_root: Path | None = None
    static_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class ModalDefaults:
    """Workspace defaults for Modal-backed workflow runners."""

    model_volume: str | None = None
    model_volume_path: str = "/models"
    vllm_cache_volume: str | None = None
    vllm_cache_root: str | None = None
    use_vllm_torch_compile_cache: bool = False

    def resolved_vllm_cache_volume(self) -> str | None:
        if self.vllm_cache_volume is not None:
            return self.vllm_cache_volume
        if self.use_vllm_torch_compile_cache:
            return self.model_volume
        return None

    def resolved_vllm_cache_root(self) -> str | None:
        if self.vllm_cache_root is not None:
            return self.vllm_cache_root
        if self.use_vllm_torch_compile_cache:
            return self.model_volume_path
        return None


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    """Resolved workspace config plus convenience accessors."""

    workspace_root: Path
    config_path: Path | None = None
    catalog_postgres_env: str | None = None
    local_catalog_root: Path | None = None
    workflow: WorkflowDefaults = field(default_factory=WorkflowDefaults)
    dashboard: DashboardDefaults = field(default_factory=DashboardDefaults)
    modal: ModalDefaults = field(default_factory=ModalDefaults)

    def workflow_catalog_postgres_env(self) -> str | None:
        return self.workflow.catalog_postgres_env or self.catalog_postgres_env

    def workflow_local_catalog_root(self) -> Path | None:
        return self.workflow.local_catalog_root or self.local_catalog_root

    def dashboard_catalog_postgres_env(self) -> str | None:
        if self.dashboard.catalog_postgres_env is not None:
            return self.dashboard.catalog_postgres_env
        if self.dashboard.use_workspace_catalog:
            return self.catalog_postgres_env
        return None

    def dashboard_local_catalog_root(self) -> Path | None:
        return self.dashboard.local_catalog_root or self.local_catalog_root

    def dashboard_static_dir(self) -> Path | None:
        return self.dashboard.static_dir


def load_workspace_config(start: Path | None = None) -> WorkspaceConfig:
    """Load `xenon.toml` from the detected workspace root if present."""

    workspace_root = find_workspace_root(start or Path.cwd())
    config_path = workspace_root / CONFIG_FILENAME
    if not config_path.exists():
        return WorkspaceConfig(workspace_root=workspace_root)

    with config_path.open("rb") as f:
        payload = tomllib.load(f)

    section = _mapping(payload.get("pipelines_v2"))
    workflow_section = _mapping(section.get("workflow"))
    dashboard_section = _mapping(section.get("dashboard"))
    modal_section = _mapping(section.get("modal"))

    return WorkspaceConfig(
        workspace_root=workspace_root,
        config_path=config_path,
        catalog_postgres_env=_string(section.get("catalog_postgres_env")),
        local_catalog_root=_path(section.get("local_catalog_root"), workspace_root=workspace_root),
        workflow=WorkflowDefaults(
            catalog_postgres_env=_string(workflow_section.get("catalog_postgres_env")),
            local_catalog_root=_path(workflow_section.get("local_catalog_root"), workspace_root=workspace_root),
        ),
        dashboard=DashboardDefaults(
            use_workspace_catalog=_bool(dashboard_section.get("use_workspace_catalog"), default=True),
            catalog_postgres_env=_string(dashboard_section.get("catalog_postgres_env")),
            local_catalog_root=_path(dashboard_section.get("local_catalog_root"), workspace_root=workspace_root),
            static_dir=_path(dashboard_section.get("static_dir"), workspace_root=workspace_root),
        ),
        modal=ModalDefaults(
            model_volume=_string(modal_section.get("model_volume")),
            model_volume_path=_string(modal_section.get("model_volume_path")) or "/models",
            vllm_cache_volume=_string(modal_section.get("vllm_cache_volume")),
            vllm_cache_root=_string(modal_section.get("vllm_cache_root")),
            use_vllm_torch_compile_cache=_bool(
                modal_section.get("use_vllm_torch_compile_cache"),
                default=False,
            ),
        ),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected TOML table, got {type(value).__name__}")
    return value


def _string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _path(value: Any, *, workspace_root: Path) -> Path | None:
    text = _string(value)
    if text is None:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (workspace_root / path).resolve()
