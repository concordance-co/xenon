from __future__ import annotations

import os
from pathlib import Path

from pipelines_v2.api import NullCatalog, PostgresCatalog, PostgresSource
from pipelines_v2.core.config import load_workspace_config


def build_prompt_confusion_catalog(start: str | Path | None = None):
    config = load_workspace_config(Path(start).resolve() if start is not None else Path(__file__).resolve())
    env_var = config.workflow_catalog_postgres_env()
    if env_var and os.environ.get(env_var):
        return PostgresCatalog(source=PostgresSource.from_env(env_var))
    return NullCatalog()
