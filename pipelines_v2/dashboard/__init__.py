"""Read-only observability dashboard for pipelines_v2 runs."""

from pipelines_v2.dashboard.catalog import build_catalog, DashboardCatalog
from pipelines_v2.dashboard.server import create_app

__all__ = ["build_catalog", "DashboardCatalog", "create_app"]
