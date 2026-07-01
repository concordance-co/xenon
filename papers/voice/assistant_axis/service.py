"""Compatibility Modal entrypoint for the Assistant Axis service.

Prefer:

    uv run python -m pipelines_v2.cli deployment deploy --file papers/voice/assistant_axis/deployment.py --target prod

This module remains importable for existing `modal deploy
papers/voice/assistant_axis/service.py` workflows, but the service is now built
through the generic pipelines_v2 deployment runtime.
"""

from __future__ import annotations

from papers.voice.assistant_axis.deployment import build_deployment, build_deployment_targets
from pipelines_v2.runtime.modal_deployments import build_modal_app


app = build_modal_app(
    build_deployment(),
    build_deployment_targets()["prod"],
    target_name="prod",
)
