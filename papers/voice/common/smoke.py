"""Shared constants and fixtures for voice paper smoke workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipelines_v2.api import FileCatalog, LocalArtifactStore, LocalRunnerSpec
from pipelines_v2.operations.common.builders import TransformResult

DEFAULT_SMOKE_MODEL = "Qwen/Qwen3-8B"


def token_metadata(*names: str, token_count: int = 6) -> dict[str, Any]:
    """Return stable section metadata for tiny ToyEngine examples."""

    positions = list(range(int(token_count)))
    sections = {name: positions for name in names}
    return {
        "token_sections": sections,
        "section_records": [
            {
                "name": name,
                "unit": name,
                "index": index,
                "token_positions": positions,
            }
            for index, name in enumerate(names)
        ],
    }


def local_runner_specs(*, artifact_name: str) -> dict[str, object]:
    root = Path("artifacts") / "papers_voice" / artifact_name
    catalog = FileCatalog(root=root / "catalog")
    return {
        "capture_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(root / "artifacts"),
            catalog=catalog,
        ),
        "analysis_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(root / "artifacts"),
            catalog=catalog,
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(root / "artifacts"),
            catalog=catalog,
        ),
    }


def coordinate_to_direction(*, coordinate: Any, name: str = "voice_direction") -> TransformResult:
    payload = coordinate.result() if hasattr(coordinate, "result") else coordinate
    if not isinstance(payload, dict):
        raise TypeError(f"coordinate must resolve to a dict, got {type(payload).__name__}")
    layers = payload.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError("coordinate payload must contain non-empty layers")
    return TransformResult(
        payload={
            "kind": "direction_result",
            "feature": payload.get("feature"),
            "name": str(name),
            "layers": layers,
            "metadata": {
                **(payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}),
                "source": "papers.voice.common.smoke.coordinate_to_direction",
                "source_coordinate_kind": payload.get("kind"),
            },
            "summary": {
                **(payload.get("summary") if isinstance(payload.get("summary"), dict) else {}),
                "layer_count": len(layers),
            },
        },
        example_keys=[],
    )


__all__ = ["DEFAULT_SMOKE_MODEL", "coordinate_to_direction", "local_runner_specs", "token_metadata"]
