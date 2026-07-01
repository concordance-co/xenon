"""Shared storage layout for reusable model-vector assets."""

from __future__ import annotations

from pathlib import Path


ARTIFACT_VOLUME_NAME = "xenon-data"
XENON_MODEL_VOLUME_NAME = "xenon-models"
YORA_MODEL_VOLUME_NAME = "yora-models"
MODEL_VOLUME_PATH = "/models"

REMOTE_ARTIFACT_ROOT = "/data/artifacts"
LOCAL_ARTIFACT_ROOT = Path("artifacts")
MODEL_ASSET_VECTOR_NAMESPACE = ("model-assets", "vectors")


def modal_vector_root(*parts: str) -> str:
    """Return a Modal artifact path under shared model-vector storage."""

    return "/".join(
        (
            REMOTE_ARTIFACT_ROOT.rstrip("/"),
            *MODEL_ASSET_VECTOR_NAMESPACE,
            *_clean_parts(parts),
        )
    )


def local_vector_root(*parts: str) -> Path:
    """Return the matching local artifact path for vector workflow reports."""

    path = LOCAL_ARTIFACT_ROOT
    for part in (*MODEL_ASSET_VECTOR_NAMESPACE, *_clean_parts(parts)):
        path /= part
    return path


def _clean_parts(parts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(part.strip("/") for part in parts if part and part.strip("/"))
