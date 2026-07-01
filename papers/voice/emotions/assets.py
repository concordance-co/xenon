"""Packaged emotion-vector asset helpers."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines_v2.api import EmotionPrecomputedVectorSpaceSpec


PACKAGE_ROOT = Path(__file__).parents[3]
EMOTION_REPLICATION_ROOT = Path(__file__).with_name("replication")
EMOTION_LIST_PATH = EMOTION_REPLICATION_ROOT / "data" / "emotions.txt"
TOPIC_LIST_PATH = EMOTION_REPLICATION_ROOT / "data" / "topics.txt"
DEFAULT_LLAMA33_70B_MANIFEST = (
    Path(__file__).parents[1]
    / "model_assets"
    / "vectors"
    / "emotions"
    / "llama-3.3-70b"
    / "sofroniew-2026"
    / "v1"
    / "manifest.toml"
)


def load_asset_manifest(path: str | Path = DEFAULT_LLAMA33_70B_MANIFEST) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def emotion_concepts(*, mode: str = "full", manifest: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Return pilot or full concept names for the emotion vector asset."""

    payload = manifest or load_asset_manifest()
    if mode == "pilot":
        usage = _mapping(payload.get("usage"))
        return tuple(str(item) for item in usage.get("pilot_concepts", ()) if str(item).strip())
    if mode != "full":
        raise ValueError(f"unknown emotion asset mode: {mode!r}")
    return tuple(_load_lines(EMOTION_LIST_PATH))


def emotion_topics() -> tuple[str, ...]:
    return tuple(_load_lines(TOPIC_LIST_PATH))


def planned_asset_root(manifest: Mapping[str, Any] | None = None) -> str:
    payload = manifest or load_asset_manifest()
    return str(_mapping(payload.get("storage")).get("asset_root") or "")


def artifact_ids(manifest: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Return completed artifact ids recorded in the manifest, if any."""

    payload = manifest or load_asset_manifest()
    artifacts = _mapping(payload.get("artifacts"))
    return {str(key): str(value) for key, value in artifacts.items() if str(value).strip()}


def precomputed_vector_space_spec(
    manifest: Mapping[str, Any] | None = None,
    *,
    path: str | None = None,
) -> EmotionPrecomputedVectorSpaceSpec:
    """Build a precomputed-vector loader once the asset has been materialized."""

    payload = manifest or load_asset_manifest()
    vector_path = path or str(_mapping(payload.get("artifacts")).get("vector_space_path") or "")
    if not vector_path:
        raise ValueError("emotion vector-space asset is not materialized; manifest has no vector_space_path")
    method = _mapping(payload.get("method"))
    return EmotionPrecomputedVectorSpaceSpec(
        path=vector_path,
        format="json",
        select_layer=int(_mapping(payload.get("model")).get("target_layer", 40)),
        normalize=str(method.get("coordinate_normalization") or "l2"),
        vector_space_kind=str(method.get("vector_space_kind") or "story"),
        metadata={
            "asset_id": _mapping(payload.get("asset")).get("id"),
            "asset_status": _mapping(payload.get("asset")).get("status"),
            "paper": _mapping(payload.get("source")).get("paper"),
            "asset_root": planned_asset_root(payload),
        },
    )


def direction_step_name(concept: str) -> str:
    return "direction_" + concept_key(concept)


def concept_key(concept: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(concept)).strip("_")


def _load_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
