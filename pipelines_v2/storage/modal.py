"""Modal volume artifact store."""

from __future__ import annotations

import json
import os
import posixpath
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from safetensors.numpy import load_file, save_file

from pipelines_v2.core.types import TransferPolicy, TransferPolicyError
from pipelines_v2.storage.json import json_default


_MODAL_VOLUME_GET_MAX_ATTEMPTS = 3
_MODAL_VOLUME_GET_TRANSIENT_MARKERS = (
    "deadline_exceeded",
    "deadline exceeded",
    "statuscode.unavailable",
    "statuscode.resource_exhausted",
    "temporarily unavailable",
    "connection reset",
    "connection aborted",
)


@dataclass(frozen=True, slots=True)
class ModalVolumeStore:
    """Artifact store backed by a Modal volume with optional local caching."""
    name: str
    root: str
    local_cache_root: Path | str | None = None
    transfer_policy: TransferPolicy = TransferPolicy()

    kind: str = "modal_volume"

    def __post_init__(self) -> None:
        if self.local_cache_root is not None:
            object.__setattr__(self, "local_cache_root", Path(self.local_cache_root))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModalVolumeStore":
        return cls(
            name=str(payload["name"]),
            root=str(payload["root"]),
            local_cache_root=payload.get("local_cache_root"),
            transfer_policy=TransferPolicy.from_dict(payload.get("transfer_policy")),
        )

    def identity(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "name": self.name,
            "root": self.root,
        }
        if self.local_cache_root is not None:
            payload["local_cache_root"] = str(self.local_cache_root)
        payload["transfer_policy"] = self.transfer_policy.to_dict()
        return payload

    def make_artifact_dir(self, artifact_id: str) -> Path:
        path = Path(self.root) / artifact_id
        path.mkdir(parents=True, exist_ok=False)
        return path

    def ensure_artifact_dir(self, artifact_id: str) -> Path:
        path = Path(self.root) / artifact_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def has_local_artifact(self, artifact_id: str) -> bool:
        artifact_root = Path(self.root) / artifact_id
        if artifact_root.exists():
            return True
        return (self._cache_root() / artifact_id).exists()

    def write_safetensors(self, artifact_id: str, relative_path: str, tensors: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_inside_artifact(artifact_id, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        save_file(tensors, str(tmp))
        os.replace(tmp, path)
        return {
            "store": self.kind,
            "name": self.name,
            "path": str(path),
            "format": "safetensors",
            "bytes": path.stat().st_size,
        }

    def write_json(self, artifact_id: str, relative_path: str, payload: Any) -> dict[str, Any]:
        path = self._resolve_inside_artifact(artifact_id, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, indent=2, default=json_default)
        os.replace(tmp, path)
        return {
            "store": self.kind,
            "name": self.name,
            "path": str(path),
            "format": "json",
            "bytes": path.stat().st_size,
        }

    def has_local_ref(self, ref: dict[str, Any]) -> bool:
        path = Path(str(ref["path"]))
        if path.exists():
            return True
        artifact_cached = self._artifact_cache_path(path)
        if artifact_cached is not None and artifact_cached.exists():
            return True
        remote_path = self._volume_relative_path(path)
        return (self._cache_root() / "_refs" / remote_path).exists()

    def localize(self, artifact_id: str) -> Path:
        artifact_root = Path(self.root) / artifact_id
        if artifact_root.exists():
            return artifact_root

        target_root = self._cache_root() / artifact_id
        if target_root.exists():
            return target_root

        target_root.parent.mkdir(parents=True, exist_ok=True)
        self._run_modal_volume_get(self._volume_relative_path(artifact_root), target_root.parent)
        return target_root

    def read_json_ref(self, ref: dict[str, Any]) -> Any:
        path = Path(str(ref["path"]))
        if path.exists():
            return self._read_json(path)
        artifact_cached = self._artifact_cache_path(path)
        if artifact_cached is not None and artifact_cached.exists():
            return self._read_json(artifact_cached)

        remote_path = self._volume_relative_path(path)
        cache_root = self._cache_root() / "_refs"
        local_path = cache_root / remote_path
        if local_path.exists():
            return self._read_json(local_path)
        self.validate_transfer(
            bytes=self.estimate_download_bytes(ref),
            label=f"remote json ref {ref.get('path', '<unknown>')}",
        )
        self._run_modal_volume_get(remote_path, local_path)
        return self._read_json(local_path)

    def read_safetensors_ref(self, ref: dict[str, Any]) -> dict[str, Any]:
        path = Path(str(ref["path"]))
        if path.exists():
            return load_file(str(path))
        artifact_cached = self._artifact_cache_path(path)
        if artifact_cached is not None and artifact_cached.exists():
            return load_file(str(artifact_cached))

        remote_path = self._volume_relative_path(path)
        cache_root = self._cache_root() / "_refs"
        local_path = cache_root / remote_path
        if local_path.exists():
            return load_file(str(local_path))
        self.validate_transfer(
            bytes=self.estimate_download_bytes(ref),
            label=f"remote safetensors ref {ref.get('path', '<unknown>')}",
        )
        self._run_modal_volume_get(remote_path, local_path)
        return load_file(str(local_path))

    def estimate_download_bytes(self, ref: dict[str, Any]) -> int | None:
        if "bytes" in ref and ref["bytes"] is not None:
            return int(ref["bytes"])
        path = Path(str(ref["path"]))
        if path.exists():
            return path.stat().st_size
        return None

    def validate_transfer(self, *, bytes: int | None, label: str) -> None:
        if bytes is None:
            if self.transfer_policy.allow_large_transfer:
                return
            raise TransferPolicyError(
                f"Cannot estimate transfer size for {label}. Set allow_large_transfer=True to override."
            )
        if self.transfer_policy.allow_large_transfer:
            return
        if bytes > self.transfer_policy.max_download_bytes:
            raise TransferPolicyError(
                f"Refusing remote transfer for {label}: estimated {bytes} bytes exceeds "
                f"max_download_bytes={self.transfer_policy.max_download_bytes}. "
                "Set allow_large_transfer=True to override."
            )

    def _resolve_inside_artifact(self, artifact_id: str, relative_path: str) -> Path:
        root = PurePosixPath(str(Path(self.root) / artifact_id))
        path = PurePosixPath(posixpath.normpath(str(root / relative_path)))
        if path.is_absolute() is False:
            path = PurePosixPath("/", str(path))
        if root != path and root not in path.parents:
            raise ValueError(f"Refusing to write outside artifact root: {relative_path!r}")
        return Path(str(path))

    def _cache_root(self) -> Path:
        if self.local_cache_root is not None:
            return Path(self.local_cache_root)
        return Path(tempfile.gettempdir()) / "pipelines_v2_modal_volume" / self.name

    def _volume_relative_path(self, path: Path) -> str:
        absolute = PurePosixPath(str(path))
        mount = PurePosixPath(modal_volume_mount_path(self.root))
        if absolute.is_absolute():
            try:
                return str(absolute.relative_to(mount))
            except ValueError as exc:
                raise ValueError(
                    f"Path {path!s} is not inside Modal mount root {mount!s} for store {self.name!r}"
                ) from exc
        return str(absolute)

    def _artifact_cache_path(self, path: Path) -> Path | None:
        absolute = PurePosixPath(str(path))
        root = PurePosixPath(str(self.root))
        if not absolute.is_absolute():
            return None
        try:
            relative = absolute.relative_to(root)
        except ValueError:
            return None
        return self._cache_root() / relative

    def _run_modal_volume_get(self, remote_path: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = ["modal", "volume", "get", self.name, remote_path, str(destination)]
        for attempt in range(1, _MODAL_VOLUME_GET_MAX_ATTEMPTS + 1):
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return
            stderr = result.stderr.strip()
            detail = stderr or result.stdout.strip()
            normalized = detail.lower()
            transient = any(
                marker in normalized for marker in _MODAL_VOLUME_GET_TRANSIENT_MARKERS
            )
            if transient and attempt < _MODAL_VOLUME_GET_MAX_ATTEMPTS:
                time.sleep(float(2 ** (attempt - 1)))
                continue
            raise RuntimeError(
                f"modal volume get failed for {self.name!r}:{remote_path!r} "
                f"after {attempt} attempt(s): {detail}"
            )

    @staticmethod
    def _read_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)


def modal_volume_mount_path(volume_root: str) -> str:
    path = PurePosixPath(volume_root)
    if path.is_absolute() and len(path.parts) > 1:
        return str(PurePosixPath("/", path.parts[1]))
    return "/data"
