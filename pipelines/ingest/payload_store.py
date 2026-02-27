from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


@dataclass(slots=True)
class RawPayloadMetadata:
    payload_path: str
    payload_sha256: str
    payload_size_bytes: int


class RawPayloadStore:
    """Stores full-log payloads as compressed JSON files."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def _payload_path(self, log_id: int) -> Path:
        shard = f"{log_id // 1000:06d}"
        return self.root_dir / shard / f"{log_id}.json.gz"

    def write(self, log_id: int, payload: dict[str, Any]) -> RawPayloadMetadata:
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
        payload_sha256 = hashlib.sha256(encoded).hexdigest()
        payload_path = self._payload_path(log_id)
        payload_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write so partial files are not left behind on interruption.
        with NamedTemporaryFile(dir=payload_path.parent, suffix=".tmp", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with gzip.open(tmp_path, "wb") as gz_file:
                gz_file.write(encoded)
            tmp_path.replace(payload_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        return RawPayloadMetadata(
            payload_path=str(payload_path.resolve()),
            payload_sha256=payload_sha256,
            payload_size_bytes=len(encoded),
        )

