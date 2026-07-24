from __future__ import annotations

import types
from typing import Any

from pipelines_v2.engine.vllm import hidden_states_connector


def test_load_and_cleanup_hidden_states_uses_vllm_synchronized_helpers(
    monkeypatch: Any,
) -> None:
    events: list[tuple[str, str]] = []
    expected = {"hidden_states": object(), "token_ids": object()}

    def _load(path: str) -> dict[str, object]:
        events.append(("load", path))
        return expected

    def _cleanup(path: str) -> None:
        events.append(("cleanup", path))

    fake_upstream = types.SimpleNamespace(
        load_hidden_states=_load,
        cleanup_hidden_states=_cleanup,
    )
    monkeypatch.setattr(
        hidden_states_connector,
        "import_module",
        lambda module_name: fake_upstream,
    )

    result = hidden_states_connector.load_and_cleanup_hidden_states(
        "/tmp/async-result.safetensors"
    )

    assert result is expected
    assert events == [
        ("load", "/tmp/async-result.safetensors"),
        ("cleanup", "/tmp/async-result.safetensors"),
    ]
