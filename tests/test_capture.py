"""Smoke tests for pipelines.interp.capture.

These tests validate the capture pipeline logic without downloading or loading
a real 16GB model. Torch/safetensors are required (install with --extra interp).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from pipelines.interp.capture import (
    CaptureConfig,
    _capture_one,
    _load_examples,
    _make_hook,
    _parse_messages,
    _save_activations,
    main,
    run_capture,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a trading agent."},
    {"role": "user", "content": "What should I do with ETH?"},
]


def _make_parquet(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write a minimal parquet file matching the expected schema."""
    path = tmp_path / "examples.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return path


def _make_example_row(log_id: int = 42, messages: list[dict] | None = None) -> dict[str, Any]:
    if messages is None:
        messages = SAMPLE_MESSAGES
    return {
        "log_id": log_id,
        "prompt_messages_json": json.dumps(messages),
        "decision_type": "trade",
        "vault_address": "0xabc",
    }


class FakeDecoderLayer(torch.nn.Module):
    """Minimal stand-in for a transformer decoder layer."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        return (self.linear(x),)


class FakeModel(torch.nn.Module):
    """Minimal stand-in for a causal LM with .model.layers and .config."""

    def __init__(self, num_layers: int = 4, hidden_dim: int = 16) -> None:
        super().__init__()
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList(
                [FakeDecoderLayer(hidden_dim) for _ in range(num_layers)]
            )
        )
        self.config = SimpleNamespace(
            hidden_size=hidden_dim,
            max_position_embeddings=8192,
        )
        self._num_layers = num_layers
        self._hidden_dim = hidden_dim

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
        batch, seq_len = input_ids.shape
        x = torch.randn(batch, seq_len, self._hidden_dim)
        for layer in self.model.layers:
            x = layer(x)[0]
        return SimpleNamespace(logits=x)

    def eval(self) -> "FakeModel":
        return self

    def to(self, device: str) -> "FakeModel":
        return self


class FakeTokenizer:
    """Minimal stand-in for a HuggingFace tokenizer."""

    def __init__(self, vocab_size: int = 100, seq_len: int = 10) -> None:
        self._vocab_size = vocab_size
        self._seq_len = seq_len

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = True,
        return_tensors: str | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int] | torch.Tensor:
        # Deterministic fake token IDs based on message content
        ids = list(range(1, self._seq_len + 1))
        if add_generation_prompt:
            ids.append(self._seq_len + 1)
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids: list[int]) -> str:
        return f"<decoded {len(ids)} tokens>"


# ---------------------------------------------------------------------------
# Unit tests: _parse_messages
# ---------------------------------------------------------------------------


class TestParseMessages:
    def test_valid_json_string(self) -> None:
        row = {"prompt_messages_json": json.dumps(SAMPLE_MESSAGES)}
        result = _parse_messages(row)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_already_parsed_list(self) -> None:
        row = {"prompt_messages_json": SAMPLE_MESSAGES}
        result = _parse_messages(row)
        assert len(result) == 2

    def test_empty_string(self) -> None:
        assert _parse_messages({"prompt_messages_json": ""}) == []

    def test_none(self) -> None:
        assert _parse_messages({"prompt_messages_json": None}) == []
        assert _parse_messages({}) == []

    def test_skips_malformed_entries(self) -> None:
        messages = [
            {"role": "system", "content": "ok"},
            "not a dict",
            {"no_role": True},
        ]
        row = {"prompt_messages_json": json.dumps(messages)}
        result = _parse_messages(row)
        assert len(result) == 1
        assert result[0]["role"] == "system"


# ---------------------------------------------------------------------------
# Unit tests: _make_hook
# ---------------------------------------------------------------------------


class TestMakeHook:
    def test_hook_captures_output(self) -> None:
        storage: dict[int, Any] = {}
        hook = _make_hook(7, storage)
        fake_output = (torch.randn(1, 5, 16),)
        hook(None, None, fake_output)
        assert 7 in storage
        assert storage[7].shape == (1, 5, 16)
        assert storage[7].dtype == torch.float16

    def test_hook_detaches_to_cpu(self) -> None:
        storage: dict[int, Any] = {}
        hook = _make_hook(0, storage)
        t = torch.randn(1, 3, 8, requires_grad=True)
        hook(None, None, (t,))
        assert not storage[0].requires_grad
        assert storage[0].device.type == "cpu"

    def test_closure_captures_correct_index(self) -> None:
        """Verify the make_hook closure pattern avoids the loop variable bug."""
        storage: dict[int, Any] = {}
        hooks = [_make_hook(i, storage) for i in range(4)]
        for i, hook in enumerate(hooks):
            hook(None, None, (torch.randn(1, 2, 4),))
        assert set(storage.keys()) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Unit tests: _load_examples
# ---------------------------------------------------------------------------


class TestLoadExamples:
    def test_loads_all_rows(self, tmp_path: Path) -> None:
        rows = [_make_example_row(i) for i in range(5)]
        path = _make_parquet(tmp_path, rows)
        config = CaptureConfig(parquet_path=path)
        result = _load_examples(config)
        assert len(result) == 5

    def test_respects_limit(self, tmp_path: Path) -> None:
        rows = [_make_example_row(i) for i in range(10)]
        path = _make_parquet(tmp_path, rows)
        config = CaptureConfig(parquet_path=path, limit=3)
        result = _load_examples(config)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Integration tests: _capture_one
# ---------------------------------------------------------------------------


class TestCaptureOne:
    def test_captures_all_layers(self) -> None:
        model = FakeModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)
        config = CaptureConfig(device="cpu")

        activations, input_ids = _capture_one(
            model=model,
            tokenizer=tokenizer,
            messages=SAMPLE_MESSAGES,
            config=config,
        )
        assert activations.shape[0] == 4  # num_layers
        assert activations.shape[1] == 10  # seq_len
        assert activations.shape[2] == 16  # hidden_dim
        assert activations.dtype == torch.float16
        assert input_ids.shape == (1, 10)

    def test_captures_subset_of_layers(self) -> None:
        model = FakeModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)
        config = CaptureConfig(device="cpu", layers=[0, 3])

        activations, _ = _capture_one(
            model=model,
            tokenizer=tokenizer,
            messages=SAMPLE_MESSAGES,
            config=config,
        )
        assert activations.shape[0] == 2  # only layers 0 and 3

    def test_hooks_are_removed_after_forward(self) -> None:
        model = FakeModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)
        config = CaptureConfig(device="cpu")

        # Count hooks before
        hooks_before = sum(
            len(layer._forward_hooks) for layer in model.model.layers
        )

        _capture_one(
            model=model,
            tokenizer=tokenizer,
            messages=SAMPLE_MESSAGES,
            config=config,
        )

        # Count hooks after — should be same as before
        hooks_after = sum(
            len(layer._forward_hooks) for layer in model.model.layers
        )
        assert hooks_after == hooks_before

    def test_add_generation_prompt_changes_seq_len(self) -> None:
        model = FakeModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)

        config_no_gen = CaptureConfig(device="cpu", add_generation_prompt=False)
        config_gen = CaptureConfig(device="cpu", add_generation_prompt=True)

        act_no, ids_no = _capture_one(
            model=model, tokenizer=tokenizer,
            messages=SAMPLE_MESSAGES, config=config_no_gen,
        )
        act_gen, ids_gen = _capture_one(
            model=model, tokenizer=tokenizer,
            messages=SAMPLE_MESSAGES, config=config_gen,
        )
        # FakeTokenizer adds 1 token for generation prompt
        assert ids_gen.shape[1] == ids_no.shape[1] + 1
        assert act_gen.shape[1] == act_no.shape[1] + 1


# ---------------------------------------------------------------------------
# Integration tests: _save_activations round-trip
# ---------------------------------------------------------------------------


class TestSaveActivations:
    def test_round_trip(self, tmp_path: Path) -> None:
        from safetensors import safe_open

        tensor = torch.randn(4, 10, 16, dtype=torch.float16)
        out_path = tmp_path / "residual_stream" / "test.safetensors"
        file_size = _save_activations(tensor, out_path)

        assert out_path.exists()
        assert file_size > 0

        with safe_open(str(out_path), framework="pt") as f:
            loaded = f.get_tensor("residual_stream")
        assert loaded.shape == (4, 10, 16)
        assert loaded.dtype == torch.float16
        assert torch.allclose(tensor, loaded)


# ---------------------------------------------------------------------------
# End-to-end: run_capture with fake model
# ---------------------------------------------------------------------------


class TestRunCapture:
    def _setup(self, tmp_path: Path, num_rows: int = 3) -> CaptureConfig:
        rows = [_make_example_row(i) for i in range(num_rows)]
        parquet_path = _make_parquet(tmp_path, rows)
        output_dir = tmp_path / "activations"
        return CaptureConfig(
            parquet_path=parquet_path,
            output_dir=output_dir,
            device="cpu",
        )

    @patch("pipelines.interp.capture._load_model")
    def test_captures_all_examples(self, mock_load: MagicMock, tmp_path: Path) -> None:
        model = FakeModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)
        mock_load.return_value = (model, tokenizer)

        config = self._setup(tmp_path, num_rows=3)
        result = run_capture(config)

        assert result["processed"] == 3
        assert result["skipped"] == 0
        assert result["errors"] == 0

        # Check safetensor files
        residual_dir = config.output_dir / "residual_stream"
        safetensor_files = list(residual_dir.glob("*.safetensors"))
        assert len(safetensor_files) == 3

        # Check metadata parquet
        meta_path = config.output_dir / "metadata.parquet"
        assert meta_path.exists()
        meta = pq.read_table(meta_path).to_pylist()
        assert len(meta) == 3
        assert all(m["num_layers_captured"] == 4 for m in meta)
        assert all(m["hidden_dim"] == 16 for m in meta)
        assert all(m["seq_len"] == 10 for m in meta)
        assert all(m["prompt_hash"] for m in meta)

    @patch("pipelines.interp.capture._load_model")
    def test_skip_existing(self, mock_load: MagicMock, tmp_path: Path) -> None:
        model = FakeModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)
        mock_load.return_value = (model, tokenizer)

        config = self._setup(tmp_path, num_rows=3)
        config.skip_existing = True

        # Pre-create one file
        residual_dir = config.output_dir / "residual_stream"
        residual_dir.mkdir(parents=True)
        (residual_dir / "0.safetensors").write_bytes(b"fake")

        result = run_capture(config)
        assert result["processed"] == 2
        assert result["skipped"] == 1

    @patch("pipelines.interp.capture._load_model")
    def test_skips_rows_without_messages(self, mock_load: MagicMock, tmp_path: Path) -> None:
        model = FakeModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)
        mock_load.return_value = (model, tokenizer)

        rows = [
            _make_example_row(0),
            {"log_id": 1, "prompt_messages_json": "", "decision_type": "trade"},
            _make_example_row(2),
        ]
        parquet_path = _make_parquet(tmp_path, rows)
        config = CaptureConfig(
            parquet_path=parquet_path,
            output_dir=tmp_path / "activations",
            device="cpu",
        )

        result = run_capture(config)
        assert result["processed"] == 2
        assert result["skipped"] == 1

    @patch("pipelines.interp.capture._load_model")
    def test_validate_tokens_mode(self, mock_load: MagicMock, tmp_path: Path) -> None:
        model = FakeModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)
        mock_load.return_value = (model, tokenizer)

        config = self._setup(tmp_path, num_rows=5)
        config.validate_tokens = True

        result = run_capture(config)
        assert "validated" in result
        assert result["validated"] == 3  # always caps at 3

        # Should NOT create any safetensor files
        residual_dir = config.output_dir / "residual_stream"
        assert not residual_dir.exists() or not list(residual_dir.glob("*.safetensors"))

    def test_missing_parquet_raises(self, tmp_path: Path) -> None:
        config = CaptureConfig(
            parquet_path=tmp_path / "nonexistent.parquet",
            device="cpu",
        )
        with pytest.raises(FileNotFoundError):
            run_capture(config)

    @patch("pipelines.interp.capture._load_model")
    def test_limit_flag(self, mock_load: MagicMock, tmp_path: Path) -> None:
        model = FakeModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)
        mock_load.return_value = (model, tokenizer)

        config = self._setup(tmp_path, num_rows=5)
        config.limit = 2

        result = run_capture(config)
        assert result["processed"] == 2


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLI:
    def test_defaults(self) -> None:
        from pipelines.interp.capture import _build_parser

        args = _build_parser().parse_args([])
        assert args.model_id == "Qwen/Qwen3-8B"
        assert args.device == "mps"
        assert args.limit is None
        assert args.layers is None
        assert args.skip_existing is False
        assert args.validate_tokens is False
        assert args.add_generation_prompt is False

    def test_all_flags(self) -> None:
        from pipelines.interp.capture import _build_parser

        args = _build_parser().parse_args([
            "--model-id", "test/model",
            "--device", "cpu",
            "--limit", "5",
            "--layers", "0,12,24,35",
            "--skip-existing",
            "--validate-tokens",
            "--add-generation-prompt",
            "--parquet-path", "/tmp/test.parquet",
            "--output-dir", "/tmp/out",
        ])
        assert args.model_id == "test/model"
        assert args.device == "cpu"
        assert args.limit == 5
        assert args.layers == "0,12,24,35"
        assert args.skip_existing is True
        assert args.validate_tokens is True
        assert args.add_generation_prompt is True

    @patch("pipelines.interp.capture.run_capture")
    def test_main_parses_layers(self, mock_run: MagicMock) -> None:
        mock_run.return_value = {"processed": 0, "skipped": 0, "errors": 0}
        main(["--layers", "0,12,35", "--parquet-path", "/tmp/x.parquet", "--device", "cpu"])
        cfg = mock_run.call_args[0][0]
        assert cfg.layers == [0, 12, 35]

    @patch("pipelines.interp.capture.run_capture")
    def test_main_no_layers(self, mock_run: MagicMock) -> None:
        mock_run.return_value = {"processed": 0, "skipped": 0, "errors": 0}
        main(["--parquet-path", "/tmp/x.parquet", "--device", "cpu"])
        cfg = mock_run.call_args[0][0]
        assert cfg.layers is None
