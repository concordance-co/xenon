"""Smoke tests for pipelines.interp.local_capture.

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

from pipelines.interp.local_capture import (
    CaptureConfig,
    _capture_one,
    _flush_metadata,
    _is_moe_model,
    _load_examples,
    _load_existing_metadata,
    _make_hook,
    _make_router_hook,
    _parse_messages,
    _save_activations,
    _save_router,
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


class FakeGate(torch.nn.Module):
    """Minimal stand-in for Qwen3MoeTopKRouter gate."""

    def __init__(self, num_experts: int = 128, top_k: int = 8) -> None:
        super().__init__()
        self._num_experts = num_experts
        self._top_k = top_k
        # Needs a parameter so register_forward_hook works
        self.weight = torch.nn.Parameter(torch.empty(0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = x.shape[0] if x.dim() == 2 else x.shape[1]
        batch_seq = seq_len  # flattened
        router_logits = torch.randn(batch_seq, self._num_experts)
        router_scores = torch.softmax(router_logits, dim=-1)
        router_indices = torch.randint(0, self._num_experts, (batch_seq, self._top_k))
        return router_logits, router_scores, router_indices


class FakeMoEMLP(torch.nn.Module):
    """MLP with a gate attribute to simulate MoE."""

    def __init__(self, hidden_dim: int, num_experts: int = 128, top_k: int = 8) -> None:
        super().__init__()
        self.gate = FakeGate(num_experts, top_k)
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call gate so hooks fire
        self.gate(x.view(-1, x.shape[-1]))
        return self.linear(x)


class FakeMoEDecoderLayer(torch.nn.Module):
    """Decoder layer with an MoE MLP."""

    def __init__(self, hidden_dim: int, num_experts: int = 128, top_k: int = 8) -> None:
        super().__init__()
        self.mlp = FakeMoEMLP(hidden_dim, num_experts, top_k)
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        out = self.linear(x) + self.mlp(x)
        return (out,)


class FakeMoEModel(torch.nn.Module):
    """Minimal MoE causal LM with .model.layers[i].mlp.gate."""

    def __init__(
        self, num_layers: int = 4, hidden_dim: int = 16,
        num_experts: int = 128, top_k: int = 8,
    ) -> None:
        super().__init__()
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList(
                [FakeMoEDecoderLayer(hidden_dim, num_experts, top_k) for _ in range(num_layers)]
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

    def eval(self) -> "FakeMoEModel":
        return self

    def to(self, device: str) -> "FakeMoEModel":
        return self


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
        config = CaptureConfig(device="cpu", capture_router=False)

        residual, router_logits, router_indices, input_ids = _capture_one(
            model=model,
            tokenizer=tokenizer,
            messages=SAMPLE_MESSAGES,
            config=config,
        )
        assert residual.shape[0] == 4  # num_layers
        assert residual.shape[1] == 10  # seq_len
        assert residual.shape[2] == 16  # hidden_dim
        assert residual.dtype == torch.float16
        assert input_ids.shape == (1, 10)
        assert router_logits is None
        assert router_indices is None

    def test_captures_subset_of_layers(self) -> None:
        model = FakeModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)
        config = CaptureConfig(device="cpu", layers=[0, 3], capture_router=False)

        residual, _, _, _ = _capture_one(
            model=model,
            tokenizer=tokenizer,
            messages=SAMPLE_MESSAGES,
            config=config,
        )
        assert residual.shape[0] == 2  # only layers 0 and 3

    def test_hooks_are_removed_after_forward(self) -> None:
        model = FakeModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)
        config = CaptureConfig(device="cpu", capture_router=False)

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

        config_no_gen = CaptureConfig(device="cpu", add_generation_prompt=False, capture_router=False)
        config_gen = CaptureConfig(device="cpu", add_generation_prompt=True, capture_router=False)

        res_no, _, _, ids_no = _capture_one(
            model=model, tokenizer=tokenizer,
            messages=SAMPLE_MESSAGES, config=config_no_gen,
        )
        res_gen, _, _, ids_gen = _capture_one(
            model=model, tokenizer=tokenizer,
            messages=SAMPLE_MESSAGES, config=config_gen,
        )
        # FakeTokenizer adds 1 token for generation prompt
        assert ids_gen.shape[1] == ids_no.shape[1] + 1
        assert res_gen.shape[1] == res_no.shape[1] + 1


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
            capture_router=False,
        )

    @patch("pipelines.interp.local_capture._load_model")
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

    @patch("pipelines.interp.local_capture._load_model")
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

    @patch("pipelines.interp.local_capture._load_model")
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

    @patch("pipelines.interp.local_capture._load_model")
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

    @patch("pipelines.interp.local_capture._load_model")
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
        from pipelines.interp.local_capture import _build_parser

        args = _build_parser().parse_args([])
        assert args.model_id == "Qwen/Qwen3-8B"
        assert args.device == "mps"
        assert args.limit is None
        assert args.layers is None
        assert args.skip_existing is False
        assert args.validate_tokens is False
        assert args.add_generation_prompt is False
        assert args.router_dtype == "float16"
        assert args.metadata_flush_interval == 10

    def test_all_flags(self) -> None:
        from pipelines.interp.local_capture import _build_parser

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

    @patch("pipelines.interp.local_capture.run_capture")
    def test_main_parses_layers(self, mock_run: MagicMock) -> None:
        mock_run.return_value = {"processed": 0, "skipped": 0, "errors": 0}
        main(["--layers", "0,12,35", "--parquet-path", "/tmp/x.parquet", "--device", "cpu"])
        cfg = mock_run.call_args[0][0]
        assert cfg.layers == [0, 12, 35]

    @patch("pipelines.interp.local_capture.run_capture")
    def test_main_no_layers(self, mock_run: MagicMock) -> None:
        mock_run.return_value = {"processed": 0, "skipped": 0, "errors": 0}
        main(["--parquet-path", "/tmp/x.parquet", "--device", "cpu"])
        cfg = mock_run.call_args[0][0]
        assert cfg.layers is None

    @patch("pipelines.interp.local_capture.run_capture")
    def test_capture_router_flags(self, mock_run: MagicMock) -> None:
        mock_run.return_value = {"processed": 0, "skipped": 0, "errors": 0}
        main(["--parquet-path", "/tmp/x.parquet", "--device", "cpu",
              "--no-capture-residual"])
        cfg = mock_run.call_args[0][0]
        assert cfg.capture_residual is False
        assert cfg.capture_router is True

    @patch("pipelines.interp.local_capture.run_capture")
    def test_no_capture_router_flag(self, mock_run: MagicMock) -> None:
        mock_run.return_value = {"processed": 0, "skipped": 0, "errors": 0}
        main(["--parquet-path", "/tmp/x.parquet", "--device", "cpu",
              "--no-capture-router"])
        cfg = mock_run.call_args[0][0]
        assert cfg.capture_router is False
        assert cfg.capture_residual is True

    @patch("pipelines.interp.local_capture.run_capture")
    def test_router_dtype_default(self, mock_run: MagicMock) -> None:
        mock_run.return_value = {"processed": 0, "skipped": 0, "errors": 0}
        main(["--parquet-path", "/tmp/x.parquet", "--device", "cpu"])
        cfg = mock_run.call_args[0][0]
        assert cfg.router_dtype == "float16"

    @patch("pipelines.interp.local_capture.run_capture")
    def test_router_dtype_fp32(self, mock_run: MagicMock) -> None:
        mock_run.return_value = {"processed": 0, "skipped": 0, "errors": 0}
        main(["--parquet-path", "/tmp/x.parquet", "--device", "cpu",
              "--router-dtype", "float32"])
        cfg = mock_run.call_args[0][0]
        assert cfg.router_dtype == "float32"

    @patch("pipelines.interp.local_capture.run_capture")
    def test_metadata_flush_interval_default(self, mock_run: MagicMock) -> None:
        mock_run.return_value = {"processed": 0, "skipped": 0, "errors": 0}
        main(["--parquet-path", "/tmp/x.parquet", "--device", "cpu"])
        cfg = mock_run.call_args[0][0]
        assert cfg.metadata_flush_interval == 10

    @patch("pipelines.interp.local_capture.run_capture")
    def test_metadata_flush_interval_custom(self, mock_run: MagicMock) -> None:
        mock_run.return_value = {"processed": 0, "skipped": 0, "errors": 0}
        main(["--parquet-path", "/tmp/x.parquet", "--device", "cpu",
              "--metadata-flush-interval", "25"])
        cfg = mock_run.call_args[0][0]
        assert cfg.metadata_flush_interval == 25


# ---------------------------------------------------------------------------
# MoE auto-detection
# ---------------------------------------------------------------------------


class TestMoEDetection:
    def test_dense_model_not_moe(self) -> None:
        model = FakeModel(num_layers=4, hidden_dim=16)
        assert _is_moe_model(model) is False

    def test_moe_model_detected(self) -> None:
        model = FakeMoEModel(num_layers=4, hidden_dim=16)
        assert _is_moe_model(model) is True


# ---------------------------------------------------------------------------
# MoE router hook tests
# ---------------------------------------------------------------------------


class TestMakeRouterHook:
    def test_router_hook_captures_logits_and_indices(self) -> None:
        logits_storage: dict[int, Any] = {}
        indices_storage: dict[int, Any] = {}
        hook = _make_router_hook(3, logits_storage, indices_storage)

        fake_logits = torch.randn(5, 128)
        fake_scores = torch.randn(5, 128)
        fake_indices = torch.randint(0, 128, (5, 8))
        hook(None, None, (fake_logits, fake_scores, fake_indices))

        assert 3 in logits_storage
        assert 3 in indices_storage
        assert logits_storage[3].dtype == torch.float32
        assert indices_storage[3].dtype == torch.int16
        assert logits_storage[3].shape == (5, 128)
        assert indices_storage[3].shape == (5, 8)

    def test_router_hook_detaches_to_cpu(self) -> None:
        logits_storage: dict[int, Any] = {}
        indices_storage: dict[int, Any] = {}
        hook = _make_router_hook(0, logits_storage, indices_storage)

        logits = torch.randn(3, 128, requires_grad=True)
        scores = torch.randn(3, 128)
        indices = torch.randint(0, 128, (3, 8))
        hook(None, None, (logits, scores, indices))

        assert not logits_storage[0].requires_grad
        assert logits_storage[0].device.type == "cpu"
        assert indices_storage[0].device.type == "cpu"


# ---------------------------------------------------------------------------
# MoE capture integration tests
# ---------------------------------------------------------------------------


class TestMoECapture:
    def test_capture_router_logits(self) -> None:
        model = FakeMoEModel(num_layers=4, hidden_dim=16, num_experts=128, top_k=8)
        tokenizer = FakeTokenizer(seq_len=10)
        config = CaptureConfig(device="cpu", capture_router=True, capture_residual=True)

        residual, router_logits, router_indices, input_ids = _capture_one(
            model=model,
            tokenizer=tokenizer,
            messages=SAMPLE_MESSAGES,
            config=config,
        )
        assert residual is not None
        assert residual.shape == (4, 10, 16)

        assert router_logits is not None
        assert router_indices is not None
        assert router_logits.shape == (4, 10, 128)
        assert router_indices.shape == (4, 10, 8)
        assert router_logits.dtype == torch.float32
        assert router_indices.dtype == torch.int16

    def test_capture_router_only(self) -> None:
        model = FakeMoEModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)
        config = CaptureConfig(device="cpu", capture_router=True, capture_residual=False)

        residual, router_logits, router_indices, input_ids = _capture_one(
            model=model,
            tokenizer=tokenizer,
            messages=SAMPLE_MESSAGES,
            config=config,
        )
        assert residual is None
        assert router_logits is not None
        assert router_indices is not None

    def test_dense_model_skips_router(self) -> None:
        model = FakeModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)
        config = CaptureConfig(device="cpu", capture_router=True, capture_residual=True)

        residual, router_logits, router_indices, input_ids = _capture_one(
            model=model,
            tokenizer=tokenizer,
            messages=SAMPLE_MESSAGES,
            config=config,
        )
        assert residual is not None
        assert router_logits is None
        assert router_indices is None

    def test_moe_hooks_removed_after_forward(self) -> None:
        model = FakeMoEModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)
        config = CaptureConfig(device="cpu", capture_router=True, capture_residual=True)

        hooks_before = sum(
            len(layer._forward_hooks) + len(layer.mlp.gate._forward_hooks)
            for layer in model.model.layers
        )

        _capture_one(
            model=model,
            tokenizer=tokenizer,
            messages=SAMPLE_MESSAGES,
            config=config,
        )

        hooks_after = sum(
            len(layer._forward_hooks) + len(layer.mlp.gate._forward_hooks)
            for layer in model.model.layers
        )
        assert hooks_after == hooks_before


# ---------------------------------------------------------------------------
# Router save/load round-trip
# ---------------------------------------------------------------------------


class TestSaveRouter:
    def test_round_trip_fp16_default(self, tmp_path: Path) -> None:
        from safetensors import safe_open

        router_logits = torch.randn(4, 10, 128, dtype=torch.float32)
        router_indices = torch.randint(0, 128, (4, 10, 8)).to(torch.int16)
        out_path = tmp_path / "router_logits" / "test.safetensors"
        file_size = _save_router(router_logits, router_indices, out_path)

        assert out_path.exists()
        assert file_size > 0

        with safe_open(str(out_path), framework="pt") as f:
            loaded_logits = f.get_tensor("router_logits")
            loaded_indices = f.get_tensor("router_indices")
        assert loaded_logits.shape == (4, 10, 128)
        assert loaded_logits.dtype == torch.float16  # default is fp16
        assert loaded_indices.shape == (4, 10, 8)
        assert loaded_indices.dtype == torch.int16
        assert torch.equal(router_indices, loaded_indices)

    def test_round_trip_fp32_explicit(self, tmp_path: Path) -> None:
        from safetensors import safe_open

        router_logits = torch.randn(4, 10, 128, dtype=torch.float32)
        router_indices = torch.randint(0, 128, (4, 10, 8)).to(torch.int16)
        out_path = tmp_path / "router_logits" / "test_fp32.safetensors"
        file_size = _save_router(router_logits, router_indices, out_path, router_dtype="float32")

        assert out_path.exists()
        assert file_size > 0

        with safe_open(str(out_path), framework="pt") as f:
            loaded_logits = f.get_tensor("router_logits")
            loaded_indices = f.get_tensor("router_indices")
        assert loaded_logits.shape == (4, 10, 128)
        assert loaded_logits.dtype == torch.float32
        assert torch.allclose(router_logits, loaded_logits)
        assert loaded_indices.shape == (4, 10, 8)
        assert loaded_indices.dtype == torch.int16
        assert torch.equal(router_indices, loaded_indices)

    def test_fp16_halves_storage(self, tmp_path: Path) -> None:
        router_logits = torch.randn(4, 10, 128, dtype=torch.float32)
        router_indices = torch.randint(0, 128, (4, 10, 8)).to(torch.int16)

        fp16_path = tmp_path / "fp16.safetensors"
        fp32_path = tmp_path / "fp32.safetensors"
        size_fp16 = _save_router(router_logits, router_indices, fp16_path, router_dtype="float16")
        size_fp32 = _save_router(router_logits, router_indices, fp32_path, router_dtype="float32")

        # fp16 logits should be roughly half the size of fp32 logits
        # (indices are the same size in both)
        assert size_fp16 < size_fp32


# ---------------------------------------------------------------------------
# End-to-end MoE run_capture
# ---------------------------------------------------------------------------


class TestRunCaptureMoE:
    def _setup(self, tmp_path: Path, num_rows: int = 3) -> CaptureConfig:
        rows = [_make_example_row(i) for i in range(num_rows)]
        parquet_path = _make_parquet(tmp_path, rows)
        output_dir = tmp_path / "activations"
        return CaptureConfig(
            parquet_path=parquet_path,
            output_dir=output_dir,
            device="cpu",
            capture_router=True,
            capture_residual=True,
        )

    @patch("pipelines.interp.local_capture._load_model")
    def test_captures_router_and_residual(self, mock_load: MagicMock, tmp_path: Path) -> None:
        model = FakeMoEModel(num_layers=4, hidden_dim=16, num_experts=128, top_k=8)
        tokenizer = FakeTokenizer(seq_len=10)
        mock_load.return_value = (model, tokenizer)

        config = self._setup(tmp_path, num_rows=2)
        result = run_capture(config)

        assert result["processed"] == 2
        assert result["errors"] == 0

        # Check residual files
        residual_dir = config.output_dir / "residual_stream"
        assert len(list(residual_dir.glob("*.safetensors"))) == 2

        # Check router files
        router_dir = config.output_dir / "router_logits"
        assert len(list(router_dir.glob("*.safetensors"))) == 2

        # Check metadata
        meta = pq.read_table(config.output_dir / "metadata.parquet").to_pylist()
        assert all(m["has_router"] is True for m in meta)
        assert all(m["num_experts"] == 128 for m in meta)

    @patch("pipelines.interp.local_capture._load_model")
    def test_router_only_no_residual(self, mock_load: MagicMock, tmp_path: Path) -> None:
        model = FakeMoEModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)
        mock_load.return_value = (model, tokenizer)

        config = self._setup(tmp_path, num_rows=1)
        config.capture_residual = False

        result = run_capture(config)
        assert result["processed"] == 1

        residual_dir = config.output_dir / "residual_stream"
        assert not residual_dir.exists() or len(list(residual_dir.glob("*.safetensors"))) == 0

        router_dir = config.output_dir / "router_logits"
        assert len(list(router_dir.glob("*.safetensors"))) == 1


# ---------------------------------------------------------------------------
# Incremental metadata tests
# ---------------------------------------------------------------------------


class TestFlushMetadata:
    def test_atomic_write(self, tmp_path: Path) -> None:
        rows = [{"log_id": 1, "seq_len": 10}, {"log_id": 2, "seq_len": 20}]
        _flush_metadata(rows, tmp_path)

        meta_path = tmp_path / "metadata.parquet"
        assert meta_path.exists()
        # .tmp file should not remain
        assert not (tmp_path / "metadata.parquet.tmp").exists()

        loaded = pq.read_table(meta_path).to_pylist()
        assert len(loaded) == 2
        assert loaded[0]["log_id"] == 1

    def test_overwrites_previous(self, tmp_path: Path) -> None:
        _flush_metadata([{"log_id": 1}], tmp_path)
        _flush_metadata([{"log_id": 1}, {"log_id": 2}, {"log_id": 3}], tmp_path)

        loaded = pq.read_table(tmp_path / "metadata.parquet").to_pylist()
        assert len(loaded) == 3


class TestLoadExistingMetadata:
    def test_no_file_returns_empty(self, tmp_path: Path) -> None:
        result = _load_existing_metadata(tmp_path)
        assert result == []

    def test_loads_existing(self, tmp_path: Path) -> None:
        rows = [{"log_id": 42, "seq_len": 100}]
        _flush_metadata(rows, tmp_path)
        result = _load_existing_metadata(tmp_path)
        assert len(result) == 1
        assert result[0]["log_id"] == 42


class TestIncrementalMetadata:
    def _setup(self, tmp_path: Path, num_rows: int = 15) -> CaptureConfig:
        rows = [_make_example_row(i) for i in range(num_rows)]
        parquet_path = _make_parquet(tmp_path, rows)
        output_dir = tmp_path / "activations"
        return CaptureConfig(
            parquet_path=parquet_path,
            output_dir=output_dir,
            device="cpu",
            capture_router=False,
            metadata_flush_interval=5,
        )

    @patch("pipelines.interp.local_capture._load_model")
    def test_metadata_written_incrementally(self, mock_load: MagicMock, tmp_path: Path) -> None:
        model = FakeModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)
        mock_load.return_value = (model, tokenizer)

        config = self._setup(tmp_path, num_rows=12)
        result = run_capture(config)

        assert result["processed"] == 12
        # Final metadata should contain all rows
        meta = pq.read_table(config.output_dir / "metadata.parquet").to_pylist()
        assert len(meta) == 12

    @patch("pipelines.interp.local_capture._load_model")
    def test_resume_from_existing_metadata(self, mock_load: MagicMock, tmp_path: Path) -> None:
        model = FakeModel(num_layers=4, hidden_dim=16)
        tokenizer = FakeTokenizer(seq_len=10)
        mock_load.return_value = (model, tokenizer)

        config = self._setup(tmp_path, num_rows=5)
        config.skip_existing = True

        # Run once
        result1 = run_capture(config)
        assert result1["processed"] == 5

        # Run again -- all should be skipped (files + metadata exist)
        result2 = run_capture(config)
        assert result2["processed"] == 0
        assert result2["skipped"] == 5

        # Metadata should still have 5 rows (loaded existing + no new)
        meta = pq.read_table(config.output_dir / "metadata.parquet").to_pylist()
        assert len(meta) == 5
