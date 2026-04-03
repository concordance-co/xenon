"""Capture pipeline for counterfactual experiment.

Runs forward passes on pre-built counterfactual prompts and saves
section-pooled + per-row activations. Built on top of existing
capture infrastructure but with a different input format and output schema.

Output structure:
    /data/activations/counterfactual/{experiment_id}/
        residual/
            {snapshot_id}_{variant}.safetensors     # section-pooled residuals
        router/
            {snapshot_id}_{variant}.safetensors      # router indices at row positions
        metadata.parquet                              # capture metadata

Each safetensors file contains keys like:
    - "row_mean_{i}": (num_layers, hidden_dim)   per-row mean (symbol-masked)
    - "row_eos_{i}":  (num_layers, hidden_dim)   per-row last token
    - "market_mean":  (num_layers, hidden_dim)    full market section mean
    - "market_eos":   (num_layers, hidden_dim)    last token of market section
    - "last_token":   (num_layers, hidden_dim)    generation position
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(slots=True)
class CounterfactualCaptureConfig:
    """Config for counterfactual capture runs."""
    output_dir: Path = field(
        default_factory=lambda: Path("data/activations/counterfactual"),
    )
    experiment_id: str = "default"
    model_id: str = "Qwen/Qwen3-30B-A3B"
    device: str = "cuda"
    capture_router: bool = True
    capture_residual: bool = True
    router_dtype: str = "float16"
    router_top_k: int = 8
    skip_existing: bool = True
    metadata_flush_interval: int = 10
    add_generation_prompt: bool = False

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.experiment_id


# ---------------------------------------------------------------------------
# Per-row pooling (applied after full-sequence forward pass)
# ---------------------------------------------------------------------------

def pool_per_row(
    residual: Any,  # (num_layers, seq_len, hidden_dim)
    row_boundaries: list[dict[str, Any]],
    section_boundaries: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    """Pool full-sequence residuals into per-row and section-level representations.

    Returns a dict of named tensors suitable for safetensors storage.
    """
    import torch

    result: dict[str, Any] = {}

    if residual is None:
        return result

    num_layers, seq_len, hidden_dim = residual.shape

    # Per-row pooling (symbol-masked: uses content_start:content_end)
    for rb in row_boundaries:
        i = rb["row_index"]
        c_start = rb["content_start"]
        c_end = rb["content_end"]
        f_start = rb["full_start"]
        f_end = rb["full_end"]

        if c_start < c_end and c_end <= seq_len:
            # row_mean: mean over content tokens (excludes symbol prefix)
            row_slice = residual[:, c_start:c_end, :]  # (layers, row_tokens, dim)
            result[f"row_mean_{i}"] = row_slice.mean(dim=1).to(torch.float16)  # (layers, dim)
            # row_eos: last token of the content region
            result[f"row_eos_{i}"] = residual[:, c_end - 1, :].clone().to(torch.float16)
        elif f_start < f_end and f_end <= seq_len:
            # Fallback to full row if content boundaries failed
            row_slice = residual[:, f_start:f_end, :]
            result[f"row_mean_{i}"] = row_slice.mean(dim=1).to(torch.float16)
            result[f"row_eos_{i}"] = residual[:, f_end - 1, :].clone().to(torch.float16)

    # Section-level pooling
    if "market" in section_boundaries:
        m_start, m_end = section_boundaries["market"]
        if m_start < m_end and m_end <= seq_len:
            market_slice = residual[:, m_start:m_end, :]
            result["market_mean"] = market_slice.mean(dim=1).to(torch.float16)
            result["market_eos"] = residual[:, m_end - 1, :].clone().to(torch.float16)

    # Last token (generation position)
    result["last_token"] = residual[:, -1, :].clone().to(torch.float16)

    # Preamble pooling (useful for position controls)
    if "preamble" in section_boundaries:
        p_start, p_end = section_boundaries["preamble"]
        if p_start < p_end and p_end <= seq_len:
            preamble_slice = residual[:, p_start:p_end, :]
            result["preamble_mean"] = preamble_slice.mean(dim=1).to(torch.float16)

    # Downstream section pooling (for Questions B & C)
    # These positions can attend to both market AND settings tokens.
    downstream_sections = [
        "active_settings", "portfolio", "constraints", "prev_decisions",
    ]
    for section_name in downstream_sections:
        if section_name in section_boundaries:
            s_start, s_end = section_boundaries[section_name]
            if s_start < s_end and s_end <= seq_len:
                section_slice = residual[:, s_start:s_end, :]
                result[f"{section_name}_mean"] = section_slice.mean(dim=1).to(torch.float16)
                result[f"{section_name}_eos"] = residual[:, s_end - 1, :].clone().to(torch.float16)

    return result


def pool_router_per_row(
    router_indices: Any,  # (num_moe_layers, seq_len, top_k) int16
    row_boundaries: list[dict[str, Any]],
    section_boundaries: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    """Extract router indices at row positions for routing divergence analysis."""
    import torch

    result: dict[str, Any] = {}
    if router_indices is None:
        return result

    num_layers, seq_len, top_k = router_indices.shape

    # Per-row: save all router indices at content positions
    for rb in row_boundaries:
        i = rb["row_index"]
        c_start = rb["content_start"]
        c_end = rb["content_end"]
        if c_start < c_end and c_end <= seq_len:
            # (num_layers, row_tokens, top_k)
            result[f"router_row_{i}"] = router_indices[:, c_start:c_end, :].contiguous()

    # Market section router indices
    if "market" in section_boundaries:
        m_start, m_end = section_boundaries["market"]
        if m_start < m_end and m_end <= seq_len:
            result["router_market"] = router_indices[:, m_start:m_end, :].contiguous()

    # Downstream section router indices
    for section_name in ["active_settings", "portfolio", "constraints", "prev_decisions"]:
        if section_name in section_boundaries:
            s_start, s_end = section_boundaries[section_name]
            if s_start < s_end and s_end <= seq_len:
                result[f"router_{section_name}"] = router_indices[:, s_start:s_end, :].contiguous()

    return result


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_pooled(
    tensors: dict[str, Any],
    output_path: Path,
) -> int:
    """Save a dict of named tensors to safetensors."""
    from safetensors.torch import save_file

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure all tensors are contiguous
    clean = {k: v.contiguous() for k, v in tensors.items()}
    save_file(clean, str(output_path))
    return sum(v.nelement() * v.element_size() for v in clean.values())


# ---------------------------------------------------------------------------
# Capture one prompt
# ---------------------------------------------------------------------------

def capture_one_counterfactual(
    *,
    model: Any,
    tokenizer: Any,
    system_text: str,
    user_text: str,
    row_boundaries: list[dict[str, Any]],
    section_boundaries: dict[str, tuple[int, int]],
    config: CounterfactualCaptureConfig,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Run a single counterfactual prompt and return pooled activations.

    Returns (residual_pooled, router_pooled, seq_len).
    """
    from pipelines.interp.local_capture import (
        _capture_one,
        CaptureConfig,
    )

    # Build messages in the format expected by _capture_one
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

    # Create a minimal CaptureConfig for the underlying capture function
    inner_config = CaptureConfig(
        output_dir=config.run_dir,
        model_id=config.model_id,
        device=config.device,
        capture_router=config.capture_router,
        capture_residual=config.capture_residual,
        pool_on_capture=None,  # We handle pooling ourselves
        add_generation_prompt=config.add_generation_prompt,
    )

    residual, router_logits, router_indices, input_ids = _capture_one(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        config=inner_config,
    )

    seq_len = int(input_ids.shape[1])

    # Pool residuals per-row and per-section
    residual_pooled = pool_per_row(residual, row_boundaries, section_boundaries)
    router_pooled = pool_router_per_row(router_indices, row_boundaries, section_boundaries)

    return residual_pooled, router_pooled, seq_len


# ---------------------------------------------------------------------------
# Batch capture loop
# ---------------------------------------------------------------------------

def run_counterfactual_capture(
    prompts: list[dict[str, Any]],
    config: CounterfactualCaptureConfig,
) -> dict[str, Any]:
    """Run capture on a list of counterfactual prompts.

    Each prompt dict must have:
        - capture_id: str (e.g. "{snapshot_id}_{variant}")
        - system_text: str
        - user_text: str
        - row_boundaries: list[dict]
        - section_boundaries: dict
        - snapshot_id: str
        - variant: str

    Returns summary stats.
    """
    import torch
    from pipelines.interp.local_capture import _load_model

    inner_config_for_model = type('C', (), {
        'model_id': config.model_id,
        'device': config.device,
    })()

    print(f"Loading model {config.model_id}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16,
        device_map=config.device,
    )
    model.eval()

    # Prepare output dirs
    residual_dir = config.run_dir / "residual"
    router_dir = config.run_dir / "router"
    residual_dir.mkdir(parents=True, exist_ok=True)
    router_dir.mkdir(parents=True, exist_ok=True)

    # Load existing metadata for resume
    meta_path = config.run_dir / "metadata.parquet"
    metadata_rows: list[dict[str, Any]] = []
    existing_ids: set[str] = set()
    if meta_path.exists():
        table = pq.read_table(meta_path)
        metadata_rows = table.to_pylist()
        existing_ids = {r["capture_id"] for r in metadata_rows}
        print(f"Resuming: {len(existing_ids)} existing captures")

    processed = 0
    skipped = 0
    errors = 0

    for idx, prompt in enumerate(prompts):
        capture_id = prompt["capture_id"]

        if config.skip_existing and capture_id in existing_ids:
            skipped += 1
            continue

        try:
            t0 = time.monotonic()
            residual_pooled, router_pooled, seq_len = capture_one_counterfactual(
                model=model,
                tokenizer=tokenizer,
                system_text=prompt["system_text"],
                user_text=prompt["user_text"],
                row_boundaries=prompt["row_boundaries"],
                section_boundaries=prompt["section_boundaries"],
                config=config,
            )
            elapsed = time.monotonic() - t0

            # Save pooled activations
            file_size = 0
            if residual_pooled:
                file_size += _save_pooled(
                    residual_pooled,
                    residual_dir / f"{capture_id}.safetensors",
                )
            if router_pooled:
                file_size += _save_pooled(
                    router_pooled,
                    router_dir / f"{capture_id}.safetensors",
                )

            # Metadata
            meta_row = {
                "capture_id": capture_id,
                "snapshot_id": prompt["snapshot_id"],
                "variant": prompt["variant"],
                "seq_len": seq_len,
                "n_rows": len(prompt["row_boundaries"]),
                "n_residual_keys": len(residual_pooled),
                "n_router_keys": len(router_pooled),
                "file_size_bytes": file_size,
                "elapsed_s": round(elapsed, 2),
                "capture_timestamp": datetime.now(UTC).isoformat(),
            }
            metadata_rows.append(meta_row)
            existing_ids.add(capture_id)
            processed += 1

            if processed % config.metadata_flush_interval == 0:
                _flush_metadata(meta_path, metadata_rows)

            print(
                f"  [{idx + 1}/{len(prompts)}] {capture_id}: "
                f"seq_len={seq_len}, {len(residual_pooled)} residual keys, "
                f"{file_size / 1024:.0f}KB, {elapsed:.1f}s"
            )

        except Exception as e:
            print(f"  [{idx + 1}/{len(prompts)}] ERROR {capture_id}: {e}")
            errors += 1

    # Final flush
    _flush_metadata(meta_path, metadata_rows)

    result = {"processed": processed, "skipped": skipped, "errors": errors}
    print(f"\nCounterfactual capture complete: {result}")
    return result


def _flush_metadata(meta_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, meta_path, compression="snappy")


# ---------------------------------------------------------------------------
# Prepare prompts for capture
# ---------------------------------------------------------------------------

def load_prompts_from_db(
    dataset: str = "a",
    include_padded: bool = True,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Load counterfactual prompts from DB and compute token boundaries.

    Args:
        dataset: 'a', 'b', or 'both'
        include_padded: If True and dataset includes 'a', generate padded
            variants (low_pad, high_pad) using the tokenizer. Requires tokenizer.
        tokenizer: HuggingFace tokenizer, required for boundary computation
            and padded variant generation.

    Returns list of capture-ready prompt dicts.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for boundary computation")

    from projects.counterfactual import (
        find_section_boundaries,
        find_row_boundaries,
        find_downstream_section_boundaries,
        compute_padding,
        build_canonical_user_text,
        parse_market_section,
        build_market_rows,
        randomize_rows,
        CanonicalPrompt,
        Snapshot,
        MarketRow,
        MARKET_HEADER,
    )
    from pipelines.db import connect_neon

    conn = connect_neon()
    try:
        # Load prompts
        ds_filter = ""
        if dataset == "a":
            ds_filter = "AND dataset = 'a'"
        elif dataset == "b":
            ds_filter = "AND dataset = 'b'"

        prompts = conn.execute(f"""
            SELECT p.prompt_id, p.snapshot_id, p.dataset, p.variant,
                   p.config_tag, p.system_text, p.user_text, p.row_order,
                   s.labels, s.market_json, s.roster
            FROM counterfactual_prompts p
            JOIN counterfactual_snapshots s ON s.snapshot_id = p.snapshot_id
            WHERE 1=1 {ds_filter}
            ORDER BY p.snapshot_id, p.variant
        """).fetchall()

        # Load templates for padding
        templates = {}
        tmpl_rows = conn.execute(
            "SELECT key, value FROM counterfactual_templates"
        ).fetchall()
        for r in tmpl_rows:
            templates[r["key"]] = r["value"]
    finally:
        conn.close()

    print(f"Loaded {len(prompts)} prompts from DB (dataset={dataset})")

    # Build snapshot cache for row boundary computation
    snap_cache: dict[str, Snapshot] = {}
    for p in prompts:
        sid = p["snapshot_id"]
        if sid not in snap_cache:
            mj = json.loads(p["market_json"]) if isinstance(p["market_json"], str) else p["market_json"]
            # Parse market rows from user_text
            try:
                header, row_texts = parse_market_section(p["user_text"])
                market_rows = build_market_rows(mj, row_texts)
                # Randomize to match stored row_order
                market_rows = randomize_rows(market_rows, sid)
            except Exception:
                market_rows = []
                header = ""

            snap_cache[sid] = Snapshot(
                snapshot_id=sid,
                vault_address="",
                snap_date="",
                week_num=0,
                market_json=mj,
                market_header=header,
                rows=market_rows,
            )

    capture_prompts: list[dict[str, Any]] = []

    for p in prompts:
        snap = snap_cache[p["snapshot_id"]]
        # Create a CanonicalPrompt-like object for boundary functions
        prompt_obj = CanonicalPrompt(
            snapshot_id=p["snapshot_id"],
            variant=p["variant"],
            system_text=p["system_text"],
            user_text=p["user_text"],
            row_order=list(p["row_order"]),
        )

        if p["dataset"] == "a":
            section_bounds = find_section_boundaries(tokenizer, prompt_obj)
            row_bounds = find_row_boundaries(tokenizer, prompt_obj, snap)
        else:
            # Dataset B: use downstream section boundaries
            section_bounds = find_downstream_section_boundaries(
                tokenizer, p["system_text"], p["user_text"],
            )
            row_bounds = find_row_boundaries(tokenizer, prompt_obj, snap)

        capture_prompts.append({
            "capture_id": p["prompt_id"],
            "snapshot_id": p["snapshot_id"],
            "dataset": p["dataset"],
            "variant": p["variant"],
            "config_tag": p["config_tag"],
            "system_text": p["system_text"],
            "user_text": p["user_text"],
            "section_boundaries": section_bounds,
            "row_boundaries": row_bounds,
        })

    # Generate padded variants for Dataset A
    if include_padded and dataset in ("a", "both"):
        low_preamble = templates.get("low_preamble", "")
        high_preamble = templates.get("high_preamble", "")
        if low_preamble and high_preamble:
            print("Computing padding for Dataset A variants...")
            low_padded, high_padded, target = compute_padding(
                tokenizer, low_preamble, high_preamble,
            )
            print(f"  Padding target: {target} tokens")

            # For each snapshot, build low_pad and high_pad
            # Get unique snapshots from Dataset A
            a_snapshots = set()
            a_system_text = ""
            for cp in capture_prompts:
                if cp["dataset"] == "a":
                    a_snapshots.add(cp["snapshot_id"])
                    a_system_text = cp["system_text"]

            from projects.counterfactual import render_market_section
            for sid in a_snapshots:
                snap = snap_cache[sid]
                market_section = render_market_section(snap.market_header, snap.rows)
                row_order = [r.symbol for r in snap.rows]

                for risk_tag, preamble in [("low", low_padded), ("high", high_padded)]:
                    variant = f"{risk_tag}_pad"
                    user_text = build_canonical_user_text(preamble, market_section)
                    prompt_obj = CanonicalPrompt(
                        snapshot_id=sid,
                        variant=variant,
                        system_text=a_system_text,
                        user_text=user_text,
                        row_order=row_order,
                    )
                    section_bounds = find_section_boundaries(tokenizer, prompt_obj)
                    row_bounds = find_row_boundaries(tokenizer, prompt_obj, snap)

                    capture_prompts.append({
                        "capture_id": f"{sid}_{variant}",
                        "snapshot_id": sid,
                        "dataset": "a",
                        "variant": variant,
                        "config_tag": None,
                        "system_text": a_system_text,
                        "user_text": user_text,
                        "section_boundaries": section_bounds,
                        "row_boundaries": row_bounds,
                    })

            print(f"  Added {len(a_snapshots) * 2} padded variants")

    print(f"Total capture prompts: {len(capture_prompts)}")
    return capture_prompts


# Keep old function as alias for backwards compatibility
def prepare_capture_prompts(
    spec: Any,  # DatasetASpec
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """DEPRECATED: Use load_prompts_from_db() instead."""
    from projects.counterfactual import (
        find_section_boundaries,
        find_row_boundaries,
    )

    snap_map = {s.snapshot_id: s for s in spec.snapshots}
    capture_prompts: list[dict[str, Any]] = []

    for prompt in spec.prompts:
        snap = snap_map[prompt.snapshot_id]
        section_bounds = find_section_boundaries(tokenizer, prompt)
        row_bounds = find_row_boundaries(tokenizer, prompt, snap)

        capture_prompts.append({
            "capture_id": f"{prompt.snapshot_id}_{prompt.variant}",
            "snapshot_id": prompt.snapshot_id,
            "variant": prompt.variant,
            "system_text": prompt.system_text,
            "user_text": prompt.user_text,
            "section_boundaries": section_bounds,
            "row_boundaries": row_bounds,
        })

    return capture_prompts
