# Xenon

Data pipeline for Terminal Markets inference logs → mechanistic interpretability.

Terminal Markets is a live AI trading competition where ~hundreds of AI vaults trade 16 meme tokens on Base 24/7, all running Qwen3-235B-A22B. Xenon captures the full decision-making pipeline — prompts, context, model outputs, trade executions — and then replays those prompts through smaller Qwen3 models to capture internal activations for interpretability research.

## How data flows

### Phase 1: Ingest (API → Neon Postgres)

The ingest pipeline pulls raw data from the Terminal Markets API: vault configs, inference logs (every LLM decision), full-log payloads (complete prompt + completion + context snapshots), and swap records (on-chain trade executions).

Everything lands in Neon Postgres. Full API payloads are stored as JSONB in the `full_logs.raw_payload` column. The pipeline is idempotent — re-running updates existing rows via upserts.

```bash
# Run on Modal (production)
./scripts/modal_capture.sh modal-ingest --top-n 10 --selection random

# Or run locally (requires XENON_NEON_DATABASE_URL in .env)
uv run -m pipelines.ingest --top-n 3
```

Verify with the Neon query tool:

```bash
uv run python scripts/db/neon_query.py tables
uv run python scripts/db/neon_query.py sample vaults
```

### Phase 2: Interp data prep (Neon → Neon)

The prepare pipeline joins inference logs with their full payloads (JSONB), extracts structured context blocks (messages, market data, portfolio, strategy, config, memory), normalizes decisions (trade vs observation), and assigns quality tiers.

Output: a clean `interp_examples_v0` table in Neon filtered to high-quality rows with complete context.

```bash
# Run on Modal (production)
./scripts/modal_capture.sh modal-prep
```

For market-manifold work, you can also export structured research tables directly from `full_logs.raw_payload`:

```bash
./scripts/modal_capture.sh manifold-export --output-dir data/interp_exports/manifolds --limit 1000
```

This writes:

- `tick_records.parquet`
- `asset_records.parquet`
- `pairwise_records.parquet`

`tick_records.parquet` can be used directly as an analysis label source:

```bash
uv run -m pipelines.interp.analysis \
  --activations-dir data/activations \
  --labels-path data/interp_exports/manifolds/tick_records.parquet \
  --target executed_valence
```

Useful downstream targets on that export include `executed_valence`, `forced_observe`, `decision_type`, `trade_side`, and `asset` (which falls back to `target_asset` when needed).

To build the first synthetic market-manifold dataset described in `MARKET_COUNTING_MANIFOLDS_PLAN.md`:

```bash
uv run python scripts/datasets/synthetic_market/build_synthetic_market_dataset.py \
  --output-dir data/interp_exports/synthetic_market
```

This writes:

- `synthetic_market_prompts.jsonl`
- `synthetic_market_tick_records.parquet`
- `synthetic_market_asset_records.parquet`
- `synthetic_market_pairwise_records.parquet`
- `synthetic_market_summary.json`

The initial scaffold includes:

- scalar sweep families
- pairwise tradeoff families
- archetype families
- a minimal settings ladder: `market_only`, `low_risk`, `high_risk`

All assets use neutral symbols like `A/B/C/D` rather than real token identities.

For real-decision section/row timing on full-sequence captures:

```bash
./scripts/modal_capture.sh decision-structure-pool --limit 500
```

This pools full residual captures into `data/activations/decision_structure/` and writes:

- `residual/{log_id}.safetensors` with `row_mean_i`, `row_eos_i`, `market_*`, `active_settings_*`, `portfolio_*`, `constraints_*`, `prev_decisions_*`, and `last_token`
- `tick_labels.parquet` with decision-level targets like `executed_valence`
- `asset_labels.parquet` with per-row targets like `is_buy_target`, `is_sell_target`, and `asset_executed_valence`

To probe when the model binds a buy/sell action to an asset:

```bash
./scripts/modal_capture.sh decision-structure-analyze --layers 16,24,32
```

This writes `data/analysis_results/decision_structure/decision_structure_results.json` with:

- per-layer decodability for `is_target_asset`, `is_buy_target`, and `is_sell_target`
- comparisons between pre-market row states and row+downstream representations
- a summary of the best pre vs best post AUROC for each target

To make capture selection cohort-aware instead of pulling the earliest generic slice, install the decision cohort views in Neon:

```bash
uv run python scripts/db/apply_decision_cohort_views.py apply
uv run python scripts/db/apply_decision_cohort_views.py stats
uv run python scripts/db/apply_decision_cohort_views.py refresh
```

This creates:

- `decision_capture_base_v1`
- `decision_trade_candidates_v1`
- `decision_sell_candidates_v1`
- `decision_blocked_observe_candidates_v1`
- `decision_policy_tension_candidates_v1`
- `decision_capture_priority_v1`

`apply` rebuilds the materialized base/priority relations. `refresh` is cheaper and is the right thing to run after backfill ingest adds more logs.

You can then drive capture or pooling from a specific cohort:

```bash
uv run --extra interp --extra modal modal run pipelines/interp/modal_vllm_capture.py \
  --mode capture \
  --cohort-view decision_capture_priority_v1 \
  --order-mode capture_priority_desc \
  --limit 500

./scripts/modal_capture.sh decision-structure-pool \
  --cohort-view decision_capture_priority_v1 \
  --order-mode capture_priority_desc \
  --limit 500
```

To build a balanced full-sequence decision-capture manifest for the next probe run:

```bash
uv run python scripts/manifests/build_decision_capture_manifest.py build
uv run python scripts/manifests/build_decision_capture_manifest.py stats
```

The default manifest publishes `decision_capture_manifest_v1` with a quota/diversity plan tuned for the decision-structure probes:

- target mix: `300 buy`, `300 sell`, `200 policy_tension_observe`, `200 blocked_observe`
- diversity caps: `max 1 row per cohort per vault`, `max 4 rows total per vault`
- trade-asset cap: `60` per side per asset
- deep scan: up to `70k` rows per cohort so concentrated priority rankings can still reach long-tail vaults

Once built, use the manifest directly for capture/pooling:

```bash
uv run --extra interp --extra modal modal run pipelines/interp/modal_vllm_capture.py \
  --mode capture \
  --cohort-view decision_capture_manifest_v1 \
  --order-mode selection_rank_asc \
  --limit 959

./scripts/modal_capture.sh decision-structure-pool \
  --cohort-view decision_capture_manifest_v1 \
  --order-mode selection_rank_asc \
  --limit 959 \
  --num-shards 10 \
  --num-workers 8 \
  --no-skip-existing
```

For pre/post settings structure on counterfactual captures:

```bash
./scripts/modal_capture.sh counterfactual-structure --experiment-id init --layers 16,24,32
```

This writes a `structure_results.json` summary under `data/analysis_results/counterfactual_structure/<experiment_id>/` with:

- position-wise probe retention from `row_mean` / `row_eos` into downstream sections
- market-subspace retention for `settings_eos`, `portfolio_eos`, `constraints_eos`, `prev_decisions_eos`, and `last_token`
- settings-shift decomposition into parallel-vs-orthogonal movement relative to the market subspace

### Phase 2b: Trade outcomes (API → Neon)

Enriches swaps with forward-looking PnL by fetching candle data from the Terminal Markets API.

```bash
./scripts/modal_capture.sh modal-outcomes
```

### Phase 3: Modal activation capture (Qwen3-30B-A3B, MoE)

Runs on Modal with A100-80GB GPUs, using Qwen3-30B-A3B which has MoE layers (128 experts, top-8 active). Captures both residual-stream activations and MoE router logits — the router logits are the primary signal for interpretability since they reveal which experts the model selects for each token.

```bash
# One-time: cache model weights to Modal volume
./scripts/modal_capture.sh download

# Smoke test: 1 example, layer 24
./scripts/modal_capture.sh smoke

# Router logits only (recommended for scale — small and information-dense)
./scripts/modal_capture.sh router

# Full capture (residual + router)
./scripts/modal_capture.sh full
```

## Data layout

```
Neon Postgres (XENON_NEON_DATABASE_URL):
├── vaults                  # vault configs, persona, risk params
├── strategies              # per-vault trading strategies
├── inference_logs          # every LLM decision (tool, args, status)
├── full_logs               # parsed fields + raw_payload JSONB
├── swaps                   # on-chain trade executions
├── trade_outcomes          # forward-looking PnL (1h, 4h, 1d)
├── interp_examples_v0      # denormalized prep output
└── ingest_cursors          # pagination state for incremental ingest

Modal Volume (xenon-data):
├── activations/
│   ├── residual_stream/
│   │   └── {log_id}.safetensors     # (L, seq_len, hidden_dim) fp16
│   ├── router_logits/
│   │   └── {log_id}.safetensors     # (L, seq_len, 128) fp32 + indices
│   └── metadata.parquet
└── dashboard_stats.json             # cached stats for dashboard
```

## Setup

```bash
uv sync                              # base deps (ingest + data prep)
uv sync --extra interp               # + torch, transformers, safetensors
uv sync --extra modal                # + modal (remote capture only)
uv sync --extra dev                  # + pytest
```

Environment:
```bash
# .env file (or set directly)
XENON_NEON_DATABASE_URL=postgresql://...   # Neon Postgres DSN
```

For Modal:
```bash
modal token new                                        # authenticate
modal secret create xenon-neon XENON_NEON_DATABASE_URL=postgresql://...
modal secret create huggingface HF_TOKEN=<your-token>  # for model downloads
```

## Tests

```bash
uv run --extra dev -m pytest tests/ -v
```
