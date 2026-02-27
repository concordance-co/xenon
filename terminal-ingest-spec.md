# Terminal Markets Inference Ingestion Pipeline — Spec

## Goal

Build a data ingestion pipeline that captures every AI inference log and its associated trade outcome from the DX Terminal Pro platform (terminal.markets). The resulting dataset will be used for mechanistic interpretability research — specifically, replaying prompts through a smaller Qwen3 model to capture activations and train SAEs to isolate features involved in financial decision-making.

The final dataset maps: **full inference context → decision → outcome (PnL)**.

---

## Platform Context

DX Terminal Pro is a live trading competition on Base (L2). ~hundreds of AI agent "vaults" trade 16 meme tokens against each other 24/7. Every agent runs **Qwen3-235B-A22B** (MoE, 128 experts, 8 active per token) on the same inference stack. The only variable between agents is the prompt context: vault configuration, strategy text, market state, and token data fed to the model.

The competition runs from Feb 26 – Mar 19, 2026. Tokens are eliminated ("reaped") at intervals. One token graduates and becomes publicly tradable.

**Why this matters for interp:** Same model weights, different inputs, different decisions, real financial outcomes. Naturally controlled experiment. Every inference has a ground-truth label (trade PnL).

---

## API Reference

**Base URL:** `https://api.terminal.markets/api/v1`

No authentication required (public read API). No documented rate limits — be conservative and implement backoff. All responses are JSON.

The full Swagger 2.0 spec is attached separately. Key endpoints below.

### Vault Discovery

**`GET /leaderboard`** — Paginated list of all vaults ranked by PnL. Use this to seed the vault list.

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `limit` | int | yes | 0-50 |
| `cursor` | string | no | Pagination cursor from previous response |
| `sortBy` | string | yes | `total_pnl_usd`, `realized_pnl_usd`, or `unrealized_pnl_usd` |
| `vaultAddress` | string | no | Include a specific vault's rank |

Response includes `items[]` of vault entries with `vaultAddress`, `ownerAddress`, `nftId`, `nftName`, PnL fields, `rank`, and `cursor`. `hasMoreItems` boolean for pagination. `totalCount` for total vaults.

**`GET /vault`** — Get vault config by address or owner.

| Param | Type | Notes |
|-------|------|-------|
| `vaultAddress` | string | |
| `ownerAddress` | string | |

Returns vault config: `persona`, `tradeSize`, `tradingActivity`, `holdingStyle`, `diversification`, `assetRiskPreference`, `maxTradeAmount`, `slippageBps`, `paused`, `state`. These are the human-set parameters that shape each agent's behavior — important context for the inference prompt.

### Inference Logs (Primary Target)

**`GET /logs/{vaultAddress}`** — Paginated inference log metadata for a vault.

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `vaultAddress` | string (path) | yes | |
| `limit` | int | yes | 1-50 |
| `order` | string | yes | `asc` or `desc` |
| `status` | string | no | Filter by status |
| `tool` | string | no | Filter by tool |
| `cursor` | string | no | Pagination cursor |

Each log entry contains: `id`, `vault_address`, `status`, `tool` (the action taken), `tool_args` (JSON), `reasoning` (text summary), `strategyId`, `transactionHash` (if a swap resulted), `inference_duration_ms`, `error`, `created_at`, `completed_at`, `cursor`.

**`GET /full-log/{id}`** — The complete inference payload for a given log ID.

Returns a raw JSON object — this is the full prompt + completion. This is the critical endpoint. The exact schema of the payload isn't documented in Swagger (it's typed as `object`), so you'll need to inspect the first few responses to understand the structure. Expect it to contain the system prompt, user message(s) with market data/strategy context, and the model's response including tool calls.

**`GET /log/{id}`** — Single inference log metadata by ID (same shape as items from `/logs/{vaultAddress}`).

### Trade Outcomes

**`GET /swaps`** — Paginated swaps, filterable by vault and/or token.

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `vaultAddress` | string | no | |
| `tokenAddress` | string | no | |
| `strategyIds` | string | no | Comma-separated, requires vaultAddress |
| `limit` | int | yes | 1-50 |
| `order` | string | yes | `asc` or `desc` |
| `cursor` | string | no | |

Each swap includes: `transactionHash`, `vaultAddress`, `tokenAddress`, `tokenSymbol`, `tokenName`, `side` (buy/sell), `ethAmount` (wei), `tokenAmount` (raw), `effectivePriceEth`, `effectivePriceUsd`, `ethPriceUsd`, `timestamp`, `blockNumber`, `logId` (links back to inference), `strategyId`, `isReapTwap`, `poolId`.

**The `logId` field is the join key** — it links a swap execution back to the inference that triggered it.

**`GET /swap/{transactionHash}`** — Single swap lookup.

### Enrichment / Context

**`GET /positions/{vaultAddress}`** — Current portfolio state. Per-token balances, cost basis, realized/unrealized PnL. Useful for understanding what the model "saw" at inference time.

**`GET /strategies/{vaultAddress}`** — Strategy text set by the vault owner. The `content` field is the user-written strategy that gets injected into the inference prompt. `activeOnly=true` filters to current strategies.

**`GET /tokens?includeMarketData=true`** — All tokens with price, volume, and timeframe stats (15m/1h/4h/1d). Useful for computing post-trade price movement.

**`GET /candles/{tokenAddress}`** — OHLCV candles. Timeframes: 1m/5m/15m/1h/4h/1d. Useful for computing PnL at specific time horizons after a trade.

**`GET /eth-price`** — ETH/USD price, optionally at a specific block or timestamp.

**`GET /activity/{vaultAddress}`** — Combined activity feed (swaps, deposits, withdrawals, presale events, reap events, vault summaries). Each activity has a `type` field and the corresponding sub-object populated. Useful as an alternative to polling swaps separately.

### Streaming

**`GET /stream`** — SSE stream of vault and narration events. Optional `vaultAddress` param for vault-specific events.

**`GET /stream/activity`** — SSE stream of global activity events (buy/sell/deposit/observe across all vaults). High volume. Useful for real-time ingestion triggers but not required if polling is sufficient.

---

## Data Model

### Core Tables

```
vaults
├── vault_address (PK)
├── owner_address
├── nft_id, nft_name
├── persona (JSON)
├── trade_size, trading_activity, holding_style
├── diversification, asset_risk_preference
├── max_trade_amount, slippage_bps
├── paused, state
├── leaderboard_rank
├── total_pnl_usd, realized_pnl_usd, unrealized_pnl_usd
└── updated_at

inference_logs
├── id (PK)
├── vault_address (FK → vaults)
├── status
├── tool
├── tool_args (JSON)
├── reasoning
├── strategy_id
├── transaction_hash
├── execution_key, request_id
├── inference_duration_ms
├── error
├── created_at, completed_at
├── cursor
└── fetched_at

full_logs
├── log_id (PK, FK → inference_logs)
├── payload (JSON) — the raw full inference payload
├── prompt_text — extracted/concatenated prompt
├── completion_text — extracted model output
├── tool_calls (JSON) — extracted structured tool calls
├── token_count_estimate — rough token count for budgeting replay compute
└── fetched_at

swaps
├── transaction_hash (PK)
├── vault_address
├── log_id (FK → inference_logs, nullable — not all swaps are inference-triggered)
├── token_address, token_symbol, token_name
├── side (buy/sell)
├── eth_amount_wei, token_amount_raw
├── effective_price_eth, effective_price_usd
├── eth_price_usd
├── strategy_id
├── is_reap_twap
├── block_number, timestamp
└── fetched_at

strategies
├── strategy_id (PK composite with vault_address)
├── vault_address (FK → vaults)
├── content — the user-written strategy text
├── enabled
├── expiry
├── strategy_priority
├── created_block, updated_block
└── fetched_at

tokens
├── token_address (PK)
├── symbol, name, description
├── type (presale_token / agent_token)
├── reaped (bool)
├── total_supply
├── pool_id
└── updated_at
```

### Derived / Analysis Tables

```
trade_outcomes
├── log_id (PK, FK → inference_logs)
├── transaction_hash (FK → swaps)
├── side
├── token_address
├── entry_price_usd
├── price_1h_usd — token price 1 hour after trade
├── price_4h_usd
├── price_1d_usd
├── pnl_1h_pct — percentage price change at 1h
├── pnl_4h_pct
├── pnl_1d_pct
├── was_profitable_1h (bool)
└── computed_at
```

Build `trade_outcomes` as a post-processing step using the candles endpoint to look up prices at t+1h, t+4h, t+1d after each swap timestamp.

### Storage

Use **SQLite** for development / small scale. The full dataset over the 21-day competition is likely manageable (hundreds of vaults × maybe 50-200 inferences/day each = low millions of rows max, but full_logs payloads could be large).

For the actual interp pipeline, export to **Parquet** — the prompt/completion text columns compress well and Parquet is what you'll want for batch processing with PyTorch datasets.

---

## Ingestion Pipeline Architecture

### Phase 1: Vault Discovery

1. Paginate through `GET /leaderboard?sortBy=total_pnl_usd&limit=50` until `hasMoreItems=false`
2. For each vault, fetch `GET /vault?vaultAddress={addr}` to get config
3. For each vault, fetch `GET /strategies/{addr}` to get strategy text
4. Store all in `vaults` and `strategies` tables
5. Re-run periodically (hourly?) to catch new vaults and updated configs

### Phase 2: Inference Log Collection (Primary Loop)

For each known vault, poll for new inference logs:

1. `GET /logs/{vaultAddress}?limit=50&order=asc&cursor={last_cursor}`
2. Store each log in `inference_logs`
3. For each log ID not yet in `full_logs`, fetch `GET /full-log/{id}`
4. Parse the full payload — extract prompt text, completion text, tool calls
5. Store in `full_logs`
6. Track the cursor per vault for incremental polling

**Polling interval:** The competition description says agents trade "24/7" but inference frequency depends on the agent's `tradingActivity` setting. Start with polling every 60 seconds per vault. Adjust based on observed throughput.

**Concurrency:** Run vault polling concurrently (asyncio) but with a global semaphore to avoid hammering the API. Start with 5-10 concurrent requests. Implement exponential backoff on 429/5xx.

### Phase 3: Swap Collection

Two approaches, use both:

1. **Via inference logs:** When an inference log has a `transactionHash`, fetch `GET /swap/{txHash}` and store in `swaps`. This gives you the direct inference→swap link.
2. **Bulk backfill:** Periodically `GET /swaps?vaultAddress={addr}&limit=50&order=asc&cursor={last_cursor}` per vault to catch any swaps not linked to inference logs (e.g., reap TWAPs).

### Phase 4: Outcome Labeling (Post-Processing)

After swaps are collected, compute trade outcomes:

1. For each swap, note the `tokenAddress` and `timestamp`
2. Fetch `GET /candles/{tokenAddress}?timeframe=1h&from={timestamp}&to={timestamp+86400}` to get price data after the trade
3. Compute price at t+1h, t+4h, t+1d relative to the swap's `effectivePriceUsd`
4. Store in `trade_outcomes`

This can run as a separate batch job since it's backward-looking. Run it with a delay (e.g., only compute outcomes for trades older than 24h so the 1d price is available).

### Phase 5: Token & Market Snapshots (Optional but Useful)

Periodically snapshot `GET /tokens?includeMarketData=true` to build a time series of token prices and volumes. This helps reconstruct what market state the model was seeing at inference time, even if the full-log payload already contains it.

---

## Pipeline Implementation Notes

### Language & Stack

Python with asyncio + aiohttp. The pipeline is I/O bound (API polling), not compute bound.

Dependencies: `aiohttp`, `aiosqlite` (or just `sqlite3` if synchronous is fine), `pyarrow` (for Parquet export).

### Key Design Decisions

**Idempotency:** Inference log IDs and transaction hashes are natural deduplication keys. Use `INSERT OR IGNORE` / `ON CONFLICT DO NOTHING` so re-running the pipeline is safe.

**Cursor management:** Store the last cursor per vault in a simple `cursors` table (`vault_address`, `endpoint`, `last_cursor`, `updated_at`). On restart, resume from last cursor.

**Full log parsing:** The `/full-log/{id}` payload structure isn't documented. On first fetch, log the raw JSON and inspect it. Build a parser that extracts:
- System prompt text
- User message(s) — likely contains market data, portfolio state, strategy text
- Assistant response — the model's reasoning and tool call
- Tool call name and arguments

This parser will need to be adjusted once you see the actual payload shape. Wrap it in a try/except and always store the raw payload regardless of parsing success.

**Rate limiting:** No documented rate limits. Implement:
- Global semaphore (start at 10 concurrent requests)
- Per-request retry with exponential backoff (1s, 2s, 4s, 8s, max 60s)
- Back off on 429 or 5xx responses
- Log all rate limit events

**Graceful shutdown:** Handle SIGINT/SIGTERM, flush pending writes, save cursors.

### Export Format

For the interp pipeline, export a Parquet dataset with this schema per row:

| Column | Type | Description |
|--------|------|-------------|
| `log_id` | int | Unique inference ID |
| `vault_address` | string | |
| `prompt_text` | string | Full concatenated prompt |
| `completion_text` | string | Model output |
| `tool` | string | Action taken (buy/sell/hold/observe) |
| `tool_args` | string (JSON) | Structured action parameters |
| `reasoning` | string | Model's stated reasoning |
| `strategy_content` | string | User-set strategy text for this vault |
| `side` | string | buy/sell/null |
| `token_symbol` | string | Token traded (if any) |
| `effective_price_usd` | float | Execution price |
| `eth_price_usd` | float | ETH/USD at time of trade |
| `pnl_1h_pct` | float | Price change 1h after trade |
| `pnl_4h_pct` | float | Price change 4h after trade |
| `pnl_1d_pct` | float | Price change 1d after trade |
| `was_profitable_1h` | bool | Binary label |
| `inference_duration_ms` | int | |
| `timestamp` | int | Unix timestamp |
| `vault_trade_size` | int | Vault config |
| `vault_trading_activity` | int | Vault config |
| `vault_holding_style` | int | Vault config |
| `vault_diversification` | int | Vault config |
| `vault_risk_preference` | int | Vault config |

This is the dataset you feed to the replay + activation capture pipeline.

---

## Replay & Interp Context (Out of Scope for Ingestion, But Informing Design)

The downstream plan is:

1. Take the `prompt_text` from each row
2. Run it through a smaller Qwen3 model (e.g., Qwen3-8B or Qwen3-4B) — same tokenizer, same architecture family
3. Capture activations at selected layers during forward pass
4. Train SAEs on those activations, labeled by trade outcome
5. Look for features that correlate with: profitable vs unprofitable decisions, buy vs sell decisions, risk-taking patterns, specific token preferences

The ingestion pipeline's job is to produce a clean, complete, well-labeled dataset so the replay pipeline doesn't need to touch the API at all.

---

## Operational Concerns

**Timeline:** The competition runs Feb 26 – Mar 19. Trading is live now. Start ingesting immediately and backfill historical data. The reaping schedule means tokens get eliminated over time — earlier data has more tokens and more diverse market conditions.

**Data volume estimate:** With ~hundreds of vaults, each running inference maybe every few minutes, expect roughly 100K-1M+ inference logs over the full competition. Full log payloads could be 5-50KB each (depending on how much market context is in the prompt), so total storage in the range of 5-50GB for raw payloads.

**Backfill:** Use `order=asc` with no cursor to start from the beginning of each vault's history. The API supports cursor-based pagination in both directions.

**Monitoring:** Log ingestion rate (logs/minute), full-log fetch success rate, parsing success rate, and lag (time between log creation and ingestion). Alert if any vault stops producing logs (might indicate the agent is paused or the vault is inactive).
