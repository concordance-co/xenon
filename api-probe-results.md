# Terminal Markets API Probe Results

**Date:** 2026-02-27
**Base URL:** `https://api.terminal.markets/api/v1`
**Auth:** None required (public read API)
**Note:** The frontend at `terminal.markets` is behind Vercel security checkpoint. The API subdomain `api.terminal.markets` works directly with curl.

---

## 1. Leaderboard

### Request

```
GET https://api.terminal.markets/api/v1/leaderboard?offset=0&limit=5&sortBy=total_pnl_usd
```

### HTTP Status: 200

### Response

```json
{
  "totalCount": 2181,
  "hasMoreItems": true,
  "items": [
    {
      "cursor": "1",
      "rank": 1,
      "vaultAddress": "0x496D1400BB933fF4683dBe42445D5c89cFc10F95",
      "ownerAddress": "0x5d3b78C158e3856DFb4334bfF2a6F3c09B74bd93",
      "nftId": "10875",
      "nftName": "NetSpy60",
      "realizedPnlEth": "59121758743752160145",
      "unrealizedPnlEth": "49458383578496190836",
      "totalPnlEth": "108580142322248350981",
      "realizedPnlUsd": 119569.27558929,
      "unrealizedPnlUsd": 95169.11378201336,
      "totalPnlUsd": 214738.38937130335
    },
    {
      "cursor": "2",
      "rank": 2,
      "vaultAddress": "0x5bA79C21e0D5BECdF3FdC24Bc401e48Ae96cFa4D",
      "ownerAddress": "0x17412B2B513584C382E4E8dc34fC776A41395636",
      "nftId": "29980",
      "nftName": "TallyCat",
      "realizedPnlEth": "73523018513244755954",
      "unrealizedPnlEth": "0",
      "totalPnlEth": "73523018513244755954",
      "realizedPnlUsd": 148215.34292981,
      "unrealizedPnlUsd": 0,
      "totalPnlUsd": 148215.34292981
    },
    {
      "cursor": "3",
      "rank": 3,
      "vaultAddress": "0xE5bfF72e19ccDDad190D1E6B10DaB8490f51430E",
      "ownerAddress": "0x076f3814A9F7E4d8f4DdB653850Af7400D3182a1",
      "nftId": "24051",
      "nftName": "PolicyNerd",
      "realizedPnlEth": "70395220347364812027",
      "unrealizedPnlEth": "0",
      "totalPnlEth": "70395220347364812027",
      "realizedPnlUsd": 142300.06364781,
      "unrealizedPnlUsd": 0,
      "totalPnlUsd": 142300.06364781
    },
    {
      "cursor": "4",
      "rank": 4,
      "vaultAddress": "0x5faDb4857B1e22f03E7f90114E7304B24D393C8E",
      "ownerAddress": "0xA3e3376b2395d6E598dba44e3609310b0B6f90bC",
      "nftId": "27916",
      "nftName": "FunyunEcho",
      "realizedPnlEth": "16953266732116722234",
      "unrealizedPnlEth": "35263656416735232083",
      "totalPnlEth": "52216923148851954317",
      "realizedPnlUsd": 34298.28451753,
      "unrealizedPnlUsd": 67855.24894010583,
      "totalPnlUsd": 102153.53345763581
    },
    {
      "cursor": "5",
      "rank": 5,
      "vaultAddress": "0xCA5e5296EF231eF4B4718390dC551D5D9d5c4dbF",
      "ownerAddress": "0x000461A73d3985eef4923655782aA5d0De75C111",
      "nftId": "19415",
      "nftName": "RetroNautX",
      "realizedPnlEth": "16737291127890228584",
      "unrealizedPnlEth": "34893526300623229906",
      "totalPnlEth": "51630817428513458490",
      "realizedPnlUsd": 33839.31497495,
      "unrealizedPnlUsd": 67143.03490103383,
      "totalPnlUsd": 100982.34987598383
    }
  ]
}
```

### Schema Summary

| Field | Type | Notes |
|-------|------|-------|
| `totalCount` | int | Total vaults in leaderboard (2181) |
| `hasMoreItems` | bool | Pagination flag |
| `items[]` | array | Vault entries |
| `items[].cursor` | string | Pagination cursor (numeric string) |
| `items[].rank` | int | Leaderboard rank |
| `items[].vaultAddress` | string | Ethereum address (0x-prefixed) |
| `items[].ownerAddress` | string | Ethereum address |
| `items[].nftId` | string | NFT identifier (numeric string) |
| `items[].nftName` | string | Human-readable vault name |
| `items[].realizedPnlEth` | string | Wei value as string (big integer) |
| `items[].unrealizedPnlEth` | string | Wei value as string |
| `items[].totalPnlEth` | string | Wei value as string |
| `items[].realizedPnlUsd` | float | USD value |
| `items[].unrealizedPnlUsd` | float | USD value |
| `items[].totalPnlUsd` | float | USD value |

**Key observations:**
- ETH values are returned as string-encoded wei (18 decimals). USD values are floats.
- Pagination uses cursor-based approach (not offset). The `cursor` field on each item is the value to pass as `cursor` param for next page.
- The `sortBy` param is required. Options: `total_pnl_usd`, `realized_pnl_usd`, `unrealized_pnl_usd`.

---

## 2. Vault Details

### 2a. Vault Config

#### Request

```
GET https://api.terminal.markets/api/v1/vault?vaultAddress=0x496D1400BB933fF4683dBe42445D5c89cFc10F95
```

#### HTTP Status: 200

#### Response

```json
{
  "vaultAddress": "0x496D1400BB933fF4683dBe42445D5c89cFc10F95",
  "nftId": "10875",
  "nftName": "NetSpy60",
  "ownerAddress": "0x5d3b78C158e3856DFb4334bfF2a6F3c09B74bd93",
  "maxTradeAmount": "3300",
  "slippageBps": "1500",
  "tradingActivity": 5,
  "assetRiskPreference": 1,
  "tradeSize": 3,
  "holdingStyle": 3,
  "diversification": 3,
  "persona": null,
  "paused": false,
  "state": "open",
  "createdBlock": 42669076,
  "updatedBlock": 42714421
}
```

#### Schema Summary

| Field | Type | Notes |
|-------|------|-------|
| `vaultAddress` | string | |
| `nftId` | string | |
| `nftName` | string | |
| `ownerAddress` | string | |
| `maxTradeAmount` | string | Numeric string (bps or USD unclear) |
| `slippageBps` | string | Basis points as string |
| `tradingActivity` | int | 1-5 scale |
| `assetRiskPreference` | int | 1-5 scale |
| `tradeSize` | int | 1-5 scale |
| `holdingStyle` | int | 1-5 scale |
| `diversification` | int | 1-5 scale |
| `persona` | object/null | Can be null; when populated contains personality traits |
| `paused` | bool | |
| `state` | string | e.g. "open" |
| `createdBlock` | int | Base L2 block number |
| `updatedBlock` | int | Base L2 block number |

### 2b. Strategies

#### Request

```
GET https://api.terminal.markets/api/v1/strategies/0x496D1400BB933fF4683dBe42445D5c89cFc10F95
```

#### HTTP Status: 200

#### Response (13 strategies returned, showing first 3)

```json
[
  {
    "vaultAddress": "0x496D1400BB933fF4683dBe42445D5c89cFc10F95",
    "strategyId": "0",
    "vaultOwnerAddress": "0x5d3b78C158e3856DFb4334bfF2a6F3c09B74bd93",
    "content": "sell 25% of tokens on token generative event if the market cap is at least $600,000",
    "expiry": 1772216483,
    "enabled": false,
    "strategyPriority": "high",
    "createdBlock": 42670370,
    "updatedBlock": 42670393
  },
  {
    "vaultAddress": "0x496D1400BB933fF4683dBe42445D5c89cFc10F95",
    "strategyId": "1",
    "vaultOwnerAddress": "0x5d3b78C158e3856DFb4334bfF2a6F3c09B74bd93",
    "content": "sell 25% of LOOKSMAX tokens on token generative event if the market cap is at least $600,000",
    "expiry": 1772216580,
    "enabled": false,
    "strategyPriority": "med",
    "createdBlock": 42670420,
    "updatedBlock": 42670794
  },
  {
    "vaultAddress": "0x496D1400BB933fF4683dBe42445D5c89cFc10F95",
    "strategyId": "12",
    "vaultOwnerAddress": "0x5d3b78C158e3856DFb4334bfF2a6F3c09B74bd93",
    "content": "Sell 5% of LMAO holdings every 15 minutes to capture volatility gains (current 1h +49.50%) while maintaining exposure to potential continuation",
    "expiry": 1772304560,
    "enabled": true,
    "strategyPriority": "high",
    "createdBlock": 42714409,
    "updatedBlock": 42714409
  }
]
```

#### Schema Summary

| Field | Type | Notes |
|-------|------|-------|
| `vaultAddress` | string | |
| `strategyId` | string | Numeric string, incrementing per vault |
| `vaultOwnerAddress` | string | |
| `content` | string | Free-text strategy instruction (injected into prompt) |
| `expiry` | int | Unix timestamp |
| `enabled` | bool | Only enabled strategies are active |
| `strategyPriority` | string | "high", "med", or "low" |
| `createdBlock` | int | |
| `updatedBlock` | int | |

**Key observations:**
- Response is a flat array (not paginated).
- Most strategies for this vault are disabled; only strategyId "12" is enabled.
- Strategy `content` is natural language -- this is what gets injected into the LLM prompt.

---

## 3. Inference Logs

### 3a. Log List

#### Request

```
GET https://api.terminal.markets/api/v1/logs/0x496D1400BB933fF4683dBe42445D5c89cFc10F95?limit=3&order=desc
```

#### HTTP Status: 200

#### Response

```json
{
  "totalCount": 102,
  "hasMoreItems": true,
  "items": [
    {
      "cursor": "434158",
      "id": 434158,
      "vault_address": "0x496D1400BB933fF4683dBe42445D5c89cFc10F95",
      "request_id": "c8142e59-51ad-461b-ba90-017d9d848bba:0",
      "execution_key": "0x496D1400BB933fF4683dBe42445D5c89cFc10F95:c8142e59-51ad-461b-ba90-017d9d848bba:0",
      "tool": "record_observation",
      "tool_args": {
        "content": "HIGH status: blocked by timing for strategy12 (next sell due in ~10.5m); Risk=1 preference rules out buys as HOLE shows strongest momentum (1h +32.43%) but exceeds low-volatility tolerance, while LMAO 5m flat price fails fresh-signal gate for re-entry.",
        "strategy": "strategy12"
      },
      "strategyId": "12",
      "status": "EXECUTED",
      "inference_duration_ms": 12949,
      "created_at": "2026-02-27T20:05:10.090491Z",
      "completed_at": "2026-02-27T20:05:10.095172Z"
    },
    {
      "cursor": "432750",
      "id": 432750,
      "vault_address": "0x496D1400BB933fF4683dBe42445D5c89cFc10F95",
      "request_id": "c68e2616-9e31-4d08-b785-390fe9573039:0",
      "execution_key": "0x496D1400BB933fF4683dBe42445D5c89cFc10F95:c68e2616-9e31-4d08-b785-390fe9573039:0",
      "tool": "sell_token",
      "tool_args": {
        "token": "LMAO",
        "content": "HIGH status: unfulfilled -> executing strategy12 interval (14m31s elapsed); LMAO 5m momentum (+0.00%) with stable flow confirms systematic volatility capture per persistent directive despite minor price stability, aligning with 15m profit-taking schedule.",
        "strategy": "strategy12",
        "spend_pct": 5.0
      },
      "strategyId": "12",
      "status": "EXECUTED",
      "inference_duration_ms": 20918,
      "created_at": "2026-02-27T20:00:17.550233Z",
      "completed_at": "2026-02-27T20:00:18.661511Z",
      "transactionHash": "0x19c3fefbc4ed41992a656bcb714e6bb43a11655fb983bf74088721975df29b4a"
    },
    {
      "cursor": "431672",
      "id": 431672,
      "vault_address": "0x496D1400BB933fF4683dBe42445D5c89cFc10F95",
      "request_id": "eff9fd7d-e77a-4831-88a6-76772f61d949:0",
      "execution_key": "0x496D1400BB933fF4683dBe42445D5c89cFc10F95:eff9fd7d-e77a-4831-88a6-76772f61d949:0",
      "tool": "record_observation",
      "tool_args": {
        "content": "HIGH status: blocked by timing for strategy12 (next sell due in ~5.5m); Risk=1 preference blocks buys as AIGF shows weakest volatility (1h -0.24%) but negative 5m momentum (-0.71%) and net outflow fails fresh-signal gate, while other tokens exceed low-volatility tolerance with >7% 1h swings.",
        "strategy": "strategy12"
      },
      "strategyId": "12",
      "status": "EXECUTED",
      "inference_duration_ms": 52786,
      "created_at": "2026-02-27T19:55:48.930274Z",
      "completed_at": "2026-02-27T19:55:48.935575Z"
    }
  ]
}
```

#### Schema Summary

| Field | Type | Notes |
|-------|------|-------|
| `totalCount` | int | Total logs for this vault |
| `hasMoreItems` | bool | |
| `items[].cursor` | string | Pagination cursor (same as id, stringified) |
| `items[].id` | int | Unique log ID (globally unique) |
| `items[].vault_address` | string | |
| `items[].request_id` | string | UUID format with `:0` suffix |
| `items[].execution_key` | string | `{vaultAddress}:{request_id}` composite |
| `items[].tool` | string | Action taken: `buy_token`, `sell_token`, `record_observation` |
| `items[].tool_args` | object | Varies by tool (see below) |
| `items[].strategyId` | string | Which strategy triggered this inference |
| `items[].status` | string | e.g. "EXECUTED" |
| `items[].inference_duration_ms` | int | Time for LLM inference |
| `items[].created_at` | string | ISO 8601 timestamp |
| `items[].completed_at` | string | ISO 8601 timestamp |
| `items[].transactionHash` | string | Only present if a swap was executed |

**tool_args shapes by tool type:**

- `record_observation`: `{ content: string, strategy: string }`
- `sell_token`: `{ token: string, content: string, strategy: string, spend_pct: float }`
- `buy_token`: `{ token: string, content: string, strategy: string|null, spend_pct: float }`

### 3b. Full Log (Complete Inference Payload)

#### Request

```
GET https://api.terminal.markets/api/v1/full-log/434158
```

#### HTTP Status: 200

#### Response (truncated -- full response is ~52KB)

The full-log is the most complex and data-rich endpoint. Top-level structure:

```json
{
  "id": "0x496D...cFc10F95:c8142e59-...:0",
  "vault_address": "0x496D1400BB933fF4683dBe42445D5c89cFc10F95",
  "nft_id": "10875",
  "request_id": "c8142e59-51ad-461b-ba90-017d9d848bba:0",
  "status": "LOGGED",
  "error": "",
  "created_at": "2026-02-27T20:05:10.090032773Z",
  "completed_at": "2026-02-27T20:05:10.089973497Z",
  "inference_duration_ms": 12949,
  "tool": "record_observation",
  "tool_args": { "content": "...", "strategy": "strategy12" },
  "reasoning": "Invoking record_observation tool.",
  "snapshot": { "..." },
  "llm_request_payload": { "..." },
  "llm_completion_payload": { "..." },
  "llm_request_hash": "0x692b13ff...",
  "llm_completion_hash": "0xa7140c00..."
}
```

#### Schema Summary -- Top Level

| Field | Type | Notes |
|-------|------|-------|
| `id` | string | Composite: `{vaultAddress}:{requestId}` |
| `vault_address` | string | |
| `nft_id` | string | |
| `request_id` | string | |
| `status` | string | "LOGGED" |
| `error` | string | Empty on success |
| `created_at` | string | ISO 8601 with nanoseconds |
| `completed_at` | string | ISO 8601 with nanoseconds |
| `inference_duration_ms` | int | |
| `tool` | string | Action taken |
| `tool_args` | object | Tool call arguments |
| `reasoning` | string | Brief reasoning summary |
| `snapshot` | object | Full state snapshot at inference time |
| `llm_request_payload` | object | Complete LLM API request |
| `llm_completion_payload` | object | Complete LLM API response |
| `llm_request_hash` | string | Hash of request payload |
| `llm_completion_hash` | string | Hash of completion payload |

#### Schema Summary -- `snapshot`

```
snapshot
├── Agent
│   ├── VaultAddress: string
│   ├── OwnerAddress: string
│   ├── CurrentNftId: string
│   ├── State: string
│   ├── Paused: bool
│   ├── Options
│   │   ├── max_trade_amount: int
│   │   ├── slippage_bps: int
│   │   ├── max_price_impact_bps: int
│   │   ├── trading_activity: int
│   │   ├── asset_risk_preference: int
│   │   ├── trade_size: int
│   │   ├── holding_style: int
│   │   └── diversification: int
│   ├── Persona
│   │   ├── Animal: string
│   │   ├── AgeRange: string
│   │   ├── Gender: string
│   │   ├── Hobbies: array|null
│   │   ├── Occupation: string
│   │   └── WritingStyle: string
│   └── Strategies[]
│       ├── strategyId: string
│       ├── content: string
│       ├── expiryUnix: int
│       └── strategyPriority: string
├── Portfolio
│   ├── EthBalance: int (wei)
│   └── Tokens[]
│       ├── TokenAddress: string
│       ├── Symbol: string
│       ├── Name: string
│       ├── Balance: int (raw token amount)
│       ├── AvgEntryPriceInEth: float
│       ├── UnrealizedPnlPercent: float
│       ├── TimeSinceLastSwapOrGenesisOrReap: int (seconds)
│       └── TimeHeld: int (seconds)
├── Market
│   ├── GeneratedAt: string (ISO 8601)
│   ├── ChainID: int (8453 = Base)
│   ├── Tokens[]
│   │   ├── Address: string
│   │   ├── Symbol: string
│   │   ├── Name: string
│   │   ├── Description: string
│   │   ├── Decimals: int
│   │   ├── PoolAddress: string
│   │   ├── PoolID: string
│   │   ├── Token0IsWETH: bool
│   │   ├── CreatedTimestamp: int
│   │   ├── PriceInEth: float
│   │   ├── Metrics
│   │   │   ├── PctChange1m: float
│   │   │   ├── PctChange5m: float
│   │   │   ├── PctChange1h: float
│   │   │   ├── PctChange6h: float
│   │   │   ├── PctChange24h: float
│   │   │   ├── PctChange7d: float|null
│   │   │   ├── PctChangeAll: float
│   │   │   ├── VolumeInEth5m: float
│   │   │   ├── VolumeInEth1h: float
│   │   │   ├── VolumeInEth6h: float
│   │   │   ├── VolumeInEth24h: float
│   │   │   ├── VolumeInEth7d: float
│   │   │   ├── VolumeInEthAll: float
│   │   │   ├── NetFlowInEth5m: float
│   │   │   ├── NetFlowInEth1h: float
│   │   │   ├── HolderCount: int
│   │   │   ├── HoldersChange1h: int
│   │   │   ├── UniqueTraders5m: int
│   │   │   └── Top20HolderPct: float
│   │   └── Pool
│   │       ├── SqrtPriceX96: int (large)
│   │       ├── CurrentTick: int
│   │       ├── Fee: int
│   │       ├── TickSpacing: int
│   │       ├── Ticks[]
│   │       │   ├── Tick: int
│   │       │   └── LiquidityNet: int (large, can be negative)
│   │       └── HookFeeBps: int
│   ├── Reaps: (observed in payload, details TBD)
│   └── EthPriceUsd: float
├── AllowedTools: string[] (e.g. ["buy_token", "sell_token", "record_observation"])
└── Memories[]
    ├── tool: string
    ├── args: object (same shape as tool_args)
    └── timestamp: string (ISO 8601)
```

#### Schema Summary -- `llm_request_payload`

```
llm_request_payload
├── version: int (4)
├── model: string ("qwen/Qwen3/Qwen3-235B-A22B-Thinking-2507-FP8")
├── options: object (same as Agent.Options -- vault trading params)
└── llm_input
    ├── model: string
    ├── max_completion_tokens: int (15000)
    ├── max_tokens: int (20000)
    ├── temperature: float (0.8)
    ├── top_p: float (0.95)
    ├── parallel_tool_calls: bool (false)
    ├── metadata: object
    ├── messages[]
    │   ├── [0] role: "system", content: string (1426 chars -- system prompt)
    │   └── [1] role: "user", content: string (27269 chars -- full context dump)
    └── tools[]
        ├── buy_token: { token, spend_pct, content, strategy }
        ├── sell_token: { token, spend_pct, content, strategy }
        └── record_observation: { content, strategy }
```

#### Schema Summary -- `llm_completion_payload`

```
llm_completion_payload
├── id: string
├── object: "chat.completion"
├── created: int (unix timestamp)
├── model: string ("qwen3-235b-a22b-thinking-fp8")
├── choices[]
│   └── [0]
│       └── message
│           ├── role: "assistant"
│           ├── content: string (can be empty when tool_calls present)
│           ├── reasoning_content: string (chain-of-thought, ~2000 chars)
│           └── tool_calls[]
│               └── function
│                   ├── name: string
│                   └── arguments: string (JSON-encoded)
├── usage
│   ├── prompt_tokens: int
│   ├── total_tokens: int
│   ├── completion_tokens: int
│   ├── prompt_tokens_details: null
│   └── reasoning_tokens: int
└── metadata
    └── weight_version: string
```

**Key observations:**
- The full-log is ~52KB. The bulk is in `snapshot.Market.Tokens` (all token data with Uniswap V4 pool state) and the `llm_input.messages[1]` user message (~27K chars of context).
- `reasoning_content` contains the model's chain-of-thought reasoning (~2107 chars in this sample). This is the "thinking" output from Qwen3's reasoning mode.
- `Memories` is an array of the last ~20 actions taken by this vault, providing recent history context.
- The `llm_input.messages` array has exactly 2 messages: a system prompt and a user message. The user message contains the full state dump.
- Three tools are available: `buy_token`, `sell_token`, `record_observation`.

---

## 4. Swaps

### Request

```
GET https://api.terminal.markets/api/v1/swaps?vaultAddress=0x496D1400BB933fF4683dBe42445D5c89cFc10F95&limit=3&order=desc
```

### HTTP Status: 200

### Response

```json
{
  "totalCount": 58,
  "hasMoreItems": true,
  "items": [
    {
      "cursor": "42716536:688",
      "blockNumber": 42716536,
      "transactionHash": "0x19c3fefbc4ed41992a656bcb714e6bb43a11655fb983bf74088721975df29b4a",
      "logIndex": 688,
      "timestamp": 1772222419,
      "poolId": "0xa64c52eea6bd4b9e92253a2d8ae06eeeefe79ed417d56d5df5099d268637572b",
      "tokenAddress": "0x5d065DAF8667A2Da5124bf10F9F3B72fE09fBC6F",
      "tokenName": "LMAO",
      "tokenSymbol": "LMAO",
      "vaultAddress": "0x496D1400BB933fF4683dBe42445D5c89cFc10F95",
      "isReapTwap": false,
      "side": "sell",
      "tokenAmount": "3919535337482267708375257",
      "ethAmount": "561927974337734110",
      "ethPriceUsd": "1921.30000000",
      "effectivePriceEth": "0.000000143365967125",
      "effectivePriceUsd": "0.000275449032636760",
      "logId": 432750,
      "strategyId": "12"
    },
    {
      "cursor": "42716089:775",
      "blockNumber": 42716089,
      "transactionHash": "0x94ae595341fd34921dfab9a0903ed0a4acd775488000fa83db53c31b6ac214c2",
      "logIndex": 775,
      "timestamp": 1772221525,
      "poolId": "0xa64c52eea6bd4b9e92253a2d8ae06eeeefe79ed417d56d5df5099d268637572b",
      "tokenAddress": "0x5d065DAF8667A2Da5124bf10F9F3B72fE09fBC6F",
      "tokenName": "LMAO",
      "tokenSymbol": "LMAO",
      "vaultAddress": "0x496D1400BB933fF4683dBe42445D5c89cFc10F95",
      "isReapTwap": false,
      "side": "sell",
      "tokenAmount": "4125826671033966008816060",
      "ethAmount": "694297282003457071",
      "ethPriceUsd": "1915.28650000",
      "effectivePriceEth": "0.000000168280768283",
      "effectivePriceUsd": "0.000322305883701765",
      "logId": 428426,
      "strategyId": "12"
    },
    {
      "cursor": "42715814:203",
      "blockNumber": 42715814,
      "transactionHash": "0x7c318740d7eb4561e9778ee45b1342033339dac5a4f8ffe4847089360a045258",
      "logIndex": 203,
      "timestamp": 1772220975,
      "poolId": "0x77ec461b4e226e36bdc08ed882c47c966e44a2af2d912d77f090b709a246339b",
      "tokenAddress": "0x781BAe1c8E0DbB4950845A6d776d94C33b326D8a",
      "tokenName": "Hotdogz",
      "tokenSymbol": "HOTDOGZ",
      "vaultAddress": "0x496D1400BB933fF4683dBe42445D5c89cFc10F95",
      "isReapTwap": false,
      "side": "buy",
      "tokenAmount": "970374736266389602558946",
      "ethAmount": "256161874783136385",
      "ethPriceUsd": "1921.55000000",
      "effectivePriceEth": "0.000000263982423706",
      "effectivePriceUsd": "0.000507255426273184",
      "logId": 426132
    }
  ]
}
```

### Schema Summary

| Field | Type | Notes |
|-------|------|-------|
| `totalCount` | int | Total swaps for this vault |
| `hasMoreItems` | bool | |
| `items[].cursor` | string | Composite: `{blockNumber}:{logIndex}` |
| `items[].blockNumber` | int | Base L2 block |
| `items[].transactionHash` | string | 0x-prefixed tx hash |
| `items[].logIndex` | int | Event log index within the transaction |
| `items[].timestamp` | int | Unix timestamp |
| `items[].poolId` | string | Uniswap V4 pool ID |
| `items[].tokenAddress` | string | Token contract address |
| `items[].tokenName` | string | |
| `items[].tokenSymbol` | string | |
| `items[].vaultAddress` | string | |
| `items[].isReapTwap` | bool | Whether this was an automated reap TWAP trade |
| `items[].side` | string | "buy" or "sell" |
| `items[].tokenAmount` | string | Raw token amount (big integer string) |
| `items[].ethAmount` | string | Wei amount (big integer string) |
| `items[].ethPriceUsd` | string | ETH/USD price at time of swap (decimal string) |
| `items[].effectivePriceEth` | string | Token price in ETH (decimal string) |
| `items[].effectivePriceUsd` | string | Token price in USD (decimal string) |
| `items[].logId` | int | Links to inference log ID -- **the join key** |
| `items[].strategyId` | string | Optional, present when strategy-driven |

**Key observations:**
- `logId` is the critical join key linking swaps back to the inference that triggered them.
- `strategyId` is only present on strategy-driven trades, absent on autonomous buys (see 3rd swap above).
- All monetary values are strings to preserve precision. ETH amounts are wei (18 decimals). Token amounts are raw (18 decimals).
- Cursor format is `{blockNumber}:{logIndex}` which provides a natural ordering.

---

## 5. Candles

### Request

```
GET https://api.terminal.markets/api/v1/candles/0x5d065DAF8667A2Da5124bf10F9F3B72fE09fBC6F?timeframe=1h&countback=5&to=1772224697
```

**Note:** The `limit` param does not exist. Required params are `timeframe`, `to` (unix timestamp), and either `from` (unix timestamp) or `countback` (number of candles).

### HTTP Status: 200

### Response

```json
{
  "s": "ok",
  "t": [1772208000, 1772211600, 1772215200, 1772218800, 1772222400],
  "o": [1.78050267204e-7, 1.47889048432e-7, 1.41064054021e-7, 2.02764372273e-7, 1.49265561887e-7],
  "h": [1.78473065341e-7, 1.47889048432e-7, 2.13493274952e-7, 2.02764372273e-7, 1.49266765698e-7],
  "l": [1.47857205936e-7, 1.41064054021e-7, 1.40150256297e-7, 1.49265561887e-7, 1.42212762212e-7],
  "c": [1.47889048432e-7, 1.41064054021e-7, 2.02764372273e-7, 1.49265561887e-7, 1.43577979129e-7],
  "v": [3.947800804330296, 0.8179947527671741, 12.450752470755027, 9.075821375009777, 1.1178466580310569]
}
```

### Schema Summary

| Field | Type | Notes |
|-------|------|-------|
| `s` | string | Status: "ok" or "error" |
| `t` | int[] | Unix timestamps (candle open time) |
| `o` | float[] | Open prices (in ETH) |
| `h` | float[] | High prices (in ETH) |
| `l` | float[] | Low prices (in ETH) |
| `c` | float[] | Close prices (in ETH) |
| `v` | float[] | Volume (in ETH) |

**Required parameters:**
- `timeframe`: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`
- `to`: Unix timestamp (end of range)
- `from` OR `countback`: Start of range (unix timestamp) or number of candles to look back

**Error responses observed:**
- Missing `to`: `{"errmsg":"to timestamp is required","s":"error"}` (HTTP 400)
- Missing `from`/`countback`: `{"errmsg":"either from or countback is required","s":"error"}` (HTTP 400)

**Key observations:**
- Uses TradingView-style OHLCV format with parallel arrays (not array of objects).
- All prices are in ETH (not USD). Multiply by ETH/USD price for USD values.
- Volumes are in ETH.
- Arrays are parallel -- index `i` across all arrays corresponds to the same candle.

---

## Cross-Endpoint Join Keys

| From | To | Join Key |
|------|----|----------|
| `swaps.logId` | `logs.id` / `full-log.id` | Inference log ID |
| `swaps.transactionHash` | `logs.transactionHash` | Transaction hash |
| `logs.strategyId` | `strategies.strategyId` | Strategy ID (scoped to vault) |
| `swaps.tokenAddress` | `candles/{tokenAddress}` | Token contract address |
| `swaps.vaultAddress` | `vault?vaultAddress=` | Vault address |
| `leaderboard.vaultAddress` | all other endpoints | Vault address |

---

## API Quirks and Notes

1. **Base URL discrepancy:** The frontend is at `terminal.markets` but the API is at `api.terminal.markets/api/v1`. Direct curl to `terminal.markets` hits a Vercel security checkpoint.

2. **Pagination:** All paginated endpoints use cursor-based pagination with `hasMoreItems` boolean. The leaderboard uses `cursor` param from the last item's `cursor` field. Logs and swaps use the same pattern.

3. **String vs numeric types:** ETH amounts and token amounts are always strings (big integers). USD values on the leaderboard are floats. Price fields on swaps are decimal strings. The API is inconsistent -- be prepared to parse both.

4. **Candles use TradingView format:** Parallel arrays, not objects. Requires `to` + (`from` or `countback`). Does not accept `limit`.

5. **Full log size:** A single full-log response is ~50KB. At scale (100K+ inferences), this is 5-50GB of raw payloads. The bulk of the size comes from `snapshot.Market.Tokens` (full market state with pool ticks) and the LLM user message.

6. **Model identification:** The model is `qwen/Qwen3/Qwen3-235B-A22B-Thinking-2507-FP8` (Qwen3 235B MoE with FP8 quantization and thinking/reasoning enabled). The completion payload confirms reasoning tokens are generated (`reasoning_content` field).

7. **Strategies endpoint:** Returns a flat array (not paginated, no cursor). Filter with `?activeOnly=true` if only enabled strategies are needed.

8. **Leaderboard requires sortBy:** The `sortBy` parameter is required. Omitting it may cause errors or unexpected behavior.
