from __future__ import annotations

"""Neon Postgres database module for Xenon.

Provides connection helpers (sync + async) and DDL for all tables.
No SQLite. No ORM. Just psycopg against Neon Postgres.
"""

import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

# ---------------------------------------------------------------------------
# Environment / DSN helpers
# ---------------------------------------------------------------------------

_ENV_VAR = "XENON_NEON_DATABASE_URL"


def load_dotenv_if_present() -> None:
    """Load variables from a .env file in the project root, if it exists.

    Only sets variables that are not already in the environment.
    No third-party dependency required.
    """
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for env_path in candidates:
        if env_path.is_file():
            with env_path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            break  # stop after the first .env found


def require_neon_dsn() -> str:
    """Return the Neon DSN or raise with a clear message."""
    load_dotenv_if_present()
    dsn = os.environ.get(_ENV_VAR)
    if not dsn:
        raise RuntimeError(
            f"Environment variable {_ENV_VAR} is not set. "
            "Set it directly or add it to a .env file."
        )
    return dsn


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def connect_neon(*, autocommit: bool = True) -> psycopg.Connection:
    """Return a sync psycopg Connection with dict_row cursors."""
    dsn = require_neon_dsn()
    return psycopg.connect(dsn, autocommit=autocommit, row_factory=dict_row)


async def connect_neon_async(
    *, autocommit: bool = True,
) -> psycopg.AsyncConnection:
    """Return an async psycopg AsyncConnection with dict_row cursors."""
    dsn = require_neon_dsn()
    return await psycopg.AsyncConnection.connect(
        dsn, autocommit=autocommit, row_factory=dict_row,
    )


# ---------------------------------------------------------------------------
# Table DDL
# ---------------------------------------------------------------------------

DDL_VAULTS = """\
CREATE TABLE IF NOT EXISTS vaults (
    vault_address       TEXT PRIMARY KEY,
    owner_address       TEXT,
    nft_id              TEXT,
    nft_name            TEXT,
    persona_json        TEXT,
    trade_size          INT,
    trading_activity    INT,
    holding_style       INT,
    diversification     INT,
    asset_risk_preference INT,
    max_trade_amount    TEXT,
    slippage_bps        TEXT,
    paused              BOOLEAN,
    state               TEXT,
    leaderboard_rank    INT,
    total_pnl_usd      DOUBLE PRECISION,
    realized_pnl_usd   DOUBLE PRECISION,
    unrealized_pnl_usd DOUBLE PRECISION,
    created_block       BIGINT,
    updated_block       BIGINT,
    fetched_at          TEXT NOT NULL
);
"""

DDL_STRATEGIES = """\
CREATE TABLE IF NOT EXISTS strategies (
    vault_address       TEXT NOT NULL,
    strategy_id         TEXT NOT NULL,
    vault_owner_address TEXT,
    content             TEXT,
    expiry              BIGINT,
    enabled             BOOLEAN,
    strategy_priority   TEXT,
    created_block       BIGINT,
    updated_block       BIGINT,
    fetched_at          TEXT NOT NULL,
    PRIMARY KEY (vault_address, strategy_id),
    FOREIGN KEY (vault_address) REFERENCES vaults(vault_address)
);
"""

DDL_INFERENCE_LOGS = """\
CREATE TABLE IF NOT EXISTS inference_logs (
    id                   BIGINT PRIMARY KEY,
    cursor               TEXT,
    vault_address        TEXT NOT NULL,
    request_id           TEXT,
    execution_key        TEXT,
    tool                 TEXT,
    tool_args_json       TEXT,
    strategy_id          TEXT,
    status               TEXT,
    inference_duration_ms INT,
    error                TEXT,
    transaction_hash     TEXT,
    created_at           TEXT,
    completed_at         TEXT,
    fetched_at           TEXT NOT NULL,
    FOREIGN KEY (vault_address) REFERENCES vaults(vault_address)
);
"""

DDL_INFERENCE_LOGS_IDX = """\
CREATE INDEX IF NOT EXISTS idx_inference_logs_vault
    ON inference_logs(vault_address, id);
"""

DDL_FULL_LOGS = """\
CREATE TABLE IF NOT EXISTS full_logs (
    log_id              BIGINT PRIMARY KEY,
    vault_address       TEXT,
    prompt_text         TEXT,
    completion_text     TEXT,
    reasoning_content   TEXT,
    tool_calls_json     TEXT,
    llm_model           TEXT,
    prompt_tokens       INT,
    completion_tokens   INT,
    reasoning_tokens    INT,
    total_tokens        INT,
    parse_error         TEXT,
    raw_payload         JSONB,
    fetched_at          TEXT NOT NULL,
    FOREIGN KEY (log_id) REFERENCES inference_logs(id)
);
"""

DDL_FULL_LOGS_GIN_IDX = """\
CREATE INDEX IF NOT EXISTS idx_full_logs_raw_payload_gin
    ON full_logs USING GIN (raw_payload jsonb_path_ops);
"""

DDL_SWAPS = """\
CREATE TABLE IF NOT EXISTS swaps (
    transaction_hash    TEXT NOT NULL,
    block_number        BIGINT NOT NULL,
    log_index           BIGINT NOT NULL,
    timestamp           BIGINT,
    pool_id             TEXT,
    token_address       TEXT,
    token_name          TEXT,
    token_symbol        TEXT,
    vault_address       TEXT NOT NULL,
    is_reap_twap        BOOLEAN,
    side                TEXT,
    token_amount        TEXT,
    eth_amount          TEXT,
    eth_price_usd       TEXT,
    effective_price_eth TEXT,
    effective_price_usd TEXT,
    log_id              BIGINT,
    strategy_id         TEXT,
    fetched_at          TEXT NOT NULL,
    PRIMARY KEY (transaction_hash, log_index),
    FOREIGN KEY (vault_address) REFERENCES vaults(vault_address)
);
"""

DDL_SWAPS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_swaps_log_id ON swaps(log_id);",
    "CREATE INDEX IF NOT EXISTS idx_swaps_vault ON swaps(vault_address, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_swaps_token_timestamp ON swaps(token_address, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_swaps_vault_token ON swaps(vault_address, token_address);",
]

DDL_FULL_LOGS_IDX_VAULT = """\
CREATE INDEX IF NOT EXISTS idx_full_logs_vault
    ON full_logs(vault_address);
"""

DDL_INGEST_CURSORS = """\
CREATE TABLE IF NOT EXISTS ingest_cursors (
    vault_address TEXT NOT NULL,
    resource      TEXT NOT NULL,
    last_cursor   TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (vault_address, resource)
);
"""

DDL_DEFERRED_LOGS = """\
CREATE TABLE IF NOT EXISTS deferred_logs (
    log_id          BIGINT PRIMARY KEY,
    vault_address   TEXT NOT NULL,
    last_status     INT NOT NULL,
    attempts        INT NOT NULL DEFAULT 1,
    first_seen_at   TEXT NOT NULL,
    last_attempted_at TEXT NOT NULL
);
"""

DDL_INTERP_EXAMPLES = """\
CREATE TABLE IF NOT EXISTS interp_examples_v0 (
    example_id           TEXT PRIMARY KEY,
    log_id               BIGINT NOT NULL,
    vault_address        TEXT NOT NULL,
    created_at           TEXT,
    strategy_id          TEXT,
    transaction_hash     TEXT,
    is_trade             BOOLEAN NOT NULL,
    prompt_messages_json TEXT,
    system_text          TEXT,
    user_text            TEXT,
    tools_available_json TEXT,
    market_snapshot_json TEXT,
    portfolio_snapshot_json TEXT,
    strategy_snapshot_json TEXT,
    config_snapshot_json TEXT,
    memory_snapshot_json TEXT,
    model_source         TEXT,
    assistant_content    TEXT,
    reasoning_content    TEXT,
    tool_calls_json      TEXT,
    action_name          TEXT,
    decision_type        TEXT NOT NULL,
    trade_side           TEXT,
    asset                TEXT,
    size                 TEXT,
    observation_text     TEXT,
    joined_swap          BOOLEAN NOT NULL,
    swap_side            TEXT,
    swap_token_address   TEXT,
    swap_token_symbol    TEXT,
    swap_price_usd       DOUBLE PRECISION,
    pnl_1h_pct           DOUBLE PRECISION,
    pnl_4h_pct           DOUBLE PRECISION,
    pnl_1d_pct           DOUBLE PRECISION,
    was_profitable_1h    BOOLEAN,
    entry_price_usd      DOUBLE PRECISION,
    entry_price_eth      DOUBLE PRECISION,
    vault_trade_size     INT,
    vault_trading_activity INT,
    vault_holding_style  INT,
    vault_diversification INT,
    vault_risk_preference INT,
    parse_ok             BOOLEAN NOT NULL,
    parse_error          TEXT,
    has_messages          BOOLEAN NOT NULL,
    has_tools             BOOLEAN NOT NULL,
    has_market            BOOLEAN NOT NULL,
    has_portfolio         BOOLEAN NOT NULL,
    has_strategy          BOOLEAN NOT NULL,
    has_config            BOOLEAN NOT NULL,
    has_memory            BOOLEAN NOT NULL,
    context_complete      BOOLEAN NOT NULL,
    missing_blocks_json   TEXT,
    label_quality         TEXT NOT NULL,
    label_confidence      TEXT NOT NULL,
    ingest_version        TEXT,
    transform_version     TEXT NOT NULL,
    built_at              TEXT NOT NULL
);
"""

DDL_INTERP_EXAMPLES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_interp_examples_v0_decision ON interp_examples_v0(decision_type, context_complete, parse_ok, created_at);",
    "CREATE INDEX IF NOT EXISTS idx_interp_examples_v0_vault_log ON interp_examples_v0(vault_address, log_id);",
]

DDL_TRADE_OUTCOMES = """\
CREATE TABLE IF NOT EXISTS trade_outcomes (
    log_id                BIGINT PRIMARY KEY,
    transaction_hash      TEXT NOT NULL,
    side                  TEXT NOT NULL,
    token_address         TEXT NOT NULL,
    entry_price_eth       DOUBLE PRECISION,
    entry_price_usd       DOUBLE PRECISION,
    eth_price_usd_at_entry DOUBLE PRECISION,
    price_1h_eth          DOUBLE PRECISION,
    price_4h_eth          DOUBLE PRECISION,
    price_1d_eth          DOUBLE PRECISION,
    price_1h_usd          DOUBLE PRECISION,
    price_4h_usd          DOUBLE PRECISION,
    price_1d_usd          DOUBLE PRECISION,
    pnl_1h_pct            DOUBLE PRECISION,
    pnl_4h_pct            DOUBLE PRECISION,
    pnl_1d_pct            DOUBLE PRECISION,
    was_profitable_1h     BOOLEAN,
    candle_data_json      TEXT,
    computed_at           TEXT NOT NULL
);
"""

DDL_PREP_TARGET_SPECS = """\
CREATE TABLE IF NOT EXISTS prep_target_specs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    spec_json   JSONB NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""

DDL_WORKFLOW_SPECS = """\
CREATE TABLE IF NOT EXISTS workflow_specs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    version     INT NOT NULL DEFAULT 1,
    spec_json   JSONB NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""

DDL_WORKFLOW_RUNS = """\
CREATE TABLE IF NOT EXISTS workflow_runs (
    id                 TEXT PRIMARY KEY,
    spec_id            TEXT NOT NULL,
    spec_version       INT NOT NULL,
    run_type           TEXT NOT NULL,
    status             TEXT NOT NULL,
    source             TEXT NOT NULL,
    spec_snapshot_json JSONB NOT NULL,
    config_json        JSONB NOT NULL,
    result_json        JSONB NOT NULL,
    error_text         TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    completed_at       TEXT
);
"""

DDL_DATASET_PUBLICATIONS = """\
CREATE TABLE IF NOT EXISTS dataset_publications (
    id               TEXT PRIMARY KEY,
    spec_id          TEXT NOT NULL,
    spec_version     INT NOT NULL,
    run_id           TEXT NOT NULL,
    relation_name    TEXT NOT NULL,
    publish_mode     TEXT NOT NULL,
    row_count        BIGINT,
    publication_json JSONB NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
"""

DDL_WORKFLOW_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_workflow_specs_updated_at ON workflow_specs(updated_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_workflow_runs_spec_created ON workflow_runs(spec_id, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_workflow_runs_type_status ON workflow_runs(run_type, status, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_dataset_publications_spec_created ON dataset_publications(spec_id, created_at DESC);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_publications_relation_name ON dataset_publications(relation_name);",
]

# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

TABLE_DDLS: list[str] = [
    DDL_VAULTS,
    DDL_STRATEGIES,
    DDL_INFERENCE_LOGS,
    DDL_FULL_LOGS,
    DDL_SWAPS,
    DDL_INGEST_CURSORS,
    DDL_DEFERRED_LOGS,
    DDL_TRADE_OUTCOMES,
    DDL_INTERP_EXAMPLES,
    DDL_PREP_TARGET_SPECS,
    DDL_WORKFLOW_SPECS,
    DDL_WORKFLOW_RUNS,
    DDL_DATASET_PUBLICATIONS,
]

INDEX_DDLS: list[str] = [
    DDL_INFERENCE_LOGS_IDX,
    DDL_FULL_LOGS_GIN_IDX,
    DDL_FULL_LOGS_IDX_VAULT,
    *DDL_SWAPS_INDEXES,
    *DDL_INTERP_EXAMPLES_INDEXES,
    *DDL_WORKFLOW_INDEXES,
]

_ALL_DDL: list[str] = TABLE_DDLS + INDEX_DDLS

# Columns to migrate from INT to BOOLEAN on existing databases.
_BOOLEAN_MIGRATIONS: list[tuple[str, list[str]]] = [
    ("vaults", ["paused"]),
    ("strategies", ["enabled"]),
    ("swaps", ["is_reap_twap"]),
    ("trade_outcomes", ["was_profitable_1h"]),
    (
        "interp_examples_v0",
        [
            "is_trade", "joined_swap", "parse_ok", "was_profitable_1h",
            "has_messages", "has_tools", "has_market", "has_portfolio",
            "has_strategy", "has_config", "has_memory", "context_complete",
        ],
    ),
]

# Columns to migrate from INT to BIGINT on existing databases.
_BIGINT_MIGRATIONS: list[tuple[str, list[str]]] = [
    ("vaults", ["created_block", "updated_block"]),
    ("strategies", ["expiry", "created_block", "updated_block"]),
]

# Dead columns to drop from existing databases.
_DEAD_COLUMNS: list[tuple[str, list[str]]] = [
    ("full_logs", ["payload_path", "payload_sha256", "payload_size_bytes"]),
]


def _migrate_existing_schema(conn: psycopg.Connection) -> None:
    """Migrate existing databases: drop dead columns, INT -> BOOLEAN.

    Runs all ALTER statements inside a single transaction to avoid
    split-brain schema on partial failure.
    """
    was_autocommit = conn.autocommit
    if was_autocommit:
        conn.autocommit = False

    try:
        with conn.cursor() as cur:
            # Drop dead columns
            for table, columns in _DEAD_COLUMNS:
                for col in columns:
                    row = cur.execute(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_schema = 'public' AND table_name = %s AND column_name = %s",
                        [table, col],
                    ).fetchone()
                    if row:
                        cur.execute(f'ALTER TABLE "{table}" DROP COLUMN IF EXISTS "{col}"')

            # Migrate INT -> BIGINT
            for table, columns in _BIGINT_MIGRATIONS:
                for col in columns:
                    row = cur.execute(
                        "SELECT data_type FROM information_schema.columns "
                        "WHERE table_schema = 'public' AND table_name = %s AND column_name = %s",
                        [table, col],
                    ).fetchone()
                    if row and row["data_type"] == "integer":
                        cur.execute(
                            f'ALTER TABLE "{table}" ALTER COLUMN "{col}" TYPE BIGINT'
                        )

            # Migrate INT -> BOOLEAN
            for table, columns in _BOOLEAN_MIGRATIONS:
                for col in columns:
                    row = cur.execute(
                        "SELECT data_type FROM information_schema.columns "
                        "WHERE table_schema = 'public' AND table_name = %s AND column_name = %s",
                        [table, col],
                    ).fetchone()
                    if row and row["data_type"] in ("integer", "bigint", "smallint"):
                        cur.execute(
                            f'ALTER TABLE "{table}" ALTER COLUMN "{col}" '
                            f'TYPE BOOLEAN USING "{col}"::int::boolean'
                        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if was_autocommit:
            conn.autocommit = True


def cleanup_stale_cursors(conn: psycopg.Connection) -> int:
    """Delete ingest_cursors for vault_addresses no longer in the vaults table."""
    row = conn.execute(
        "DELETE FROM ingest_cursors "
        "WHERE vault_address NOT IN (SELECT vault_address FROM vaults) "
        "RETURNING vault_address"
    ).fetchall()
    count = len(row)
    if count:
        print(f"Cleaned up {count} stale ingest cursor(s)")
    return count


def reset_cursors(conn: psycopg.Connection) -> int:
    """Wipe all ingest_cursors.

    Safe because ingest upserts handle duplicates and full_log fetches
    skip logs already in the DB via fetch_existing_full_log_ids.
    """
    row = conn.execute("DELETE FROM ingest_cursors RETURNING vault_address").fetchall()
    count = len(row)
    if count:
        print(f"Cleared {count} ingest cursor(s) — next run will re-paginate from start")
    else:
        print("No ingest cursors to clear")
    return count


def ensure_schema(conn: psycopg.Connection) -> None:
    """Create all tables and indexes if they do not already exist.

    Also runs lightweight migrations (dead column drops, INT->BOOLEAN)
    for existing databases.

    Expects a sync psycopg connection. Commits after all DDL statements
    have been executed (safe to call with autocommit=True as well).
    """
    with conn.cursor() as cur:
        for ddl in _ALL_DDL:
            cur.execute(ddl)
    if not conn.autocommit:
        conn.commit()

    _migrate_existing_schema(conn)
