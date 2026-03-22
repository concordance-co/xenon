CREATE TABLE IF NOT EXISTS synthetic_market_examples_v0 (
    log_id               INT PRIMARY KEY,
    phase_name           TEXT NOT NULL,
    example_id           TEXT NOT NULL,
    family               TEXT NOT NULL,
    family_variant       TEXT NOT NULL,
    context_variant      TEXT NOT NULL,
    system_prompt        TEXT NOT NULL,
    user_prompt          TEXT NOT NULL,
    prompt_messages_json JSONB NOT NULL,
    labels_json          JSONB NOT NULL,
    num_assets           INT NOT NULL,
    best_asset           TEXT,
    buy_any              INT NOT NULL,
    observe_vs_act       TEXT NOT NULL,
    capture_enabled      BOOLEAN NOT NULL DEFAULT TRUE,
    selection_rank       INT NOT NULL,
    capture_priority     DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_synthetic_market_examples_phase_ctx
    ON synthetic_market_examples_v0 (phase_name, context_variant, selection_rank, log_id);

CREATE INDEX IF NOT EXISTS idx_synthetic_market_examples_family
    ON synthetic_market_examples_v0 (phase_name, family, family_variant);

CREATE TABLE IF NOT EXISTS synthetic_market_assets_v0 (
    log_id                           INT NOT NULL REFERENCES synthetic_market_examples_v0(log_id) ON DELETE CASCADE,
    phase_name                       TEXT NOT NULL,
    example_id                       TEXT NOT NULL,
    family                           TEXT NOT NULL,
    family_variant                   TEXT NOT NULL,
    context_variant                  TEXT NOT NULL,
    row_index                        INT NOT NULL,
    symbol                           TEXT NOT NULL,
    profile_id                       TEXT,
    archetype                        TEXT NOT NULL,
    pct_5m                           DOUBLE PRECISION NOT NULL,
    pct_1h                           DOUBLE PRECISION NOT NULL,
    net_flow_5m                      DOUBLE PRECISION NOT NULL,
    vol_5m                           DOUBLE PRECISION NOT NULL,
    vol_1h                           DOUBLE PRECISION NOT NULL,
    unique_traders_5m                INT NOT NULL,
    top20_holder_pct                 DOUBLE PRECISION NOT NULL,
    age_bucket                       TEXT NOT NULL,
    momentum_score                   DOUBLE PRECISION NOT NULL,
    participation_score              DOUBLE PRECISION NOT NULL,
    flow_score                       DOUBLE PRECISION NOT NULL,
    concentration_penalty            DOUBLE PRECISION NOT NULL,
    riskiness_score                  DOUBLE PRECISION NOT NULL,
    attractiveness_score             DOUBLE PRECISION NOT NULL,
    risk_adjusted_score              DOUBLE PRECISION NOT NULL,
    edge_after_fee_score             DOUBLE PRECISION NOT NULL,
    edge_gt_fee                      DOUBLE PRECISION NOT NULL,
    attractiveness_rank              INT NOT NULL,
    risk_adjusted_rank               INT NOT NULL,
    is_best_asset                    INT NOT NULL,
    buyable_if_unconstrained         INT NOT NULL,
    acceptable_under_risk_setting    INT NOT NULL,
    PRIMARY KEY (log_id, row_index)
);

CREATE INDEX IF NOT EXISTS idx_synthetic_market_assets_phase
    ON synthetic_market_assets_v0 (phase_name, family, context_variant, log_id, row_index);

CREATE INDEX IF NOT EXISTS idx_synthetic_market_assets_symbol
    ON synthetic_market_assets_v0 (symbol, archetype, phase_name);

ALTER TABLE synthetic_market_assets_v0
    ADD COLUMN IF NOT EXISTS profile_id TEXT;

CREATE TABLE IF NOT EXISTS synthetic_market_pairs_v0 (
    log_id                           INT NOT NULL REFERENCES synthetic_market_examples_v0(log_id) ON DELETE CASCADE,
    phase_name                       TEXT NOT NULL,
    example_id                       TEXT NOT NULL,
    family                           TEXT NOT NULL,
    family_variant                   TEXT NOT NULL,
    context_variant                  TEXT NOT NULL,
    asset_a                          TEXT NOT NULL,
    asset_b                          TEXT NOT NULL,
    a_beats_b_on_attractiveness      INT NOT NULL,
    a_beats_b_on_risk_adjusted       INT NOT NULL,
    delta_pct_5m                     DOUBLE PRECISION NOT NULL,
    delta_pct_1h                     DOUBLE PRECISION NOT NULL,
    delta_net_flow_5m                DOUBLE PRECISION NOT NULL,
    delta_vol_5m                     DOUBLE PRECISION NOT NULL,
    delta_unique_traders_5m          INT NOT NULL,
    delta_top20_holder_pct           DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (log_id, asset_a, asset_b)
);

CREATE INDEX IF NOT EXISTS idx_synthetic_market_pairs_phase
    ON synthetic_market_pairs_v0 (phase_name, family, context_variant, log_id);

CREATE OR REPLACE VIEW synthetic_market_phase1_capture_v0 AS
SELECT
    log_id,
    prompt_messages_json,
    phase_name,
    example_id,
    family,
    family_variant,
    context_variant,
    selection_rank,
    capture_priority,
    created_at
FROM synthetic_market_examples_v0
WHERE phase_name = 'phase1'
  AND context_variant = 'market_only'
  AND capture_enabled = TRUE;

CREATE OR REPLACE VIEW synthetic_market_phase1_context_ladder_v0 AS
SELECT
    log_id,
    prompt_messages_json,
    phase_name,
    example_id,
    family,
    family_variant,
    context_variant,
    selection_rank,
    capture_priority,
    created_at
FROM synthetic_market_examples_v0
WHERE phase_name = 'phase1'
  AND capture_enabled = TRUE;
