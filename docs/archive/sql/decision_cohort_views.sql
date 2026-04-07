DROP VIEW IF EXISTS decision_trade_candidates_v1;
DROP VIEW IF EXISTS decision_sell_candidates_v1;
DROP VIEW IF EXISTS decision_blocked_observe_candidates_v1;
DROP VIEW IF EXISTS decision_policy_tension_candidates_v1;
DROP MATERIALIZED VIEW IF EXISTS decision_capture_priority_v1;
DROP VIEW IF EXISTS decision_capture_base_v1;
DROP MATERIALIZED VIEW IF EXISTS decision_capture_base_mv_v1;


CREATE MATERIALIZED VIEW decision_capture_base_mv_v1 AS
WITH strategy_summary AS (
    SELECT
        ie.log_id,
        COUNT(*) FILTER (WHERE priority = 'high') AS high_strategy_count,
        COUNT(*) FILTER (
            WHERE priority = 'high'
              AND content ~ '(do not buy any token|do not buy any tokens|don''t buy any token|don''t buy any tokens|observe only|stay flat)'
        ) AS high_block_buy_all_count,
        COUNT(*) FILTER (
            WHERE priority = 'high'
              AND content ~ '(do not sell any token|do not sell any tokens|don''t sell any token|don''t sell any tokens|never sell any token|never sell any tokens)'
        ) AS high_block_sell_all_count,
        COUNT(*) FILTER (
            WHERE priority = 'high'
              AND content ~ '(do not buy|don''t buy|avoid|only buy|only trade|stay flat|observe only)'
        ) AS high_restriction_count,
        COUNT(*) FILTER (
            WHERE priority = 'high'
              AND content ~ '(do not sell|don''t sell|never sell|hold)'
        ) AS high_hold_rule_count,
        COUNT(*) FILTER (
            WHERE priority = 'high'
              AND content ~ '\m(if|when|once)\M.*\m(buy|sell|liquidate|exit|enter)\M'
        ) AS high_triggered_action_count,
        COUNT(*) FILTER (
            WHERE priority = 'high'
              AND content ~ '\m(buy|sell|liquidate|exit|enter|take profit)\M.*\m(now|immediately)\M'
        ) AS high_immediate_action_count
    FROM interp_examples_v0 ie
    JOIN full_logs fl ON fl.log_id = ie.log_id
    LEFT JOIN LATERAL (
        SELECT
            lower(COALESCE(value ->> 'strategyPriority', '')) AS priority,
            lower(COALESCE(value ->> 'content', '')) AS content
        FROM jsonb_array_elements(COALESCE(fl.raw_payload #> '{snapshot,Agent,Strategies}', '[]'::jsonb)) AS s(value)
    ) strat ON TRUE
    WHERE ie.label_quality IN ('high', 'medium')
      AND ie.prompt_messages_json IS NOT NULL
      AND ie.market_snapshot_json IS NOT NULL
    GROUP BY ie.log_id
),
held_tokens AS (
    SELECT
        ie.log_id,
        COUNT(*) FILTER (
            WHERE COALESCE(NULLIF(value ->> 'Balance', ''), '0')::double precision > 0
        ) AS held_token_count
    FROM interp_examples_v0 ie
    JOIN full_logs fl ON fl.log_id = ie.log_id
    LEFT JOIN LATERAL jsonb_array_elements(COALESCE(fl.raw_payload #> '{snapshot,Portfolio,Tokens}', '[]'::jsonb)) AS t(value) ON TRUE
    WHERE ie.label_quality IN ('high', 'medium')
      AND ie.prompt_messages_json IS NOT NULL
      AND ie.market_snapshot_json IS NOT NULL
    GROUP BY ie.log_id
)
SELECT
    ie.log_id,
    NULLIF(ps.created_at, '')::timestamptz AS created_at,
    ps.vault_address,
    ps.llm_model,
    ps.tool,
    ie.decision_type AS prep_decision_type,
    ie.trade_side AS prep_trade_side,
    ie.asset AS prep_asset,
    ie.label_quality,
    CASE
        WHEN ps.tool = 'buy_token' THEN 'buy'
        WHEN ps.tool = 'sell_token' THEN 'sell'
        ELSE NULL
    END AS trade_side,
    (ps.tool IN ('buy_token', 'sell_token')) AS is_trade,
    (ps.tool = 'record_observation') AS is_observe,
    UPPER(NULLIF(ps.trade_token, '')) AS target_asset,
    ps.trade_spend_pct,
    ps.trade_size,
    ps.trading_activity,
    ps.holding_style,
    ps.diversification,
    ps.risk_preference,
    COALESCE(ps.eth_balance, 0) AS eth_balance,
    (COALESCE(ps.eth_balance, 0) <= 0) AS zero_eth,
    ps.portfolio_token_count,
    COALESCE(ht.held_token_count, 0) AS held_token_count,
    COALESCE(array_length(ps.market_tokens, 1), 0) AS market_token_count,
    ps.strategy_count,
    ps.memory_depth,
    COALESCE(ss.high_strategy_count, 0) AS high_strategy_count,
    COALESCE(ss.high_restriction_count, 0) AS high_restriction_count,
    COALESCE(ss.high_hold_rule_count, 0) AS high_hold_rule_count,
    COALESCE(ss.high_immediate_action_count, 0) AS high_immediate_action_count,
    COALESCE(ss.high_triggered_action_count, 0) AS high_triggered_action_count,
    (COALESCE(ss.high_block_buy_all_count, 0) > 0) AS blocks_all_buys,
    (COALESCE(ss.high_block_sell_all_count, 0) > 0) AS blocks_all_sells,
    ('buy_token' = ANY(COALESCE(ps.allowed_tools, ARRAY[]::text[]))) AS allow_buy_tool,
    ('sell_token' = ANY(COALESCE(ps.allowed_tools, ARRAY[]::text[]))) AS allow_sell_tool,
    (
        ('buy_token' = ANY(COALESCE(ps.allowed_tools, ARRAY[]::text[])))
        AND COALESCE(ps.eth_balance, 0) > 0
        AND COALESCE(ss.high_block_buy_all_count, 0) = 0
    ) AS can_buy_any,
    (
        ('sell_token' = ANY(COALESCE(ps.allowed_tools, ARRAY[]::text[])))
        AND COALESCE(ht.held_token_count, 0) > 0
        AND COALESCE(ss.high_block_sell_all_count, 0) = 0
    ) AS can_sell_any,
    (
        ps.tool = 'record_observation'
        AND NOT (
            ('buy_token' = ANY(COALESCE(ps.allowed_tools, ARRAY[]::text[])))
            AND COALESCE(ps.eth_balance, 0) > 0
            AND COALESCE(ss.high_block_buy_all_count, 0) = 0
        )
        AND NOT (
            ('sell_token' = ANY(COALESCE(ps.allowed_tools, ARRAY[]::text[])))
            AND COALESCE(ht.held_token_count, 0) > 0
            AND COALESCE(ss.high_block_sell_all_count, 0) = 0
        )
    ) AS forced_observe,
    (
        ps.tool = 'record_observation'
        AND (
            COALESCE(ss.high_strategy_count, 0) > 0
            OR COALESCE(ss.high_block_buy_all_count, 0) > 0
            OR COALESCE(ss.high_block_sell_all_count, 0) > 0
            OR (
                COALESCE(ps.eth_balance, 0) <= 0
                AND NOT (
                    ('sell_token' = ANY(COALESCE(ps.allowed_tools, ARRAY[]::text[])))
                    AND COALESCE(ht.held_token_count, 0) > 0
                    AND COALESCE(ss.high_block_sell_all_count, 0) = 0
                )
            )
        )
    ) AS blocked_observe_candidate,
    (
        ps.tool = 'record_observation'
        AND COALESCE(ss.high_strategy_count, 0) = 0
        AND (
            (
                ('buy_token' = ANY(COALESCE(ps.allowed_tools, ARRAY[]::text[])))
                AND COALESCE(ps.eth_balance, 0) > 0
                AND COALESCE(ss.high_block_buy_all_count, 0) = 0
            )
            OR (
                ('sell_token' = ANY(COALESCE(ps.allowed_tools, ARRAY[]::text[])))
                AND COALESCE(ht.held_token_count, 0) > 0
                AND COALESCE(ss.high_block_sell_all_count, 0) = 0
            )
        )
        AND (
            COALESCE(ps.trading_activity, 0) >= 4
            OR COALESCE(ps.risk_preference, 0) IN (1, 5)
            OR COALESCE(ps.holding_style, 0) IN (1, 5)
            OR COALESCE(ps.diversification, 0) IN (1, 5)
        )
    ) AS policy_tension_candidate,
    (
        CASE WHEN COALESCE(ps.trade_size, 0) IN (1, 5) THEN 1 ELSE 0 END
        + CASE WHEN COALESCE(ps.trading_activity, 0) IN (1, 5) THEN 1 ELSE 0 END
        + CASE WHEN COALESCE(ps.holding_style, 0) IN (1, 5) THEN 1 ELSE 0 END
        + CASE WHEN COALESCE(ps.diversification, 0) IN (1, 5) THEN 1 ELSE 0 END
        + CASE WHEN COALESCE(ps.risk_preference, 0) IN (1, 5) THEN 1 ELSE 0 END
    ) AS extreme_settings_count
FROM interp_examples_v0 ie
JOIN payload_stats ps ON ps.log_id = ie.log_id
LEFT JOIN strategy_summary ss ON ss.log_id = ie.log_id
LEFT JOIN held_tokens ht ON ht.log_id = ie.log_id
WHERE ie.label_quality IN ('high', 'medium')
  AND ie.prompt_messages_json IS NOT NULL
  AND ie.market_snapshot_json IS NOT NULL;


CREATE UNIQUE INDEX IF NOT EXISTS idx_decision_capture_base_mv_log_id
    ON decision_capture_base_mv_v1 (log_id);

CREATE INDEX IF NOT EXISTS idx_decision_capture_base_mv_created_at
    ON decision_capture_base_mv_v1 (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_decision_capture_base_mv_trade
    ON decision_capture_base_mv_v1 (trade_side, target_asset, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_decision_capture_base_mv_blocked
    ON decision_capture_base_mv_v1 (blocked_observe_candidate, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_decision_capture_base_mv_policy_tension
    ON decision_capture_base_mv_v1 (policy_tension_candidate, created_at DESC);


CREATE OR REPLACE VIEW decision_capture_base_v1 AS
SELECT *
FROM decision_capture_base_mv_v1;


CREATE MATERIALIZED VIEW decision_capture_priority_v1 AS
WITH asset_freq AS (
    SELECT target_asset, COUNT(*) AS trade_asset_count
    FROM decision_capture_base_mv_v1
    WHERE is_trade AND target_asset IS NOT NULL
    GROUP BY 1
),
scored AS (
    SELECT
        b.*,
        COALESCE(af.trade_asset_count, 0) AS trade_asset_count,
        CASE
            WHEN b.trade_side = 'sell' THEN 'sell'
            WHEN b.trade_side = 'buy' THEN 'buy'
            WHEN b.policy_tension_candidate THEN 'policy_tension_observe'
            WHEN b.blocked_observe_candidate THEN 'blocked_observe'
            ELSE 'other_observe'
        END AS cohort_label,
        (
            CASE
                WHEN b.trade_side = 'sell' THEN 1000
                WHEN b.trade_side = 'buy' THEN 800
                WHEN b.policy_tension_candidate THEN 650
                WHEN b.blocked_observe_candidate THEN 550
                ELSE 100
            END
            + LEAST(COALESCE(b.extreme_settings_count, 0), 4) * 25
            + LEAST(COALESCE(b.high_strategy_count, 0), 3) * 20
            + CASE WHEN b.zero_eth THEN 15 ELSE 0 END
            + CASE
                WHEN COALESCE(af.trade_asset_count, 0) > 0
                    THEN ROUND(200.0 / SQRT(af.trade_asset_count))::int
                ELSE 0
              END
        ) AS capture_priority
    FROM decision_capture_base_mv_v1 b
    LEFT JOIN asset_freq af ON af.target_asset = b.target_asset
)
SELECT
    s.*,
    ROW_NUMBER() OVER (
        PARTITION BY s.cohort_label, COALESCE(s.target_asset, 'NONE')
        ORDER BY s.capture_priority DESC, s.created_at DESC NULLS LAST, s.log_id DESC
    ) AS within_cohort_asset_rank
FROM scored s;


CREATE UNIQUE INDEX IF NOT EXISTS idx_decision_capture_priority_log_id
    ON decision_capture_priority_v1 (log_id);

CREATE INDEX IF NOT EXISTS idx_decision_capture_priority_rank
    ON decision_capture_priority_v1 (capture_priority DESC, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_decision_capture_priority_cohort_asset
    ON decision_capture_priority_v1 (cohort_label, target_asset, within_cohort_asset_rank);


CREATE OR REPLACE VIEW decision_trade_candidates_v1 AS
SELECT
    p.*,
    ROW_NUMBER() OVER (
        PARTITION BY COALESCE(p.trade_side, 'none'), COALESCE(p.target_asset, 'NONE')
        ORDER BY p.created_at DESC NULLS LAST, p.log_id DESC
    ) AS within_asset_rank
FROM decision_capture_priority_v1 p
WHERE p.is_trade;


CREATE OR REPLACE VIEW decision_sell_candidates_v1 AS
SELECT
    p.*,
    ROW_NUMBER() OVER (
        PARTITION BY COALESCE(p.target_asset, 'NONE')
        ORDER BY p.created_at DESC NULLS LAST, p.log_id DESC
    ) AS within_asset_rank
FROM decision_capture_priority_v1 p
WHERE p.trade_side = 'sell';


CREATE OR REPLACE VIEW decision_blocked_observe_candidates_v1 AS
SELECT
    p.*,
    CASE
        WHEN p.blocks_all_buys AND p.blocks_all_sells THEN 'strategy_blocks_both'
        WHEN p.blocks_all_buys THEN 'strategy_blocks_buys'
        WHEN p.blocks_all_sells THEN 'strategy_blocks_sells'
        WHEN p.high_strategy_count > 0 THEN 'high_strategy_present'
        WHEN p.zero_eth AND p.held_token_count = 0 THEN 'no_eth_no_holdings'
        WHEN p.zero_eth THEN 'no_eth'
        WHEN p.forced_observe THEN 'forced_observe'
        ELSE 'other_block'
    END AS block_reason,
    ROW_NUMBER() OVER (
        PARTITION BY
            CASE
                WHEN p.blocks_all_buys AND p.blocks_all_sells THEN 'strategy_blocks_both'
                WHEN p.blocks_all_buys THEN 'strategy_blocks_buys'
                WHEN p.blocks_all_sells THEN 'strategy_blocks_sells'
                WHEN p.high_strategy_count > 0 THEN 'high_strategy_present'
                WHEN p.zero_eth AND p.held_token_count = 0 THEN 'no_eth_no_holdings'
                WHEN p.zero_eth THEN 'no_eth'
                WHEN p.forced_observe THEN 'forced_observe'
                ELSE 'other_block'
            END
        ORDER BY p.created_at DESC NULLS LAST, p.log_id DESC
    ) AS within_reason_rank
FROM decision_capture_priority_v1 p
WHERE p.blocked_observe_candidate;


CREATE OR REPLACE VIEW decision_policy_tension_candidates_v1 AS
SELECT
    p.*,
    ROW_NUMBER() OVER (
        PARTITION BY
            CASE
                WHEN p.trading_activity >= 4 THEN 'active'
                ELSE 'patient'
            END,
            CASE
                WHEN p.risk_preference IN (1, 2) THEN 'low_risk'
                WHEN p.risk_preference IN (4, 5) THEN 'high_risk'
                ELSE 'mid_risk'
            END
        ORDER BY p.created_at DESC NULLS LAST, p.log_id DESC
    ) AS within_settings_rank
FROM decision_capture_priority_v1 p
WHERE p.policy_tension_candidate;
