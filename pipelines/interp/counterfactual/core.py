"""Dataset construction for the Config Entanglement vs. Late Policy experiment.

Builds two datasets:
  - Dataset A (Canonical Mechanism Set): stripped prompts with swapped preambles
  - Dataset B (Real Prompt Validation Set): real prompts with preamble/settings edits

See plan: .claude/plans/woolly-snacking-bear.md
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOMINANT_SYSTEM_MSG_HASH_PREFIX = "a0b1d568"  # most common system message (10755 of 19929)

# Risk bucket → (vault_risk_preference values) used for preamble selection
LOW_RISK_LEVELS = (1, 2)
HIGH_RISK_LEVELS = (4, 5)

# Market section header marker
MARKET_HEADER = "## MARKET SNAPSHOT"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MarketRow:
    """One asset row from the market section of a prompt."""
    symbol: str
    name: str
    text_block: str  # the full rendered text for this row (multi-line)
    # Metrics extracted from market_snapshot_json
    pct_5m: float = 0.0
    pct_1h: float = 0.0
    net_flow_5m: float = 0.0
    vol_1h: float = 0.0
    vol_5m: float = 0.0
    unique_traders_5m: float = 0.0


@dataclass
class Snapshot:
    """A sampled market snapshot with parsed rows and labels."""
    snapshot_id: str  # example_id from the source row
    vault_address: str
    snap_date: str
    week_num: int
    market_json: dict[str, Any]
    market_header: str  # rendered header lines (ETH price, supply, etc.)
    rows: list[MarketRow] = field(default_factory=list)
    # Labels computed per-row (keyed by row index after randomization)
    labels: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class CanonicalPrompt:
    """A single prompt variant for Dataset A."""
    snapshot_id: str
    variant: str  # low_raw, high_raw, low_pad, high_pad
    system_text: str
    user_text: str
    row_order: list[str]  # symbol order after randomization
    # Token-level boundaries (populated after tokenization)
    section_boundaries: dict[str, tuple[int, int]] = field(default_factory=dict)
    row_boundaries: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DatasetASpec:
    """Full specification of Dataset A."""
    snapshots: list[Snapshot]
    train_ids: list[str]
    test_ids: list[str]
    system_text: str
    low_preamble: str
    high_preamble: str
    prompts: list[CanonicalPrompt] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _connect_neon():
    """Get a read-only Neon connection."""
    import psycopg
    from psycopg.rows import dict_row
    from pipelines.db import require_neon_dsn

    return psycopg.connect(require_neon_dsn(), autocommit=True, row_factory=dict_row)


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Template extraction
# ---------------------------------------------------------------------------

def extract_system_message(conn: Any) -> str:
    """Get the dominant system message from HQ observation prompts."""
    rows = conn.execute("""
        SELECT system_text, COUNT(*) AS cnt
        FROM interp_examples_v0
        WHERE has_market = true AND label_quality = 'high'
          AND decision_type = 'record_observation'
        GROUP BY system_text
        ORDER BY cnt DESC
        LIMIT 1
    """).fetchall()
    return rows[0]["system_text"]


def extract_preamble(conn: Any, risk_levels: tuple[int, ...]) -> str:
    """Get the dominant preamble template for given risk levels.

    Finds the most common preamble (text before MARKET SNAPSHOT) across all
    HQ observation prompts with the given risk levels. No date filter — the
    dominant template is selected by frequency across the full dataset.
    """
    row = conn.execute("""
        SELECT
            SUBSTRING(user_text FROM 1 FOR POSITION(%s IN user_text) - 1) AS preamble,
            COUNT(*) AS cnt
        FROM interp_examples_v0
        WHERE vault_risk_preference = ANY(%s)
          AND has_market = true AND label_quality = 'high'
          AND decision_type = 'record_observation'
          AND POSITION(%s IN user_text) > 0
        GROUP BY preamble
        ORDER BY cnt DESC
        LIMIT 1
    """, [
        MARKET_HEADER,
        list(risk_levels),
        MARKET_HEADER,
    ]).fetchone()
    if row is None:
        raise RuntimeError(
            f"No preamble found for risk_levels={risk_levels}. "
            "Check that interp_examples_v0 has data for these configs."
        )
    return row["preamble"]


# ---------------------------------------------------------------------------
# Market section parsing
# ---------------------------------------------------------------------------

def parse_market_section(user_text: str) -> tuple[str, list[str]]:
    """Parse the market section from user_text into header + per-asset row blocks.

    Returns (header_text, [row_block_text, ...]).
    Each row_block_text is the full multi-line text for one asset row.
    """
    mkt_start = user_text.find(MARKET_HEADER)
    if mkt_start < 0:
        raise ValueError("No MARKET SNAPSHOT section found in user_text")

    # Find next section header
    rest = user_text[mkt_start:]
    next_section = rest.find("\n## ", len(MARKET_HEADER))
    if next_section > 0:
        market_block = rest[:next_section]
    else:
        market_block = rest

    lines = market_block.split("\n")

    header_lines: list[str] = []
    asset_blocks: list[list[str]] = []
    current_block: list[str] = []
    in_header = True

    for line in lines:
        # Asset row starts with "  - NAME (SYMBOL) | Price:"
        stripped = line.lstrip()
        if stripped.startswith("- ") and "|" in stripped and "Price:" in stripped:
            if current_block:
                asset_blocks.append(current_block)
            current_block = [line]
            in_header = False
        elif in_header:
            header_lines.append(line)
        else:
            current_block.append(line)

    if current_block:
        asset_blocks.append(current_block)

    header_text = "\n".join(header_lines)
    row_texts = ["\n".join(block) for block in asset_blocks]
    return header_text, row_texts


def _extract_symbol_from_row(row_text: str) -> tuple[str, str]:
    """Extract (name, symbol) from a market row text block.

    Expected format: '  - AI GF (AIGF) | Price: ...'
    """
    match = re.search(r"-\s+(.+?)\s+\((\w+)\)\s*\|", row_text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "?", "?"


def build_market_rows(
    market_json: dict[str, Any],
    row_texts: list[str],
) -> list[MarketRow]:
    """Pair rendered row text blocks with metric data from market_snapshot_json."""
    tokens = market_json.get("Tokens", [])
    # Build lookup by symbol
    metrics_by_symbol: dict[str, dict[str, Any]] = {}
    for t in tokens:
        sym = t.get("Symbol", "")
        m = t.get("Metrics", {})
        metrics_by_symbol[sym] = {
            "name": t.get("Name", ""),
            "pct_5m": _safe_float(m.get("PctChange5m")),
            "pct_1h": _safe_float(m.get("PctChange1h")),
            "net_flow_5m": _safe_float(m.get("NetFlowInEth5m")),
            "vol_1h": _safe_float(m.get("VolumeInEth1h")),
            "vol_5m": _safe_float(m.get("VolumeInEth5m")),
            "unique_traders_5m": _safe_float(m.get("UniqueTraders5m")),
        }

    rows: list[MarketRow] = []
    for rt in row_texts:
        name, symbol = _extract_symbol_from_row(rt)
        m = metrics_by_symbol.get(symbol, {})
        rows.append(MarketRow(
            symbol=symbol,
            name=name,
            text_block=rt,
            pct_5m=m.get("pct_5m", 0.0),
            pct_1h=m.get("pct_1h", 0.0),
            net_flow_5m=m.get("net_flow_5m", 0.0),
            vol_1h=m.get("vol_1h", 0.0),
            vol_5m=m.get("vol_5m", 0.0),
            unique_traders_5m=m.get("unique_traders_5m", 0.0),
        ))
    return rows


# ---------------------------------------------------------------------------
# Row randomization
# ---------------------------------------------------------------------------

def randomize_rows(
    rows: list[MarketRow],
    snapshot_id: str,
) -> list[MarketRow]:
    """Deterministically randomize row order using snapshot_id as seed.

    The order is identical across all variants for the same snapshot.
    """
    seed = int(hashlib.sha256(snapshot_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled


# ---------------------------------------------------------------------------
# Label computation (per-row binary labels)
# ---------------------------------------------------------------------------

def compute_labels(rows: list[MarketRow]) -> dict[str, list[int]]:
    """Compute all label sets for a list of MarketRow objects.

    Returns dict mapping label_name -> list of 0/1 values (one per row).
    """
    n = len(rows)
    if n == 0:
        return {}

    labels: dict[str, list[int]] = {}

    # --- Sanity labels ---
    # is_top_5m_gainer
    best_5m_idx = max(range(n), key=lambda i: rows[i].pct_5m)
    labels["is_top_5m_gainer"] = [1 if i == best_5m_idx else 0 for i in range(n)]

    # is_top_net_flow
    best_flow_idx = max(range(n), key=lambda i: rows[i].net_flow_5m)
    labels["is_top_net_flow"] = [1 if i == best_flow_idx else 0 for i in range(n)]

    # --- Synthesis labels ---
    # is_momentum_divergence_leader: highest (PctChange5m - PctChange1h)
    best_div_idx = max(range(n), key=lambda i: rows[i].pct_5m - rows[i].pct_1h)
    labels["is_momentum_divergence_leader"] = [1 if i == best_div_idx else 0 for i in range(n)]

    # is_flow_surprise: highest NetFlowInEth5m / (VolumeInEth1h / 12 + eps)
    eps = 1e-10
    best_fs_idx = max(
        range(n),
        key=lambda i: rows[i].net_flow_5m / (rows[i].vol_1h / 12 + eps),
    )
    labels["is_flow_surprise"] = [1 if i == best_fs_idx else 0 for i in range(n)]

    # is_participation_momentum_leader: highest UniqueTraders5m × PctChange5m
    best_pm_idx = max(
        range(n),
        key=lambda i: rows[i].unique_traders_5m * rows[i].pct_5m,
    )
    labels["is_participation_momentum_leader"] = [1 if i == best_pm_idx else 0 for i in range(n)]

    return labels


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def render_market_section(
    header: str,
    rows: list[MarketRow],
) -> str:
    """Re-render the market section with rows in the given order."""
    parts = [header]
    for row in rows:
        parts.append(row.text_block)
    return "\n".join(parts)


def build_canonical_user_text(
    preamble: str,
    market_section: str,
) -> str:
    """Build a Dataset A user text: preamble + market + ## END."""
    return f"{preamble}{market_section}\n\n## END\n"


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def compute_padding(
    tokenizer: Any,
    low_preamble: str,
    high_preamble: str,
) -> tuple[str, str, int]:
    """Compute padded preamble variants that produce identical token counts.

    Returns (low_padded, high_padded, target_length).
    Padding is inserted at the end of the preamble (before market section).
    Uses a neutral padding string repeated to fill the gap.
    """
    low_tokens = tokenizer.encode(low_preamble, add_special_tokens=False)
    high_tokens = tokenizer.encode(high_preamble, add_special_tokens=False)

    low_len = len(low_tokens)
    high_len = len(high_tokens)
    target = max(low_len, high_len)

    # Padding uses a single space per unit — most tokenizers encode " " as
    # exactly 1 token, but consecutive spaces may merge.  We try single-char
    # pad units in preference order, falling back to dot-space which always
    # tokenizes to a predictable count.
    _PAD_CANDIDATES = [" ", ".", "\n", ". "]

    def pad_to_target(text: str, current_len: int, target_len: int) -> str:
        if current_len >= target_len:
            return text
        deficit = target_len - current_len
        # Try each candidate pad unit
        for pad_unit in _PAD_CANDIDATES:
            padded = text + (pad_unit * deficit)
            new_len = len(tokenizer.encode(padded, add_special_tokens=False))
            if new_len == target_len:
                return padded
            # Binary-search refinement: add/remove pad units to hit target
            if new_len < target_len:
                # Need more — add one at a time
                while new_len < target_len:
                    padded += pad_unit
                    new_len = len(tokenizer.encode(padded, add_special_tokens=False))
                if new_len == target_len:
                    return padded
            elif new_len > target_len:
                # Overshot — remove one at a time
                while new_len > target_len and padded.endswith(pad_unit):
                    padded = padded[:-len(pad_unit)]
                    new_len = len(tokenizer.encode(padded, add_special_tokens=False))
                if new_len == target_len:
                    return padded
        # Last resort: append spaces one-by-one with per-step verification
        padded = text
        cur = current_len
        while cur < target_len:
            padded += " "
            cur = len(tokenizer.encode(padded, add_special_tokens=False))
        return padded

    low_padded = pad_to_target(low_preamble, low_len, target)
    high_padded = pad_to_target(high_preamble, high_len, target)

    # Verify
    final_low = len(tokenizer.encode(low_padded, add_special_tokens=False))
    final_high = len(tokenizer.encode(high_padded, add_special_tokens=False))

    if final_low != final_high:
        print(f"WARNING: padding mismatch: low={final_low}, high={final_high}, target={target}")

    return low_padded, high_padded, target


# ---------------------------------------------------------------------------
# Snapshot sampling
# ---------------------------------------------------------------------------

def check_interp_examples(conn: Any) -> int:
    """Check that interp_examples_v0 exists and return row count.

    Raises RuntimeError if the table doesn't exist or is empty.
    Run the prep pipeline first: `uv run python -m pipelines.interp.prep`
    """
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM interp_examples_v0"
        ).fetchone()
    except Exception as e:
        raise RuntimeError(
            "interp_examples_v0 does not exist. "
            "Run the prep pipeline first to build it."
        ) from e
    cnt = row["cnt"]
    if cnt == 0:
        raise RuntimeError(
            "interp_examples_v0 is empty. "
            "Run the prep pipeline first to populate it."
        )
    return cnt


def find_best_roster(
    conn: Any,
    require_all_configs: bool = False,
) -> tuple[int, ...]:
    """Find the best asset roster (exact symbol set) for the experiment.

    The market adds/removes tokens over time, so different snapshots have
    different asset rosters. We lock to a single roster so every snapshot
    has the same assets — required for consistent probe dimensions, per-row
    label alignment, and avoiding asset-identity confounds.

    If require_all_configs=True, only considers rosters present across all
    three experiment configs (1/1, 3/3, 5/5). This is needed for Dataset B.

    Returns the roster as a sorted tuple of symbols.
    """
    # Extract sorted roster arrays in SQL to avoid fetching all JSON client-side
    _ROSTER_SQL = """
        SELECT (
            SELECT array_agg(sym ORDER BY sym)
            FROM jsonb_array_elements(market_snapshot_json::jsonb -> 'Tokens') AS t,
                 LATERAL (SELECT t->>'Symbol' AS sym) AS s
        ) AS roster
    """

    if require_all_configs:
        # Count per (config, roster) in SQL
        rows = conn.execute(f"""
            WITH rosters AS (
                {_ROSTER_SQL},
                vault_risk_preference || '/' || vault_trading_activity AS cfg
                FROM interp_examples_v0
                WHERE has_market = true
                  AND label_quality = 'high'
                  AND decision_type = 'record_observation'
                  AND market_snapshot_json IS NOT NULL
                  AND (vault_risk_preference, vault_trading_activity) IN ((1,1), (3,3), (5,5))
            )
            SELECT cfg, roster, COUNT(*) AS cnt
            FROM rosters
            GROUP BY cfg, roster
        """).fetchall()

        from collections import defaultdict
        config_rosters: dict[str, dict[tuple[str, ...], int]] = defaultdict(dict)
        for r in rows:
            roster_key = tuple(r["roster"])
            config_rosters[r["cfg"]][roster_key] = r["cnt"]

        available_cfgs = set(config_rosters.keys())
        if len(available_cfgs) < 2:
            return find_best_roster(conn, require_all_configs=False)

        roster_sets = [set(cfg_rosters.keys()) for cfg_rosters in config_rosters.values()]
        shared = roster_sets[0]
        for s in roster_sets[1:]:
            shared &= s

        if not shared:
            return find_best_roster(conn, require_all_configs=False)

        best_roster = max(
            shared,
            key=lambda r: min(config_rosters[c].get(r, 0) for c in available_cfgs),
        )
        min_count = min(config_rosters[c].get(best_roster, 0) for c in available_cfgs)
        print(f"  Best shared roster ({len(best_roster)} assets): {best_roster}")
        print(f"  Min coverage across configs: {min_count}")
        return best_roster
    else:
        # Most common roster overall, computed in SQL
        row = conn.execute(f"""
            WITH rosters AS (
                {_ROSTER_SQL}
                FROM interp_examples_v0
                WHERE has_market = true
                  AND label_quality = 'high'
                  AND decision_type = 'record_observation'
                  AND market_snapshot_json IS NOT NULL
            )
            SELECT roster, COUNT(*) AS cnt
            FROM rosters
            GROUP BY roster
            ORDER BY cnt DESC
            LIMIT 1
        """).fetchone()
        best_roster = tuple(row["roster"])
        print(f"  Modal roster ({len(best_roster)} assets, {row['cnt']} snapshots): {best_roster}")
        return best_roster


def sample_snapshots(
    conn: Any,
    n: int = 120,
    seed: int = 42,
    required_roster: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Sample n market snapshots, 1 per vault_address × UTC day, stratified across weeks.

    If required_roster is given, only samples snapshots whose asset symbols
    (sorted) exactly match this tuple. This ensures all snapshots have the
    same assets — critical because the market adds/removes tokens over time,
    and mixing rosters would confound asset identity with market features.
    """
    asset_filter = ""
    params: list[Any] = [str(seed), MARKET_HEADER]
    if required_roster is not None:
        # Filter: the sorted symbol set must match exactly.
        # We use a subquery that extracts and sorts symbols from the JSON.
        asset_filter = """
              AND (
                SELECT array_agg(sym ORDER BY sym)
                FROM jsonb_array_elements(market_snapshot_json::jsonb -> 'Tokens') AS t,
                     LATERAL (SELECT t->>'Symbol' AS sym) AS s
              ) = %s::text[]
        """
        params.append(list(required_roster))

    params.append(n)

    rows = conn.execute(f"""
        WITH ranked AS (
            SELECT
                example_id,
                vault_address,
                user_text,
                market_snapshot_json,
                created_at::date AS snap_date,
                EXTRACT(WEEK FROM created_at::date)::int AS week_num,
                jsonb_array_length(market_snapshot_json::jsonb -> 'Tokens') AS n_assets,
                ROW_NUMBER() OVER (
                    PARTITION BY vault_address, created_at::date
                    ORDER BY MD5(example_id || %s)
                ) AS rn
            FROM interp_examples_v0
            WHERE has_market = true
              AND label_quality = 'high'
              AND decision_type = 'record_observation'
              AND market_snapshot_json IS NOT NULL
              AND POSITION(%s IN user_text) > 0
              {asset_filter}
        )
        SELECT example_id, vault_address, user_text, market_snapshot_json,
               snap_date::text, week_num, n_assets
        FROM ranked
        WHERE rn = 1
        ORDER BY week_num, snap_date, vault_address
        LIMIT %s
    """, params).fetchall()

    return [dict(r) for r in rows]


def build_snapshots(raw_rows: list[dict[str, Any]]) -> list[Snapshot]:
    """Convert raw DB rows to Snapshot objects with parsed market rows and labels."""
    snapshots: list[Snapshot] = []

    for row in raw_rows:
        market_json = json.loads(row["market_snapshot_json"])
        header, row_texts = parse_market_section(row["user_text"])
        market_rows = build_market_rows(market_json, row_texts)

        snap = Snapshot(
            snapshot_id=row["example_id"],
            vault_address=row["vault_address"],
            snap_date=row["snap_date"],
            week_num=int(row["week_num"]),
            market_json=market_json,
            market_header=header,
            rows=market_rows,
        )

        # Randomize row order (deterministic per snapshot_id)
        snap.rows = randomize_rows(snap.rows, snap.snapshot_id)

        # Compute labels on randomized order
        snap.labels = compute_labels(snap.rows)

        snapshots.append(snap)

    return snapshots


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def split_train_test(
    snapshots: list[Snapshot],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split snapshot IDs into train/test sets (by snapshot_id, not by row).

    Returns (train_ids, test_ids).
    """
    rng = random.Random(seed)
    ids = [s.snapshot_id for s in snapshots]
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * test_fraction))
    return ids[n_test:], ids[:n_test]


# ---------------------------------------------------------------------------
# Dataset A construction
# ---------------------------------------------------------------------------

def build_dataset_a(
    conn: Any,
    n_snapshots: int = 120,
    seed: int = 42,
) -> DatasetASpec:
    """Build the full Dataset A specification.

    This samples snapshots, extracts templates, and constructs all 4 prompt
    variants per snapshot. Tokenizer-dependent padding is deferred to
    `finalize_padding()`.

    Prerequisite: interp_examples_v0 must exist (run prep pipeline first).
    """
    print("Checking prerequisites...")
    total = check_interp_examples(conn)
    print(f"  interp_examples_v0: {total:,} rows")

    print("Finding best asset roster (shared across all configs)...")
    roster = find_best_roster(conn, require_all_configs=True)
    print(f"  Locked roster ({len(roster)} assets): {roster}")

    print("Extracting templates...")
    system_text = extract_system_message(conn)
    low_preamble = extract_preamble(conn, LOW_RISK_LEVELS)
    high_preamble = extract_preamble(conn, HIGH_RISK_LEVELS)

    print(f"  System message: {len(system_text)} chars")
    print(f"  Low-risk preamble: {len(low_preamble)} chars")
    print(f"  High-risk preamble: {len(high_preamble)} chars")

    print(f"Sampling {n_snapshots} snapshots (roster: {roster})...")
    raw = sample_snapshots(conn, n=n_snapshots, seed=seed, required_roster=roster)
    snapshots = build_snapshots(raw)
    print(f"  Got {len(snapshots)} snapshots, "
          f"total rows: {sum(len(s.rows) for s in snapshots)}")

    train_ids, test_ids = split_train_test(snapshots, seed=seed)
    print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}")

    spec = DatasetASpec(
        snapshots=snapshots,
        train_ids=train_ids,
        test_ids=test_ids,
        system_text=system_text,
        low_preamble=low_preamble,
        high_preamble=high_preamble,
    )

    # Build raw (unpadded) prompts
    for snap in snapshots:
        market_section = render_market_section(snap.market_header, snap.rows)
        row_order = [r.symbol for r in snap.rows]

        for risk_tag, preamble in [("low", low_preamble), ("high", high_preamble)]:
            user_text = build_canonical_user_text(preamble, market_section)
            spec.prompts.append(CanonicalPrompt(
                snapshot_id=snap.snapshot_id,
                variant=f"{risk_tag}_raw",
                system_text=system_text,
                user_text=user_text,
                row_order=row_order,
            ))

    print(f"  Built {len(spec.prompts)} raw prompts (padded variants added after tokenizer)")
    return spec


def finalize_padding(
    spec: DatasetASpec,
    tokenizer: Any,
) -> None:
    """Add padded variants to the spec. Requires a tokenizer for token counting."""
    print("Computing padding...")
    low_padded, high_padded, target = compute_padding(
        tokenizer, spec.low_preamble, spec.high_preamble,
    )
    print(f"  Padding target: {target} tokens")

    snap_map = {s.snapshot_id: s for s in spec.snapshots}
    padded_prompts: list[CanonicalPrompt] = []

    for snap in spec.snapshots:
        market_section = render_market_section(snap.market_header, snap.rows)
        row_order = [r.symbol for r in snap.rows]

        for risk_tag, preamble in [("low", low_padded), ("high", high_padded)]:
            user_text = build_canonical_user_text(preamble, market_section)
            padded_prompts.append(CanonicalPrompt(
                snapshot_id=snap.snapshot_id,
                variant=f"{risk_tag}_pad",
                system_text=snap_map[snap.snapshot_id].market_header[:0] or spec.system_text,
                user_text=user_text,
                row_order=row_order,
            ))

    # Fix system_text for padded prompts
    for p in padded_prompts:
        p.system_text = spec.system_text

    spec.prompts.extend(padded_prompts)
    print(f"  Total prompts: {len(spec.prompts)} ({len(spec.snapshots)} snapshots × 4 variants)")


# ---------------------------------------------------------------------------
# Section boundary detection (post-tokenization)
# ---------------------------------------------------------------------------

def find_section_boundaries(
    tokenizer: Any,
    prompt: CanonicalPrompt,
) -> dict[str, tuple[int, int]]:
    """Find token-level boundaries for key sections in a canonical prompt.

    Returns dict of section_name -> (start_token_idx, end_token_idx)
    in the full chat-templated sequence (matching vLLM capture tokenization).
    """
    user_text = prompt.user_text
    system_text = prompt.system_text
    boundaries: dict[str, tuple[int, int]] = {}

    # Build the full tokenized input the same way the capture pipeline does
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    full_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, return_tensors=None,
    )
    full_len = len(full_ids)

    def _token_offset_at_user_char(char_pos: int) -> int:
        truncated_user = user_text[:char_pos]
        msgs = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": truncated_user},
        ]
        ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=False, return_tensors=None,
        )
        return len(ids)

    # Find market header position
    mkt_start = user_text.find(MARKET_HEADER)
    if mkt_start >= 0:
        preamble_end = _token_offset_at_user_char(mkt_start)
        boundaries["preamble"] = (0, preamble_end)

        # Find ## END
        end_marker = user_text.find("\n## END", mkt_start)
        if end_marker >= 0:
            market_end = _token_offset_at_user_char(end_marker)
            boundaries["market"] = (preamble_end, market_end)
            boundaries["suffix"] = (market_end, full_len)

    return boundaries


def find_row_boundaries(
    tokenizer: Any,
    prompt: CanonicalPrompt,
    snap: Snapshot,
) -> list[dict[str, Any]]:
    """Find token-level boundaries for each asset row within the market section.

    For each row, finds:
      - symbol_start, symbol_end: token range for "TOKEN_NAME (SYMBOL)" prefix
      - content_start, content_end: token range AFTER the "|" delimiter (metrics only)
      - full_start, full_end: token range for the entire row

    All positions are in the full chat-templated sequence (matching vLLM capture).
    The symbol-masked representation uses content_start:content_end.
    """
    user_text = prompt.user_text
    system_text = prompt.system_text
    row_bounds: list[dict[str, Any]] = []

    def _token_offset_at_user_char(char_pos: int) -> int:
        truncated_user = user_text[:char_pos]
        msgs = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": truncated_user},
        ]
        ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=False, return_tensors=None,
        )
        return len(ids)

    for i, market_row in enumerate(snap.rows):
        row_text = market_row.text_block
        row_start_char = user_text.find(row_text)
        if row_start_char < 0:
            print(f"WARNING: could not find row text for {market_row.symbol}")
            continue

        row_end_char = row_start_char + len(row_text)

        # Find the "|" delimiter position within the row
        pipe_pos = row_text.find("|")
        if pipe_pos < 0:
            pipe_pos = 0  # fallback: no delimiter found

        symbol_end_char = row_start_char + pipe_pos
        content_start_char = symbol_end_char  # includes the "|"

        # Convert char positions to token positions in full chat-templated sequence
        full_start = _token_offset_at_user_char(row_start_char)
        symbol_end = _token_offset_at_user_char(symbol_end_char)
        full_end = _token_offset_at_user_char(row_end_char)

        row_bounds.append({
            "row_index": i,
            "symbol": market_row.symbol,
            "full_start": full_start,
            "full_end": full_end,
            "symbol_start": full_start,
            "symbol_end": symbol_end,
            "content_start": symbol_end,
            "content_end": full_end,
        })

    return row_bounds


# ---------------------------------------------------------------------------
# Dataset B construction
# ---------------------------------------------------------------------------

@dataclass
class DatasetBPrompt:
    """A single prompt variant for Dataset B."""
    base_log_id: int
    example_id: str
    config_tag: str  # e.g. "1_1", "3_3", "5_5"
    variant: str  # "original", "preamble_swap_raw", "preamble_swap_pad", "settings_all1", "settings_all5"
    system_text: str
    user_text: str
    vault_risk_preference: int
    vault_trading_activity: int


def sample_dataset_b_prompts(
    conn: Any,
    n_per_config: int = 30,
    seed: int = 42,
    required_roster: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Sample real prompts for Dataset B.

    30 from 1/1, 30 from 3/3, 30 from 5/5.
    Cap at 2 per vault×day. Require dominant preamble hash in narrow window.
    If required_roster is set, only includes snapshots with that exact asset set.
    """
    configs = [
        (1, 1, "1_1"),
        (3, 3, "3_3"),
        (5, 5, "5_5"),
    ]

    asset_filter = ""
    extra_params: list[Any] = []
    if required_roster is not None:
        asset_filter = """
              AND (
                SELECT array_agg(sym ORDER BY sym)
                FROM jsonb_array_elements(market_snapshot_json::jsonb -> 'Tokens') AS t,
                     LATERAL (SELECT t->>'Symbol' AS sym) AS s
              ) = %s::text[]
        """
        extra_params = [list(required_roster)]

    all_rows: list[dict[str, Any]] = []

    for risk, activity, tag in configs:
        rows = conn.execute(f"""
            WITH ranked AS (
                SELECT
                    example_id,
                    log_id,
                    vault_address,
                    system_text,
                    user_text,
                    market_snapshot_json,
                    vault_risk_preference,
                    vault_trading_activity,
                    created_at::date AS snap_date,
                    ROW_NUMBER() OVER (
                        PARTITION BY vault_address, created_at::date
                        ORDER BY MD5(example_id || %s)
                    ) AS rn
                FROM interp_examples_v0
                WHERE vault_risk_preference = %s
                  AND vault_trading_activity = %s
                  AND has_market = true
                  AND label_quality = 'high'
                  AND decision_type = 'record_observation'
                  AND market_snapshot_json IS NOT NULL
                  AND POSITION(%s IN user_text) > 0
                  {asset_filter}
            )
            SELECT *
            FROM ranked
            WHERE rn <= 2
            ORDER BY MD5(example_id || %s)
            LIMIT %s
        """, [
            str(seed), risk, activity, MARKET_HEADER,
            *extra_params,
            str(seed), n_per_config,
        ]).fetchall()

        for r in rows:
            d = dict(r)
            d["config_tag"] = tag
            all_rows.append(d)

    return all_rows


def build_preamble_swapped_variant(
    original_user_text: str,
    new_preamble: str,
) -> str:
    """Replace the preamble in a real prompt with a different one."""
    mkt_start = original_user_text.find(MARKET_HEADER)
    if mkt_start < 0:
        raise ValueError("No MARKET SNAPSHOT section found")
    return new_preamble + original_user_text[mkt_start:]


def build_settings_edited_variant(
    user_text: str,
    slider_values: dict[str, int],
) -> str:
    """Replace ACTIVE SETTINGS slider values in a real prompt.

    slider_values maps setting name -> new value (1-5).
    """
    # ACTIVE SETTINGS section has lines like:
    #   - Trade Size: 1/5
    #   - Trading Activity: 1/5
    # etc.
    result = user_text
    for name, val in slider_values.items():
        # Match both legacy and current prompt formats:
        #   "Trade Size: 1/5"
        #   "Trade Size (Size): 1 / 5"
        pattern = rf"({re.escape(name)}(?:\s*\([^)]*\))?:\s*)\d(\s*/\s*5)"
        result = re.sub(pattern, rf"\g<1>{val}\2", result)
    return result


# ---------------------------------------------------------------------------
# Downstream section boundary detection (Dataset B)
# ---------------------------------------------------------------------------

# Section headers in causal order after MARKET SNAPSHOT
DOWNSTREAM_SECTIONS = [
    ("active_strategies", "## ACTIVE STRATEGIES"),
    ("active_settings", "## ACTIVE SETTINGS"),
    ("portfolio", "## PORTFOLIO CONTEXT"),
    ("constraints", "## CONSTRAINTS"),
    ("prev_decisions", "## PREVIOUS DECISIONS"),
]


def find_downstream_section_boundaries(
    tokenizer: Any,
    system_text: str,
    user_text: str,
) -> dict[str, tuple[int, int]]:
    """Find token-level boundaries for downstream sections in a full (Dataset B) prompt.

    Returns dict of section_name -> (start_token_idx, end_token_idx) in the
    full tokenized sequence (system + user, with chat template applied).

    Sections detected: active_strategies, active_settings, portfolio,
    constraints, prev_decisions, market.
    """
    # Build the full tokenized input the same way the capture pipeline does
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    full_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, return_tensors=None,
    )
    full_len = len(full_ids)

    # We need char→token mapping within user_text.
    # Strategy: encode prefix up to each section header start char to get token offset.
    # The chat template adds tokens around system/user, so we encode the full
    # template up to markers within the user_text portion.

    # First, find char offsets of each section header in user_text
    section_char_offsets: list[tuple[str, int]] = []
    for name, header in DOWNSTREAM_SECTIONS:
        idx = user_text.find(header)
        if idx >= 0:
            section_char_offsets.append((name, idx))

    # Also find market section
    mkt_idx = user_text.find(MARKET_HEADER)
    if mkt_idx >= 0:
        section_char_offsets.append(("market", mkt_idx))

    # Sort by char offset
    section_char_offsets.sort(key=lambda x: x[1])

    if not section_char_offsets:
        return {}

    # To convert char offsets in user_text to token offsets in the full sequence,
    # we build a reference: encode the chat template with truncated user_text.
    # This accounts for system message tokens + template tokens.
    def _token_offset_at_user_char(char_pos: int) -> int:
        """Get the token offset in the full sequence corresponding to a char
        position within user_text."""
        truncated_user = user_text[:char_pos]
        msgs = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": truncated_user},
        ]
        ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=False, return_tensors=None,
        )
        return len(ids)

    # Build boundaries: each section runs from its header to the next section's header
    boundaries: dict[str, tuple[int, int]] = {}
    for i, (name, char_off) in enumerate(section_char_offsets):
        start_tok = _token_offset_at_user_char(char_off)
        if i + 1 < len(section_char_offsets):
            end_tok = _token_offset_at_user_char(section_char_offsets[i + 1][1])
        else:
            end_tok = full_len
        boundaries[name] = (start_tok, end_tok)

    # Also add preamble (everything before market)
    if mkt_idx >= 0:
        preamble_end = _token_offset_at_user_char(mkt_idx)
        boundaries["preamble"] = (0, preamble_end)

    return boundaries


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _connect_neon_rw():
    """Get a read-write Neon connection (with explicit transactions)."""
    import psycopg
    from psycopg.rows import dict_row
    from pipelines.db import require_neon_dsn

    return psycopg.connect(require_neon_dsn(), autocommit=False, row_factory=dict_row)


# ---------------------------------------------------------------------------
# DB table DDL
# ---------------------------------------------------------------------------

_DDL_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS counterfactual_snapshots (
    snapshot_id   TEXT PRIMARY KEY,
    vault_address TEXT NOT NULL,
    snap_date     DATE NOT NULL,
    week_num      INT NOT NULL,
    n_rows        INT NOT NULL,
    roster        TEXT[] NOT NULL,
    row_order     TEXT[] NOT NULL,
    labels        JSONB NOT NULL,
    split         TEXT NOT NULL,  -- 'train' or 'test'
    market_json   JSONB NOT NULL
);
"""

_DDL_PROMPTS = """
CREATE TABLE IF NOT EXISTS counterfactual_prompts (
    prompt_id     TEXT PRIMARY KEY,  -- snapshot_id || '_' || variant
    snapshot_id   TEXT NOT NULL REFERENCES counterfactual_snapshots(snapshot_id),
    dataset       TEXT NOT NULL,     -- 'a' or 'b'
    variant       TEXT NOT NULL,     -- low_raw, high_raw, low_pad, high_pad, original, settings_all1, etc.
    config_tag    TEXT,              -- NULL for dataset A, '1_1'/'3_3'/'5_5' for B
    system_text   TEXT NOT NULL,
    user_text     TEXT NOT NULL,
    row_order     TEXT[] NOT NULL,
    n_rows        INT NOT NULL
);
"""

_DDL_TEMPLATES = """
CREATE TABLE IF NOT EXISTS counterfactual_templates (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_DDL_CAPTURES = """
CREATE TABLE IF NOT EXISTS counterfactual_captures (
    capture_id        TEXT PRIMARY KEY,
    experiment_id     TEXT NOT NULL,
    snapshot_id       TEXT NOT NULL,
    dataset           TEXT NOT NULL,
    variant           TEXT NOT NULL,
    seq_len           INT NOT NULL,
    n_rows            INT NOT NULL,
    n_residual_keys   INT NOT NULL DEFAULT 0,
    n_router_keys     INT NOT NULL DEFAULT 0,
    file_size_bytes   BIGINT NOT NULL DEFAULT 0,
    elapsed_s         REAL NOT NULL DEFAULT 0,
    capture_timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def create_tables(conn: Any) -> None:
    """Create counterfactual experiment tables (idempotent)."""
    with conn.cursor() as cur:
        cur.execute(_DDL_SNAPSHOTS)
        cur.execute(_DDL_PROMPTS)
        cur.execute(_DDL_TEMPLATES)
        cur.execute(_DDL_CAPTURES)
    conn.commit()


def save_dataset_a(spec: DatasetASpec, conn: Any) -> None:
    """Save Dataset A to counterfactual_* tables in Neon."""
    create_tables(conn)

    with conn.cursor() as cur:
        # Clear previous dataset A data
        cur.execute("DELETE FROM counterfactual_prompts WHERE dataset = 'a'")
        cur.execute("DELETE FROM counterfactual_snapshots")

        # Save templates
        for key, val in [
            ("system_text", spec.system_text),
            ("low_preamble", spec.low_preamble),
            ("high_preamble", spec.high_preamble),
            ("roster", json.dumps([r.symbol for r in spec.snapshots[0].rows] if spec.snapshots else [])),
            ("n_snapshots", str(len(spec.snapshots))),
        ]:
            cur.execute(
                """INSERT INTO counterfactual_templates (key, value)
                   VALUES (%s, %s)
                   ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
                [key, val],
            )

        train_set = set(spec.train_ids)

        # Save snapshots
        for s in spec.snapshots:
            cur.execute(
                """INSERT INTO counterfactual_snapshots
                   (snapshot_id, vault_address, snap_date, week_num, n_rows,
                    roster, row_order, labels, split, market_json)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    s.snapshot_id,
                    s.vault_address,
                    s.snap_date,
                    s.week_num,
                    len(s.rows),
                    [r.symbol for r in s.rows],
                    [r.symbol for r in s.rows],
                    json.dumps(s.labels),
                    "train" if s.snapshot_id in train_set else "test",
                    json.dumps(s.market_json),
                ],
            )

        # Save prompts
        for p in spec.prompts:
            cur.execute(
                """INSERT INTO counterfactual_prompts
                   (prompt_id, snapshot_id, dataset, variant, config_tag,
                    system_text, user_text, row_order, n_rows)
                   VALUES (%s, %s, 'a', %s, NULL, %s, %s, %s, %s)""",
                [
                    f"{p.snapshot_id}_{p.variant}",
                    p.snapshot_id,
                    p.variant,
                    p.system_text,
                    p.user_text,
                    p.row_order,
                    len(p.row_order),
                ],
            )

    conn.commit()
    print(f"Saved Dataset A to DB: {len(spec.snapshots)} snapshots, {len(spec.prompts)} prompts")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Build counterfactual datasets and save to Neon."""
    import argparse

    parser = argparse.ArgumentParser(description="Build counterfactual datasets")
    parser.add_argument("--n-snapshots", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["a", "b", "both"], default="both")
    args = parser.parse_args()

    # Read connection for sampling
    read_conn = _connect_neon()
    # Write connection for saving
    write_conn = _connect_neon_rw()

    try:
        if args.dataset in ("a", "both"):
            spec = build_dataset_a(read_conn, n_snapshots=args.n_snapshots, seed=args.seed)
            save_dataset_a(spec, write_conn)

        if args.dataset in ("b", "both"):
            print("\nBuilding Dataset B...")
            roster = find_best_roster(read_conn, require_all_configs=True)
            print(f"  Shared roster ({len(roster)} assets): {roster}")
            raw = sample_dataset_b_prompts(
                read_conn, seed=args.seed, required_roster=roster,
            )
            print(f"  Sampled {len(raw)} prompts across configs")
            save_dataset_b(raw, write_conn, roster)
    finally:
        read_conn.close()
        write_conn.close()


def save_dataset_b(
    raw_prompts: list[dict[str, Any]],
    conn: Any,
    roster: tuple[str, ...],
) -> None:
    """Save Dataset B prompts to counterfactual_prompts table."""
    create_tables(conn)

    with conn.cursor() as cur:
        # Clear previous dataset B data
        cur.execute("DELETE FROM counterfactual_prompts WHERE dataset = 'b'")

        # Save roster used for dataset B
        cur.execute(
            """INSERT INTO counterfactual_templates (key, value)
               VALUES ('roster_b', %s)
               ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
            [json.dumps(list(roster))],
        )

        for row in raw_prompts:
            example_id = row["example_id"]
            config_tag = row["config_tag"]
            system_text = row["system_text"]
            user_text = row["user_text"]
            market_json = json.loads(row["market_snapshot_json"])
            _, row_texts = parse_market_section(user_text)
            market_rows = build_market_rows(market_json, row_texts)
            row_order = [r.symbol for r in market_rows]

            # Ensure snapshot exists (may already be there from dataset A)
            cur.execute(
                """INSERT INTO counterfactual_snapshots
                   (snapshot_id, vault_address, snap_date, week_num, n_rows,
                    roster, row_order, labels, split, market_json)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'b_source', %s)
                   ON CONFLICT (snapshot_id) DO NOTHING""",
                [
                    example_id,
                    row["vault_address"],
                    row["snap_date"],
                    0,  # week_num not critical for B
                    len(market_rows),
                    list(roster),
                    row_order,
                    json.dumps(compute_labels(market_rows)),
                    json.dumps(market_json),
                ],
            )

            # Save original prompt
            cur.execute(
                """INSERT INTO counterfactual_prompts
                   (prompt_id, snapshot_id, dataset, variant, config_tag,
                    system_text, user_text, row_order, n_rows)
                   VALUES (%s, %s, 'b', 'original', %s, %s, %s, %s, %s)
                   ON CONFLICT (prompt_id) DO NOTHING""",
                [
                    f"{example_id}_original",
                    example_id,
                    config_tag,
                    system_text,
                    user_text,
                    row_order,
                    len(row_order),
                ],
            )

            # Settings-edited variants: all sliders to 1 and all to 5
            slider_names = [
                "Trade Size", "Trading Activity", "Holding Style",
                "Diversification", "Asset Risk Preference",
            ]
            for val, tag in [(1, "settings_all1"), (5, "settings_all5")]:
                edited = build_settings_edited_variant(
                    user_text,
                    {name: val for name in slider_names},
                )
                cur.execute(
                    """INSERT INTO counterfactual_prompts
                       (prompt_id, snapshot_id, dataset, variant, config_tag,
                        system_text, user_text, row_order, n_rows)
                       VALUES (%s, %s, 'b', %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (prompt_id) DO NOTHING""",
                    [
                        f"{example_id}_{tag}",
                        example_id,
                        tag,
                        config_tag,
                        system_text,
                        edited,
                        row_order,
                        len(row_order),
                    ],
                )

    conn.commit()
    n_prompts = len(raw_prompts) * 3  # original + 2 settings variants
    print(f"Saved Dataset B to DB: {len(raw_prompts)} base prompts, {n_prompts} total variants")


if __name__ == "__main__":
    main()
