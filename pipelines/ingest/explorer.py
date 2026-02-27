from __future__ import annotations

import argparse
import gzip
import json
import math
import sqlite3
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Terminal Ingest Explorer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-a: #f7f6ef;
      --bg-b: #e6f1f0;
      --ink: #1f2430;
      --muted: #5b6374;
      --card: rgba(255, 255, 255, 0.82);
      --line: rgba(31, 36, 48, 0.14);
      --accent: #0f766e;
      --accent-2: #dc2626;
      --shadow: 0 8px 24px rgba(31, 36, 48, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 500px at 0% 0%, rgba(15, 118, 110, 0.15), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(220, 38, 38, 0.10), transparent 55%),
        linear-gradient(180deg, var(--bg-a), var(--bg-b));
      min-height: 100vh;
    }
    .wrap { max-width: 1400px; margin: 0 auto; padding: 20px; }
    .top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 16px;
      animation: slideIn 400ms ease-out;
    }
    .title {
      font-size: clamp(1.3rem, 2vw, 2rem);
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    .muted { color: var(--muted); font-size: 0.95rem; }
    .btn {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--card);
      padding: 8px 12px;
      font-weight: 600;
      cursor: pointer;
      transition: transform 140ms ease, background 140ms ease;
    }
    .btn:hover { transform: translateY(-1px); background: white; }
    .cards {
      display: grid;
      grid-template-columns: repeat(6, minmax(140px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .cards.mini {
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      margin-bottom: 10px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--card);
      box-shadow: var(--shadow);
      padding: 10px 12px;
      animation: fadeUp 350ms ease-out;
    }
    .card h4 {
      margin: 0;
      color: var(--muted);
      font-size: 0.76rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .card .value {
      margin-top: 8px;
      font-weight: 700;
      font-size: 1.2rem;
      font-variant-numeric: tabular-nums;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1.35fr;
      gap: 12px;
      align-items: start;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--card);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .panel-head {
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.8);
    }
    .panel-head h3 {
      margin: 0;
      font-size: 0.95rem;
      letter-spacing: 0.02em;
    }
    .search {
      width: 230px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 9px;
      background: white;
      font: inherit;
    }
    .body-pad { padding: 10px 12px; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.86rem;
    }
    th, td {
      text-align: left;
      padding: 7px 8px;
      border-bottom: 1px solid var(--line);
      font-variant-numeric: tabular-nums;
    }
    th { color: var(--muted); font-weight: 600; }
    tr.pick { background: rgba(15, 118, 110, 0.10); }
    tr:hover { background: rgba(15, 118, 110, 0.06); cursor: pointer; }
    .mono {
      font-family: "IBM Plex Mono", "SFMono-Regular", Menlo, monospace;
      font-size: 0.82rem;
    }
    .split {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      padding: 10px 12px 12px;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 0.78rem;
      background: rgba(255, 255, 255, 0.8);
    }
    .chip.ok { border-color: rgba(15, 118, 110, 0.38); color: #0f766e; }
    .chip.bad { border-color: rgba(220, 38, 38, 0.38); color: #b91c1c; }
    .box {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.78);
      padding: 10px;
      overflow: auto;
      max-height: 300px;
    }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; }
    .empty {
      color: var(--muted);
      padding: 8px;
      border: 1px dashed var(--line);
      border-radius: 10px;
    }
    .two {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .h4-tight {
      margin: 4px 0 8px;
      font-size: 0.86rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }
    @media (max-width: 1100px) {
      .cards { grid-template-columns: repeat(3, minmax(120px, 1fr)); }
      .cards.mini { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
      .grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 640px) {
      .cards { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .search { width: 100%; }
      .panel-head { flex-direction: column; align-items: stretch; }
      .two { grid-template-columns: 1fr; }
    }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideIn {
      from { opacity: 0; transform: translateY(-6px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div>
        <div class="title">Terminal Ingest Explorer</div>
        <div class="muted">SQLite metadata + raw full-log pointers, browsable by vault and log.</div>
      </div>
      <button class="btn" id="refreshBtn">Refresh</button>
    </div>

    <section class="cards" id="cards"></section>

    <section class="grid">
      <div class="panel">
        <div class="panel-head">
          <h3>Vaults</h3>
          <input id="vaultSearch" class="search" placeholder="Search address / name / owner" />
        </div>
        <div class="body-pad">
          <table id="vaultTable">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Name</th>
                <th>Vault</th>
                <th>Total PnL USD</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <h3 id="vaultTitle">Vault Details</h3>
          <button class="btn" id="payloadBtn" disabled>Load Payload Preview</button>
        </div>
        <div class="split">
          <div id="vaultMeta" class="empty">Select a vault from the left table.</div>
          <div class="two">
            <div>
              <h4>Strategies</h4>
              <div id="strategyList" class="box mono"></div>
            </div>
            <div>
              <h4>Recent Logs</h4>
              <div id="logList" class="box mono"></div>
            </div>
          </div>
          <div>
            <h4>Selected Log</h4>
            <div id="logDetail" class="box mono"></div>
          </div>
          <div>
            <h4>Payload Preview</h4>
            <div id="payloadPreview" class="box mono"></div>
          </div>
        </div>
      </div>
    </section>

    <section class="panel" style="margin-top: 12px;">
      <div class="panel-head">
        <h3>Dataset Readiness (Mech Interp)</h3>
        <button class="btn" id="refreshDatasetBtn">Refresh Dataset Stats</button>
      </div>
      <div class="split">
        <div class="cards mini" id="datasetCards"></div>
        <div class="two">
          <div>
            <div class="h4-tight">Join Coverage</div>
            <div id="joinCoverage" class="box mono"></div>
          </div>
          <div>
            <div class="h4-tight">Tool Mix</div>
            <div id="toolMix" class="box mono"></div>
          </div>
        </div>
        <div>
          <div class="h4-tight">Recommended Next Steps</div>
          <div id="nextSteps" class="box mono"></div>
        </div>
        <div>
          <div class="h4-tight">Candidate Rows (Full Logs Prioritized For Labeling)</div>
          <div class="box">
            <table id="candidateTable">
              <thead>
                <tr>
                  <th>Log ID</th>
                  <th>Tool</th>
                  <th>Join State</th>
                  <th>Trade?</th>
                  <th>Prompt?</th>
                  <th>Decision Text?</th>
                  <th>Prompt Tok</th>
                  <th>Completion Tok</th>
                  <th>Reasoning Tok</th>
                  <th>Vault</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  </div>

  <script>
    const state = {
      vaults: [],
      selectedVault: null,
      selectedLogId: null,
      payloadLoadedFor: null,
    };

    const cardsEl = document.getElementById("cards");
    const vaultTableBody = document.querySelector("#vaultTable tbody");
    const vaultSearch = document.getElementById("vaultSearch");
    const vaultTitle = document.getElementById("vaultTitle");
    const vaultMeta = document.getElementById("vaultMeta");
    const strategyList = document.getElementById("strategyList");
    const logList = document.getElementById("logList");
    const logDetail = document.getElementById("logDetail");
    const payloadPreview = document.getElementById("payloadPreview");
    const payloadBtn = document.getElementById("payloadBtn");
    const datasetCards = document.getElementById("datasetCards");
    const joinCoverage = document.getElementById("joinCoverage");
    const toolMix = document.getElementById("toolMix");
    const nextSteps = document.getElementById("nextSteps");
    const candidateTableBody = document.querySelector("#candidateTable tbody");

    function fmt(n) {
      if (n === null || n === undefined) return "—";
      if (typeof n === "number") return n.toLocaleString();
      return String(n);
    }

    function fmtMoney(n) {
      if (n === null || n === undefined) return "—";
      return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(n);
    }

    function shortAddr(v) {
      if (!v || v.length < 12) return v || "—";
      return `${v.slice(0, 8)}…${v.slice(-6)}`;
    }

    function escapeHtml(s) {
      return String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
    }

    async function api(path) {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    }

    function renderCards(overview) {
      const defs = [
        ["Vaults", overview.vault_count],
        ["Strategies", overview.strategy_count],
        ["Inference Logs", overview.log_count],
        ["Full Logs", overview.full_log_count],
        ["Parse Errors", overview.parse_error_count],
        ["Coverage", `${overview.full_log_coverage_pct}%`],
      ];
      cardsEl.innerHTML = defs.map(([label, val]) => `
        <article class="card">
          <h4>${label}</h4>
          <div class="value">${fmt(val)}</div>
        </article>
      `).join("");
    }

    function renderDatasetCards(readiness) {
      const defs = [
        ["Logs", readiness.total_logs],
        ["Full-Log Coverage", `${readiness.full_log_coverage_pct}%`],
        ["Trade Logs", readiness.logs_with_tx_hash],
        ["Current Candidates", readiness.candidate_rows_now],
        ["Labeled Candidates", readiness.candidate_rows_with_outcomes],
        ["Avg Prompt Tok", readiness.avg_prompt_tokens],
        ["Avg Completion Tok", readiness.avg_completion_tokens],
        ["Avg Reasoning Tok", readiness.avg_reasoning_tokens],
      ];
      datasetCards.innerHTML = defs.map(([label, val]) => `
        <article class="card">
          <h4>${label}</h4>
          <div class="value">${fmt(val)}</div>
        </article>
      `).join("");
    }

    function renderReadiness(readiness) {
      renderDatasetCards(readiness);

      const joinLines = [
        `logs -> full_logs: ${readiness.logs_with_full_logs} / ${readiness.total_logs} (${readiness.full_log_coverage_pct}%)`,
        `logs with tx hash: ${readiness.logs_with_tx_hash}`,
        `swaps table present: ${readiness.has_swaps_table ? "yes" : "no"}`,
        `trade_outcomes table present: ${readiness.has_trade_outcomes_table ? "yes" : "no"}`,
        `swap rows: ${readiness.swaps_total}`,
        `swaps with log_id: ${readiness.swaps_with_log_id}`,
        `logs joined from swaps.log_id: ${readiness.logs_joined_from_swaps}`,
        `outcome rows: ${readiness.trade_outcomes_total}`,
        `logs joined with outcomes: ${readiness.logs_with_outcomes}`
      ];
      joinCoverage.innerHTML = `<pre>${escapeHtml(joinLines.join("\\n"))}</pre>`;

      if (!readiness.tool_distribution.length) {
        toolMix.innerHTML = '<div class="empty">No logs available yet.</div>';
      } else {
        const lines = readiness.tool_distribution.map((t) => {
          const pct = t.total ? ((t.trade_count / t.total) * 100).toFixed(1) : "0.0";
          return `${t.tool || "unknown"}: total=${t.total}, trade-linked=${t.trade_count} (${pct}%)`;
        });
        toolMix.innerHTML = `<pre>${escapeHtml(lines.join("\\n"))}</pre>`;
      }

      if (!readiness.next_steps.length) {
        nextSteps.innerHTML = "<pre>No blocking gaps detected for current schema snapshot.</pre>";
      } else {
        nextSteps.innerHTML = `<pre>${escapeHtml(readiness.next_steps.map((s, idx) => `${idx + 1}. ${s}`).join("\\n"))}</pre>`;
      }
    }

    function renderCandidates(items) {
      candidateTableBody.innerHTML = "";
      for (const row of items) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td class="mono">${row.id}</td>
          <td>${escapeHtml(row.tool || "—")}</td>
          <td class="mono">${escapeHtml(row.join_state || "—")}</td>
          <td>${row.is_trade ? "yes" : "no"}</td>
          <td>${row.has_prompt_text ? "yes" : "no"}</td>
          <td>${row.has_decision_text ? "yes" : "no"}</td>
          <td>${fmt(row.prompt_tokens)}</td>
          <td>${fmt(row.completion_tokens)}</td>
          <td>${fmt(row.reasoning_tokens)}</td>
          <td class="mono">${escapeHtml(shortAddr(row.vault_address))}</td>
        `;
        tr.onclick = async () => {
          if (row.vault_address) {
            await selectVault(row.vault_address);
          }
          await selectLog(row.id);
        };
        candidateTableBody.appendChild(tr);
      }
      if (!items.length) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="10" class="muted">No candidates yet. Ingest more logs/full-logs first.</td>`;
        candidateTableBody.appendChild(tr);
      }
    }

    function renderVaults() {
      vaultTableBody.innerHTML = "";
      for (const vault of state.vaults) {
        const tr = document.createElement("tr");
        if (state.selectedVault && state.selectedVault.vault_address === vault.vault_address) tr.className = "pick";
        tr.innerHTML = `
          <td>${fmt(vault.leaderboard_rank)}</td>
          <td>${escapeHtml(vault.nft_name || "—")}</td>
          <td class="mono">${escapeHtml(shortAddr(vault.vault_address))}</td>
          <td>${fmtMoney(vault.total_pnl_usd)}</td>
        `;
        tr.onclick = () => selectVault(vault.vault_address);
        vaultTableBody.appendChild(tr);
      }
    }

    function renderVaultDetails(detail) {
      if (!detail || !detail.vault) {
        vaultMeta.innerHTML = '<div class="empty">Vault not found.</div>';
        strategyList.innerHTML = "";
        logList.innerHTML = "";
        return;
      }

      const v = detail.vault;
      vaultTitle.textContent = `Vault Details: ${v.nft_name || shortAddr(v.vault_address)}`;
      vaultMeta.innerHTML = `
        <div class="chips">
          <span class="chip">rank: ${fmt(v.leaderboard_rank)}</span>
          <span class="chip">trade_size: ${fmt(v.trade_size)}</span>
          <span class="chip">activity: ${fmt(v.trading_activity)}</span>
          <span class="chip">holding: ${fmt(v.holding_style)}</span>
          <span class="chip">diversification: ${fmt(v.diversification)}</span>
          <span class="chip">risk: ${fmt(v.asset_risk_preference)}</span>
          <span class="chip ${v.paused ? "bad" : "ok"}">${v.paused ? "paused" : "active"}</span>
          <span class="chip">logs: ${fmt(detail.stats.log_count)}</span>
          <span class="chip">full_logs: ${fmt(detail.stats.full_log_count)}</span>
        </div>
        <pre class="mono">${escapeHtml(JSON.stringify({
          vault_address: v.vault_address,
          owner_address: v.owner_address,
          nft_id: v.nft_id,
          total_pnl_usd: v.total_pnl_usd,
          realized_pnl_usd: v.realized_pnl_usd,
          unrealized_pnl_usd: v.unrealized_pnl_usd,
          latest_log_at: detail.stats.latest_log_at
        }, null, 2))}</pre>
      `;

      if (!detail.strategies.length) {
        strategyList.innerHTML = '<div class="empty">No strategies ingested.</div>';
      } else {
        strategyList.innerHTML = detail.strategies.map((s) => `
          <div style="padding-bottom:8px; margin-bottom:8px; border-bottom:1px solid var(--line);">
            <div><strong>#${escapeHtml(String(s.strategy_id))}</strong> <span class="chip ${s.enabled ? "ok" : ""}">${s.enabled ? "enabled" : "disabled"}</span></div>
            <div class="muted">${escapeHtml(s.strategy_priority || "n/a")} / expiry: ${fmt(s.expiry)}</div>
            <div>${escapeHtml(s.content || "")}</div>
          </div>
        `).join("");
      }

      if (!detail.recent_logs.length) {
        logList.innerHTML = '<div class="empty">No logs ingested for this vault.</div>';
      } else {
        logList.innerHTML = detail.recent_logs.map((log) => `
          <div data-log-id="${log.id}" style="padding:6px 4px; border-bottom:1px solid var(--line); cursor:pointer;">
            <div><strong>#${log.id}</strong> ${escapeHtml(log.tool || "unknown")} <span class="muted">${escapeHtml(log.status || "")}</span></div>
            <div class="muted">${escapeHtml(log.created_at || "n/a")} / full: ${log.has_full_log ? "yes" : "no"}</div>
          </div>
        `).join("");

        logList.querySelectorAll("[data-log-id]").forEach((el) => {
          el.onclick = () => selectLog(Number(el.getAttribute("data-log-id")));
        });
      }
    }

    function renderLogDetail(detail, withPayload = false) {
      if (!detail || !detail.log) {
        logDetail.innerHTML = '<div class="empty">No log selected.</div>';
        return;
      }
      const log = detail.log;
      logDetail.innerHTML = `<pre>${escapeHtml(JSON.stringify(log, null, 2))}</pre>`;
      payloadBtn.disabled = false;
      if (withPayload && detail.payload_preview) {
        payloadPreview.innerHTML = `<pre>${escapeHtml(detail.payload_preview)}</pre>`;
        state.payloadLoadedFor = log.id;
      }
    }

    async function loadOverview() {
      const data = await api("/api/overview");
      renderCards(data);
    }

    async function loadDatasetReadiness() {
      const readiness = await api("/api/dataset-readiness");
      renderReadiness(readiness);
      const candidates = await api("/api/dataset-candidates?limit=120");
      renderCandidates(candidates.items || []);
    }

    async function loadVaults() {
      const q = encodeURIComponent(vaultSearch.value.trim());
      const data = await api(`/api/vaults?limit=100&offset=0&q=${q}`);
      state.vaults = data.items;
      renderVaults();
    }

    async function selectVault(vaultAddress) {
      state.selectedVault = state.vaults.find((v) => v.vault_address === vaultAddress) || null;
      state.selectedLogId = null;
      state.payloadLoadedFor = null;
      payloadBtn.disabled = true;
      payloadPreview.innerHTML = '<div class="empty">Payload not loaded.</div>';
      logDetail.innerHTML = '<div class="empty">Select a log in "Recent Logs".</div>';
      renderVaults();
      const detail = await api(`/api/vault/${vaultAddress}`);
      renderVaultDetails(detail);
    }

    async function selectLog(logId) {
      state.selectedLogId = logId;
      state.payloadLoadedFor = null;
      payloadPreview.innerHTML = '<div class="empty">Payload not loaded.</div>';
      const detail = await api(`/api/log/${logId}`);
      renderLogDetail(detail, false);
    }

    async function loadPayloadForSelectedLog() {
      if (!state.selectedLogId) return;
      const detail = await api(`/api/log/${state.selectedLogId}?include_payload=1`);
      renderLogDetail(detail, true);
    }

    document.getElementById("refreshBtn").onclick = async () => {
      await loadOverview();
      await loadVaults();
      if (state.selectedVault) await selectVault(state.selectedVault.vault_address);
      await loadDatasetReadiness();
    };
    document.getElementById("refreshDatasetBtn").onclick = loadDatasetReadiness;

    payloadBtn.onclick = loadPayloadForSelectedLog;

    let searchTimer = null;
    vaultSearch.oninput = () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(loadVaults, 220);
    };

    async function boot() {
      payloadPreview.innerHTML = '<div class="empty">Payload not loaded.</div>';
      logDetail.innerHTML = '<div class="empty">Select a log in "Recent Logs".</div>';
      await loadOverview();
      await loadVaults();
      await loadDatasetReadiness();
      if (state.vaults[0]) await selectVault(state.vaults[0].vault_address);
    }
    boot().catch((err) => {
      console.error(err);
      alert("Failed to load explorer data. Check server logs.");
    });
  </script>
</body>
</html>
"""


class ExplorerStore:
    def __init__(self, db_path: Path, payload_preview_chars: int = 12000) -> None:
        self.db_path = db_path
        self.payload_preview_chars = payload_preview_chars

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _table_exists(self, table_name: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
                [table_name],
            ).fetchone()
        return row is not None

    @staticmethod
    def _percentile(values: list[int], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        rank = (len(ordered) - 1) * percentile
        low = math.floor(rank)
        high = math.ceil(rank)
        if low == high:
            return float(ordered[low])
        low_val = ordered[low]
        high_val = ordered[high]
        return float(low_val + (high_val - low_val) * (rank - low))

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return dict(row)

    def get_overview(self) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM vaults) AS vault_count,
                    (SELECT COUNT(*) FROM strategies) AS strategy_count,
                    (SELECT COUNT(*) FROM inference_logs) AS log_count,
                    (SELECT COUNT(*) FROM full_logs) AS full_log_count,
                    (SELECT COUNT(*) FROM full_logs WHERE parse_error IS NOT NULL AND parse_error != '') AS parse_error_count,
                    (SELECT MAX(created_at) FROM inference_logs) AS latest_log_at,
                    (SELECT MAX(fetched_at) FROM full_logs) AS latest_full_log_fetched_at
                """
            ).fetchone()
        result = dict(row) if row else {}
        log_count = int(result.get("log_count") or 0)
        full_log_count = int(result.get("full_log_count") or 0)
        coverage = round((full_log_count / log_count) * 100.0, 2) if log_count else 0.0
        result["full_log_coverage_pct"] = coverage
        return result

    def list_vaults(self, *, limit: int, offset: int, q: str | None) -> dict[str, Any]:
        where = ""
        params: list[Any] = []
        if q:
            where = """
            WHERE
                lower(vault_address) LIKE ?
                OR lower(owner_address) LIKE ?
                OR lower(nft_name) LIKE ?
            """
            query = f"%{q.lower()}%"
            params.extend([query, query, query])

        with self._connect() as conn:
            total_row = conn.execute(f"SELECT COUNT(*) AS total_count FROM vaults {where}", params).fetchone()
            rows = conn.execute(
                f"""
                SELECT
                    vault_address, owner_address, nft_id, nft_name,
                    leaderboard_rank, total_pnl_usd, realized_pnl_usd, unrealized_pnl_usd,
                    trade_size, trading_activity, holding_style, diversification,
                    asset_risk_preference, paused, state, fetched_at
                FROM vaults
                {where}
                ORDER BY
                    CASE WHEN leaderboard_rank IS NULL THEN 1 ELSE 0 END,
                    leaderboard_rank ASC,
                    total_pnl_usd DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return {
            "total_count": int(total_row["total_count"]) if total_row else 0,
            "items": [dict(r) for r in rows],
        }

    def get_vault_detail(self, vault_address: str) -> dict[str, Any]:
        with self._connect() as conn:
            vault = conn.execute(
                "SELECT * FROM vaults WHERE vault_address = ?",
                [vault_address],
            ).fetchone()
            strategies = conn.execute(
                """
                SELECT
                    strategy_id, content, enabled, expiry, strategy_priority,
                    created_block, updated_block, fetched_at
                FROM strategies
                WHERE vault_address = ?
                ORDER BY enabled DESC, CAST(strategy_id AS INTEGER) DESC
                """,
                [vault_address],
            ).fetchall()
            recent_logs = conn.execute(
                """
                SELECT
                    l.id, l.created_at, l.completed_at, l.status, l.tool, l.strategy_id,
                    l.transaction_hash, l.inference_duration_ms,
                    CASE WHEN f.log_id IS NOT NULL THEN 1 ELSE 0 END AS has_full_log
                FROM inference_logs l
                LEFT JOIN full_logs f ON f.log_id = l.id
                WHERE l.vault_address = ?
                ORDER BY l.id DESC
                LIMIT 60
                """,
                [vault_address],
            ).fetchall()
            stats = conn.execute(
                """
                SELECT
                    COUNT(*) AS log_count,
                    SUM(CASE WHEN f.log_id IS NOT NULL THEN 1 ELSE 0 END) AS full_log_count,
                    MAX(l.created_at) AS latest_log_at
                FROM inference_logs l
                LEFT JOIN full_logs f ON f.log_id = l.id
                WHERE l.vault_address = ?
                """,
                [vault_address],
            ).fetchone()

        return {
            "vault": self._row_to_dict(vault),
            "strategies": [dict(r) for r in strategies],
            "recent_logs": [dict(r) for r in recent_logs],
            "stats": dict(stats) if stats else {},
        }

    def get_log_detail(self, log_id: int, *, include_payload: bool) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    l.id, l.vault_address, l.cursor, l.request_id, l.execution_key,
                    l.tool, l.tool_args_json, l.strategy_id, l.status, l.inference_duration_ms,
                    l.error, l.transaction_hash, l.created_at, l.completed_at, l.fetched_at,
                    f.payload_path, f.payload_sha256, f.payload_size_bytes,
                    f.prompt_text, f.completion_text, f.reasoning_content, f.tool_calls_json,
                    f.llm_model, f.prompt_tokens, f.completion_tokens, f.reasoning_tokens,
                    f.total_tokens, f.parse_error, f.fetched_at AS full_log_fetched_at
                FROM inference_logs l
                LEFT JOIN full_logs f ON f.log_id = l.id
                WHERE l.id = ?
                """,
                [log_id],
            ).fetchone()

        if row is None:
            return {"log": None, "payload_preview": None}

        item = dict(row)
        for key in ["tool_args_json", "tool_calls_json"]:
            value = item.get(key)
            if not value:
                item[key] = None
                continue
            try:
                item[key] = json.loads(value)
            except json.JSONDecodeError:
                pass

        payload_preview: str | None = None
        if include_payload:
            payload_preview = self._load_payload_preview(item.get("payload_path"))

        return {"log": item, "payload_preview": payload_preview}

    def get_dataset_readiness(self) -> dict[str, Any]:
        has_swaps_table = self._table_exists("swaps")
        has_trade_outcomes_table = self._table_exists("trade_outcomes")

        with self._connect() as conn:
            core = conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM inference_logs) AS total_logs,
                    (SELECT COUNT(*) FROM full_logs) AS logs_with_full_logs,
                    (SELECT COUNT(*) FROM inference_logs WHERE transaction_hash IS NOT NULL AND transaction_hash != '') AS logs_with_tx_hash,
                    (SELECT COUNT(*) FROM full_logs WHERE parse_error IS NOT NULL AND parse_error != '') AS full_logs_with_parse_errors,
                    (SELECT COUNT(*) FROM full_logs WHERE prompt_text IS NOT NULL AND prompt_text != '') AS full_logs_with_prompt_text,
                    (SELECT COUNT(*) FROM full_logs WHERE completion_text IS NOT NULL AND completion_text != '') AS full_logs_with_completion_text,
                    (SELECT COUNT(*) FROM full_logs WHERE reasoning_content IS NOT NULL AND reasoning_content != '') AS full_logs_with_reasoning_text,
                    (SELECT COUNT(*) FROM full_logs WHERE prompt_tokens IS NOT NULL) AS full_logs_with_prompt_tokens,
                    (SELECT COUNT(*) FROM full_logs WHERE completion_tokens IS NOT NULL) AS full_logs_with_completion_tokens,
                    (SELECT COUNT(*) FROM full_logs WHERE reasoning_tokens IS NOT NULL) AS full_logs_with_reasoning_tokens
                """
            ).fetchone()

            core_map = dict(core) if core else {}
            total_logs = int(core_map.get("total_logs") or 0)
            logs_with_full_logs = int(core_map.get("logs_with_full_logs") or 0)
            logs_with_tx_hash = int(core_map.get("logs_with_tx_hash") or 0)
            full_log_coverage_pct = round((logs_with_full_logs / total_logs) * 100.0, 2) if total_logs else 0.0

            token_rows = conn.execute(
                """
                SELECT prompt_tokens, completion_tokens, reasoning_tokens, total_tokens
                FROM full_logs
                WHERE prompt_tokens IS NOT NULL OR completion_tokens IS NOT NULL OR reasoning_tokens IS NOT NULL
                """
            ).fetchall()
            prompt_values = [int(r["prompt_tokens"]) for r in token_rows if r["prompt_tokens"] is not None]
            completion_values = [int(r["completion_tokens"]) for r in token_rows if r["completion_tokens"] is not None]
            reasoning_values = [int(r["reasoning_tokens"]) for r in token_rows if r["reasoning_tokens"] is not None]
            total_values = [int(r["total_tokens"]) for r in token_rows if r["total_tokens"] is not None]

            tool_distribution_rows = conn.execute(
                """
                SELECT
                    tool,
                    COUNT(*) AS total,
                    SUM(CASE WHEN transaction_hash IS NOT NULL AND transaction_hash != '' THEN 1 ELSE 0 END) AS trade_count
                FROM inference_logs
                GROUP BY tool
                ORDER BY total DESC
                """
            ).fetchall()
            tool_distribution = [dict(row) for row in tool_distribution_rows]

            candidate_now_row = conn.execute(
                """
                SELECT COUNT(*) AS count_now
                FROM inference_logs l
                INNER JOIN full_logs f ON f.log_id = l.id
                WHERE l.transaction_hash IS NOT NULL AND l.transaction_hash != ''
                AND (f.parse_error IS NULL OR f.parse_error = '')
                """
            ).fetchone()
            candidate_rows_now = int(candidate_now_row["count_now"]) if candidate_now_row else 0

            swaps_total = 0
            swaps_with_log_id = 0
            logs_joined_from_swaps = 0
            if has_swaps_table:
                swap_row = conn.execute(
                    """
                    SELECT
                        COUNT(*) AS swaps_total,
                        SUM(CASE WHEN log_id IS NOT NULL THEN 1 ELSE 0 END) AS swaps_with_log_id
                    FROM swaps
                    """
                ).fetchone()
                if swap_row:
                    swaps_total = int(swap_row["swaps_total"] or 0)
                    swaps_with_log_id = int(swap_row["swaps_with_log_id"] or 0)
                joined_row = conn.execute(
                    """
                    SELECT COUNT(DISTINCT l.id) AS joined
                    FROM inference_logs l
                    INNER JOIN swaps s ON s.log_id = l.id
                    """
                ).fetchone()
                logs_joined_from_swaps = int(joined_row["joined"]) if joined_row else 0

            trade_outcomes_total = 0
            logs_with_outcomes = 0
            if has_trade_outcomes_table:
                outcome_row = conn.execute(
                    "SELECT COUNT(*) AS outcomes_total FROM trade_outcomes"
                ).fetchone()
                if outcome_row:
                    trade_outcomes_total = int(outcome_row["outcomes_total"] or 0)
                joined_outcome_row = conn.execute(
                    """
                    SELECT COUNT(DISTINCT l.id) AS joined
                    FROM inference_logs l
                    INNER JOIN trade_outcomes t ON t.log_id = l.id
                    """
                ).fetchone()
                logs_with_outcomes = int(joined_outcome_row["joined"]) if joined_outcome_row else 0

        next_steps: list[str] = []
        if logs_with_full_logs < total_logs:
            missing = total_logs - logs_with_full_logs
            next_steps.append(
                f"Backfill missing full logs ({missing} logs) so prompt/completion context is complete."
            )
        if logs_with_tx_hash > 0 and not has_swaps_table:
            next_steps.append(
                "Implement swaps ingestion (Phase 3). You already have transaction hashes in logs to anchor joins."
            )
        if has_swaps_table and not has_trade_outcomes_table:
            next_steps.append(
                "Implement outcome labeling (Phase 4) so each trade can be mapped to horizon-based PnL labels."
            )
        if has_trade_outcomes_table and logs_with_outcomes < candidate_rows_now:
            next_steps.append(
                "Increase trade_outcomes coverage; some trade-linked logs still do not have labels."
            )
        if not next_steps:
            next_steps.append(
                "Current schema has the key joins; next is tokenizer replay to materialize per-token IDs for activations."
            )

        def _avg(values: list[int]) -> float | None:
            if not values:
                return None
            return round(sum(values) / len(values), 2)

        return {
            "total_logs": total_logs,
            "logs_with_full_logs": logs_with_full_logs,
            "logs_with_tx_hash": logs_with_tx_hash,
            "full_log_coverage_pct": full_log_coverage_pct,
            "candidate_rows_now": candidate_rows_now,
            "candidate_rows_with_outcomes": logs_with_outcomes,
            "full_logs_with_parse_errors": int(core_map.get("full_logs_with_parse_errors") or 0),
            "full_logs_with_prompt_text": int(core_map.get("full_logs_with_prompt_text") or 0),
            "full_logs_with_completion_text": int(core_map.get("full_logs_with_completion_text") or 0),
            "full_logs_with_reasoning_text": int(core_map.get("full_logs_with_reasoning_text") or 0),
            "full_logs_with_prompt_tokens": int(core_map.get("full_logs_with_prompt_tokens") or 0),
            "full_logs_with_completion_tokens": int(core_map.get("full_logs_with_completion_tokens") or 0),
            "full_logs_with_reasoning_tokens": int(core_map.get("full_logs_with_reasoning_tokens") or 0),
            "avg_prompt_tokens": _avg(prompt_values),
            "avg_completion_tokens": _avg(completion_values),
            "avg_reasoning_tokens": _avg(reasoning_values),
            "p50_prompt_tokens": self._percentile(prompt_values, 0.50),
            "p90_prompt_tokens": self._percentile(prompt_values, 0.90),
            "p50_total_tokens": self._percentile(total_values, 0.50),
            "has_swaps_table": has_swaps_table,
            "has_trade_outcomes_table": has_trade_outcomes_table,
            "swaps_total": swaps_total,
            "swaps_with_log_id": swaps_with_log_id,
            "logs_joined_from_swaps": logs_joined_from_swaps,
            "trade_outcomes_total": trade_outcomes_total,
            "logs_with_outcomes": logs_with_outcomes,
            "tool_distribution": tool_distribution,
            "next_steps": next_steps,
        }

    def list_dataset_candidates(self, *, limit: int) -> dict[str, Any]:
        has_swaps_table = self._table_exists("swaps")
        has_trade_outcomes_table = self._table_exists("trade_outcomes")

        swap_join = "LEFT JOIN swaps s ON s.log_id = l.id" if has_swaps_table else ""
        swap_cols = (
            ", s.transaction_hash AS swap_transaction_hash, s.side AS swap_side, s.token_symbol, s.effective_price_usd"
            if has_swaps_table
            else ", NULL AS swap_transaction_hash, NULL AS swap_side, NULL AS token_symbol, NULL AS effective_price_usd"
        )
        outcome_join = "LEFT JOIN trade_outcomes o ON o.log_id = l.id" if has_trade_outcomes_table else ""
        outcome_cols = (
            ", o.pnl_1h_pct, o.pnl_4h_pct, o.pnl_1d_pct, o.was_profitable_1h"
            if has_trade_outcomes_table
            else ", NULL AS pnl_1h_pct, NULL AS pnl_4h_pct, NULL AS pnl_1d_pct, NULL AS was_profitable_1h"
        )

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    l.id, l.vault_address, l.tool, l.strategy_id, l.transaction_hash, l.created_at, l.status,
                    f.prompt_tokens, f.completion_tokens, f.reasoning_tokens, f.total_tokens, f.parse_error,
                    CASE WHEN f.prompt_text IS NOT NULL AND f.prompt_text != '' THEN 1 ELSE 0 END AS has_prompt_text,
                    CASE
                        WHEN (f.completion_text IS NOT NULL AND f.completion_text != '')
                             OR (f.tool_calls_json IS NOT NULL AND f.tool_calls_json != '')
                        THEN 1
                        ELSE 0
                    END AS has_decision_text
                    {swap_cols}
                    {outcome_cols}
                FROM inference_logs l
                INNER JOIN full_logs f ON f.log_id = l.id
                {swap_join}
                {outcome_join}
                WHERE f.parse_error IS NULL OR f.parse_error = ''
                ORDER BY
                    CASE WHEN l.transaction_hash IS NOT NULL AND l.transaction_hash != '' THEN 0 ELSE 1 END,
                    l.id DESC
                LIMIT ?
                """,
                [limit],
            ).fetchall()

        items: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            is_trade = bool(item.get("transaction_hash"))
            item["is_trade"] = is_trade
            if not is_trade:
                item["join_state"] = "NO_TRADE"
            elif has_swaps_table and not item.get("swap_transaction_hash"):
                item["join_state"] = "TX_NOT_JOINED_TO_SWAPS"
            elif has_swaps_table and has_trade_outcomes_table and item.get("pnl_1h_pct") is None:
                item["join_state"] = "SWAP_WITHOUT_OUTCOME"
            elif has_swaps_table and has_trade_outcomes_table:
                item["join_state"] = "READY_FOR_LABEL"
            elif has_swaps_table:
                item["join_state"] = "AWAITING_OUTCOMES"
            else:
                item["join_state"] = "AWAITING_SWAPS"
            items.append(item)
        return {"items": items}

    def _load_payload_preview(self, payload_path: Any) -> str:
        if not payload_path:
            return "No payload path stored for this log."
        path = Path(str(payload_path))
        if not path.exists():
            return f"Payload file is missing: {path}"
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                raw = handle.read()
            if len(raw) > self.payload_preview_chars:
                return raw[: self.payload_preview_chars] + (
                    f"\n\n... truncated to {self.payload_preview_chars} characters ..."
                )
            return raw
        except Exception as exc:  # pragma: no cover - defensive
            return f"Failed to read payload: {type(exc).__name__}: {exc}"


def _parse_int(query: dict[str, list[str]], key: str, default: int, minimum: int, maximum: int) -> int:
    raw = query.get(key, [str(default)])[0]
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _make_handler(store: ExplorerStore) -> type[BaseHTTPRequestHandler]:
    class ExplorerHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_not_found(self) -> None:
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def _send_bad_request(self, message: str) -> None:
            self._send_json({"error": message}, status=HTTPStatus.BAD_REQUEST)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path == "/":
                self._send_html(HTML_PAGE)
                return

            if path == "/api/overview":
                self._send_json(store.get_overview())
                return

            if path == "/api/dataset-readiness":
                self._send_json(store.get_dataset_readiness())
                return

            if path == "/api/dataset-candidates":
                limit = _parse_int(query, "limit", 120, 1, 1000)
                self._send_json(store.list_dataset_candidates(limit=limit))
                return

            if path == "/api/vaults":
                limit = _parse_int(query, "limit", 100, 1, 500)
                offset = _parse_int(query, "offset", 0, 0, 1_000_000)
                q = query.get("q", [""])[0].strip() or None
                self._send_json(store.list_vaults(limit=limit, offset=offset, q=q))
                return

            if path.startswith("/api/vault/"):
                vault_address = path.removeprefix("/api/vault/").strip()
                if not vault_address:
                    self._send_bad_request("vault address is required")
                    return
                self._send_json(store.get_vault_detail(vault_address))
                return

            if path.startswith("/api/log/"):
                raw_log_id = path.removeprefix("/api/log/").strip()
                try:
                    log_id = int(raw_log_id)
                except ValueError:
                    self._send_bad_request("log id must be an integer")
                    return
                include_payload = query.get("include_payload", ["0"])[0] == "1"
                self._send_json(store.get_log_detail(log_id, include_payload=include_payload))
                return

            self._send_not_found()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    return ExplorerHandler


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local UI explorer for ingested terminal data")
    parser.add_argument("--db-path", type=Path, default=Path("data/terminal_ingest.db"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--payload-preview-chars", type=int, default=12000)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    db_path: Path = args.db_path
    if not db_path.exists():
        print(f"Database file not found: {db_path}")
        print("Run ingestion first, e.g. uv run -m pipelines.ingest --top-n 3")
        return 1

    store = ExplorerStore(db_path=db_path, payload_preview_chars=args.payload_preview_chars)
    handler = _make_handler(store)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Explorer running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
