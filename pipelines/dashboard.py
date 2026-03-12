"""Xenon Pipeline Dashboard — manage all 4 phases from a single UI.

Serves a local web app for monitoring pipeline status, browsing data,
viewing analysis results, and running pipeline commands.

Usage:
    uv run -m pipelines.dashboard
    uv run -m pipelines.dashboard --port 8800
    uv run -m pipelines.dashboard --db-path data/terminal_ingest.db
"""

from __future__ import annotations

import argparse
import json
import os
import selectors
import sqlite3
import subprocess
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# HTML page (inline SPA)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Xenon Pipeline Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #f5f3ee;
      --bg-subtle: #eae7df;
      --surface: rgba(255,255,255,0.88);
      --surface-raised: rgba(255,255,255,0.96);
      --border: rgba(31,36,48,0.12);
      --border-strong: rgba(31,36,48,0.22);
      --text: #1a1d23;
      --text-dim: #5b6170;
      --text-faint: #8b8f9a;
      --accent: #0f766e;
      --accent-dim: rgba(15,118,110,0.12);
      --accent-2: #b45309;
      --accent-2-dim: rgba(180,83,9,0.10);
      --danger: #be123c;
      --danger-dim: rgba(190,18,60,0.10);
      --success: #16a34a;
      --success-dim: rgba(22,163,74,0.10);
      --warn: #ca8a04;
      --warn-dim: rgba(202,138,4,0.10);
      --mono: "Fira Code", "SF Mono", Menlo, monospace;
      --sans: "DM Sans", "Avenir Next", system-ui, sans-serif;
      --shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
      --shadow: 0 4px 16px rgba(0,0,0,0.08);
      --radius: 10px;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #141519;
        --bg-subtle: #1c1d24;
        --surface: rgba(30,32,40,0.92);
        --surface-raised: rgba(40,42,52,0.96);
        --border: rgba(255,255,255,0.08);
        --border-strong: rgba(255,255,255,0.16);
        --text: #e4e5ea;
        --text-dim: #8b8f9a;
        --text-faint: #5b5f6a;
        --accent: #2dd4bf;
        --accent-dim: rgba(45,212,191,0.12);
        --accent-2: #f59e0b;
        --accent-2-dim: rgba(245,158,11,0.10);
        --danger: #f43f5e;
        --danger-dim: rgba(244,63,94,0.10);
        --success: #4ade80;
        --success-dim: rgba(74,222,128,0.10);
        --warn: #facc15;
        --warn-dim: rgba(250,204,21,0.10);
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.2);
        --shadow: 0 4px 16px rgba(0,0,0,0.3);
      }
    }
    * { box-sizing: border-box; margin: 0; }
    body {
      font-family: var(--sans);
      color: var(--text);
      background: var(--bg);
      min-height: 100vh;
      line-height: 1.5;
    }
    .wrap { max-width: 1320px; margin: 0 auto; padding: 16px 20px; }

    /* Header */
    .header {
      display: flex; justify-content: space-between; align-items: center;
      padding: 12px 0 16px; border-bottom: 1px solid var(--border);
      margin-bottom: 16px;
    }
    .header h1 { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.01em; }
    .header-sub { color: var(--text-dim); font-size: 0.85rem; margin-top: 2px; }
    .header-actions { display: flex; gap: 8px; }

    /* Buttons */
    .btn {
      display: inline-flex; align-items: center; gap: 6px;
      border: 1px solid var(--border-strong); border-radius: 8px;
      background: var(--surface); padding: 7px 14px;
      font: 500 0.82rem var(--sans); color: var(--text);
      cursor: pointer; transition: all 120ms ease;
    }
    .btn:hover { background: var(--surface-raised); transform: translateY(-1px); box-shadow: var(--shadow-sm); }
    .btn--accent { background: var(--accent-dim); border-color: var(--accent); color: var(--accent); }
    .btn--accent:hover { background: var(--accent); color: white; }
    .btn--sm { padding: 4px 10px; font-size: 0.78rem; }
    .btn--danger { background: var(--danger-dim); border-color: var(--danger); color: var(--danger); }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

    /* Phase status strip */
    .phases {
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
      margin-bottom: 16px;
    }
    .phase-card {
      border: 1px solid var(--border); border-radius: var(--radius);
      background: var(--surface); padding: 14px 16px;
      box-shadow: var(--shadow-sm); cursor: pointer;
      transition: all 150ms ease; position: relative;
    }
    .phase-card:hover { box-shadow: var(--shadow); transform: translateY(-2px); }
    .phase-card.active { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent), var(--shadow); }
    .phase-card .phase-num {
      font: 600 0.68rem var(--mono); text-transform: uppercase; letter-spacing: 0.08em;
      color: var(--text-dim); margin-bottom: 4px;
    }
    .phase-card .phase-title { font-weight: 600; font-size: 0.95rem; margin-bottom: 6px; }
    .phase-card .phase-stat {
      font: 500 1.3rem var(--mono); font-variant-numeric: tabular-nums;
    }
    .phase-card .phase-label {
      font-size: 0.75rem; color: var(--text-dim); margin-top: 2px;
    }
    .dot {
      display: inline-block; width: 8px; height: 8px; border-radius: 50%;
      margin-right: 6px; vertical-align: middle;
    }
    .dot--green { background: var(--success); }
    .dot--yellow { background: var(--warn); }
    .dot--red { background: var(--danger); }
    .dot--gray { background: var(--text-faint); }

    /* Tabs (phase detail) */
    .tab-content { display: none; }
    .tab-content.visible { display: block; }

    /* Panels */
    .panel {
      border: 1px solid var(--border); border-radius: var(--radius);
      background: var(--surface); box-shadow: var(--shadow-sm);
      overflow: hidden; margin-bottom: 14px;
    }
    .panel-head {
      display: flex; justify-content: space-between; align-items: center;
      padding: 10px 16px; border-bottom: 1px solid var(--border);
      background: var(--surface-raised);
    }
    .panel-head h3 { font-size: 0.9rem; font-weight: 600; }
    .panel-body { padding: 14px 16px; }

    /* Stat cards row */
    .stats-row {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 10px; margin-bottom: 14px;
    }
    .stat-card {
      border: 1px solid var(--border); border-radius: 8px;
      background: var(--surface); padding: 10px 14px;
    }
    .stat-card .stat-label {
      font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.06em; color: var(--text-dim);
    }
    .stat-card .stat-value {
      font: 600 1.2rem var(--mono); font-variant-numeric: tabular-nums;
      margin-top: 4px;
    }

    /* Tables */
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    th, td {
      text-align: left; padding: 8px 10px;
      border-bottom: 1px solid var(--border);
      font-variant-numeric: tabular-nums;
    }
    th { color: var(--text-dim); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }
    tr:hover { background: var(--accent-dim); }

    /* Mono */
    .mono { font-family: var(--mono); font-size: 0.82rem; }

    /* Code / command blocks */
    .cmd-block {
      background: var(--bg-subtle); border: 1px solid var(--border);
      border-radius: 8px; padding: 10px 14px;
      font: 400 0.82rem var(--mono); white-space: pre-wrap;
      word-break: break-word; overflow: auto; max-height: 400px;
    }

    /* Badge / chip */
    .badge {
      display: inline-block; padding: 2px 8px; border-radius: 999px;
      font: 500 0.72rem var(--mono); letter-spacing: 0.02em;
    }
    .badge--green { background: var(--success-dim); color: var(--success); }
    .badge--yellow { background: var(--warn-dim); color: var(--warn); }
    .badge--red { background: var(--danger-dim); color: var(--danger); }
    .badge--dim { background: var(--bg-subtle); color: var(--text-dim); }

    /* Chart container */
    .chart-wrap { position: relative; height: 320px; margin: 8px 0; }

    /* Grid layouts */
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }

    /* PCA gallery */
    .pca-gallery {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px;
    }
    .pca-gallery img {
      width: 100%; border-radius: 8px; border: 1px solid var(--border);
      cursor: pointer; transition: transform 120ms;
    }
    .pca-gallery img:hover { transform: scale(1.02); }

    /* Run log */
    .run-log {
      background: #0d1117; color: #c9d1d9; border-radius: 8px;
      padding: 12px 16px; font: 400 0.78rem var(--mono);
      white-space: pre-wrap; word-break: break-word;
      max-height: 500px; overflow: auto; line-height: 1.6;
    }
    .run-log .run-err { color: #f85149; }
    .run-log .run-ok { color: #56d364; }

    /* Empty state */
    .empty { color: var(--text-faint); font-style: italic; padding: 20px; text-align: center; }

    /* Responsive */
    @media (max-width: 900px) {
      .phases { grid-template-columns: repeat(2, 1fr); }
      .grid-2, .grid-3 { grid-template-columns: 1fr; }
    }
    @media (max-width: 600px) {
      .phases { grid-template-columns: 1fr; }
      .stats-row { grid-template-columns: repeat(2, 1fr); }
    }

    /* Animations */
    @keyframes fadeUp { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
    .phase-card, .stat-card, .panel { animation: fadeUp 300ms ease-out; }
  </style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div>
      <h1>Xenon Pipeline</h1>
      <div class="header-sub">Ingest → Data Prep → Activation Capture → Analysis</div>
    </div>
    <div class="header-actions">
      <button class="btn" onclick="refreshAll(true)">Refresh</button>
    </div>
  </div>

  <!-- Phase status strip -->
  <section class="phases" id="phases"></section>

  <!-- Phase detail tabs -->
  <div id="tab-ingest" class="tab-content"></div>
  <div id="tab-prep" class="tab-content"></div>
  <div id="tab-capture" class="tab-content"></div>
  <div id="tab-analysis" class="tab-content"></div>
</div>

<script>
// ---------- State ----------
let currentTab = "ingest";
let statusData = null;

function fmt(n) {
  if (n === null || n === undefined) return "\u2014";
  if (typeof n === "number") return n.toLocaleString();
  return String(n);
}

function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

// ---------- Phase cards ----------
function selectTab(tab) {
  currentTab = tab;
  document.querySelectorAll(".phase-card").forEach(c => c.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("visible"));
  const card = document.querySelector(`.phase-card[data-phase="${tab}"]`);
  if (card) card.classList.add("active");
  const content = document.getElementById(`tab-${tab}`);
  if (content) content.classList.add("visible");
}

function dotClass(status) {
  if (status === "ready") return "dot--green";
  if (status === "partial") return "dot--yellow";
  if (status === "empty") return "dot--gray";
  return "dot--red";
}

function renderPhaseCards(s) {
  const phases = [
    { key: "ingest", num: "Phase 1", title: "Ingest", stat: fmt(s.ingest.log_count), label: "inference logs", status: s.ingest.status },
    { key: "prep", num: "Phase 2", title: "Data Prep", stat: fmt(s.prep.total_examples), label: "interp examples", status: s.prep.status },
    { key: "capture", num: "Phase 3", title: "Capture", stat: fmt(s.capture.total_files), label: "activations", status: s.capture.status },
    { key: "analysis", num: "Phase 4", title: "Analysis", stat: fmt(s.analysis.total_results), label: "result files", status: s.analysis.status },
  ];
  document.getElementById("phases").innerHTML = phases.map(p => `
    <div class="phase-card${currentTab === p.key ? ' active' : ''}" data-phase="${p.key}" onclick="selectTab('${p.key}'); loadTab('${p.key}');">
      <div class="phase-num"><span class="dot ${dotClass(p.status)}"></span>${p.num}</div>
      <div class="phase-title">${p.title}</div>
      <div class="phase-stat">${p.stat}</div>
      <div class="phase-label">${p.label} &middot; <span class="badge badge--${p.status === 'ready' ? 'green' : p.status === 'partial' ? 'yellow' : 'dim'}">${p.status}</span></div>
    </div>
  `).join("");
}

// ---------- Ingest tab ----------
function renderIngest(data) {
  const el = document.getElementById("tab-ingest");
  const s = data;
  el.innerHTML = `
    <div class="stats-row">
      ${[["Vaults", s.vault_count], ["Strategies", s.strategy_count], ["Inference Logs", s.log_count],
         ["Full Logs", s.full_log_count], ["Full Log Coverage", s.full_log_coverage_pct + "%"],
         ["Parse Errors", s.parse_error_count]].map(([l,v]) =>
        `<div class="stat-card"><div class="stat-label">${l}</div><div class="stat-value">${fmt(v)}</div></div>`
      ).join("")}
    </div>
    <div class="grid-2">
      <div class="panel">
        <div class="panel-head"><h3>Commands</h3></div>
        <div class="panel-body">
          <p style="color:var(--text-dim);font-size:0.82rem;margin-bottom:10px;">Run these from the project root:</p>
          <div class="cmd-block"># Ingest top N vaults
uv run -m pipelines.ingest --top-n 3

# With options
uv run -m pipelines.ingest --top-n 10 --request-concurrency 20

# Browse data
uv run -m pipelines.ingest.explorer --port 8765</div>
          <div style="margin-top:10px;">
            <button class="btn btn--accent btn--sm" onclick="runCmd('ingest', 'uv run -m pipelines.ingest --top-n 3')">Run Ingest (top 3)</button>
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head"><h3>Database Tables</h3></div>
        <div class="panel-body">
          <table>
            <thead><tr><th>Table</th><th>Rows</th></tr></thead>
            <tbody>
              ${(s.tables || []).map(t => `<tr><td class="mono">${esc(t.name)}</td><td class="mono">${fmt(t.count)}</td></tr>`).join("")}
            </tbody>
          </table>
          ${!s.tables || !s.tables.length ? '<div class="empty">No database found. Run ingest first.</div>' : ''}
        </div>
      </div>
    </div>
  `;
}

// ---------- Prep tab ----------
function renderPrep(data) {
  const el = document.getElementById("tab-prep");
  const s = data;
  el.innerHTML = `
    <div class="stats-row">
      ${[["Total Examples", s.total_examples], ["High Quality", s.high_quality], ["Medium Quality", s.medium_quality],
         ["Low Quality", s.low_quality], ["Trades", s.trade_count], ["Observations", s.observation_count],
         ["Parquet Exported", s.parquet_exported ? "Yes" : "No"]].map(([l,v]) =>
        `<div class="stat-card"><div class="stat-label">${l}</div><div class="stat-value">${fmt(v)}</div></div>`
      ).join("")}
    </div>
    <div class="grid-2">
      <div class="panel">
        <div class="panel-head"><h3>Commands</h3></div>
        <div class="panel-body">
          <div class="cmd-block"># Build interp dataset + export parquet
uv run -m pipelines.interp.prepare \\
  --db-path data/terminal_ingest.db \\
  --export-parquet

# Optional: compute trade outcomes
uv run -m pipelines.interp.outcomes \\
  --db-path data/terminal_ingest.db</div>
          <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
            <button class="btn btn--accent btn--sm" onclick="runCmd('prep', 'uv run -m pipelines.interp.prepare --db-path data/terminal_ingest.db --export-parquet')">Run Prep + Export</button>
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head"><h3>Export Files</h3></div>
        <div class="panel-body">
          <table>
            <thead><tr><th>File</th><th>Size</th></tr></thead>
            <tbody>
              ${(s.export_files || []).map(f => `<tr><td class="mono">${esc(f.name)}</td><td class="mono">${f.size}</td></tr>`).join("")}
            </tbody>
          </table>
          ${!s.export_files || !s.export_files.length ? '<div class="empty">No exports yet. Run data prep with --export-parquet.</div>' : ''}
        </div>
      </div>
    </div>
    ${s.label_distribution && s.label_distribution.length ? `
    <div class="panel">
      <div class="panel-head"><h3>Label Distribution</h3></div>
      <div class="panel-body">
        <table>
          <thead><tr><th>Decision Type</th><th>Count</th><th>Trade Side</th><th>Avg Risk Pref</th></tr></thead>
          <tbody>${s.label_distribution.map(r => `
            <tr>
              <td>${esc(r.decision_type || '—')}</td>
              <td class="mono">${fmt(r.count)}</td>
              <td>${esc(r.trade_side || '—')}</td>
              <td class="mono">${r.avg_risk !== null ? r.avg_risk.toFixed(1) : '—'}</td>
            </tr>`).join("")}
          </tbody>
        </table>
      </div>
    </div>` : ''}
  `;
}

// ---------- Capture tab ----------
function renderCapture(data) {
  const el = document.getElementById("tab-capture");
  const s = data;
  el.innerHTML = `
    <div class="stats-row">
      ${[["Residual Files", s.residual_count], ["Router Files", s.router_count],
         ["Total Size", s.total_size_mb + " MB"], ["Avg Seq Len", s.avg_seq_len],
         ["Layers Captured", s.num_layers], ["Hidden Dim", s.hidden_dim],
         ["Num Experts", s.num_experts || "N/A (dense)"]].map(([l,v]) =>
        `<div class="stat-card"><div class="stat-label">${l}</div><div class="stat-value">${fmt(v)}</div></div>`
      ).join("")}
    </div>
    <div class="grid-2">
      <div class="panel">
        <div class="panel-head"><h3>Local Capture</h3></div>
        <div class="panel-body">
          <div class="cmd-block"># Validate tokenization
uv run --extra interp -m pipelines.interp.capture --validate-tokens

# Capture (local, Qwen3-8B)
uv run --extra interp -m pipelines.interp.capture \\
  --limit 5 --layers 0,12,24,35

# Resume
uv run --extra interp -m pipelines.interp.capture --skip-existing</div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head"><h3>Modal Capture (Qwen3-30B-A3B)</h3></div>
        <div class="panel-body">
          <div class="cmd-block"># One-time setup
modal token new
modal secret create huggingface HF_TOKEN=&lt;token&gt;
./scripts/modal_capture.sh download

# Capture
./scripts/modal_capture.sh router          # router only
./scripts/modal_capture.sh full            # residual + router
./scripts/modal_capture.sh router --limit 10

# Inspect / download
./scripts/modal_capture.sh inspect
./scripts/modal_capture.sh download-activations</div>
          <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
            <button class="btn btn--accent btn--sm" onclick="runCmd('capture', './scripts/modal_capture.sh inspect')">Inspect Volume</button>
            <button class="btn btn--sm" onclick="runCmd('capture', './scripts/modal_capture.sh meta')">Show Metadata</button>
          </div>
        </div>
      </div>
    </div>
    ${s.recent_captures && s.recent_captures.length ? `
    <div class="panel">
      <div class="panel-head"><h3>Recent Captures (metadata.parquet)</h3></div>
      <div class="panel-body" style="overflow-x:auto;">
        <table>
          <thead><tr><th>Log ID</th><th>Seq Len</th><th>Size (MB)</th><th>Time (s)</th><th>Router</th><th>Layers</th><th>Timestamp</th></tr></thead>
          <tbody>${s.recent_captures.map(r => `
            <tr>
              <td class="mono">${fmt(r.log_id)}</td>
              <td class="mono">${fmt(r.seq_len)}</td>
              <td class="mono">${(r.file_size_bytes / 1024 / 1024).toFixed(1)}</td>
              <td class="mono">${r.elapsed_s}</td>
              <td>${r.has_router ? '<span class="badge badge--green">yes</span>' : '<span class="badge badge--dim">no</span>'}</td>
              <td class="mono">${fmt(r.num_layers_captured)}</td>
              <td class="mono" style="font-size:0.75rem;">${esc(r.capture_timestamp || '—')}</td>
            </tr>`).join("")}
          </tbody>
        </table>
      </div>
    </div>` : ''}
  `;
}

// ---------- Analysis tab ----------
let probeChart = null;

function renderAnalysis(data) {
  const el = document.getElementById("tab-analysis");
  const s = data;

  const probeFiles = (s.probe_files || []);
  const expertFile = s.has_expert_specialization;
  const pcaImages = (s.pca_images || []);

  el.innerHTML = `
    <div class="stats-row">
      ${[["Probe Results", probeFiles.length], ["Expert Analysis", expertFile ? "Yes" : "No"],
         ["PCA Plots", pcaImages.length], ["Total Results", s.total_results]].map(([l,v]) =>
        `<div class="stat-card"><div class="stat-label">${l}</div><div class="stat-value">${fmt(v)}</div></div>`
      ).join("")}
    </div>

    <div class="grid-2">
      <div class="panel">
        <div class="panel-head"><h3>Run Analysis</h3></div>
        <div class="panel-body">
          <div class="cmd-block"># On Modal (primary — no download needed)
./scripts/modal_capture.sh analyze --mode probe --target decision_type
./scripts/modal_capture.sh analyze --mode all --target decision_type
./scripts/modal_capture.sh analyze --mode probe --target risk_tolerance

# Download results
./scripts/modal_capture.sh download-results

# Or locally (after downloading activations)
uv run --extra analysis -m pipelines.interp.analysis \\
  --mode all --target decision_type</div>
          <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
            <button class="btn btn--accent btn--sm" onclick="runLocalAnalysis('probe', 'decision_type')">Probe: decision_type</button>
            <button class="btn btn--sm" onclick="runLocalAnalysis('probe', 'trade_side')">Probe: trade_side</button>
            <button class="btn btn--sm" onclick="runLocalAnalysis('probe', 'risk_tolerance')">Probe: risk_tolerance</button>
            <button class="btn btn--sm" onclick="runLocalAnalysis('all', 'decision_type')">All: decision_type</button>
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head"><h3>Available Results</h3></div>
        <div class="panel-body">
          <table>
            <thead><tr><th>File</th><th>Size</th></tr></thead>
            <tbody>
              ${(s.result_files || []).map(f =>
                `<tr style="cursor:pointer;" onclick="loadProbeResult('${esc(f.name)}')">
                  <td class="mono">${esc(f.name)}</td><td class="mono">${f.size}</td>
                </tr>`
              ).join("")}
            </tbody>
          </table>
          ${!s.result_files || !s.result_files.length ? '<div class="empty">No results yet. Run analysis first.</div>' : ''}
        </div>
      </div>
    </div>

    <div class="panel" id="probe-panel" style="display:none;">
      <div class="panel-head"><h3 id="probe-title">Probe Results</h3></div>
      <div class="panel-body">
        <div class="chart-wrap"><canvas id="probeChart"></canvas></div>
        <div id="probe-table-wrap" style="margin-top:14px; overflow-x:auto;"></div>
      </div>
    </div>

    ${expertFile ? `
    <div class="panel">
      <div class="panel-head">
        <h3>Expert Specialization</h3>
        <button class="btn btn--sm" onclick="loadExperts()">Load</button>
      </div>
      <div class="panel-body" id="expert-body">
        <div class="empty">Click Load to view expert specialization data.</div>
      </div>
    </div>` : ''}

    ${pcaImages.length ? `
    <div class="panel">
      <div class="panel-head"><h3>PCA Visualizations</h3></div>
      <div class="panel-body">
        <div class="pca-gallery">
          ${pcaImages.map(img => `<img src="/api/analysis/pca/${esc(img)}" alt="${esc(img)}" title="${esc(img)}" />`).join("")}
        </div>
      </div>
    </div>` : ''}
  `;
}

async function loadProbeResult(filename) {
  if (!filename.endsWith('.parquet')) return;
  try {
    const data = await api(`/api/analysis/probe-data?file=${encodeURIComponent(filename)}`);
    if (!data.rows || !data.rows.length) return;

    document.getElementById("probe-panel").style.display = "block";
    document.getElementById("probe-title").textContent = `Probe: ${filename.replace('.parquet','')}`;

    const rows = data.rows;
    const layers = rows.map(r => r.layer);
    const accs = rows.map(r => r.accuracy_mean);
    const balAccs = rows.map(r => r.balanced_accuracy);
    const baselines = rows.map(r => r.baseline_majority);
    const shuffled = rows.map(r => r.baseline_shuffled);

    if (probeChart) probeChart.destroy();
    const ctx = document.getElementById("probeChart").getContext("2d");
    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    probeChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: layers,
        datasets: [
          { label: "Accuracy", data: accs, borderColor: "#0f766e", backgroundColor: "rgba(15,118,110,0.1)", fill: true, tension: 0.3, pointRadius: 3 },
          { label: "Balanced Acc", data: balAccs, borderColor: "#b45309", backgroundColor: "transparent", borderDash: [4,4], tension: 0.3, pointRadius: 2 },
          { label: "Majority Baseline", data: baselines, borderColor: isDark ? "#555" : "#bbb", backgroundColor: "transparent", borderDash: [8,4], pointRadius: 0 },
          { label: "Shuffled Control", data: shuffled, borderColor: isDark ? "#444" : "#ccc", backgroundColor: "transparent", borderDash: [2,2], pointRadius: 0 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: "Layer" }, ticks: { color: isDark ? '#888' : '#666' }, grid: { color: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)' } },
          y: { title: { display: true, text: "Score" }, min: 0, max: 1, ticks: { color: isDark ? '#888' : '#666' }, grid: { color: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.06)' } },
        },
        plugins: {
          legend: { labels: { color: isDark ? '#ccc' : '#333', usePointStyle: true, pointStyle: 'line' } },
          tooltip: { mode: 'index', intersect: false },
        }
      }
    });

    // Table
    document.getElementById("probe-table-wrap").innerHTML = `
      <table>
        <thead><tr><th>Layer</th><th>Acc</th><th>Bal Acc</th><th>Selectivity</th><th>Majority</th><th>Shuffled</th><th>N</th></tr></thead>
        <tbody>${rows.map(r => `
          <tr${r.selectivity > 0.05 ? ' style="background:var(--accent-dim);"' : ''}>
            <td class="mono">${r.layer}</td>
            <td class="mono">${r.accuracy_mean.toFixed(3)} ±${r.accuracy_std.toFixed(3)}</td>
            <td class="mono">${r.balanced_accuracy.toFixed(3)}</td>
            <td class="mono" style="font-weight:${r.selectivity > 0.05 ? '700' : '400'};">${r.selectivity > 0 ? '+' : ''}${r.selectivity.toFixed(3)}</td>
            <td class="mono">${r.baseline_majority.toFixed(3)}</td>
            <td class="mono">${r.baseline_shuffled.toFixed(3)}</td>
            <td class="mono">${r.n_examples}</td>
          </tr>`).join("")}
        </tbody>
      </table>
    `;
  } catch(e) { console.error(e); }
}

async function loadExperts() {
  try {
    const data = await api("/api/analysis/expert-data");
    if (!data.rows || !data.rows.length) {
      document.getElementById("expert-body").innerHTML = '<div class="empty">No expert data found.</div>';
      return;
    }
    // Group by layer, show top 5 per layer
    const byLayer = {};
    for (const r of data.rows) {
      if (!byLayer[r.layer]) byLayer[r.layer] = [];
      if (byLayer[r.layer].length < 5) byLayer[r.layer].push(r);
    }
    const layers = Object.keys(byLayer).sort((a,b) => a - b);
    document.getElementById("expert-body").innerHTML = `
      <table>
        <thead><tr><th>Layer</th><th>Expert ID</th><th>Rank</th><th>Discriminative Score</th></tr></thead>
        <tbody>${layers.flatMap(l => byLayer[l].map(r => `
          <tr>
            <td class="mono">${r.layer}</td>
            <td class="mono">${r.expert_id}</td>
            <td class="mono">${r.rank}</td>
            <td class="mono" style="font-weight:${Math.abs(r.discriminative_score) > 1 ? '700' : '400'};">${r.discriminative_score > 0 ? '+' : ''}${r.discriminative_score.toFixed(3)}</td>
          </tr>`)).join("")}
        </tbody>
      </table>
    `;
  } catch(e) { console.error(e); }
}

function runLocalAnalysis(mode, target) {
  runCmd('analysis', `uv run --extra analysis -m pipelines.interp.analysis --mode ${mode} --target ${target} --data-source router`);
}

// ---------- Command runner ----------
let runnerEl = null;
function runCmd(phase, cmd) {
  // Create or reuse a floating run log
  if (!runnerEl) {
    runnerEl = document.createElement("div");
    runnerEl.className = "panel";
    runnerEl.style.cssText = "position:fixed;bottom:16px;right:16px;width:600px;max-width:90vw;z-index:100;box-shadow:0 8px 32px rgba(0,0,0,0.3);";
    document.body.appendChild(runnerEl);
  }
  runnerEl.innerHTML = `
    <div class="panel-head">
      <h3>Running command...</h3>
      <button class="btn btn--sm btn--danger" onclick="this.closest('.panel').remove(); runnerEl=null;">Close</button>
    </div>
    <div class="panel-body">
      <div class="cmd-block" style="margin-bottom:8px;">$ ${esc(cmd)}</div>
      <div class="run-log" id="runLog">Starting...</div>
    </div>
  `;
  const logEl = document.getElementById("runLog");

  fetch("/api/run", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({command: cmd}),
  }).then(r => r.json()).then(data => {
    let html = "";
    if (data.stdout) html += esc(data.stdout);
    if (data.stderr) html += `\n<span class="run-err">${esc(data.stderr)}</span>`;
    html += `\n<span class="${data.returncode === 0 ? 'run-ok' : 'run-err'}">\nExit code: ${data.returncode}</span>`;
    logEl.innerHTML = html;
    // Refresh after run
    setTimeout(() => refreshAll(true), 500);
  }).catch(err => {
    logEl.innerHTML = `<span class="run-err">Error: ${esc(err.message)}</span>`;
  });
}

// ---------- Load tabs ----------
async function loadTab(tab) {
  try {
    if (tab === "ingest") {
      const data = await api("/api/ingest");
      renderIngest(data);
    } else if (tab === "prep") {
      const data = await api("/api/prep");
      renderPrep(data);
    } else if (tab === "capture") {
      const data = await api("/api/capture");
      renderCapture(data);
    } else if (tab === "analysis") {
      const data = await api("/api/analysis");
      renderAnalysis(data);
      // Auto-load first probe file if available
      if (data.probe_files && data.probe_files.length) {
        loadProbeResult(data.probe_files[0]);
      }
    }
  } catch(e) { console.error("loadTab error:", e); }
}

async function refreshAll(force = false) {
  try {
    statusData = await api(force ? "/api/status?refresh=1" : "/api/status");
    renderPhaseCards(statusData);
    await loadTab(currentTab);
  } catch(e) { console.error("refreshAll error:", e); }
}

// Boot
refreshAll();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Backend store
# ---------------------------------------------------------------------------

class _ModalStatsCache:
    """Cache for Modal DB stats. Downloads a small JSON from the volume."""

    def __init__(self, ttl_s: float = 300) -> None:
        self.ttl_s = ttl_s
        self._data: dict[str, Any] | None = None
        self._fetched_at: float = 0
        self._lock = threading.Lock()
        self._fetching = False

    def get(self) -> dict[str, Any] | None:
        if self._data and (time.monotonic() - self._fetched_at) < self.ttl_s:
            return self._data
        # Try local file first (already downloaded)
        self._try_load_local()
        # Trigger background download from volume
        self._start_fetch()
        return self._data

    def invalidate(self) -> None:
        """Force re-fetch on next get()."""
        self._fetched_at = 0

    def force_refresh(self, timeout_s: float = 45.0) -> dict[str, Any] | None:
        """Refresh stats synchronously for explicit UI refresh actions."""
        self.invalidate()

        # If another request is already fetching, wait briefly for it.
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                fetching = self._fetching
                if not fetching:
                    self._fetching = True
                    break
            if time.monotonic() >= deadline:
                self._try_load_local()
                return self._data
            time.sleep(0.05)

        try:
            # Explicit refresh should recompute snapshot on Modal first.
            self._fetch_once(recompute=True)
        finally:
            with self._lock:
                self._fetching = False
        return self._data

    def _try_load_local(self) -> None:
        try:
            stats_path = Path("data/dashboard_stats.json")
            if stats_path.exists():
                self._data = json.loads(stats_path.read_text())
                if self._fetched_at == 0:
                    self._fetched_at = time.monotonic()
        except Exception:
            pass

    def _start_fetch(self) -> None:
        with self._lock:
            if self._fetching:
                return
            self._fetching = True
        t = threading.Thread(target=self._fetch, daemon=True)
        t.start()

    def _fetch(self) -> None:
        try:
            # Background refresh only downloads the latest existing snapshot.
            self._fetch_once(recompute=False)
        finally:
            with self._lock:
                self._fetching = False

    def _fetch_once(self, recompute: bool) -> None:
        try:
            cmd = ["./scripts/modal_capture.sh", "modal-snapshot"] if recompute else [
                "./scripts/modal_capture.sh",
                "modal-stats",
            ]
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60 if recompute else 30,
            )
            stats_path = Path("data/dashboard_stats.json")
            if stats_path.exists():
                self._data = json.loads(stats_path.read_text())
                self._fetched_at = time.monotonic()
        except Exception as exc:
            print(f"[modal-stats] fetch failed: {exc}")


_modal_stats = _ModalStatsCache()


class DashboardStore:
    def __init__(self, db_path: Path, data_dir: Path) -> None:
        self.db_path = db_path
        self.data_dir = data_dir
        self.activations_dir = data_dir / "activations"
        self.exports_dir = data_dir / "interp_exports"
        self.results_dir = data_dir / "analysis_results"

    def _connect(self) -> sqlite3.Connection | None:
        if not self.db_path.exists():
            return None
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _table_exists(self, conn: sqlite3.Connection, name: str) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", [name]
        ).fetchone()
        return row is not None

    def _table_count(self, conn: sqlite3.Connection, name: str) -> int:
        if not self._table_exists(conn, name):
            return 0
        row = conn.execute(f"SELECT COUNT(*) AS n FROM [{name}]").fetchone()
        return int(row["n"]) if row else 0

    # --- Status ---

    def get_status(self) -> dict[str, Any]:
        conn = self._connect()
        modal = _modal_stats.get()

        # Ingest
        ingest = {"log_count": 0, "status": "empty"}
        if modal and modal.get("ingest"):
            lc = modal["ingest"].get("log_count", 0)
            ingest["log_count"] = lc
            ingest["status"] = "ready" if lc > 0 else "empty"
        elif conn:
            lc = self._table_count(conn, "inference_logs")
            ingest["log_count"] = lc
            ingest["status"] = "ready" if lc > 0 else "empty"

        # Prep
        prep = {"total_examples": 0, "status": "empty"}
        if modal and modal.get("prep"):
            tc = modal["prep"].get("total_examples", 0)
            prep["total_examples"] = tc
            has_exports = len(modal["prep"].get("export_files", [])) > 0
            if tc > 0 and has_exports:
                prep["status"] = "ready"
            elif tc > 0:
                prep["status"] = "partial"
        elif conn and self._table_exists(conn, "interp_examples_v0"):
            tc = self._table_count(conn, "interp_examples_v0")
            prep["total_examples"] = tc
            hq = self.exports_dir / "interp_examples_v0_high_quality.parquet"
            if tc > 0 and hq.exists():
                prep["status"] = "ready"
            elif tc > 0:
                prep["status"] = "partial"

        # Capture
        capture = {"total_files": 0, "status": "empty"}
        meta = self.activations_dir / "metadata.parquet"
        if meta.exists():
            try:
                import pyarrow.parquet as pq
                t = pq.read_table(meta)
                capture["total_files"] = t.num_rows
                capture["status"] = "ready" if t.num_rows > 0 else "empty"
            except Exception:
                pass
        else:
            # Count safetensors files directly
            res_dir = self.activations_dir / "residual_stream"
            rtr_dir = self.activations_dir / "router_logits"
            rc = len(list(res_dir.glob("*.safetensors"))) if res_dir.exists() else 0
            rtc = len(list(rtr_dir.glob("*.safetensors"))) if rtr_dir.exists() else 0
            capture["total_files"] = rc + rtc
            if rc + rtc > 0:
                capture["status"] = "partial"

        # Analysis
        analysis = {"total_results": 0, "status": "empty"}
        if self.results_dir.exists():
            files = list(self.results_dir.glob("*"))
            analysis["total_results"] = len(files)
            analysis["status"] = "ready" if files else "empty"

        if conn:
            conn.close()

        return {
            "ingest": ingest,
            "prep": prep,
            "capture": capture,
            "analysis": analysis,
        }

    # --- Ingest detail ---

    def get_ingest(self) -> dict[str, Any]:
        empty = {
            "vault_count": 0, "strategy_count": 0, "log_count": 0,
            "full_log_count": 0, "full_log_coverage_pct": 0,
            "parse_error_count": 0, "tables": [],
        }

        # Try Modal stats first
        modal = _modal_stats.get()
        if modal and modal.get("ingest"):
            return modal["ingest"]

        # Fall back to local DB
        conn = self._connect()
        if not conn:
            return empty

        table_names = ["vaults", "strategies", "inference_logs", "full_logs", "swaps",
                        "trade_outcomes", "interp_examples_v0"]
        tables = []
        for tn in table_names:
            if self._table_exists(conn, tn):
                tables.append({"name": tn, "count": self._table_count(conn, tn)})

        vc = self._table_count(conn, "vaults")
        sc = self._table_count(conn, "strategies")
        lc = self._table_count(conn, "inference_logs")
        flc = self._table_count(conn, "full_logs")
        cov = round((flc / lc) * 100, 1) if lc else 0

        pe = 0
        if self._table_exists(conn, "full_logs"):
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM full_logs WHERE parse_error IS NOT NULL AND parse_error != ''"
            ).fetchone()
            pe = int(row["n"]) if row else 0

        conn.close()
        return {
            "vault_count": vc, "strategy_count": sc, "log_count": lc,
            "full_log_count": flc, "full_log_coverage_pct": cov,
            "parse_error_count": pe, "tables": tables,
        }

    # --- Prep detail ---

    def get_prep(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "total_examples": 0, "high_quality": 0, "medium_quality": 0, "low_quality": 0,
            "trade_count": 0, "observation_count": 0,
            "parquet_exported": False, "export_files": [], "label_distribution": [],
        }

        # Try Modal stats first
        modal = _modal_stats.get()
        if modal and modal.get("prep"):
            return modal["prep"]

        # Fall back to local DB
        conn = self._connect()
        if conn and self._table_exists(conn, "interp_examples_v0"):
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM interp_examples_v0"
            ).fetchone()
            result["total_examples"] = int(row["n"]) if row else 0

            # Quality breakdown
            for quality in ("high", "medium", "low"):
                qrow = conn.execute(
                    "SELECT COUNT(*) AS n FROM interp_examples_v0 WHERE label_quality = ?",
                    [quality],
                ).fetchone()
                result[f"{quality}_quality"] = int(qrow["n"]) if qrow else 0

            # Decision type breakdown
            dt_rows = conn.execute(
                """SELECT decision_type, COUNT(*) AS count,
                   GROUP_CONCAT(DISTINCT trade_side) AS trade_side,
                   AVG(vault_risk_preference) AS avg_risk
                   FROM interp_examples_v0 GROUP BY decision_type"""
            ).fetchall()
            result["label_distribution"] = [
                {
                    "decision_type": r["decision_type"],
                    "count": r["count"],
                    "trade_side": r["trade_side"],
                    "avg_risk": r["avg_risk"],
                }
                for r in dt_rows
            ]

            result["trade_count"] = sum(
                r["count"] for r in result["label_distribution"]
                if r["decision_type"] == "trade"
            )
            result["observation_count"] = sum(
                r["count"] for r in result["label_distribution"]
                if r["decision_type"] == "record_observation"
            )

        if conn:
            conn.close()

        # Export files (local)
        if self.exports_dir.exists():
            for f in sorted(self.exports_dir.iterdir()):
                if f.suffix in (".parquet", ".jsonl"):
                    size = f.stat().st_size
                    if size > 1024 * 1024:
                        size_str = f"{size / 1024 / 1024:.1f} MB"
                    elif size > 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size} B"
                    result["export_files"].append({"name": f.name, "size": size_str})
                    if f.name == "interp_examples_v0_high_quality.parquet":
                        result["parquet_exported"] = True

        return result

    # --- Outcomes detail ---

    def get_outcomes(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "total_outcomes": 0, "unlabeled_swaps": 0, "total_swaps": 0,
            "avg_pnl_1h": None, "avg_pnl_4h": None, "avg_pnl_1d": None,
            "win_rate_1h": None, "risk_breakdown": [],
        }

        modal = _modal_stats.get()
        if modal and modal.get("outcomes"):
            return modal["outcomes"]

        # Legacy snapshot (pre-outcomes key): extract counts from ingest.tables
        if modal and modal.get("ingest", {}).get("tables"):
            tables = {t["name"]: t["count"] for t in modal["ingest"]["tables"]}
            sw = tables.get("swaps", 0)
            oc = tables.get("trade_outcomes", 0)
            result["total_swaps"] = sw
            result["total_outcomes"] = oc
            result["unlabeled_swaps"] = sw - oc

        return result

    # --- Capture detail ---

    def get_capture(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "residual_count": 0, "router_count": 0, "total_size_mb": 0,
            "avg_seq_len": 0, "num_layers": 0, "hidden_dim": 0, "num_experts": 0,
            "recent_captures": [],
        }

        res_dir = self.activations_dir / "residual_stream"
        rtr_dir = self.activations_dir / "router_logits"
        result["residual_count"] = len(list(res_dir.glob("*.safetensors"))) if res_dir.exists() else 0
        result["router_count"] = len(list(rtr_dir.glob("*.safetensors"))) if rtr_dir.exists() else 0

        meta_path = self.activations_dir / "metadata.parquet"
        if meta_path.exists():
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(meta_path)
                rows = table.to_pylist()

                total_bytes = sum(r.get("file_size_bytes", 0) for r in rows)
                result["total_size_mb"] = round(total_bytes / 1024 / 1024, 1)

                seq_lens = [r.get("seq_len", 0) for r in rows if r.get("seq_len")]
                result["avg_seq_len"] = round(sum(seq_lens) / len(seq_lens)) if seq_lens else 0

                if rows:
                    result["num_layers"] = rows[0].get("num_layers_captured", 0)
                    result["hidden_dim"] = rows[0].get("hidden_dim", 0)
                    result["num_experts"] = rows[0].get("num_experts", 0)

                result["recent_captures"] = rows[:50]
            except Exception:
                pass

        return result

    # --- Analysis detail ---

    def get_analysis(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "total_results": 0, "probe_files": [], "has_expert_specialization": False,
            "pca_images": [], "result_files": [],
        }

        if not self.results_dir.exists():
            return result

        for f in sorted(self.results_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            size = f.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"

            result["result_files"].append({"name": f.name, "size": size_str})
            result["total_results"] += 1

            if f.name.startswith("probe_") and f.suffix == ".parquet":
                result["probe_files"].append(f.name)
            elif f.name == "expert_specialization.parquet":
                result["has_expert_specialization"] = True
            elif f.suffix == ".png":
                result["pca_images"].append(f.name)

        return result

    def get_probe_data(self, filename: str) -> dict[str, Any]:
        path = self.results_dir / filename
        if not path.exists() or not path.suffix == ".parquet":
            return {"rows": []}
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            return {"rows": table.to_pylist()}
        except Exception:
            return {"rows": []}

    def get_expert_data(self) -> dict[str, Any]:
        path = self.results_dir / "expert_specialization.parquet"
        if not path.exists():
            return {"rows": []}
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            return {"rows": table.to_pylist()}
        except Exception:
            return {"rows": []}

    def get_pca_image(self, filename: str) -> bytes | None:
        # Sanitize filename to prevent path traversal
        clean = Path(filename).name
        path = self.results_dir / clean
        if not path.exists() or path.suffix != ".png":
            return None
        return path.read_bytes()


# ---------------------------------------------------------------------------
# Job registry — tracks running/completed processes for reconnection
# ---------------------------------------------------------------------------

class _Job:
    __slots__ = ("job_id", "command", "started_at", "proc", "lines", "return_code", "lock")

    def __init__(self, job_id: str, command: str, proc: subprocess.Popen):
        self.job_id = job_id
        self.command = command
        self.started_at = time.time()
        self.proc = proc
        self.lines: list[tuple[str, str]] = []  # (event, data)
        self.return_code: int | None = None
        self.lock = threading.Lock()


class JobRegistry:
    """Thread-safe registry of running and recently completed jobs."""

    def __init__(self, max_finished: int = 20):
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
        self._max_finished = max_finished

    def start(self, cmd: str) -> _Job:
        job_id = uuid.uuid4().hex[:12]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
            cwd=str(Path.cwd()),
            env=env,
        )
        job = _Job(job_id, cmd, proc)
        with self._lock:
            self._jobs[job_id] = job

        # Background thread reads output and buffers it
        t = threading.Thread(target=self._reader, args=(job,), daemon=True)
        t.start()
        return job

    def _reader(self, job: _Job) -> None:
        sel = selectors.DefaultSelector()
        sel.register(job.proc.stdout, selectors.EVENT_READ)   # type: ignore[arg-type]
        sel.register(job.proc.stderr, selectors.EVENT_READ)   # type: ignore[arg-type]

        open_streams = 2
        while open_streams > 0:
            for key, _ in sel.select(timeout=1.0):
                line = key.fileobj.readline()  # type: ignore[union-attr]
                if not line:
                    sel.unregister(key.fileobj)
                    open_streams -= 1
                    continue
                event = "stderr" if key.fileobj is job.proc.stderr else "stdout"
                with job.lock:
                    job.lines.append((event, line.rstrip("\n")))

        job.proc.wait(timeout=30)
        with job.lock:
            job.return_code = job.proc.returncode
            job.lines.append(("done", json.dumps({"returncode": job.return_code})))

        self._prune_finished()

    def get(self, job_id: str) -> _Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            result = []
            for job in self._jobs.values():
                with job.lock:
                    result.append({
                        "job_id": job.job_id,
                        "command": job.command,
                        "started_at": job.started_at,
                        "running": job.return_code is None,
                        "return_code": job.return_code,
                        "line_count": len(job.lines),
                    })
            return sorted(result, key=lambda j: j["started_at"], reverse=True)

    def _prune_finished(self) -> None:
        """Remove old finished jobs if we have too many."""
        with self._lock:
            finished = [j for j in self._jobs.values() if j.return_code is not None]
            finished.sort(key=lambda j: j.started_at)
            while len(finished) > self._max_finished:
                old = finished.pop(0)
                del self._jobs[old.job_id]


# Global job registry
_job_registry = JobRegistry()


# ---------------------------------------------------------------------------
# Allowed commands (whitelist for safety)
# ---------------------------------------------------------------------------

ALLOWED_COMMANDS = [
    "uv run -m pipelines.ingest",
    "uv run -m pipelines.interp.prepare",
    "uv run -m pipelines.interp.outcomes",
    "uv run --extra interp -m pipelines.interp.capture",
    "uv run --extra analysis -m pipelines.interp.analysis",
    "./scripts/modal_capture.sh",
    "modal volume put",
]


def _is_command_allowed(cmd: str) -> bool:
    return any(cmd.startswith(prefix) for prefix in ALLOWED_COMMANDS)


def _run_command(cmd: str) -> dict[str, Any]:
    if not _is_command_allowed(cmd):
        return {"stdout": "", "stderr": f"Command not allowed: {cmd}", "returncode": -1}
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=600,
            cwd=str(Path.cwd()),
        )
        return {
            "stdout": result.stdout[-10000:] if len(result.stdout) > 10000 else result.stdout,
            "stderr": result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Command timed out after 600s", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}


def _run_command_streaming(handler: BaseHTTPRequestHandler, cmd: str) -> None:
    """Start a command via the job registry and stream output as SSE."""
    if not _is_command_allowed(cmd):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.end_headers()
        _sse_write(handler, "stderr", f"Command not allowed: {cmd}")
        _sse_write(handler, "done", json.dumps({"returncode": -1}))
        return

    job = _job_registry.start(cmd)

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.end_headers()

    # Send the job_id so the client can reconnect
    _sse_write(handler, "job_id", job.job_id)

    _stream_job(handler, job, from_line=0)


def _reconnect_job_streaming(handler: BaseHTTPRequestHandler, job_id: str) -> None:
    """Reconnect to a running or completed job and stream its output."""
    job = _job_registry.get(job_id)
    if not job:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.end_headers()
        _sse_write(handler, "stderr", f"Job not found: {job_id}")
        _sse_write(handler, "done", json.dumps({"returncode": -1}))
        return

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.end_headers()

    # Send command so UI knows what's running
    _sse_write(handler, "command", job.command)
    _stream_job(handler, job, from_line=0)


def _stream_job(handler: BaseHTTPRequestHandler, job: _Job, from_line: int) -> None:
    """Stream buffered + live lines from a job as SSE."""
    cursor = from_line
    try:
        while True:
            with job.lock:
                new_lines = job.lines[cursor:]
                done = job.return_code is not None

            for event, data in new_lines:
                _sse_write(handler, event, data)
                cursor += 1

            if done:
                # Invalidate modal stats cache so next request fetches fresh data
                _modal_stats.invalidate()
                break

            time.sleep(0.1)
    except (BrokenPipeError, ConnectionResetError):
        pass  # Client disconnected — job keeps running in background


def _sse_write(handler: BaseHTTPRequestHandler, event: str, data: str) -> None:
    try:
        payload = f"event: {event}\ndata: {data}\n\n"
        handler.wfile.write(payload.encode())
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def _make_handler(store: DashboardStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, data: Any, status: int = 200) -> None:
            body = json.dumps(data, ensure_ascii=True, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, html: str) -> None:
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _png(self, data: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _not_found(self) -> None:
            self._json({"error": "not found"}, 404)

        def _serve_static(self, file_path: Path) -> bool:
            """Serve a static file from the React build. Returns True if served."""
            if not file_path.exists() or not file_path.is_file():
                return False
            content = file_path.read_bytes()
            suffix = file_path.suffix
            mime = {
                ".html": "text/html", ".js": "application/javascript",
                ".css": "text/css", ".json": "application/json",
                ".png": "image/png", ".svg": "image/svg+xml",
                ".ico": "image/x-icon", ".woff2": "font/woff2",
            }.get(suffix, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return True

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            # API routes
            if path == "/api/status":
                refresh = (query.get("refresh", ["0"])[0] or "").lower() in {"1", "true", "yes"}
                if refresh:
                    _modal_stats.force_refresh()
                self._json(store.get_status())
            elif path == "/api/ingest":
                self._json(store.get_ingest())
            elif path == "/api/prep":
                self._json(store.get_prep())
            elif path == "/api/outcomes":
                self._json(store.get_outcomes())
            elif path == "/api/capture":
                self._json(store.get_capture())
            elif path == "/api/analysis":
                self._json(store.get_analysis())
            elif path == "/api/analysis/probe-data":
                filename = query.get("file", [""])[0]
                self._json(store.get_probe_data(filename))
            elif path == "/api/analysis/expert-data":
                self._json(store.get_expert_data())
            elif path.startswith("/api/analysis/pca/"):
                filename = path.removeprefix("/api/analysis/pca/")
                img = store.get_pca_image(filename)
                if img:
                    self._png(img)
                else:
                    self._not_found()
            elif path == "/api/jobs":
                self._json(_job_registry.list_jobs())
            elif path == "/api/backend-url":
                url = os.environ.get("XENON_BACKEND_URL")
                if not url:
                    url_file = os.path.expanduser("~/.xenon_backend_url")
                    if os.path.exists(url_file):
                        with open(url_file) as f:
                            url = f.read().strip()
                if url:
                    self._json({"url": url.rstrip("/")})
                else:
                    self._json({"url": None, "error": "No backend URL configured. Set XENON_BACKEND_URL or write to ~/.xenon_backend_url"})

            # Static files from React build (production)
            else:
                static_dir = Path(__file__).parent / "dashboard-ui" / "dist"
                # Try exact path first
                clean = path.lstrip("/")
                if clean and self._serve_static(static_dir / clean):
                    return
                # SPA fallback: serve index.html for all non-API, non-asset routes
                if not self._serve_static(static_dir / "index.html"):
                    self._html(HTML_PAGE)  # Fallback to inline HTML if no build

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/run":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                cmd = body.get("command", "")
                result = _run_command(cmd)
                self._json(result)
            elif parsed.path == "/api/run-stream":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                cmd = body.get("command", "")
                _run_command_streaming(self, cmd)
            elif parsed.path == "/api/job-reconnect":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                job_id = body.get("job_id", "")
                _reconnect_job_streaming(self, job_id)
            elif parsed.path == "/api/status/refresh":
                _modal_stats.force_refresh()
                self._json(store.get_status())
            else:
                self._not_found()

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Xenon Pipeline Dashboard")
    p.add_argument("--db-path", type=Path, default=Path("data/terminal_ingest.db"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8800)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    store = DashboardStore(db_path=args.db_path, data_dir=args.data_dir)
    handler = _make_handler(store)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Xenon Dashboard running at {url}")
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
