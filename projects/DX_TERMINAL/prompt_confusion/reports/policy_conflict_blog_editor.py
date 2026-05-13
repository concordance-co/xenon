from __future__ import annotations

import html
import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "policy_conflict_blog_skeleton_state_v2.json"
MD_PATH = ROOT / "policy_conflict_blog_working.md"
TYP_PATH = ROOT / "policy_conflict_blog_working.typ"
PDF_PATH = ROOT / "policy_conflict_blog_working.pdf"
LOCAL_STORAGE_KEY = "policy-conflict-skeleton-v7"


DEFAULT_SECTIONS = [
    {
        "id": "opener",
        "title": "Opener",
        "hint": "Set the hook and the claim boundary. Keep this short.",
        "markdown": """This work follows up on our recent blog post documenting our collaboration with DX Terminal to use mechanistic interpretability in real world financial contexts. In part 1, we described our research probing for, and discovering, the early signs of interpretable "Market Perception" geometries in LLMs when given real market data.

Part 2 of our work examined a common problem DXRG saw in their agents: strange behaviors when policies collide. Users often ask their agents to execute on strategies they create that directly conflict with the vault settings they inputted when configuring the initial setup.

Our thesis was that we'd be able to find interpretable probe directions associated with "conflict" that might allow us to detect difficult-to-spot policy-conflict scenarios in the prompts.

We used a semi-synthetic pipeline to discover 3 directions associated with distinct "conflict families" that shared geometric structure with each other, and have achieved early results that show the directions are active in real data in the cases they were trained to fire on.""",
    },
    {
        "id": "problem",
        "title": "Problem",
        "hint": "Add concrete DX Terminal examples in plain language.",
        "markdown": """Users operating an agent in the DX Terminal experiment set an initial set of sliders that correspond to different elements of trading for their *vaults*.

Those settings include:

- `Trading Activity`: how readily the agent should take action instead of observing.
- `Trade Size`: how large each trade should be.
- `Risk Preference`: whether the agent should prefer safer or more aggressive opportunities.
- `Holding Style`: how long the agent should generally expect to hold positions.
- `Diversification`: whether the agent should broaden exposure or concentrate into stronger opportunities.

When deployed, agents are prompted at regular tick intervals with updated market information, and asked to call one of three tools:

- `record_observation(content, strategy?)`: records an observation without trading. `strategy` is present when the observation is tied to a specific strategy.
- `buy_token(token, spend_pct, content, strategy?)`: buys `token` using `spend_pct` percent of available ETH, with `content` explaining the decision.
- `sell_token(token, spend_pct, content, strategy?)`: sells `spend_pct` percent of the current `token` position, with `content` explaining the decision.

Users can also chat with their agent to come up with *strategies* that provide more explicit guidance for the agent. Sometimes, the user curated strategies conflict with the initial vault settings the agent was configured with, and this can lead to strange agent behavior. For example, a user may set their initial vault to `trade_size = 1`, and then after a conversation with the agent create the strategy to "Sell all positions and go full port into X Token if momentum is strong". This would imply the agent should take action with the full portfolio, which contradicts the small setting used to configure the vault.

While there are instructions in the system prompt for how to resolve this kind of conflict ("ACTIVE SETTINGS are binding execution constraints" and "STRATEGY expresses preferences that apply only within what ACTIVE SETTINGS allow"), it can still create undesirable behavior in situations where it's not clear which path to take.

We wanted to see if the model is aware of these conflicts when processing a prompt.""",
    },
    {
        "id": "synthetic_setup",
        "title": "Synthetic Abstraction",
        "hint": "Explain why controlled prompts come before real transfer.",
        "markdown": """Real world data can be incredibly messy, but to ask mechanistic questions, one needs a clean dataset to amplify potentially active features before bringing learnings back into real settings.

In our case, we wanted to examine specific conflicts between user-configured *strategies* and vault-configured *settings*, so we created a synthetic dataset of ~1100 rows to make the read stronger.

Synthetic Prompt Structure:

```text
[system]
Role: trading agent.
Core rule: each prompt contains STRATEGY and ACTIVE SETTINGS.
Priority rule: ACTIVE SETTINGS are binding execution constraints.
Decision order:
  1. decide whether ACTIVE SETTINGS permit entry
  2. choose asset according to the allowed risk/diversification posture
  3. choose size according to ACTIVE SETTINGS
Output format: strict JSON only.

[user]
TASK
Choose exactly one action for this tick.

STRATEGY
A compact preference statement, e.g.:
  - prefer high-conviction trades
  - prefer large/small size
  - prefer concentrated/diversified exposure
Strategy applies only within ACTIVE SETTINGS.

ACTIVE SETTINGS
Slider-like constraints:
  - Trading Activity
  - Trade Size
  - Risk Preference
  - Holding Style
  - Diversification

PORTFOLIO
Small controlled portfolio state.

MARKET
Four synthetic assets:
  - ALPHA
  - BETA
  - DELTA
  - GAMMA
Each has controlled evidence/risk/diversification language.
```

Using this structure, we can create a synthetic dataset that creates both aligned and conflicted rows between three setting types: `trade_size`, `risk_preference`, and `diversification_preference`. For example, to create a conflict row for `trade_size`, we could set the `Trade Size` slider to 5 (highest size), and then add a strategy like "Never trade more than 10% of portfolio" while keeping all other things constant. Our decision to isolate conflicts into three families came from an earlier unsupervised discovery phase that showed there might exist different conflict resolution circuits depending on the *type* of conflict in the prompt.

To avoid lexical confounds, we came up with strategy and other contextual information variation and split the data for strict holdouts to ensure there is minimal leakage between train and test sets.

TODO: Add lexical variation examples here.""",
    },
    {
        "id": "synthetic_results_intro",
        "title": "Synthetic Probe Results",
        "hint": "Lead into the locked results table.",
        "markdown": """After a few iterations on the synthetic prompt structure and confound isolation, we achieved the following results:""",
    },
    {
        "id": "synthetic_results_takeaway",
        "title": "Synthetic Results Takeaway",
        "hint": "Interpret the charts and geometry without overclaiming.",
        "markdown": """We found that while the three aforementioned families had noticeably distinct geometry (see cosine similarities/PCA projections), there was a significant shared space when looking at the `shared_mean` across all trained probe vectors that hinted there is enough structure to continue in our tests.

- clean synthetic policy-conflict directions are strongly readable
- validated synthetic families:
  - `trade_size`
  - `risk_preference`
  - `diversification_preference`
- families share meaningful geometry
- not collinear""",
    },
    {
        "id": "transfer_failure",
        "title": "First Real Transfer Attempt",
        "hint": "Frame failure as ontology sharpening, not as a dead end.",
        "markdown": (
            "The first direct transfer pass projected synthetic conflict directions onto full production "
            "prompts at coarse global sites. It did not cleanly separate complaint rows from baseline controls.\n\n"
            "That failure mattered. It showed that synthetic success does not automatically become production "
            "success."
        ),
    },
    {
        "id": "bridge",
        "title": "Bridge Program",
        "hint": "Use this as a short failure-analysis interlude before Phase 13.",
        "markdown": (
            "We then used bridge datasets to separate template mismatch from content and ontology mismatch. "
            "The bridge evidence was real but weak: buy-only filtering helped, but the deeper issue was an "
            "unresolved ontology and representation mismatch."
        ),
    },
    {
        "id": "phase13",
        "title": "Direct Projection on Real Prompts",
        "hint": "Explain the simpler question: fixed directions, no classifier, no threshold.",
        "markdown": (
            "Phase 13 asked a simpler question: if we do not train a classifier and do not set thresholds, "
            "do fixed synthetic directions produce scalar structure anywhere on real DX Terminal prompts?"
        ),
    },
    {
        "id": "row_reading",
        "title": "Row Reading / Ontology Correction",
        "hint": "This should become the core narrative section.",
        "markdown": (
            "The most important interpretability move was reading the top and bottom rows. The preregistered "
            "root-cause proxy was wrong for the `trade_size` target. Root-cause labels diagnose why a "
            "complaint happened; the probe target is visible current-prefix conflict shape."
        ),
    },
    {
        "id": "claim_boundary",
        "title": "Claim Boundary",
        "hint": "Keep this crisp and conservative.",
        "markdown": (
            "Fixed synthetic directions recover a real production signal at L32 `settings_end`. `trade_size` "
            "is selective for current-prefix concrete sized-action conflict. `shared_mean` appears to track "
            "broader policy tension, but the shared-family interpretation still needs more audit.\n\n"
            "This is not a final detector, not a deployment monitor, and not a causal mechanism claim."
        ),
    },
    {
        "id": "closing",
        "title": "Closing / Next Steps",
        "hint": "Close with the practical loop and next work.",
        "markdown": (
            "The useful loop is: real data exposes a messy failure mode; synthetic prompts isolate a clean "
            "abstraction; probes find a candidate internal signal; bridge tests expose transfer mismatch; "
            "real-data projection finds a narrower shape-specific signal; row reading improves the ontology."
        ),
    },
]


DATA_BLOCKS = {
    "synthetic_setup": [
        {
            "title": "Families Tested",
            "body": [
                "`trade_size`: buy small vs large; output size/action axis.",
                "`risk_preference`: asset selection by allowed risk posture.",
                "`diversification_preference`: concentration vs broadening; portfolio-conditioned.",
            ],
        }
    ],
    "synthetic_results_intro": [
        {
            "title": "Synthetic Probe Table",
            "table": {
                "columns": ["Family", "Standard probe results", "Strict holdout"],
                "rows": [
                    ["trade_size", "XOR 0.9948 / 1.0000; strategy 1.0000 / 1.0000; settings 0.9948 / 1.0000", "0.990 / 1.000 at L40"],
                    ["risk_preference", "XOR 0.9635 / 0.9766; strategy 0.9844 / 0.9937; settings 0.9740 / 0.9839", "0.8854 / 0.9119"],
                    ["diversification_preference", "behavior aligned 1.0000, conflict 0.8542; XOR 0.9896 / 0.9995; strategy 1.0000 / 1.0000; settings 0.9792 / 0.9957", "0.8333 / 0.8819"],
                ],
            },
        },
        {
            "title": "Figures",
            "figures": [
                {"src": "phase_12/reports/dx_terminal_brief_assets/family_within_auroc_by_layer.png", "caption": "Representative within-family AUROC curves."},
                {"src": "phase_12/reports/dx_terminal_brief_assets/strict_family_auroc_by_layer.png", "caption": "Strict lexical-holdout AUROC curves."},
            ],
        },
    ],
    "synthetic_results_takeaway": [
        {
            "title": "L36 Same-Capture Geometry",
            "table": {
                "columns": ["Pair", "Cosine"],
                "rows": [
                    ["risk_preference vs trade_size", "0.6449"],
                    ["diversification_preference vs risk_preference", "0.4684"],
                    ["diversification_preference vs trade_size", "0.4883"],
                ],
            },
        },
        {
            "title": "Geometry Figures",
            "figures": [
                {"src": "phase_12/reports/three_family_visuals/shared_axis_distributions.png", "caption": "Shared-axis distributions: separation exists, but family baselines are offset."},
                {"src": "phase_12/reports/three_family_visuals/directed_subspace_scatter_by_family_conflict_v2.png", "caption": "Directed subspace view: related conflict geometry, not one collinear axis."},
            ],
        },
    ],
    "bridge": [
        {
            "title": "Bridge Dataset Counts",
            "table": {
                "columns": ["Dataset", "Rows", "Aligned", "Conflict"],
                "rows": [
                    ["Stage 1a template control", "768", "384", "384"],
                    ["Stage 1b loose adapter", "258", "168", "90"],
                    ["Stage 1b strict adapter", "118", "81", "37"],
                    ["Stage 1b strict buy-only", "33", "27", "6"],
                ],
            },
        }
    ],
    "phase13": [
        {
            "title": "L32 Settings-End Cohort Means",
            "table": {
                "columns": ["Direction", "Anchor", "Complaint", "Control", "Anchor-control", "Complaint-control"],
                "rows": [
                    ["trade_size", "4.425", "3.803", "3.278", "+1.147", "+0.526"],
                    ["shared_mean", "3.462", "3.137", "2.760", "+0.703", "+0.377"],
                ],
            },
        }
    ],
    "row_reading": [
        {
            "title": "Top/Bottom Shape Audit",
            "table": {
                "columns": ["Direction", "Top action/size", "Top strategy ignored", "Bottom action/size", "Bottom strategy ignored"],
                "rows": [
                    ["trade_size", "20/25", "5/25", "15/25", "10/25"],
                    ["shared_mean", "20/25", "5/25", "9/25", "16/25"],
                ],
            },
        },
        {
            "title": "Top trade_size Complaint Types",
            "table": {
                "columns": ["Type", "Count"],
                "rows": [
                    ["UNWANTED_BUY", "10/25"],
                    ["UNWANTED_SELL", "6/25"],
                    ["WRONG_SIZE", "4/25"],
                    ["Concrete action/size combined", "20/25"],
                ],
            },
        },
    ],
}


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Policy Conflict Blog Skeleton</title>
  <style>
    :root {
      --ink: #1b242c;
      --muted: #60707d;
      --paper: #fbfaf7;
      --line: #d8d1c5;
      --panel: #f0eee8;
      --data: #eef4f2;
      --data-line: #bdcbc6;
      --accent: #a83d2d;
      --focus: #2d6f90;
      --todo: #fff1c7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: Charter, "Iowan Old Style", Georgia, serif;
    }
    button, textarea, input {
      font: inherit;
    }
    .app {
      display: grid;
      grid-template-columns: 290px minmax(0, 1fr) 380px;
      min-height: 100vh;
    }
    .rail, .inspector {
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      padding: 18px;
      background: var(--panel);
      border-color: var(--line);
    }
    .rail { border-right: 1px solid var(--line); }
    .inspector { border-left: 1px solid var(--line); }
    .brand {
      color: var(--accent);
      font: 700 11px ui-sans-serif, system-ui, sans-serif;
      letter-spacing: .12em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    h1 {
      font-size: 25px;
      line-height: 1.05;
      margin: 0 0 10px;
    }
    .sub {
      color: var(--muted);
      font: 13px/1.35 ui-sans-serif, system-ui, sans-serif;
      margin: 0 0 18px;
    }
    .actions {
      display: grid;
      gap: 8px;
      margin: 16px 0;
    }
    button {
      border: 1px solid var(--ink);
      background: var(--ink);
      color: #fff;
      padding: 9px 10px;
      cursor: pointer;
      border-radius: 3px;
      font: 700 12px ui-sans-serif, system-ui, sans-serif;
      text-align: left;
    }
    button.secondary {
      background: transparent;
      color: var(--ink);
      border-color: var(--line);
    }
    button:focus-visible, textarea:focus-visible {
      outline: 2px solid var(--focus);
      outline-offset: 2px;
    }
    .nav {
      display: grid;
      gap: 3px;
      margin-top: 18px;
    }
    .nav a {
      color: var(--ink);
      text-decoration: none;
      padding: 7px 8px;
      border-left: 3px solid transparent;
      font: 13px ui-sans-serif, system-ui, sans-serif;
    }
    .nav a:hover {
      background: #fff7;
      border-left-color: var(--accent);
    }
    main {
      padding: 24px clamp(22px, 4vw, 58px) 80px;
      max-width: 1050px;
      width: 100%;
      justify-self: center;
    }
    .section {
      border-top: 1px solid var(--line);
      padding: 28px 0 34px;
    }
    .section-head {
      display: flex;
      align-items: start;
      gap: 18px;
      margin-bottom: 14px;
    }
    .section h2 {
      font-size: 28px;
      line-height: 1;
      margin: 0;
      min-width: 260px;
    }
    .hint {
      color: var(--muted);
      font: 13px/1.35 ui-sans-serif, system-ui, sans-serif;
      max-width: 560px;
    }
    textarea {
      width: 100%;
      min-height: 160px;
      resize: vertical;
      border: 1px solid var(--line);
      background: #fffefa;
      color: var(--ink);
      border-radius: 2px;
      padding: 14px 15px;
      line-height: 1.45;
      box-shadow: inset 0 1px 0 #fff;
    }
    .data-block {
      margin: 14px 0;
      background: var(--data);
      border: 1px solid var(--data-line);
      padding: 12px;
    }
    .data-title {
      font: 800 12px ui-sans-serif, system-ui, sans-serif;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: #27584e;
      margin-bottom: 9px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font: 12px ui-sans-serif, system-ui, sans-serif;
      background: #fffdfa;
    }
    th, td {
      border: 1px solid var(--data-line);
      text-align: left;
      vertical-align: top;
      padding: 7px 8px;
    }
    th {
      background: #dfe9e6;
      font-weight: 800;
    }
    .fig-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    figure {
      margin: 0;
      background: #fffefa;
      border: 1px solid var(--data-line);
      padding: 8px;
    }
    figure img {
      display: block;
      width: 100%;
      height: auto;
    }
    figcaption, .source {
      color: var(--muted);
      font: 12px/1.35 ui-sans-serif, system-ui, sans-serif;
      margin-top: 7px;
    }
    .preview {
      background: #fffefa;
      border: 1px solid var(--line);
      padding: 16px;
      min-height: 400px;
    }
    .preview h2 {
      margin: 20px 0 8px;
      font-size: 20px;
    }
    .preview p, .preview li {
      font-size: 14px;
      line-height: 1.45;
    }
    .preview code {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: .92em;
      background: #eee7d8;
      padding: 1px 3px;
    }
    .status {
      min-height: 20px;
      color: var(--muted);
      font: 12px ui-sans-serif, system-ui, sans-serif;
      margin-top: 10px;
    }
    .meter {
      color: var(--muted);
      font: 12px ui-sans-serif, system-ui, sans-serif;
      margin: 10px 0;
    }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 230px minmax(0, 1fr); }
      .inspector { display: none; }
    }
    @media (max-width: 760px) {
      .app { display: block; }
      .rail { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }
      .section-head { display: block; }
      .section h2 { margin-bottom: 8px; }
      .fig-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="rail">
      <div class="brand">DX Terminal</div>
      <h1>Policy Conflict Blog Skeleton</h1>
      <p class="sub">Markdown boxes around locked data. Save whenever you want. Build PDF only on demand.</p>
      <div class="actions">
        <button id="saveBtn">Save Markdown</button>
        <button id="buildBtn" class="secondary">Build PDF Now</button>
        <button id="downloadBtn" class="secondary">Download .md</button>
      </div>
      <div class="meter" id="meter"></div>
      <div class="status" id="status"></div>
      <nav class="nav" id="nav"></nav>
    </aside>
    <main id="editor"></main>
    <aside class="inspector">
      <div class="brand">Live Read</div>
      <h1>Markdown Preview</h1>
      <p class="sub">Quick text-only preview. Locked data stays in the main column.</p>
      <div class="preview" id="preview"></div>
    </aside>
  </div>
  <script>
    const DEFAULT_SECTIONS = __DEFAULT_SECTIONS__;
    const DATA_BLOCKS = __DATA_BLOCKS__;
    let sections = structuredClone(DEFAULT_SECTIONS);

    const editor = document.getElementById("editor");
    const nav = document.getElementById("nav");
    const preview = document.getElementById("preview");
    const statusEl = document.getElementById("status");
    const meter = document.getElementById("meter");

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function renderInline(value) {
      return escapeHtml(value).replace(/`([^`]+)`/g, "<code>$1</code>");
    }

    function markdownToHtml(md) {
      const lines = md.split(/\r?\n/);
      let html = "";
      let list = false;
      for (const line of lines) {
        if (/^\s*-\s+/.test(line)) {
          if (!list) { html += "<ul>"; list = true; }
          html += `<li>${renderInline(line.replace(/^\s*-\s+/, ""))}</li>`;
        } else {
          if (list) { html += "</ul>"; list = false; }
          if (line.trim()) html += `<p>${renderInline(line)}</p>`;
        }
      }
      if (list) html += "</ul>";
      return html;
    }

    function dataBlockHtml(block) {
      let out = `<section class="data-block"><div class="data-title">${escapeHtml(block.title)}</div>`;
      if (block.body) {
        out += "<ul>" + block.body.map(item => `<li>${renderInline(item)}</li>`).join("") + "</ul>";
      }
      if (block.table) {
        out += "<table><thead><tr>" + block.table.columns.map(c => `<th>${escapeHtml(c)}</th>`).join("") + "</tr></thead><tbody>";
        out += block.table.rows.map(row => "<tr>" + row.map(cell => `<td>${renderInline(cell)}</td>`).join("") + "</tr>").join("");
        out += "</tbody></table>";
      }
      if (block.figures) {
        out += '<div class="fig-grid">';
        out += block.figures.map(fig => `<figure><img src="../${escapeHtml(fig.src)}" alt=""><figcaption>${escapeHtml(fig.caption)}</figcaption></figure>`).join("");
        out += "</div>";
      }
      if (block.source) out += `<div class="source">Source: ${escapeHtml(block.source)}</div>`;
      return out + "</section>";
    }

    function render() {
      nav.innerHTML = sections.map(s => `<a href="#${s.id}">${escapeHtml(s.title)}</a>`).join("");
      editor.innerHTML = sections.map(section => `
        <section class="section" id="${section.id}">
          <div class="section-head">
            <h2>${escapeHtml(section.title)}</h2>
            <div class="hint">${escapeHtml(section.hint)}</div>
          </div>
          <textarea data-id="${section.id}" spellcheck="true">${escapeHtml(section.markdown)}</textarea>
          ${(DATA_BLOCKS[section.id] || []).map(dataBlockHtml).join("")}
        </section>
      `).join("");
      editor.querySelectorAll("textarea").forEach(area => {
        area.addEventListener("input", () => {
          const target = sections.find(s => s.id === area.dataset.id);
          target.markdown = area.value;
          persistLocal();
          renderPreview();
        });
      });
      renderPreview();
    }

    function renderPreview() {
      const words = sections.reduce((n, s) => n + (s.markdown.match(/[A-Za-z][A-Za-z'_-]*/g) || []).length, 0);
      meter.textContent = `${words} prose words in editable boxes`;
      preview.innerHTML = sections.map(s => `<h2>${escapeHtml(s.title)}</h2>${markdownToHtml(s.markdown)}`).join("");
    }

    function assembleMarkdown() {
      let out = "# Policy Conflict Internals in Real Agentic Contexts\n\n";
      for (const section of sections) {
        out += `## ${section.title}\n\n${section.markdown.trim()}\n\n`;
        for (const block of (DATA_BLOCKS[section.id] || [])) {
          out += `### DATA: ${block.title}\n\n`;
          if (block.body) out += block.body.map(item => `- ${item}`).join("\n") + "\n\n";
          if (block.table) {
            out += `| ${block.table.columns.join(" | ")} |\n`;
            out += `| ${block.table.columns.map(() => "---").join(" | ")} |\n`;
            out += block.table.rows.map(row => `| ${row.join(" | ")} |`).join("\n") + "\n\n";
          }
          if (block.figures) out += block.figures.map(fig => `![${fig.caption}](${fig.src})`).join("\n") + "\n\n";
          if (block.source) out += `Source: ${block.source}\n\n`;
        }
      }
      return out;
    }

    function persistLocal() {
      localStorage.setItem("__LOCAL_STORAGE_KEY__", JSON.stringify(sections));
    }

    function restoreLocal() {
      const saved = localStorage.getItem("__LOCAL_STORAGE_KEY__");
      if (!saved) return;
      try {
        const parsed = JSON.parse(saved);
        sections = DEFAULT_SECTIONS.map(def => ({...def, markdown: (parsed.find(s => s.id === def.id) || def).markdown}));
      } catch {
        // Ignore malformed local state.
      }
    }

    async function post(path, body) {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "request failed");
      return payload;
    }

    document.getElementById("saveBtn").addEventListener("click", async () => {
      statusEl.textContent = "Saving...";
      try {
        const result = await post("/api/save", {sections, markdown: assembleMarkdown()});
        statusEl.textContent = `Saved ${result.markdown_path}`;
      } catch (error) {
        statusEl.textContent = error.message;
      }
    });

    document.getElementById("buildBtn").addEventListener("click", async () => {
      statusEl.textContent = "Building PDF...";
      try {
        const result = await post("/api/build-pdf", {sections, markdown: assembleMarkdown()});
        statusEl.innerHTML = `Built <a href="${result.pdf_url}" target="_blank">PDF</a>`;
      } catch (error) {
        statusEl.textContent = error.message;
      }
    });

    document.getElementById("downloadBtn").addEventListener("click", () => {
      const blob = new Blob([assembleMarkdown()], {type: "text/markdown"});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "policy_conflict_blog_working.md";
      link.click();
      URL.revokeObjectURL(url);
    });

    restoreLocal();
    render();
  </script>
</body>
</html>"""


def load_sections() -> list[dict]:
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text())
            saved = {section["id"]: section for section in state.get("sections", [])}
            return [{**default, "markdown": saved.get(default["id"], default).get("markdown", default["markdown"])} for default in DEFAULT_SECTIONS]
        except Exception:
            pass
    return DEFAULT_SECTIONS


def write_json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json_request(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def markdown_to_typst(md: str) -> str:
    lines = md.splitlines()
    output: list[str] = []
    in_code = False
    code_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                output.append("#block(fill: rgb(\"#f7f3ea\"), inset: 7pt)[")
                output.append("#raw(" + json.dumps("\n".join(code_lines)) + ", lang: \"text\")")
                output.append("]")
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        stripped = line.strip()
        if not stripped:
            output.append("")
        elif stripped.startswith("### "):
            output.append("== " + stripped[4:])
        elif stripped.startswith("## "):
            output.append("= " + stripped[3:])
        elif stripped.startswith("# "):
            output.append("#text(size: 20pt, weight: \"bold\")[" + typ_text(stripped[2:]) + "]")
        elif stripped.startswith("- "):
            output.append("- " + typ_text(stripped[2:]))
        else:
            output.append(typ_text(stripped))
    return "\n".join(output).strip()


def typ_text(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("*", "\\*")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("$", "\\$")
    )


def typ_table(columns: list[str], rows: list[list[str]]) -> str:
    entries = [[f"*{col}*" for col in columns], *rows]
    cells = ",\n  ".join(f"[{typ_text(str(cell))}]" for row in entries for cell in row)
    col_spec = ", ".join(["1fr"] * len(columns))
    return (
        "#table(\n"
        f"  columns: ({col_spec}),\n"
        "  inset: 4pt,\n"
        "  stroke: 0.35pt + rgb(\"#ccd6de\"),\n"
        "  fill: (x, y) => if y == 0 { rgb(\"#e8eef3\") } else if calc.odd(y) { rgb(\"#f8fafb\") } else { white },\n"
        f"  {cells},\n"
        ")"
    )


def typ_data_blocks(section_id: str) -> str:
    chunks: list[str] = []
    for block in DATA_BLOCKS.get(section_id, []):
        chunks.append(f"== {block['title']}")
        if "body" in block:
            chunks.extend("- " + typ_text(item) for item in block["body"])
        if "table" in block:
            chunks.append(typ_table(block["table"]["columns"], block["table"]["rows"]))
        if "figures" in block:
            for fig in block["figures"]:
                chunks.append(f"#figure(image(\"../{fig['src']}\", width: 100%), caption: [{typ_text(fig['caption'])}])")
        if "source" in block:
            chunks.append(f"#text(size: 7.5pt, fill: rgb(\"#60707d\"))[Source: {typ_text(block['source'])}]")
    return "\n\n".join(chunks)


def build_typst(sections: list[dict]) -> str:
    chunks = [
        "#set page(paper: \"us-letter\", margin: 1.45cm, numbering: \"1\")",
        "#set text(font: \"Georgia\", size: 9pt)",
        "#set par(justify: true, leading: 0.55em)",
        "#set heading(numbering: none)",
        "#show heading.where(level: 1): it => { set text(size: 15pt, weight: \"bold\"); v(0.55em); it; v(0.15em); line(length: 100%, stroke: 0.7pt + rgb(\"#ccd6de\")); v(0.2em) }",
        "#show heading.where(level: 2): it => { set text(size: 10.5pt, weight: \"bold\"); v(0.35em); it; v(0.1em) }",
        "#text(size: 21pt, weight: \"bold\")[Policy Conflict Internals in Real Agentic Contexts]",
        "#v(0.5em)",
    ]
    for section in sections:
        chunks.append(f"= {section['title']}")
        chunks.append(markdown_to_typst(section.get("markdown", "")))
        data = typ_data_blocks(section["id"])
        if data:
            chunks.append(data)
    return "\n\n".join(chunks)


def save_payload(payload: dict) -> list[dict]:
    incoming = payload.get("sections", [])
    by_id = {section.get("id"): section for section in incoming}
    sections = [{**default, "markdown": by_id.get(default["id"], default).get("markdown", default["markdown"])} for default in DEFAULT_SECTIONS]
    STATE_PATH.write_text(json.dumps({"sections": sections}, indent=2) + "\n")
    MD_PATH.write_text(payload.get("markdown") or assemble_markdown(sections))
    return sections


def assemble_markdown(sections: list[dict]) -> str:
    output = ["# Policy Conflict Internals in Real Agentic Contexts", ""]
    for section in sections:
        output.extend([f"## {section['title']}", "", section.get("markdown", "").strip(), ""])
    return "\n".join(output)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/policy_conflict_blog_editor.html"}:
            page = (
                HTML_PAGE
                .replace("__DEFAULT_SECTIONS__", json.dumps(load_sections()))
                .replace("__DATA_BLOCKS__", json.dumps(DATA_BLOCKS))
                .replace("__LOCAL_STORAGE_KEY__", LOCAL_STORAGE_KEY)
            )
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        candidates = [
            (ROOT / path.lstrip("/")).resolve(),
            (ROOT.parent / path.lstrip("/")).resolve(),
        ]
        requested = next(
            (
                candidate
                for candidate in candidates
                if candidate.is_file()
                and (ROOT in candidate.parents or ROOT.parent in candidate.parents)
            ),
            None,
        )
        if requested is not None:
            content_type = "application/pdf" if requested.suffix == ".pdf" else "text/plain"
            if requested.suffix == ".png":
                content_type = "image/png"
            elif requested.suffix == ".html":
                content_type = "text/html; charset=utf-8"
            body = requested.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().send_error(404)

    def do_POST(self) -> None:
        try:
            payload = read_json_request(self)
            if self.path == "/api/save":
                save_payload(payload)
                write_json_response(self, 200, {"ok": True, "markdown_path": MD_PATH.name, "state_path": STATE_PATH.name})
                return
            if self.path == "/api/build-pdf":
                sections = save_payload(payload)
                TYP_PATH.write_text(build_typst(sections))
                subprocess.run(["typst", "compile", "--root", str(ROOT.parent), str(TYP_PATH), str(PDF_PATH)], check=True, cwd=ROOT)
                write_json_response(self, 200, {"ok": True, "typ_path": TYP_PATH.name, "pdf_url": "/" + PDF_PATH.name})
                return
            write_json_response(self, 404, {"error": "unknown endpoint"})
        except Exception as exc:
            write_json_response(self, 500, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        print(format % args)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Serving editor at http://127.0.0.1:8765/policy_conflict_blog_editor.html")
    server.serve_forever()
