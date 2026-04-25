from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from projects.DX_TERMINAL.prompt_confusion.neon import connect_neon, validate_table_name
from projects.DX_TERMINAL.prompt_confusion.paths import phase_root


DEFAULT_TABLE = "dx_terminal_trade_size_stage1b_adapter_strict_v1"
DEFAULT_OUTPUT = phase_root("phase_12", __file__) / "reports" / "stage1b_strict_prompt_review.html"


LABEL_FIELDS = (
    "adapter_alignment_label",
    "strategy_size_preference",
    "slider_size_bucket",
    "complaint_type",
    "root_cause",
    "label",
    "fault",
    "slider_ts",
    "slider_ta",
    "slider_arp",
    "slider_hs",
    "slider_div",
    "config_conflict_like",
    "system_fault",
    "size_relevant_complaint",
    "activity_relevant_complaint",
    "extracted_portfolio_present",
    "extracted_market_present",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Stage 1b strict prompt review HTML.")
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_rows(table_name: str) -> list[dict[str, Any]]:
    table_name = validate_table_name(table_name)
    with connect_neon() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM {table_name}
            ORDER BY adapter_alignment_label, strategy_size_preference, slider_size_bucket, example_id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def parse_messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    return [dict(message) for message in value or [] if isinstance(message, dict)]


def message_html(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "unknown")
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        parts.append(
            '<section class="message">'
            f'<div class="role">{esc(role)}</div>'
            f'<pre>{esc(content)}</pre>'
            "</section>"
        )
    return "\n".join(parts)


def label_html(row: dict[str, Any]) -> str:
    items = []
    for field in LABEL_FIELDS:
        items.append(
            f'<div class="kv"><span>{esc(field)}</span><strong>{esc(row.get(field))}</strong></div>'
        )
    return "\n".join(items)


def summary_html(rows: list[dict[str, Any]], table_name: str) -> str:
    alignment = Counter(str(row.get("adapter_alignment_label")) for row in rows)
    buckets = Counter(
        (
            str(row.get("adapter_alignment_label")),
            str(row.get("strategy_size_preference")),
            str(row.get("slider_size_bucket")),
        )
        for row in rows
    )
    complaint = Counter(str(row.get("complaint_type")) for row in rows)
    bucket_lines = "".join(
        f"<li><code>{esc(label)} / {esc(pref)} / {esc(bucket)}</code>: {count}</li>"
        for (label, pref, bucket), count in buckets.most_common()
    )
    complaint_lines = "".join(
        f"<li><code>{esc(name)}</code>: {count}</li>" for name, count in complaint.most_common()
    )
    return f"""
    <section class="summary">
      <div>
        <h2>Dataset</h2>
        <p><code>{esc(table_name)}</code></p>
        <p>{len(rows)} rows total. {alignment.get("aligned", 0)} aligned, {alignment.get("conflict", 0)} conflict.</p>
      </div>
      <div>
        <h2>Alignment / Preference / Bucket</h2>
        <ul>{bucket_lines}</ul>
      </div>
      <div>
        <h2>Complaint Types</h2>
        <ul>{complaint_lines}</ul>
      </div>
    </section>
    """


def row_html(row: dict[str, Any], index: int) -> str:
    messages = parse_messages(row.get("prompt_messages_json"))
    label = str(row.get("adapter_alignment_label") or "")
    pref = str(row.get("strategy_size_preference") or "")
    bucket = str(row.get("slider_size_bucket") or "")
    search_blob = " ".join(
        str(row.get(field) or "")
        for field in (
            "example_id",
            "trace_id",
            "source_example_id",
            "adapter_alignment_label",
            "strategy_size_preference",
            "slider_size_bucket",
            "complaint_type",
            "root_cause",
            "complaint_text",
            "prompt_text",
        )
    ).lower()
    return f"""
    <article
      class="card"
      data-label="{esc(label)}"
      data-pref="{esc(pref)}"
      data-bucket="{esc(bucket)}"
      data-search="{esc(search_blob)}"
    >
      <header>
        <div>
          <div class="index">#{index}</div>
          <h2>{esc(row.get("example_id"))}</h2>
          <p>{esc(row.get("trace_id"))}</p>
        </div>
        <div class="badges">
          <span class="badge {esc(label)}">{esc(label)}</span>
          <span class="badge">{esc(pref)} strategy</span>
          <span class="badge">{esc(bucket)} setting</span>
        </div>
      </header>
      <div class="labels">{label_html(row)}</div>
      <details>
        <summary>Complaint / source context</summary>
        <div class="context">
          <div><strong>source_example_id</strong><br><code>{esc(row.get("source_example_id"))}</code></div>
          <div><strong>complaint_text</strong><pre>{esc(row.get("complaint_text"))}</pre></div>
        </div>
      </details>
      <div class="prompt">{message_html(messages)}</div>
    </article>
    """


def render(rows: list[dict[str, Any]], table_name: str) -> str:
    cards = "\n".join(row_html(row, idx + 1) for idx, row in enumerate(rows))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage 1b Strict Prompt Review</title>
  <style>
    :root {{
      --bg: #0f1115;
      --panel: #171a21;
      --panel-2: #1f2430;
      --text: #edf0f4;
      --muted: #a4adbb;
      --line: #343b49;
      --accent: #7dd3fc;
      --conflict: #f97373;
      --aligned: #74d99f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ margin: 0; font-size: 15px; }}
    p {{ margin: 4px 0 0; color: var(--muted); }}
    code, pre {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 16px 0;
      background: linear-gradient(var(--bg), rgba(15,17,21,0.92));
      border-bottom: 1px solid var(--line);
      margin-bottom: 18px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) repeat(3, minmax(140px, 180px));
      gap: 10px;
      margin-top: 14px;
    }}
    input, select {{
      width: 100%;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      padding: 9px 10px;
      border-radius: 6px;
      font-size: 13px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 12px;
      margin-bottom: 18px;
    }}
    .summary > div, .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .summary > div {{ padding: 14px; }}
    .summary ul {{ margin: 8px 0 0; padding-left: 18px; color: var(--muted); }}
    .card {{ margin-bottom: 18px; overflow: hidden; }}
    .card.hidden {{ display: none; }}
    .card > header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
    }}
    .index {{ color: var(--accent); font: 12px ui-monospace, monospace; margin-bottom: 4px; }}
    .badges {{ display: flex; flex-wrap: wrap; align-content: flex-start; gap: 6px; justify-content: flex-end; }}
    .badge {{
      border: 1px solid var(--line);
      color: var(--muted);
      padding: 3px 7px;
      border-radius: 999px;
      font-size: 12px;
      white-space: nowrap;
    }}
    .badge.conflict {{ color: var(--conflict); border-color: color-mix(in srgb, var(--conflict), transparent 45%); }}
    .badge.aligned {{ color: var(--aligned); border-color: color-mix(in srgb, var(--aligned), transparent 45%); }}
    .labels {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 1px;
      background: var(--line);
      border-bottom: 1px solid var(--line);
    }}
    .kv {{ background: var(--panel); padding: 8px 10px; min-width: 0; }}
    .kv span {{ display: block; color: var(--muted); font-size: 11px; }}
    .kv strong {{ display: block; margin-top: 2px; font-size: 13px; overflow-wrap: anywhere; }}
    details {{ border-bottom: 1px solid var(--line); }}
    summary {{ cursor: pointer; padding: 10px 16px; color: var(--accent); font-size: 13px; }}
    .context {{ padding: 0 16px 14px; color: var(--muted); }}
    .context pre {{
      white-space: pre-wrap;
      color: var(--text);
      background: #10131a;
      border: 1px solid var(--line);
      padding: 10px;
      border-radius: 6px;
    }}
    .prompt {{ padding: 16px; display: grid; gap: 12px; }}
    .message {{
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #10131a;
    }}
    .role {{
      padding: 7px 10px;
      color: var(--accent);
      background: #151923;
      border-bottom: 1px solid var(--line);
      font: 12px ui-monospace, monospace;
      text-transform: uppercase;
    }}
    .message pre {{
      margin: 0;
      padding: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 12px;
      color: #e8edf5;
    }}
    .count {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
    @media (max-width: 900px) {{
      main {{ padding: 14px; }}
      .controls, .summary {{ grid-template-columns: 1fr; }}
      .card > header {{ flex-direction: column; }}
      .badges {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="topbar">
      <h1>Stage 1b Strict Prompt Review</h1>
      <p>All prompts and labels from <code>{esc(table_name)}</code>.</p>
      <div class="controls">
        <input id="search" placeholder="Search id, label, complaint, root cause, prompt text...">
        <select id="label">
          <option value="">all labels</option>
          <option value="aligned">aligned</option>
          <option value="conflict">conflict</option>
        </select>
        <select id="pref">
          <option value="">all strategy prefs</option>
          <option value="small">small strategy</option>
          <option value="large">large strategy</option>
        </select>
        <select id="bucket">
          <option value="">all setting buckets</option>
          <option value="small">small setting</option>
          <option value="large">large setting</option>
        </select>
      </div>
      <div class="count"><span id="visible-count">{len(rows)}</span> visible / {len(rows)} total</div>
    </div>
    {summary_html(rows, table_name)}
    <section id="cards">{cards}</section>
  </main>
  <script>
    const cards = Array.from(document.querySelectorAll('.card'));
    const search = document.getElementById('search');
    const label = document.getElementById('label');
    const pref = document.getElementById('pref');
    const bucket = document.getElementById('bucket');
    const count = document.getElementById('visible-count');
    function applyFilters() {{
      const q = search.value.trim().toLowerCase();
      let visible = 0;
      for (const card of cards) {{
        const ok =
          (!q || card.dataset.search.includes(q)) &&
          (!label.value || card.dataset.label === label.value) &&
          (!pref.value || card.dataset.pref === pref.value) &&
          (!bucket.value || card.dataset.bucket === bucket.value);
        card.classList.toggle('hidden', !ok);
        if (ok) visible++;
      }}
      count.textContent = String(visible);
    }}
    for (const el of [search, label, pref, bucket]) {{
      el.addEventListener('input', applyFilters);
    }}
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    table_name = validate_table_name(args.table)
    rows = load_rows(table_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows, table_name), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "table": table_name, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

