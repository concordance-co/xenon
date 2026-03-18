import { useFetch } from '../hooks/useApi'
import { useConfig, buildIngestCmd, buildOutcomesCmd } from '../hooks/useConfig'
import type { IngestData, OutcomesData } from '../types/api'
import { Tip } from './Tip'
import s from './shared.module.css'

interface Props {
  onRun: (cmd: string) => void
}

export function IngestView({ onRun }: Props) {
  const { data, loading, error } = useFetch<IngestData>('/api/ingest')
  const { data: outcomesData } = useFetch<OutcomesData>('/api/outcomes')
  const { config, update } = useConfig()
  const c = config.ingest
  const oc = config.outcomes

  const cmd = buildIngestCmd(c)
  const outcomesCmd = buildOutcomesCmd(oc)

  if (error) return <div className={s.empty}>Failed to load data: {error}</div>
  if (loading || !data) return <div className={s.empty}>Loading...</div>

  const fmtPct = (v: number | null) => v != null ? `${v.toFixed(2)}%` : '\u2014'
  const fmtRate = (v: number | null) => v != null ? `${(v * 100).toFixed(1)}%` : '\u2014'

  return (
    <div>
      <p className={s.phaseDesc}>
        Pulls trading agent data from the Terminal Markets API on Modal cloud.
        Discovers vaults from the leaderboard — either the top performers by
        PnL or a random sample for diversity — then fetches their strategies,
        inference logs (the LLM's decision-making records), full-log payloads
        (stored as JSONB), and on-chain swap history. All data goes directly
        to Neon Postgres.
      </p>

      <div className={s.statsRow}>
        <Stat label="Vaults" value={data.vault_count} />
        <Stat label="Strategies" value={data.strategy_count} />
        <Stat label="Inference Logs" value={data.log_count} />
        <Stat label="Full Logs" value={data.full_log_count} />
        <Stat label="Coverage" value={`${data.full_log_coverage_pct}%`} />
        <Stat label="Parse Errors" value={data.parse_error_count} />
      </div>

      <div className={s.grid2}>
        <div>
          <div className={s.panel}>
            <div className={s.panelHead}>
              <span className={s.panelTitle}>Database Tables</span>
            </div>
            <div className={s.panelBody}>
              {data.tables.length > 0 ? (
                <table className={s.table}>
                  <thead>
                    <tr><th>Table</th><th style={{ textAlign: 'right' }}>Rows</th></tr>
                  </thead>
                  <tbody>
                    {data.tables.map(t => (
                      <tr key={t.name}>
                        <td className="mono">{t.name}</td>
                        <td className="mono" style={{ textAlign: 'right' }}>{t.count.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className={s.empty}>No database found. Run ingest first.</div>
              )}
            </div>
          </div>

          {outcomesData && (
            <div className={s.panel}>
              <div className={s.panelHead}>
                <span className={s.panelTitle}>Trade Outcomes</span>
              </div>
              <div className={s.panelBody}>
                <div className={s.statsRow}>
                  <Stat label="Outcomes" value={outcomesData.total_outcomes} tip="Swaps that have had their forward-looking PnL computed from candle data." />
                  <Stat label="Unlabeled" value={outcomesData.unlabeled_swaps} tip="Swaps still missing outcome labels. Run outcomes to fetch candle prices at 1h/4h/1d after each trade and compute PnL." />
                  <Stat label="Win Rate 1h" value={fmtRate(outcomesData.win_rate_1h)} tip="Fraction of trades that were profitable 1 hour after execution. A trade is profitable if PnL > 0% (price moved in the direction of the trade)." />
                  <Stat label="Avg PnL 1h" value={fmtPct(outcomesData.avg_pnl_1h)} tip="Mean percentage return 1 hour after trade execution, across all labeled swaps." />
                  <Stat label="Avg PnL 4h" value={fmtPct(outcomesData.avg_pnl_4h)} tip="Mean percentage return 4 hours after trade execution." />
                  <Stat label="Avg PnL 1d" value={fmtPct(outcomesData.avg_pnl_1d)} tip="Mean percentage return 1 day after trade execution." />
                </div>
                {outcomesData.risk_breakdown.length > 0 && (
                  <table className={s.table}>
                    <thead>
                      <tr>
                        <th>Risk</th>
                        <th style={{ textAlign: 'right' }}>Count</th>
                        <th style={{ textAlign: 'right' }}>Avg PnL 1h</th>
                        <th style={{ textAlign: 'right' }}>Avg PnL 4h</th>
                        <th style={{ textAlign: 'right' }}>Avg PnL 1d</th>
                        <th style={{ textAlign: 'right' }}>Win Rate 1h</th>
                      </tr>
                    </thead>
                    <tbody>
                      {outcomesData.risk_breakdown.map(r => (
                        <tr key={r.risk_level}>
                          <td className="mono">{r.risk_level}</td>
                          <td className="mono" style={{ textAlign: 'right' }}>{r.count.toLocaleString()}</td>
                          <td className="mono" style={{ textAlign: 'right' }}>{fmtPct(r.avg_pnl_1h)}</td>
                          <td className="mono" style={{ textAlign: 'right' }}>{fmtPct(r.avg_pnl_4h)}</td>
                          <td className="mono" style={{ textAlign: 'right' }}>{fmtPct(r.avg_pnl_1d)}</td>
                          <td className="mono" style={{ textAlign: 'right' }}>{fmtRate(r.win_rate_1h)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
                <div style={{ marginTop: 12 }}>
                  <div className={s.configForm}>
                    <div className={s.field}>
                      <label className={s.fieldLabel}>
                        Concurrency
                        <Tip text="Parallel candle API requests. Higher values speed up processing but may hit rate limits." />
                      </label>
                      <input
                        type="number"
                        className={s.fieldInput}
                        value={oc.concurrency}
                        min={1}
                        max={50}
                        onChange={e => update('outcomes', { concurrency: Number(e.target.value) || 5 })}
                      />
                    </div>
                    <div className={s.field}>
                      <label className={s.fieldLabel}>
                        Limit
                        <Tip text="Max swaps to process. 0 = all unlabeled." />
                      </label>
                      <input
                        type="number"
                        className={s.fieldInput}
                        value={oc.limit}
                        min={0}
                        onChange={e => update('outcomes', { limit: Number(e.target.value) || 0 })}
                      />
                    </div>
                  </div>
                  <div className={s.generatedCmd}>{outcomesCmd}</div>
                  <div className={s.btnRow}>
                    <button
                      className={`${s.btn} ${s.btnAccent}`}
                      onClick={() => onRun(outcomesCmd)}
                    >
                      Run Outcomes
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className={s.panel}>
          <div className={s.panelHead}>
            <span className={s.panelTitle}>Ingest Configuration</span>
          </div>
          <div className={s.panelBody}>
            <div className={s.configForm}>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Selection Mode
                  <Tip text="'Top' picks the highest-ranked vaults by PnL. 'Random' pages through all vaults on the leaderboard and randomly samples N of them. 'Existing' skips the leaderboard and only processes vaults already in the database. 'Backfill' fills in missing full logs and null payloads for existing data, then runs swaps." />
                </label>
                <div className={s.modeToggle}>
                  <button
                    className={c.selection === 'top' ? s.modeBtnActive : s.modeBtn}
                    onClick={() => update('ingest', { selection: 'top' })}
                  >
                    Top N
                  </button>
                  <button
                    className={c.selection === 'random' ? s.modeBtnActive : s.modeBtn}
                    onClick={() => update('ingest', { selection: 'random' })}
                  >
                    Random
                  </button>
                  <button
                    className={c.selection === 'existing' ? s.modeBtnActive : s.modeBtn}
                    onClick={() => update('ingest', { selection: 'existing' })}
                  >
                    Existing
                  </button>
                  <button
                    className={c.selection === 'backfill' ? s.modeBtnActive : s.modeBtn}
                    onClick={() => update('ingest', { selection: 'backfill' })}
                  >
                    Backfill
                  </button>
                </div>
              </div>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  {c.selection === 'random' ? 'Sample Size' : 'Top N Vaults'}
                  <Tip text={c.selection === 'random'
                    ? "How many vaults to randomly sample from the full leaderboard."
                    : "Number of top-performing vaults to ingest from the leaderboard, ranked by total PnL. Higher = more data but longer ingest time."
                  } />
                </label>
                <input
                  type="number"
                  className={s.fieldInput}
                  value={c.topN}
                  min={1}
                  onChange={e => update('ingest', { topN: Number(e.target.value) || 1 })}
                />
              </div>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Concurrency
                  <Tip text="Max parallel API requests when fetching full logs. Higher = faster but risks rate limiting. The API client has built-in retry logic." />
                </label>
                <input
                  type="number"
                  className={s.fieldInput}
                  value={c.concurrency}
                  min={1}
                  max={100}
                  onChange={e => update('ingest', { concurrency: Number(e.target.value) || 1 })}
                />
              </div>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Requests/sec
                  <Tip text="API rate limit in requests per second. Higher = faster ingest but risks 429s. 502s are automatically deferred and retried later." />
                </label>
                <input
                  type="number"
                  className={s.fieldInput}
                  value={c.rps}
                  min={1}
                  max={50}
                  step={1}
                  onChange={e => update('ingest', { rps: Number(e.target.value) || 6 })}
                />
              </div>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Exclude Reasoning
                  <Tip text="Skip storing the LLM's chain-of-thought reasoning from full logs. Reduces DB size but loses the model's internal reasoning trace." />
                </label>
                <label className={s.toggle}>
                  <span
                    className={c.excludeReasoning ? s.toggleTrackOn : s.toggleTrack}
                    onClick={() => update('ingest', { excludeReasoning: !c.excludeReasoning })}
                  />
                  <span>{c.excludeReasoning ? 'Yes' : 'No'}</span>
                </label>
              </div>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Skip Deferred
                  <Tip text="Skip retrying previously deferred logs (502s) at the end of the run. Useful for faster runs when you don't need to recover failed fetches." />
                </label>
                <label className={s.toggle}>
                  <span
                    className={c.skipDeferred ? s.toggleTrackOn : s.toggleTrack}
                    onClick={() => update('ingest', { skipDeferred: !c.skipDeferred })}
                  />
                  <span>{c.skipDeferred ? 'Yes' : 'No'}</span>
                </label>
              </div>
            </div>

            <div className={s.generatedCmd}>{cmd}</div>

            <div className={s.btnRow}>
              <button
                className={`${s.btn} ${s.btnAccent}`}
                onClick={() => onRun(cmd)}
              >
                Run Ingest
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value, tip }: { label: string; value: string | number; tip?: string }) {
  return (
    <div className={s.stat}>
      <div className={s.statLabel}>{label}{tip && <Tip text={tip} />}</div>
      <div className={s.statValue}>
        {typeof value === 'number' ? value.toLocaleString() : value}
      </div>
    </div>
  )
}
