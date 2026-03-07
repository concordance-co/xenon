import { useFetch } from '../hooks/useApi'
import { useConfig, buildIngestCmd } from '../hooks/useConfig'
import type { IngestData } from '../types/api'
import { Tip } from './Tip'
import s from './shared.module.css'

interface Props {
  onRun: (cmd: string) => void
}

export function IngestView({ onRun }: Props) {
  const { data, loading } = useFetch<IngestData>('/api/ingest')
  const { config, update } = useConfig()
  const c = config.ingest

  const cmd = buildIngestCmd(c, c.mode)

  if (loading || !data) return <div className={s.empty}>Loading...</div>

  return (
    <div>
      <p className={s.phaseDesc}>
        Pulls trading agent data from the Terminal Markets API. Discovers
        vaults from the leaderboard — either the top performers by PnL or
        a random sample for diversity — then fetches their strategies,
        inference logs (the LLM's decision-making records), and on-chain
        swap history. Run locally or on Modal (cloud). For Modal, upload
        your local DB first to continue from existing data.
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

        <div className={s.panel}>
          <div className={s.panelHead}>
            <span className={s.panelTitle}>Configuration</span>
          </div>
          <div className={s.panelBody}>
            <div className={s.modeToggle}>
              <button
                className={c.mode === 'local' ? s.modeBtnActive : s.modeBtn}
                onClick={() => update('ingest', { mode: 'local' })}
              >
                Local
                <Tip text="Run ingest on your machine. Data saved to the local SQLite DB." />
              </button>
              <button
                className={c.mode === 'modal' ? s.modeBtnActive : s.modeBtn}
                onClick={() => update('ingest', { mode: 'modal' })}
              >
                Modal
                <Tip text="Run ingest on Modal cloud. DB lives on the xenon-data volume. Upload your local DB first to continue from existing data, or start fresh on Modal." />
              </button>
            </div>

            <div className={s.configForm}>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Selection Mode
                  <Tip text="'Top' picks the highest-ranked vaults by PnL. 'Random' pages through all vaults on the leaderboard and randomly samples N of them — useful for getting a diverse cross-section of traders instead of just the top performers." />
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
              {c.mode === 'local' && (
                <div className={s.field}>
                  <label className={s.fieldLabel}>
                    DB Path
                    <Tip text="Path to the SQLite database file. All ingested data (vaults, strategies, logs, swaps) is stored here. Shared with the prep phase." />
                  </label>
                  <input
                    type="text"
                    className={s.fieldInput}
                    value={c.dbPath}
                    onChange={e => update('ingest', { dbPath: e.target.value })}
                  />
                </div>
              )}
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
            </div>

            <div className={s.generatedCmd}>{cmd}</div>

            <div className={s.btnRow}>
              <button
                className={`${s.btn} ${s.btnAccent}`}
                onClick={() => onRun(cmd)}
              >
                Run Ingest
              </button>
              {c.mode === 'modal' && (
                <>
                  <button className={s.btn} onClick={() => onRun('./scripts/modal_capture.sh upload-db')}>
                    Upload DB
                  </button>
                  <button className={s.btn} onClick={() => onRun('./scripts/modal_capture.sh download-db')}>
                    Download DB
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className={s.stat}>
      <div className={s.statLabel}>{label}</div>
      <div className={s.statValue}>
        {typeof value === 'number' ? value.toLocaleString() : value}
      </div>
    </div>
  )
}
