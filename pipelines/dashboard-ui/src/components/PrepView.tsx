import { useFetch } from '../hooks/useApi'
import { useConfig, buildPrepCmd } from '../hooks/useConfig'
import type { PrepData } from '../types/api'
import { Tip } from './Tip'
import s from './shared.module.css'

interface Props {
  onRun: (cmd: string) => void
}

export function PrepView({ onRun }: Props) {
  const { data, loading } = useFetch<PrepData>('/api/prep')
  const { config, update } = useConfig()
  const c = config.prep

  const cmd = buildPrepCmd(c, c.mode)

  if (loading || !data) return <div className={s.empty}>Loading...</div>

  const total = data.high_quality + data.medium_quality + data.low_quality
  const pct = (n: number) => total > 0 ? `${((n / total) * 100).toFixed(0)}%` : '—'

  return (
    <div>
      <p className={s.phaseDesc}>
        Transforms raw ingested data into labeled examples for interpretability
        research. Parses each inference log to extract the LLM's decision
        (trade vs. observe), trade side (buy/sell), asset, risk tolerance, and
        profitability outcomes. Assigns quality tiers based on completeness.
        Samples balanced subsets of trades, observations, and paired examples
        from the same vault. Exports to parquet for downstream capture.
      </p>

      <div className={s.statsRow}>
        <Stat label="Total Examples" value={data.total_examples} />
        <Stat label="High Quality" value={`${data.high_quality} (${pct(data.high_quality)})`} />
        <Stat label="Medium Quality" value={`${data.medium_quality} (${pct(data.medium_quality)})`} />
        <Stat label="Low Quality" value={`${data.low_quality} (${pct(data.low_quality)})`} />
        <Stat label="Trades" value={data.trade_count} />
        <Stat label="Observations" value={data.observation_count} />
      </div>

      <div className={s.grid2}>
        <div>
          {data.label_distribution.length > 0 && (
            <div className={s.panel}>
              <div className={s.panelHead}>
                <span className={s.panelTitle}>Label Distribution</span>
              </div>
              <div className={s.panelBody}>
                <table className={s.table}>
                  <thead>
                    <tr>
                      <th>Decision Type</th>
                      <th style={{ textAlign: 'right' }}>Count</th>
                      <th>Trade Side</th>
                      <th style={{ textAlign: 'right' }}>Avg Risk</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.label_distribution.map((r, i) => (
                      <tr key={i}>
                        <td>{r.decision_type ?? '—'}</td>
                        <td className="mono" style={{ textAlign: 'right' }}>{r.count.toLocaleString()}</td>
                        <td className="mono">{r.trade_side ?? '—'}</td>
                        <td className="mono" style={{ textAlign: 'right' }}>{r.avg_risk?.toFixed(1) ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className={s.panel}>
            <div className={s.panelHead}>
              <span className={s.panelTitle}>Export Files</span>
            </div>
            <div className={s.panelBody}>
              {data.export_files.length > 0 ? (
                <table className={s.table}>
                  <thead><tr><th>File</th><th style={{ textAlign: 'right' }}>Size</th></tr></thead>
                  <tbody>
                    {data.export_files.map(f => (
                      <tr key={f.name}>
                        <td className="mono">{f.name}</td>
                        <td className="mono" style={{ textAlign: 'right' }}>{f.size}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className={s.empty}>No exports yet. Run data prep with export enabled.</div>
              )}
            </div>
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
                onClick={() => update('prep', { mode: 'local' })}
              >
                Local
                <Tip text="Run prep on your machine using the local SQLite DB." />
              </button>
              <button
                className={c.mode === 'modal' ? s.modeBtnActive : s.modeBtn}
                onClick={() => update('prep', { mode: 'modal' })}
              >
                Modal
                <Tip text="Run prep on Modal cloud. Uses the DB on the xenon-data volume. Exports (parquet) are written directly to the volume for capture to use." />
              </button>
            </div>
            <div className={s.configForm}>
              {c.mode === 'local' && (
                <div className={s.field}>
                  <label className={s.fieldLabel}>
                    DB Path
                    <Tip text="Path to the SQLite database created by ingest. Must match the ingest DB path to find the raw data." />
                  </label>
                  <input
                    type="text"
                    className={s.fieldInput}
                    value={c.dbPath}
                    onChange={e => update('prep', { dbPath: e.target.value })}
                  />
                </div>
              )}
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Trade Sample
                  <Tip text="Max trade examples to include. Controls dataset balance — too many trades relative to observations skews probes. Set to 0 for all." />
                </label>
                <input
                  type="number"
                  className={s.fieldInput}
                  value={c.tradeSampleSize}
                  min={0}
                  onChange={e => update('prep', { tradeSampleSize: Number(e.target.value) || 0 })}
                />
              </div>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Observation Sample
                  <Tip text="Max observation (non-trade) examples. Balance this with trade sample size for clean binary classification in probes." />
                </label>
                <input
                  type="number"
                  className={s.fieldInput}
                  value={c.observationSampleSize}
                  min={0}
                  onChange={e => update('prep', { observationSampleSize: Number(e.target.value) || 0 })}
                />
              </div>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Paired Sample
                  <Tip text="Max paired examples — trade + observation from the same vault, enabling controlled comparisons where context is held constant." />
                </label>
                <input
                  type="number"
                  className={s.fieldInput}
                  value={c.pairedSampleSize}
                  min={0}
                  onChange={e => update('prep', { pairedSampleSize: Number(e.target.value) || 0 })}
                />
              </div>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Export Parquet
                  <Tip text="Write labeled examples as a parquet file. Required for the capture phase — this is the input format it reads." />
                </label>
                <label className={s.toggle}>
                  <span
                    className={c.exportParquet ? s.toggleTrackOn : s.toggleTrack}
                    onClick={() => update('prep', { exportParquet: !c.exportParquet })}
                  />
                  <span>{c.exportParquet ? 'Yes' : 'No'}</span>
                </label>
              </div>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Export JSONL
                  <Tip text="Also write examples as JSONL. Useful for manual inspection or loading into other tools. Not required for the pipeline." />
                </label>
                <label className={s.toggle}>
                  <span
                    className={c.exportJsonl ? s.toggleTrackOn : s.toggleTrack}
                    onClick={() => update('prep', { exportJsonl: !c.exportJsonl })}
                  />
                  <span>{c.exportJsonl ? 'Yes' : 'No'}</span>
                </label>
              </div>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  All Decisions
                  <Tip text="Include all decision types, not just high-quality ones. Increases dataset size but adds noisier examples that may reduce probe accuracy." />
                </label>
                <label className={s.toggle}>
                  <span
                    className={c.includeAllDecisions ? s.toggleTrackOn : s.toggleTrack}
                    onClick={() => update('prep', { includeAllDecisions: !c.includeAllDecisions })}
                  />
                  <span>{c.includeAllDecisions ? 'Yes' : 'No'}</span>
                </label>
              </div>
            </div>

            <div className={s.generatedCmd}>{cmd}</div>

            <div className={s.btnRow}>
              <button
                className={`${s.btn} ${s.btnAccent}`}
                onClick={() => onRun(cmd)}
              >
                Run Prep
              </button>
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
