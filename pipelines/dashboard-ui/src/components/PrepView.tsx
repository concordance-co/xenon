import { useFetch } from '../hooks/useApi'
import { useConfig, buildPrepCmd } from '../hooks/useConfig'
import type { PrepData } from '../types/api'
import { Tip } from './Tip'
import s from './shared.module.css'

interface Props {
  onRun: (cmd: string) => void
}

export function PrepView({ onRun }: Props) {
  const { data, loading, error } = useFetch<PrepData>('/api/prep')
  const { config, update } = useConfig()
  const c = config.prep

  const cmd = buildPrepCmd(c)

  if (error) return <div className={s.empty}>Failed to load data: {error}</div>
  if (loading || !data) return <div className={s.empty}>Loading...</div>

  const total = data.high_quality + data.medium_quality + data.low_quality
  const pct = (n: number) => total > 0 ? `${((n / total) * 100).toFixed(0)}%` : '—'

  return (
    <div>
      <p className={s.phaseDesc}>
        Transforms raw ingested data into labeled examples for interpretability
        research. Reads full-log JSONB payloads from Neon Postgres, parses each
        inference log to extract the LLM's decision (trade vs. observe), trade
        side (buy/sell), asset, risk tolerance, and profitability outcomes.
        Assigns quality tiers based on completeness. Writes the denormalized
        <code> interp_examples_v0</code> table back to Neon.
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
        </div>

        <div className={s.panel}>
          <div className={s.panelHead}>
            <span className={s.panelTitle}>Configuration</span>
          </div>
          <div className={s.panelBody}>
            <div className={s.configForm}>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Limit
                  <Tip text="Max number of full logs to process. 0 = default (50,000)." />
                </label>
                <input
                  type="number"
                  className={s.fieldInput}
                  value={c.limit}
                  min={0}
                  onChange={e => update('prep', { limit: Number(e.target.value) || 0 })}
                />
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
