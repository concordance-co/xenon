import { useState, useEffect, useCallback } from 'react'
import { useFetch, fetchJson } from '../hooks/useApi'
import { useConfig, buildAnalysisCmd } from '../hooks/useConfig'
import type { AnalysisData, ProbeRow, ExpertRow, FileTreeEntry } from '../types/api'
import { ProbeChart } from './ProbeChart'
import { Tip } from './Tip'
import s from './shared.module.css'

interface Props {
  onRun: (cmd: string) => void
  refreshKey?: number
}

export function AnalysisView({ onRun, refreshKey }: Props) {
  const { data, loading, error, refetch } = useFetch<AnalysisData>('/api/analysis')
  const { config, update } = useConfig()
  const c = config.analysis
  const [probeData, setProbeData] = useState<ProbeRow[] | null>(null)
  const [probeFile, setProbeFile] = useState<string | null>(null)
  const [expertData, setExpertData] = useState<ExpertRow[] | null>(null)
  const [showExperts, setShowExperts] = useState(false)
  const [lightboxImg, setLightboxImg] = useState<string | null>(null)

  // Lightbox keyboard nav
  const handleKey = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') setLightboxImg(null)
    if (!data?.pca_images?.length) return
    const imgNames = data.pca_images.map(i => i.name)
    setLightboxImg(cur => {
      if (!cur) return cur
      const idx = imgNames.indexOf(cur)
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') return imgNames[Math.min(idx + 1, imgNames.length - 1)]
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') return imgNames[Math.max(idx - 1, 0)]
      return cur
    })
  }, [data?.pca_images])
  useEffect(() => {
    if (lightboxImg) {
      window.addEventListener('keydown', handleKey)
      return () => window.removeEventListener('keydown', handleKey)
    }
  }, [lightboxImg, handleKey])

  const cmd = buildAnalysisCmd(c)

  // Refetch when a command finishes
  useEffect(() => {
    if (refreshKey) {
      refetch()
      setProbeFile(null)
      setProbeData(null)
    }
  }, [refreshKey]) // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-load first probe file
  useEffect(() => {
    if (data?.probe_files?.length && !probeFile) {
      loadProbe(data.probe_files[0])
    }
  }, [data]) // eslint-disable-line react-hooks/exhaustive-deps

  async function loadProbe(file: string) {
    setProbeFile(file)
    try {
      const result = await fetchJson<{ rows: ProbeRow[] }>(
        `/api/analysis/probe-data?file=${encodeURIComponent(file)}`
      )
      setProbeData(result.rows)
    } catch {
      setProbeData([])
    }
  }

  async function loadExperts() {
    setShowExperts(true)
    try {
      const result = await fetchJson<{ rows: ExpertRow[] }>('/api/analysis/expert-data')
      setExpertData(result.rows)
    } catch {
      setExpertData([])
    }
  }

  if (error) return <div className={s.empty}>Failed to load data: {error}</div>
  if (loading || !data) return <div className={s.empty}>Loading...</div>

  return (
    <div>
      <p className={s.phaseDesc}>
        Tests whether the model's internal routing patterns encode
        meaningful information about trading decisions. Runs linear probes
        (logistic regression) at each layer to predict targets like
        decision type, trade side, profitability, and risk tolerance from
        router logits. Computes selectivity (accuracy above shuffled
        control) to find layers where routing is genuinely informative vs.
        noise. Also runs expert specialization analysis (which of the 128
        experts discriminate between classes via Cohen's d) and PCA
        visualization of the activation space.
      </p>

      <div className={s.statsRow}>
        <Stat label="Probe Results" value={data.probe_files.length} />
        <Stat label="Expert Analysis" value={data.has_expert_specialization ? 'Available' : 'None'} />
        <Stat label="PCA Plots" value={data.pca_images.length} />
        <Stat label="Total Files" value={data.total_results} />
      </div>

      <div className={s.grid2}>
        <div className={s.panel}>
          <div className={s.panelHead}>
            <span className={s.panelTitle}>Analysis Configuration</span>
          </div>
          <div className={s.panelBody}>
            <div className={s.configForm}>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Target
                  <Tip text="What to predict from activations. decision_type: trade vs. observe. trade_side: buy vs. sell (trades only). executed_valence: sell=bearish, observe=neutral, buy=bullish. risk_tolerance: low/mid/high (3-class). asset: which token was traded (multi-class)." />
                </label>
                <select
                  className={s.fieldSelect}
                  value={c.target}
                  onChange={e => update('analysis', { target: e.target.value as typeof c.target })}
                >
                  <option value="decision_type">Decision Type</option>
                  <option value="trade_side">Trade Side</option>
                  <option value="was_profitable_1h">Profitability (1h)</option>
                  <option value="executed_valence">Executed Valence</option>
                  <option value="risk_tolerance">Risk Tolerance</option>
                  <option value="asset">Asset</option>
                </select>
              </div>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Analyses
                  <Tip text="Select which analyses to run. Probe: linear classifier per layer. Experts: which MoE experts discriminate between classes. PCA: 2D scatter plots of activations." />
                </label>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xs)' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)', fontSize: '0.8125rem', cursor: 'pointer' }}>
                    <input type="checkbox" checked={c.runProbe} onChange={e => update('analysis', { runProbe: e.target.checked })} />
                    Probe
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)', fontSize: '0.8125rem', cursor: 'pointer' }}>
                    <input type="checkbox" checked={c.runExperts} onChange={e => update('analysis', { runExperts: e.target.checked })} />
                    Experts
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)', fontSize: '0.8125rem', cursor: 'pointer' }}>
                    <input type="checkbox" checked={c.runPca} onChange={e => update('analysis', { runPca: e.target.checked })} />
                    PCA
                  </label>
                </div>
              </div>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Data Source
                  <Tip text="router: use MoE router logits (128-dim per layer, which experts are selected). residual: use hidden state vectors (much higher dim). Router is the primary signal for MoE interpretability." />
                </label>
                <select
                  className={s.fieldSelect}
                  value={c.dataSource}
                  onChange={e => update('analysis', { dataSource: e.target.value as 'router' | 'residual' })}
                >
                  <option value="router">Router Logits</option>
                  <option value="residual">Residual Stream</option>
                </select>
              </div>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Pooling
                  <Tip text="How to reduce token-level activations to a single vector. last_token: use the final token (where the model commits to its decision). mean_pool: average across all tokens." />
                </label>
                <select
                  className={s.fieldSelect}
                  value={c.pooling}
                  onChange={e => update('analysis', { pooling: e.target.value as 'last_token' | 'mean_pool' })}
                >
                  <option value="last_token">Last Token</option>
                  <option value="mean_pool">Mean Pool</option>
                </select>
              </div>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Layers
                  <Tip text="Comma-separated layer indices to analyze. Empty = all layers. Analyzing all 48 layers gives the full picture but takes longer. Try a subset first." />
                </label>
                <input
                  type="text"
                  className={s.fieldInput}
                  value={c.layers}
                  placeholder="all layers"
                  onChange={e => update('analysis', { layers: e.target.value })}
                />
              </div>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  CV Folds
                  <Tip text="Number of cross-validation folds for probe accuracy. More folds = more robust estimate but slower. 5 is standard, 3 is fine for quick checks." />
                </label>
                <input
                  type="number"
                  className={s.fieldInput}
                  value={c.nFolds}
                  min={2}
                  max={20}
                  onChange={e => update('analysis', { nFolds: Number(e.target.value) || 5 })}
                />
              </div>
              <div className={s.field}>
                <label className={s.fieldLabel}>
                  Limit
                  <Tip text="Max examples to use in analysis. 0 = all. A smaller limit runs faster for quick sanity checks before full analysis." />
                </label>
                <input
                  type="number"
                  className={s.fieldInput}
                  value={c.limit}
                  min={0}
                  placeholder="0 = all"
                  onChange={e => update('analysis', { limit: Number(e.target.value) || 0 })}
                />
              </div>
            </div>

            <div className={s.generatedCmd}>{cmd}</div>

            <div className={s.btnRow}>
              <button
                className={`${s.btn} ${s.btnAccent}`}
                onClick={() => onRun(cmd)}
              >
                Run Analysis
              </button>
              <button className={s.btn} onClick={() => onRun('./scripts/modal_capture.sh download-results')}>
                Download Results
              </button>
            </div>
          </div>
        </div>

        <div className={s.panel}>
          <div className={s.panelHead}>
            <span className={s.panelTitle}>Result Files</span>
          </div>
          <div className={s.panelBody} style={{ maxHeight: 400, overflowY: 'auto' }}>
            {data.file_tree?.length > 0 ? (
              <FileTree entries={data.file_tree} depth={0} probeFile={probeFile} onLoadProbe={loadProbe} />
            ) : (
              <div className={s.empty}>No results yet. Run analysis first.</div>
            )}
          </div>
        </div>
      </div>

      {/* Probe chart */}
      {probeData && probeData.length > 0 && (
        <div className={s.panel}>
          <div className={s.panelHead}>
            <span className={s.panelTitle}>
              Probe Accuracy
              {probeFile && (
                <span style={{ fontWeight: 400, color: 'var(--text-3)', marginLeft: 8, fontSize: '0.75rem' }}>
                  {probeFile.replace('.parquet', '')}
                </span>
              )}
            </span>
            {data.probe_files.length > 1 && (
              <div style={{ display: 'flex', gap: 4 }}>
                {data.probe_files.map(f => (
                  <button
                    key={f}
                    className={`${s.btn} ${probeFile === f ? s.btnAccent : ''}`}
                    onClick={() => loadProbe(f)}
                    style={{ fontSize: '0.6875rem' }}
                  >
                    {f.replace('probe_', '').replace('.parquet', '')}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className={s.panelBody}>
            <ProbeChart data={probeData} />

            <div style={{ marginTop: 'var(--space-xl)', overflowX: 'auto' }}>
              <table className={s.table}>
                <thead>
                  <tr>
                    <th>Layer</th>
                    <th style={{ textAlign: 'right' }}>Accuracy</th>
                    <th style={{ textAlign: 'right' }}>Balanced</th>
                    <th style={{ textAlign: 'right' }}>Selectivity</th>
                    <th style={{ textAlign: 'right' }}>Majority</th>
                    <th style={{ textAlign: 'right' }}>Shuffled</th>
                    <th style={{ textAlign: 'right' }}>N</th>
                  </tr>
                </thead>
                <tbody>
                  {probeData.map(r => (
                    <tr key={r.layer} className={r.selectivity > 0.05 ? s.highlight : ''}>
                      <td className="mono">{r.layer}</td>
                      <td className="mono" style={{ textAlign: 'right' }}>{r.accuracy_mean.toFixed(3)} <span style={{ color: 'var(--text-4)' }}>±{r.accuracy_std.toFixed(3)}</span></td>
                      <td className="mono" style={{ textAlign: 'right' }}>{r.balanced_accuracy.toFixed(3)}</td>
                      <td className="mono" style={{ textAlign: 'right', fontWeight: r.selectivity > 0.05 ? 700 : 400 }}>
                        {r.selectivity > 0 ? '+' : ''}{r.selectivity.toFixed(3)}
                      </td>
                      <td className="mono" style={{ textAlign: 'right', color: 'var(--text-3)' }}>{r.baseline_majority.toFixed(3)}</td>
                      <td className="mono" style={{ textAlign: 'right', color: 'var(--text-3)' }}>{r.baseline_shuffled.toFixed(3)}</td>
                      <td className="mono" style={{ textAlign: 'right' }}>{r.n_examples}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Expert specialization */}
      {data.has_expert_specialization && (
        <div className={s.panel}>
          <div className={s.panelHead}>
            <span className={s.panelTitle}>Expert Specialization</span>
            {!showExperts && (
              <button className={s.btn} onClick={loadExperts}>Load</button>
            )}
          </div>
          <div className={s.panelBody}>
            {!showExperts && (
              <div className={s.empty}>Click Load to view expert specialization data.</div>
            )}
            {showExperts && expertData && expertData.length > 0 && (
              <div style={{ overflowX: 'auto' }}>
                <table className={s.table}>
                  <thead>
                    <tr>
                      <th>Layer</th>
                      <th style={{ textAlign: 'right' }}>Expert</th>
                      <th style={{ textAlign: 'right' }}>Rank</th>
                      <th style={{ textAlign: 'right' }}>Discriminative Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {expertData.filter(r => r.rank < 5).map((r, i) => (
                      <tr key={i}>
                        <td className="mono">{r.layer}</td>
                        <td className="mono" style={{ textAlign: 'right' }}>{r.expert_id}</td>
                        <td className="mono" style={{ textAlign: 'right' }}>{r.rank}</td>
                        <td className="mono" style={{ textAlign: 'right', fontWeight: Math.abs(r.discriminative_score) > 1 ? 700 : 400 }}>
                          {r.discriminative_score > 0 ? '+' : ''}{r.discriminative_score.toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {showExperts && expertData?.length === 0 && (
              <div className={s.empty}>No expert data found.</div>
            )}
          </div>
        </div>
      )}

      {/* PCA gallery */}
      {data.pca_images.length > 0 && (
        <div className={s.panel}>
          <div className={s.panelHead}>
            <span className={s.panelTitle}>PCA Visualizations</span>
          </div>
          <div className={s.panelBody}>
            <p style={{ fontSize: '0.8125rem', color: 'var(--text-3)', margin: '0 0 var(--space-md) 0', lineHeight: 1.5 }}>
              Three visualization types per layer, each answering a different question:
            </p>
            <ul style={{ fontSize: '0.8125rem', color: 'var(--text-3)', margin: '0 0 var(--space-md) 0', lineHeight: 1.6, paddingLeft: '1.2em' }}>
              <li><strong>PCA</strong> (unsupervised) — projects onto the top-2 variance directions.
              Shows whether the target is the <em>dominant</em> source of variance. Overlapping clouds
              don't mean probes fail — just that the signal isn't in the top-2 PCs.</li>
              <li><strong>LDA</strong> (supervised) — projects onto directions that <em>maximally
              separate</em> classes. If LDA shows separation but PCA doesn't, the signal exists but
              lives in lower-variance dimensions. This is the plot most directly comparable to probe accuracy.</li>
              <li><strong>Diff-in-Means / Centroid PCA</strong> — for binary targets, projects all
              points onto the vector between class means (1D histogram). For multi-class, does PCA on
              class centroids and projects all points into that space. Stars mark centroids.</li>
            </ul>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 'var(--space-md)' }}>
              {data.pca_images.map(img => (
                <div key={img.name} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <img
                    src={`/api/analysis/pca/${img.name}`}
                    alt={img.name}
                    title={img.name}
                    onClick={() => setLightboxImg(img.name)}
                    style={{
                      width: '100%',
                      borderRadius: 'var(--radius)',
                      border: '1px solid var(--border)',
                      cursor: 'pointer',
                      transition: 'transform 120ms var(--ease-out)',
                    }}
                    onMouseEnter={e => (e.currentTarget.style.transform = 'scale(1.02)')}
                    onMouseLeave={e => (e.currentTarget.style.transform = 'scale(1)')}
                  />
                  <span style={{ fontSize: '0.625rem', color: 'var(--text-4)', textAlign: 'center' }}>
                    {new Date(img.modified_at).toLocaleString()}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* PCA lightbox */}
      {lightboxImg && (
        <div
          onClick={() => setLightboxImg(null)}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 9999,
            background: 'rgba(0,0,0,0.85)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'zoom-out',
            padding: 'var(--space-xl)',
          }}
        >
          <img
            src={`/api/analysis/pca/${lightboxImg}`}
            alt={lightboxImg}
            style={{
              maxWidth: '90vw',
              maxHeight: '90vh',
              borderRadius: 'var(--radius)',
              boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
            }}
            onClick={e => e.stopPropagation()}
          />
          <div style={{
            position: 'absolute',
            top: 'var(--space-lg)',
            right: 'var(--space-lg)',
            color: 'rgba(255,255,255,0.7)',
            fontSize: '0.8125rem',
            cursor: 'pointer',
          }}
            onClick={() => setLightboxImg(null)}
          >
            ESC or click to close
          </div>
        </div>
      )}
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

function FileTree({ entries, depth, probeFile, onLoadProbe }: {
  entries: FileTreeEntry[]
  depth: number
  probeFile: string | null
  onLoadProbe: (file: string) => void
}) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())

  return (
    <div style={{ fontSize: '0.8125rem' }}>
      {entries.map(entry => {
        const isDir = entry.type === 'dir'
        const isCollapsed = collapsed.has(entry.path)
        const isProbe = !isDir && entry.name.startsWith('probe_') && entry.name.endsWith('.parquet')
        const isActive = probeFile === entry.path

        return (
          <div key={entry.path}>
            <div
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '3px 0', paddingLeft: depth * 16,
                cursor: isDir || isProbe ? 'pointer' : 'default',
                background: isActive ? 'oklch(22% 0.02 185 / 0.3)' : 'transparent',
                borderRadius: 'var(--radius)',
              }}
              onClick={() => {
                if (isDir) setCollapsed(prev => {
                  const next = new Set(prev)
                  isCollapsed ? next.delete(entry.path) : next.add(entry.path)
                  return next
                })
                else if (isProbe) onLoadProbe(entry.path)
              }}
              onMouseEnter={e => { if (!isActive) e.currentTarget.style.background = 'oklch(20% 0.010 70)' }}
              onMouseLeave={e => { if (!isActive) e.currentTarget.style.background = 'transparent' }}
            >
              <span style={{ width: 14, textAlign: 'center', fontSize: '0.6875rem', color: 'var(--text-4)', flexShrink: 0 }}>
                {isDir ? (isCollapsed ? '+' : '-') : ' '}
              </span>
              <span style={{
                fontFamily: 'var(--font-mono)', color: isDir ? 'oklch(75% 0.06 185)' : 'var(--text-2)',
                fontWeight: isDir ? 600 : 400, flex: 1,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {entry.name}
              </span>
              {!isDir && entry.size && (
                <span style={{ fontSize: '0.6875rem', color: 'var(--text-4)', flexShrink: 0 }}>
                  {entry.size}
                </span>
              )}
              <span style={{ fontSize: '0.625rem', color: 'var(--text-4)', flexShrink: 0 }}>
                {new Date(entry.modified_at).toLocaleString()}
              </span>
            </div>
            {isDir && !isCollapsed && entry.children && (
              <FileTree entries={entry.children} depth={depth + 1} probeFile={probeFile} onLoadProbe={onLoadProbe} />
            )}
          </div>
        )
      })}
    </div>
  )
}
