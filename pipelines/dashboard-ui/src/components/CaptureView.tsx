import { useFetch } from '../hooks/useApi'
import { useConfig, buildCaptureCmd } from '../hooks/useConfig'
import type { CaptureData } from '../types/api'
import { Tip } from './Tip'
import s from './shared.module.css'

interface Props {
  onRun: (cmd: string) => void
}

export function CaptureView({ onRun }: Props) {
  const { data, loading } = useFetch<CaptureData>('/api/capture')
  const { config, update } = useConfig()
  const c = config.capture

  const cmd = buildCaptureCmd(c)

  if (loading || !data) return <div className={s.empty}>Loading...</div>

  return (
    <div>
      <p className={s.phaseDesc}>
        Feeds each labeled example through a Qwen3 MoE model on Modal
        (A100-80GB) and records internal activations. For each input,
        captures the router logits (which experts the model selects at each
        layer) and optionally residual stream hidden states. Qwen3-30B-A3B
        has 48 layers with 128 experts and top-8 routing — the router
        logits encode which computational pathways the model uses for
        trading decisions vs. observations. Outputs safetensors files to the
        Modal volume.
      </p>

      <div className={s.statsRow}>
        <Stat label="Residual Files" value={data.residual_count} />
        <Stat label="Router Files" value={data.router_count} />
        <Stat label="Total Size" value={`${data.total_size_mb} MB`} />
        <Stat label="Avg Seq Len" value={data.avg_seq_len} />
        <Stat label="Layers" value={data.num_layers || '—'} />
        <Stat label="Hidden Dim" value={data.hidden_dim || '—'} />
        <Stat label="Experts" value={data.num_experts || 'dense'} />
      </div>

      <div className={s.panel}>
        <div className={s.panelHead}>
          <span className={s.panelTitle}>Capture Configuration</span>
        </div>
        <div className={s.panelBody}>
          <div className={s.configForm}>
            <div className={s.field}>
              <label className={s.fieldLabel}>
                Capture Router
                <Tip text="Record MoE router logits — the raw scores the model uses to select which experts process each token. This is the primary signal for interpretability probes." />
              </label>
              <label className={s.toggle}>
                <span
                  className={c.captureRouter ? s.toggleTrackOn : s.toggleTrack}
                  onClick={() => update('capture', { captureRouter: !c.captureRouter })}
                />
                <span>{c.captureRouter ? 'Yes' : 'No'}</span>
              </label>
            </div>
            <div className={s.field}>
              <label className={s.fieldLabel}>
                Capture Residual
                <Tip text="Record residual stream hidden states — the full activation vector at each layer. Much larger files (~hidden_dim per layer vs. ~128 for router) but captures richer information." />
              </label>
              <label className={s.toggle}>
                <span
                  className={c.captureResidual ? s.toggleTrackOn : s.toggleTrack}
                  onClick={() => update('capture', { captureResidual: !c.captureResidual })}
                />
                <span>{c.captureResidual ? 'Yes' : 'No'}</span>
              </label>
            </div>
            <div className={s.field}>
              <label className={s.fieldLabel}>
                Limit
                <Tip text="Max examples to process. 0 = all. Use a small limit (5-10) for testing to verify capture works before running the full dataset." />
              </label>
              <input
                type="number"
                className={s.fieldInput}
                value={c.limit}
                min={0}
                placeholder="0 = all"
                onChange={e => update('capture', { limit: Number(e.target.value) || 0 })}
              />
            </div>
            <div className={s.field}>
              <label className={s.fieldLabel}>
                Layers
                <Tip text="Comma-separated layer indices to capture. Empty = all layers. Use a subset like 0,16,32,47 for faster iteration — you can always re-capture all layers later." />
              </label>
              <input
                type="text"
                className={s.fieldInput}
                value={c.layers}
                placeholder="e.g. 0,16,32,47"
                onChange={e => update('capture', { layers: e.target.value })}
              />
            </div>
            <div className={s.field}>
              <label className={s.fieldLabel}>
                Batch Size
                <Tip text="Examples per Modal batch. Higher = better GPU utilization but more memory. 10 is safe for A100-80GB with the 30B model." />
              </label>
              <input
                type="number"
                className={s.fieldInput}
                value={c.batchSize}
                min={1}
                onChange={e => update('capture', { batchSize: Number(e.target.value) || 1 })}
              />
            </div>
            <div className={s.field}>
              <label className={s.fieldLabel}>
                Pool on Capture
                <Tip text="Collapse the sequence dimension during capture to drastically reduce file size (~9300x for residual). 'last_token' keeps only the final token's activations (where the model commits to a decision). 'mean_pool' averages across all tokens. Leave empty to save full sequences for flexible post-hoc pooling." />
              </label>
              <select
                className={s.fieldSelect}
                value={c.poolOnCapture}
                onChange={e => update('capture', { poolOnCapture: e.target.value as '' | 'last_token' | 'mean_pool' })}
              >
                <option value="">None (full sequence)</option>
                <option value="last_token">Last Token</option>
                <option value="mean_pool">Mean Pool</option>
              </select>
            </div>
          </div>

          <div className={s.generatedCmd}>{cmd}</div>

          <div className={s.btnRow}>
            <button
              className={`${s.btn} ${s.btnAccent}`}
              onClick={() => onRun(cmd)}
            >
              Run Capture
            </button>
            <button className={s.btn} onClick={() => onRun('./scripts/modal_capture.sh inspect')}>
              Inspect Volume
            </button>
            <button className={s.btn} onClick={() => onRun('./scripts/modal_capture.sh meta')}>
              Show Metadata
            </button>
          </div>
        </div>
      </div>

      {data.recent_captures.length > 0 && (
        <div className={s.panel}>
          <div className={s.panelHead}>
            <span className={s.panelTitle}>
              Captures
              <span style={{ fontWeight: 400, color: 'var(--text-3)', marginLeft: 8, fontSize: '0.75rem' }}>
                {data.recent_captures.length} rows from metadata.parquet
              </span>
            </span>
          </div>
          <div className={s.panelBody} style={{ overflowX: 'auto' }}>
            <table className={s.table}>
              <thead>
                <tr>
                  <th>Log ID</th>
                  <th style={{ textAlign: 'right' }}>Seq Len</th>
                  <th style={{ textAlign: 'right' }}>Size (MB)</th>
                  <th style={{ textAlign: 'right' }}>Time (s)</th>
                  <th>Router</th>
                  <th style={{ textAlign: 'right' }}>Layers</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {data.recent_captures.map(r => (
                  <tr key={r.log_id}>
                    <td className="mono">{r.log_id}</td>
                    <td className="mono" style={{ textAlign: 'right' }}>{r.seq_len.toLocaleString()}</td>
                    <td className="mono" style={{ textAlign: 'right' }}>{(r.file_size_bytes / 1024 / 1024).toFixed(1)}</td>
                    <td className="mono" style={{ textAlign: 'right' }}>{r.elapsed_s}</td>
                    <td>
                      {r.has_router
                        ? <span className={s.badgeGreen + ' ' + s.badge}>yes</span>
                        : <span className={s.badgeDim + ' ' + s.badge}>no</span>}
                    </td>
                    <td className="mono" style={{ textAlign: 'right' }}>{r.num_layers_captured}</td>
                    <td className="mono" style={{ fontSize: '0.6875rem', color: 'var(--text-3)' }}>
                      {r.capture_timestamp?.slice(0, 19) ?? '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
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
