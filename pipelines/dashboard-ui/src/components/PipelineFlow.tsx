import type { PipelineStatus } from '../types/api'
import type { Phase } from '../App'
import styles from './PipelineFlow.module.css'

const PHASES: { key: Phase; label: string; desc: string }[] = [
  { key: 'ingest', label: 'Ingest', desc: 'Terminal API → Neon' },
  { key: 'prep', label: 'Data Prep', desc: 'JSONB → interp_examples' },
  { key: 'capture', label: 'Capture', desc: 'LLM activations → Safetensors' },
  { key: 'analysis', label: 'Analysis', desc: 'Probe + PCA on routing' },
]

function statusColor(s: string) {
  if (s === 'ready') return 'var(--green)'
  if (s === 'partial') return 'var(--amber)'
  return 'var(--text-4)'
}

interface Props {
  status: PipelineStatus | null
  active: Phase
  onSelect: (p: Phase) => void
}

export function PipelineFlow({ status, active, onSelect }: Props) {
  return (
    <div className={styles.flow}>
      {PHASES.map((phase, i) => {
        const s = status?.[phase.key]?.status ?? 'empty'
        return (
          <div key={phase.key} className={styles.step}>
            <button
              className={`${styles.node} ${active === phase.key ? styles.nodeActive : ''}`}
              onClick={() => onSelect(phase.key)}
              style={{ '--status-color': statusColor(s) } as React.CSSProperties}
            >
              <span className={styles.nodeNum}>{i + 1}</span>
              <span className={styles.nodeLabel}>{phase.label}</span>
              <span className={styles.nodeDesc}>{phase.desc}</span>
              <span className={styles.nodeDot} />
            </button>
            {i < PHASES.length - 1 && (
              <svg className={styles.arrow} viewBox="0 0 40 12" fill="none">
                <path d="M0 6h32m0 0l-5-4.5M32 6l-5 4.5" stroke="var(--border-strong)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            )}
          </div>
        )
      })}
    </div>
  )
}
