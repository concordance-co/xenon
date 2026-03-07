import type { PipelineStatus } from '../types/api'
import type { Phase } from '../App'
import styles from './PhaseStrip.module.css'

const META: Record<Phase, { label: string; unit: string; countKey: string }> = {
  ingest: { label: 'Ingest', unit: 'logs', countKey: 'log_count' },
  prep: { label: 'Data Prep', unit: 'examples', countKey: 'total_examples' },
  capture: { label: 'Capture', unit: 'files', countKey: 'total_files' },
  analysis: { label: 'Analysis', unit: 'results', countKey: 'total_results' },
}

interface Props {
  status: PipelineStatus
  active: Phase
  onSelect: (p: Phase) => void
}

export function PhaseStrip({ status, active, onSelect }: Props) {
  return (
    <div className={styles.strip}>
      {(Object.keys(META) as Phase[]).map(key => {
        const m = META[key]
        const phase = status[key] as unknown as Record<string, unknown>
        const count = phase[m.countKey] as number | undefined
        const s = phase.status as string

        return (
          <button
            key={key}
            className={`${styles.item} ${active === key ? styles.active : ''}`}
            onClick={() => onSelect(key)}
          >
            <span className={`${styles.dot} ${styles[`dot_${s}`]}`} />
            <span className={styles.label}>{m.label}</span>
            <span className={styles.count}>
              {count?.toLocaleString() ?? '—'}
            </span>
            <span className={styles.unit}>{m.unit}</span>
          </button>
        )
      })}
    </div>
  )
}
