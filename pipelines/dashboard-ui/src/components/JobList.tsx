import { useState, useEffect } from 'react'
import { fetchJson } from '../hooks/useApi'
import s from './shared.module.css'
import styles from './CommandRunner.module.css'

interface Job {
  job_id: string
  command: string
  started_at: number
  running: boolean
  return_code: number | null
  line_count: number
}

interface Props {
  onReconnect: (jobId: string) => void
}

export function JobList({ onReconnect }: Props) {
  const [jobs, setJobs] = useState<Job[]>([])
  const [open, setOpen] = useState(false)

  useEffect(() => {
    let timer: ReturnType<typeof setInterval>
    async function poll() {
      try {
        const data = await fetchJson<Job[]>('/api/jobs')
        setJobs(data)
      } catch { /* ignore */ }
    }
    poll()
    timer = setInterval(poll, 3000)
    return () => clearInterval(timer)
  }, [])

  const running = jobs.filter(j => j.running)
  const finished = jobs.filter(j => !j.running)

  if (jobs.length === 0) return null

  if (!open) {
    return (
      <button
        className={styles.pill}
        style={{ bottom: 'calc(var(--space-lg) + 40px)', right: 'var(--space-lg)' }}
        onClick={() => setOpen(true)}
      >
        {running.length > 0 && <span className={styles.spinner} />}
        <span>{running.length} running</span>
        <span className={styles.pillLine}>{jobs.length} total</span>
      </button>
    )
  }

  return (
    <div style={{
      position: 'fixed',
      bottom: 'calc(var(--space-lg) + 40px)',
      right: 'var(--space-lg)',
      zIndex: 99,
      width: 380,
      maxHeight: '50vh',
      overflow: 'auto',
      background: 'oklch(14% 0.012 70)',
      border: '1px solid oklch(26% 0.010 70)',
      borderRadius: 'var(--radius-lg)',
      boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
    }}>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: 'var(--space-sm) var(--space-md)',
        borderBottom: '1px solid oklch(22% 0.010 70)',
      }}>
        <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'oklch(88% 0.006 70)' }}>
          Jobs ({running.length} running)
        </span>
        <button className={styles.closeBtn} onClick={() => setOpen(false)}>&times;</button>
      </div>
      <div style={{ padding: 'var(--space-xs)' }}>
        {jobs.map(j => (
          <div
            key={j.job_id}
            onClick={() => onReconnect(j.job_id)}
            style={{
              padding: 'var(--space-xs) var(--space-sm)',
              borderRadius: 'var(--radius)',
              cursor: 'pointer',
              marginBottom: 2,
              background: j.running ? 'oklch(18% 0.015 185 / 0.3)' : 'transparent',
              transition: 'background 100ms',
            }}
            onMouseEnter={e => { if (!j.running) e.currentTarget.style.background = 'oklch(20% 0.010 70)' }}
            onMouseLeave={e => { if (!j.running) e.currentTarget.style.background = 'transparent' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
              {j.running ? (
                <span className={styles.spinner} />
              ) : (
                <span style={{
                  display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
                  background: j.return_code === 0 ? 'oklch(72% 0.16 145)' : 'oklch(65% 0.18 15)',
                }} />
              )}
              <span style={{
                fontSize: '0.6875rem', fontFamily: 'var(--font-mono)',
                color: 'oklch(72% 0.008 70)',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                flex: 1,
              }}>
                {j.command.length > 50 ? j.command.slice(0, 50) + '...' : j.command}
              </span>
              <span style={{ fontSize: '0.625rem', color: 'oklch(52% 0.008 70)' }}>
                {j.line_count}L
              </span>
            </div>
            <div style={{ fontSize: '0.625rem', color: 'oklch(45% 0.008 70)', marginTop: 2, paddingLeft: 20 }}>
              {new Date(j.started_at * 1000).toLocaleTimeString()}
              {j.return_code !== null && ` · exit ${j.return_code}`}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
