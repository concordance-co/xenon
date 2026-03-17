import { useState, useEffect } from 'react'
import { fetchJson } from '../hooks/useApi'
import styles from './CommandRunner.module.css'

interface LocalJob {
  job_id: string
  command: string
  started_at: number
  running: boolean
  return_code: number | null
  line_count: number
}

interface ModalJob {
  app_id: string
  name: string
  state: string
  tasks: number
  created_at: string | null
  stopped_at: string | null
  active: boolean
}

interface Props {
  onReconnect: (jobId: string) => void
  onViewModalLogs: (appId: string, appName: string) => void
}

export function JobList({ onReconnect, onViewModalLogs }: Props) {
  const [localJobs, setLocalJobs] = useState<LocalJob[]>([])
  const [modalJobs, setModalJobs] = useState<ModalJob[]>([])
  const [open, setOpen] = useState(false)

  // Poll local subprocess jobs
  useEffect(() => {
    async function poll() {
      try {
        const data = await fetchJson<LocalJob[]>('/api/jobs')
        setLocalJobs(data)
      } catch { /* ignore */ }
    }
    poll()
    const timer = setInterval(poll, 3000)
    return () => clearInterval(timer)
  }, [])

  // Poll Modal jobs from local dashboard server
  useEffect(() => {
    let timer: ReturnType<typeof setInterval>
    async function poll() {
      try {
        const data = await fetchJson<{ jobs: ModalJob[] }>('/api/modal-jobs')
        setModalJobs(data.jobs)
      } catch (err) {
        console.warn('[JobList] Modal jobs poll failed:', (err as Error).message)
      }
    }
    poll()
    timer = setInterval(poll, 5000)
    return () => clearInterval(timer)
  }, [])

  const runningLocal = localJobs.filter(j => j.running)
  const activeModal = modalJobs.filter(j => j.active)
  const totalRunning = runningLocal.length + activeModal.length
  const totalJobs = localJobs.length + modalJobs.length

  if (totalJobs === 0) return null

  if (!open) {
    return (
      <button
        className={styles.pill}
        style={{ bottom: 'calc(var(--space-lg) + 40px)', right: 'var(--space-lg)' }}
        onClick={() => setOpen(true)}
      >
        {totalRunning > 0 && <span className={styles.spinner} />}
        <span>{totalRunning} running</span>
        <span className={styles.pillLine}>{totalJobs} total</span>
      </button>
    )
  }

  return (
    <div style={{
      position: 'fixed',
      bottom: 'calc(var(--space-lg) + 40px)',
      right: 'var(--space-lg)',
      zIndex: 99,
      width: 400,
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
          Jobs ({totalRunning} running)
        </span>
        <button className={styles.closeBtn} onClick={() => setOpen(false)}>&times;</button>
      </div>
      <div style={{ padding: 'var(--space-xs)' }}>
        {/* Modal jobs */}
        {modalJobs.length > 0 && (
          <>
            <div style={{
              fontSize: '0.625rem', fontWeight: 600, color: 'oklch(55% 0.008 70)',
              padding: 'var(--space-xs) var(--space-sm)',
              textTransform: 'uppercase', letterSpacing: '0.05em',
            }}>
              Modal
            </div>
            {modalJobs.map(j => (
              <div
                key={j.app_id}
                onClick={() => onViewModalLogs(j.app_id, j.name)}
                style={{
                  padding: 'var(--space-xs) var(--space-sm)',
                  borderRadius: 'var(--radius)',
                  cursor: 'pointer',
                  marginBottom: 2,
                  background: j.active ? 'oklch(18% 0.015 185 / 0.3)' : 'transparent',
                  transition: 'background 100ms',
                }}
                onMouseEnter={e => e.currentTarget.style.background = j.active ? 'oklch(22% 0.02 185 / 0.4)' : 'oklch(20% 0.010 70)'}
                onMouseLeave={e => e.currentTarget.style.background = j.active ? 'oklch(18% 0.015 185 / 0.3)' : 'transparent'}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
                  {j.active ? (
                    <span className={styles.spinner} />
                  ) : (
                    <span style={{
                      display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
                      background: j.state === 'stopped' ? 'oklch(52% 0.008 70)' : 'oklch(65% 0.18 15)',
                    }} />
                  )}
                  <span style={{
                    fontSize: '0.6875rem', fontFamily: 'var(--font-mono)',
                    color: 'oklch(72% 0.008 70)',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    flex: 1,
                  }}>
                    {j.name}
                  </span>
                  <span style={{
                    fontSize: '0.5625rem',
                    padding: '1px 5px',
                    borderRadius: 3,
                    background: j.active ? 'oklch(30% 0.04 185)' : 'oklch(22% 0.010 70)',
                    color: j.active ? 'oklch(80% 0.06 185)' : 'oklch(55% 0.008 70)',
                  }}>
                    {j.state}
                  </span>
                  {j.active && j.tasks > 0 && (
                    <span style={{ fontSize: '0.625rem', color: 'oklch(52% 0.008 70)' }}>
                      {j.tasks} task{j.tasks !== 1 ? 's' : ''}
                    </span>
                  )}
                </div>
                <div style={{ fontSize: '0.625rem', color: 'oklch(45% 0.008 70)', marginTop: 2, paddingLeft: 20 }}>
                  {j.created_at && new Date(j.created_at).toLocaleString()}
                  {j.stopped_at && ` — stopped ${new Date(j.stopped_at).toLocaleString()}`}
                </div>
              </div>
            ))}
          </>
        )}
        {/* Local subprocess jobs */}
        {localJobs.length > 0 && (
          <>
            <div style={{
              fontSize: '0.625rem', fontWeight: 600, color: 'oklch(55% 0.008 70)',
              padding: 'var(--space-xs) var(--space-sm)',
              textTransform: 'uppercase', letterSpacing: '0.05em',
            }}>
              Local
            </div>
            {localJobs.map(j => (
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
          </>
        )}
      </div>
    </div>
  )
}
