import { useState, useEffect, useRef, useCallback } from 'react'
import styles from './CommandRunner.module.css'

interface Props {
  command?: string
  jobId?: string  // reconnect to existing local job
  modalAppId?: string  // stream logs from a Modal app
  modalAppName?: string
  minimized: boolean
  onMinimize: () => void
  onRestore: () => void
  onClose: () => void
  onJobId?: (id: string) => void
}

interface LogLine {
  type: 'stdout' | 'stderr'
  text: string
}

export function CommandRunner({ command, jobId, modalAppId, modalAppName, minimized, onMinimize, onRestore, onClose, onJobId }: Props) {
  const [lines, setLines] = useState<LogLine[]>([])
  const [running, setRunning] = useState(true)
  const [returnCode, setReturnCode] = useState<number | null>(null)
  const [displayCmd, setDisplayCmd] = useState(command ?? modalAppName ?? '')
  const logRef = useRef<HTMLPreElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const startedRef = useRef(false)

  const startStream = useCallback(async () => {
    // Guard against React StrictMode double-invocation
    if (startedRef.current) return
    startedRef.current = true

    setRunning(true)
    setLines([])
    setReturnCode(null)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      let resp: Response

      if (modalAppId) {
        // Mode 3: stream logs from Modal app via local dashboard
        resp = await fetch('/api/modal-logs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ app_id: modalAppId }),
          signal: controller.signal,
        })
      } else {
        // Mode 1 & 2: local command or reconnect
        const url = jobId ? '/api/job-reconnect' : '/api/run-stream'
        const body = jobId ? { job_id: jobId } : { command }
        resp = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: controller.signal,
        })
      }

      if (!resp.body) {
        setLines([{ type: 'stderr', text: 'No response body' }])
        setRunning(false)
        setReturnCode(-1)
        return
      }

      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        const parts = buffer.split('\n\n')
        buffer = parts.pop() ?? ''

        for (const part of parts) {
          let event = 'message'
          let data = ''
          for (const line of part.split('\n')) {
            if (line.startsWith('event: ')) event = line.slice(7)
            else if (line.startsWith('data: ')) data = line.slice(6)
          }

          if (event === 'stdout') {
            setLines(prev => [...prev, { type: 'stdout', text: data }])
          } else if (event === 'stderr') {
            setLines(prev => [...prev, { type: 'stderr', text: data }])
          } else if (event === 'state') {
            setLines(prev => [...prev, { type: 'stderr', text: `[${data}]` }])
          } else if (event === 'done') {
            if (modalAppId) {
              setReturnCode(0)
            } else {
              try {
                const parsed = JSON.parse(data)
                setReturnCode(parsed.returncode)
              } catch {
                setReturnCode(-1)
              }
            }
            setRunning(false)
          } else if (event === 'error') {
            setLines(prev => [...prev, { type: 'stderr', text: data }])
            setReturnCode(-1)
            setRunning(false)
          } else if (event === 'job_id') {
            onJobId?.(data)
          } else if (event === 'command') {
            setDisplayCmd(data)
          }
        }
      }

      // If stream ended without a done event
      if (running) {
        setRunning(false)
        if (returnCode === null) setReturnCode(modalAppId ? 0 : -1)
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setLines(prev => [...prev, { type: 'stderr', text: (err as Error).message }])
        setReturnCode(-1)
      }
      setRunning(false)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    startStream()
    return () => {
      abortRef.current?.abort()
    }
  }, [startStream])

  // Auto-scroll
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [lines, returnCode])

  const handleClose = () => {
    abortRef.current?.abort()
    onClose()
  }

  const isModal = !!modalAppId
  const statusText = running
    ? (isModal ? 'Streaming...' : 'Running...')
    : returnCode === 0 ? 'Complete' : (isModal ? 'Disconnected' : 'Failed')

  if (minimized) {
    return (
      <button className={styles.pill} onClick={onRestore}>
        {running && <span className={styles.spinner} />}
        <span>{statusText}</span>
        <span className={styles.pillLine}>{lines.length} lines</span>
      </button>
    )
  }

  return (
    <div className={styles.overlay} onClick={e => { if (e.target === e.currentTarget) onMinimize() }}>
      <div className={styles.panel}>
        <div className={styles.head}>
          <div className={styles.headLeft}>
            {running && <span className={styles.spinner} />}
            <span className={styles.headTitle}>{statusText}</span>
          </div>
          <div className={styles.headRight}>
            <button className={styles.closeBtn} onClick={onMinimize} title="Minimize">
              &minus;
            </button>
            <button className={styles.closeBtn} onClick={handleClose}>
              {running ? (isModal ? 'Stop' : 'Cancel') : 'Close'}
            </button>
          </div>
        </div>
        <div className={styles.cmd}>{isModal ? displayCmd : `$ ${displayCmd}`}</div>
        <pre className={styles.log} ref={logRef}>
          {lines.length === 0 && running && (isModal ? 'Connecting to Modal logs...\n' : 'Waiting for output...\n')}
          {lines.map((line, i) => (
            <span key={i} className={line.type === 'stderr' ? styles.stderr : undefined}>
              {line.text}{'\n'}
            </span>
          ))}
          {returnCode !== null && (
            <span className={returnCode === 0 ? styles.ok : styles.err}>
              {'\n'}{isModal ? 'App finished.' : `Exit code: ${returnCode}`}
            </span>
          )}
        </pre>
      </div>
    </div>
  )
}
