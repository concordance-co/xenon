import { useState, useCallback } from 'react'
import { useFetch } from './hooks/useApi'
import type { PipelineStatus } from './types/api'
import { ConfigContext, loadConfig, saveConfig, type PipelineConfig } from './hooks/useConfig'
import { PhaseStrip } from './components/PhaseStrip'
import { IngestView } from './components/IngestView'
import { PrepView } from './components/PrepView'
import { CaptureView } from './components/CaptureView'
import { AnalysisView } from './components/AnalysisView'
import { CounterfactualView } from './components/CounterfactualView'
import { ExplorerView } from './components/ExplorerView'
import { CommandRunner } from './components/CommandRunner'
import { JobList } from './components/JobList'
import { PipelineFlow } from './components/PipelineFlow'
import styles from './App.module.css'

export type Phase = 'ingest' | 'prep' | 'capture' | 'analysis'
type TopView = 'pipeline' | 'explorer' | 'counterfactual'

interface RunnerState {
  command?: string
  jobId?: string
  modalAppId?: string
  modalAppName?: string
}

const noop = () => {}

export default function App() {
  const [view, setView] = useState<TopView>('pipeline')
  const [activePhase, setActivePhase] = useState<Phase>('ingest')
  const [runner, setRunner] = useState<RunnerState | null>(null)
  const [cmdMinimized, setCmdMinimized] = useState(false)
  const [refreshKey, setRefreshKey] = useState(0)
  const { data: status, refetch } = useFetch<PipelineStatus>('/api/status')
  const [config, setConfig] = useState<PipelineConfig>(loadConfig)

  const updateConfig = useCallback(<K extends keyof PipelineConfig>(
    phase: K,
    partial: Partial<PipelineConfig[K]>,
  ) => {
    setConfig(prev => {
      const next = { ...prev, [phase]: { ...prev[phase], ...partial } }
      saveConfig(next)
      return next
    })
  }, [])

  const handleRun = useCallback((cmd: string) => {
    setRunner({ command: cmd })
    setCmdMinimized(false)
  }, [])

  const handleReconnect = useCallback((jobId: string) => {
    setRunner({ jobId })
    setCmdMinimized(false)
  }, [])

  const handleViewModalLogs = useCallback((appId: string, appName: string) => {
    setRunner({ modalAppId: appId, modalAppName: appName })
    setCmdMinimized(false)
  }, [])

  const handleRunDone = useCallback(() => {
    setRunner(null)
    setCmdMinimized(false)
    setRefreshKey(k => k + 1)
    refetch('/api/status?refresh=1')
  }, [refetch])

  return (
    <ConfigContext.Provider value={{ config, update: updateConfig }}>
      <div className={styles.layout}>
        <header className={styles.header}>
          <div className={styles.headerLeft}>
            <h1 className={styles.title}>
              <span className={styles.titleAccent}>Xe</span>non
            </h1>
            <span className={styles.subtitle}>
              {view === 'pipeline' ? 'Pipeline Control' : view === 'explorer' ? 'Data Explorer' : 'Counterfactual Experiment'}
            </span>
          </div>
          <div className={styles.headerNav}>
            <button
              className={view === 'pipeline' ? styles.navActive : styles.navBtn}
              onClick={() => setView('pipeline')}
            >
              Pipeline
            </button>
            <button
              className={view === 'explorer' ? styles.navActive : styles.navBtn}
              onClick={() => setView('explorer')}
            >
              Explorer
            </button>
            <button
              className={view === 'counterfactual' ? styles.navActive : styles.navBtn}
              onClick={() => setView('counterfactual')}
            >
              Counterfactual
            </button>
          </div>
        </header>

        {view === 'pipeline' && (
          <>
            <PipelineFlow status={status} active={activePhase} onSelect={setActivePhase} />

            {status && (
              <PhaseStrip
                status={status}
                active={activePhase}
                onSelect={setActivePhase}
              />
            )}

            <main className={styles.main}>
              {activePhase === 'ingest' && <IngestView onRun={handleRun} />}
              {activePhase === 'prep' && <PrepView onRun={handleRun} />}
              {activePhase === 'capture' && <CaptureView onRun={handleRun} />}
              {activePhase === 'analysis' && <AnalysisView onRun={handleRun} refreshKey={refreshKey} />}
            </main>

            <JobList onReconnect={handleReconnect} onViewModalLogs={handleViewModalLogs} />
          </>
        )}

        {view === 'explorer' && (
          <main className={styles.main}>
            <ExplorerView />
          </main>
        )}

        {view === 'counterfactual' && (
          <>
            <main className={styles.main}>
              <CounterfactualView onRun={handleRun} refreshKey={refreshKey} />
            </main>
            <JobList onReconnect={handleReconnect} onViewModalLogs={handleViewModalLogs} />
          </>
        )}

        {runner && (
          <CommandRunner
            key={runner.modalAppId ?? runner.jobId ?? runner.command ?? ''}
            command={runner.command}
            jobId={runner.jobId}
            modalAppId={runner.modalAppId}
            modalAppName={runner.modalAppName}
            minimized={cmdMinimized}
            onMinimize={() => setCmdMinimized(true)}
            onRestore={() => setCmdMinimized(false)}
            onClose={handleRunDone}
            onJobId={noop}
          />
        )}
      </div>
    </ConfigContext.Provider>
  )
}
