import { createContext, useContext } from 'react'

export type RunMode = 'local' | 'modal'

export interface IngestConfig {
  mode: RunMode
  topN: number
  selection: 'top' | 'random'
  concurrency: number
  excludeReasoning: boolean
  dbPath: string
}

export interface PrepConfig {
  mode: RunMode
  dbPath: string
  exportParquet: boolean
  exportJsonl: boolean
  tradeSampleSize: number
  observationSampleSize: number
  pairedSampleSize: number
  includeAllDecisions: boolean
}

export interface CaptureConfig {
  mode: RunMode
  // Local
  localDevice: 'mps' | 'cpu' | 'cuda'
  localModelId: string
  // Modal
  modalModelId: string
  // Shared
  captureRouter: boolean
  captureResidual: boolean
  limit: number
  layers: string
  skipExisting: boolean
  batchSize: number
  poolOnCapture: '' | 'last_token' | 'mean_pool'
}

export interface AnalysisConfig {
  mode: RunMode
  target: 'decision_type' | 'trade_side' | 'was_profitable_1h' | 'risk_tolerance' | 'asset'
  analysisMode: 'probe' | 'experts' | 'pca' | 'all'
  runProbe: boolean
  runExperts: boolean
  runPca: boolean
  dataSource: 'router' | 'residual'
  pooling: 'last_token' | 'mean_pool'
  layers: string
  nFolds: number
  limit: number
}

export interface PipelineConfig {
  ingest: IngestConfig
  prep: PrepConfig
  capture: CaptureConfig
  analysis: AnalysisConfig
}

export const DEFAULT_CONFIG: PipelineConfig = {
  ingest: {
    mode: 'local',
    topN: 3,
    selection: 'top',
    concurrency: 20,
    excludeReasoning: false,
    dbPath: 'data/terminal_ingest.db',
  },
  prep: {
    mode: 'local',
    dbPath: 'data/terminal_ingest.db',
    exportParquet: true,
    exportJsonl: false,
    tradeSampleSize: 150,
    observationSampleSize: 150,
    pairedSampleSize: 100,
    includeAllDecisions: false,
  },
  capture: {
    mode: 'modal',
    localDevice: 'mps',
    localModelId: 'Qwen/Qwen3-8B',
    modalModelId: 'Qwen/Qwen3-30B-A3B',
    captureRouter: true,
    captureResidual: false,
    limit: 0,
    layers: '',
    skipExisting: false,
    batchSize: 10,
    poolOnCapture: '',
  },
  analysis: {
    mode: 'local',
    target: 'decision_type',
    analysisMode: 'all',
    runProbe: true,
    runExperts: true,
    runPca: false,
    dataSource: 'router',
    pooling: 'last_token',
    layers: '',
    nFolds: 5,
    limit: 0,
  },
}

const STORAGE_KEY = 'xenon-pipeline-config'

export function loadConfig(): PipelineConfig {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULT_CONFIG
    const saved = JSON.parse(raw)
    // Deep merge with defaults so new fields are picked up
    return {
      ingest: { ...DEFAULT_CONFIG.ingest, ...saved.ingest },
      prep: { ...DEFAULT_CONFIG.prep, ...saved.prep },
      capture: { ...DEFAULT_CONFIG.capture, ...saved.capture },
      analysis: { ...DEFAULT_CONFIG.analysis, ...saved.analysis },
    }
  } catch {
    return DEFAULT_CONFIG
  }
}

export function saveConfig(config: PipelineConfig) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config))
}

// ─── Command builders ───

export function buildIngestCmd(c: IngestConfig, mode: RunMode): string {
  if (mode === 'modal') {
    let cmd = `./scripts/modal_capture.sh modal-ingest --top-n ${c.topN}`
    if (c.selection !== 'top') cmd += ` --selection ${c.selection}`
    if (c.concurrency !== 20) cmd += ` --concurrency ${c.concurrency}`
    if (c.excludeReasoning) cmd += ' --exclude-reasoning'
    return cmd
  }
  let cmd = `uv run -m pipelines.ingest --top-n ${c.topN}`
  if (c.selection !== 'top') cmd += ` --selection ${c.selection}`
  if (c.concurrency !== 20) cmd += ` --request-concurrency ${c.concurrency}`
  if (c.dbPath !== 'data/terminal_ingest.db') cmd += ` --db-path ${c.dbPath}`
  if (c.excludeReasoning) cmd += ' --exclude-reasoning'
  return cmd
}

export function buildPrepCmd(c: PrepConfig, mode: RunMode): string {
  if (mode === 'modal') {
    let cmd = './scripts/modal_capture.sh modal-prep'
    if (c.exportParquet) cmd += ' --export-parquet'
    if (c.exportJsonl) cmd += ' --export-jsonl'
    if (c.tradeSampleSize !== 150) cmd += ` --trade-sample-size ${c.tradeSampleSize}`
    if (c.observationSampleSize !== 150) cmd += ` --observation-sample-size ${c.observationSampleSize}`
    if (c.pairedSampleSize !== 100) cmd += ` --paired-sample-size ${c.pairedSampleSize}`
    if (c.includeAllDecisions) cmd += ' --include-all-decisions'
    return cmd
  }
  let cmd = `uv run -m pipelines.interp.prepare --db-path ${c.dbPath}`
  if (c.exportParquet) cmd += ' --export-parquet'
  if (c.exportJsonl) cmd += ' --export-jsonl'
  if (c.tradeSampleSize !== 150) cmd += ` --trade-sample-size ${c.tradeSampleSize}`
  if (c.observationSampleSize !== 150) cmd += ` --observation-sample-size ${c.observationSampleSize}`
  if (c.pairedSampleSize !== 100) cmd += ` --paired-sample-size ${c.pairedSampleSize}`
  if (c.includeAllDecisions) cmd += ' --include-all-decisions'
  return cmd
}

export function buildCaptureCmd(c: CaptureConfig): string {
  if (c.mode === 'modal') {
    let sub = c.captureResidual && c.captureRouter ? 'full' : c.captureRouter ? 'router' : 'full'
    if (!c.captureResidual && !c.captureRouter) sub = 'full' // fallback
    let cmd = `./scripts/modal_capture.sh ${sub}`
    if (c.limit > 0) cmd += ` --limit ${c.limit}`
    if (c.layers) cmd += ` --layers ${c.layers}`
    if (c.batchSize !== 10) cmd += ` --batch-size ${c.batchSize}`
    if (c.poolOnCapture) cmd += ` --pool ${c.poolOnCapture}`
    return cmd
  }

  // Local
  let cmd = 'uv run --extra interp -m pipelines.interp.capture'
  if (c.localDevice !== 'mps') cmd += ` --device ${c.localDevice}`
  if (c.limit > 0) cmd += ` --limit ${c.limit}`
  if (c.layers) cmd += ` --layers ${c.layers}`
  if (c.skipExisting) cmd += ' --skip-existing'
  if (!c.captureRouter) cmd += ' --no-capture-router'
  if (!c.captureResidual) cmd += ' --no-capture-residual'
  if (c.poolOnCapture) cmd += ` --pool-on-capture ${c.poolOnCapture}`
  return cmd
}

function _analysisMode(c: AnalysisConfig): string {
  const all = c.runProbe && c.runExperts && c.runPca
  if (all) return 'all'
  // If exactly one, use it directly
  const parts: string[] = []
  if (c.runProbe) parts.push('probe')
  if (c.runExperts) parts.push('experts')
  if (c.runPca) parts.push('pca')
  if (parts.length === 0) return 'probe' // fallback
  if (parts.length === 1) return parts[0]
  return 'all' // 2 of 3 — run all, the extra one is cheap
}

export function buildAnalysisCmd(c: AnalysisConfig): string {
  const mode = _analysisMode(c)

  if (c.mode === 'modal') {
    let cmd = `./scripts/modal_capture.sh analyze --mode ${mode} --target ${c.target}`
    if (c.dataSource !== 'router') cmd += ` --data-source ${c.dataSource}`
    if (c.pooling !== 'last_token') cmd += ` --pooling ${c.pooling}`
    if (c.layers) cmd += ` --layers ${c.layers}`
    if (c.nFolds !== 5) cmd += ` --n-folds ${c.nFolds}`
    if (c.limit > 0) cmd += ` --limit ${c.limit}`
    return cmd
  }

  // Local
  let cmd = `uv run --extra analysis -m pipelines.interp.analysis --mode ${mode} --target ${c.target}`
  cmd += ` --data-source ${c.dataSource}`
  cmd += ` --pooling ${c.pooling}`
  if (c.layers) cmd += ` --layers ${c.layers}`
  if (c.nFolds !== 5) cmd += ` --n-folds ${c.nFolds}`
  if (c.limit > 0) cmd += ` --limit ${c.limit}`
  return cmd
}

// Context
export interface ConfigContextValue {
  config: PipelineConfig
  update: <K extends keyof PipelineConfig>(phase: K, partial: Partial<PipelineConfig[K]>) => void
}

export const ConfigContext = createContext<ConfigContextValue>({
  config: DEFAULT_CONFIG,
  update: () => {},
})

export function useConfig() {
  return useContext(ConfigContext)
}
