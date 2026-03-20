import { createContext, useContext } from 'react'

export interface IngestConfig {
  topN: number
  selection: 'top' | 'random' | 'existing' | 'backfill'
  concurrency: number
  rps: number
  excludeReasoning: boolean
  skipDeferred: boolean
}

export interface PrepConfig {
  limit: number
  includeAllDecisions: boolean
}

export interface OutcomesConfig {
  concurrency: number
  limit: number
}

export interface CaptureConfig {
  captureRouter: boolean
  captureResidual: boolean
  limit: number
  layers: string
  batchSize: number
  poolOnCapture: '' | 'last_token' | 'mean_pool'
}

export interface AnalysisConfig {
  target: 'decision_type' | 'trade_side' | 'was_profitable_1h' | 'executed_valence' | 'forced_observe' | 'risk_tolerance' | 'asset'
  runProbe: boolean
  runExperts: boolean
  runPca: boolean
  dataSource: 'router' | 'residual'
  pooling: 'last_token' | 'mean_pool'
  layers: string
  nFolds: number
  limit: number
}

export interface CounterfactualConfig {
  experimentId: string
  dataset: 'a' | 'b' | 'both'
  batchSize: number
  gpu: 'A100-80GB' | 'H200' | 'H100'
  layers: string
  questions: 'a' | 'b' | 'c' | 'all'
  nBootstrap: number
  seed: number
}

export interface PipelineConfig {
  ingest: IngestConfig
  prep: PrepConfig
  outcomes: OutcomesConfig
  capture: CaptureConfig
  analysis: AnalysisConfig
  counterfactual: CounterfactualConfig
}

export const DEFAULT_CONFIG: PipelineConfig = {
  ingest: {
    topN: 3,
    selection: 'top',
    concurrency: 20,
    rps: 6,
    excludeReasoning: false,
    skipDeferred: false,
  },
  prep: {
    limit: 0,
    includeAllDecisions: false,
  },
  outcomes: {
    concurrency: 5,
    limit: 0,
  },
  capture: {
    captureRouter: true,
    captureResidual: false,
    limit: 0,
    layers: '',
    batchSize: 10,
    poolOnCapture: '',
  },
  analysis: {
    target: 'decision_type',
    runProbe: true,
    runExperts: true,
    runPca: false,
    dataSource: 'router',
    pooling: 'last_token',
    layers: '',
    nFolds: 5,
    limit: 0,
  },
  counterfactual: {
    experimentId: 'default',
    dataset: 'both',
    batchSize: 10,
    gpu: 'H200',
    layers: '',
    questions: 'all',
    nBootstrap: 1000,
    seed: 42,
  },
}

const STORAGE_KEY = 'xenon-pipeline-config'

export function loadConfig(): PipelineConfig {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULT_CONFIG
    const saved = JSON.parse(raw)
    return {
      ingest: { ...DEFAULT_CONFIG.ingest, ...saved.ingest },
      prep: { ...DEFAULT_CONFIG.prep, ...saved.prep },
      outcomes: { ...DEFAULT_CONFIG.outcomes, ...saved.outcomes },
      capture: { ...DEFAULT_CONFIG.capture, ...saved.capture },
      analysis: { ...DEFAULT_CONFIG.analysis, ...saved.analysis },
      counterfactual: { ...DEFAULT_CONFIG.counterfactual, ...saved.counterfactual },
    }
  } catch {
    return DEFAULT_CONFIG
  }
}

export function saveConfig(config: PipelineConfig) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config))
}

// ─── Command builders (all Modal) ───

export function buildIngestCmd(c: IngestConfig): string {
  let cmd = `./scripts/modal_capture.sh modal-ingest --top-n ${c.topN}`
  if (c.selection !== 'top') cmd += ` --selection ${c.selection}`
  if (c.concurrency !== 20) cmd += ` --concurrency ${c.concurrency}`
  if (c.rps !== 6) cmd += ` --rps ${c.rps}`
  if (c.excludeReasoning) cmd += ' --exclude-reasoning'
  if (c.skipDeferred) cmd += ' --skip-deferred'
  return cmd
}

export function buildPrepCmd(c: PrepConfig): string {
  let cmd = './scripts/modal_capture.sh modal-prep'
  if (c.limit > 0) cmd += ` --limit ${c.limit}`
  if (c.includeAllDecisions) cmd += ' --include-all-decisions'
  return cmd
}

export function buildOutcomesCmd(c: OutcomesConfig): string {
  let cmd = './scripts/modal_capture.sh modal-outcomes'
  if (c.concurrency !== 5) cmd += ` --concurrency ${c.concurrency}`
  if (c.limit > 0) cmd += ` --outcomes-limit ${c.limit}`
  return cmd
}

export function buildCaptureCmd(c: CaptureConfig): string {
  let sub = c.captureResidual && c.captureRouter ? 'full' : c.captureRouter ? 'router' : 'full'
  if (!c.captureResidual && !c.captureRouter) sub = 'full'
  let cmd = `./scripts/modal_capture.sh ${sub}`
  if (c.limit > 0) cmd += ` --limit ${c.limit}`
  if (c.layers) cmd += ` --layers ${c.layers}`
  if (c.batchSize !== 10) cmd += ` --batch-size ${c.batchSize}`
  if (c.poolOnCapture) cmd += ` --pool ${c.poolOnCapture}`
  return cmd
}

function _analysisMode(c: AnalysisConfig): string {
  const all = c.runProbe && c.runExperts && c.runPca
  if (all) return 'all'
  const parts: string[] = []
  if (c.runProbe) parts.push('probe')
  if (c.runExperts) parts.push('experts')
  if (c.runPca) parts.push('pca')
  if (parts.length === 0) return 'probe'
  if (parts.length === 1) return parts[0]
  return 'all'
}

export function buildAnalysisCmd(c: AnalysisConfig): string {
  const mode = _analysisMode(c)
  let cmd = `./scripts/modal_capture.sh analyze --mode ${mode} --target ${c.target}`
  if (c.dataSource !== 'router') cmd += ` --data-source ${c.dataSource}`
  if (c.pooling !== 'last_token') cmd += ` --pooling ${c.pooling}`
  if (c.layers) cmd += ` --layers ${c.layers}`
  if (c.nFolds !== 5) cmd += ` --n-folds ${c.nFolds}`
  if (c.limit > 0) cmd += ` --limit ${c.limit}`
  return cmd
}

export function buildCounterfactualBuildCmd(c: CounterfactualConfig): string {
  return `uv run python -m pipelines.interp.counterfactual --dataset ${c.dataset} --seed ${c.seed}`
}

export function buildCounterfactualCaptureCmd(c: CounterfactualConfig): string {
  let cmd = `./scripts/modal_capture.sh counterfactual-capture --experiment-id ${c.experimentId}`
  cmd += ` --batch-size ${c.batchSize} --dataset ${c.dataset} --gpu ${c.gpu}`
  return cmd
}

export function buildCounterfactualAnalyzeCmd(c: CounterfactualConfig): string {
  let cmd = `./scripts/modal_capture.sh counterfactual-analyze --experiment-id ${c.experimentId}`
  cmd += ` --questions ${c.questions}`
  if (c.layers) cmd += ` --layers ${c.layers}`
  cmd += ` --n-bootstrap ${c.nBootstrap} --seed ${c.seed}`
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
