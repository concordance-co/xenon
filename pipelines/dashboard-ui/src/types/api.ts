export interface PhaseStatus {
  status: 'ready' | 'partial' | 'empty'
}

export interface IngestStatus extends PhaseStatus {
  log_count: number
}

export interface PrepStatus extends PhaseStatus {
  total_examples: number
}

export interface CaptureStatus extends PhaseStatus {
  total_files: number
}

export interface AnalysisStatus extends PhaseStatus {
  total_results: number
}

export interface PipelineStatus {
  ingest: IngestStatus
  prep: PrepStatus
  capture: CaptureStatus
  analysis: AnalysisStatus
}

export interface TableInfo {
  name: string
  count: number
}

export interface IngestData {
  vault_count: number
  strategy_count: number
  log_count: number
  full_log_count: number
  full_log_coverage_pct: number
  parse_error_count: number
  tables: TableInfo[]
}

export interface ExportFile {
  name: string
  size: string
}

export interface LabelDistRow {
  decision_type: string | null
  count: number
  trade_side: string | null
  avg_risk: number | null
}

export interface PrepData {
  total_examples: number
  high_quality: number
  medium_quality: number
  low_quality: number
  trade_count: number
  observation_count: number
  parquet_exported: boolean
  export_files: ExportFile[]
  label_distribution: LabelDistRow[]
}

export interface RiskBreakdownRow {
  risk_level: number
  count: number
  avg_pnl_1h: number | null
  avg_pnl_4h: number | null
  avg_pnl_1d: number | null
  win_rate_1h: number | null
}

export interface OutcomesData {
  total_outcomes: number
  unlabeled_swaps: number
  total_swaps: number
  avg_pnl_1h: number | null
  avg_pnl_4h: number | null
  avg_pnl_1d: number | null
  win_rate_1h: number | null
  risk_breakdown: RiskBreakdownRow[]
}

export interface CaptureRow {
  log_id: number
  seq_len: number
  file_size_bytes: number
  elapsed_s: number
  has_router: boolean
  num_layers_captured: number
  capture_timestamp: string
}

export interface CaptureData {
  residual_count: number
  router_count: number
  total_size_mb: number
  avg_seq_len: number
  num_layers: number
  hidden_dim: number
  num_experts: number
  recent_captures: CaptureRow[]
}

export interface ResultFile {
  name: string
  size: string
}

export interface AnalysisData {
  total_results: number
  probe_files: string[]
  has_expert_specialization: boolean
  pca_images: string[]
  result_files: ResultFile[]
}

export interface ProbeRow {
  layer: number
  accuracy_mean: number
  accuracy_std: number
  balanced_accuracy: number
  baseline_majority: number
  baseline_shuffled: number
  selectivity: number
  n_examples: number
  n_classes: number
}

export interface ExpertRow {
  layer: number
  expert_id: number
  rank: number
  discriminative_score: number
}

export interface RunResult {
  stdout: string
  stderr: string
  returncode: number
}

export interface Job {
  job_id: string
  command: string
  started_at: number
  running: boolean
  return_code: number | null
  line_count: number
}

// --- Explorer (Backend API) ---

export interface BackendTableInfo {
  name: string
  count: number
}

export interface BackendColumnInfo {
  name: string
  type: string
  notnull: boolean
  pk: boolean
  default: string | null
}

export interface BackendSchemaResponse {
  table: string
  columns: BackendColumnInfo[]
}

export interface BackendSampleResponse {
  table: string
  columns: string[]
  rows: Record<string, unknown>[]
  row_count: number
}

export interface BackendQueryResponse {
  columns: string[]
  rows: Record<string, unknown>[]
  row_count: number
  sql: string
}
