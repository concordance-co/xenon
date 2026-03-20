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

export interface CounterfactualStatus extends PhaseStatus {
  total_files: number
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

export interface FileTreeEntry {
  name: string
  path: string
  type: 'file' | 'dir'
  size?: string
  modified_at: string
  children?: FileTreeEntry[]
}

export interface PcaImage {
  name: string
  modified_at: string
}

export interface AnalysisData {
  total_results: number
  probe_files: string[]
  has_expert_specialization: boolean
  pca_images: PcaImage[]
  file_tree: FileTreeEntry[]
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

export interface PrepTargetSource {
  mode: 'table' | 'sql'
  table?: string
  sql?: string
}

export interface PrepTargetFilters {
  sql_where?: string
}

export interface PrepTargetBucket {
  name: string
  min?: number
  max?: number
}

export interface PrepTargetLabel {
  mode: 'direct' | 'binary_rule' | 'bucket'
  expression_sql: string
  classes?: string[]
  buckets?: PrepTargetBucket[]
}

export interface PrepTargetSplit {
  mode: 'random_stratified' | 'time_based' | 'group_holdout'
  train_pct: number
  val_pct: number
  test_pct: number
  group_key?: string
  time_key?: string
}

export interface PrepTargetProbeDefaults {
  data_source: 'router' | 'residual'
  pooling: 'last_token' | 'mean_pool'
  n_folds: number
  layers?: string
  limit?: number
}

export interface PrepTargetSpec {
  id?: string
  name: string
  description?: string
  source: PrepTargetSource
  filters?: PrepTargetFilters
  label: PrepTargetLabel
  split: PrepTargetSplit
  probe_defaults?: PrepTargetProbeDefaults
  created_at?: string
  updated_at?: string
}

export interface DatasetProfileColumn {
  column: string
  null_count: number
  null_rate: number
  distinct_count: number
  type: string
  sample_values: unknown[]
}

export interface DatasetLabelBalanceRow {
  label: string
  count: number
  pct: number
}

export interface DatasetStratifiedRow {
  stratum: string
  label: string
  count: number
}

export interface BackendDatasetProfileResponse {
  source_sql: string
  sample_limit: number
  row_count: number
  columns: string[]
  profiles: DatasetProfileColumn[]
  label_balance: DatasetLabelBalanceRow[]
  stratified: DatasetStratifiedRow[]
}

export interface SplitViabilityResponse {
  viable: boolean
  reasons: string[]
  counts: {
    train: number
    val: number
    test: number
    total: number
  }
}

export interface ActivationCoverageResponse {
  available: boolean
  reason?: string
  eligible_labeled: number
  matched: number
  coverage: number | null
}

export interface ProbeReadinessResponse {
  can_probe: boolean
  class_count: number
  min_class_count: number
  imbalance_ratio: number | null
  recommended_n_folds: number
  reasons: string[]
}

export interface BackendLabelPreviewResponse {
  sample_limit: number
  row_count: number
  labeled_count: number
  columns: string[]
  label_distribution: DatasetLabelBalanceRow[]
  missing_labels: {
    count: number
    rate: number
  }
  split_viability: SplitViabilityResponse
  activation_coverage: ActivationCoverageResponse
  probe_readiness: ProbeReadinessResponse
  generated_sql: string
}

export interface BackendPrepTargetsResponse {
  specs: PrepTargetSpec[]
}

// --- Payload Explorer ---

export interface DistRow {
  [key: string]: string | number | null
}

export interface PayloadStatsResponse {
  total_logs: number
  unique_vaults?: number
  tool_distribution?: DistRow[]
  model_distribution?: DistRow[]
  slider_trade_size?: DistRow[]
  slider_trading_activity?: DistRow[]
  slider_holding_style?: DistRow[]
  slider_diversification?: DistRow[]
  slider_asset_risk_preference?: DistRow[]
  eth_balance_buckets?: DistRow[]
  portfolio_token_count?: DistRow[]
  strategy_count_dist?: DistRow[]
  memory_depth?: DistRow[]
  token_usage?: Record<string, number | null>
  inference_duration?: Record<string, number | null>
  allowed_tools_combos?: DistRow[]
  risk_activity_heatmap?: DistRow[]
  trade_token_dist?: DistRow[]
  held_token_dist?: DistRow[]
  market_token_dist?: DistRow[]
}

// --- Counterfactual Experiment ---

export interface CounterfactualLayerRow {
  layer: number
  cka: number | null
  transfer_gap_auroc: number
  within_auroc: number | null
  transfer_auroc: number | null
}

export interface CounterfactualLabelSummary {
  n_layers: number
  mean_cka: number | null
  mean_auroc_gap: number | null
  per_layer: CounterfactualLayerRow[]
}

export interface DeltaLayerRow {
  layer: number
  cos: number | null
  cka?: number | null
}

export interface DeltaPositionSummary {
  mean_cos: number | null
  mean_cka?: number | null
  per_layer: DeltaLayerRow[]
}

export interface CounterfactualDecision {
  decision: string
  reasoning: string
  metrics: Record<string, unknown>
}

export interface CounterfactualData {
  dataset_a: {
    status: string
    n_snapshots: number
    n_prompts: number
    n_train?: number
    n_test?: number
    dir?: string
  }
  dataset_b: {
    status: string
    n_prompts: number
  }
  capture: {
    status: string
    total_files: number
    total_size_mb: number
    recent?: Record<string, unknown>[]
  }
  analysis: {
    status: string
    decision: CounterfactualDecision | null
    question_a: {
      labels: Record<string, CounterfactualLabelSummary>
      n_labels: number
    } | null
    question_b: {
      delta_consistency: Record<string, DeltaPositionSummary>
    } | null
    question_c: {
      positions: Record<string, DeltaPositionSummary>
    } | null
  }
}
