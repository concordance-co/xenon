import { useState } from 'react'
import { useFetch } from '../hooks/useApi'
import {
  useConfig,
  buildCounterfactualBuildCmd,
  buildCounterfactualCaptureCmd,
  buildCounterfactualAnalyzeCmd,
} from '../hooks/useConfig'
import type {
  CounterfactualData,
  CounterfactualLayerRow,
  CounterfactualLabelSummary,
  DeltaLayerRow,
  DeltaPositionSummary,
} from '../types/api'
import { Tip } from './Tip'
import s from './shared.module.css'
import cx from './CounterfactualView.module.css'

interface Props {
  onRun: (cmd: string) => void
  refreshKey?: number
}

const DECISION_STYLES: Record<string, { label: string; tone: string }> = {
  objective_market_first: { label: 'Objective Market First', tone: 'green' },
  early_entanglement: { label: 'Early Entanglement', tone: 'rose' },
  late_reinterpretation: { label: 'Late Reinterpretation', tone: 'amber' },
  mixed: { label: 'Mixed / Inconclusive', tone: 'dim' },
  insufficient_data: { label: 'Insufficient Data', tone: 'dim' },
}

export function CounterfactualView({ onRun }: Props) {
  const { data, loading, error } = useFetch<CounterfactualData>('/api/counterfactual')
  const { config, update } = useConfig()
  const c = config.counterfactual
  const [activeStep, setActiveStep] = useState<'build' | 'capture' | 'analyze'>('build')
  const [expandedQuestion, setExpandedQuestion] = useState<string | null>(null)

  if (error) return <div className={s.empty}>Failed to load: {error}</div>
  if (loading || !data) return <div className={s.empty}>Loading...</div>

  const decision = data.analysis.decision
  const decisionStyle = decision ? DECISION_STYLES[decision.decision] ?? DECISION_STYLES.mixed : null

  return (
    <div>
      <p className={s.phaseDesc}>
        Tests whether the Qwen3 surrogate forms an objective market
        understanding first and applies config as a late policy layer, or
        whether config warps market perception from early layers. Three
        questions, tested on stripped canonical prompts (Dataset A) and full
        production prompts (Dataset B).
      </p>

      {/* ── Verdict banner ── */}
      {decision && (
        <div className={cx.verdict} data-tone={decisionStyle?.tone}>
          <div className={cx.verdictLabel}>Experiment Verdict</div>
          <div className={cx.verdictDecision}>{decisionStyle?.label}</div>
          <div className={cx.verdictReasoning}>{decision.reasoning}</div>
        </div>
      )}

      {/* ── Pipeline progress ── */}
      <div className={cx.progress}>
        <ProgressNode
          label="Dataset"
          status={data.dataset_a.status}
          detail={data.dataset_a.status === 'ready'
            ? `${data.dataset_a.n_snapshots} snapshots, ${data.dataset_a.n_prompts} prompts`
            : 'Not built'}
          active={activeStep === 'build'}
          onClick={() => setActiveStep('build')}
        />
        <span className={cx.progressArrow} />
        <ProgressNode
          label="Capture"
          status={data.capture.status}
          detail={data.capture.status === 'ready'
            ? `${data.capture.total_files} files, ${data.capture.total_size_mb} MB`
            : 'No captures'}
          active={activeStep === 'capture'}
          onClick={() => setActiveStep('capture')}
        />
        <span className={cx.progressArrow} />
        <ProgressNode
          label="Analysis"
          status={data.analysis.status}
          detail={data.analysis.status === 'ready'
            ? decisionStyle?.label ?? 'Complete'
            : data.analysis.status === 'partial' ? 'Partial results' : 'Not run'}
          active={activeStep === 'analyze'}
          onClick={() => setActiveStep('analyze')}
        />
      </div>

      {/* ── Step panels ── */}
      {activeStep === 'build' && (
        <BuildPanel config={c} update={update} onRun={onRun} data={data} />
      )}
      {activeStep === 'capture' && (
        <CapturePanel config={c} update={update} onRun={onRun} data={data} />
      )}
      {activeStep === 'analyze' && (
        <AnalyzePanel
          config={c}
          update={update}
          onRun={onRun}
          data={data}
          expandedQuestion={expandedQuestion}
          setExpandedQuestion={setExpandedQuestion}
        />
      )}
    </div>
  )
}


/* ═══════════════════════════════════════════════════════════
   Progress node
   ═══════════════════════════════════════════════════════════ */

function ProgressNode({ label, status, detail, active, onClick }: {
  label: string
  status: string
  detail: string
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      className={`${cx.progressNode} ${active ? cx.progressNodeActive : ''}`}
      onClick={onClick}
    >
      <span className={cx.progressDot} data-status={status} />
      <span className={cx.progressNodeLabel}>{label}</span>
      <span className={cx.progressNodeDetail}>{detail}</span>
    </button>
  )
}


/* ═══════════════════════════════════════════════════════════
   Build panel
   ═══════════════════════════════════════════════════════════ */

function BuildPanel({ config: c, update, onRun, data }: {
  config: Props extends never ? never : ReturnType<typeof useConfig>['config']['counterfactual']
  update: ReturnType<typeof useConfig>['update']
  onRun: (cmd: string) => void
  data: CounterfactualData
}) {
  const cmd = buildCounterfactualBuildCmd(c)

  return (
    <div className={s.panel}>
      <div className={s.panelHead}>
        <span className={s.panelTitle}>Build Datasets</span>
        <StatusBadge status={data.dataset_a.status} />
      </div>
      <div className={s.panelBody}>
        <div className={s.configForm}>
          <div className={s.field}>
            <label className={s.fieldLabel}>
              Experiment ID
              <Tip text="Identifier for this experiment run. Captures and results are stored under this ID." />
            </label>
            <input
              type="text"
              className={s.fieldInput}
              value={c.experimentId}
              onChange={e => update('counterfactual', { experimentId: e.target.value })}
            />
          </div>
          <div className={s.field}>
            <label className={s.fieldLabel}>
              Dataset
              <Tip text="Which dataset to capture. 'both' runs Dataset A (canonical mechanism test) and Dataset B (ecological validation)." />
            </label>
            <select
              className={s.fieldInput}
              value={c.dataset}
              onChange={e => update('counterfactual', { dataset: e.target.value as 'a' | 'b' | 'both' })}
            >
              <option value="both">Both (A + B)</option>
              <option value="a">Dataset A only</option>
              <option value="b">Dataset B only</option>
            </select>
          </div>
          <div className={s.field}>
            <label className={s.fieldLabel}>
              Seed
              <Tip text="Random seed for snapshot sampling, row randomization, and train/test split. Deterministic — same seed = same dataset." />
            </label>
            <input
              type="number"
              className={s.fieldInput}
              value={c.seed}
              onChange={e => update('counterfactual', { seed: Number(e.target.value) || 42 })}
            />
          </div>
        </div>

        {data.dataset_a.status === 'ready' && (
          <div className={cx.datasetStats}>
            <Metric label="Snapshots" value={data.dataset_a.n_snapshots} />
            <Metric label="Prompts" value={data.dataset_a.n_prompts} />
            <Metric label="Train" value={data.dataset_a.n_train ?? '—'} />
            <Metric label="Test" value={data.dataset_a.n_test ?? '—'} />
          </div>
        )}

        <div className={s.generatedCmd}>{cmd}</div>

        <div className={s.btnRow}>
          <button className={`${s.btn} ${s.btnAccent}`} onClick={() => onRun(cmd)}>
            Build Dataset A
          </button>
        </div>
      </div>
    </div>
  )
}


/* ═══════════════════════════════════════════════════════════
   Capture panel
   ═══════════════════════════════════════════════════════════ */

function CapturePanel({ config: c, update, onRun, data }: {
  config: ReturnType<typeof useConfig>['config']['counterfactual']
  update: ReturnType<typeof useConfig>['update']
  onRun: (cmd: string) => void
  data: CounterfactualData
}) {
  const cmd = buildCounterfactualCaptureCmd(c)

  return (
    <div className={s.panel}>
      <div className={s.panelHead}>
        <span className={s.panelTitle}>Counterfactual Capture</span>
        <StatusBadge status={data.capture.status} />
      </div>
      <div className={s.panelBody}>
        <div className={s.configForm}>
          <div className={s.field}>
            <label className={s.fieldLabel}>
              GPU
              <Tip text="GPU type for capture workers. H200 (141GB) fits model + all-layer residuals comfortably. A100-80GB is tighter." />
            </label>
            <select
              className={s.fieldInput}
              value={c.gpu}
              onChange={e => update('counterfactual', { gpu: e.target.value as typeof c.gpu })}
            >
              <option value="H200">H200</option>
              <option value="H100">H100</option>
              <option value="A100-80GB">A100-80GB</option>
            </select>
          </div>
          <div className={s.field}>
            <label className={s.fieldLabel}>
              Batch Size
              <Tip text="Prompts per GPU batch. 750 sends all prompts to a single worker. Lower values fan out across multiple workers." />
            </label>
            <input
              type="number"
              className={s.fieldInput}
              value={c.batchSize}
              min={1}
              max={1000}
              onChange={e => update('counterfactual', { batchSize: Number(e.target.value) || 750 })}
            />
          </div>
        </div>

        {data.capture.total_files > 0 && (
          <div className={cx.datasetStats}>
            <Metric label="Captured" value={data.capture.total_files} />
            <Metric label="Size" value={`${data.capture.total_size_mb} MB`} />
          </div>
        )}

        <div className={s.generatedCmd}>{cmd}</div>

        <div className={s.btnRow}>
          <button className={`${s.btn} ${s.btnAccent}`} onClick={() => onRun(cmd)}>
            Run Capture
          </button>
        </div>
      </div>
    </div>
  )
}


/* ═══════════════════════════════════════════════════════════
   Analyze panel — the heart of it
   ═══════════════════════════════════════════════════════════ */

function AnalyzePanel({ config: c, update, onRun, data, expandedQuestion, setExpandedQuestion }: {
  config: ReturnType<typeof useConfig>['config']['counterfactual']
  update: ReturnType<typeof useConfig>['update']
  onRun: (cmd: string) => void
  data: CounterfactualData
  expandedQuestion: string | null
  setExpandedQuestion: (q: string | null) => void
}) {
  const cmd = buildCounterfactualAnalyzeCmd(c)

  return (
    <>
      <div className={s.panel}>
        <div className={s.panelHead}>
          <span className={s.panelTitle}>Run Analysis</span>
          <StatusBadge status={data.analysis.status} />
        </div>
        <div className={s.panelBody}>
          <div className={s.configForm}>
            <div className={s.field}>
              <label className={s.fieldLabel}>
                Questions
                <Tip text="Which questions to run. A = pre-market entanglement (Dataset A). B = post-market reinterpretation (Dataset B). C = decision-layer interaction (Dataset B). 'all' runs all three." />
              </label>
              <select
                className={s.fieldSelect}
                value={c.questions}
                onChange={e => update('counterfactual', { questions: e.target.value as typeof c.questions })}
              >
                <option value="all">All Questions</option>
                <option value="a">A — Pre-market entanglement</option>
                <option value="b">B — Post-market reinterpretation</option>
                <option value="c">C — Decision-layer interaction</option>
              </select>
            </div>
            <div className={s.field}>
              <label className={s.fieldLabel}>
                Layers
                <Tip text="Comma-separated layer indices to analyze. Empty = all 48. Subset like 0,8,16,24,32,40,47 for faster iteration." />
              </label>
              <input
                type="text"
                className={s.fieldInput}
                value={c.layers}
                placeholder="all layers"
                onChange={e => update('counterfactual', { layers: e.target.value })}
              />
            </div>
            <div className={s.field}>
              <label className={s.fieldLabel}>
                Bootstrap
                <Tip text="Number of bootstrap resamples for confidence intervals. 1000 is standard. 200 for quick checks." />
              </label>
              <input
                type="number"
                className={s.fieldInput}
                value={c.nBootstrap}
                min={100}
                max={10000}
                onChange={e => update('counterfactual', { nBootstrap: Number(e.target.value) || 1000 })}
              />
            </div>
          </div>

          <div className={s.generatedCmd}>{cmd}</div>

          <div className={s.btnRow}>
            <button className={`${s.btn} ${s.btnAccent}`} onClick={() => onRun(cmd)}>
              Run Analysis
            </button>
            <button className={s.btn} onClick={() => onRun('./scripts/modal_capture.sh download-results')}>
              Download Results
            </button>
          </div>
        </div>
      </div>

      {/* ── Three questions ── */}
      <QuestionCard
        id="a"
        title="Pre-market Entanglement"
        question="Does the config-conditional preamble change how the model reads the market?"
        description="Trains probes on low-risk preamble activations at market-row positions, tests transfer to high-risk. Position-controlled via padding."
        dataset="Dataset A (canonical)"
        expanded={expandedQuestion === 'a'}
        onToggle={() => setExpandedQuestion(expandedQuestion === 'a' ? null : 'a')}
        result={data.analysis.question_a}
        renderResult={() => <QuestionAResult data={data.analysis.question_a!} />}
      />

      <QuestionCard
        id="b"
        title="Post-market Reinterpretation"
        question="After seeing ACTIVE SETTINGS, does the model reinterpret market state at downstream positions?"
        description="Tests market-feature probe transfer across settings variants at portfolio, constraints, and previous-decisions positions. Computes delta consistency of config effect vectors."
        dataset="Dataset B (production)"
        expanded={expandedQuestion === 'b'}
        onToggle={() => setExpandedQuestion(expandedQuestion === 'b' ? null : 'b')}
        result={data.analysis.question_b}
        renderResult={() => <QuestionBResult data={data.analysis.question_b!} />}
      />

      <QuestionCard
        id="c"
        title="Decision-layer Interaction"
        question="At the final decision point, is config an additive policy layer or does it interact with market content?"
        description="Measures consistency of config effect vectors (Δ = h_all5 − h_all1) across prompts at last_token. High consistency = additive. Low = content interaction."
        dataset="Dataset B (production)"
        expanded={expandedQuestion === 'c'}
        onToggle={() => setExpandedQuestion(expandedQuestion === 'c' ? null : 'c')}
        result={data.analysis.question_c}
        renderResult={() => <QuestionCResult data={data.analysis.question_c!} />}
      />
    </>
  )
}


/* ═══════════════════════════════════════════════════════════
   Question cards
   ═══════════════════════════════════════════════════════════ */

function QuestionCard({ id, title, question, description, dataset, expanded, onToggle, result, renderResult }: {
  id: string
  title: string
  question: string
  description: string
  dataset: string
  expanded: boolean
  onToggle: () => void
  result: unknown
  renderResult: () => React.ReactNode
}) {
  return (
    <div className={cx.questionCard} data-has-result={result != null}>
      <button className={cx.questionHeader} onClick={onToggle}>
        <span className={cx.questionId}>Q{id.toUpperCase()}</span>
        <div className={cx.questionText}>
          <span className={cx.questionTitle}>{title}</span>
          <span className={cx.questionQ}>{question}</span>
        </div>
        <span className={cx.questionDataset}>{dataset}</span>
        {result != null && (
          <span className={cx.questionChevron} data-expanded={expanded}>
            {expanded ? '−' : '+'}
          </span>
        )}
        {result == null && (
          <span className={cx.questionEmpty}>awaiting data</span>
        )}
      </button>

      {expanded && result != null && (
        <div className={cx.questionBody}>
          <div className={cx.questionDesc}>{description}</div>
          {renderResult()}
        </div>
      )}
    </div>
  )
}


/* ═══════════════════════════════════════════════════════════
   Question A results — CKA heatstrip + transfer gap
   ═══════════════════════════════════════════════════════════ */

function QuestionAResult({ data }: {
  data: { labels: Record<string, CounterfactualLabelSummary>; n_labels: number }
}) {
  const labels = Object.entries(data.labels)

  return (
    <div className={cx.resultGrid}>
      {labels.map(([name, summary]) => (
        <div key={name} className={cx.labelBlock}>
          <div className={cx.labelName}>{formatLabelName(name)}</div>
          <div className={cx.labelMeta}>
            {summary.mean_cka != null && (
              <span>CKA {summary.mean_cka.toFixed(3)}</span>
            )}
            {summary.mean_auroc_gap != null && (
              <span>Gap {summary.mean_auroc_gap > 0 ? '+' : ''}{summary.mean_auroc_gap.toFixed(3)}</span>
            )}
          </div>

          {/* Layer strip visualization */}
          <LayerStrip
            layers={summary.per_layer}
            valueKey="cka"
            colorScale={ckaColor}
            label="CKA"
          />

          <LayerStrip
            layers={summary.per_layer}
            valueKey="transfer_gap_auroc"
            colorScale={gapColor}
            label="Transfer Gap"
          />
        </div>
      ))}
    </div>
  )
}


/* ═══════════════════════════════════════════════════════════
   Question B results — delta consistency per position
   ═══════════════════════════════════════════════════════════ */

function QuestionBResult({ data }: {
  data: { delta_consistency: Record<string, DeltaPositionSummary> }
}) {
  const entries = Object.entries(data.delta_consistency)

  return (
    <div className={cx.resultGrid}>
      {entries.map(([pos, summary]) => (
        <div key={pos} className={cx.labelBlock}>
          <div className={cx.labelName}>{formatPosition(pos)}</div>
          <div className={cx.labelMeta}>
            {summary.mean_cos != null && (
              <span>Cos {summary.mean_cos.toFixed(3)}</span>
            )}
          </div>

          <LayerStrip
            layers={summary.per_layer}
            valueKey="cos"
            colorScale={cosColor}
            label="Delta Consistency"
          />
        </div>
      ))}
    </div>
  )
}


/* ═══════════════════════════════════════════════════════════
   Question C results — additive policy check
   ═══════════════════════════════════════════════════════════ */

function QuestionCResult({ data }: {
  data: { positions: Record<string, DeltaPositionSummary> }
}) {
  const entries = Object.entries(data.positions)

  return (
    <div className={cx.resultGrid}>
      {entries.map(([pos, summary]) => (
        <div key={pos} className={cx.labelBlock}>
          <div className={cx.labelName}>{formatPosition(pos)}</div>
          <div className={cx.labelMeta}>
            {summary.mean_cos != null && (
              <span>Cos {summary.mean_cos.toFixed(3)}</span>
            )}
            {summary.mean_cka != null && (
              <span>CKA {summary.mean_cka.toFixed(3)}</span>
            )}
          </div>

          <LayerStrip
            layers={summary.per_layer}
            valueKey="cos"
            colorScale={cosColor}
            label="Effect Consistency"
          />

          {summary.per_layer[0]?.cka != null && (
            <LayerStrip
              layers={summary.per_layer}
              valueKey="cka"
              colorScale={ckaColor}
              label="CKA"
            />
          )}
        </div>
      ))}
    </div>
  )
}


/* ═══════════════════════════════════════════════════════════
   Layer strip — a horizontal heatmap of per-layer values
   ═══════════════════════════════════════════════════════════ */

function LayerStrip({ layers, valueKey, colorScale, label }: {
  layers: (CounterfactualLayerRow | DeltaLayerRow)[]
  valueKey: string
  colorScale: (v: number) => string
  label: string
}) {
  if (!layers.length) return null

  return (
    <div className={cx.strip}>
      <span className={cx.stripLabel}>{label}</span>
      <div className={cx.stripTrack}>
        {layers.map((d, i) => {
          const rec = d as unknown as Record<string, number | null | undefined>
          const v = (rec[valueKey] ?? null) as number | null
          return (
            <div
              key={i}
              className={cx.stripCell}
              style={{ background: v != null ? colorScale(v) : 'var(--border)' }}
              title={`Layer ${d.layer ?? i}: ${v != null ? v.toFixed(4) : '—'}`}
            />
          )
        })}
      </div>
      {layers.length > 0 && (
        <span className={cx.stripRange}>
          L0 — L{layers[layers.length - 1].layer ?? layers.length - 1}
        </span>
      )}
    </div>
  )
}


/* ═══════════════════════════════════════════════════════════
   Shared small components
   ═══════════════════════════════════════════════════════════ */

function StatusBadge({ status }: { status: string }) {
  const cls = status === 'ready' ? s.badgeGreen
    : status === 'partial' ? s.badgeAmber
    : s.badgeDim
  return <span className={`${s.badge} ${cls}`}>{status}</span>
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className={cx.metric}>
      <span className={cx.metricValue}>
        {typeof value === 'number' ? value.toLocaleString() : value}
      </span>
      <span className={cx.metricLabel}>{label}</span>
    </div>
  )
}


/* ═══════════════════════════════════════════════════════════
   Color scales
   ═══════════════════════════════════════════════════════════ */

function ckaColor(v: number): string {
  // 0.85 → rose, 0.90 → amber, 0.95+ → green
  if (v >= 0.95) return 'oklch(62% 0.14 145 / 0.7)'
  if (v >= 0.90) return 'oklch(72% 0.12 75 / 0.6)'
  if (v >= 0.85) return 'oklch(72% 0.12 75 / 0.3)'
  return 'oklch(58% 0.16 15 / 0.5)'
}

function gapColor(v: number): string {
  const abs = Math.abs(v)
  if (abs < 0.02) return 'oklch(62% 0.14 145 / 0.6)'
  if (abs < 0.05) return 'oklch(72% 0.12 75 / 0.5)'
  if (abs < 0.10) return 'oklch(72% 0.12 75 / 0.3)'
  return 'oklch(58% 0.16 15 / 0.5)'
}

function cosColor(v: number): string {
  // High consistency → green, low → rose
  if (v >= 0.7) return 'oklch(62% 0.14 145 / 0.7)'
  if (v >= 0.5) return 'oklch(72% 0.12 75 / 0.5)'
  if (v >= 0.3) return 'oklch(72% 0.12 75 / 0.3)'
  return 'oklch(58% 0.16 15 / 0.5)'
}


/* ═══════════════════════════════════════════════════════════
   Formatters
   ═══════════════════════════════════════════════════════════ */

function formatLabelName(name: string): string {
  return name.replace(/^is_/, '').replace(/_/g, ' ')
}

function formatPosition(pos: string): string {
  return pos.replace(/_eos$/, '').replace(/_/g, ' ')
}
