import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useBackendUrl } from '../hooks/useBackend'
import type {
  BackendDatasetProfileResponse,
  BackendLabelPreviewResponse,
  BackendPrepTargetsResponse,
  BackendQueryResponse,
  BackendSampleResponse,
  BackendSchemaResponse,
  BackendTableInfo,
  DistRow,
  PayloadStatsResponse,
  PrepTargetSpec,
} from '../types/api'
import s from './shared.module.css'
import x from './ExplorerView.module.css'

type ExplorerWorkspace = 'query' | 'label' | 'probe' | 'payload'

const QUERY_HISTORY_KEY = 'xenon-explorer-query-history'

function newSpec(defaultTable = 'interp_examples_v0'): PrepTargetSpec {
  return {
    name: 'New prep target',
    description: '',
    source: { mode: 'table', table: defaultTable },
    filters: { sql_where: '' },
    label: {
      mode: 'direct',
      expression_sql: 'decision_type',
      classes: ['negative', 'positive'],
      buckets: [
        { name: 'loss', max: 0 },
        { name: 'win', min: 0 },
      ],
    },
    split: {
      mode: 'random_stratified',
      train_pct: 70,
      val_pct: 15,
      test_pct: 15,
    },
    probe_defaults: {
      data_source: 'router',
      pooling: 'last_token',
      n_folds: 5,
      layers: '',
      limit: 0,
    },
  }
}

function bucketsToText(spec: PrepTargetSpec): string {
  if (!spec.label.buckets?.length) return ''
  return spec.label.buckets
    .map(b => `${b.name}:${b.min ?? ''}:${b.max ?? ''}`)
    .join('\n')
}

function parseBucketText(text: string) {
  return text
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)
    .map(line => {
      const [nameRaw, minRaw, maxRaw] = line.split(':')
      const name = (nameRaw || '').trim() || 'bucket'
      const min = (minRaw || '').trim() === '' ? undefined : Number(minRaw)
      const max = (maxRaw || '').trim() === '' ? undefined : Number(maxRaw)
      return {
        name,
        min: Number.isFinite(min ?? NaN) ? min : undefined,
        max: Number.isFinite(max ?? NaN) ? max : undefined,
      }
    })
}

function inferBuiltInTarget(spec: PrepTargetSpec) {
  if (spec.label.mode !== 'direct') return null
  const expr = spec.label.expression_sql.trim()
  if (expr === 'decision_type') return 'decision_type'
  if (expr === 'trade_side') return 'trade_side'
  if (expr === 'was_profitable_1h') return 'was_profitable_1h'
  if (expr === 'asset') return 'asset'
  if (expr === 'vault_risk_preference') return 'risk_tolerance'
  return null
}

export function ExplorerView() {
  const {
    url,
    loading: urlLoading,
    error: urlError,
    backendFetch,
    backendPost,
    backendDelete,
  } = useBackendUrl()

  const [workspace, setWorkspace] = useState<ExplorerWorkspace>('query')

  const [tables, setTables] = useState<BackendTableInfo[]>([])
  const [selectedTable, setSelectedTable] = useState('')
  const [schema, setSchema] = useState<BackendSchemaResponse | null>(null)
  const [sample, setSample] = useState<BackendSampleResponse | null>(null)
  const [sampleSize, setSampleSize] = useState(20)

  const [sql, setSql] = useState('')
  const [queryResult, setQueryResult] = useState<BackendQueryResponse | null>(null)
  const [queryError, setQueryError] = useState<string | null>(null)
  const [queryLoading, setQueryLoading] = useState(false)
  const [queryHistory, setQueryHistory] = useState<string[]>([])

  const [specs, setSpecs] = useState<PrepTargetSpec[]>([])
  const [specLoading, setSpecLoading] = useState(false)
  const [selectedSpecId, setSelectedSpecId] = useState<string>('')
  const [draftSpec, setDraftSpec] = useState<PrepTargetSpec>(newSpec())
  const [bucketText, setBucketText] = useState<string>(bucketsToText(newSpec()))
  const [profile, setProfile] = useState<BackendDatasetProfileResponse | null>(null)
  const [preview, setPreview] = useState<BackendLabelPreviewResponse | null>(null)
  const [labelError, setLabelError] = useState<string | null>(null)
  const [labelLoading, setLabelLoading] = useState(false)

  const [copied, setCopied] = useState(false)

  useEffect(() => {
    try {
      const raw = localStorage.getItem(QUERY_HISTORY_KEY)
      if (!raw) return
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) {
        setQueryHistory(parsed.filter(v => typeof v === 'string').slice(0, 12))
      }
    } catch {
      // ignore local storage parse issues
    }
  }, [])

  const refreshSpecs = useCallback(async () => {
    if (!url) return
    setSpecLoading(true)
    try {
      const result = await backendFetch<BackendPrepTargetsResponse>('/prep-targets')
      setSpecs(result.specs)
    } finally {
      setSpecLoading(false)
    }
  }, [url, backendFetch])

  useEffect(() => {
    if (!url) return

    backendFetch<{ tables: BackendTableInfo[] }>('/tables')
      .then(result => {
        setTables(result.tables)
        if (!selectedTable && result.tables.length > 0) {
          const fallback = result.tables[0].name
          setSelectedTable(fallback)
          setSql(`SELECT * FROM ${fallback}`)
          setDraftSpec(prev => {
            if (prev.source.mode === 'table' && !prev.source.table) {
              return { ...prev, source: { mode: 'table', table: fallback } }
            }
            return prev
          })
        }
      })
      .catch(() => {
        // ignore, UI will reflect missing table data
      })

    refreshSpecs()
  }, [url, backendFetch, refreshSpecs])

  useEffect(() => {
    if (!selectedTable || !url) {
      setSchema(null)
      setSample(null)
      return
    }
    backendFetch<BackendSchemaResponse>(`/schema?table=${selectedTable}`).then(setSchema).catch(() => setSchema(null))
    backendFetch<BackendSampleResponse>(`/sample/${selectedTable}?n=${sampleSize}`).then(setSample).catch(() => setSample(null))
  }, [selectedTable, sampleSize, url, backendFetch])

  useEffect(() => {
    setBucketText(bucketsToText(draftSpec))
  }, [draftSpec.id, draftSpec.label.mode])

  const resample = useCallback(() => {
    if (!selectedTable || !url) return
    backendFetch<BackendSampleResponse>(`/sample/${selectedTable}?n=${sampleSize}`).then(setSample).catch(() => setSample(null))
  }, [selectedTable, sampleSize, url, backendFetch])

  const handleSelectTable = useCallback((name: string) => {
    setSelectedTable(name)
    setSql(`SELECT * FROM ${name}`)
    setQueryResult(null)
    setQueryError(null)
  }, [])

  const recordQueryHistory = useCallback((statement: string) => {
    const normalized = statement.trim()
    if (!normalized) return
    setQueryHistory(prev => {
      const next = [normalized, ...prev.filter(item => item !== normalized)].slice(0, 12)
      localStorage.setItem(QUERY_HISTORY_KEY, JSON.stringify(next))
      return next
    })
  }, [])

  const runQuery = useCallback(async () => {
    if (!sql.trim()) return
    setQueryLoading(true)
    setQueryError(null)
    try {
      const result = await backendPost<BackendQueryResponse>('/query', { sql, limit: 400 })
      setQueryResult(result)
      recordQueryHistory(sql)
    } catch (e) {
      setQueryError(e instanceof Error ? e.message : 'Query failed')
      setQueryResult(null)
    } finally {
      setQueryLoading(false)
    }
  }, [sql, backendPost, recordQueryHistory])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault()
      runQuery()
    }
  }, [runQuery])

  const handleLoadSpec = useCallback((spec: PrepTargetSpec) => {
    setDraftSpec(spec)
    setSelectedSpecId(spec.id ?? '')
    setLabelError(null)
  }, [])

  const handleNewSpec = useCallback(() => {
    const fallbackTable = selectedTable || tables[0]?.name || 'interp_examples_v0'
    setSelectedSpecId('')
    setDraftSpec(newSpec(fallbackTable))
    setLabelError(null)
  }, [selectedTable, tables])

  const handleSaveSpec = useCallback(async () => {
    setLabelLoading(true)
    setLabelError(null)
    try {
      const response = await backendPost<{ spec: PrepTargetSpec }>('/prep-targets', draftSpec)
      setDraftSpec(response.spec)
      setSelectedSpecId(response.spec.id ?? '')
      await refreshSpecs()
    } catch (e) {
      setLabelError(e instanceof Error ? e.message : 'Failed to save prep target')
    } finally {
      setLabelLoading(false)
    }
  }, [backendPost, draftSpec, refreshSpecs])

  const handleDeleteSpec = useCallback(async () => {
    if (!draftSpec.id) return
    setLabelLoading(true)
    setLabelError(null)
    try {
      await backendDelete(`/prep-targets/${draftSpec.id}`)
      await refreshSpecs()
      handleNewSpec()
    } catch (e) {
      setLabelError(e instanceof Error ? e.message : 'Failed to delete prep target')
    } finally {
      setLabelLoading(false)
    }
  }, [backendDelete, draftSpec.id, handleNewSpec, refreshSpecs])

  const handleProfile = useCallback(async () => {
    setLabelLoading(true)
    setLabelError(null)
    try {
      const result = await backendPost<BackendDatasetProfileResponse>('/profile/dataset', {
        source: draftSpec.source,
        limit: 1000,
      })
      setProfile(result)
    } catch (e) {
      setLabelError(e instanceof Error ? e.message : 'Dataset profile failed')
      setProfile(null)
    } finally {
      setLabelLoading(false)
    }
  }, [backendPost, draftSpec.source])

  const handlePreview = useCallback(async () => {
    setLabelLoading(true)
    setLabelError(null)
    try {
      const result = await backendPost<BackendLabelPreviewResponse>('/label/preview', {
        spec: draftSpec,
        limit: 2000,
      })
      setPreview(result)
    } catch (e) {
      setLabelError(e instanceof Error ? e.message : 'Label preview failed')
      setPreview(null)
    } finally {
      setLabelLoading(false)
    }
  }, [backendPost, draftSpec])

  const chartData = useMemo(() => {
    if (!queryResult || queryResult.rows.length === 0) return null

    const numericColumn = queryResult.columns.find(col => {
      return queryResult.rows.some(row => {
        const value = row[col]
        return typeof value === 'number'
      })
    })

    if (!numericColumn) return null

    const categoryColumn = queryResult.columns.find(col => col !== numericColumn)
    const rows = queryResult.rows.slice(0, 20)

    return rows.map((row, idx) => ({
      key: categoryColumn ? String(row[categoryColumn] ?? `row_${idx + 1}`) : `row_${idx + 1}`,
      value: Number(row[numericColumn] ?? 0),
    }))
  }, [queryResult])

  const probeCommand = useMemo(() => {
    const defaults = draftSpec.probe_defaults ?? {
      data_source: 'router' as const,
      pooling: 'last_token' as const,
      n_folds: 5,
      layers: '',
      limit: 0,
    }
    const target = inferBuiltInTarget(draftSpec)
    let command = './scripts/modal_capture.sh analyze --mode probe '
    command += `--target ${target ?? 'decision_type'}`
    if (defaults.data_source === 'residual') command += ' --data-source residual'
    if (defaults.pooling === 'mean_pool') command += ' --pooling mean_pool'
    if (defaults.n_folds !== 5) command += ` --n-folds ${defaults.n_folds}`
    if (defaults.layers && defaults.layers.trim()) command += ` --layers ${defaults.layers.trim()}`
    if ((defaults.limit ?? 0) > 0) command += ` --limit ${defaults.limit}`

    if (!target) {
      command += '\n# Note: custom labels are not yet wired into analysis CLI targets.'
      command += '\n# Use this as a baseline command and attach prep-target metadata in your run notes.'
    }

    return command
  }, [draftSpec])

  const copyProbeCommand = useCallback(async () => {
    await navigator.clipboard.writeText(probeCommand)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1500)
  }, [probeCommand])

  if (urlLoading) return <div className={s.empty}>Connecting to backend...</div>

  if (urlError) {
    return (
      <div className={s.empty}>
        <p style={{ marginBottom: 8 }}>Backend not configured</p>
        <code style={{ fontSize: '0.75rem', color: 'var(--text-3)' }}>
          Set XENON_BACKEND_URL or write URL to ~/.xenon_backend_url
        </code>
      </div>
    )
  }

  return (
    <div className={x.explorerRoot}>
      <div className={x.workspaceNav}>
        <button
          className={workspace === 'query' ? x.workspaceTabActive : x.workspaceTab}
          onClick={() => setWorkspace('query')}
        >
          Query Lab
        </button>
        <button
          className={workspace === 'label' ? x.workspaceTabActive : x.workspaceTab}
          onClick={() => setWorkspace('label')}
        >
          Label Lab
        </button>
        <button
          className={workspace === 'probe' ? x.workspaceTabActive : x.workspaceTab}
          onClick={() => setWorkspace('probe')}
        >
          Probe Prep
        </button>
        <button
          className={workspace === 'payload' ? x.workspaceTabActive : x.workspaceTab}
          onClick={() => setWorkspace('payload')}
        >
          Payload Explorer
        </button>
      </div>

      {workspace === 'query' && (
        <div>
          <div className={x.toolbar}>
            <select
              className={s.fieldSelect}
              value={selectedTable}
              onChange={e => handleSelectTable(e.target.value)}
            >
              <option value="">Select a table...</option>
              {tables.map(t => (
                <option key={t.name} value={t.name}>
                  {t.name} ({t.count.toLocaleString()} rows)
                </option>
              ))}
            </select>

            <input
              type="number"
              className={s.fieldInput}
              value={sampleSize}
              min={1}
              max={500}
              style={{ width: 80 }}
              onChange={e => setSampleSize(Number(e.target.value) || 20)}
            />
            <button className={s.btn} onClick={resample}>Resample</button>
          </div>

          <div className={x.queryLayout}>
            <div className={s.panel}>
              <div className={s.panelHead}>
                <span className={s.panelTitle}>Schema</span>
                <span className={x.rowCount}>{schema?.columns.length ?? 0} columns</span>
              </div>
              <div className={s.panelBody}>
                {schema ? (
                  <div className={x.dataGrid}>
                    <table className={s.table}>
                      <thead>
                        <tr>
                          <th>Column</th>
                          <th>Type</th>
                          <th>PK</th>
                          <th>Not Null</th>
                        </tr>
                      </thead>
                      <tbody>
                        {schema.columns.map(c => (
                          <tr key={c.name}>
                            <td className="mono">{c.name}</td>
                            <td className="mono">{c.type || 'ANY'}</td>
                            <td>{c.pk ? 'yes' : ''}</td>
                            <td>{c.notnull ? 'yes' : ''}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className={s.empty}>Select a table to inspect schema</div>
                )}
              </div>
            </div>

            <div className={s.panel}>
              <div className={s.panelHead}>
                <span className={s.panelTitle}>Sample</span>
                <span className={x.rowCount}>{sample?.row_count ?? 0} rows</span>
              </div>
              <div className={s.panelBody}>
                {sample ? <DataGrid columns={sample.columns} rows={sample.rows} /> : <div className={s.empty}>No sample loaded</div>}
              </div>
            </div>
          </div>

          <div className={s.panel}>
            <div className={s.panelHead}>
              <span className={s.panelTitle}>SQL Query</span>
              <span className={x.hint}>Cmd+Enter to run</span>
            </div>
            <div className={s.panelBody}>
              <textarea
                className={x.queryInput}
                value={sql}
                onChange={e => setSql(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="SELECT * FROM interp_examples_v0 LIMIT 100"
                spellCheck={false}
              />

              <div className={s.btnRow}>
                <button
                  className={`${s.btn} ${s.btnAccent}`}
                  onClick={runQuery}
                  disabled={queryLoading || !sql.trim()}
                >
                  {queryLoading ? 'Running...' : 'Run Query'}
                </button>
                {queryResult && <span className={x.rowCount}>{queryResult.row_count} rows returned</span>}
              </div>

              {queryHistory.length > 0 && (
                <div className={x.historyStrip}>
                  <span className={x.historyLabel}>Recent</span>
                  {queryHistory.map((item, idx) => (
                    <button
                      key={`${item}-${idx}`}
                      className={x.historyBtn}
                      onClick={() => setSql(item)}
                      title={item}
                    >
                      {item.length > 52 ? `${item.slice(0, 52)}...` : item}
                    </button>
                  ))}
                </div>
              )}

              {queryError && <div className={x.errorBanner}>{queryError}</div>}

              {queryResult && queryResult.rows.length > 0 && (
                <div className={x.queryResultGrid}>
                  <DataGrid columns={queryResult.columns} rows={queryResult.rows} />
                  <div className={s.panel}>
                    <div className={s.panelHead}>
                      <span className={s.panelTitle}>Chart Preview</span>
                    </div>
                    <div className={s.panelBody}>
                      {chartData && chartData.length > 0 ? (
                        <div className={x.chartWrap}>
                          <ResponsiveContainer>
                            <BarChart data={chartData} margin={{ top: 8, right: 12, left: 8, bottom: 20 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                              <XAxis
                                dataKey="key"
                                angle={-22}
                                textAnchor="end"
                                height={56}
                                tick={{ fontSize: 10, fill: 'var(--text-3)' }}
                              />
                              <YAxis tick={{ fontSize: 10, fill: 'var(--text-3)' }} />
                              <Tooltip />
                              <Bar dataKey="value" fill="var(--accent)" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      ) : (
                        <div className={s.empty}>No numeric column available for preview chart</div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {workspace === 'label' && (
        <div className={x.labelLayout}>
          <div className={s.panel}>
            <div className={s.panelHead}>
              <span className={s.panelTitle}>Shared Prep Targets</span>
              <span className={x.rowCount}>{specLoading ? 'Loading...' : `${specs.length} specs`}</span>
            </div>
            <div className={s.panelBody}>
              <div className={x.specList}>
                {specs.map(spec => (
                  <button
                    key={spec.id ?? spec.name}
                    className={selectedSpecId === spec.id ? x.specItemActive : x.specItem}
                    onClick={() => handleLoadSpec(spec)}
                  >
                    <span>{spec.name}</span>
                    <span className={x.specItemMeta}>{spec.updated_at?.slice(0, 10) ?? 'unsaved'}</span>
                  </button>
                ))}
              </div>
              <div className={s.btnRow}>
                <button className={s.btn} onClick={refreshSpecs}>Refresh</button>
                <button className={s.btn} onClick={handleNewSpec}>New</button>
              </div>
            </div>
          </div>

          <div className={s.panel}>
            <div className={s.panelHead}>
              <span className={s.panelTitle}>Prep Target Builder</span>
              <span className={x.rowCount}>{draftSpec.id ? `id: ${draftSpec.id}` : 'new spec'}</span>
            </div>
            <div className={s.panelBody}>
              <div className={x.formGrid}>
                <label className={s.field}>
                  <span className={s.fieldLabel}>Name</span>
                  <input
                    className={s.fieldInput}
                    value={draftSpec.name}
                    onChange={e => setDraftSpec(prev => ({ ...prev, name: e.target.value }))}
                  />
                </label>

                <label className={s.field}>
                  <span className={s.fieldLabel}>Description</span>
                  <input
                    className={s.fieldInput}
                    value={draftSpec.description ?? ''}
                    onChange={e => setDraftSpec(prev => ({ ...prev, description: e.target.value }))}
                  />
                </label>

                <label className={s.field}>
                  <span className={s.fieldLabel}>Source Mode</span>
                  <select
                    className={s.fieldSelect}
                    value={draftSpec.source.mode}
                    onChange={e => {
                      const mode = e.target.value as 'table' | 'sql'
                      setDraftSpec(prev => ({
                        ...prev,
                        source: mode === 'table'
                          ? { mode, table: prev.source.table || selectedTable || tables[0]?.name || 'interp_examples_v0' }
                          : { mode, sql: prev.source.sql || `SELECT * FROM ${selectedTable || 'interp_examples_v0'}` },
                      }))
                    }}
                  >
                    <option value="table">Table</option>
                    <option value="sql">SQL</option>
                  </select>
                </label>

                {draftSpec.source.mode === 'table' ? (
                  <label className={s.field}>
                    <span className={s.fieldLabel}>Source Table</span>
                    <select
                      className={s.fieldSelect}
                      value={draftSpec.source.table ?? ''}
                      onChange={e => setDraftSpec(prev => ({ ...prev, source: { mode: 'table', table: e.target.value } }))}
                    >
                      {tables.map(t => <option key={t.name} value={t.name}>{t.name}</option>)}
                    </select>
                  </label>
                ) : (
                  <label className={s.field}>
                    <span className={s.fieldLabel}>Source SQL</span>
                    <textarea
                      className={x.codeArea}
                      value={draftSpec.source.sql ?? ''}
                      onChange={e => setDraftSpec(prev => ({ ...prev, source: { mode: 'sql', sql: e.target.value } }))}
                    />
                  </label>
                )}

                <label className={s.field}>
                  <span className={s.fieldLabel}>Filter (SQL WHERE)</span>
                  <input
                    className={s.fieldInput}
                    value={draftSpec.filters?.sql_where ?? ''}
                    placeholder="decision_type = 'trade'"
                    onChange={e => setDraftSpec(prev => ({
                      ...prev,
                      filters: { ...(prev.filters ?? {}), sql_where: e.target.value },
                    }))}
                  />
                </label>

                <label className={s.field}>
                  <span className={s.fieldLabel}>Label Mode</span>
                  <select
                    className={s.fieldSelect}
                    value={draftSpec.label.mode}
                    onChange={e => setDraftSpec(prev => ({
                      ...prev,
                      label: { ...prev.label, mode: e.target.value as 'direct' | 'binary_rule' | 'bucket' },
                    }))}
                  >
                    <option value="direct">Direct</option>
                    <option value="binary_rule">Binary Rule</option>
                    <option value="bucket">Bucket</option>
                  </select>
                </label>

                <label className={s.field}>
                  <span className={s.fieldLabel}>Label Expression (SQL)</span>
                  <input
                    className={s.fieldInput}
                    value={draftSpec.label.expression_sql}
                    onChange={e => setDraftSpec(prev => ({ ...prev, label: { ...prev.label, expression_sql: e.target.value } }))}
                  />
                </label>

                {draftSpec.label.mode === 'binary_rule' && (
                  <label className={s.field}>
                    <span className={s.fieldLabel}>Class Names (neg,pos)</span>
                    <input
                      className={s.fieldInput}
                      value={(draftSpec.label.classes ?? ['negative', 'positive']).join(',')}
                      onChange={e => {
                        const parsed = e.target.value.split(',').map(v => v.trim()).filter(Boolean)
                        setDraftSpec(prev => ({ ...prev, label: { ...prev.label, classes: parsed } }))
                      }}
                    />
                  </label>
                )}

                {draftSpec.label.mode === 'bucket' && (
                  <label className={s.field}>
                    <span className={s.fieldLabel}>Buckets (name:min:max per line)</span>
                    <textarea
                      className={x.codeArea}
                      value={bucketText}
                      onChange={e => {
                        const value = e.target.value
                        setBucketText(value)
                        setDraftSpec(prev => ({
                          ...prev,
                          label: { ...prev.label, buckets: parseBucketText(value) },
                        }))
                      }}
                    />
                  </label>
                )}

                <label className={s.field}>
                  <span className={s.fieldLabel}>Split Mode</span>
                  <select
                    className={s.fieldSelect}
                    value={draftSpec.split.mode}
                    onChange={e => setDraftSpec(prev => ({ ...prev, split: { ...prev.split, mode: e.target.value as 'random_stratified' | 'time_based' | 'group_holdout' } }))}
                  >
                    <option value="random_stratified">Random Stratified</option>
                    <option value="time_based">Time Based</option>
                    <option value="group_holdout">Group Holdout</option>
                  </select>
                </label>

                <label className={s.field}>
                  <span className={s.fieldLabel}>Train %</span>
                  <input
                    type="number"
                    className={s.fieldInput}
                    value={draftSpec.split.train_pct}
                    onChange={e => setDraftSpec(prev => ({ ...prev, split: { ...prev.split, train_pct: Number(e.target.value) || 0 } }))}
                  />
                </label>
                <label className={s.field}>
                  <span className={s.fieldLabel}>Val %</span>
                  <input
                    type="number"
                    className={s.fieldInput}
                    value={draftSpec.split.val_pct}
                    onChange={e => setDraftSpec(prev => ({ ...prev, split: { ...prev.split, val_pct: Number(e.target.value) || 0 } }))}
                  />
                </label>
                <label className={s.field}>
                  <span className={s.fieldLabel}>Test %</span>
                  <input
                    type="number"
                    className={s.fieldInput}
                    value={draftSpec.split.test_pct}
                    onChange={e => setDraftSpec(prev => ({ ...prev, split: { ...prev.split, test_pct: Number(e.target.value) || 0 } }))}
                  />
                </label>
              </div>

              <div className={s.btnRow}>
                <button className={`${s.btn} ${s.btnAccent}`} onClick={handlePreview} disabled={labelLoading}>
                  Preview Labels
                </button>
                <button className={s.btn} onClick={handleProfile} disabled={labelLoading}>
                  Profile Source
                </button>
                <button className={s.btn} onClick={handleSaveSpec} disabled={labelLoading}>
                  Save Shared Spec
                </button>
                <button className={s.btn} onClick={handleDeleteSpec} disabled={labelLoading || !draftSpec.id}>
                  Delete
                </button>
              </div>

              {labelError && <div className={x.errorBanner}>{labelError}</div>}

              {preview && (
                <div className={x.previewBlocks}>
                  <div className={x.metricRow}>
                    <Metric label="Rows" value={preview.row_count} />
                    <Metric label="Labeled" value={preview.labeled_count} />
                    <Metric label="Missing" value={preview.missing_labels.count} />
                    <Metric label="Probe Ready" value={preview.probe_readiness.can_probe ? 'yes' : 'no'} />
                  </div>

                  <div className={x.previewGrid}>
                    <div>
                      <div className={x.subhead}>Label Distribution</div>
                      <table className={s.table}>
                        <thead>
                          <tr><th>Label</th><th style={{ textAlign: 'right' }}>Count</th><th style={{ textAlign: 'right' }}>Pct</th></tr>
                        </thead>
                        <tbody>
                          {preview.label_distribution.map(row => (
                            <tr key={row.label}>
                              <td className="mono">{row.label}</td>
                              <td className="mono" style={{ textAlign: 'right' }}>{row.count}</td>
                              <td className="mono" style={{ textAlign: 'right' }}>{(row.pct * 100).toFixed(1)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    <div>
                      <div className={x.subhead}>Split Viability</div>
                      <div className={x.readinessBox}>
                        <div className="mono">Viable: {preview.split_viability.viable ? 'yes' : 'no'}</div>
                        <div className="mono">Train: {preview.split_viability.counts.train}</div>
                        <div className="mono">Val: {preview.split_viability.counts.val}</div>
                        <div className="mono">Test: {preview.split_viability.counts.test}</div>
                        {preview.split_viability.reasons.length > 0 && (
                          <ul className={x.reasonList}>
                            {preview.split_viability.reasons.map(reason => <li key={reason}>{reason}</li>)}
                          </ul>
                        )}
                      </div>

                      <div className={x.subhead}>Activation Coverage</div>
                      <div className={x.readinessBox}>
                        <div className="mono">Available: {preview.activation_coverage.available ? 'yes' : 'no'}</div>
                        <div className="mono">Eligible labeled: {preview.activation_coverage.eligible_labeled}</div>
                        <div className="mono">Matched: {preview.activation_coverage.matched}</div>
                        <div className="mono">Coverage: {preview.activation_coverage.coverage == null ? 'n/a' : `${(preview.activation_coverage.coverage * 100).toFixed(1)}%`}</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {profile && (
                <div style={{ marginTop: 'var(--space-xl)' }}>
                  <div className={x.subhead}>Source Profile ({profile.row_count} sampled rows)</div>
                  <div className={x.dataGrid}>
                    <table className={s.table}>
                      <thead>
                        <tr>
                          <th>Column</th>
                          <th>Type</th>
                          <th style={{ textAlign: 'right' }}>Null %</th>
                          <th style={{ textAlign: 'right' }}>Distinct</th>
                          <th>Samples</th>
                        </tr>
                      </thead>
                      <tbody>
                        {profile.profiles.map(col => (
                          <tr key={col.column}>
                            <td className="mono">{col.column}</td>
                            <td className="mono">{col.type}</td>
                            <td className="mono" style={{ textAlign: 'right' }}>{(col.null_rate * 100).toFixed(1)}%</td>
                            <td className="mono" style={{ textAlign: 'right' }}>{col.distinct_count}</td>
                            <td className="mono">{col.sample_values.map(String).join(', ')}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {workspace === 'probe' && (
        <div className={x.probeLayout}>
          <div className={s.panel}>
            <div className={s.panelHead}>
              <span className={s.panelTitle}>Probe Prep</span>
              <span className={x.rowCount}>Read-only planning</span>
            </div>
            <div className={s.panelBody}>
              <p className={x.probeText}>
                This workspace does not run pipeline commands. It packages your selected prep target,
                readiness signals, and a copyable analysis command template.
              </p>

              <div className={x.metricRow}>
                <Metric label="Spec" value={draftSpec.name || 'Untitled'} mono={false} />
                <Metric label="Target" value={inferBuiltInTarget(draftSpec) ?? 'custom'} />
                <Metric label="Ready" value={preview?.probe_readiness.can_probe ? 'yes' : 'unknown'} />
                <Metric label="Rec Folds" value={preview?.probe_readiness.recommended_n_folds ?? 'n/a'} />
              </div>

              <div className={x.subhead}>Generated Command</div>
              <pre className={x.cmdBlock}>{probeCommand}</pre>

              <div className={s.btnRow}>
                <button className={`${s.btn} ${s.btnAccent}`} onClick={copyProbeCommand}>
                  {copied ? 'Copied' : 'Copy Command'}
                </button>
                <button className={s.btn} onClick={handlePreview}>Refresh Readiness</button>
              </div>

              {preview?.probe_readiness.reasons && preview.probe_readiness.reasons.length > 0 && (
                <div style={{ marginTop: 'var(--space-lg)' }}>
                  <div className={x.subhead}>Readiness Notes</div>
                  <ul className={x.reasonList}>
                    {preview.probe_readiness.reasons.map(reason => <li key={reason}>{reason}</li>)}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {workspace === 'payload' && (
        <PayloadExplorer backendFetch={backendFetch} />
      )}
    </div>
  )
}

function Metric({ label, value, mono = true }: { label: string; value: string | number; mono?: boolean }) {
  return (
    <div className={x.metricCard}>
      <div className={x.metricLabel}>{label}</div>
      <div className={mono ? 'mono' : undefined}>{value}</div>
    </div>
  )
}

function DataGrid({ columns, rows }: { columns: string[]; rows: Record<string, unknown>[] }) {
  if (!rows.length) return <div className={s.empty}>No rows</div>

  const fmtCell = (v: unknown): string => {
    if (v === null || v === undefined) return '\u2014'
    if (typeof v === 'object') return JSON.stringify(v)
    return String(v)
  }

  return (
    <div className={x.dataGrid}>
      <table className={s.table}>
        <thead>
          <tr>
            {columns.map(c => <th key={c}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map(c => {
                const val = fmtCell(row[c])
                return (
                  <td key={c} className={`mono ${x.cellTruncate}`} title={val}>
                    {val}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ────────────────────────── Payload Explorer ────────────────────────── */

const ACCENT_COLORS = [
  'var(--accent)',
  'oklch(72% 0.15 160)',
  'oklch(72% 0.15 280)',
  'oklch(72% 0.15 40)',
  'oklch(72% 0.15 200)',
  'oklch(65% 0.14 100)',
]

function MiniBar({
  title,
  data,
  dataKey = 'count',
  nameKey = 'name',
  color = 'var(--accent)',
}: {
  title: string
  data: DistRow[] | undefined
  dataKey?: string
  nameKey?: string
  color?: string
}) {
  if (!data || data.length === 0) return null
  const chartData = data.map(row => {
    const keys = Object.keys(row)
    const name = String(row[nameKey] ?? row[keys.find(k => k !== dataKey) ?? keys[0]] ?? '')
    return { name, value: Number(row[dataKey] ?? row[keys.find(k => typeof row[k] === 'number') ?? dataKey] ?? 0) }
  })

  return (
    <div className={x.payloadChart}>
      <div className={x.subhead}>{title}</div>
      <div className={x.miniChartWrap}>
        <ResponsiveContainer>
          <BarChart data={chartData} margin={{ top: 4, right: 4, left: 4, bottom: 28 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="name"
              angle={-30}
              textAnchor="end"
              height={48}
              tick={{ fontSize: 10, fill: 'var(--text-3)' }}
              interval={0}
            />
            <YAxis tick={{ fontSize: 10, fill: 'var(--text-3)' }} width={48} />
            <Tooltip
              contentStyle={{
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius)',
                fontSize: '0.75rem',
              }}
            />
            <Bar dataKey="value" fill={color} radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function SliderChart({
  title,
  data,
}: {
  title: string
  data: DistRow[] | undefined
}) {
  if (!data || data.length === 0) return null
  const chartData = data.map(row => ({
    name: String(row.value ?? ''),
    value: Number(row.count ?? 0),
  }))

  return (
    <div className={x.payloadChart}>
      <div className={x.subhead}>{title}</div>
      <div className={x.miniChartWrap}>
        <ResponsiveContainer>
          <BarChart data={chartData} margin={{ top: 4, right: 4, left: 4, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'var(--text-3)' }} />
            <YAxis tick={{ fontSize: 10, fill: 'var(--text-3)' }} width={48} />
            <Tooltip
              contentStyle={{
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius)',
                fontSize: '0.75rem',
              }}
            />
            <Bar dataKey="value" radius={[3, 3, 0, 0]}>
              {chartData.map((_, i) => (
                <Cell key={i} fill={ACCENT_COLORS[i % ACCENT_COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function HeatmapTable({
  title,
  data,
}: {
  title: string
  data: DistRow[] | undefined
}) {
  if (!data || data.length === 0) return null

  const riskVals = [...new Set(data.map(r => Number(r.risk)))].sort((a, b) => a - b)
  const actVals = [...new Set(data.map(r => Number(r.activity)))].sort((a, b) => a - b)
  const lookup = new Map(data.map(r => [`${r.risk}-${r.activity}`, Number(r.count)]))
  const maxCount = Math.max(...data.map(r => Number(r.count)), 1)

  return (
    <div className={x.payloadChart}>
      <div className={x.subhead}>{title}</div>
      <table className={x.heatmap}>
        <thead>
          <tr>
            <th className={x.heatmapCorner}>Risk \ TA</th>
            {actVals.map(a => <th key={a} className={x.heatmapHead}>{a}</th>)}
          </tr>
        </thead>
        <tbody>
          {riskVals.map(r => (
            <tr key={r}>
              <td className={x.heatmapLabel}>{r}</td>
              {actVals.map(a => {
                const v = lookup.get(`${r}-${a}`) ?? 0
                const intensity = v / maxCount
                return (
                  <td
                    key={a}
                    className={x.heatmapCell}
                    style={{
                      background: v > 0
                        ? `oklch(${65 - intensity * 30}% ${0.08 + intensity * 0.12} 250 / ${0.15 + intensity * 0.85})`
                        : 'var(--bg-sub)',
                    }}
                  >
                    {v > 0 ? v.toLocaleString() : ''}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className={x.metricCard}>
      <div className={x.metricLabel}>{label}</div>
      <div className="mono" style={{ fontSize: '1.1rem', fontWeight: 600 }}>
        {typeof value === 'number' ? value.toLocaleString() : value}
      </div>
      {sub && <div className={x.metricLabel} style={{ marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function PayloadExplorer({
  backendFetch,
}: {
  backendFetch: <T>(path: string) => Promise<T>
}) {
  const [stats, setStats] = useState<PayloadStatsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await backendFetch<PayloadStatsResponse>('/payload-stats')
      setStats(result)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load payload stats')
    } finally {
      setLoading(false)
    }
  }, [backendFetch])

  useEffect(() => { load() }, [load])

  if (loading) return <div className={s.empty}>Loading payload distributions...</div>
  if (error) return <div className={x.errorBanner}>{error}</div>
  if (!stats || stats.total_logs === 0) {
    return <div className={s.empty}>No full logs with raw_payload found. Run ingest first.</div>
  }

  const fmtMs = (v: number | null | undefined) => v != null ? `${(v / 1000).toFixed(1)}s` : '\u2014'
  const fmtTok = (v: number | null | undefined) => v != null ? Math.round(v).toLocaleString() : '\u2014'

  return (
    <div className={x.payloadRoot}>
      <p className={s.phaseDesc}>
        Distribution analysis of <strong>{stats.total_logs.toLocaleString()}</strong> full log payloads.
        Explores the structure and diversity of agent decisions, market contexts,
        and LLM behavior across the dataset.
      </p>

      {/* Top-level stats */}
      <div className={x.payloadStats}>
        <StatCard label="Full Logs" value={stats.total_logs} />
        <StatCard label="Unique Vaults" value={stats.unique_vaults ?? '\u2014'} />
        <StatCard
          label="Avg Prompt"
          value={fmtTok(stats.token_usage?.avg_prompt)}
          sub={`p50: ${fmtTok(stats.token_usage?.p50_prompt)} / p95: ${fmtTok(stats.token_usage?.p95_prompt)}`}
        />
        <StatCard
          label="Avg Completion"
          value={fmtTok(stats.token_usage?.avg_completion)}
          sub={`p50: ${fmtTok(stats.token_usage?.p50_completion)} / p95: ${fmtTok(stats.token_usage?.p95_completion)}`}
        />
        <StatCard
          label="Avg Reasoning"
          value={fmtTok(stats.token_usage?.avg_reasoning)}
        />
        <StatCard
          label="Inference Time"
          value={fmtMs(stats.inference_duration?.avg_ms)}
          sub={`p50: ${fmtMs(stats.inference_duration?.p50_ms)} / p95: ${fmtMs(stats.inference_duration?.p95_ms)}`}
        />
      </div>

      {/* Decision distribution */}
      <div className={s.sectionLabel}>Decision Distribution</div>
      <div className={x.payloadGrid2}>
        <MiniBar title="Tool Calls" data={stats.tool_distribution} nameKey="tool" />
        <MiniBar title="LLM Models" data={stats.model_distribution} nameKey="model" color="oklch(72% 0.15 160)" />
      </div>

      {/* Agent configuration */}
      <div className={s.sectionLabel}>Agent Configuration Space</div>
      <div className={x.payloadGrid3}>
        <SliderChart title="Trading Activity" data={stats.slider_trading_activity} />
        <SliderChart title="Risk Preference" data={stats.slider_asset_risk_preference} />
        <SliderChart title="Trade Size" data={stats.slider_trade_size} />
        <SliderChart title="Holding Style" data={stats.slider_holding_style} />
        <SliderChart title="Diversification" data={stats.slider_diversification} />
        <HeatmapTable title="Risk x Activity" data={stats.risk_activity_heatmap} />
      </div>

      {/* Token universe */}
      <div className={s.sectionLabel}>Token Universe</div>
      <div className={x.payloadGrid2}>
        <MiniBar title="Most Traded" data={stats.trade_token_dist} nameKey="token" color="oklch(72% 0.16 30)" />
        <MiniBar title="ETH Balance" data={stats.eth_balance_buckets} nameKey="bucket" color="oklch(72% 0.15 280)" />
      </div>

      {/* Context richness */}
      <div className={s.sectionLabel}>Context & Portfolio</div>
      <div className={x.payloadGrid3}>
        <MiniBar title="Portfolio Tokens Held" data={stats.portfolio_token_count} nameKey="token_count" color="oklch(72% 0.15 40)" />
        <MiniBar title="Active Strategies" data={stats.strategy_count_dist} nameKey="strategy_count" color="oklch(72% 0.15 200)" />
        <MiniBar title="Memory Depth" data={stats.memory_depth} nameKey="bucket" color="oklch(65% 0.14 100)" />
      </div>
    </div>
  )
}
