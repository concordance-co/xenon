import { useState, useEffect, useCallback } from 'react'
import { useBackendUrl } from '../hooks/useBackend'
import type {
  BackendTableInfo, BackendSchemaResponse,
  BackendSampleResponse, BackendQueryResponse,
} from '../types/api'
import s from './shared.module.css'
import x from './ExplorerView.module.css'

export function ExplorerView() {
  const { url, loading: urlLoading, error: urlError, backendFetch, backendPost } = useBackendUrl()

  const [tables, setTables] = useState<BackendTableInfo[]>([])
  const [selectedTable, setSelectedTable] = useState('')
  const [schema, setSchema] = useState<BackendSchemaResponse | null>(null)
  const [sample, setSample] = useState<BackendSampleResponse | null>(null)
  const [sampleSize, setSampleSize] = useState(20)
  const [tableLoading, setTableLoading] = useState(false)

  const [sql, setSql] = useState('')
  const [queryResult, setQueryResult] = useState<BackendQueryResponse | null>(null)
  const [queryError, setQueryError] = useState<string | null>(null)
  const [queryLoading, setQueryLoading] = useState(false)

  // Load tables when URL is ready
  useEffect(() => {
    if (!url) return
    setTableLoading(true)
    backendFetch<{ tables: BackendTableInfo[] }>('/tables')
      .then(d => setTables(d.tables))
      .catch(() => {})
      .finally(() => setTableLoading(false))
  }, [url, backendFetch])

  // Load schema + sample when table changes
  useEffect(() => {
    if (!selectedTable || !url) {
      setSchema(null)
      setSample(null)
      return
    }
    backendFetch<BackendSchemaResponse>(`/schema?table=${selectedTable}`).then(setSchema)
    backendFetch<BackendSampleResponse>(`/sample/${selectedTable}?n=${sampleSize}`).then(setSample)
  }, [selectedTable, url, backendFetch]) // eslint-disable-line react-hooks/exhaustive-deps

  const resample = useCallback(() => {
    if (!selectedTable || !url) return
    backendFetch<BackendSampleResponse>(`/sample/${selectedTable}?n=${sampleSize}`).then(setSample)
  }, [selectedTable, sampleSize, url, backendFetch])

  const handleSelectTable = useCallback((name: string) => {
    setSelectedTable(name)
    setSql(`SELECT * FROM ${name}`)
    setQueryResult(null)
    setQueryError(null)
  }, [])

  const runQuery = useCallback(async () => {
    if (!sql.trim()) return
    setQueryLoading(true)
    setQueryError(null)
    try {
      const result = await backendPost<BackendQueryResponse>('/query', { sql, limit: 100 })
      setQueryResult(result)
    } catch (e) {
      setQueryError(e instanceof Error ? e.message : 'Query failed')
      setQueryResult(null)
    } finally {
      setQueryLoading(false)
    }
  }, [sql, backendPost])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault()
      runQuery()
    }
  }, [runQuery])

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
    <div>
      {/* Table picker */}
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
        {tableLoading && <span className={x.rowCount}>Loading...</span>}
      </div>

      {/* Schema + Sample */}
      {selectedTable && (
        <div className={s.grid2}>
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
                <div className={s.empty}>Loading...</div>
              )}
            </div>
          </div>

          <div className={s.panel}>
            <div className={s.panelHead}>
              <span className={s.panelTitle}>Sample</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className={x.rowCount}>{sample?.row_count ?? 0} rows</span>
                <input
                  type="number"
                  className={s.fieldInput}
                  value={sampleSize}
                  min={1}
                  max={500}
                  style={{ width: 60 }}
                  onChange={e => setSampleSize(Number(e.target.value) || 20)}
                />
                <button className={s.btn} onClick={resample}>Resample</button>
              </div>
            </div>
            <div className={s.panelBody}>
              {sample ? (
                <DataGrid columns={sample.columns} rows={sample.rows} />
              ) : (
                <div className={s.empty}>Loading...</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Query editor */}
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
            placeholder="SELECT * FROM vaults LIMIT 10"
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
            {queryResult && (
              <span className={x.rowCount}>
                {queryResult.row_count} rows returned
              </span>
            )}
          </div>

          {queryError && <div className={x.errorBanner}>{queryError}</div>}

          {queryResult && queryResult.rows.length > 0 && (
            <div style={{ marginTop: 'var(--space-lg)' }}>
              <DataGrid columns={queryResult.columns} rows={queryResult.rows} />
            </div>
          )}
        </div>
      </div>
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
