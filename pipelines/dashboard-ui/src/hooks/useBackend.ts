import { useState, useEffect, useCallback } from 'react'

export function useBackendUrl() {
  const [url, setUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/backend-url')
      .then(r => r.json())
      .then(data => {
        if (data.url) setUrl(data.url)
        else setError(data.error || 'Backend not configured')
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const backendFetch = useCallback(async <T,>(path: string): Promise<T> => {
    if (!url) throw new Error('Backend not configured')
    const res = await fetch(`${url}${path}`)
    if (!res.ok) {
      const body = await res.text().catch(() => '')
      let detail = `HTTP ${res.status}`
      try { detail = JSON.parse(body).detail ?? detail } catch { /* */ }
      throw new Error(detail)
    }
    return res.json()
  }, [url])

  const backendPost = useCallback(async <T,>(path: string, body: unknown): Promise<T> => {
    if (!url) throw new Error('Backend not configured')
    const res = await fetch(`${url}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      let detail = `HTTP ${res.status}`
      try { detail = JSON.parse(text).detail ?? detail } catch { /* */ }
      throw new Error(detail)
    }
    return res.json()
  }, [url])

  const backendDelete = useCallback(async <T,>(path: string): Promise<T> => {
    if (!url) throw new Error('Backend not configured')
    const res = await fetch(`${url}${path}`, { method: 'DELETE' })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      let detail = `HTTP ${res.status}`
      try { detail = JSON.parse(text).detail ?? detail } catch { /* */ }
      throw new Error(detail)
    }
    return res.json()
  }, [url])

  return { url, loading, error, backendFetch, backendPost, backendDelete }
}
