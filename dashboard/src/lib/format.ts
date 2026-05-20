/** Utility formatters used across panels. */

export function formatRelative(iso: string | null | undefined): string {
  if (!iso) return "—";
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return iso;
  const diff = Math.max(0, Date.now() - then) / 1000;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86_400) return `${Math.floor(diff / 3600)}h ago`;
  const days = Math.floor(diff / 86_400);
  if (days < 14) return `${days}d ago`;
  return new Date(then).toISOString().slice(0, 10);
}

export function formatDuration(startIso: string | null, endIso: string | null): string {
  if (!startIso || !endIso) return "—";
  const start = Date.parse(startIso);
  const end = Date.parse(endIso);
  if (Number.isNaN(start) || Number.isNaN(end)) return "—";
  const secs = Math.max(0, end - start) / 1000;
  if (secs < 60) return `${secs.toFixed(1)}s`;
  if (secs < 3600) {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}m ${s}s`;
  }
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  return `${h}h ${m}m`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)}GB`;
}

export function shortHash(hash: string | null | undefined, len = 10): string {
  if (!hash) return "—";
  if (hash.length <= len) return hash;
  return hash.slice(0, len);
}

export function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return `${s.slice(0, n - 1)}…`;
}
