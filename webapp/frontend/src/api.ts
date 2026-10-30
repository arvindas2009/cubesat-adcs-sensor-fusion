// REST wrappers for /api/adcs (docs/specs/webapp_protocol.md §2). DOM-free: only needs `fetch`.
import type { Presets, SessionInfoResponse, SessionParams } from './types';

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/** Turn FastAPI's error bodies (422 list or `{detail: str}`) into one readable line. */
export function describeError(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const parts = detail.map((d) => {
      const e = d as { loc?: unknown[]; msg?: string };
      const field = Array.isArray(e.loc) ? e.loc.filter((x) => x !== 'body').join('.') : '';
      return field ? `${field}: ${e.msg ?? 'invalid'}` : (e.msg ?? 'invalid');
    });
    if (parts.length) return parts.join('; ');
  }
  return `HTTP ${status}`;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      /* no JSON body */
    }
    throw new ApiError(res.status, describeError(res.status, body));
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export function getPresets(): Promise<Presets> {
  return request<Presets>('/api/adcs/presets');
}

/** Create a session. A 15-orbit precompute takes a couple of seconds; callers show "Preparing...". */
export function createSession(params: Partial<SessionParams> = {}): Promise<SessionInfoResponse> {
  return request<SessionInfoResponse>('/api/adcs/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
}

export function getSession(id: string): Promise<SessionInfoResponse> {
  return request<SessionInfoResponse>(`/api/adcs/sessions/${encodeURIComponent(id)}`);
}

export function deleteSession(id: string): Promise<void> {
  return request<void>(`/api/adcs/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

/** ws:// or wss:// URL of a session's socket, relative to the page origin. */
export function wsUrl(id: string, loc: { protocol: string; host: string }): string {
  const scheme = loc.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${scheme}//${loc.host}/api/adcs/sessions/${encodeURIComponent(id)}/ws`;
}
