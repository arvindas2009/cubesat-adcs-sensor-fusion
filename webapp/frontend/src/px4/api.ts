// REST wrappers for /api/px4 (docs/specs/px4_replay.md §2). DOM-free: only needs `fetch`. Errors reuse
// ApiError / describeError from ../api, so `detail` strings reach the banner unchanged.
import { request } from '../api';
import type { LogsResponse, Replay } from './types';

export function getLogs(): Promise<LogsResponse> {
  return request<LogsResponse>('/api/px4/logs');
}

/**
 * Run (or fetch from the server cache) the replay of one log: the whole run as a single JSON document.
 * A first build of a log takes 10-35 s; the caller shows a "Running the filter..." message meanwhile.
 */
export function loadReplay(log: string): Promise<Replay> {
  return request<Replay>(`/api/px4/replay/${encodeURIComponent(log)}`, { method: 'POST' });
}
