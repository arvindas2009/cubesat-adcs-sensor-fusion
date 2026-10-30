// Typed WebSocket client for one ADCS session (docs/specs/webapp_protocol.md §3).
// No auto-reconnect: close codes 4001 ("replaced") and 4404 (unknown session) are surfaced to the UI.
import type { Axis, Command, ErrorMsg, Frame, Hello, ServerMessage } from './types';

export interface WsHandlers {
  onHello?: (h: Hello) => void;
  onFrame?: (f: Frame) => void;
  onError?: (e: ErrorMsg) => void;
  onClose?: (code: number) => void;
}

/** Parse one server text message; returns null for anything that is not a known message. */
export function parseServerMessage(text: string): ServerMessage | null {
  let obj: unknown;
  try {
    obj = JSON.parse(text);
  } catch {
    return null;
  }
  const type = (obj as { type?: unknown } | null)?.type;
  if (type === 'hello' || type === 'frame' || type === 'error') return obj as ServerMessage;
  return null;
}

export class SessionSocket {
  private readonly ws: WebSocket;
  private readonly handlers: WsHandlers;

  constructor(url: string, handlers: WsHandlers = {}) {
    this.handlers = handlers;
    this.ws = new WebSocket(url);
    this.ws.onmessage = (ev: MessageEvent) => {
      if (typeof ev.data !== 'string') return;
      const msg = parseServerMessage(ev.data);
      if (!msg) return;
      if (msg.type === 'hello') this.handlers.onHello?.(msg);
      else if (msg.type === 'frame') this.handlers.onFrame?.(msg);
      else this.handlers.onError?.(msg);
    };
    this.ws.onclose = (ev: CloseEvent) => this.handlers.onClose?.(ev.code);
  }

  get isOpen(): boolean {
    return this.ws.readyState === WebSocket.OPEN;
  }

  send(cmd: Command): void {
    if (this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(cmd));
  }

  play(): void {
    this.send({ cmd: 'play' });
  }
  pause(): void {
    this.send({ cmd: 'pause' });
  }
  speed(value: number): void {
    this.send({ cmd: 'speed', value });
  }
  kick(axis: Axis, deg_s: number): void {
    this.send({ cmd: 'kick', axis, deg_s });
  }
  step(n: number): void {
    this.send({ cmd: 'step', n });
  }
  reset(): void {
    this.send({ cmd: 'reset' });
  }

  /** Close without firing `onClose` (the caller initiated it). */
  close(): void {
    this.ws.onclose = null;
    this.ws.close();
  }
}
