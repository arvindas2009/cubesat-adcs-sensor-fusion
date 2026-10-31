// ADCS slot of the top bar (docs/specs/ui_console.md §2): the session-state chip, then the simulated time,
// the orbit count and the speed multiplier in mono. Updated from Panels.onHello / onFrame / onClosed; every
// number comes from server frames. Fixed-width cells so the bar never shifts while the values tick.
import { DASH, fmtFixed, fmtSpeed } from './format';

export type SessionState = 'none' | 'running' | 'paused' | 'complete';

export const SESSION_LABEL: Record<SessionState, string> = {
  none: 'no session',
  running: 'running',
  paused: 'paused',
  complete: 'run complete',
};

/** Session state from the connection and the server's playing / done flags. */
export function sessionState(connected: boolean, playing: boolean, done: boolean): SessionState {
  if (!connected) return 'none';
  if (done) return 'complete';
  return playing ? 'running' : 'paused';
}

function cell(cls: string, label: string): { box: HTMLElement; value: HTMLElement } {
  const box = document.createElement('span');
  box.className = `stat ${cls}`;
  const l = document.createElement('span');
  l.className = 'stat-label';
  l.textContent = label;
  const value = document.createElement('span');
  value.className = 'stat-value num';
  value.textContent = DASH;
  box.append(l, value);
  return { box, value };
}

export class StatusBar {
  private readonly chip = document.createElement('span');
  private readonly t = cell('stat-t', 't');
  private readonly orbit = cell('stat-orbit', 'orbit');
  private readonly speed = cell('stat-speed', '');
  private state: SessionState | null = null;
  private readonly shown = { t: '', orbit: '', speed: '' };

  constructor(private readonly root: HTMLElement) {
    this.chip.className = 'chip session-chip';
    this.speed.box.title = 'Actual speed, in times real time';
    root.append(this.chip, this.t.box, this.orbit.box, this.speed.box);
    this.setSession(false, false, false);
    this.clearValues();
  }

  setSession(connected: boolean, playing: boolean, done: boolean): void {
    const s = sessionState(connected, playing, done);
    if (s === this.state) return;
    this.state = s;
    this.chip.textContent = SESSION_LABEL[s];
    this.chip.dataset.state = s;
    this.root.dataset.state = s;
  }

  /** Simulated time [s], orbits completed and the actual speed multiplier (null = not measured yet). */
  setFrame(t: number, orbit: number, speedActual: number | null, playing: boolean): void {
    this.put('t', this.t.value, `${fmtFixed(t, 0)} s`);
    this.put('orbit', this.orbit.value, fmtFixed(orbit, 3));
    this.put('speed', this.speed.value, playing && speedActual !== null ? fmtSpeed(speedActual) : DASH);
  }

  clearValues(): void {
    this.put('t', this.t.value, DASH);
    this.put('orbit', this.orbit.value, DASH);
    this.put('speed', this.speed.value, DASH);
  }

  private put(key: 't' | 'orbit' | 'speed', e: HTMLElement, text: string): void {
    if (this.shown[key] === text) return;
    this.shown[key] = text;
    e.textContent = text;
  }
}
