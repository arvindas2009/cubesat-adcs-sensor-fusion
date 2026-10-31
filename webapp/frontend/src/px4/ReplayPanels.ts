// Side panels, timeline and plot strips of the PX4 replay page (docs/specs/px4_replay.md §3.3).
// Browser-only: the non-trivial logic (series, clock, readout values) is pure and unit-tested in
// series.ts / clock.ts. Every number shown comes from the replay document.
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import { DASH, fmtFixed } from '../adcs/format';
import { clampLog, logRange } from '../adcs/logic';
import { chromeColors, dataColors, nisFallback, onThemeChange, currentTheme, whenFontsReady, type DataColors } from '../theme';
import { SPEEDS, DEFAULT_SPEED, indexAt, xToTime } from './clock';
import type { PlaybackClock } from './clock';
import { alignColumns, diffColumns, heightColumns, nextReset, nisBands, nisSeries, prevReset, readoutsAt } from './series';
import type { NisBand, ReadoutValues } from './series';
import type { LogEntry, LogsResponse, Replay } from './types';

export interface ReplayHandlers {
  onLoad(role: string): void;
  onTogglePlay(): void;
  onSpeed(v: number): void;
  onSeek(t: number): void;
  onPrevReset(): void;
  onNextReset(): void;
}

export interface ReplayPanelRoots {
  log: HTMLElement;
  transport: HTMLElement;
  readouts: HTMLElement;
  timeline: HTMLElement;
  plots: [HTMLElement, HTMLElement, HTMLElement, HTMLElement];
}

export const LOADING_TEXT = 'Running the filter… (first load of a log takes 10–35 s; cached afterwards)';
const FONT_FAMILY = '"Schibsted Grotesk Variable", system-ui, sans-serif';
/** Event-marker role (magenta) of the active mode, shared by the timeline and the plots. */
const resetColor = (): string => dataColors().kick;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function body(panel: HTMLElement): HTMLElement {
  return panel.querySelector<HTMLElement>('.panel-body') ?? panel;
}

function niceStep(x: number): number {
  const e = 10 ** Math.floor(Math.log10(x));
  for (const m of [1, 2, 5, 10]) if (m * e >= x) return m * e;
  return 10 * e;
}

// ===================================================================== log picker

type Chip = { text: string; cls: string };

function chipFor(l: LogEntry): Chip {
  if (!l.present) return { text: 'not downloaded', cls: 'missing' };
  return l.cached ? { text: 'cached', cls: 'cached' } : { text: 'ready', cls: 'ready' };
}

class LogPanel {
  private readonly list = el('div', 'log-list');
  private readonly hint = el('div', 'log-hint');
  private readonly status = el('div', 'log-status');
  private resp: LogsResponse | null = null;
  private busy: string | null = null;
  private active: string | null = null;

  constructor(
    root: HTMLElement,
    private readonly handlers: ReplayHandlers,
  ) {
    body(root).append(this.list, this.hint, this.status);
    this.status.textContent = 'Loading the log list…';
  }

  setLogs(resp: LogsResponse): void {
    this.resp = resp;
    if (!this.busy) this.status.textContent = '';
    this.status.classList.remove('bad');
    this.render();
  }

  setFailed(message: string): void {
    this.status.textContent = message;
    this.status.classList.add('bad');
  }

  /** `role` while its POST is pending (all Load buttons disabled), null when idle. */
  setBusy(role: string | null): void {
    this.busy = role;
    this.status.classList.remove('bad');
    this.status.textContent = role ? LOADING_TEXT : '';
    this.render();
  }

  setActive(role: string | null): void {
    this.active = role;
    this.render();
  }

  private render(): void {
    const r = this.resp;
    this.list.replaceChildren();
    this.hint.replaceChildren();
    if (!r) return;
    let anyMissing = false;
    for (const l of r.logs) {
      anyMissing ||= !l.present;
      const row = el('div', 'log-row');
      row.classList.toggle('active', this.active === l.role);
      row.title = l.notes;
      const chip = chipFor(l);
      const btn = el('button', this.busy === l.role ? 'busy' : undefined, this.busy === l.role ? 'Running…' : this.active === l.role ? 'Reload' : 'Load');
      btn.type = 'button';
      btn.disabled = !l.present || this.busy !== null;
      btn.addEventListener('click', () => this.handlers.onLoad(l.role));
      row.append(el('span', 'log-role', l.role), el('span', 'log-id num', l.id.slice(0, 8)), el('span', `chip ${chip.cls}`, chip.text), btn);
      this.list.append(row);
    }
    if (anyMissing) {
      this.hint.append(document.createTextNode('Missing logs: run '));
      const code = el('code', 'copyable', r.download_hint);
      code.title = 'Click to copy';
      code.addEventListener('click', () => {
        navigator.clipboard?.writeText(r.download_hint).then(
          () => (code.title = 'Copied'),
          () => undefined,
        );
      });
      this.hint.append(code);
    }
  }
}

// ===================================================================== transport

class TransportPanel {
  private readonly playBtn = el('button', 'primary wide', 'Play');
  private readonly speedSel = el('select', 'speed-select');
  private readonly prevBtn = el('button', undefined, '◀ reset');
  private readonly nextBtn = el('button', undefined, 'reset ▶');
  private readonly slider = el('input', 'scrub-slider');
  private readonly timeText = el('span', 'num scrub-time', DASH);
  private t0 = 0;
  private t1 = 1;
  private lastText = '';
  private enabled = false;
  private scrubbing = false;

  constructor(
    root: HTMLElement,
    private readonly handlers: ReplayHandlers,
  ) {
    for (const s of SPEEDS) this.speedSel.append(new Option(`${s}×`, String(s)));
    this.speedSel.value = String(DEFAULT_SPEED);
    this.speedSel.setAttribute('aria-label', 'Playback speed');
    this.playBtn.type = 'button';
    this.prevBtn.type = 'button';
    this.nextBtn.type = 'button';
    this.prevBtn.title = 'Seek to the previous EKF2 reset';
    this.nextBtn.title = 'Seek to the next EKF2 reset';
    const r1 = el('div', 'row');
    r1.append(this.playBtn, el('span', 'field-label', 'Speed'), this.speedSel, this.prevBtn, this.nextBtn);

    this.slider.type = 'range';
    this.slider.min = '0';
    this.slider.max = '1';
    this.slider.step = 'any';
    this.slider.value = '0';
    this.slider.setAttribute('aria-label', 'Scrub through the log');
    const r2 = el('div', 'row scrub-row');
    r2.append(this.slider, this.timeText);
    body(root).append(r1, r2);

    this.playBtn.addEventListener('click', () => this.handlers.onTogglePlay());
    this.speedSel.addEventListener('change', () => this.handlers.onSpeed(Number(this.speedSel.value)));
    this.prevBtn.addEventListener('click', () => this.handlers.onPrevReset());
    this.nextBtn.addEventListener('click', () => this.handlers.onNextReset());
    this.slider.addEventListener('input', () => this.handlers.onSeek(Number(this.slider.value)));
    this.slider.addEventListener('pointerdown', () => (this.scrubbing = true));
    const release = (): void => {
      this.scrubbing = false;
    };
    window.addEventListener('pointerup', release);
    window.addEventListener('pointercancel', release);
    this.refreshEnabled();
  }

  setReplay(rep: Replay | null, clock: PlaybackClock | null): void {
    this.enabled = rep !== null && clock !== null;
    if (rep && clock) {
      this.t0 = clock.t0;
      this.t1 = clock.t1;
      this.slider.min = String(clock.t0);
      this.slider.max = String(clock.t1);
      this.speedSel.value = String(clock.speed);
    }
    this.lastText = '';
    this.refreshEnabled();
    if (clock) this.update(clock, rep);
    else this.timeText.textContent = DASH;
  }

  private refreshEnabled(): void {
    this.playBtn.disabled = !this.enabled;
    this.speedSel.disabled = !this.enabled;
    this.slider.disabled = !this.enabled;
    if (!this.enabled) {
      this.prevBtn.disabled = true;
      this.nextBtn.disabled = true;
      this.playBtn.textContent = 'Play';
    }
  }

  update(clock: PlaybackClock, rep: Replay | null): void {
    this.playBtn.textContent = clock.playing ? 'Pause' : 'Play';
    if (!this.scrubbing) this.slider.value = String(clock.t);
    const elapsed = clock.t - this.t0;
    const text = `${elapsed.toFixed(2).padStart(8, ' ')} / ${(this.t1 - this.t0).toFixed(2)} s`;
    if (text !== this.lastText) {
      this.lastText = text;
      this.timeText.textContent = text;
    }
    if (rep) {
      this.prevBtn.disabled = prevReset(rep, clock.t) === null;
      this.nextBtn.disabled = nextReset(rep, clock.t) === null;
    }
  }
}

// ===================================================================== readouts

class Readouts {
  private readonly vals = new Map<string, HTMLElement>();
  private readonly cache = new Map<string, string>();

  constructor(root: HTMLElement) {
    const b = body(root);
    const table = el('div', 'ro-table');
    table.append(el('span'), el('span', 'ro-head ours', 'ours'), el('span', 'ro-head ekf2', 'EKF2 (reference)'));
    const addPair = (key: string, label: string): void => {
      const a = el('span', 'readout-value', DASH);
      const c = el('span', 'readout-value', DASH);
      table.append(el('span', 'readout-label', label), a, c);
      this.vals.set(`${key}.o`, a);
      this.vals.set(`${key}.e`, c);
    };
    addPair('roll', 'roll [°]');
    addPair('pitch', 'pitch [°]');
    addPair('yaw', 'yaw [°]');
    addPair('height', 'height above start [m]');
    const grid = el('div', 'readout-grid');
    const add = (key: string, label: string, title?: string): void => {
      const lab = el('span', 'readout-label', label);
      if (title) lab.title = title;
      const val = el('span', 'readout-value', DASH);
      const cell = el('div', 'ro');
      cell.append(lab, val);
      grid.append(cell);
      this.vals.set(key, val);
    };
    add('t', 't [s]', 'Log time of the current sample');
    add('el', 'elapsed [s]');
    add('frac', 'log position');
    add('s3', '3σ attitude (max) [°]', 'Largest of the three attitude-error 3-sigma values of our filter');
    add('tilt', 'tilt diff [°]', 'ours − EKF2');
    add('yawd', 'yaw diff [°]', 'ours − EKF2, signed');
    add('posh', 'horiz. pos diff [m]', 'ours − EKF2');
    add('posd', 'vert. pos diff [m]', 'ours − EKF2, positive = ours lower');
    b.append(table, grid);
  }

  private set(key: string, text: string): void {
    if (this.cache.get(key) === text) return;
    this.cache.set(key, text);
    const e = this.vals.get(key);
    if (e) e.textContent = text;
  }

  update(v: ReadoutValues, t0: number): void {
    const names = ['roll', 'pitch', 'yaw'];
    names.forEach((n, k) => {
      this.set(`${n}.o`, fmtFixed(v.oursEuler[k], 2, 8));
      this.set(`${n}.e`, fmtFixed(v.ekf2Euler[k], 2, 8));
    });
    this.set('height.o', fmtFixed(v.heightOurs, 2, 8));
    this.set('height.e', fmtFixed(v.heightEkf2, 2, 8));
    this.set('t', fmtFixed(v.t, 2));
    this.set('el', fmtFixed(v.t - t0, 2));
    this.set('frac', `${(100 * v.frac).toFixed(1)} %`);
    this.set('s3', fmtFixed(v.sigma3, 3));
    this.set('tilt', fmtFixed(v.tilt, 3));
    this.set('yawd', fmtFixed(v.yaw, 3, 0, true));
    this.set('posh', fmtFixed(v.posH, 3));
    this.set('posd', fmtFixed(v.posD, 3, 0, true));
  }

  clear(): void {
    this.cache.clear();
    for (const v of this.vals.values()) v.textContent = DASH;
  }
}

// ===================================================================== timeline

class TimelineBar {
  private readonly canvas = el('canvas', 'timeline-canvas');
  private readonly ctx: CanvasRenderingContext2D | null;
  private rep: Replay | null = null;
  private t = 0;
  private w = 0;
  private h = 0;
  private dpr = 1;
  private dragging = false;

  constructor(
    root: HTMLElement,
    private readonly onSeek: (t: number) => void,
  ) {
    root.append(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.canvas.classList.add('seekable');
    new ResizeObserver(() => {
      this.dpr = window.devicePixelRatio || 1;
      this.w = Math.max(1, Math.floor(root.clientWidth));
      this.h = Math.max(1, Math.floor(root.clientHeight));
      this.canvas.width = Math.round(this.w * this.dpr);
      this.canvas.height = Math.round(this.h * this.dpr);
      this.draw();
    }).observe(root);

    const seek = (e: PointerEvent): void => {
      const r = this.rep;
      if (!r) return;
      const rect = this.canvas.getBoundingClientRect();
      this.onSeek(xToTime(e.clientX - rect.left, this.x0, this.x1, r.t[0], r.t[r.t.length - 1]));
    };
    this.canvas.addEventListener('pointerdown', (e) => {
      this.dragging = true;
      this.canvas.setPointerCapture(e.pointerId);
      seek(e);
    });
    this.canvas.addEventListener('pointermove', (e) => {
      if (this.dragging) seek(e);
    });
    const end = (): void => {
      this.dragging = false;
    };
    this.canvas.addEventListener('pointerup', end);
    this.canvas.addEventListener('pointercancel', end);
  }

  private get x0(): number {
    return 12;
  }
  private get x1(): number {
    return this.w - 12;
  }

  setReplay(rep: Replay | null): void {
    this.rep = rep;
    this.t = rep ? rep.t[0] : 0;
    this.draw();
  }

  setTime(t: number): void {
    if (t === this.t) return;
    this.t = t;
    this.draw();
  }

  draw(): void {
    const g = this.ctx;
    if (!g || this.w < 2) return;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.clearRect(0, 0, this.w, this.h);
    const c = chromeColors(); // the active mode's card / band / marker colours
    const d = dataColors();
    g.font = `12px ${FONT_FAMILY}`;
    g.textBaseline = 'middle';
    const { x0, x1 } = this;
    const bandTop = 18;
    const bandBot = Math.max(bandTop + 20, this.h - 16);

    // legend row
    let lx = x0;
    const swatch = (color: string, label: string): void => {
      g.fillStyle = color;
      g.fillRect(lx, 4, 9, 9);
      g.fillStyle = c.dim;
      g.textAlign = 'left';
      g.fillText(label, lx + 13, 9);
      lx += 13 + g.measureText(label).width + 14;
    };
    swatch(c.excluded, 'excluded from report statistics');
    swatch(d.kick, 'EKF2 reset');
    swatch(c.cursor, 'cursor');

    g.fillStyle = c.band;
    g.fillRect(x0, bandTop, x1 - x0, bandBot - bandTop);
    g.strokeStyle = c.bandEdge;
    g.lineWidth = 1;
    g.strokeRect(x0 + 0.5, bandTop + 0.5, x1 - x0 - 1, bandBot - bandTop - 1);
    const rep = this.rep;
    if (!rep || rep.t.length < 2) return;
    const t0 = rep.t[0];
    const dur = rep.t[rep.t.length - 1] - t0;
    const px = (t: number): number => x0 + (Math.min(1, Math.max(0, (t - t0) / dur)) * (x1 - x0));

    // the part the reports exclude (the first skip_init_s)
    const xs = px(t0 + rep.events.skip_init_s);
    g.fillStyle = c.excluded;
    g.fillRect(x0 + 1, bandTop + 1, Math.max(0, xs - x0 - 1), bandBot - bandTop - 2);
    g.fillStyle = c.dim;
    g.textAlign = 'left';
    if (xs - x0 > g.measureText('excluded from report statistics').width + 8) {
      g.fillText('excluded from report statistics', x0 + 5, (bandTop + bandBot) / 2);
    }

    // time ticks (elapsed seconds)
    const step = niceStep(dur / 10);
    g.strokeStyle = c.tick;
    g.fillStyle = c.dim;
    g.textAlign = 'center';
    for (let s = 0; s <= dur + 1e-9; s += step) {
      const x = Math.round(px(t0 + s)) + 0.5;
      g.beginPath();
      g.moveTo(x, bandBot);
      g.lineTo(x, bandBot + 4);
      g.stroke();
      g.fillText(String(+s.toFixed(3)), x, bandBot + 10);
    }
    g.textAlign = 'left';
    g.fillText('s', x1 + 2, bandBot + 10);

    // EKF2 resets
    g.strokeStyle = d.kick;
    g.lineWidth = 1.5;
    for (const r of rep.events.ekf2_resets_s) {
      const x = Math.round(px(r)) + 0.5;
      g.beginPath();
      g.moveTo(x, bandTop + 1);
      g.lineTo(x, bandBot - 1);
      g.stroke();
    }

    // cursor
    const cx = Math.round(px(this.t)) + 0.5;
    g.strokeStyle = c.cursor;
    g.lineWidth = 1.5;
    g.beginPath();
    g.moveTo(cx, bandTop - 3);
    g.lineTo(cx, bandBot + 2);
    g.stroke();
    g.lineWidth = 1;
  }
}

// ===================================================================== plots

interface PlotModel {
  data: uPlot.AlignedData;
  series: uPlot.Series[];
  log: boolean;
  bands: NisBand[];
  /** Indices (into `series`, 0-based after x) of legend rows to hide. */
  hideLegend: number[];
}

type ModelBuilder = (rep: Replay) => PlotModel;

function tickLabel(v: number): string {
  const a = Math.abs(v);
  if (a >= 1000) return `${+(v / 1000).toFixed(1)}k`;
  if (a > 0 && a < 0.01) return v.toExponential(0);
  return String(+v.toFixed(2));
}

/** NIS sensor name -> colour role (the server may name sensors the table does not know: those cycle a fallback list). */
const NIS_ROLES: Record<string, keyof DataColors> = {
  gps_pos: 'nisGpsPos',
  gps_vel: 'nisGpsVel',
  baro: 'nisBaro',
  mag_heading: 'nisMag',
};
const NIS_FLOOR = 1e-3;

function nisColor(sensor: string, index: number): string {
  const role = NIS_ROLES[sensor];
  if (role) return dataColors()[role];
  const fb = nisFallback(currentTheme());
  return fb[index % fb.length];
}

function lineSeries(label: string, color: string, dash?: number[], fmt: (v: number) => string = (v) => fmtFixed(v, 3)): uPlot.Series {
  return {
    label,
    stroke: color,
    width: 2,
    dash,
    spanGaps: false,
    points: { show: false },
    value: (_u, v) => (v === null || v === undefined ? DASH : fmt(v)),
  };
}

function pointSeries(label: string, color: string, hollow: boolean): uPlot.Series {
  return {
    label,
    paths: () => null, // markers only, no connecting line
    points: hollow
      ? { show: true, size: 4, width: 1, stroke: `${color}88`, fill: 'transparent' }
      : { show: true, size: 3, width: 0, stroke: color, fill: color },
    value: (_u, v) => (v === null || v === undefined ? DASH : fmtFixed(v, 2)),
  };
}

const PLOT_HEADS: { title: string; sub?: string }[] = [
  { title: 'Attitude difference, ours − EKF2 [°]' },
  { title: 'Position difference, ours − EKF2 [m]' },
  { title: 'NIS by sensor', sub: 'log scale; hollow = rejected; dashed = 95 % band by dof' },
  { title: 'Height above start [m]' },
];

const MODELS: ModelBuilder[] = [
  (rep) => {
    const { att } = diffColumns(rep);
    const d = dataColors();
    return {
      data: att as unknown as uPlot.AlignedData,
      series: [lineSeries('tilt', d.nadir), lineSeries('yaw (signed)', d.field)],
      log: false,
      bands: [],
      hideLegend: [],
    };
  },
  (rep) => {
    const { pos } = diffColumns(rep);
    const d = dataColors();
    return {
      data: pos as unknown as uPlot.AlignedData,
      series: [lineSeries('horizontal', d.nadir), lineSeries('vertical (+ = ours lower)', d.field)],
      log: false,
      bands: [],
      hideLegend: [],
    };
  },
  (rep) => {
    const sensors = nisSeries(rep);
    const parts = sensors.flatMap((s) => [s.accepted, s.rejected]);
    const cols = alignColumns(parts);
    const data = [cols[0], ...cols.slice(1).map((c) => c.map((v) => clampLog(v, NIS_FLOOR)))];
    const series: uPlot.Series[] = [];
    const hide: number[] = [];
    sensors.forEach((s, i) => {
      const color = nisColor(s.sensor, i);
      series.push(pointSeries(s.sensor, color, false), pointSeries(`${s.sensor} (rejected)`, color, true));
      hide.push(series.length - 1); // legend: one entry per sensor
    });
    return { data: data as unknown as uPlot.AlignedData, series, log: true, bands: nisBands(rep), hideLegend: hide };
  },
  (rep) => {
    const d = dataColors();
    return {
      data: heightColumns(rep) as unknown as uPlot.AlignedData,
      series: [lineSeries('ours', d.ours), lineSeries('EKF2 (reference)', d.ekf2, [6, 4])],
      log: false,
      bands: [],
      hideLegend: [],
    };
  },
];

class ReplayPlot {
  private readonly host = el('div', 'plot-host');
  private u: uPlot | null = null;
  private nowLine: HTMLElement | null = null;
  private lastX = NaN;
  private lastT = 0;
  private bands: NisBand[] = [];
  private rep: Replay | null = null;

  constructor(
    root: HTMLElement,
    head: { title: string; sub?: string },
    private readonly model: ModelBuilder,
    private readonly onSeek: (t: number) => void,
  ) {
    const h = el('div', 'card-head');
    h.append(el('h2', undefined, head.title));
    if (head.sub) h.append(el('span', 'card-sub', head.sub));
    root.append(h, this.host);
    new ResizeObserver(() => this.resize()).observe(this.host);
  }

  /** Mode change: rebuild from the loaded replay with the new palette, keeping the "now" line where it was. */
  rebuild(): void {
    if (!this.rep) return;
    this.setReplay(this.rep);
    this.setTime(this.lastT);
  }

  redraw(): void {
    this.u?.redraw();
  }

  private size(): { width: number; height: number } {
    return { width: Math.max(100, Math.floor(this.host.clientWidth)), height: Math.max(80, Math.floor(this.host.clientHeight) - 22) };
  }

  /** (Re)build the plot for a replay; null just destroys the old one. */
  setReplay(rep: Replay | null): void {
    this.u?.destroy();
    this.u = null;
    this.nowLine = null;
    this.lastX = NaN;
    this.rep = rep;
    if (!rep || rep.t.length < 2) return;
    const m = this.model(rep);
    this.bands = m.bands;
    const t0 = rep.t[0];
    const t1 = rep.t[rep.t.length - 1];
    const c = chromeColors();
    const resetStroke = resetColor();
    // axis text dim, grid a faint tint of the card's block colour, ticks hidden (spec §3)
    const axisBase = { stroke: c.dim, grid: { stroke: c.grid, width: 1 }, ticks: { show: false }, font: `12px ${FONT_FAMILY}` };
    const resets = rep.events.ekf2_resets_s;
    const opts: uPlot.Options = {
      ...this.size(),
      padding: [8, 14, 0, 0],
      legend: { show: true, live: true },
      cursor: { sync: { key: 'px4', setSeries: false }, drag: { x: false, y: false }, points: { show: false } },
      scales: {
        x: { time: false, range: () => [t0, t1] },
        y: m.log
          ? { distr: 3, log: 10, range: (_u, min, max) => logRange(min, max, NIS_FLOOR, this.bands.flatMap((b) => [b.lo, b.hi])) }
          : {},
      },
      axes: [
        { ...axisBase, size: 24, values: (_u, splits) => splits.map(tickLabel) },
        { ...axisBase, size: 54, values: (_u, splits) => splits.map((v) => (v === null ? '' : tickLabel(v))) },
      ],
      series: [{ label: 't [s]', value: (_u, v) => (v === null || v === undefined ? DASH : `${v.toFixed(2)} s`) }, ...m.series],
      hooks: {
        draw: [
          (u) => {
            const ctx = u.ctx;
            const dpr = window.devicePixelRatio || 1;
            ctx.save();
            ctx.beginPath();
            ctx.rect(u.bbox.left, u.bbox.top, u.bbox.width, u.bbox.height);
            ctx.clip();
            // EKF2 resets: thin vertical lines
            ctx.strokeStyle = `${resetStroke}aa`;
            ctx.lineWidth = dpr;
            for (const r of resets) {
              const x = Math.round(u.valToPos(r, 'x', true)) + 0.5;
              ctx.beginPath();
              ctx.moveTo(x, u.bbox.top);
              ctx.lineTo(x, u.bbox.top + u.bbox.height);
              ctx.stroke();
            }
            // chi-square 95 % bands (NIS plot)
            ctx.setLineDash([6 * dpr, 4 * dpr]);
            ctx.font = `${11 * dpr}px ${FONT_FAMILY}`;
            ctx.strokeStyle = c.dim;
            for (const b of this.bands) {
              for (const v of [b.lo, b.hi]) {
                const y = u.valToPos(Math.max(v, NIS_FLOOR), 'y', true);
                ctx.beginPath();
                ctx.moveTo(u.bbox.left, y);
                ctx.lineTo(u.bbox.left + u.bbox.width, y);
                ctx.stroke();
              }
            }
            // Labels sit at the right end of their line, on a card-coloured backing so data under them stays
            // readable. Bands whose upper lines are close on the log axis are stacked downward, never overlapping.
            ctx.setLineDash([]);
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            const xr = u.bbox.left + u.bbox.width - 6 * dpr;
            const rowH = 14 * dpr;
            let prevY = -Infinity;
            for (const b of [...this.bands].sort((p, q) => q.hi - p.hi)) {
              let y = u.valToPos(Math.max(b.hi, NIS_FLOOR), 'y', true) - 8 * dpr;
              y = Math.max(y, u.bbox.top + rowH / 2, prevY + rowH);
              prevY = y;
              const text = `dof ${b.dof}`;
              const w = ctx.measureText(text).width;
              ctx.globalAlpha = 0.85;
              ctx.fillStyle = c.card;
              ctx.fillRect(xr - w - 4 * dpr, y - rowH / 2, w + 8 * dpr, rowH);
              ctx.globalAlpha = 1;
              ctx.fillStyle = c.dim;
              ctx.fillText(text, xr, y);
            }
            ctx.restore();
          },
        ],
        ready: [
          (u) => {
            const rows = u.root.querySelectorAll<HTMLElement>('.u-legend .u-series');
            for (const i of m.hideLegend) if (rows[i + 1]) rows[i + 1].style.display = 'none';
            // the moving "now" line, a cheap DOM overlay (no redraw of the full plot per frame)
            const line = el('div', 'plot-now');
            u.over.append(line);
            this.nowLine = line;
            // click or drag to seek
            let down = false;
            const seek = (e: PointerEvent): void => {
              const x = e.clientX - u.over.getBoundingClientRect().left;
              this.onSeek(Math.min(t1, Math.max(t0, u.posToVal(x, 'x'))));
            };
            u.over.addEventListener('pointerdown', (e) => {
              down = true;
              u.over.setPointerCapture(e.pointerId);
              seek(e);
            });
            u.over.addEventListener('pointermove', (e) => {
              if (down) seek(e);
            });
            const end = (): void => {
              down = false;
            };
            u.over.addEventListener('pointerup', end);
            u.over.addEventListener('pointercancel', end);
          },
        ],
      },
    };
    this.u = new uPlot(opts, m.data, this.host);
  }

  setTime(t: number): void {
    this.lastT = t;
    const u = this.u;
    const line = this.nowLine;
    if (!u || !line) return;
    const x = Math.round(u.valToPos(t, 'x'));
    if (x === this.lastX) return;
    this.lastX = x;
    line.style.transform = `translateX(${x}px)`;
    line.style.display = Number.isFinite(x) ? 'block' : 'none';
  }

  private resize(): void {
    if (!this.u) return;
    const { width, height } = this.size();
    if (width > 20 && height > 40) {
      this.u.setSize({ width, height });
      this.lastX = NaN;
      this.setTime(this.lastT);
    }
  }
}

// ===================================================================== composite

export class ReplayPanels {
  private readonly log: LogPanel;
  private readonly transport: TransportPanel;
  private readonly readouts: Readouts;
  private readonly timeline: TimelineBar;
  private readonly plots: ReplayPlot[];

  constructor(roots: ReplayPanelRoots, handlers: ReplayHandlers) {
    this.log = new LogPanel(roots.log, handlers);
    this.transport = new TransportPanel(roots.transport, handlers);
    this.readouts = new Readouts(roots.readouts);
    this.timeline = new TimelineBar(roots.timeline, handlers.onSeek);
    this.plots = roots.plots.map((r, i) => new ReplayPlot(r, PLOT_HEADS[i], MODELS[i], handlers.onSeek));
    // Mode change: plots are rebuilt from the loaded replay with the new palette, the timeline is redrawn.
    onThemeChange(() => {
      for (const p of this.plots) p.rebuild();
      this.timeline.draw();
    });
    // Canvas text drawn before the web font arrived used the fallback font.
    whenFontsReady(() => {
      for (const p of this.plots) p.redraw();
      this.timeline.draw();
    });
  }

  setLogs(resp: LogsResponse): void {
    this.log.setLogs(resp);
  }
  setLogsFailed(message: string): void {
    this.log.setFailed(message);
  }
  setBusy(role: string | null): void {
    this.log.setBusy(role);
  }
  setActiveLog(role: string | null): void {
    this.log.setActive(role);
  }

  /** A new replay (or null): rebuild the plots, reset the timeline, transport and readouts. */
  setReplay(rep: Replay | null, clock: PlaybackClock | null): void {
    this.transport.setReplay(rep, clock);
    this.timeline.setReplay(rep);
    this.readouts.clear();
    for (const p of this.plots) p.setReplay(rep);
    if (rep && clock) this.onTime(rep, clock);
  }

  /** Per animation frame (cheap: cached text, one line per plot). */
  onTime(rep: Replay, clock: PlaybackClock): number {
    const i = indexAt(rep.t, clock.t);
    this.readouts.update(readoutsAt(rep, i, clock.t), rep.t[0]);
    this.transport.update(clock, rep);
    this.timeline.setTime(clock.t);
    for (const p of this.plots) p.setTime(clock.t);
    return i;
  }

  /** Transport state only (e.g. after a Play/Pause or speed change). */
  onTransport(clock: PlaybackClock, rep: Replay | null): void {
    this.transport.update(clock, rep);
  }
}
