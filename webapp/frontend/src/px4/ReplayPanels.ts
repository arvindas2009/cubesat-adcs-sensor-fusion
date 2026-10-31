// Side panels, timeline and plot strips of the PX4 replay page (docs/specs/px4_replay.md §3.3).
// Browser-only: the non-trivial logic (series, clock, readout values) is pure and unit-tested in
// series.ts / clock.ts. Every number shown comes from the replay document.
import { DASH, fmtFixed } from '../adcs/format';
import { FONT_MONO, FONT_UI, chromeColors, dataColors, onThemeChange, whenFontsReady } from '../theme';
import { buildPlotCard } from '../ui/chart';
import { sliderKey, sliderText } from '../ui/a11y';
import { appendHelp } from '../ui/help';
import { readoutGroup, readoutRow, type RowSpec } from '../ui/readout';
import { chartDialog } from '../ui/ChartDialog';
import { CHART_HELP } from '../ui/glossary';
import { glyphSvg, px4TimelineItems } from '../ui/glyphs';
import { renderLegend, type LegendHandle } from '../ui/legend';
import { SPEEDS, DEFAULT_SPEED, indexAt, xToTime } from './clock';
import type { PlaybackClock } from './clock';
import { PLOT_HEADS, ReplayChart } from './charts';
import { nextReset, prevReset, readoutsAt } from './series';
import type { ReadoutValues } from './series';
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
    b.classList.add('ro-body');

    // Attitude and height: ours next to EKF2 (reference), one table.
    const g1 = readoutGroup('Attitude and height');
    const table = el('table', 'ro-table');
    const head = el('tr');
    const corner = el('th', 'visually-hidden', 'Quantity');
    corner.scope = 'col';
    head.append(corner, this.colHead('ours', 'ours', false), this.colHead('ekf2', 'EKF2 (reference)', true));
    const thead = el('thead');
    thead.append(head);
    const tbody = el('tbody');
    const addPair = (key: string, label: string, unit: string): void => {
      const tr = el('tr');
      const th = el('th', 'readout-label', label);
      th.scope = 'row';
      th.append(' ', el('span', 'ro-unit', unit));
      const a = el('td', 'readout-value', DASH);
      const c = el('td', 'readout-value', DASH);
      tr.append(th, a, c);
      tbody.append(tr);
      this.vals.set(`${key}.o`, a);
      this.vals.set(`${key}.e`, c);
    };
    addPair('roll', 'roll', '°');
    addPair('pitch', 'pitch', '°');
    addPair('yaw', 'yaw', '°');
    addPair('height', 'height above start', 'm');
    const cols = el('colgroup');
    cols.append(el('col'), el('col', 'c-val'), el('col', 'c-val'));
    table.append(cols, thead, tbody);
    g1.section.replaceChildren(g1.section.firstElementChild as Element, table);
    b.append(g1.section);

    const groups: { heading: string; rows: RowSpec[] }[] = [
      {
        heading: 'Difference, ours − EKF2',
        rows: [
          { key: 'tilt', label: 'Tilt', unit: '°', help: 'tilt', title: 'ours − EKF2' },
          { key: 'yawd', label: 'Yaw (signed)', unit: '°', title: 'ours − EKF2, signed' },
          { key: 'posh', label: 'Horizontal position', unit: 'm', title: 'ours − EKF2' },
          { key: 'posd', label: 'Vertical position (+ = ours lower)', unit: 'm', title: 'ours − EKF2, positive = ours lower' },
        ],
      },
      {
        heading: 'Filter confidence',
        rows: [{ key: 's3', label: '3σ attitude, largest axis', unit: '°', help: 'sigma3', title: 'Largest of the three attitude-error 3-sigma values of our filter' }],
      },
      {
        heading: 'Time',
        rows: [
          { key: 't', label: 'Log time', unit: 's', title: 'Log time of the current sample' },
          { key: 'el', label: 'Elapsed', unit: 's' },
          { key: 'frac', label: 'Position', unit: 'of log' },
        ],
      },
    ];
    for (const g of groups) {
      const grp = readoutGroup(g.heading);
      for (const spec of g.rows) {
        const r = readoutRow(spec, DASH);
        grp.list.append(r.row);
        this.vals.set(spec.key, r.value);
      }
      b.append(grp.section);
    }
  }

  /** Column head of the table: a legend glyph (colour from the role via CSS), the name, and for EKF2 its help. */
  private colHead(role: 'ours' | 'ekf2', label: string, help: boolean): HTMLElement {
    const th = el('th', `ro-head ${role}`);
    th.scope = 'col';
    const g = el('span', 'lg-glyph');
    g.innerHTML = glyphSvg({ kind: 'line', dash: role === 'ekf2' }, 'currentColor');
    th.append(g, ' ', label);
    if (help) appendHelp(th, 'ekf2');
    return th;
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
  private readonly legend: LegendHandle;
  private describedAt = -Infinity;
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
    // DOM legend row above the canvas (the same legend system as the plots), then the canvas band.
    const legendHost = el('div', 'timeline-legend');
    this.legend = renderLegend(px4TimelineItems(dataColors(), chromeColors()), { surface: 'panel' });
    legendHost.append(this.legend.el);
    const canvasHost = el('div', 'timeline-body');
    canvasHost.append(this.canvas);
    root.append(legendHost, canvasHost);
    this.ctx = this.canvas.getContext('2d');
    this.canvas.classList.add('seekable');
    // The canvas is a slider over the log: arrows move 1 s (Shift 10 s), Home / End jump; all through onSeek.
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute('role', 'slider');
    this.canvas.setAttribute('aria-label', 'Position in the log');
    this.canvas.setAttribute('aria-valuemin', '0');
    this.canvas.setAttribute('aria-valuemax', '0');
    this.canvas.setAttribute('aria-valuenow', '0');
    this.canvas.setAttribute('aria-valuetext', 'no log loaded');
    this.canvas.addEventListener('keydown', (e) => {
      const r = this.rep;
      const move = sliderKey(e.key, e.shiftKey);
      if (!r || !move || e.ctrlKey || e.metaKey || e.altKey) return;
      e.preventDefault();
      const t0 = r.t[0];
      const t1 = r.t[r.t.length - 1];
      const to = 'delta' in move ? this.t + move.delta : move.to === 'start' ? t0 : t1;
      this.onSeek(Math.min(t1, Math.max(t0, to)));
      this.describe(true);
    });
    new ResizeObserver(() => {
      this.dpr = window.devicePixelRatio || 1;
      this.w = Math.max(1, Math.floor(canvasHost.clientWidth));
      this.h = Math.max(1, Math.floor(canvasHost.clientHeight));
      this.canvas.width = Math.round(this.w * this.dpr);
      this.canvas.height = Math.round(this.h * this.dpr);
      this.draw();
    }).observe(canvasHost);

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
    if (rep) {
      this.canvas.setAttribute('aria-valuemax', (rep.t[rep.t.length - 1] - rep.t[0]).toFixed(1));
    } else {
      this.canvas.setAttribute('aria-valuemax', '0');
      this.canvas.setAttribute('aria-valuetext', 'no log loaded');
    }
    this.describe(true);
  }

  /** aria-valuenow / aria-valuetext of the slider, at most 4 times a second while playing. */
  private describe(force: boolean): void {
    const r = this.rep;
    if (!r) return;
    const now = performance.now();
    if (!force && now - this.describedAt < 250) return;
    this.describedAt = now;
    const dur = r.t[r.t.length - 1] - r.t[0];
    const el = this.t - r.t[0];
    this.canvas.setAttribute('aria-valuenow', el.toFixed(1));
    this.canvas.setAttribute('aria-valuetext', sliderText(el, dur));
  }

  /** Mode change: the legend glyphs take the new palette (the canvas is redrawn by the caller). */
  recolor(): void {
    this.legend.setColors(px4TimelineItems(dataColors(), chromeColors()));
  }

  setTime(t: number): void {
    if (t === this.t) return;
    this.t = t;
    this.draw();
    this.describe(false);
  }

  draw(): void {
    const g = this.ctx;
    if (!g || this.w < 2) return;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.clearRect(0, 0, this.w, this.h);
    const c = chromeColors(); // the active mode's card / band / marker colours
    const d = dataColors();
    g.font = `12px ${FONT_UI}`;
    g.textBaseline = 'middle';
    const { x0, x1 } = this;
    const bandTop = 4;
    const bandBot = Math.max(bandTop + 20, this.h - 18); // tick numbers sit 10 px below it, inside the canvas

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

    // time ticks (elapsed seconds, numerals in mono)
    g.font = `12px ${FONT_MONO}`;
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
    g.font = `12px ${FONT_UI}`;

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

// ===================================================================== composite

export class ReplayPanels {
  private readonly log: LogPanel;
  private readonly transport: TransportPanel;
  private readonly readouts: Readouts;
  private readonly timeline: TimelineBar;
  private readonly cards: ReplayChart[];
  private readonly expandBtns: HTMLButtonElement[] = [];
  private readonly summaryEls: HTMLElement[] = [];
  private summaryAt = -Infinity;
  /** Every chart that must follow the replay, theme and fonts: the four cards plus the chart of the open dialog. */
  private readonly live = new Set<ReplayChart>();
  private rep: Replay | null = null;
  private lastT = 0;
  private readonly onSeek: (t: number) => void;

  constructor(roots: ReplayPanelRoots, handlers: ReplayHandlers) {
    this.log = new LogPanel(roots.log, handlers);
    this.transport = new TransportPanel(roots.transport, handlers);
    this.readouts = new Readouts(roots.readouts);
    this.timeline = new TimelineBar(roots.timeline, handlers.onSeek);
    this.onSeek = handlers.onSeek;
    this.cards = roots.plots.map((r, i) => {
      const parts = buildPlotCard(r, { title: PLOT_HEADS[i].title, unit: PLOT_HEADS[i].unit }, () => this.expand(i));
      this.expandBtns.push(parts.expandBtn);
      this.summaryEls.push(parts.summaryEl);
      return new ReplayChart(parts.host, i, handlers.onSeek, { variant: 'card', legendHost: parts.legendHost });
    });
    for (const c of this.cards) this.live.add(c);
    // Mode change: plots are rebuilt from the loaded replay with the new palette, the timeline is redrawn.
    onThemeChange(() => {
      for (const c of this.live) c.rebuild();
      this.timeline.recolor();
      this.timeline.draw();
    });
    // Canvas text drawn before the web font arrived used the fallback font.
    whenFontsReady(() => {
      for (const c of this.live) c.redraw();
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

  /** The sheet settled (or the layout changed): uPlot caches its bounding rect, so refresh it on every chart. */
  syncCharts(): void {
    for (const c of this.live) c.syncRect();
  }

  /** Open the dialog with a large, zoomable chart of plot `i` (the now-line keeps moving while it is open). */
  expand(i: number): void {
    const head = PLOT_HEADS[i];
    let chart: ReplayChart | null = null;
    chartDialog().open({
      title: head.title,
      unit: head.unit,
      help: CHART_HELP[head.id],
      returnFocus: this.expandBtns[i],
      mount: (chartHost, legendHost) => {
        const c = new ReplayChart(chartHost, i, this.onSeek, { variant: 'large', legendHost });
        chart = c;
        this.live.add(c);
        c.setReplay(this.rep);
        c.setTime(this.lastT);
        return { chart: c, onZoom: (cb) => c.onZoom(cb) };
      },
      onClose: () => {
        if (chart) this.live.delete(chart);
      },
    });
  }

  /** A new replay (or null): rebuild the plots, reset the timeline, transport and readouts. */
  setReplay(rep: Replay | null, clock: PlaybackClock | null): void {
    this.transport.setReplay(rep, clock);
    this.timeline.setReplay(rep);
    this.readouts.clear();
    this.rep = rep;
    for (const c of this.live) c.setReplay(rep);
    this.refreshSummaries(true);
    if (rep && clock) this.onTime(rep, clock);
  }

  /** Per animation frame (cheap: cached text, one line per plot). */
  onTime(rep: Replay, clock: PlaybackClock): number {
    const i = indexAt(rep.t, clock.t);
    this.readouts.update(readoutsAt(rep, i, clock.t), rep.t[0]);
    this.transport.update(clock, rep);
    this.timeline.setTime(clock.t);
    this.lastT = clock.t;
    for (const c of this.live) c.setTime(clock.t);
    this.refreshSummaries(false);
    return i;
  }

  /** Hidden text alternatives of the plot cards: the latest values at the playback time, at most every 2 s. */
  private refreshSummaries(force: boolean): void {
    const now = performance.now();
    if (!force && now - this.summaryAt < 2000) return;
    this.summaryAt = now;
    this.cards.forEach((c, i) => {
      this.summaryEls[i].textContent = c.summaryText();
    });
  }

  /** Transport state only (e.g. after a Play/Pause or speed change). */
  onTransport(clock: PlaybackClock, rep: Replay | null): void {
    this.transport.update(clock, rep);
  }
}
