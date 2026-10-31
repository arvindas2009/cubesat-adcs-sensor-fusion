// Side panels, timeline and plot strips of the ADCS page (docs/specs/frontend.md §2).
// Browser-only: all non-trivial logic lives in logic.ts / format.ts / history.ts and is unit-tested.
import type { Axis, Command, Frame, FrameEvents, Info, Presets, SessionParams, Status } from '../types';
import { DASH, fmtAngle, fmtFixed, fmtPercent, fmtSpeed, fmtVec3 } from './format';
import { eclipseFraction, type Columns, type ModeSegment } from './history';
import {
  KICK_DEFAULT_DEG_S,
  KICK_MAX_DEG_S,
  SLIDER_MAX,
  buildKickCommand,
  fieldSpecs,
  intervalRect,
  orbitTicks,
  sliderToSpeed,
  speedToSlider,
  timeToX,
  validateAll,
  type FieldSpec,
} from './logic';
import { FONT_MONO, FONT_UI, chromeColors, dataColors, onThemeChange, whenFontsReady } from '../theme';
import { buildPlotCard } from '../ui/chart';
import { appendHelp } from '../ui/help';
import { readoutGroup, readoutRow, type RowSpec } from '../ui/readout';
import { chartDialog } from '../ui/ChartDialog';
import { timelineSummary } from '../ui/a11y';
import { CHART_HELP, type TermId } from '../ui/glossary';
import { ADCS_EVENT_SHAPE, adcsTimelineItems, drawMarker } from '../ui/glyphs';
import { renderLegend, type LegendHandle } from '../ui/legend';
import { AdcsChart, PLOTS } from './charts';
import { StatusBar } from './StatusBar';
import { modeName } from './theme';



export interface PanelHandlers {
  onStart(params: SessionParams): void;
  onPlay(): void;
  onPause(): void;
  onSpeed(value: number): void;
  onStep(n: number): void;
  onReset(): void;
  onKick(cmd: Extract<Command, { cmd: 'kick' }>): void;
}

export interface PanelRoots {
  form: HTMLElement;
  transport: HTMLElement;
  kick: HTMLElement;
  readouts: HTMLElement;
  timeline: HTMLElement;
  plots: [HTMLElement, HTMLElement, HTMLElement, HTMLElement];
  banner: HTMLElement;
  toasts: HTMLElement;
  /** The ADCS slot of the top bar (session state, time, orbit, speed). */
  status: HTMLElement;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function body(panel: HTMLElement): HTMLElement {
  return panel.querySelector<HTMLElement>('.panel-body') ?? panel;
}

// ===================================================================== parameter form

class FormPanel {
  private specs: FieldSpec[] = [];
  private readonly inputs = new Map<string, HTMLInputElement | HTMLSelectElement>();
  private readonly hints = new Map<string, HTMLElement>();
  private readonly fields: HTMLElement;
  private readonly startBtn: HTMLButtonElement;
  private readonly status: HTMLElement;
  private hasSession = false;
  private preparing = false;
  private statusFromValidation = false;

  constructor(root: HTMLElement, private readonly handlers: PanelHandlers) {
    const b = body(root);
    this.fields = el('div', 'form-grid');
    this.startBtn = el('button', 'primary', 'Start session');
    this.startBtn.type = 'button';
    this.startBtn.disabled = true;
    this.status = el('span', 'form-status', 'Loading presets…');
    // status text and the primary button sit at the end of the card head (saves a row)
    const row = el('div', 'form-actions head-end');
    row.append(this.status, this.startBtn);
    (root.querySelector('.card-head') ?? b).append(row);
    b.append(this.fields);
    this.startBtn.addEventListener('click', () => this.submit());
  }

  setPresets(p: Presets): void {
    this.specs = fieldSpecs(p.params_schema.properties);
    this.fields.replaceChildren();
    this.inputs.clear();
    this.hints.clear();
    for (const spec of this.specs) {
      const def = (p.defaults as unknown as Record<string, unknown>)[spec.name] ?? spec.def;
      const wrap = el('label', `field field-${spec.kind}`);
      const lab = el('span', 'field-label', spec.label);
      let input: HTMLInputElement | HTMLSelectElement;
      if (spec.kind === 'enum') {
        const sel = el('select');
        for (const o of spec.options ?? []) sel.append(new Option(o, o));
        sel.value = String(def);
        input = sel;
      } else if (spec.kind === 'boolean') {
        const cb = el('input');
        cb.type = 'checkbox';
        cb.checked = Boolean(def);
        input = cb;
      } else {
        const num = el('input');
        num.type = 'number';
        num.step = spec.kind === 'integer' ? '1' : 'any';
        if (spec.min !== undefined) num.min = String(spec.min);
        if (spec.max !== undefined) num.max = String(spec.max);
        num.value = String(def);
        input = num;
      }
      input.name = spec.name;
      const hint = el('small', 'field-hint', spec.hint);
      wrap.title = spec.hint; // the allowed range is a tooltip; an error shows in the card head
      this.hints.set(spec.name, hint);
      this.inputs.set(spec.name, input);
      input.addEventListener('input', () => this.validate());
      input.addEventListener('change', () => this.validate());
      wrap.append(lab, input, hint);
      this.fields.append(wrap);
    }
    this.setStatus('', false);
    this.validate();
  }

  setPresetsFailed(message: string): void {
    this.setStatus(message, true);
  }

  private setStatus(text: string, bad: boolean): void {
    this.status.textContent = text;
    this.status.title = text;
    this.status.classList.toggle('bad', bad);
  }

  private raw(): Record<string, string | boolean> {
    const r: Record<string, string | boolean> = {};
    for (const [name, input] of this.inputs) {
      r[name] = input instanceof HTMLInputElement && input.type === 'checkbox' ? input.checked : input.value;
    }
    return r;
  }

  private validate(): boolean {
    const v = validateAll(this.specs, this.raw());
    let first: string | null = null;
    for (const spec of this.specs) {
      const hint = this.hints.get(spec.name);
      const input = this.inputs.get(spec.name);
      if (!hint || !input) continue;
      const err = v.errors[spec.name];
      hint.textContent = err ?? spec.hint;
      hint.classList.toggle('bad', Boolean(err));
      input.classList.toggle('invalid', Boolean(err));
      input.setAttribute('aria-invalid', String(Boolean(err)));
      if (err && first === null) first = `${spec.label}: ${err}`;
    }
    if (first !== null) {
      this.setStatus(first, true);
      this.statusFromValidation = true;
    } else if (this.statusFromValidation) {
      this.setStatus('', false);
      this.statusFromValidation = false;
    }
    this.startBtn.disabled = !v.valid || this.preparing || this.specs.length === 0;
    return v.valid;
  }

  private submit(): void {
    const v = validateAll(this.specs, this.raw());
    if (!v.valid) return;
    this.handlers.onStart(v.values as SessionParams);
  }

  /** "Preparing…" while the POST is pending (a 15-orbit build takes a second or two). */
  setPreparing(on: boolean, label = 'Preparing…'): void {
    this.preparing = on;
    this.startBtn.textContent = on ? label : this.hasSession ? 'Start new session' : 'Start session';
    this.startBtn.classList.toggle('busy', on);
    for (const i of this.inputs.values()) i.disabled = on;
    this.setStatus(on ? 'Building the simulation…' : '', false);
    this.statusFromValidation = false;
    this.validate();
  }

  setSessionActive(on: boolean): void {
    this.hasSession = on;
    if (!this.preparing) this.startBtn.textContent = on ? 'Start new session' : 'Start session';
  }

  setError(message: string): void {
    this.setStatus(message, true);
  }
}

// ===================================================================== transport

class TransportPanel {
  private readonly playBtn: HTMLButtonElement;
  private readonly resetBtn: HTMLButtonElement;
  private readonly slider: HTMLInputElement;
  private readonly speedText: HTMLElement;
  private readonly stepBtns: HTMLButtonElement[] = [];
  private connected = false;
  private playing = false;
  private done = false;
  private speedSet = 50;
  private speedActual: number | null = null;
  private lastUserSpeedAt = -Infinity;
  private speedTimer: number | undefined;
  private pendingSpeed = 50;

  constructor(root: HTMLElement, private readonly handlers: PanelHandlers) {
    const b = body(root);
    this.playBtn = el('button', 'primary wide', 'Play');
    this.resetBtn = el('button', undefined, 'Reset');
    const r1 = el('div', 'row');
    r1.append(this.playBtn, this.resetBtn);

    this.slider = el('input', 'speed-slider');
    this.slider.type = 'range';
    this.slider.min = '0';
    this.slider.max = String(SLIDER_MAX);
    this.slider.step = '1';
    this.slider.setAttribute('aria-label', 'Simulation speed (log scale, 1 to 2000 times real time)');
    this.speedText = el('span', 'speed-text num');
    const r2 = el('div', 'row speed-row');
    const lab = el('span', 'field-label', 'Speed');
    r2.append(lab, this.slider, this.speedText);

    r1.append(el('span', 'field-label grow', 'Step (paused)'));
    for (const n of [1, 60, 600]) {
      const btn = el('button', undefined, `+${n}`);
      btn.type = 'button';
      btn.title = `Advance ${n} control step${n > 1 ? 's' : ''}`;
      btn.addEventListener('click', () => this.handlers.onStep(n));
      this.stepBtns.push(btn);
      r1.append(btn);
    }
    b.append(r1, r2);

    this.playBtn.type = 'button';
    this.resetBtn.type = 'button';
    this.playBtn.addEventListener('click', () => {
      if (this.playing) {
        this.playing = false;
        this.handlers.onPause();
      } else {
        this.playing = true;
        this.handlers.onPlay();
      }
      this.refresh();
    });
    this.resetBtn.addEventListener('click', () => this.handlers.onReset());
    this.slider.addEventListener('input', () => {
      this.lastUserSpeedAt = performance.now();
      this.speedSet = sliderToSpeed(Number(this.slider.value));
      this.pendingSpeed = this.speedSet;
      this.refresh();
      if (this.speedTimer === undefined) {
        this.speedTimer = window.setTimeout(() => {
          this.speedTimer = undefined;
          this.handlers.onSpeed(this.pendingSpeed);
        }, 80);
      }
    });
    this.refresh();
  }

  setConnected(on: boolean): void {
    this.connected = on;
    if (!on) this.playing = false;
    this.refresh();
  }

  /** Apply status from `hello` or a frame. The slider follows the server unless the user just moved it. */
  setState(s: { playing: boolean; done: boolean; speedSet: number; speedActual: number | null }): void {
    this.playing = s.playing;
    this.done = s.done;
    this.speedActual = s.speedActual;
    if (performance.now() - this.lastUserSpeedAt > 1200) {
      this.speedSet = s.speedSet;
      this.slider.value = String(Math.round(speedToSlider(s.speedSet)));
    }
    this.refresh();
  }

  private refresh(): void {
    const on = this.connected;
    this.playBtn.textContent = this.playing ? 'Pause' : 'Play';
    this.playBtn.disabled = !on || (this.done && !this.playing);
    this.resetBtn.disabled = !on;
    this.slider.disabled = !on;
    const canStep = on && !this.playing && !this.done;
    for (const b of this.stepBtns) b.disabled = !canStep;
    const actual = this.speedActual !== null && this.playing ? fmtSpeed(this.speedActual) : DASH;
    this.speedText.textContent = `set ${fmtSpeed(this.speedSet)} / actual ${actual}`;
    if (!this.slider.matches(':active')) this.slider.value = String(Math.round(speedToSlider(this.speedSet)));
  }
}

// ===================================================================== kick

class KickPanel {
  private axis: Axis = 'x';
  private sign: 1 | -1 = 1;
  private readonly axisBtns = new Map<Axis, HTMLButtonElement>();
  private readonly signBtns = new Map<number, HTMLButtonElement>();
  private readonly mag: HTMLInputElement;
  private readonly apply: HTMLButtonElement;
  private readonly err: HTMLElement;
  private enabled = false;

  constructor(root: HTMLElement, private readonly handlers: PanelHandlers, private readonly onLocalError: (m: string) => void) {
    const b = body(root);
    const row = el('div', 'row');
    row.append(el('span', 'field-label', 'Axis'));
    const seg = el('span', 'seg');
    for (const a of ['x', 'y', 'z'] as Axis[]) {
      const btn = el('button', undefined, a);
      btn.type = 'button';
      btn.style.setProperty('--seg-color', `var(--axis-${a})`); // light / dark value of the axis role
      btn.addEventListener('click', () => {
        this.axis = a;
        this.refresh();
      });
      this.axisBtns.set(a, btn);
      seg.append(btn);
    }
    const sgn = el('span', 'seg');
    for (const [s, txt] of [
      [1, '+'],
      [-1, '−'],
    ] as [1 | -1, string][]) {
      const btn = el('button', undefined, txt);
      btn.type = 'button';
      btn.setAttribute('aria-label', s > 0 ? 'positive' : 'negative');
      btn.addEventListener('click', () => {
        this.sign = s;
        this.refresh();
      });
      this.signBtns.set(s, btn);
      sgn.append(btn);
    }
    this.mag = el('input', 'num kick-mag');
    this.mag.type = 'number';
    this.mag.min = '0.1';
    this.mag.max = String(KICK_MAX_DEG_S);
    this.mag.step = '0.5';
    this.mag.value = String(KICK_DEFAULT_DEG_S);
    this.mag.setAttribute('aria-label', 'Kick magnitude in degrees per second');
    this.apply = el('button', 'kick-apply', 'Apply kick');
    this.apply.type = 'button';
    row.append(seg, sgn, this.mag, el('span', 'unit', '°/s'), this.apply);
    this.err = el('span', 'card-sub', `Body-axis rate step, 0 – ${KICK_MAX_DEG_S} °/s. The server caps total rate at 30 °/s.`);
    (root.querySelector('.card-head') ?? b).append(this.err);
    b.append(row);
    this.apply.addEventListener('click', () => this.submit());
    this.mag.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') this.submit();
    });
    this.refresh();
  }

  private submit(): void {
    const r = buildKickCommand(this.axis, this.sign, Number(this.mag.value));
    if (!r.ok) {
      this.onLocalError(r.error);
      return;
    }
    this.handlers.onKick(r.cmd);
  }

  setEnabled(on: boolean): void {
    this.enabled = on;
    this.refresh();
  }

  private refresh(): void {
    for (const [a, btn] of this.axisBtns) btn.setAttribute('aria-pressed', String(a === this.axis));
    for (const [s, btn] of this.signBtns) btn.setAttribute('aria-pressed', String(s === this.sign));
    this.apply.disabled = !this.enabled;
    this.mag.disabled = !this.enabled;
  }
}

// ===================================================================== readouts

class Readouts {
  private readonly vals = new Map<string, HTMLElement>();
  private readonly base = new Map<string, string>(); // initial class of each value element
  private readonly cache = new Map<string, string>();
  private readonly badge: HTMLElement;
  private readonly light: HTMLElement;
  private readonly dump: HTMLElement;
  private modeHelp: HTMLElement[] = []; // [DETUMBLE tip button, POINTING tip button]; only the current mode's shows
  private eclipseText = DASH;

  constructor(root: HTMLElement) {
    const b = body(root);
    b.classList.add('ro-body');

    // Status: the three state chips, each with the help of what it names.
    const status = readoutGroup('Status');
    const chips = el('div', 'ro-chips');
    const chip = (key: string, cls: string, title: string, help: TermId[]): HTMLElement => {
      const wrap = el('span', 'ro-chip');
      const e = el('span', cls, DASH);
      e.title = title;
      wrap.append(e);
      for (const h of help) appendHelp(wrap, h);
      chips.append(wrap);
      this.vals.set(key, e);
      this.base.set(key, cls);
      return e;
    };
    this.badge = chip('mode', 'readout-value badge', 'Control mode', ['detumble', 'pointing']);
    this.modeHelp = [...(this.badge.parentElement?.querySelectorAll<HTMLElement>('button.help') ?? [])];
    if (this.modeHelp[1]) this.modeHelp[1].hidden = true;
    this.light = chip('light', 'readout-value pill-state', 'Illumination', ['eclipse']);
    this.dump = chip('dump', 'readout-value pill-state', 'Momentum dumping', ['dumping']);
    status.section.replaceChildren(status.section.firstElementChild as Element, chips);

    const groups: { heading: string; rows: RowSpec[] }[] = [
      {
        heading: 'Time',
        rows: [
          { key: 't', label: 'Sim time', unit: 's' },
          { key: 'orbit', label: 'Orbits completed' },
          { key: 'triad', label: 'TRIAD start', unit: 's', help: 'triad' },
        ],
      },
      {
        heading: 'Pointing',
        rows: [
          { key: 'nadir', label: 'Nadir pointing error', unit: '°', help: 'nadir', title: 'Nadir pointing error' },
          { key: 'estNadir', label: 'Estimated nadir error', unit: '°', help: 'estNadir', title: 'Estimated nadir pointing error' },
        ],
      },
      {
        heading: 'Attitude knowledge',
        rows: [
          { key: 'know', label: 'Knowledge error', unit: '°', help: 'knowledge', title: 'Attitude knowledge error' },
          { key: 'sigma3', label: 'Filter 3σ bound', unit: '°', help: 'sigma3', title: '3 × the filter attitude sigma' },
        ],
      },
      {
        heading: 'Motion',
        rows: [
          { key: 'rate', label: 'Body rate |ω|', unit: '°/s', help: 'rate', title: 'Body rate |ω|' },
          { key: 'w', label: 'Estimated rate ω̂ x, y, z', unit: '°/s', vec: true, title: 'Estimated body rate' },
        ],
      },
      {
        heading: 'Actuators',
        rows: [
          { key: 'wheel', label: 'Wheel speeds x, y, z', unit: 'rpm', vec: true },
          { key: 'wheelFrac', label: 'Wheel use', unit: 'of max', help: 'wheelUse', title: 'Wheel speed as a share of the maximum' },
          { key: 'm', label: 'Magnetorquer dipole x, y, z', unit: 'A·m²', help: 'dipole', vec: true },
        ],
      },
      {
        heading: 'Orbit',
        rows: [{ key: 'eclipse', label: 'Eclipse share of run', unit: 'of run', help: 'eclipse', title: 'Share of the run spent in eclipse' }],
      },
    ];
    b.append(status.section);
    for (const g of groups) {
      const grp = readoutGroup(g.heading);
      for (const spec of g.rows) {
        const r = readoutRow(spec, DASH);
        grp.list.append(r.row);
        this.vals.set(spec.key, r.value);
        this.base.set(spec.key, 'readout-value');
      }
      b.append(grp.section);
    }
  }

  setInfo(info: Info): void {
    this.eclipseText = fmtPercent(eclipseFraction(info.eclipses_s, info.t_end_s), 1);
    this.set('eclipse', this.eclipseText);
  }

  private set(key: string, text: string): void {
    if (this.cache.get(key) === text) return;
    this.cache.set(key, text);
    const e = this.vals.get(key);
    if (e) e.textContent = text;
  }

  update(f: Frame): void {
    const m = f.metrics;
    this.set('t', fmtFixed(f.t, 0));
    this.set('orbit', fmtFixed(f.orbit, 3));
    this.set('triad', f.events.t_init === null ? 'pending' : fmtFixed(f.events.t_init, 0));
    this.set('nadir', fmtAngle(m.nadir_err_deg));
    this.set('estNadir', fmtAngle(m.est_nadir_err_deg));
    this.set('know', fmtAngle(m.know_err_deg));
    this.set('sigma3', m.att_sigma_deg === null ? DASH : fmtAngle(3 * m.att_sigma_deg));
    this.set('rate', fmtFixed(m.rate_deg_s, 3));
    this.set('w', fmtVec3(m.w_est_deg_s, 3, 8));
    this.set('wheel', fmtVec3(m.wheel_speed_rpm, 0, 7));
    this.set('wheelFrac', fmtPercent(m.wheel_frac, 1));
    this.set('m', fmtVec3(m.m_Am2, 3, 8));

    const modeTxt = modeName(f.mode);
    if (this.cache.get('mode') !== modeTxt) {
      this.cache.set('mode', modeTxt);
      this.badge.textContent = modeTxt;
      this.badge.className = `readout-value badge ${f.mode === 1 ? 'pointing' : 'detumble'}`;
      this.modeHelp.forEach((b, i) => (b.hidden = i !== (f.mode === 1 ? 1 : 0)));
    }
    const lightTxt = f.lit ? 'Sunlit' : 'Eclipse';
    if (this.cache.get('light') !== lightTxt) {
      this.cache.set('light', lightTxt);
      this.light.textContent = lightTxt;
      this.light.className = `readout-value pill-state ${f.lit ? 'lit' : 'eclipsed'}`;
    }
    const dumpTxt = f.dump ? 'dumping: active' : 'dumping: off';
    if (this.cache.get('dump') !== dumpTxt) {
      this.cache.set('dump', dumpTxt);
      this.dump.textContent = dumpTxt;
      this.dump.className = `readout-value pill-state ${f.dump ? 'dump-on' : 'dim'}`;
    }
  }

  clear(): void {
    this.cache.clear();
    for (const [k, v] of this.vals) {
      v.textContent = DASH;
      v.className = this.base.get(k) ?? 'readout-value';
    }
  }
}

// ===================================================================== timeline

class Timeline {
  private readonly canvas = el('canvas', 'timeline-canvas');
  private readonly ctx: CanvasRenderingContext2D | null;
  private readonly legend: LegendHandle;
  private describedAt = -Infinity;
  private info: Info | null = null;
  private w = 0;
  private h = 0;
  private dpr = 1;
  private last: { t: number; segments: ModeSegment[]; events: FrameEvents | null } = { t: 0, segments: [], events: null };

  constructor(root: HTMLElement) {
    // DOM legend row above the canvas (the same legend system as the plots), then the canvas band.
    const legendHost = el('div', 'timeline-legend');
    this.legend = renderLegend(adcsTimelineItems(dataColors(), chromeColors()), { surface: 'panel' });
    legendHost.append(this.legend.el);
    const canvasHost = el('div', 'timeline-body');
    canvasHost.append(this.canvas);
    root.append(legendHost, canvasHost);
    this.ctx = this.canvas.getContext('2d');
    this.canvas.setAttribute('role', 'img');
    this.canvas.setAttribute('aria-label', timelineSummary(null, [], null, 0));
    new ResizeObserver(() => {
      this.dpr = window.devicePixelRatio || 1;
      this.w = Math.max(1, Math.floor(canvasHost.clientWidth));
      this.h = Math.max(1, Math.floor(canvasHost.clientHeight));
      this.canvas.width = Math.round(this.w * this.dpr);
      this.canvas.height = Math.round(this.h * this.dpr);
      this.draw();
    }).observe(canvasHost);
  }

  /** Mode change: the legend glyphs take the new palette (the canvas is redrawn by the caller). */
  recolor(): void {
    this.legend.setColors(adcsTimelineItems(dataColors(), chromeColors()));
  }

  setInfo(info: Info | null): void {
    this.info = info;
    this.last = { t: 0, segments: [], events: null };
    this.draw();
    this.describe(true);
  }

  update(t: number, segments: ModeSegment[], events: FrameEvents | null): void {
    this.last = { t, segments, events };
    this.draw();
    this.describe(false);
  }

  /** Text alternative of the canvas (`role="img"`), refreshed at most once per second. */
  private describe(force: boolean): void {
    const now = performance.now();
    if (!force && now - this.describedAt < 1000) return;
    this.describedAt = now;
    this.canvas.setAttribute('aria-label', timelineSummary(this.info, this.last.segments, this.last.events, this.last.t));
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
    const x0 = 12;
    const x1 = this.w - 42; // room at the right for the unit label after the last tick
    const bandTop = 4;
    const bandBot = Math.max(bandTop + 20, this.h - 18); // tick numbers sit 10 px below it, inside the canvas
    const info = this.info;

    // band background
    g.fillStyle = c.band;
    g.fillRect(x0, bandTop, x1 - x0, bandBot - bandTop);
    g.strokeStyle = c.bandEdge;
    g.lineWidth = 1;
    g.strokeRect(x0 + 0.5, bandTop + 0.5, x1 - x0 - 1, bandBot - bandTop - 1);
    if (!info) return;
    const tEnd = info.t_end_s;

    // eclipse bands (darker, with a hairline edge so they read on the dark panel)
    for (const [a, b] of info.eclipses_s) {
      const r = intervalRect(a, b, tEnd, x0, x1, 2);
      g.fillStyle = c.eclipse;
      g.fillRect(r.x, bandTop + 1, r.w, bandBot - bandTop - 2);
      g.strokeStyle = c.eclipseEdge;
      g.strokeRect(r.x + 0.5, bandTop + 1.5, Math.max(1, r.w - 1), bandBot - bandTop - 3);
    }

    // orbit ticks (numerals in mono)
    g.font = `12px ${FONT_MONO}`;
    g.fillStyle = c.dim;
    g.strokeStyle = c.tick;
    for (const tk of orbitTicks(tEnd, info.period_s)) {
      const x = Math.round(timeToX(tk.t, tEnd, x0, x1)) + 0.5;
      g.beginPath();
      g.moveTo(x, bandBot);
      g.lineTo(x, bandBot + 4);
      g.stroke();
      g.textAlign = 'center';
      g.fillText(`${tk.n}`, x, bandBot + 10);
    }
    g.textAlign = 'left';
    g.fillText('orbit', x1 + 12, bandBot + 10);
    g.font = `12px ${FONT_UI}`;

    // mode strip along the bottom of the band
    const stripH = 7;
    for (const s of this.last.segments) {
      const r = intervalRect(s.t0, s.t1, tEnd, x0, x1, 1);
      g.fillStyle = s.mode === 1 ? d.pointing : d.detumble;
      g.fillRect(r.x, bandBot - stripH - 1, r.w, stripH);
    }

    // event markers
    const ev = this.last.events;
    const markTop = bandTop + 1;
    const markBot = bandBot - stripH - 2;
    const vline = (t: number, color: string) => {
      const x = Math.round(timeToX(t, tEnd, x0, x1)) + 0.5;
      g.strokeStyle = color;
      g.lineWidth = 1.5;
      g.beginPath();
      g.moveTo(x, markTop);
      g.lineTo(x, markBot);
      g.stroke();
      return x;
    };
    if (ev) {
      for (const t of ev.switch_times) vline(t, c.marker);
      for (const t of ev.fallback_times) {
        // Triangle at the bottom of the lane: a fallback usually lands < 1 px after its kick,
        // whose dot sits at the top, so both stay visible.
        const x = vline(t, d.danger);
        drawMarker(g, ADCS_EVENT_SHAPE.fallback, x, markBot - 4, 8, d.danger);
      }
      for (const k of ev.kicks) {
        const x = vline(k.t, d.kick);
        drawMarker(g, ADCS_EVENT_SHAPE.kick, x, markTop + 4, 8, d.kick);
      }
      if (ev.t_init !== null) {
        const x = vline(ev.t_init, d.estimate);
        drawMarker(g, ADCS_EVENT_SHAPE.tInit, x, markTop + 5, 8, d.estimate);
      }
      g.lineWidth = 1;
    }

    // cursor
    const cx = Math.round(timeToX(this.last.t, tEnd, x0, x1)) + 0.5;
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

export class Panels {
  private readonly form: FormPanel;
  private readonly transport: TransportPanel;
  private readonly kick: KickPanel;
  private readonly readouts: Readouts;
  private readonly timeline: Timeline;
  private readonly cards: AdcsChart[];
  private readonly expandBtns: HTMLButtonElement[] = [];
  private readonly summaryEls: HTMLElement[] = [];
  private summaryAt = -Infinity;
  /** Every chart that must follow data, theme and fonts: the four cards plus the chart of the open dialog. */
  private readonly live = new Set<AdcsChart>();
  private lastCols: Columns | null = null;
  private readonly status: StatusBar;
  private readonly banner: HTMLElement;
  private readonly toasts: HTMLElement;
  private info: Info | null = null;

  constructor(roots: PanelRoots, handlers: PanelHandlers) {
    this.banner = roots.banner;
    this.toasts = roots.toasts;
    this.status = new StatusBar(roots.status);
    this.form = new FormPanel(roots.form, handlers);
    this.transport = new TransportPanel(roots.transport, handlers);
    this.kick = new KickPanel(roots.kick, handlers, (m) => this.toast(m));
    this.readouts = new Readouts(roots.readouts);
    this.timeline = new Timeline(roots.timeline);
    this.cards = roots.plots.map((r, i) => {
      const spec = PLOTS[i];
      const parts = buildPlotCard(r, { title: spec.title, unit: spec.unit }, () => this.expand(i));
      this.expandBtns.push(parts.expandBtn);
      this.summaryEls.push(parts.summaryEl);
      return new AdcsChart(parts.host, spec, { variant: 'card', info: () => this.info, legendHost: parts.legendHost });
    });
    for (const c of this.cards) this.live.add(c);
    // Mode change: charts are rebuilt with the new palette from the stored data (no reload, no lost history).
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

  setPresets(p: Presets): void {
    this.form.setPresets(p);
  }
  setPresetsFailed(message: string): void {
    this.form.setPresetsFailed(message);
  }
  setPreparing(on: boolean, label?: string): void {
    this.form.setPreparing(on, label);
  }
  setFormError(message: string): void {
    this.form.setError(message);
  }

  /** `hello` (new session or reset): static info and initial status; clears readouts and the timeline. */
  onHello(info: Info, status: Status): void {
    this.info = info;
    this.form.setSessionActive(true);
    this.transport.setConnected(true);
    this.transport.setState({ playing: status.playing, done: status.done, speedSet: status.speed, speedActual: null });
    this.status.setSession(true, status.playing, status.done);
    this.status.clearValues();
    this.kick.setEnabled(!status.done);
    this.readouts.clear();
    this.readouts.setInfo(info);
    this.timeline.setInfo(info);
    for (const c of this.live) c.setInfo(info);
  }

  onClosed(): void {
    this.status.setSession(false, false, false);
    this.status.clearValues();
    this.transport.setConnected(false);
    this.kick.setEnabled(false);
  }

  onFrame(f: Frame): void {
    this.readouts.update(f);
    this.transport.setState({ playing: f.playing, done: f.done, speedSet: f.speed_set, speedActual: f.speed_actual });
    this.kick.setEnabled(!f.done);
    this.status.setSession(true, f.playing, f.done);
    this.status.setFrame(f.t, f.orbit, f.speed_actual, f.playing);
  }

  /** Throttled by the caller (about 8 Hz): timeline and plot strips. */
  refreshCharts(cols: Columns, t: number, segments: ModeSegment[], events: FrameEvents | null): void {
    this.timeline.update(t, segments, events);
    this.lastCols = cols;
    for (const c of this.live) c.setData(cols);
    this.refreshSummaries(false);
  }

  /** Hidden text alternatives of the plot cards: the latest values, at most every 2 s. */
  private refreshSummaries(force: boolean): void {
    const now = performance.now();
    if (!force && now - this.summaryAt < 2000) return;
    this.summaryAt = now;
    this.cards.forEach((c, i) => {
      this.summaryEls[i].textContent = c.summaryText();
    });
  }

  /** The sheet settled (or the layout changed): uPlot caches its bounding rect, so refresh it on every chart. */
  syncCharts(): void {
    for (const c of this.live) c.syncRect();
  }

  /** Open the dialog with a large, zoomable chart of plot `i` (it keeps streaming while open). */
  expand(i: number): void {
    const spec = PLOTS[i];
    let chart: AdcsChart | null = null;
    chartDialog().open({
      title: spec.title,
      unit: spec.unit,
      help: CHART_HELP[spec.id],
      returnFocus: this.expandBtns[i],
      mount: (chartHost, legendHost) => {
        const c = new AdcsChart(chartHost, spec, { variant: 'large', info: () => this.info, legendHost });
        chart = c;
        this.live.add(c);
        if (this.lastCols) c.setData(this.lastCols);
        return { chart: c, onZoom: (cb) => c.onZoom(cb) };
      },
      onClose: () => {
        if (chart) this.live.delete(chart);
      },
    });
  }

  clearCharts(): void {
    this.timeline.update(0, [], null);
    const empty: Columns = Array.from({ length: 9 }, () => []);
    this.lastCols = empty;
    for (const c of this.live) c.setData(empty);
    this.refreshSummaries(true);
  }

  setBanner(message: string | null): void {
    this.banner.textContent = message ?? '';
    this.banner.classList.toggle('visible', Boolean(message));
  }

  toast(message: string): void {
    const t = el('div', 'toast', message);
    t.setAttribute('role', 'status');
    this.toasts.append(t);
    while (this.toasts.children.length > 4) this.toasts.firstElementChild?.remove();
    window.setTimeout(() => t.remove(), 7000);
  }

  get currentInfo(): Info | null {
    return this.info;
  }
}
