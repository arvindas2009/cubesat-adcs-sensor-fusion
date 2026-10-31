// Side panels, timeline and plot strips of the ADCS page (docs/specs/frontend.md §2).
// Browser-only: all non-trivial logic lives in logic.ts / format.ts / history.ts and is unit-tested.
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import type { Axis, Command, Frame, FrameEvents, Info, Presets, SessionParams, Status } from '../types';
import { DASH, fmtAngle, fmtFixed, fmtPercent, fmtSpeed, fmtVec3 } from './format';
import { eclipseFraction, type Columns, type ModeSegment } from './history';
import {
  KICK_DEFAULT_DEG_S,
  KICK_MAX_DEG_S,
  SLIDER_MAX,
  buildKickCommand,
  clampLog,
  fieldSpecs,
  intervalRect,
  logRange,
  orbitTicks,
  sliderToSpeed,
  speedToSlider,
  timeRange,
  timeToX,
  validateAll,
  type FieldSpec,
} from './logic';
import { chromeColors, dataColors, onThemeChange, whenFontsReady, type DataColors } from '../theme';
import { modeName } from './theme';

const FONT_FAMILY = '"Schibsted Grotesk Variable", system-ui, sans-serif';

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
  private readonly stateText: HTMLElement;
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
    this.stateText = el('span', 'chip transport-state');
    (root.querySelector('.card-head') ?? b).append(this.stateText);
    this.stateText.classList.add('push');
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
    this.stateText.textContent = !on ? 'no session' : this.done ? 'run complete' : this.playing ? 'running' : 'paused';
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
  private eclipseText = DASH;

  constructor(root: HTMLElement) {
    const pills = el('div', 'ro-pills');
    const grid = el('div', 'readout-grid');
    body(root).append(pills, grid);
    const pill = (key: string, cls: string, title: string): HTMLElement => {
      const e = el('span', cls, DASH);
      e.title = title;
      pills.append(e);
      this.vals.set(key, e);
      this.base.set(key, cls);
      return e;
    };
    this.badge = pill('mode', 'readout-value badge', 'Control mode');
    this.light = pill('light', 'readout-value pill-state', 'Illumination');
    this.dump = pill('dump', 'readout-value pill-state', 'Momentum dumping');
    const add = (key: string, label: string, title?: string, wide = false): void => {
      const lab = el('span', 'readout-label', label);
      if (title) lab.title = title;
      const val = el('span', 'readout-value', DASH);
      const cell = el('div', `ro${wide ? ' wide' : ''}`);
      cell.append(lab, val);
      grid.append(cell);
      this.vals.set(key, val);
      this.base.set(key, 'readout-value');
    };
    add('t', 't [s]');
    add('orbit', 'orbits');
    add('triad', 'TRIAD init [s]');
    add('nadir', 'nadir err [°]', 'Nadir pointing error');
    add('estNadir', 'est. nadir [°]', 'Estimated nadir pointing error');
    add('know', 'knowledge [°]', 'Attitude knowledge error');
    add('sigma3', '3σ [°]', '3 × the filter attitude sigma');
    add('rate', 'rate [°/s]', 'Body rate |ω|');
    add('eclipse', 'eclipse [% run]', 'Share of the run spent in eclipse');
    add('wheelFrac', 'wheels [% max]', 'Wheel speed as a share of the maximum');
    add('w', 'ω̂ est. [°/s]', 'Estimated body rate', true);
    add('wheel', 'wheels [rpm]', undefined, true);
    add('m', 'dipole m [A·m²]', undefined, true);
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
  private info: Info | null = null;
  private w = 0;
  private h = 0;
  private dpr = 1;
  private last: { t: number; segments: ModeSegment[]; events: FrameEvents | null } = { t: 0, segments: [], events: null };

  constructor(root: HTMLElement) {
    root.append(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    new ResizeObserver(() => {
      this.dpr = window.devicePixelRatio || 1;
      this.w = Math.max(1, Math.floor(root.clientWidth));
      this.h = Math.max(1, Math.floor(root.clientHeight));
      this.canvas.width = Math.round(this.w * this.dpr);
      this.canvas.height = Math.round(this.h * this.dpr);
      this.draw();
    }).observe(root);
  }

  setInfo(info: Info | null): void {
    this.info = info;
    this.last = { t: 0, segments: [], events: null };
    this.draw();
  }

  update(t: number, segments: ModeSegment[], events: FrameEvents | null): void {
    this.last = { t, segments, events };
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
    const x0 = 12;
    const x1 = this.w - 42; // room at the right for the unit label after the last tick
    const bandTop = 18;
    const bandBot = Math.max(bandTop + 20, this.h - 16);
    const info = this.info;

    // legend row
    let lx = x0;
    const swatch = (color: string, label: string) => {
      g.fillStyle = color;
      g.fillRect(lx, 4, 9, 9);
      g.fillStyle = c.dim;
      g.textAlign = 'left';
      g.fillText(label, lx + 13, 9);
      lx += 13 + g.measureText(label).width + 14;
    };
    swatch(c.eclipse, 'eclipse');
    swatch(d.detumble, 'detumble');
    swatch(d.pointing, 'pointing');
    swatch(d.estimate, 'TRIAD init');
    swatch(c.marker, 'switch');
    swatch(d.danger, 'fallback');
    swatch(d.kick, 'kick');

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

    // orbit ticks
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
        g.fillStyle = d.danger;
        g.beginPath();
        g.moveTo(x - 4, markBot);
        g.lineTo(x + 4, markBot);
        g.lineTo(x, markBot - 6);
        g.fill();
      }
      for (const k of ev.kicks) {
        const x = vline(k.t, d.kick);
        g.fillStyle = d.kick;
        g.beginPath();
        g.arc(x, markTop + 3, 3, 0, 2 * Math.PI);
        g.fill();
      }
      if (ev.t_init !== null) {
        const x = vline(ev.t_init, d.estimate);
        g.fillStyle = d.estimate;
        g.beginPath();
        g.moveTo(x, markTop + 1);
        g.lineTo(x + 4, markTop + 5);
        g.lineTo(x, markTop + 9);
        g.lineTo(x - 4, markTop + 5);
        g.fill();
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

// ===================================================================== plots

interface SeriesSpec {
  label: string;
  col: number; // index into History.columns()
  color: keyof DataColors; // role; resolved against the active mode when the chart is built
  dash?: number[];
  fmt: (v: number) => string;
}

interface PlotSpec {
  title: string;
  log: boolean;
  series: SeriesSpec[];
  thresholds?: boolean;
}

const LOG_FLOOR = 1e-3;

const PLOTS: PlotSpec[] = [
  {
    title: 'Nadir pointing error [°]',
    log: true,
    series: [
      { label: 'true', col: 1, color: 'nadir', fmt: fmtAngle },
      { label: 'estimated', col: 2, color: 'estimate', dash: [6, 4], fmt: fmtAngle },
    ],
  },
  {
    title: 'Knowledge error and 3σ [°]',
    log: true,
    series: [
      { label: 'knowledge error', col: 3, color: 'estimate', fmt: fmtAngle },
      { label: '3σ', col: 4, color: 'sigma3', dash: [6, 4], fmt: fmtAngle },
    ],
  },
  {
    title: 'Body rate [°/s]',
    log: true,
    thresholds: true,
    series: [{ label: '|ω|', col: 5, color: 'nadir', fmt: (v) => fmtFixed(v, 4) }],
  },
  {
    title: 'Wheel speeds [rpm]',
    log: false,
    series: [
      { label: 'x', col: 6, color: 'axisX', fmt: (v) => v.toFixed(0) },
      { label: 'y', col: 7, color: 'axisY', fmt: (v) => v.toFixed(0) },
      { label: 'z', col: 8, color: 'axisZ', fmt: (v) => v.toFixed(0) },
    ],
  },
];

function tickLabel(v: number): string {
  const a = Math.abs(v);
  if (a >= 1000) return `${+(v / 1000).toFixed(1)}k`;
  return String(+v.toFixed(2));
}

class PlotStrip {
  private readonly host = el('div', 'plot-host');
  private u: uPlot | null = null;
  private info: Info | null = null;
  private lastCols: Columns | null = null;
  private readonly spec: PlotSpec;

  constructor(root: HTMLElement, spec: PlotSpec) {
    this.spec = spec;
    const head = el('div', 'card-head');
    head.append(el('h2', undefined, spec.title));
    root.append(head, this.host);
    this.build();
    new ResizeObserver(() => this.resize()).observe(this.host);
  }

  /** Mode change: rebuild the chart with the new axis / grid / series colours and refill it from the last data. */
  rebuild(): void {
    this.u?.destroy();
    this.u = null;
    this.build();
    if (this.lastCols) this.setData(this.lastCols);
  }

  redraw(): void {
    this.u?.redraw();
  }

  setInfo(info: Info | null): void {
    this.info = info;
    this.u?.redraw();
  }

  private thresholds(): { v: number; color: string; label: string }[] {
    if (!this.spec.thresholds || !this.info) return [];
    const d = dataColors();
    return [
      { v: this.info.detumble_threshold_deg_s, color: d.detumble, label: `detumble ${this.info.detumble_threshold_deg_s} °/s` },
      { v: this.info.safe_rate_deg_s, color: d.danger, label: `safe rate ${this.info.safe_rate_deg_s} °/s` },
    ];
  }

  private build(): void {
    const spec = this.spec;
    const c = chromeColors();
    const d = dataColors();
    // axis text dim, grid a faint tint of the card's block colour, ticks hidden (spec §3)
    const axisBase = { stroke: c.dim, grid: { stroke: c.grid, width: 1 }, ticks: { show: false }, font: `12px ${FONT_FAMILY}` };
    const opts: uPlot.Options = {
      width: Math.max(100, this.host.clientWidth),
      height: Math.max(80, this.host.clientHeight - 22),
      padding: [8, 14, 0, 0],
      legend: { show: true, live: true },
      cursor: { sync: { key: 'adcs', setSeries: false }, drag: { x: false, y: false }, points: { show: false } },
      scales: {
        x: {
          time: false,
          range: (_u, min, max) => timeRange(min, max, this.info?.t_end_s ?? 0),
        },
        y: spec.log
          ? {
              distr: 3,
              log: 10,
              range: (_u, min, max) => logRange(min, max, LOG_FLOOR, this.thresholds().map((x) => x.v)),
            }
          : {
              range: (_u, min, max) => {
                if (!Number.isFinite(min) || !Number.isFinite(max)) return [-1000, 1000];
                const m = Math.max(Math.abs(min), Math.abs(max), 100);
                return [-m * 1.08, m * 1.08];
              },
            },
      },
      axes: [
        { ...axisBase, size: 24, values: (_u, splits) => splits.map(tickLabel) },
        { ...axisBase, size: 54, values: (_u, splits) => splits.map((v) => (v === null ? '' : tickLabel(v))) },
      ],
      series: [
        { label: 't [s]', value: (_u, v) => (v === null || v === undefined ? DASH : `${v.toFixed(0)} s`) },
        ...spec.series.map<uPlot.Series>((s) => ({
          label: s.label,
          stroke: d[s.color],
          width: 2,
          dash: s.dash,
          spanGaps: false,
          points: { show: false },
          value: (_u, v) => (v === null || v === undefined ? DASH : s.fmt(v)),
        })),
      ],
      hooks: {
        draw: [
          (u) => {
            const th = this.thresholds();
            if (!th.length) return;
            const ctx = u.ctx;
            const dpr = window.devicePixelRatio || 1;
            ctx.save();
            ctx.beginPath();
            ctx.rect(u.bbox.left, u.bbox.top, u.bbox.width, u.bbox.height);
            ctx.clip();
            ctx.setLineDash([6 * dpr, 4 * dpr]);
            ctx.lineWidth = dpr;
            ctx.font = `${11 * dpr}px ${FONT_FAMILY}`;
            ctx.textBaseline = 'bottom';
            ctx.textAlign = 'left';
            for (const t of th) {
              const y = u.valToPos(t.v, 'y', true);
              ctx.strokeStyle = t.color;
              ctx.fillStyle = t.color;
              ctx.beginPath();
              ctx.moveTo(u.bbox.left, y);
              ctx.lineTo(u.bbox.left + u.bbox.width, y);
              ctx.stroke();
              ctx.fillText(t.label, u.bbox.left + 6 * dpr, y - 2 * dpr);
            }
            ctx.restore();
          },
        ],
      },
    };
    const empty: uPlot.AlignedData = [[], ...spec.series.map(() => [])] as unknown as uPlot.AlignedData;
    this.u = new uPlot(opts, empty, this.host);
  }

  private resize(): void {
    if (!this.u) return;
    const w = Math.floor(this.host.clientWidth);
    const h = Math.floor(this.host.clientHeight) - 22; // legend row
    if (w > 20 && h > 40) this.u.setSize({ width: w, height: h });
  }

  setData(cols: Columns): void {
    this.lastCols = cols;
    if (!this.u) return;
    const t = cols[0] as number[];
    const rest = this.spec.series.map((s) => {
      const c = cols[s.col];
      return this.spec.log ? c.map((v) => clampLog(v, LOG_FLOOR)) : c;
    });
    this.u.setData([t, ...rest] as unknown as uPlot.AlignedData, true);
  }
}

// ===================================================================== composite

export class Panels {
  private readonly form: FormPanel;
  private readonly transport: TransportPanel;
  private readonly kick: KickPanel;
  private readonly readouts: Readouts;
  private readonly timeline: Timeline;
  private readonly plots: PlotStrip[];
  private readonly banner: HTMLElement;
  private readonly toasts: HTMLElement;
  private info: Info | null = null;

  constructor(roots: PanelRoots, handlers: PanelHandlers) {
    this.banner = roots.banner;
    this.toasts = roots.toasts;
    this.form = new FormPanel(roots.form, handlers);
    this.transport = new TransportPanel(roots.transport, handlers);
    this.kick = new KickPanel(roots.kick, handlers, (m) => this.toast(m));
    this.readouts = new Readouts(roots.readouts);
    this.timeline = new Timeline(roots.timeline);
    this.plots = roots.plots.map((r, i) => new PlotStrip(r, PLOTS[i]));
    // Mode change: charts are rebuilt with the new palette from the stored data (no reload, no lost history).
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
    this.kick.setEnabled(!status.done);
    this.readouts.clear();
    this.readouts.setInfo(info);
    this.timeline.setInfo(info);
    for (const p of this.plots) p.setInfo(info);
  }

  onClosed(): void {
    this.transport.setConnected(false);
    this.kick.setEnabled(false);
  }

  onFrame(f: Frame): void {
    this.readouts.update(f);
    this.transport.setState({ playing: f.playing, done: f.done, speedSet: f.speed_set, speedActual: f.speed_actual });
    this.kick.setEnabled(!f.done);
  }

  /** Throttled by the caller (about 8 Hz): timeline and plot strips. */
  refreshCharts(cols: Columns, t: number, segments: ModeSegment[], events: FrameEvents | null): void {
    this.timeline.update(t, segments, events);
    for (const p of this.plots) p.setData(cols);
  }

  clearCharts(): void {
    this.timeline.update(0, [], null);
    const empty: Columns = Array.from({ length: 9 }, () => []);
    for (const p of this.plots) p.setData(empty);
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
