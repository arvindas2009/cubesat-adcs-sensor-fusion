// The four PX4 replay plots (docs/specs/px4_replay.md §3.3, docs/specs/ui_console.md §5). `ReplayChart` is
// one uPlot chart in either variant: the card on the sheet (click or drag seeks, cursor synced with its
// siblings) or the large chart of the expand dialog (drag zooms in x, a click without movement seeks).
// The model data of a replay is computed once per replay and shared; series styles are per theme.
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import { DASH, fmtFixed } from '../adcs/format';
import { clampLog, logRange } from '../adcs/logic';
import { FONT_UI, chromeColors, currentTheme, dataColors, nisFallback, type DataColors } from '../theme';
import { axisSet, bindLegend, mountLegend, observeHost, type ChartVariant, type LiveChart } from '../ui/chart';
import type { ChartId } from '../ui/glossary';
import { seriesGlyph, type LegendItem } from '../ui/glyphs';
import { renderLegend, type LegendHandle } from '../ui/legend';
import { chartSummary } from '../ui/a11y';
import { legendValue } from '../ui/sample';
import { XZoom } from '../ui/zoom';
import { alignColumns, diffColumns, heightColumns, nisBands, nisSeries } from './series';
import type { NisBand } from './series';
import { mergeBandTags } from './tags';
import type { Replay } from './types';

export const NIS_FLOOR = 1e-3;
const CLICK_PX = 4; // the large chart: less movement than this between pointer down and up is a seek

export function tickLabel(v: number): string {
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

export function nisColor(sensor: string, index: number): string {
  const role = NIS_ROLES[sensor];
  if (role) return dataColors()[role];
  const fb = nisFallback(currentTheme());
  return fb[index % fb.length];
}

/** Theme-independent description of one series; the colour comes from `role`, or from the NIS sensor. */
interface SeriesDef {
  label: string;
  role?: keyof DataColors;
  nis?: { sensor: string; index: number };
  dash?: number[];
  /** Markers only, no connecting line. */
  points?: boolean;
  hollow?: boolean;
  /** The next series is switched together with this one (a NIS sensor's rejected updates). */
  pairNext?: boolean;
  fmt: (v: number) => string;
  /** Has a legend item (the rejected series ride on their sensor's item). */
  legend: boolean;
}

/** Everything about a plot that does not depend on the theme. Computed once per replay. */
interface PlotModel {
  data: uPlot.AlignedData;
  defs: SeriesDef[];
  log: boolean;
  bands: NisBand[];
}

const f3 = (v: number): string => fmtFixed(v, 3);
const f2 = (v: number): string => fmtFixed(v, 2);

function line(label: string, role: keyof DataColors, dash?: number[]): SeriesDef {
  return { label, role, dash, fmt: f3, legend: true };
}

const BUILDERS: ((rep: Replay) => PlotModel)[] = [
  (rep) => ({
    data: diffColumns(rep).att as unknown as uPlot.AlignedData,
    defs: [line('tilt', 'nadir'), line('yaw (signed)', 'field')],
    log: false,
    bands: [],
  }),
  (rep) => ({
    data: diffColumns(rep).pos as unknown as uPlot.AlignedData,
    defs: [line('horizontal', 'nadir'), line('vertical (+ = ours lower)', 'field')],
    log: false,
    bands: [],
  }),
  (rep) => {
    const sensors = nisSeries(rep);
    const cols = alignColumns(sensors.flatMap((s) => [s.accepted, s.rejected]));
    const data = [cols[0], ...cols.slice(1).map((c) => c.map((v) => clampLog(v, NIS_FLOOR)))];
    const defs: SeriesDef[] = [];
    sensors.forEach((s, i) => {
      defs.push({ label: s.sensor, nis: { sensor: s.sensor, index: i }, points: true, pairNext: true, fmt: f2, legend: true });
      defs.push({ label: `${s.sensor} (rejected)`, nis: { sensor: s.sensor, index: i }, points: true, hollow: true, fmt: f2, legend: false });
    });
    return { data: data as unknown as uPlot.AlignedData, defs, log: true, bands: nisBands(rep) };
  },
  (rep) => ({
    data: heightColumns(rep) as unknown as uPlot.AlignedData,
    defs: [line('ours', 'ours'), line('EKF2 (reference)', 'ekf2', [6, 4])],
    log: false,
    bands: [],
  }),
];

const MODEL_CACHE = new WeakMap<Replay, PlotModel[]>();

function modelFor(rep: Replay, idx: number): PlotModel {
  let list = MODEL_CACHE.get(rep);
  if (!list) {
    list = [];
    MODEL_CACHE.set(rep, list);
  }
  return (list[idx] ??= BUILDERS[idx](rep));
}

export const PLOT_HEADS: { id: ChartId; title: string; unit: string }[] = [
  { id: 'px4.attitude', title: 'Attitude difference, ours − EKF2', unit: '°' },
  { id: 'px4.position', title: 'Position difference, ours − EKF2', unit: 'm' },
  { id: 'px4.nis', title: 'NIS by sensor', unit: 'log scale' },
  { id: 'px4.height', title: 'Height above start', unit: 'm' },
];

function defColor(def: SeriesDef): string {
  if (def.nis) return nisColor(def.nis.sensor, def.nis.index);
  return dataColors()[def.role ?? 'nadir'];
}

function styleSeries(def: SeriesDef, color: string, show: boolean): uPlot.Series {
  const value = (_u: uPlot, v: number | null | undefined): string => (v === null || v === undefined ? DASH : def.fmt(v));
  if (def.points) {
    return {
      label: def.label,
      show,
      paths: () => null, // markers only, no connecting line
      points: def.hollow
        ? { show: true, size: 5, width: 1.5, stroke: color, fill: 'transparent' }
        : { show: true, size: 4, width: 0, stroke: color, fill: color },
      value,
    };
  }
  return { label: def.label, show, stroke: color, width: 2, dash: def.dash, spanGaps: false, points: { show: false }, value };
}

export interface ReplayChartOptions {
  variant: ChartVariant;
  /** Where the bound legend row goes (none: no legend). */
  legendHost?: HTMLElement;
}

export class ReplayChart implements LiveChart {
  private u: uPlot | null = null;
  private nowLine: HTMLElement | null = null;
  private lastX = NaN;
  private lastT = 0;
  private rep: Replay | null = null;
  private model: PlotModel | null = null;
  private readonly zoom = new XZoom();
  private zoomCb: ((zoomed: boolean) => void) | null = null;
  private readonly hidden = new Set<number>();
  private legend: LegendHandle | null = null;
  private timeEl: HTMLElement | null = null;
  private legendBinding: { update(): void } | null = null;
  private readonly ro: ResizeObserver;

  constructor(
    private readonly host: HTMLElement,
    private readonly idx: number,
    private readonly onSeek: (t: number) => void,
    private readonly opts: ReplayChartOptions,
  ) {
    this.ro = observeHost(host, () => this.resize());
  }

  private get large(): boolean {
    return this.opts.variant === 'large';
  }

  // ------------------------------------------------------------------ legend

  private legendItems(m: PlotModel): LegendItem[] {
    const items: LegendItem[] = [];
    m.defs.forEach((def, k) => {
      if (!def.legend) return;
      items.push({
        id: `s${k + 1}`,
        label: def.label,
        color: defColor(def),
        glyph: def.points ? { kind: 'marker', shape: 'circle' } : seriesGlyph({ dash: def.dash }),
      });
    });
    if (m.log && m.bands.length) {
      items.push(
        { id: 'rejected', label: 'rejected', color: chromeColors().dim, glyph: { kind: 'marker', shape: 'circle', hollow: true }, static: true },
        { id: 'band', label: '95 % band', color: chromeColors().dim, glyph: { kind: 'line', dash: true }, help: 'nisBand', static: true },
      );
    }
    return items;
  }

  private ensureLegend(m: PlotModel): void {
    const host = this.opts.legendHost;
    if (!host) return;
    if (!this.legend) {
      this.legend = renderLegend(this.legendItems(m), {
        surface: 'panel',
        live: true,
        toggle: (id, on) => {
          const i = Number(id.slice(1));
          const set = (k: number): void => {
            if (on) this.hidden.delete(k);
            else this.hidden.add(k);
            this.u?.setSeries(k, { show: on });
          };
          set(i);
          if (this.model?.defs[i - 1]?.pairNext) set(i + 1);
        },
      });
      this.timeEl = mountLegend(host, this.legend);
    } else {
      this.legend.setColors(this.legendItems(m));
    }
  }

  // ------------------------------------------------------------------ build

  private size(): { width: number; height: number } {
    return { width: Math.max(100, Math.floor(this.host.clientWidth)), height: Math.max(80, Math.floor(this.host.clientHeight)) };
  }

  /** Mode change: rebuild from the loaded replay with the new palette, keeping the "now" line where it was. */
  rebuild(): void {
    if (!this.rep) return;
    this.setReplay(this.rep);
    this.setTime(this.lastT);
  }

  /** (Re)build the plot for a replay; null just destroys the old one. */
  setReplay(rep: Replay | null): void {
    if (rep !== this.rep) {
      this.zoom.reset();
      this.zoomCb?.(false);
      this.hidden.clear();
      if (this.legend) {
        this.legend = null;
        this.opts.legendHost?.replaceChildren();
      }
    }
    this.u?.destroy();
    this.u = null;
    this.nowLine = null;
    this.legendBinding = null;
    this.lastX = NaN;
    this.rep = rep;
    this.model = null;
    if (!rep || rep.t.length < 2) return;
    const m = modelFor(rep, this.idx);
    this.model = m;
    const t0 = rep.t[0];
    const t1 = rep.t[rep.t.length - 1];
    const c = chromeColors();
    const resetStroke = dataColors().kick;
    const resets = rep.events.ekf2_resets_s;
    const zoom = this.zoom;
    const large = this.large;
    const [ax, ay] = axisSet(this.opts.variant, tickLabel);
    const opts: uPlot.Options = {
      ...this.size(),
      padding: [8, 14, 0, 0],
      legend: { show: false },
      cursor: large
        ? {
            drag: { x: true, y: false, dist: 6, setScale: true },
            points: { show: false },
            bind: {
              dblclick: () => () => {
                this.resetZoom();
                return null;
              },
            },
          }
        : { sync: { key: 'px4', setSeries: false }, drag: { x: false, y: false }, points: { show: false } },
      scales: {
        x: { time: false, range: () => (large ? zoom.range([t0, t1]) : [t0, t1]) },
        y: m.log ? { distr: 3, log: 10, range: (_u, min, max) => logRange(min, max, NIS_FLOOR, m.bands.flatMap((b) => [b.lo, b.hi])) } : {},
      },
      axes: [ax, ay],
      series: [
        { label: 't [s]', value: (_u, v) => (v === null || v === undefined ? DASH : `${v.toFixed(2)} s`) },
        ...m.defs.map((def, k) => styleSeries(def, defColor(def), !this.hidden.has(k + 1))),
      ],
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
            // chi-square 95 % bands (NIS plot): dashed lines, one small "dof N" tag per line group
            if (m.bands.length) {
              ctx.setLineDash([6 * dpr, 4 * dpr]);
              ctx.strokeStyle = c.dim;
              ctx.lineWidth = dpr;
              for (const b of m.bands) {
                for (const v of [b.lo, b.hi]) {
                  const y = u.valToPos(Math.max(v, NIS_FLOOR), 'y', true);
                  ctx.beginPath();
                  ctx.moveTo(u.bbox.left, y);
                  ctx.lineTo(u.bbox.left + u.bbox.width, y);
                  ctx.stroke();
                }
              }
              ctx.setLineDash([]);
              ctx.font = `${12 * dpr}px ${FONT_UI}`;
              ctx.textAlign = 'right';
              ctx.textBaseline = 'middle';
              const xr = u.bbox.left + u.bbox.width - 4 * dpr;
              const rowH = 14 * dpr;
              // tags within 14 css px of each other share one tag, so they never overlap
              const tags = mergeBandTags(
                m.bands.map((b) => ({ y: u.valToPos(Math.max(b.hi, NIS_FLOOR), 'y'), dof: b.dof })),
                14,
              );
              for (const tag of tags) {
                const y = Math.max(u.bbox.top + rowH / 2, u.bbox.top + tag.y * dpr - 9 * dpr);
                const w = ctx.measureText(tag.text).width;
                ctx.globalAlpha = 0.9;
                ctx.fillStyle = c.card;
                ctx.fillRect(xr - w - 3 * dpr, y - rowH / 2, w + 6 * dpr, rowH);
                ctx.globalAlpha = 1;
                ctx.fillStyle = c.dim;
                ctx.fillText(tag.text, xr, y);
              }
            }
            ctx.restore();
          },
        ],
        setScale: [
          (_u, key) => {
            if (key === 'x') {
              this.lastX = NaN;
              this.setTime(this.lastT);
            }
          },
        ],
        setSelect: [(u) => this.onSelect(u, t0, t1)],
        ready: [(u) => this.installPointer(u, t0, t1)],
      },
    };
    const u = new uPlot(opts, m.data, this.host);
    this.u = u;
    if (this.opts.legendHost) {
      this.ensureLegend(m);
      if (this.legend) {
        this.legendBinding = bindLegend(u, this.legend, (i, v) => (v === null ? DASH : m.defs[i - 1].fmt(v)), {
          restX: () => this.lastT,
          time: this.timeEl ? { el: this.timeEl, fmt: (t) => `${t.toFixed(2)} s` } : undefined,
        });
      }
    }
  }

  /** The moving "now" line and the pointer behaviour: card = click or drag seeks; large = a click seeks, a drag zooms. */
  private installPointer(u: uPlot, t0: number, t1: number): void {
    const line = document.createElement('div');
    line.className = 'plot-now';
    u.over.append(line);
    this.nowLine = line;
    const seekAt = (e: PointerEvent): void => {
      const x = e.clientX - u.over.getBoundingClientRect().left;
      this.onSeek(Math.min(t1, Math.max(t0, u.posToVal(x, 'x'))));
    };
    if (!this.large) {
      let down = false;
      u.over.addEventListener('pointerdown', (e) => {
        down = true;
        u.over.setPointerCapture(e.pointerId);
        seekAt(e);
      });
      u.over.addEventListener('pointermove', (e) => {
        if (down) seekAt(e);
      });
      const end = (): void => {
        down = false;
      };
      u.over.addEventListener('pointerup', end);
      u.over.addEventListener('pointercancel', end);
    } else {
      let x0 = NaN;
      u.over.addEventListener('pointerdown', (e) => {
        x0 = e.clientX;
      });
      u.over.addEventListener('pointerup', (e) => {
        if (Number.isFinite(x0) && Math.abs(e.clientX - x0) < CLICK_PX) seekAt(e);
        x0 = NaN;
      });
    }
  }

  private onSelect(u: uPlot, t0: number, t1: number): void {
    if (!this.large) return;
    const s = u.select;
    if (!(s.width > 0)) return;
    this.zoom.select(u.posToVal(s.left, 'x'), u.posToVal(s.left + s.width, 'x'), t0, t1, Math.max((t1 - t0) * 0.002, 1e-3));
    u.setSelect({ left: 0, width: 0, top: 0, height: 0 }, false);
    this.zoomCb?.(true);
  }

  // ------------------------------------------------------------------ public

  setTime(t: number): void {
    this.lastT = t;
    this.legendBinding?.update();
    const u = this.u;
    const line = this.nowLine;
    if (!u || !line) return;
    const x = Math.round(u.valToPos(t, 'x'));
    if (x === this.lastX) return;
    this.lastX = x;
    line.style.transform = `translateX(${x}px)`;
    line.style.display = Number.isFinite(x) ? 'block' : 'none';
  }

  /** Text alternative: the title and the latest value of every legend series at the playback time. */
  summaryText(): string {
    const head = PLOT_HEADS[this.idx];
    const u = this.u;
    const m = this.model;
    if (!u || !m) return chartSummary(`${head.title} (${head.unit})`, [], []);
    const xs = u.data[0] as ArrayLike<number>;
    const labels: string[] = [];
    const latest: (string | null)[] = [];
    m.defs.forEach((def, k) => {
      if (!def.legend) return;
      labels.push(def.label);
      const v = legendValue(xs, u.data[k + 1] as ArrayLike<number | null>, null, this.lastT);
      latest.push(v === null ? null : def.fmt(v));
    });
    return chartSummary(`${head.title} (${head.unit})`, labels, latest);
  }

  resetZoom(): void {
    if (!this.zoom.active) return;
    this.zoom.reset();
    const rep = this.rep;
    if (this.u && rep) this.u.setScale('x', { min: rep.t[0], max: rep.t[rep.t.length - 1] }); // through `range`: the default window
    this.zoomCb?.(false);
  }

  onZoom(cb: (zoomed: boolean) => void): void {
    this.zoomCb = cb;
  }

  redraw(): void {
    this.u?.redraw();
  }

  syncRect(): void {
    this.u?.syncRect(true);
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

  destroy(): void {
    this.ro.disconnect();
    this.u?.destroy();
    this.u = null;
    this.nowLine = null;
    this.zoomCb = null;
    this.legendBinding = null;
    this.opts.legendHost?.replaceChildren();
    this.host.replaceChildren();
  }
}
