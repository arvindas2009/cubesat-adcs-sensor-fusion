// The four ADCS plots (docs/specs/frontend.md §2, docs/specs/ui_console.md §5). `AdcsChart` is one uPlot
// chart in either variant: the card on the sheet (cursor synced with its siblings) or the large chart of the
// expand dialog (drag to zoom in x, double-click to reset). The legend is a bound DOM legend, not uPlot's.
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import type { Info } from '../types';
import { DASH, fmtAngle, fmtFixed } from './format';
import type { Columns } from './history';
import { clampLog, logRange, timeRange } from './logic';
import { FONT_UI, dataColors, type DataColors } from '../theme';
import { axisSet, bindLegend, mountLegend, observeHost, type ChartVariant, type LiveChart } from '../ui/chart';
import type { ChartId } from '../ui/glossary';
import { seriesGlyph, type LegendItem } from '../ui/glyphs';
import { renderLegend, type LegendHandle } from '../ui/legend';
import { chartSummary } from '../ui/a11y';
import { latestValue } from '../ui/sample';
import { XZoom } from '../ui/zoom';

export interface SeriesSpec {
  label: string;
  col: number; // index into History.columns()
  color: keyof DataColors; // role; resolved against the active mode when the chart is built
  dash?: number[];
  fmt: (v: number) => string;
}

export interface PlotSpec {
  id: ChartId;
  title: string;
  unit: string;
  log: boolean;
  series: SeriesSpec[];
  thresholds?: boolean;
}

export const LOG_FLOOR = 1e-3;

export const PLOTS: PlotSpec[] = [
  {
    id: 'adcs.nadir',
    title: 'Nadir pointing error',
    unit: '°, log scale',
    log: true,
    series: [
      { label: 'true', col: 1, color: 'nadir', fmt: fmtAngle },
      { label: 'estimated', col: 2, color: 'estimate', dash: [6, 4], fmt: fmtAngle },
    ],
  },
  {
    id: 'adcs.knowledge',
    title: 'Knowledge error and 3σ',
    unit: '°, log scale',
    log: true,
    series: [
      { label: 'knowledge error', col: 3, color: 'estimate', fmt: fmtAngle },
      { label: '3σ', col: 4, color: 'sigma3', dash: [6, 4], fmt: fmtAngle },
    ],
  },
  {
    id: 'adcs.rate',
    title: 'Body rate',
    unit: '°/s, log scale',
    log: true,
    thresholds: true,
    series: [{ label: '|ω|', col: 5, color: 'nadir', fmt: (v) => fmtFixed(v, 4) }],
  },
  {
    id: 'adcs.wheels',
    title: 'Wheel speeds',
    unit: 'rpm',
    log: false,
    series: [
      { label: 'x', col: 6, color: 'axisX', fmt: (v) => v.toFixed(0) },
      { label: 'y', col: 7, color: 'axisY', fmt: (v) => v.toFixed(0) },
      { label: 'z', col: 8, color: 'axisZ', fmt: (v) => v.toFixed(0) },
    ],
  },
];

export function tickLabel(v: number): string {
  const a = Math.abs(v);
  if (a >= 1000) return `${+(v / 1000).toFixed(1)}k`;
  return String(+v.toFixed(2));
}

export interface AdcsChartOptions {
  variant: ChartVariant;
  /** The session info (thresholds, run length); null before the first `hello`. */
  info: () => Info | null;
  /** Where the bound legend row goes (none: no legend). */
  legendHost?: HTMLElement;
}

interface Threshold {
  v: number;
  color: string;
  label: string;
}

export class AdcsChart implements LiveChart {
  private u: uPlot | null = null;
  private info: Info | null;
  private lastCols: Columns | null = null;
  private lastData: uPlot.AlignedData | null = null;
  private readonly zoom = new XZoom();
  private zoomCb: ((zoomed: boolean) => void) | null = null;
  private readonly hidden = new Set<number>(); // uPlot series indices switched off in the legend
  private readonly legend: LegendHandle | null;
  private readonly timeEl: HTMLElement | null;
  private readonly ro: ResizeObserver;

  constructor(
    private readonly host: HTMLElement,
    private readonly spec: PlotSpec,
    private readonly opts: AdcsChartOptions,
  ) {
    this.info = opts.info();
    if (opts.legendHost) {
      this.legend = renderLegend(this.legendItems(), {
        surface: 'panel',
        live: true,
        toggle: (id, on) => {
          const i = Number(id.slice(1));
          if (on) this.hidden.delete(i);
          else this.hidden.add(i);
          this.u?.setSeries(i, { show: on });
        },
      });
      this.timeEl = mountLegend(opts.legendHost, this.legend);
    } else {
      this.legend = null;
      this.timeEl = null;
    }
    this.build();
    this.ro = observeHost(host, () => this.resize());
  }

  private get large(): boolean {
    return this.opts.variant === 'large';
  }

  // ------------------------------------------------------------------ legend

  private thresholds(): Threshold[] {
    if (!this.spec.thresholds || !this.info) return [];
    const d = dataColors();
    return [
      { v: this.info.detumble_threshold_deg_s, color: d.detumble, label: `detumble ${this.info.detumble_threshold_deg_s} °/s` },
      { v: this.info.safe_rate_deg_s, color: d.danger, label: `safe rate ${this.info.safe_rate_deg_s} °/s` },
    ];
  }

  private legendItems(): LegendItem[] {
    const d = dataColors();
    const items: LegendItem[] = this.spec.series.map((s, k) => ({
      id: `s${k + 1}`,
      label: s.label,
      color: d[s.color],
      glyph: seriesGlyph({ dash: s.dash }),
    }));
    if (this.spec.thresholds) {
      items.push(
        { id: 'th-detumble', label: 'detumble threshold', color: d.detumble, glyph: { kind: 'line', dash: true }, help: 'detumble', static: true },
        { id: 'th-safe', label: 'safe rate', color: d.danger, glyph: { kind: 'line', dash: true }, static: true },
      );
    }
    return items;
  }

  private syncThresholdValues(): void {
    const th = this.thresholds();
    this.legend?.setValue('th-detumble', th[0] ? `${th[0].v} °/s` : DASH);
    this.legend?.setValue('th-safe', th[1] ? `${th[1].v} °/s` : DASH);
  }

  // ------------------------------------------------------------------ build

  private size(): { width: number; height: number } {
    return { width: Math.max(100, Math.floor(this.host.clientWidth)), height: Math.max(80, Math.floor(this.host.clientHeight)) };
  }

  private build(): void {
    const spec = this.spec;
    const d = dataColors();
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
        : { sync: { key: 'adcs', setSeries: false }, drag: { x: false, y: false }, points: { show: false } },
      scales: {
        x: {
          time: false,
          range: (_u, min, max) => {
            const def = timeRange(min, max, this.info?.t_end_s ?? 0);
            return large ? zoom.range(def) : def;
          },
        },
        y: spec.log
          ? { distr: 3, log: 10, range: (_u, min, max) => logRange(min, max, LOG_FLOOR, this.thresholds().map((x) => x.v)) }
          : {
              range: (_u, min, max) => {
                if (!Number.isFinite(min) || !Number.isFinite(max)) return [-1000, 1000];
                const m = Math.max(Math.abs(min), Math.abs(max), 100);
                return [-m * 1.08, m * 1.08];
              },
            },
      },
      axes: [ax, ay],
      series: [
        { label: 't [s]', value: (_u, v) => (v === null || v === undefined ? DASH : `${v.toFixed(0)} s`) },
        ...spec.series.map<uPlot.Series>((s, k) => ({
          label: s.label,
          show: !this.hidden.has(k + 1),
          stroke: d[s.color],
          width: 2,
          dash: s.dash,
          spanGaps: false,
          points: { show: false },
          value: (_u, v) => (v === null || v === undefined ? DASH : s.fmt(v)),
        })),
      ],
      hooks: {
        draw: [(u) => this.drawThresholds(u)],
        setSelect: [(u) => this.onSelect(u)],
      },
    };
    const empty: uPlot.AlignedData = [[], ...spec.series.map(() => [])] as unknown as uPlot.AlignedData;
    const u = new uPlot(opts, this.lastData ?? empty, this.host);
    this.u = u;
    if (this.legend) {
      this.legend.setColors(this.legendItems());
      bindLegend(u, this.legend, (i, v) => (v === null ? DASH : spec.series[i - 1].fmt(v)), {
        time: this.timeEl ? { el: this.timeEl, fmt: (t) => `${t.toFixed(0)} s` } : undefined,
      });
      this.syncThresholdValues();
    }
  }

  /** Dashed threshold lines of the body-rate plot, labelled in the plot. */
  private drawThresholds(u: uPlot): void {
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
    ctx.font = `${12 * dpr}px ${FONT_UI}`;
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
  }

  /** Drag selection (large variant): remember the zoomed window and clear the selection box. */
  private onSelect(u: uPlot): void {
    if (!this.large) return;
    const s = u.select;
    if (!(s.width > 0)) return;
    const xs = u.data[0] as ArrayLike<number>;
    if (!xs.length) return;
    const lo = xs[0];
    const hi = xs[xs.length - 1];
    this.zoom.select(u.posToVal(s.left, 'x'), u.posToVal(s.left + s.width, 'x'), lo, hi, Math.max((hi - lo) * 0.002, 1e-3));
    u.setSelect({ left: 0, width: 0, top: 0, height: 0 }, false);
    this.zoomCb?.(true);
  }

  // ------------------------------------------------------------------ public

  setInfo(info: Info | null): void {
    this.info = info;
    this.syncThresholdValues();
    this.u?.redraw();
  }

  /** New columns (History.columns()). Cards rescale every time; the dialog keeps a zoomed window. */
  setData(cols: Columns): void {
    this.lastCols = cols;
    const t = cols[0] as number[];
    const rest = this.spec.series.map((s) => {
      const c = cols[s.col];
      return this.spec.log ? c.map((v) => clampLog(v, LOG_FLOOR)) : c;
    });
    const data = [t, ...rest] as unknown as uPlot.AlignedData;
    this.lastData = data;
    this.u?.setData(data, this.large ? !this.zoom.active : true);
  }

  /** Text alternative: the title and the latest value of every series. */
  summaryText(): string {
    const cols = this.lastCols;
    const latest = this.spec.series.map((s) => {
      const v = cols ? latestValue(cols[s.col]) : null;
      return v === null ? null : s.fmt(v);
    });
    return chartSummary(`${this.spec.title} (${this.spec.unit})`, this.spec.series.map((s) => s.label), latest);
  }

  resetZoom(): void {
    if (!this.zoom.active) return;
    this.zoom.reset();
    if (this.u && this.lastData) this.u.setData(this.lastData, true);
    this.zoomCb?.(false);
  }

  onZoom(cb: (zoomed: boolean) => void): void {
    this.zoomCb = cb;
  }

  rebuild(): void {
    this.u?.destroy();
    this.u = null;
    this.build();
    if (this.lastCols) this.setData(this.lastCols);
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
    if (width > 20 && height > 40) this.u.setSize({ width, height });
  }

  destroy(): void {
    this.ro.disconnect();
    this.u?.destroy();
    this.u = null;
    this.zoomCb = null;
    this.opts.legendHost?.replaceChildren();
    this.host.replaceChildren();
  }
}
