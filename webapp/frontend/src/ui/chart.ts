// DOM helpers shared by the ADCS and PX4 charts (docs/specs/ui_console.md §5.1): the axis set, the bound DOM
// legend, the plot-card skeleton and the `LiveChart` contract the page registries (`live`) hold.
import uPlot from 'uplot';
import { FONT_MONO, chromeColors } from '../theme';
import type { LegendHandle } from './legend';
import { legendValue } from './sample';

export type ChartVariant = 'card' | 'large';

/** What a page's `live` registry holds: the four cards and the chart of the open dialog. */
export interface LiveChart {
  /** Theme change: rebuild with the new palette from the stored data. */
  rebuild(): void;
  /** Redraw (web fonts arrived). */
  redraw(): void;
  /** uPlot caches its rect: refresh it (sheet settled, page shown). */
  syncRect(): void;
  /** Free the uPlot instance, observers and listeners. */
  destroy(): void;
}

/** x and y axis of a chart: dim mono tick text, faint grid, no tick marks. 12 px on cards, 13 px and wider in the dialog. */
export function axisSet(variant: ChartVariant, tick: (v: number) => string): [uPlot.Axis, uPlot.Axis] {
  const c = chromeColors();
  const large = variant === 'large';
  const base = { stroke: c.dim, grid: { stroke: c.grid, width: 1 }, ticks: { show: false }, font: `${large ? 13 : 12}px ${FONT_MONO}` };
  return [
    { ...base, size: large ? 30 : 24, values: (_u, splits) => splits.map((v) => (v === null ? '' : tick(v))) },
    { ...base, size: large ? 72 : 54, values: (_u, splits) => splits.map((v) => (v === null ? '' : tick(v))) },
  ];
}

export interface LegendBinding {
  /** Recompute all values now. */
  update(): void;
}

/**
 * Drive a DOM legend from a chart (the chart's own legend is off): on every cursor move and data change
 * each series item `s<i>` shows the value at the cursor, or the latest sample when the pointer is away
 * (`restX`, when given, picks "the latest sample at or before x", e.g. a replay's playback time).
 * `time` shows the x of the same sample.
 */
export function bindLegend(
  u: uPlot,
  legend: LegendHandle,
  fmt: (seriesIdx: number, v: number | null) => string,
  opts: { restX?: () => number | null; time?: { el: HTMLElement; fmt: (t: number) => string } } = {},
): LegendBinding {
  const update = (): void => {
    const xs = u.data[0] as ArrayLike<number>;
    if (!xs) return;
    const hover = u.cursor.idx;
    const restX = opts.restX?.() ?? null;
    for (let i = 1; i < u.series.length; i++) {
      legend.setValue(`s${i}`, fmt(i, legendValue(xs, u.data[i] as ArrayLike<number | null>, hover, restX)));
    }
    const tm = opts.time;
    if (tm) {
      let t: number | null = null;
      if (hover !== null && hover !== undefined) t = xs[hover] ?? null;
      else if (restX !== null) t = restX;
      else if (xs.length) t = xs[xs.length - 1];
      const text = t === null || !Number.isFinite(t) ? '—' : tm.fmt(t);
      if (tm.el.textContent !== text) tm.el.textContent = text;
    }
  };
  const hooks = u.hooks as unknown as Record<string, ((self: uPlot) => void)[] | undefined>;
  for (const name of ['setCursor', 'setData']) (hooks[name] ??= []).push(update);
  update();
  return { update };
}

/** Observe a chart host; `onSize` gets CSS px whenever it changes. */
export function observeHost(host: HTMLElement, onSize: (w: number, h: number) => void): ResizeObserver {
  const ro = new ResizeObserver(() => onSize(Math.floor(host.clientWidth), Math.floor(host.clientHeight)));
  ro.observe(host);
  return ro;
}

export interface PlotCardParts {
  host: HTMLElement;
  legendHost: HTMLElement;
  expandBtn: HTMLButtonElement;
  /** Visually hidden text alternative of the plot (latest values); the caller refreshes it, throttled. */
  summaryEl: HTMLElement;
}

let cardSeq = 0;

/** Skeleton of a plot card: head (title, dim unit, expand button), legend row, chart host. */
export function buildPlotCard(root: HTMLElement, head: { title: string; unit?: string }, onExpand: () => void): PlotCardParts {
  const h = document.createElement('div');
  h.className = 'card-head';
  const n = ++cardSeq;
  const title = document.createElement('h2');
  title.id = `plot-title-${n}`;
  // Titles are uppercase in the card head; Greek letters (3σ) keep their case, as uppercase would turn σ into Σ.
  head.title.split(/(σ)/).forEach((part) => {
    if (part === 'σ') {
      const k = document.createElement('span');
      k.className = 'keep-case';
      k.textContent = part;
      title.append(k);
    } else if (part) title.append(part);
  });
  h.append(title);
  if (head.unit) {
    const u = document.createElement('span');
    u.className = 'plot-unit';
    u.textContent = head.unit;
    h.append(u);
  }
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'expand-btn';
  btn.textContent = '⤢';
  btn.setAttribute('aria-haspopup', 'dialog');
  btn.setAttribute('aria-label', `Expand plot: ${head.title}`);
  btn.title = 'Expand plot';
  btn.addEventListener('click', onExpand);
  h.append(btn);
  const legendHost = document.createElement('div');
  legendHost.className = 'plot-legend';
  const host = document.createElement('div');
  host.className = 'plot-host';
  const summaryEl = document.createElement('p');
  summaryEl.className = 'visually-hidden';
  summaryEl.id = `plot-summary-${n}`;
  root.setAttribute('aria-labelledby', title.id);
  root.setAttribute('aria-describedby', summaryEl.id);
  root.append(h, legendHost, host, summaryEl);
  return { host, legendHost, expandBtn: btn, summaryEl };
}

/** Legend row content: the legend list and a right-aligned mono time cell. */
export function mountLegend(legendHost: HTMLElement, legend: LegendHandle): HTMLElement {
  const time = document.createElement('span');
  time.className = 'lg-time num';
  time.textContent = '—';
  legendHost.replaceChildren(legend.el, time);
  return time;
}
