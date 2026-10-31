// Pure part of the one legend system (docs/specs/ui_console.md §4.1): the glyph vocabulary, the SVG text of
// each glyph, the canvas marker drawing shared by the timelines, and the item lists of every legend.
// DOM-free (no DOM globals at import time), so vitest can test it in node. The DOM part is ./legend.ts.
import type { ChartChrome, DataColors } from '../theme/palette';
import type { TermId } from './glossary';

export type MarkerShape = 'diamond' | 'triangle' | 'circle' | 'square';

export type Glyph =
  | { kind: 'line'; dash?: boolean } // series, paths, 3σ, thresholds
  | { kind: 'band' } // eclipse, excluded, mode-strip segment
  | { kind: 'tick' } // vertical event line: switch, EKF2 reset, cursor
  | { kind: 'marker'; shape: MarkerShape; hollow?: boolean }
  | { kind: 'arrow' } // 3D arrows (axes, Sun, B, nadir)
  | { kind: 'box'; outline?: boolean }; // solid body / ghost or wireframe

export interface LegendItem {
  id: string;
  label: string;
  color: string;
  glyph: Glyph;
  help?: TermId;
  /** Never a toggle button, even in a legend that has a `toggle` callback (explanatory items). */
  static?: boolean;
}

export const GLYPH_W = 16;
export const GLYPH_H = 10;
export const MARKER_SIZE = 8;

/** Neutral hairline round band glyphs, so a near-black band (eclipse) still reads on a dark panel. */
const BAND_EDGE = '#8A94A1';

function markerSvg(shape: MarkerShape, color: string, hollow: boolean): string {
  const fill = hollow ? 'none' : color;
  const stroke = hollow ? ` stroke="${color}" stroke-width="1.5"` : '';
  const a = hollow ? 3.25 : 4; // half-size (the hollow stroke grows outwards by 0.75)
  switch (shape) {
    case 'diamond':
      return `<polygon points="8,${5 - a} ${8 + a},5 8,${5 + a} ${8 - a},5" fill="${fill}"${stroke}/>`;
    case 'triangle':
      return `<polygon points="${8 - a},${5 + a} ${8 + a},${5 + a} 8,${5 - a}" fill="${fill}"${stroke}/>`;
    case 'circle':
      return `<circle cx="8" cy="5" r="${a}" fill="${fill}"${stroke}/>`;
    case 'square':
      return `<rect x="${8 - a}" y="${5 - a}" width="${2 * a}" height="${2 * a}" fill="${fill}"${stroke}/>`;
  }
}

/** SVG markup of a glyph, always `viewBox="0 0 16 10"`; dashed glyphs carry a `stroke-dasharray`. */
export function glyphSvg(g: Glyph, color: string): string {
  let inner: string;
  switch (g.kind) {
    case 'line':
      inner = `<line x1="0" y1="5" x2="16" y2="5" stroke="${color}" stroke-width="2"${g.dash ? ' stroke-dasharray="4 3"' : ''}/>`;
      break;
    case 'band':
      inner = `<rect x="0" y="2" width="16" height="6" fill="${color}" stroke="${BAND_EDGE}" stroke-opacity="0.6" stroke-width="1"/>`;
      break;
    case 'tick':
      inner = `<line x1="8" y1="0" x2="8" y2="10" stroke="${color}" stroke-width="1.5"/>`;
      break;
    case 'marker':
      inner = markerSvg(g.shape, color, g.hollow === true);
      break;
    case 'arrow':
      inner = `<line x1="0" y1="5" x2="11" y2="5" stroke="${color}" stroke-width="2"/><polygon points="16,5 10,1.5 10,8.5" fill="${color}"/>`;
      break;
    case 'box':
      inner = g.outline
        ? `<rect x="2.75" y="1.75" width="10.5" height="6.5" fill="none" stroke="${color}" stroke-width="1.5" stroke-dasharray="3 2"/>`
        : `<rect x="2" y="1" width="12" height="8" fill="${color}"/>`;
      break;
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${GLYPH_W} ${GLYPH_H}" width="${GLYPH_W}" height="${GLYPH_H}" aria-hidden="true" focusable="false">${inner}</svg>`;
}

/** Shape of each ADCS timeline event, used by both the legend and the canvas drawing. */
export const ADCS_EVENT_SHAPE = { tInit: 'diamond', fallback: 'triangle', kick: 'circle' } as const;

/** The part of a canvas 2D context the markers need (so node can import this module and test with a mock). */
export interface MinimalCtx {
  beginPath(): void;
  moveTo(x: number, y: number): void;
  lineTo(x: number, y: number): void;
  arc(x: number, y: number, r: number, a0: number, a1: number): void;
  closePath(): void;
  fill(): void;
  stroke(): void;
  fillStyle: string | object;
  strokeStyle: string | object;
  lineWidth: number;
}

/** Draw a marker of `size` px centred on (x, y): filled, or an outline when `hollow`. */
export function drawMarker(ctx: MinimalCtx, shape: MarkerShape, x: number, y: number, size: number, color: string, hollow = false): void {
  const h = size / 2;
  ctx.beginPath();
  switch (shape) {
    case 'diamond':
      ctx.moveTo(x, y - h);
      ctx.lineTo(x + h, y);
      ctx.lineTo(x, y + h);
      ctx.lineTo(x - h, y);
      ctx.closePath();
      break;
    case 'triangle':
      ctx.moveTo(x - h, y + h);
      ctx.lineTo(x + h, y + h);
      ctx.lineTo(x, y - h);
      ctx.closePath();
      break;
    case 'circle':
      ctx.arc(x, y, h, 0, 2 * Math.PI);
      break;
    case 'square':
      ctx.moveTo(x - h, y - h);
      ctx.lineTo(x + h, y - h);
      ctx.lineTo(x + h, y + h);
      ctx.lineTo(x - h, y + h);
      ctx.closePath();
      break;
  }
  if (hollow) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  } else {
    ctx.fillStyle = color;
    ctx.fill();
  }
}

/** The eight things the ADCS timeline draws. */
export function adcsTimelineItems(d: DataColors, c: ChartChrome): LegendItem[] {
  return [
    { id: 'eclipse', label: 'eclipse', color: c.eclipse, glyph: { kind: 'band' }, help: 'eclipse' },
    { id: 'detumble', label: 'detumble', color: d.detumble, glyph: { kind: 'band' }, help: 'detumble' },
    { id: 'pointing', label: 'pointing', color: d.pointing, glyph: { kind: 'band' }, help: 'pointing' },
    { id: 'triad', label: 'TRIAD init', color: d.estimate, glyph: { kind: 'marker', shape: ADCS_EVENT_SHAPE.tInit }, help: 'triad' },
    { id: 'switch', label: 'switch', color: c.marker, glyph: { kind: 'tick' } },
    { id: 'fallback', label: 'fallback', color: d.danger, glyph: { kind: 'marker', shape: ADCS_EVENT_SHAPE.fallback } },
    { id: 'kick', label: 'kick', color: d.kick, glyph: { kind: 'marker', shape: ADCS_EVENT_SHAPE.kick }, help: 'kick' },
    { id: 'cursor', label: 'cursor', color: c.cursor, glyph: { kind: 'tick' } },
  ];
}

/** The three things the PX4 timeline draws. */
export function px4TimelineItems(d: DataColors, c: ChartChrome): LegendItem[] {
  return [
    { id: 'excluded', label: 'excluded from report statistics', color: c.excluded, glyph: { kind: 'band' }, help: 'excluded' },
    { id: 'reset', label: 'EKF2 reset', color: d.kick, glyph: { kind: 'tick' }, help: 'ekf2Reset' },
    { id: 'cursor', label: 'cursor', color: c.cursor, glyph: { kind: 'tick' } },
  ];
}

/** Legend glyph of a uPlot series: markers only -> circle, dashed -> dashed line, else a line. */
export function seriesGlyph(s: { dash?: number[]; points?: boolean; hollow?: boolean }): Glyph {
  if (s.points) return { kind: 'marker', shape: 'circle', hollow: s.hollow === true };
  return { kind: 'line', dash: Array.isArray(s.dash) && s.dash.length > 0 };
}

// ---------------------------------------------------------------------- 3D view legends

/** Scene colours that are not data roles (the Earth view). Theme-independent, like `COLORS`. */
export const ORBIT_VIEW_COLORS = {
  ring: '#3F6FB0',
  trail: '#8FC2FF',
  satellite: '#E8EEFC',
  satelliteDim: '#3B4357',
} as const;

export type ViewColors = Pick<DataColors, 'axisX' | 'axisY' | 'axisZ' | 'estimate' | 'sun' | 'field' | 'nadir' | 'ours' | 'ekf2'>;

export const VIEW_LEGENDS: {
  sat: (c: ViewColors) => LegendItem[];
  drone: (c: ViewColors) => LegendItem[];
  path: (c: ViewColors) => LegendItem[];
  orbit: (c: ViewColors) => LegendItem[];
} = {
  sat: (c) => [
    { id: 'bx', label: 'body x', color: c.axisX, glyph: { kind: 'arrow' } },
    { id: 'by', label: 'body y', color: c.axisY, glyph: { kind: 'arrow' } },
    { id: 'bz', label: 'body z', color: c.axisZ, glyph: { kind: 'arrow' } },
    { id: 'nadir', label: 'nadir', color: c.nadir, glyph: { kind: 'arrow' }, help: 'nadir' },
    { id: 'sun', label: 'Sun', color: c.sun, glyph: { kind: 'arrow' } },
    { id: 'field', label: 'B field', color: c.field, glyph: { kind: 'arrow' } },
    { id: 'estimate', label: 'estimate', color: c.estimate, glyph: { kind: 'box', outline: true } },
  ],
  drone: (c) => [
    { id: 'ours', label: 'ours', color: c.nadir, glyph: { kind: 'box' } },
    { id: 'ekf2', label: 'EKF2 (reference)', color: c.ekf2, glyph: { kind: 'box', outline: true }, help: 'ekf2' },
    { id: 'bx', label: 'body x', color: c.axisX, glyph: { kind: 'arrow' } },
    { id: 'by', label: 'body y', color: c.axisY, glyph: { kind: 'arrow' } },
    { id: 'bz', label: 'body z', color: c.axisZ, glyph: { kind: 'arrow' } },
  ],
  path: (c) => [
    { id: 'ours', label: 'ours', color: c.ours, glyph: { kind: 'line' } },
    { id: 'ekf2', label: 'EKF2 (reference)', color: c.ekf2, glyph: { kind: 'line', dash: true }, help: 'ekf2' },
    { id: 'start', label: 'start', color: c.nadir, glyph: { kind: 'marker', shape: 'diamond' } },
  ],
  orbit: (c) => [
    { id: 'sun', label: 'Sun direction', color: c.sun, glyph: { kind: 'arrow' } },
    { id: 'ring', label: 'first orbit', color: ORBIT_VIEW_COLORS.ring, glyph: { kind: 'line' } },
    { id: 'trail', label: 'trail', color: ORBIT_VIEW_COLORS.trail, glyph: { kind: 'line' } },
    { id: 'satellite', label: 'satellite', color: ORBIT_VIEW_COLORS.satellite, glyph: { kind: 'box' } },
    { id: 'dim', label: 'dim in eclipse', color: ORBIT_VIEW_COLORS.satelliteDim, glyph: { kind: 'marker', shape: 'square' }, help: 'eclipse' },
  ],
};
