// Pure tests of the legend system (docs/specs/ui_console.md §4.1, §8): glyph SVG geometry, the canvas
// markers, the item lists of the timelines and 3D views, and the series -> glyph mapping.
import { describe, expect, it } from 'vitest';
import { chartChrome, dataColors, type Theme } from '../src/theme/palette';
import {
  ADCS_EVENT_SHAPE,
  GLYPH_H,
  GLYPH_W,
  ORBIT_VIEW_COLORS,
  VIEW_LEGENDS,
  adcsTimelineItems,
  drawMarker,
  glyphSvg,
  px4TimelineItems,
  seriesGlyph,
  type Glyph,
  type LegendItem,
  type MarkerShape,
  type MinimalCtx,
} from '../src/ui/glyphs';

const THEMES: Theme[] = ['light', 'dark'];
const HEX = /^#[0-9A-Fa-f]{6}$/;
const COLOR = '#3FD0E0';

const ALL_GLYPHS: Glyph[] = [
  { kind: 'line' },
  { kind: 'line', dash: true },
  { kind: 'band' },
  { kind: 'tick' },
  { kind: 'marker', shape: 'diamond' },
  { kind: 'marker', shape: 'triangle' },
  { kind: 'marker', shape: 'circle', hollow: true },
  { kind: 'marker', shape: 'square' },
  { kind: 'arrow' },
  { kind: 'box' },
  { kind: 'box', outline: true },
];

describe('glyphSvg', () => {
  it('every glyph is a 16 x 10 SVG in the requested colour', () => {
    expect(GLYPH_W).toBe(16);
    expect(GLYPH_H).toBe(10);
    for (const g of ALL_GLYPHS) {
      const svg = glyphSvg(g, COLOR);
      expect(svg, JSON.stringify(g)).toContain('viewBox="0 0 16 10"');
      expect(svg).toContain(COLOR);
      expect(svg.startsWith('<svg')).toBe(true);
      expect(svg.endsWith('</svg>')).toBe(true);
      expect(svg).toContain('aria-hidden="true"');
    }
  });

  it('only dashed glyphs carry a dash array', () => {
    expect(glyphSvg({ kind: 'line', dash: true }, COLOR)).toContain('stroke-dasharray');
    expect(glyphSvg({ kind: 'box', outline: true }, COLOR)).toContain('stroke-dasharray');
    expect(glyphSvg({ kind: 'line' }, COLOR)).not.toContain('stroke-dasharray');
    expect(glyphSvg({ kind: 'line', dash: false }, COLOR)).not.toContain('stroke-dasharray');
    expect(glyphSvg({ kind: 'band' }, COLOR)).not.toContain('stroke-dasharray');
  });

  it('band is a 16 x 6 rect, tick a 1.5 px vertical line', () => {
    const band = glyphSvg({ kind: 'band' }, COLOR);
    expect(band).toContain('width="16" height="6"');
    expect(band).toContain(`fill="${COLOR}"`);
    const tick = glyphSvg({ kind: 'tick' }, COLOR);
    expect(tick).toContain('x1="8"');
    expect(tick).toContain('x2="8"');
    expect(tick).toContain('stroke-width="1.5"');
  });

  it('markers are about 8 px: filled, or an outline when hollow', () => {
    expect(glyphSvg({ kind: 'marker', shape: 'circle' }, COLOR)).toContain('r="4"');
    expect(glyphSvg({ kind: 'marker', shape: 'square' }, COLOR)).toContain('width="8" height="8"');
    expect(glyphSvg({ kind: 'marker', shape: 'diamond' }, COLOR)).toContain('points="8,1 12,5 8,9 4,5"');
    expect(glyphSvg({ kind: 'marker', shape: 'triangle' }, COLOR)).toContain('points="4,9 12,9 8,1"');
    const hollow = glyphSvg({ kind: 'marker', shape: 'circle', hollow: true }, COLOR);
    expect(hollow).toContain('fill="none"');
    expect(hollow).toContain(`stroke="${COLOR}"`);
    expect(glyphSvg({ kind: 'marker', shape: 'circle' }, COLOR)).not.toContain('fill="none"');
  });
});

/** A context that records the path the marker draws. */
function mockCtx(): MinimalCtx & { log: string[]; filled: boolean; stroked: boolean } {
  const log: string[] = [];
  return {
    log,
    filled: false,
    stroked: false,
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 0,
    beginPath() {
      log.push('begin');
    },
    moveTo(x, y) {
      log.push(`M${x},${y}`);
    },
    lineTo(x, y) {
      log.push(`L${x},${y}`);
    },
    arc(x, y, r) {
      log.push(`A${x},${y},${r}`);
    },
    closePath() {
      log.push('close');
    },
    fill() {
      this.filled = true;
    },
    stroke() {
      this.stroked = true;
    },
  };
}

describe('drawMarker', () => {
  it('draws each shape centred on (x, y) at the given size', () => {
    const shapes: Record<MarkerShape, string[]> = {
      diamond: ['begin', 'M10,16', 'L14,20', 'L10,24', 'L6,20', 'close'],
      triangle: ['begin', 'M6,24', 'L14,24', 'L10,16', 'close'],
      circle: ['begin', 'A10,20,4'],
      square: ['begin', 'M6,16', 'L14,16', 'L14,24', 'L6,24', 'close'],
    };
    for (const [shape, path] of Object.entries(shapes) as [MarkerShape, string[]][]) {
      const ctx = mockCtx();
      drawMarker(ctx, shape, 10, 20, 8, COLOR);
      expect(ctx.log, shape).toEqual(path);
      expect(ctx.filled).toBe(true);
      expect(ctx.stroked).toBe(false);
      expect(ctx.fillStyle).toBe(COLOR);
    }
  });

  it('hollow markers are stroked, not filled', () => {
    const ctx = mockCtx();
    drawMarker(ctx, 'circle', 0, 0, 8, COLOR, true);
    expect(ctx.stroked).toBe(true);
    expect(ctx.filled).toBe(false);
    expect(ctx.strokeStyle).toBe(COLOR);
    expect(ctx.lineWidth).toBe(1.5);
  });
});

describe('timeline legends', () => {
  for (const t of THEMES) {
    const d = dataColors(t);
    const c = chartChrome(t);

    it(`ADCS: the eight drawn kinds, with the shared event shapes (${t})`, () => {
      const items = adcsTimelineItems(d, c);
      expect(items.map((i) => i.id)).toEqual(['eclipse', 'detumble', 'pointing', 'triad', 'switch', 'fallback', 'kick', 'cursor']);
      const by = new Map(items.map((i) => [i.id, i]));
      for (const id of ['eclipse', 'detumble', 'pointing']) expect(by.get(id)?.glyph).toEqual({ kind: 'band' });
      expect(by.get('triad')?.glyph).toEqual({ kind: 'marker', shape: ADCS_EVENT_SHAPE.tInit });
      expect(by.get('fallback')?.glyph).toEqual({ kind: 'marker', shape: ADCS_EVENT_SHAPE.fallback });
      expect(by.get('kick')?.glyph).toEqual({ kind: 'marker', shape: ADCS_EVENT_SHAPE.kick });
      expect(by.get('switch')?.glyph).toEqual({ kind: 'tick' });
      expect(by.get('cursor')?.glyph).toEqual({ kind: 'tick' });
      // the colours are the ones the canvas draws with
      expect(by.get('eclipse')?.color).toBe(c.eclipse);
      expect(by.get('detumble')?.color).toBe(d.detumble);
      expect(by.get('pointing')?.color).toBe(d.pointing);
      expect(by.get('triad')?.color).toBe(d.estimate);
      expect(by.get('switch')?.color).toBe(c.marker);
      expect(by.get('fallback')?.color).toBe(d.danger);
      expect(by.get('kick')?.color).toBe(d.kick);
      expect(by.get('cursor')?.color).toBe(c.cursor);
    });

    it(`PX4: excluded band, EKF2 reset tick, cursor tick (${t})`, () => {
      const items = px4TimelineItems(d, c);
      expect(items.map((i) => i.glyph.kind)).toEqual(['band', 'tick', 'tick']);
      expect(items[0].label).toBe('excluded from report statistics');
      expect(items[0].color).toBe(c.excluded);
      expect(items[1].color).toBe(d.kick);
      expect(items[2].color).toBe(c.cursor);
    });

    it(`every legend colour is #rrggbb and every item has a unique id (${t})`, () => {
      const lists: LegendItem[][] = [
        adcsTimelineItems(d, c),
        px4TimelineItems(d, c),
        VIEW_LEGENDS.sat(d),
        VIEW_LEGENDS.drone(d),
        VIEW_LEGENDS.path(d),
        VIEW_LEGENDS.orbit(d),
      ];
      for (const items of lists) {
        expect(new Set(items.map((i) => i.id)).size).toBe(items.length);
        for (const it of items) {
          expect(it.color, `${t} ${it.id}`).toMatch(HEX);
          expect(it.label.length).toBeGreaterThan(0);
        }
      }
    });
  }
});

describe('seriesGlyph', () => {
  it('maps a uPlot series description to a glyph', () => {
    expect(seriesGlyph({})).toEqual({ kind: 'line', dash: false });
    expect(seriesGlyph({ dash: [] })).toEqual({ kind: 'line', dash: false });
    expect(seriesGlyph({ dash: [6, 4] })).toEqual({ kind: 'line', dash: true });
    expect(seriesGlyph({ points: true })).toEqual({ kind: 'marker', shape: 'circle', hollow: false });
    expect(seriesGlyph({ points: true, hollow: true })).toEqual({ kind: 'marker', shape: 'circle', hollow: true });
    expect(seriesGlyph({ points: true, dash: [6, 4] })).toEqual({ kind: 'marker', shape: 'circle', hollow: false });
  });
});

describe('3D view legends', () => {
  const d = dataColors('dark');

  it('Sat: six arrows and a dashed box outline for the estimate', () => {
    const items = VIEW_LEGENDS.sat(d);
    expect(items.map((i) => i.label)).toEqual(['body x', 'body y', 'body z', 'nadir', 'Sun', 'B field', 'estimate']);
    expect(items.slice(0, 6).every((i) => i.glyph.kind === 'arrow')).toBe(true);
    expect(items[6].glyph).toEqual({ kind: 'box', outline: true });
  });

  it('Drone: solid box ours, outlined box EKF2 (reference), three arrows', () => {
    const items = VIEW_LEGENDS.drone(d);
    expect(items[0].glyph).toEqual({ kind: 'box' });
    expect(items[1].label).toBe('EKF2 (reference)');
    expect(items[1].glyph).toEqual({ kind: 'box', outline: true });
    expect(items.slice(2).map((i) => i.glyph.kind)).toEqual(['arrow', 'arrow', 'arrow']);
  });

  it('Path: line, dashed line (reference), diamond start', () => {
    const items = VIEW_LEGENDS.path(d);
    expect(items.map((i) => i.glyph)).toEqual([{ kind: 'line' }, { kind: 'line', dash: true }, { kind: 'marker', shape: 'diamond' }]);
    expect(items[1].label).toBe('EKF2 (reference)');
  });

  it('Orbit: Sun arrow, two lines, the satellite and its dimmed square', () => {
    const items = VIEW_LEGENDS.orbit(d);
    expect(items.map((i) => i.glyph.kind)).toEqual(['arrow', 'line', 'line', 'box', 'marker']);
    expect(items[0].color).toBe(d.sun);
    expect(items[1].color).toBe(ORBIT_VIEW_COLORS.ring);
    expect(items[2].color).toBe(ORBIT_VIEW_COLORS.trail);
    expect(items[4].glyph).toEqual({ kind: 'marker', shape: 'square' });
    expect(items[4].color).toBe(ORBIT_VIEW_COLORS.satelliteDim);
  });
});
