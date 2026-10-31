// Pure tests of the visual theme (docs/specs/ui_theme.md §4): stored-choice parsing, the role colours of
// each mode, WCAG contrast of every data colour on its plot card, and that style.css mirrors the palette.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { COLORS } from '../src/adcs/theme';
import {
  BRAND,
  DATA_ROLES,
  PLOT_CARD,
  SCREEN,
  chartChrome,
  contrastRatio,
  dataColors,
  nisFallback,
  otherTheme,
  parseHex,
  parseStoredTheme,
  relativeLuminance,
  resolveTheme,
  type Theme,
} from '../src/theme/palette';

const THEMES: Theme[] = ['light', 'dark'];

describe('stored choice', () => {
  it('accepts only light and dark', () => {
    expect(parseStoredTheme('light')).toBe('light');
    expect(parseStoredTheme('dark')).toBe('dark');
    for (const bad of [null, undefined, '', 'Dark', 'LIGHT', 'auto', 'system', 0, 1, true, {}, [], 'dark ']) {
      expect(parseStoredTheme(bad)).toBeNull();
    }
  });

  it('falls back to the system preference for a missing or bad value', () => {
    expect(resolveTheme(null, true)).toBe('dark');
    expect(resolveTheme(null, false)).toBe('light');
    expect(resolveTheme('garbage', true)).toBe('dark');
    expect(resolveTheme(undefined, false)).toBe('light');
  });

  it('lets a valid stored choice win over the system preference', () => {
    expect(resolveTheme('light', true)).toBe('light');
    expect(resolveTheme('dark', false)).toBe('dark');
  });

  it('toggles between the two modes', () => {
    expect(otherTheme('light')).toBe('dark');
    expect(otherTheme('dark')).toBe('light');
  });
});

describe('dataColors', () => {
  it('returns every role, as #rrggbb, in both modes', () => {
    for (const t of THEMES) {
      const c = dataColors(t);
      expect(Object.keys(c).sort()).toEqual([...DATA_ROLES].sort());
      for (const role of DATA_ROLES) expect(c[role], `${t}.${role}`).toMatch(/^#[0-9A-Fa-f]{6}$/);
    }
  });

  it('returns copies (callers cannot corrupt the palette)', () => {
    const a = dataColors('light');
    a.ours = '#000000';
    expect(dataColors('light').ours).not.toBe('#000000');
  });

  it('uses the dark values for the 3D screens (COLORS)', () => {
    const d = dataColors('dark');
    for (const role of ['axisX', 'axisY', 'axisZ', 'estimate', 'sun', 'field', 'nadir', 'detumble', 'pointing', 'kick', 'danger', 'ours', 'ekf2'] as const) {
      expect(COLORS[role].toLowerCase(), role).toBe(d[role].toLowerCase());
    }
  });
});

describe('contrast helpers', () => {
  it('matches known WCAG values', () => {
    expect(contrastRatio('#000000', '#ffffff')).toBeCloseTo(21, 6);
    expect(contrastRatio('#777777', '#ffffff')).toBeCloseTo(4.478, 2);
    expect(contrastRatio('#ffffff', '#ffffff')).toBeCloseTo(1, 9);
    expect(relativeLuminance('#ffffff')).toBeCloseTo(1, 9);
    expect(parseHex('#FFC531')).toEqual([255, 197, 49]);
    expect(() => parseHex('red')).toThrow();
  });
});

describe('contrast on the plot card (spec: data colours >= 3:1)', () => {
  for (const t of THEMES) {
    it(`every data colour on the ${t} plot card`, () => {
      const card = PLOT_CARD[t];
      const c = dataColors(t);
      for (const role of DATA_ROLES) {
        expect(contrastRatio(c[role], card), `${t}.${role} ${c[role]} on ${card}`).toBeGreaterThanOrEqual(3);
      }
      for (const f of nisFallback(t)) expect(contrastRatio(f, card), `${t} nis fallback ${f}`).toBeGreaterThanOrEqual(3);
    });

    it(`chart text, cursor and markers on the ${t} plot card`, () => {
      const ch = chartChrome(t);
      expect(ch.card).toBe(PLOT_CARD[t]);
      expect(contrastRatio(ch.dim, ch.card)).toBeGreaterThanOrEqual(4.5);
      expect(contrastRatio(ch.text, ch.card)).toBeGreaterThanOrEqual(4.5);
      expect(contrastRatio(ch.cursor, ch.band)).toBeGreaterThanOrEqual(3);
      expect(contrastRatio(ch.marker, ch.band)).toBeGreaterThanOrEqual(3);
    });

    it(`mode colours and event markers on the ${t} timeline band`, () => {
      const ch = chartChrome(t);
      const c = dataColors(t);
      for (const role of ['detumble', 'pointing', 'kick', 'danger', 'estimate'] as const) {
        expect(contrastRatio(c[role], ch.band), `${t}.${role} on band`).toBeGreaterThanOrEqual(3);
      }
    });
  }

  it('the grid is fainter than the series (grid contrast < every data colour)', () => {
    for (const t of THEMES) {
      const ch = chartChrome(t);
      const g = contrastRatio(ch.grid, ch.card);
      expect(g).toBeLessThan(1.6);
      for (const role of DATA_ROLES) expect(contrastRatio(dataColors(t)[role], ch.card)).toBeGreaterThan(g);
    }
  });
});

describe('screens and brand', () => {
  it('3D screen text is readable on the plum screen', () => {
    expect(contrastRatio(SCREEN.text, SCREEN.clear)).toBeGreaterThanOrEqual(7);
    expect(contrastRatio(SCREEN.textDim, SCREEN.clear)).toBeGreaterThanOrEqual(4.5);
  });

  it('ink text on every brand fill that carries text is >= 4.5:1', () => {
    for (const fill of [BRAND.sunflower, BRAND.mint, BRAND.periwinkle, BRAND.bubblegum, '#E08A00', '#FFB020', '#B8A6FF']) {
      expect(contrastRatio(BRAND.ink, fill), fill).toBeGreaterThanOrEqual(4.5);
    }
    expect(contrastRatio(BRAND.paper, BRAND.ink)).toBeGreaterThanOrEqual(7);
  });
});

// ---- style.css mirrors the palette -------------------------------------------------------------

const CSS_NAMES: Record<string, string> = {
  axisX: '--axis-x',
  axisY: '--axis-y',
  axisZ: '--axis-z',
  estimate: '--estimate',
  sun: '--sun',
  field: '--field',
  nadir: '--nadir',
  detumble: '--detumble',
  pointing: '--pointing',
  kick: '--kick',
  danger: '--danger',
  ours: '--ours',
  ekf2: '--ekf2',
};

function cssBlock(css: string, selector: string): string {
  const at = css.indexOf(selector);
  if (at < 0) throw new Error(`selector not found: ${selector}`);
  const open = css.indexOf('{', at);
  const close = css.indexOf('}', open);
  return css.slice(open + 1, close);
}

function cssVar(block: string, name: string): string {
  const m = new RegExp(`${name}\\s*:\\s*(#[0-9a-fA-F]{6})\\s*;`).exec(block);
  if (!m) throw new Error(`variable not found: ${name}`);
  return m[1];
}

describe('style.css mirrors the palette', () => {
  const css = readFileSync(new URL('../src/style.css', import.meta.url), 'utf8');
  const blocks: Record<Theme, string> = {
    light: cssBlock(css, ":root[data-theme='light']"),
    dark: cssBlock(css, ":root[data-theme='dark']"),
  };

  for (const t of THEMES) {
    it(`data colour variables (${t})`, () => {
      const c = dataColors(t);
      for (const [role, name] of Object.entries(CSS_NAMES)) {
        expect(cssVar(blocks[t], name).toLowerCase(), `${t} ${name}`).toBe(c[role as keyof typeof c].toLowerCase());
      }
    });

    it(`plot card variable (${t})`, () => {
      expect(cssVar(blocks[t], '--card-mint').toLowerCase()).toBe(PLOT_CARD[t].toLowerCase());
    });
  }

  it('brand colours and screens', () => {
    expect(cssVar(blocks.light, '--sunflower').toLowerCase()).toBe(BRAND.sunflower.toLowerCase());
    expect(cssVar(blocks.light, '--ink').toLowerCase()).toBe(BRAND.ink.toLowerCase());
    expect(cssVar(blocks.light, '--page').toLowerCase()).toBe(BRAND.paper.toLowerCase());
    expect(cssVar(blocks.light, '--screen').toLowerCase()).toBe(SCREEN.clear.toLowerCase());
    expect(blocks.dark).not.toMatch(/--screen\s*:/); // the screens are the same dark plum in both modes
    expect(cssVar(blocks.dark, '--page').toLowerCase()).toBe('#160e17');
  });
});
