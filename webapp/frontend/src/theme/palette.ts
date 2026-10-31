// Pure (DOM-free) part of the visual theme "Colour blocks" (docs/specs/ui_theme.md §1, §3): the data
// colour roles for each mode, the colours the canvas charts use, stored-choice parsing and the WCAG
// contrast helpers the tests use. The CSS variables in src/style.css mirror these values (a test
// compares them); the DOM-bound runtime (apply / toggle / subscribe) lives in ./index.ts.

export type Theme = 'light' | 'dark';

export const THEME_STORAGE_KEY = 'theme';

/** Brand palette (spec §1). Named here so tests and the 3D screens share one source. */
export const BRAND = {
  plum: '#3A2236',
  sunflower: '#FFC531',
  mint: '#3DDC97',
  periwinkle: '#7C80EE',
  bubblegum: '#EBA3D6',
  ink: '#1C1220',
  paper: '#F3F3F6',
} as const;

/** 3D viewports are dark "screens" in both modes (spec §1). */
export const SCREEN = {
  clear: '#24152A',
  gridMajor: '#4A3350',
  gridMinor: '#33223A',
  text: '#F2EAF0',
  textDim: '#CDBBCB',
} as const;

/** Colour roles (docs/specs/frontend.md §2) used by plots, markers and the timeline. */
export interface DataColors {
  axisX: string;
  axisY: string;
  axisZ: string;
  estimate: string;
  sun: string;
  field: string;
  nadir: string;
  detumble: string;
  pointing: string;
  kick: string;
  danger: string;
  ours: string;
  ekf2: string;
  sigma3: string;
  nisGpsPos: string;
  nisGpsVel: string;
  nisBaro: string;
  nisMag: string;
}

export const DATA_ROLES: readonly (keyof DataColors)[] = [
  'axisX',
  'axisY',
  'axisZ',
  'estimate',
  'sun',
  'field',
  'nadir',
  'detumble',
  'pointing',
  'kick',
  'danger',
  'ours',
  'ekf2',
  'sigma3',
  'nisGpsPos',
  'nisGpsVel',
  'nisBaro',
  'nisMag',
];

const LIGHT: DataColors = {
  axisX: '#D63B3B',
  axisY: '#1A8F43',
  axisZ: '#2F6BE0',
  estimate: '#0A7F94',
  sun: '#A87B00',
  field: '#6B4FD8',
  nadir: '#3A2236',
  detumble: '#B86500',
  pointing: '#0E8A56',
  kick: '#D4198F',
  danger: '#C42B3B',
  ours: '#0B8F6E',
  ekf2: '#C0680A',
  sigma3: '#4F7F88',
  nisGpsPos: '#2F6BE0',
  nisGpsVel: '#6B4FD8',
  nisBaro: '#3A2236',
  nisMag: '#D0305F',
};

const DARK: DataColors = {
  axisX: '#FF5A5A',
  axisY: '#4FD96B',
  axisZ: '#4A8DFF',
  estimate: '#2FD6E6',
  sun: '#FFD23F',
  field: '#B39BFF',
  nadir: '#F2EAF0',
  detumble: '#FFB020',
  pointing: '#3DDC97',
  kick: '#FF5FC4',
  danger: '#FF5C6C',
  ours: '#3DDC97',
  ekf2: '#FFB347',
  sigma3: '#7FB9C2',
  nisGpsPos: '#4A8DFF',
  nisGpsVel: '#B39BFF',
  nisBaro: '#F2EAF0',
  nisMag: '#FF5C8A',
};

/** Colours for NIS sensors the server names but the table above does not (cycled). */
const NIS_FALLBACK: Record<Theme, readonly string[]> = {
  light: ['#0A7F94', '#A87B00', '#0E8A56', '#B35F0A'],
  dark: ['#2FD6E6', '#FFD23F', '#3DDC97', '#FF8A3D'],
};

/** Plot / timeline card colour (the card sits in the mint block): light tint, dark card (spec §1). */
export const PLOT_CARD: Record<Theme, string> = { light: '#DDF8EB', dark: '#251826' };

/** The role colours for the given mode. */
export function dataColors(theme: Theme): DataColors {
  return { ...(theme === 'dark' ? DARK : LIGHT) };
}

export function nisFallback(theme: Theme): readonly string[] {
  return NIS_FALLBACK[theme];
}

/** Colours of the canvas charts that are not data roles: text, grid, timeline band, cursor. */
export interface ChartChrome {
  card: string;
  text: string;
  dim: string;
  grid: string;
  tick: string;
  band: string;
  bandEdge: string;
  eclipse: string;
  eclipseEdge: string;
  excluded: string;
  cursor: string;
  marker: string;
}

const CHROME: Record<Theme, ChartChrome> = {
  light: {
    card: PLOT_CARD.light,
    text: '#1C1220',
    dim: '#5B4A58',
    grid: '#BDF2DA', // faint tint of the mint block
    tick: '#9ED9BF',
    band: '#F6FDF9',
    bandEdge: '#9ED9BF',
    eclipse: '#3A2236',
    eclipseEdge: '#1C1220',
    excluded: '#C4E8D7',
    cursor: '#1C1220',
    marker: '#3A2236',
  },
  dark: {
    card: PLOT_CARD.dark,
    text: '#F2EAF0',
    dim: '#B9A8B6',
    grid: '#2A3F3D',
    tick: '#3C5552',
    band: '#1B111C',
    bandEdge: '#3C5552',
    eclipse: '#0A050B',
    eclipseEdge: '#5B4060',
    excluded: '#33243A',
    cursor: '#FFFFFF',
    marker: '#F2EAF0',
  },
};

export function chartChrome(theme: Theme): ChartChrome {
  return { ...CHROME[theme] };
}

// ---------------------------------------------------------------- stored choice

/** `localStorage` value -> a theme, or null for anything missing or unexpected. */
export function parseStoredTheme(raw: unknown): Theme | null {
  return raw === 'light' || raw === 'dark' ? raw : null;
}

/** Stored choice if valid, otherwise the system preference (`prefers-color-scheme: dark`). */
export function resolveTheme(stored: unknown, prefersDark: boolean): Theme {
  return parseStoredTheme(stored) ?? (prefersDark ? 'dark' : 'light');
}

export function otherTheme(t: Theme): Theme {
  return t === 'dark' ? 'light' : 'dark';
}

// ---------------------------------------------------------------- WCAG contrast

/** `#rrggbb` -> [r, g, b] in 0-255; throws on anything else. */
export function parseHex(hex: string): [number, number, number] {
  const m = /^#([0-9a-fA-F]{6})$/.exec(hex);
  if (!m) throw new Error(`not a #rrggbb colour: ${hex}`);
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** WCAG 2 relative luminance. */
export function relativeLuminance(hex: string): number {
  const [r, g, b] = parseHex(hex).map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** WCAG 2 contrast ratio (1 to 21) of two colours. */
export function contrastRatio(a: string, b: string): number {
  const la = relativeLuminance(a);
  const lb = relativeLuminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}
