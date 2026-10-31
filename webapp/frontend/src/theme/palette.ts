// Pure (DOM-free) part of the visual theme "Flight-deck console" (docs/specs/ui_console.md §1): the UI
// tokens, the data colour roles for each mode, the colours the canvas charts use, stored-choice parsing and
// the WCAG contrast helpers the tests use. The CSS variables in src/style.css mirror these values (a test
// compares them); the DOM-bound runtime (apply / toggle / subscribe) lives in ./index.ts.

export type Theme = 'light' | 'dark';

export const THEME_STORAGE_KEY = 'theme';

/** Fonts (spec §1.4); mirrored as `--font-ui` / `--font-mono` and used by the canvas code. */
export const FONT_UI = '"Barlow Semi Condensed", system-ui, sans-serif';
export const FONT_MONO = '"JetBrains Mono Variable", ui-monospace, Consolas, monospace';

/** UI tokens of one mode (spec §1.1). `data` and `neutral` are the data accent and its grey. */
export interface UiTokens {
  page: string;
  bar: string;
  panel: string;
  inset: string;
  edge: string;
  edgeInput: string;
  bracket: string;
  text: string;
  dim: string;
  action: string;
  actionText: string;
  onAction: string;
  actionHover: string;
  data: string;
  neutral: string;
  alarm: string;
  ok: string;
  okText: string;
  focus: string;
  fillDetumble: string;
  fillPointing: string;
  fillSunlit: string;
  fillDump: string;
  fillEclipse: string;
}

/** CSS variable that mirrors each UI token (the test parses both mode blocks of style.css). */
export const UI_CSS_NAMES: Record<keyof UiTokens, string> = {
  page: '--page',
  bar: '--bar',
  panel: '--panel',
  inset: '--inset',
  edge: '--edge',
  edgeInput: '--edge-input',
  bracket: '--bracket',
  text: '--text',
  dim: '--dim',
  action: '--action',
  actionText: '--action-text',
  onAction: '--on-action',
  actionHover: '--action-hover',
  data: '--data',
  neutral: '--neutral',
  alarm: '--alarm',
  ok: '--ok',
  okText: '--ok-text',
  focus: '--focus',
  fillDetumble: '--fill-detumble',
  fillPointing: '--fill-pointing',
  fillSunlit: '--fill-sunlit',
  fillDump: '--fill-dump',
  fillEclipse: '--fill-eclipse',
};

/** Text colour on the eclipse chip (the other chips carry `onAction` ink), the same in both modes. */
export const FILL_ECLIPSE_TEXT = '#E6E9ED';

export const UI: Record<Theme, UiTokens> = {
  dark: {
    page: '#0D0F12',
    bar: '#0A0C0F',
    panel: '#15191E',
    inset: '#0F1216',
    edge: '#2A3038',
    edgeInput: '#6B7480',
    bracket: '#5A6472',
    text: '#E6E9ED',
    dim: '#9AA4B2',
    action: '#FF8A1F',
    actionText: '#FF8A1F',
    onAction: '#0D0F12',
    actionHover: '#FFA04A',
    data: '#3FD0E0',
    neutral: '#9AA4B2',
    alarm: '#FF3B3B',
    ok: '#46D36B',
    okText: '#46D36B',
    focus: '#FF8A1F',
    fillDetumble: '#FFB020',
    fillPointing: '#46D36B',
    fillSunlit: '#FFD23F',
    fillDump: '#B39BFF',
    fillEclipse: '#05070A',
  },
  light: {
    page: '#E9ECEF',
    bar: '#F7F8FA',
    panel: '#F7F8FA',
    inset: '#FFFFFF',
    edge: '#C5CBD3',
    edgeInput: '#6B7480',
    bracket: '#8A94A1',
    text: '#14181D',
    dim: '#4A5563',
    action: '#D9600B',
    actionText: '#A84800',
    onAction: '#0D0F12',
    actionHover: '#E8742A',
    data: '#0B7C8A',
    neutral: '#4A5563',
    alarm: '#C81E1E',
    ok: '#1E8A3E',
    okText: '#176F31',
    focus: '#D9600B',
    fillDetumble: '#FFB020',
    fillPointing: '#46D36B',
    fillSunlit: '#FFD23F',
    fillDump: '#B39BFF',
    fillEclipse: '#05070A',
  },
};

/** 3D viewports are dark "screens" in both modes (spec §1.1). */
export const SCREEN = {
  clear: '#080A0D',
  gridMajor: '#1F2730',
  gridMinor: '#141A21',
  text: '#E6E9ED',
  textDim: '#9AA4B2',
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
  estimate: '#0B7C8A',
  sun: '#A87B00',
  field: '#6B4FD8',
  nadir: '#4A5563',
  detumble: '#B86500',
  pointing: '#1E8A3E',
  kick: '#D4198F',
  danger: '#C81E1E',
  ours: '#0B8F6E',
  ekf2: '#C0680A',
  sigma3: '#4F7F88',
  nisGpsPos: '#2F6BE0',
  nisGpsVel: '#6B4FD8',
  nisBaro: '#4A5563',
  nisMag: '#D0305F',
};

const DARK: DataColors = {
  axisX: '#FF5A5A',
  axisY: '#4FD96B',
  axisZ: '#4A8DFF',
  estimate: '#3FD0E0',
  sun: '#FFD23F',
  field: '#B39BFF',
  nadir: '#D7DCE3',
  detumble: '#FFB020',
  pointing: '#46D36B',
  kick: '#FF5FC4',
  danger: '#FF3B3B',
  ours: '#3DDC97',
  ekf2: '#FFB347',
  sigma3: '#7FB9C2',
  nisGpsPos: '#4A8DFF',
  nisGpsVel: '#B39BFF',
  nisBaro: '#D7DCE3',
  nisMag: '#FF5C8A',
};

/** Colours for NIS sensors the server names but the table above does not (cycled). */
const NIS_FALLBACK: Record<Theme, readonly string[]> = {
  light: ['#0B7C8A', '#A87B00', '#1E8A3E', '#B35F0A'],
  dark: ['#3FD0E0', '#FFD23F', '#46D36B', '#FF8A3D'],
};

/** Plot / timeline card colour (`--panel`). */
export const PLOT_CARD: Record<Theme, string> = { light: UI.light.panel, dark: UI.dark.panel };

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
    text: '#14181D',
    dim: '#4A5563',
    grid: '#E3E6EA',
    tick: '#C5CBD3',
    band: '#FFFFFF',
    bandEdge: '#C5CBD3',
    eclipse: '#3A4350',
    eclipseEdge: '#14181D',
    excluded: '#DDE1E6',
    cursor: '#0D0F12',
    marker: '#14181D',
  },
  dark: {
    card: PLOT_CARD.dark,
    text: '#E6E9ED',
    dim: '#9AA4B2',
    grid: '#232A32',
    tick: '#3A434E',
    band: '#0F1216',
    bandEdge: '#2A3038',
    eclipse: '#05070A',
    eclipseEdge: '#5A6472',
    excluded: '#262D36',
    cursor: '#FFFFFF',
    marker: '#E6E9ED',
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
