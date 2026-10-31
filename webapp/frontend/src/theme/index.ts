// DOM-bound runtime of the visual theme (docs/specs/ui_theme.md §3): `data-theme` on <html>, the stored
// choice, subscriptions. The first-paint application is an inline script in index.html; `initTheme()` only
// re-reads the same sources so both agree. The pure palette lives in ./palette.ts (tested in node).
import {
  THEME_STORAGE_KEY,
  chartChrome,
  dataColors as paletteDataColors,
  otherTheme,
  parseStoredTheme,
  resolveTheme,
  type ChartChrome,
  type DataColors,
  type Theme,
} from './palette';

export * from './palette';

type Listener = (t: Theme) => void;
const listeners = new Set<Listener>();
let active: Theme | null = null;

function readStored(): unknown {
  try {
    return window.localStorage.getItem(THEME_STORAGE_KEY);
  } catch {
    return null; // storage blocked (private mode, disabled cookies)
  }
}

function systemPrefersDark(): boolean {
  try {
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  } catch {
    return false;
  }
}

function apply(t: Theme): void {
  active = t;
  document.documentElement.setAttribute('data-theme', t);
  document.documentElement.style.colorScheme = t;
}

/** Resolve (stored choice, else system preference) and apply. Call once at start-up. */
export function initTheme(): Theme {
  const t = resolveTheme(readStored(), systemPrefersDark());
  apply(t);
  return t;
}

/** The active mode (the `data-theme` attribute if `initTheme()` has not run yet, else light). */
export function currentTheme(): Theme {
  if (active) return active;
  const attr = typeof document !== 'undefined' ? parseStoredTheme(document.documentElement.getAttribute('data-theme')) : null;
  return attr ?? 'light';
}

/** Switch mode: updates `data-theme`, stores the choice, notifies subscribers (no-op if unchanged). */
export function setTheme(t: Theme): void {
  if (t === active) return;
  apply(t);
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, t);
  } catch {
    // not persisted; the session keeps the choice
  }
  for (const cb of [...listeners]) cb(t);
}

export function toggleTheme(): Theme {
  const next = otherTheme(currentTheme());
  setTheme(next);
  return next;
}

/** Subscribe to mode changes; returns the unsubscribe function. */
export function onThemeChange(cb: Listener): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

/** Role colours for the given mode (default: the active one). */
export function dataColors(theme: Theme = currentTheme()): DataColors {
  return paletteDataColors(theme);
}

/** Chart / timeline chrome colours for the given mode (default: the active one). */
export function chromeColors(theme: Theme = currentTheme()): ChartChrome {
  return chartChrome(theme);
}

/** Run `cb` once the web fonts have loaded (canvas text drawn earlier used the fallback font). */
export function whenFontsReady(cb: () => void): void {
  try {
    void document.fonts?.ready.then(cb);
  } catch {
    // no FontFaceSet: nothing to wait for
  }
}
