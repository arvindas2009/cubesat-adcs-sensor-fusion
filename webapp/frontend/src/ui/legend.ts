// DOM part of the one legend system (docs/specs/ui_console.md §4.2). The same markup and typography
// everywhere: timelines, plot cards, the expand dialog and the 3D views. The glyph vocabulary and the item
// lists are in ./glyphs.ts (pure, tested).
import { glyphSvg, type LegendItem } from './glyphs';

export interface LegendOptions {
  /** `screen` adds a dark backing and edge, for legends drawn over a 3D view. */
  surface: 'panel' | 'screen';
  /** Items get a mono value cell (filled by `setValue`). */
  live?: boolean;
  /** When given, non-static items are `<button aria-pressed>` toggles. */
  toggle?: (id: string, on: boolean) => void;
}

export interface LegendHandle {
  el: HTMLElement;
  setValue(id: string, text: string): void;
  setColors(items: LegendItem[]): void;
}

export function renderLegend(items: LegendItem[], opts: LegendOptions): LegendHandle {
  const ul = document.createElement('ul');
  ul.className = `legend legend-${opts.surface}`;
  const glyphs = new Map<string, HTMLElement>();
  const labels = new Map<string, HTMLElement>();
  const values = new Map<string, HTMLElement>();

  for (const it of items) {
    const li = document.createElement('li');
    li.dataset.id = it.id;
    const asButton = opts.toggle !== undefined && it.static !== true;
    const inner = document.createElement(asButton ? 'button' : 'span');
    inner.className = 'lg-item';
    const glyph = document.createElement('span');
    glyph.className = 'lg-glyph';
    glyph.innerHTML = glyphSvg(it.glyph, it.color);
    const label = document.createElement('span');
    label.className = 'lg-label';
    label.textContent = it.label;
    inner.append(glyph, label);
    if (opts.live) {
      const v = document.createElement('span');
      v.className = 'lg-value num';
      v.textContent = '—';
      inner.append(v);
      values.set(it.id, v);
    }
    if (asButton) {
      const b = inner as HTMLButtonElement;
      b.type = 'button';
      b.setAttribute('aria-pressed', 'true');
      b.addEventListener('click', () => {
        const on = b.getAttribute('aria-pressed') !== 'true';
        b.setAttribute('aria-pressed', String(on));
        li.classList.toggle('off', !on);
        opts.toggle?.(it.id, on);
      });
    }
    li.append(inner);
    ul.append(li);
    glyphs.set(it.id, glyph);
    labels.set(it.id, label);
  }

  return {
    el: ul,
    setValue(id, text) {
      const v = values.get(id);
      if (v && v.textContent !== text) v.textContent = text;
    },
    setColors(next) {
      for (const it of next) {
        const g = glyphs.get(it.id);
        if (g) g.innerHTML = glyphSvg(it.glyph, it.color);
        const l = labels.get(it.id);
        if (l && l.textContent !== it.label) l.textContent = it.label;
      }
    },
  };
}
