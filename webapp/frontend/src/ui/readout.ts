// DOM building blocks of the grouped readouts (docs/specs/ui_console.md §6): a titled group with a <dl> of
// rows, each row a dim label (with an optional "?" help tip), a mono value and a dim unit after the value.
// The value element is what the pages write into (they keep their formatters and text caches).
import type { TermId } from './glossary';
import { appendHelp } from './help';

export interface RowSpec {
  key: string;
  label: string;
  /** Dim unit shown after the value. */
  unit?: string;
  help?: TermId;
  /** Tooltip of the label. */
  title?: string;
  /** A wide three-component value: the row may wrap the value under the label. */
  vec?: boolean;
}

export function readoutGroup(heading: string): { section: HTMLElement; list: HTMLElement } {
  const section = document.createElement('section');
  section.className = 'ro-group';
  const h = document.createElement('h3');
  h.className = 'ro-heading';
  h.textContent = heading;
  const list = document.createElement('dl');
  list.className = 'ro-list';
  section.append(h, list);
  return { section, list };
}

/** One `<div class="ro"><dt>label ?</dt><dd><span class="readout-value">value</span> unit</dd></div>` row. */
export function readoutRow(spec: RowSpec, initial: string): { row: HTMLElement; value: HTMLElement } {
  const row = document.createElement('div');
  row.className = spec.vec ? 'ro ro-vec' : 'ro';
  const dt = document.createElement('dt');
  dt.className = 'readout-label';
  const text = document.createElement('span');
  text.textContent = spec.label;
  if (spec.title) text.title = spec.title;
  dt.append(text);
  if (spec.help) appendHelp(dt, spec.help);
  const dd = document.createElement('dd');
  dd.className = 'ro-dd';
  const value = document.createElement('span');
  value.className = 'readout-value';
  value.textContent = initial;
  dd.append(value);
  if (spec.unit) {
    const u = document.createElement('span');
    u.className = 'ro-unit';
    u.textContent = spec.unit;
    dd.append(u);
  }
  row.append(dt, dd);
  return { row, value };
}
