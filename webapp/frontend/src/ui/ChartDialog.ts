// The expand dialog of a plot (docs/specs/ui_console.md §5.3): one native <dialog> opened with showModal()
// (top layer, inert background, Esc through `cancel`). The page supplies a `mount` callback that builds the
// large chart into the hosts the dialog provides; closing destroys that chart and returns focus.
import type { LiveChart } from './chart';
import type { ChartHelp } from './glossary';

export interface ChartDialogOpen {
  title: string;
  unit?: string;
  help: ChartHelp;
  /** Focus goes back here on close (the expand button). */
  returnFocus: HTMLElement;
  mount(
    chartHost: HTMLElement,
    legendHost: HTMLElement,
  ): { chart: LiveChart & { resetZoom(): void }; onZoom(cb: (zoomed: boolean) => void): void };
  onClose(): void;
}

let seq = 0;

export class ChartDialog {
  private readonly dlg: HTMLDialogElement;
  private readonly title: HTMLElement;
  private readonly unit: HTMLElement;
  private readonly resetBtn: HTMLButtonElement;
  private readonly legendHost: HTMLElement;
  private readonly chartHost: HTMLElement;
  private readonly helpBox: HTMLElement;
  private current: { o: ChartDialogOpen; chart: LiveChart & { resetZoom(): void } } | null = null;
  private pressedBackdrop = false;

  constructor() {
    const id = `chart-dialog-${++seq}`;
    const dlg = document.createElement('dialog');
    dlg.className = 'chart-dialog';
    dlg.setAttribute('aria-labelledby', `${id}-title`);
    dlg.setAttribute('aria-describedby', `${id}-help`);
    dlg.innerHTML = `
      <div class="cd-shell">
        <header class="cd-head">
          <h2 class="cd-title" id="${id}-title" tabindex="-1"></h2>
          <span class="cd-unit"></span>
          <button type="button" class="cd-reset" disabled>Reset zoom</button>
          <button type="button" class="cd-close" aria-label="Close">✕</button>
        </header>
        <div class="cd-body">
          <div class="cd-main">
            <div class="plot-legend cd-legend"></div>
            <div class="plot-host cd-chart"></div>
          </div>
          <aside class="cd-help" id="${id}-help"></aside>
        </div>
      </div>`;
    const q = <T extends HTMLElement>(sel: string): T => {
      const e = dlg.querySelector<T>(sel);
      if (!e) throw new Error(`missing ${sel}`);
      return e;
    };
    this.dlg = dlg;
    this.title = q('.cd-title');
    this.unit = q('.cd-unit');
    this.resetBtn = q<HTMLButtonElement>('.cd-reset');
    this.legendHost = q('.cd-legend');
    this.chartHost = q('.cd-chart');
    this.helpBox = q('.cd-help');
    this.resetBtn.addEventListener('click', () => this.current?.chart.resetZoom());
    q('.cd-close').addEventListener('click', () => this.close());
    // A backdrop click targets the <dialog> itself (the shell fills it); a drag that merely ends there does not count.
    dlg.addEventListener('pointerdown', (e) => {
      this.pressedBackdrop = e.target === dlg;
    });
    dlg.addEventListener('click', (e) => {
      if (e.target === dlg && this.pressedBackdrop) this.close();
      this.pressedBackdrop = false;
    });
    dlg.addEventListener('close', () => this.finish()); // Esc (cancel) and every other way of closing
    document.body.append(dlg);
  }

  get isOpen(): boolean {
    return this.dlg.open;
  }

  open(o: ChartDialogOpen): void {
    this.close();
    this.title.textContent = o.title;
    this.unit.textContent = o.unit ?? '';
    this.helpBox.replaceChildren();
    const h = document.createElement('h3');
    h.textContent = 'How to read this';
    this.helpBox.append(h);
    for (const text of o.help.paragraphs) {
      const p = document.createElement('p');
      p.textContent = text;
      this.helpBox.append(p);
    }
    this.legendHost.replaceChildren();
    this.chartHost.replaceChildren();
    this.resetBtn.disabled = true;
    this.dlg.showModal(); // laid out before the chart measures its host
    const m = o.mount(this.chartHost, this.legendHost);
    m.onZoom((zoomed) => {
      this.resetBtn.disabled = !zoomed;
    });
    this.current = { o, chart: m.chart };
    // Focus the title, not a button: Space then still reaches the PX4 play toggle.
    this.title.focus({ preventScroll: true });
  }

  close(): void {
    if (this.dlg.open) this.dlg.close();
    this.finish(); // do not wait for the asynchronous `close` event
  }

  /** Destroy the chart, tell the page, return focus. Idempotent. */
  private finish(): void {
    const cur = this.current;
    if (!cur) return;
    this.current = null;
    cur.chart.destroy();
    this.legendHost.replaceChildren();
    this.chartHost.replaceChildren();
    cur.o.onClose();
    if (cur.o.returnFocus.isConnected) cur.o.returnFocus.focus({ preventScroll: true });
  }
}

let shared: ChartDialog | null = null;

/** The one dialog of the page, created on first use. */
export function chartDialog(): ChartDialog {
  return (shared ??= new ChartDialog());
}

/** Close it if it was ever opened (tab change, new replay). */
export function closeChartDialog(): void {
  shared?.close();
}
