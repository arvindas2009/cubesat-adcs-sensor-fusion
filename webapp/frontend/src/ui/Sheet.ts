// DOM controller of the rising plot sheet (docs/specs/ui_console.md §3). The pure motion model, the wheel
// and key routing live in ./sheetModel.ts (unit-tested). This file only wires them to the page:
//  - one window wheel listener (capture, non-passive) that sends a plain wheel to the sheet, even over the
//    3D canvases (OrbitControls never sees it), and leaves Ctrl/Cmd + wheel over a canvas to the camera;
//  - PageDown / PageUp, the drag on the sheet head and the toggle button;
//  - per animation frame only `transform` / `opacity` on three elements (sheet, views, dim), so nothing
//    is laid out or resized during motion; on settle: `inert`, focus, `data-sheet`, chart rect sync.
// Below 1100 px wide or 600 px tall the layout falls back to a normal scrolling page and this is disabled.
import { SheetModel, normalizeWheel, routeKey, routeWheel, type SheetState } from './sheetModel';

/** Height of the peek (sheet head + the whole timeline) in px; the same value as `--peek` in style.css. */
export const PEEK_PX = 108;
const PARALLAX = 0.3; // the views move up by this fraction of the travel while the sheet rises
const DIM_MAX = 0.55;
const HINT_MS = 1500;
const HINT_EVERY_MS = 10000;
export const WHEEL_HINT = 'Scroll moves the plot sheet · Ctrl + scroll or pinch to zoom';
export const FALLBACK_QUERY = '(max-width: 1099px), (max-height: 599px)';

export interface SheetMarkup {
  /** DOM id of the section, e.g. `sheet-adcs`; the body gets `<id>-body`. */
  id: string;
  timelineId: string;
  plotsId: string;
  /** Extra class of the plot grid. */
  plotsClass?: string;
  plotIds: string[];
  footerId?: string;
  footerHtml: string;
}

/** HTML of the sheet: head, timeline panel, and the body holding the plot grid and the footer. */
export function sheetMarkup(o: SheetMarkup): string {
  const plots = o.plotIds.map((id) => `<figure class="panel plot-card" id="${id}"></figure>`).join('');
  return `
    <section class="sheet" id="${o.id}" aria-label="Timeline and plots">
      <div class="sheet-head">
        <span class="sheet-grip" aria-hidden="true"></span>
        <span class="sheet-title">Timeline &amp; plots</span>
        <span class="sheet-hint" aria-hidden="true">scroll ↓ · PgDn</span>
        <button type="button" class="sheet-toggle" aria-expanded="false" aria-controls="${o.id}-body">▲ Show plots</button>
      </div>
      <div class="panel timeline-card" id="${o.timelineId}"></div>
      <div class="sheet-body" id="${o.id}-body">
        <div class="plots ${o.plotsClass ?? ''}" id="${o.plotsId}">${plots}</div>
        <footer class="app-footer"${o.footerId ? ` id="${o.footerId}"` : ''}>${o.footerHtml}</footer>
      </div>
    </section>`;
}

export interface SheetOptions {
  /** True while this page is the visible tab (the other page's sheet ignores wheel and keys). */
  isActive(): boolean;
  /** The sheet has settled (or the layout mode changed): sync chart rects, resume 3D rendering when lowered. */
  onSettle(state: SheetState): void;
  /** The 3D views are about to be visible again: mark them for a redraw. */
  onResumeRender?(): void;
}

function must<T extends Element>(root: ParentNode, sel: string): T {
  const e = root.querySelector<T>(sel);
  if (!e) throw new Error(`missing element ${sel}`);
  return e;
}

type KeyTarget = 'text' | 'select' | 'range' | 'slider' | 'other';

function keyTarget(t: EventTarget | null): KeyTarget {
  const el = t as HTMLElement | null;
  if (!el || !el.tagName) return 'other';
  if (el.isContentEditable || el.tagName === 'TEXTAREA') return 'text';
  if (el.tagName === 'SELECT') return 'select';
  if (el.tagName === 'INPUT') {
    const type = (el as HTMLInputElement).type;
    if (type === 'range') return 'range';
    if (type === 'checkbox' || type === 'radio' || type === 'button' || type === 'submit' || type === 'reset') return 'other';
    return 'text'; // text, number, search, ...
  }
  if (el.closest?.('[role="slider"]')) return 'slider';
  return 'other';
}

export class Sheet {
  private readonly stage: HTMLElement;
  private readonly views: HTMLElement;
  private readonly dim: HTMLElement;
  private readonly sheet: HTMLElement;
  private readonly body: HTMLElement;
  private readonly head: HTMLElement;
  private readonly toggleBtn: HTMLButtonElement;
  private readonly hintEl: HTMLElement;
  private readonly fallbackMq: MediaQueryList;
  private readonly reducedMq: MediaQueryList;
  private model: SheetModel;
  private enabled: boolean;
  private settled: SheetState = 'lowered';
  private labelState: SheetState | null = null;
  private moving = false;
  private raf = 0;
  private travel = 0;
  private applied = { p: NaN, travel: NaN, parallax: NaN };
  private readonly ro: ResizeObserver;
  private readonly hintShown = new WeakMap<Element, number>();
  private readonly hintTimers = new WeakMap<Element, number>();
  private dragY0 = 0;
  private dragId = -1;
  private readonly cleanup: (() => void)[] = [];

  constructor(
    stage: HTMLElement,
    private readonly opts: SheetOptions,
  ) {
    this.stage = stage;
    this.views = must(stage, '.stage-views');
    this.dim = must(stage, '.stage-dim');
    this.sheet = must(stage, '.sheet');
    this.body = must(stage, '.sheet-body');
    this.head = must(stage, '.sheet-head');
    this.toggleBtn = must(stage, '.sheet-toggle');
    this.hintEl = must(stage, '.sheet-hint');
    this.fallbackMq = window.matchMedia(FALLBACK_QUERY);
    this.reducedMq = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.enabled = !this.fallbackMq.matches;
    this.model = new SheetModel({ reducedMotion: this.reducedMq.matches });

    this.ro = new ResizeObserver(() => this.measure());
    this.ro.observe(stage);
    this.ro.observe(this.sheet);

    const listen = <K extends keyof WindowEventMap>(type: K, fn: (e: WindowEventMap[K]) => void, opt?: AddEventListenerOptions): void => {
      window.addEventListener(type, fn as EventListener, opt);
      this.cleanup.push(() => window.removeEventListener(type, fn as EventListener, opt));
    };
    // Capture + non-passive: a plain wheel is claimed here before any canvas listener (OrbitControls) runs.
    listen('wheel', (e) => this.onWheel(e), { capture: true, passive: false });
    listen('keydown', (e) => this.onKey(e));

    this.toggleBtn.addEventListener('click', () => {
      if (!this.enabled) return;
      this.model.toggle(performance.now());
      this.kick();
    });
    this.head.addEventListener('pointerdown', (e) => this.onPointerDown(e));
    this.head.addEventListener('pointermove', (e) => this.onPointerMove(e));
    this.head.addEventListener('pointerup', (e) => this.onPointerEnd(e));
    this.head.addEventListener('pointercancel', (e) => this.onPointerEnd(e));

    this.fallbackMq.addEventListener('change', this.onMode);
    this.reducedMq.addEventListener('change', this.onReduced);

    this.measure();
    this.applySettled('lowered', false);
    this.writeTransforms(0, true);
  }

  /** True when the 3D views should not render: the sheet is settled raised and covers them. */
  get renderPaused(): boolean {
    return this.enabled && this.settled === 'raised' && !this.moving;
  }

  get state(): SheetState {
    return this.settled;
  }

  /** The sheet and its layout mode are in use (false in the narrow / short fallback). */
  get layoutEnabled(): boolean {
    return this.enabled;
  }

  destroy(): void {
    for (const c of this.cleanup) c();
    this.cleanup.length = 0;
    this.ro.disconnect();
    this.fallbackMq.removeEventListener('change', this.onMode);
    this.reducedMq.removeEventListener('change', this.onReduced);
    if (this.raf) cancelAnimationFrame(this.raf);
  }

  // ------------------------------------------------------------------ layout and modes

  private measure(): void {
    const peek = parseFloat(getComputedStyle(this.stage).getPropertyValue('--peek')) || PEEK_PX;
    this.travel = this.enabled ? Math.max(0, this.sheet.offsetHeight - peek) : 0;
    this.model.setTravel(this.travel);
    if (this.enabled) this.writeTransforms(this.currentP(), false);
  }

  private currentP(): number {
    return Number.isFinite(this.applied.p) ? this.applied.p : this.settled === 'raised' ? 1 : 0;
  }

  /** Layout mode changed (matchMedia): reset to lowered with no transforms and no inert, or arm the sheet. */
  private readonly onMode = (): void => {
    const enabled = !this.fallbackMq.matches;
    if (enabled === this.enabled) return;
    this.enabled = enabled;
    if (this.raf) {
      cancelAnimationFrame(this.raf);
      this.raf = 0;
    }
    this.moving = false;
    this.model = new SheetModel({ reducedMotion: this.reducedMq.matches });
    this.applied = { p: NaN, travel: NaN, parallax: NaN };
    this.measure();
    this.applySettled('lowered', true);
    this.writeTransforms(0, true);
    this.opts.onResumeRender?.();
    this.opts.onSettle('lowered');
  };

  private readonly onReduced = (): void => {
    this.model.setReducedMotion(this.reducedMq.matches);
    this.applied.parallax = NaN;
    this.kick();
  };

  // ------------------------------------------------------------------ frame loop

  private kick(): void {
    if (!this.raf && this.enabled) this.raf = requestAnimationFrame(this.frame);
  }

  private readonly frame = (now: number): void => {
    this.raf = 0;
    if (!this.enabled) return;
    const r = this.model.tick(now);
    const wasMoving = this.moving;
    this.moving = r.moving;
    if (r.moving && !wasMoving && this.settled === 'raised') this.opts.onResumeRender?.();
    if (r.moving || r.settledNow) this.writeTransforms(r.p, false);
    if (r.state !== this.labelState) this.setLabels(r.state);
    if (r.settledNow) this.applySettled(r.state, true);
    if (r.moving) this.raf = requestAnimationFrame(this.frame);
  };

  /** The only per-frame DOM writes: transform on the sheet and the views, opacity on the dim layer. */
  private writeTransforms(p: number, force: boolean): void {
    if (!this.enabled) {
      this.sheet.style.transform = '';
      this.views.style.transform = '';
      this.dim.style.opacity = '';
      return;
    }
    const parallax = this.model.config.reducedMotion ? 0 : PARALLAX;
    const a = this.applied;
    if (!force && a.p === p && a.travel === this.travel && a.parallax === parallax) return;
    this.applied = { p, travel: this.travel, parallax };
    this.sheet.style.transform = `translate3d(0, ${(1 - p) * this.travel}px, 0)`;
    this.views.style.transform = parallax === 0 ? '' : `translate3d(0, ${-p * parallax * this.travel}px, 0)`;
    this.dim.style.opacity = String(DIM_MAX * p);
  }

  private setLabels(state: SheetState): void {
    this.labelState = state;
    const raised = state === 'raised';
    this.toggleBtn.textContent = raised ? '▼ Hide plots' : '▲ Show plots';
    this.toggleBtn.setAttribute('aria-expanded', String(raised));
    this.hintEl.textContent = raised ? 'scroll ↑ · PgUp' : 'scroll ↓ · PgDn';
  }

  /** Settled state: data attribute, inert, focus, 3D rendering and chart rects. */
  private applySettled(state: SheetState, notify: boolean): void {
    this.settled = state;
    this.setLabels(state);
    if (!this.enabled) {
      this.stage.dataset.sheet = 'lowered';
      this.views.inert = false;
      this.body.inert = false;
      return;
    }
    this.stage.dataset.sheet = state;
    const active = document.activeElement;
    // Focus must leave a subtree before it turns inert (it would otherwise be lost to the page).
    if (state === 'raised' && active && this.views.contains(active)) this.toggleBtn.focus({ preventScroll: true });
    if (state === 'lowered' && active && this.body.contains(active)) this.toggleBtn.focus({ preventScroll: true });
    this.views.inert = state === 'raised';
    this.body.inert = state === 'lowered';
    if (notify) {
      if (state === 'lowered') this.opts.onResumeRender?.();
      this.opts.onSettle(state);
    }
  }

  // ------------------------------------------------------------------ input

  private onWheel(e: WheelEvent): void {
    if (!this.enabled || !this.opts.isActive() || e.deltaY === 0) return;
    const target = e.target instanceof Element ? e.target : null;
    const own = target?.closest<HTMLElement>('[data-own-scroll]') ?? null;
    const route = routeWheel({
      enabled: true,
      dialogOpen: document.querySelector('dialog[open]') !== null,
      overCanvas: target?.closest('.vp-canvas') != null,
      zoomModifier: e.ctrlKey || e.metaKey,
      overOwnScroller: own !== null && own.scrollHeight > own.clientHeight + 1,
    });
    if (route !== 'sheet') return;
    e.preventDefault();
    e.stopPropagation();
    const canvas = target?.closest('.vp-canvas');
    if (canvas) this.showHint(canvas.closest('.vp-body'));
    const mode = e.deltaMode === 1 ? 1 : e.deltaMode === 2 ? 2 : 0;
    this.model.wheel(normalizeWheel(e.deltaY, mode, this.stage.clientHeight), performance.now());
    this.kick();
  }

  private onKey(e: KeyboardEvent): void {
    if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
    if (e.key !== 'PageDown' && e.key !== 'PageUp') return;
    const own = e.target instanceof Element ? e.target.closest<HTMLElement>('[data-own-scroll]') : null;
    if (own && own.scrollHeight > own.clientHeight + 1) return; // an overflowing side column pages itself
    const r = routeKey({
      key: e.key,
      enabled: this.enabled && this.opts.isActive(),
      dialogOpen: document.querySelector('dialog[open]') !== null,
      target: keyTarget(e.target),
      state: this.model.state,
    });
    if (!r) return;
    e.preventDefault();
    this.model.set(r === 'raise' ? 'raised' : 'lowered', performance.now());
    this.kick();
  }

  private onPointerDown(e: PointerEvent): void {
    if (!this.enabled || e.button !== 0 || this.dragId >= 0) return;
    if ((e.target as Element).closest('button')) return;
    this.dragId = e.pointerId;
    this.dragY0 = e.clientY;
    this.head.setPointerCapture(e.pointerId);
    this.model.dragStart(performance.now());
    this.kick();
  }

  private onPointerMove(e: PointerEvent): void {
    if (e.pointerId !== this.dragId) return;
    this.model.dragMove(e.clientY - this.dragY0, performance.now());
    this.kick();
  }

  private onPointerEnd(e: PointerEvent): void {
    if (e.pointerId !== this.dragId) return;
    this.dragId = -1;
    if (this.head.hasPointerCapture(e.pointerId)) this.head.releasePointerCapture(e.pointerId);
    this.model.dragEnd(performance.now());
    this.kick();
  }

  /** The "Scroll moves the plot sheet" hint inside the viewport under the pointer (at most once per 10 s). */
  private showHint(vp: Element | null): void {
    if (!vp) return;
    const now = performance.now();
    const last = this.hintShown.get(vp);
    if (last !== undefined && now - last < HINT_EVERY_MS) return;
    this.hintShown.set(vp, now);
    let el = vp.querySelector<HTMLElement>(':scope > .vp-hint');
    if (!el) {
      el = document.createElement('div');
      el.className = 'vp-hint';
      el.setAttribute('aria-hidden', 'true');
      el.textContent = WHEEL_HINT;
      vp.append(el);
    }
    el.classList.add('show');
    window.clearTimeout(this.hintTimers.get(el));
    const shown = el;
    this.hintTimers.set(el, window.setTimeout(() => shown.classList.remove('show'), HINT_MS));
  }
}
