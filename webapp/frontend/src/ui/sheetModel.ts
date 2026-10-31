// Pure (DOM-free) model of the rising plot sheet (docs/specs/ui_console.md §3.3): wheel / drag follow the
// input 1:1, then snap to an end after a short idle; reduced motion flips the state instantly instead.
// `p` is the raised fraction: 0 = lowered (only the peek shows), 1 = raised (the sheet covers the views).
// Time is passed in by the caller (milliseconds, any monotonic clock), so everything here is testable.

export type SheetState = 'lowered' | 'raised';

export interface SheetConfig {
  /** Layout distance between the lowered and the raised position, in px (the sheet height minus the peek). */
  travelPx: number;
  /** Duration of the snap animation (260). */
  durationMs: number;
  /** A gesture ends after this long without input (140). */
  idleMs: number;
  /** Minimum travel fraction of a gesture that commits it to the gesture direction (0.08). */
  commitFrac: number;
  /** Reduced motion: the first this-many px of a gesture flip the state (40). */
  discretePx: number;
  reducedMotion: boolean;
}

export const DEFAULT_SHEET_CONFIG: SheetConfig = {
  travelPx: 600,
  durationMs: 260,
  idleMs: 140,
  commitFrac: 0.08,
  discretePx: 40,
  reducedMotion: false,
};

export interface SheetTick {
  /** Raised fraction in [0, 1]; exactly 0 or 1 once settled. */
  p: number;
  /** The committed state: the settled one, or the target of a running snap. */
  state: SheetState;
  /** True while the position differs from a settled one (following input, or animating). */
  moving: boolean;
  /** True exactly once per settle (a gesture that never left its state does not settle). */
  settledNow: boolean;
}

export function easeOutCubic(x: number): number {
  const u = 1 - Math.min(1, Math.max(0, x));
  return 1 - u * u * u;
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

type Gesture = 'none' | 'wheel' | 'drag';

export class SheetModel {
  private cfg: SheetConfig;
  private p: number;
  private target: SheetState; // committed state
  private gesture: Gesture = 'none';
  private gestureStartP = 0;
  private lastInputAt = 0;
  private dragP0 = 0;
  private acc = 0; // reduced motion: signed px accumulated in the current gesture
  private discreteDone = false;
  private anim: { from: number; to: number; t0: number } | null = null;
  private dirtyMove = false; // p has left (or is not at) a settled value
  private pendingSettle = false;

  constructor(cfg: Partial<SheetConfig> = {}, initial: SheetState = 'lowered') {
    this.cfg = { ...DEFAULT_SHEET_CONFIG, ...cfg };
    this.target = initial;
    this.p = initial === 'raised' ? 1 : 0;
  }

  get config(): Readonly<SheetConfig> {
    return this.cfg;
  }

  /** The committed state (the target of a running snap). */
  get state(): SheetState {
    return this.target;
  }

  setTravel(px: number): void {
    this.cfg.travelPx = Number.isFinite(px) && px > 0 ? px : 0;
  }

  setReducedMotion(b: boolean): void {
    if (b === this.cfg.reducedMotion) return;
    this.cfg.reducedMotion = b;
    this.acc = 0;
    this.discreteDone = false;
    if (b) {
      // No animation under reduced motion: finish whatever is running at once.
      if (this.anim) {
        this.p = this.anim.to;
        this.anim = null;
        this.settle();
      } else if (this.gesture !== 'none' || this.dirtyMove) {
        this.endGesture();
      }
    }
  }

  // ------------------------------------------------------------------ input

  /** Wheel input: positive = scroll down = raise. */
  wheel(deltaPx: number, now: number): void {
    if (!Number.isFinite(deltaPx) || deltaPx === 0) return;
    this.closeStaleGesture(now);
    if (this.cfg.reducedMotion) {
      this.discrete(deltaPx, now);
      return;
    }
    if (this.cfg.travelPx <= 0) return;
    this.beginGesture('wheel');
    this.anim = null;
    this.setP(this.p + deltaPx / this.cfg.travelPx);
    this.lastInputAt = now;
  }

  dragStart(now: number): void {
    this.closeStaleGesture(now);
    if (this.gesture !== 'none') this.endGesture();
    this.beginGesture('drag');
    this.anim = null;
    this.dragP0 = this.p;
    this.acc = 0;
    this.discreteDone = false;
    this.lastInputAt = now;
  }

  /** `dyPx` is the total pointer displacement since `dragStart`; positive = downwards (lowering). */
  dragMove(dyPx: number, now: number): void {
    if (this.gesture !== 'drag' || !Number.isFinite(dyPx)) return;
    this.lastInputAt = now;
    if (this.cfg.reducedMotion) {
      // Upwards drag (negative) raises; the first discretePx flips the state, the rest is ignored.
      if (!this.discreteDone && Math.abs(dyPx) >= this.cfg.discretePx) {
        this.discreteDone = true;
        this.flipTo(dyPx < 0 ? 'raised' : 'lowered');
      }
      return;
    }
    if (this.cfg.travelPx <= 0) return;
    this.setP(this.dragP0 - dyPx / this.cfg.travelPx);
  }

  dragEnd(now: number): void {
    if (this.gesture !== 'drag') return;
    this.endGesture(now);
  }

  toggle(now: number): void {
    this.set(this.target === 'raised' ? 'lowered' : 'raised', now);
  }

  /** Go to a state by animation (instantly under reduced motion). */
  set(s: SheetState, now: number): void {
    if (this.gesture !== 'none') this.gesture = 'none';
    const to = s === 'raised' ? 1 : 0;
    this.target = s;
    if (this.cfg.reducedMotion) {
      this.anim = null;
      if (this.p !== to) {
        this.p = to;
        this.dirtyMove = true;
      }
      if (this.dirtyMove) this.settle();
      return;
    }
    if (this.p === to && !this.anim) {
      if (this.dirtyMove) this.settle();
      return;
    }
    this.dirtyMove = true;
    this.anim = { from: this.p, to, t0: now };
  }

  // ------------------------------------------------------------------ time

  tick(now: number): SheetTick {
    if (this.gesture === 'wheel' && !this.anim && now - this.lastInputAt >= this.cfg.idleMs) {
      this.endGesture(now);
    }
    const a = this.anim;
    if (a) {
      const u = (now - a.t0) / this.cfg.durationMs;
      if (u >= 1) {
        this.p = a.to;
        this.anim = null;
        this.settle();
      } else {
        this.p = a.from + (a.to - a.from) * easeOutCubic(Math.max(0, u));
      }
    }
    const settledNow = this.pendingSettle;
    this.pendingSettle = false;
    const moving = this.anim !== null || (this.dirtyMove && !settledNow);
    return { p: this.p, state: this.target, moving, settledNow };
  }

  // ------------------------------------------------------------------ internals

  private beginGesture(g: Gesture): void {
    if (this.gesture === 'none') {
      this.gesture = g;
      this.gestureStartP = this.p;
    }
  }

  /** A wheel gesture that went quiet is closed before a new input starts the next one. */
  private closeStaleGesture(now: number): void {
    if (this.gesture === 'wheel' && now - this.lastInputAt >= this.cfg.idleMs) this.endGesture(now);
    if (this.cfg.reducedMotion && this.gesture === 'none') {
      this.acc = 0;
      this.discreteDone = false;
    }
  }

  private setP(p: number): void {
    const q = clamp01(p);
    if (q !== this.p) this.dirtyMove = true;
    this.p = q;
  }

  private discrete(deltaPx: number, now: number): void {
    this.beginGesture('wheel');
    this.lastInputAt = now;
    if (this.discreteDone) return;
    this.acc += deltaPx;
    if (Math.abs(this.acc) >= this.cfg.discretePx) {
      this.discreteDone = true;
      this.flipTo(this.acc > 0 ? 'raised' : 'lowered');
    }
  }

  private flipTo(s: SheetState): void {
    const to = s === 'raised' ? 1 : 0;
    this.target = s;
    if (this.p !== to) {
      this.p = to;
      this.dirtyMove = true;
      this.settle();
    }
  }

  /** Close the current gesture: commit to its direction or return, by animation. */
  private endGesture(now = this.lastInputAt): void {
    const g = this.gesture;
    this.gesture = 'none';
    this.acc = 0;
    this.discreteDone = false;
    if (g === 'none') return;
    const moved = this.p - this.gestureStartP;
    let to: 0 | 1;
    if (Math.abs(moved) >= this.cfg.commitFrac) to = moved > 0 ? 1 : 0;
    else to = this.gestureStartP >= 0.5 ? 1 : 0;
    this.target = to === 1 ? 'raised' : 'lowered';
    if (this.p !== to) this.dirtyMove = true;
    if (this.p === to || this.cfg.reducedMotion) {
      // nothing to animate (already there, or reduced motion: jump)
      this.p = to;
      if (this.dirtyMove) this.settle();
      return;
    }
    this.anim = { from: this.p, to, t0: now };
  }

  private settle(): void {
    this.dirtyMove = false;
    this.pendingSettle = true;
    this.p = this.target === 'raised' ? 1 : 0;
  }
}

// ---------------------------------------------------------------------- wheel and key routing

/** Wheel delta in px: line units count 16 px, page units a full `pagePx`. */
export function normalizeWheel(deltaY: number, deltaMode: 0 | 1 | 2, pagePx: number): number {
  if (deltaMode === 1) return deltaY * 16;
  if (deltaMode === 2) return deltaY * pagePx;
  return deltaY;
}

export type WheelRoute = 'sheet' | 'camera' | 'native';

export function routeWheel(c: {
  enabled: boolean;
  dialogOpen: boolean;
  overCanvas: boolean;
  zoomModifier: boolean;
  overOwnScroller: boolean;
}): WheelRoute {
  if (!c.enabled || c.dialogOpen) return 'native';
  if (c.zoomModifier) return c.overCanvas ? 'camera' : 'native';
  if (c.overOwnScroller) return 'native';
  return 'sheet';
}

export type KeyRoute = 'raise' | 'lower' | null;

export function routeKey(c: {
  key: string;
  enabled: boolean;
  dialogOpen: boolean;
  target: 'text' | 'select' | 'range' | 'slider' | 'other';
  state: SheetState;
}): KeyRoute {
  if (!c.enabled || c.dialogOpen || c.target !== 'other') return null;
  if (c.key === 'PageDown') return c.state === 'lowered' ? 'raise' : null;
  if (c.key === 'PageUp') return c.state === 'raised' ? 'lower' : null;
  return null;
}
