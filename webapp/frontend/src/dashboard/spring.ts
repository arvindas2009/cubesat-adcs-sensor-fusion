// Small, DOM-free motion helpers for the landing page (docs/specs/ui_dashboard.md §6). A spring is set
// by damping ratio and response time (seconds) rather than mass/stiffness: damping 1 = no overshoot,
// < 1 = bounce. It always continues from its current value and velocity, so it can be re-targeted at
// any moment without a jump.

export interface SpringParams {
  /** Damping ratio: 1 = critically damped (default), < 1 overshoots. */
  damping?: number;
  /** Response time in seconds (lower = snappier). */
  response?: number;
}

export class Spring {
  value: number;
  velocity = 0;
  target: number;
  private k: number;
  private c: number;

  constructor(value: number, params: SpringParams = {}) {
    this.value = value;
    this.target = value;
    this.k = 0;
    this.c = 0;
    this.configure(params);
  }

  /** Change damping / response mid-flight; value and velocity carry over (no jump). */
  configure({ damping = 1, response = 0.4 }: SpringParams): void {
    const w = (2 * Math.PI) / response; // undamped angular frequency (unit mass)
    this.k = w * w;
    this.c = 2 * damping * w;
  }

  /** Advance by dt seconds (sub-stepped, stable for any frame time up to 0.1 s). */
  step(dt: number): number {
    const n = Math.max(1, Math.ceil(dt / (1 / 240)));
    const h = Math.min(dt, 0.1) / n;
    for (let i = 0; i < n; i++) {
      const a = -this.k * (this.value - this.target) - this.c * this.velocity;
      this.velocity += a * h; // semi-implicit Euler
      this.value += this.velocity * h;
    }
    return this.value;
  }

  get settled(): boolean {
    return Math.abs(this.value - this.target) < 1e-3 && Math.abs(this.velocity) < 1e-3;
  }
}

/** Distance a flick travels before stopping, scroll-deceleration style (velocity in units/s). */
export function project(velocity: number, decelerationRate = 0.998): number {
  return ((velocity / 1000) * decelerationRate) / (1 - decelerationRate);
}

/** The snap point nearest to x. */
export function nearest(points: readonly number[], x: number): number {
  return points.reduce((best, p) => (Math.abs(p - x) < Math.abs(best - x) ? p : best), points[0]);
}

/** Progressive resistance past a boundary: the further past it, the less the element follows. */
export function rubberband(overshoot: number, dimension: number, constant = 0.55): number {
  return (overshoot * dimension * constant) / (dimension + constant * Math.abs(overshoot));
}

/** Velocity (units/s) from the last samples of a pointer history, newest last. */
export function velocityOf(history: readonly { x: number; t: number }[], windowMs = 100): number {
  if (history.length < 2) return 0;
  const last = history[history.length - 1];
  let first = history[history.length - 2];
  for (let i = history.length - 2; i >= 0; i--) {
    if (last.t - history[i].t > windowMs) break;
    first = history[i];
  }
  const dt = (last.t - first.t) / 1000;
  return dt > 0 ? (last.x - first.x) / dt : 0;
}
