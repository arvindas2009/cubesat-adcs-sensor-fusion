// Draggable mission timeline (docs/specs/ui_dashboard.md §6). The satellite knob tracks the pointer 1:1
// from where it was grabbed, resists past the ends (rubber band), and on release is thrown: the release
// velocity projects where it would coast, it snaps to the phase nearest that point, and a spring carries
// the same velocity there (slight bounce only because a flick preceded it). Clicks, arrow keys and the
// phase labels also move it (critically damped). Browser-only; the math lives in spring.ts.
import { Spring, nearest, project, rubberband, velocityOf } from './spring';

export class Scrubber {
  private readonly track: HTMLElement;
  private readonly knob: HTMLElement;
  private readonly fill: HTMLElement;
  private readonly stops: number[]; // 0..1 along the track
  private readonly spring = new Spring(0, { damping: 1, response: 0.35 });
  private index = 0;
  private dragging = false;
  private grab = 0;
  private hist: { x: number; t: number }[] = [];
  private raf = 0;
  private last = 0;

  constructor(
    root: HTMLElement,
    count: number,
    private readonly onChange: (i: number) => void,
  ) {
    this.track = root.querySelector<HTMLElement>('.scrub-track')!;
    this.knob = root.querySelector<HTMLElement>('.scrub-knob')!;
    this.fill = root.querySelector<HTMLElement>('.scrub-fill')!;
    this.stops = Array.from({ length: count }, (_, i) => (count === 1 ? 0 : i / (count - 1)));

    this.knob.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.knob.setPointerCapture(e.pointerId);
      this.dragging = true;
      this.knob.classList.add('held');
      this.grab = e.clientX - this.px(this.spring.value); // keep the grab offset
      this.spring.velocity = 0;
      this.hist = [{ x: e.clientX, t: e.timeStamp }];
    });
    this.knob.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      const w = this.width();
      let x = (e.clientX - this.grab) / w;
      if (x < 0) x = rubberband(x * w, w) / w;
      if (x > 1) x = 1 + rubberband((x - 1) * w, w) / w;
      this.spring.value = this.spring.target = x;
      this.hist.push({ x: e.clientX, t: e.timeStamp });
      if (this.hist.length > 8) this.hist.shift();
      this.paint();
      this.select(this.closestIndex(x), false);
    });
    const release = (e: PointerEvent): void => {
      if (!this.dragging) return;
      this.dragging = false;
      this.knob.classList.remove('held');
      const vPx = velocityOf(this.hist);
      const w = this.width();
      const landing = this.spring.value + project(vPx) / w;
      const i = this.closestIndex(landing);
      this.spring.velocity = vPx / w; // hand off the finger's velocity
      this.goTo(i, Math.abs(vPx) > 200 ? 0.8 : 1);
      if (this.knob.hasPointerCapture(e.pointerId)) this.knob.releasePointerCapture(e.pointerId);
    };
    this.knob.addEventListener('pointerup', release);
    this.knob.addEventListener('pointercancel', release);
    this.track.addEventListener('pointerdown', (e) => {
      if (e.target === this.knob) return;
      const r = this.track.getBoundingClientRect();
      this.goTo(this.closestIndex((e.clientX - r.left) / r.width), 1);
    });
    this.knob.addEventListener('keydown', (e) => {
      const d = e.key === 'ArrowRight' || e.key === 'ArrowUp' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowDown' ? -1 : 0;
      if (e.key === 'Home') this.goTo(0, 1);
      else if (e.key === 'End') this.goTo(this.stops.length - 1, 1);
      else if (d) this.goTo(Math.min(this.stops.length - 1, Math.max(0, this.index + d)), 1);
      else return;
      e.preventDefault();
    });
    new ResizeObserver(() => this.paint()).observe(this.track);
    this.paint();
    this.select(0, true);
  }

  private width(): number {
    return Math.max(1, this.track.clientWidth);
  }

  private px(v: number): number {
    return v * this.width();
  }

  private closestIndex(x: number): number {
    return this.stops.indexOf(nearest(this.stops, x));
  }

  goTo(i: number, damping: number): void {
    this.spring.configure({ damping, response: 0.4 });
    this.spring.target = this.stops[i];
    this.select(i, true);
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      this.spring.value = this.spring.target;
      this.spring.velocity = 0;
      this.paint();
      return;
    }
    if (!this.raf) {
      this.last = performance.now();
      this.raf = requestAnimationFrame(this.tick);
    }
  }

  private readonly tick = (now: number): void => {
    this.raf = 0;
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    this.spring.step(dt);
    this.paint();
    if (!this.spring.settled && !this.dragging) this.raf = requestAnimationFrame(this.tick);
    else if (!this.dragging) {
      this.spring.value = this.spring.target;
      this.paint();
    }
  };

  private select(i: number, force: boolean): void {
    if (i === this.index && !force) return;
    this.index = i;
    this.knob.setAttribute('aria-valuenow', String(i + 1));
    this.onChange(i);
  }

  private paint(): void {
    const x = this.px(this.spring.value);
    this.knob.style.transform = `translate3d(${x}px, -50%, 0)`;
    this.fill.style.transform = `scaleX(${Math.min(1, Math.max(0, this.spring.value))})`;
  }
}
